#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn


TARGET_LINEAR_NAMES = (
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
)

REQUIRED_STATE_SUFFIXES = (
  "rotate_linear.rotation.theta",
  "rotate_linear.rotation.pairs",
  "rotate_linear.rotation.channel_scales",
  "rotate_linear.qlinear.qweight",
  "rotate_linear.qlinear.qzeros",
  "rotate_linear.qlinear.scales",
)

OPTIONAL_STATE_SUFFIXES = (
  "rotate_linear.qlinear.g_idx",
  "rotate_linear.qlinear.g_idx_sort_indices",
  "rotate_linear.qlinear.input_global_scale",
  "rotate_linear.qlinear.workspace",
)


@dataclass(frozen=True)
class NestedLoadReport:
  loaded: tuple[str, ...]
  missing: tuple[str, ...]


@dataclass(frozen=True)
class ReplaceLinearsSummary:
  replaced: int
  skipped: int
  language_root: str
  target_names: tuple[str, ...]
  replaced_names: tuple[str, ...]
  skipped_reasons: tuple[str, ...]


def _ensure_positive(name: str, value: int) -> int:
  value_i = int(value)
  if value_i <= 0:
    raise ValueError(f"{name} must be > 0, got {value_i}")
  return value_i


def _infer_layout(in_features: int, out_features: int, group_size: int = 128) -> dict[str, tuple[int, ...]]:
  in_features = _ensure_positive("in_features", in_features)
  out_features = _ensure_positive("out_features", out_features)
  if out_features % 2 != 0:
    raise ValueError(f"out_features must be even for native PARO layout, got {out_features}")
  if out_features % 8 != 0:
    raise ValueError(f"out_features must be multiple of 8, got {out_features}")
  if in_features % 16 != 0:
    raise ValueError(f"in_features must be multiple of 16, got {in_features}")
  if in_features % group_size != 0:
    raise ValueError(f"in_features must be multiple of group_size={group_size}, got {in_features}")
  if (group_size & (group_size - 1)) != 0:
    raise ValueError(f"group_size must be positive power-of-two, got {group_size}")

  return {
    "rotate_linear.rotation.theta": (8, out_features // 2),
    "rotate_linear.rotation.pairs": (8, in_features),
    "rotate_linear.rotation.channel_scales": (1, out_features),
    "rotate_linear.qlinear.qweight": (in_features // 16, out_features * 2),
    "rotate_linear.qlinear.qzeros": (in_features // group_size, out_features // 8),
    "rotate_linear.qlinear.scales": (in_features // group_size, out_features),
    "rotate_linear.qlinear.g_idx": (in_features // 16,),
    "rotate_linear.qlinear.g_idx_sort_indices": (in_features // 16,),
    "rotate_linear.qlinear.input_global_scale": (1,),
    "rotate_linear.qlinear.workspace": (1,),
  }


def _set_nested_buffer(parent: nn.Module, key: str, value: torch.Tensor) -> None:
  module = parent
  parts = key.split(".")
  for part in parts[:-1]:
    module = getattr(module, part)
  name = parts[-1]
  if name in module._buffers:
    module._buffers[name] = value
    return
  module.register_buffer(name, value)


def _get_nested_buffer(module: nn.Module, key: str) -> torch.Tensor:
  target = module
  for part in key.split("."):
    target = getattr(target, part)
  if not isinstance(target, torch.Tensor):
    raise TypeError(f"expected tensor at {key}, got {type(target)!r}")
  return target


def _state_dtype(key: str) -> torch.dtype:
  if key.endswith("pairs") or key.endswith("qzeros") or key.endswith("g_idx") or key.endswith("g_idx_sort_indices") or key.endswith("workspace"):
    return torch.int32 if key.endswith("qzeros") or key.endswith("g_idx") or key.endswith("g_idx_sort_indices") or key.endswith("workspace") else torch.int16
  if key.endswith("input_global_scale"):
    return torch.float32
  if key.endswith("scales") or "channel_scales" in key or "theta" in key:
    return torch.float16
  return torch.float16


class ParoNativeMarlinLinear(nn.Module):
  """Thin PARO-native linear wrapper for vLLM Marlin apply path."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    rotation_group_size: int = 128,
    bias: bool = False,
    input_dtype: torch.dtype | None = None,
    placeholder_dtype: torch.dtype = torch.float16,
  ) -> None:
    super().__init__()
    self.in_features = _ensure_positive("in_features", in_features)
    self.out_features = _ensure_positive("out_features", out_features)
    self.rotation_group_size = rotation_group_size
    self.input_dtype = input_dtype
    self._loaded_suffixes: set[str] = set()

    layout = _infer_layout(self.in_features, self.out_features, rotation_group_size)
    self.rotate_linear = nn.Module()
    self.rotate_linear.rotation = nn.Module()
    self.rotate_linear.qlinear = nn.Module()

    for key in REQUIRED_STATE_SUFFIXES + OPTIONAL_STATE_SUFFIXES:
      target_dtype = _state_dtype(key)
      shape = layout[key]
      if key.startswith("rotate_linear.rotation.") and "channel_scales" in key:
        value = torch.ones(shape, dtype=target_dtype)
      elif key == "rotate_linear.qlinear.input_global_scale":
        value = torch.ones(shape, dtype=target_dtype)
      else:
        value = torch.zeros(shape, dtype=target_dtype)
      _set_nested_buffer(self, key, value)

    if bias:
      self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=placeholder_dtype), requires_grad=False)
    else:
      self.register_parameter("bias", None)

  @classmethod
  def from_linear_module(
    cls,
    linear: nn.Linear,
    *,
    rotation_group_size: int = 128,
  ) -> "ParoNativeMarlinLinear":
    if not isinstance(linear, nn.Linear):
      raise TypeError(f"expected nn.Linear, got {type(linear)}")
    return cls(
      linear.in_features,
      linear.out_features,
      rotation_group_size=rotation_group_size,
      bias=(linear.bias is not None),
      placeholder_dtype=linear.weight.dtype,
    )

  @classmethod
  def from_nested_state(
    cls,
    *,
    state: Mapping[str, torch.Tensor],
    module_prefix: str,
    rotation_group_size: int = 128,
    input_dtype: torch.dtype | None = None,
    strict: bool = True,
    strict_keys: bool = True,
  ) -> "ParoNativeMarlinLinear":
    prefix = f"{module_prefix}." if module_prefix and not module_prefix.endswith(".") else module_prefix
    pairs_key = f"{prefix}rotate_linear.rotation.pairs"
    scales_key = f"{prefix}rotate_linear.qlinear.scales"
    if pairs_key not in state or scales_key not in state:
      missing = [pairs_key if pairs_key not in state else None, scales_key if scales_key not in state else None]
      missing = [item for item in missing if item is not None]
      raise KeyError(f"missing required layout keys: {missing}")

    pairs = state[pairs_key]
    scales = state[scales_key]
    module = cls(
      int(pairs.shape[-1]),
      int(scales.shape[-1]),
      rotation_group_size=rotation_group_size,
      input_dtype=input_dtype,
    )
    module.load_nested_state(state, module_prefix=prefix, strict=strict)
    if strict_keys:
      missing = [key for key in REQUIRED_STATE_SUFFIXES if f"{prefix}{key}" not in state]
      if missing:
        raise KeyError(f"missing required nested PARO tensors for {module_prefix}: {missing}")
    return module

  @classmethod
  def from_nested_state_tensors(
    cls,
    module_prefix: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    rotation_group_size: int = 128,
    input_dtype: torch.dtype | None = None,
    strict: bool = True,
  ) -> "ParoNativeMarlinLinear":
    return cls.from_nested_state(
      state=tensors,
      module_prefix=module_prefix,
      rotation_group_size=rotation_group_size,
      input_dtype=input_dtype,
      strict=strict,
      strict_keys=False,
    )

  def load_nested_state(
    self,
    state: Mapping[str, torch.Tensor],
    *,
    module_prefix: str = "",
    strict: bool = False,
    include_optional: bool = True,
  ) -> NestedLoadReport:
    expected = list(REQUIRED_STATE_SUFFIXES)
    if include_optional:
      expected.extend(OPTIONAL_STATE_SUFFIXES)
    loaded: list[str] = []
    missing: list[str] = []

    for suffix in expected:
      key = f"{module_prefix}{suffix}"
      if key not in state:
        if suffix in REQUIRED_STATE_SUFFIXES:
          missing.append(suffix)
        continue

      tensor = state[key]
      if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor for {key}, got {type(tensor)}")
      tensor = tensor.to(dtype=_state_dtype(suffix), device=self._default_target_device())
      _set_nested_buffer(self, suffix, tensor)
      loaded.append(suffix)
      self._loaded_suffixes.add(suffix)

    if strict and missing:
      raise KeyError(f"missing required nested keys: {missing}")

    return NestedLoadReport(loaded=tuple(loaded), missing=tuple(missing))

  def _default_target_device(self) -> torch.device:
    if self.rotate_linear.rotation.pairs.is_cuda:
      return self.rotate_linear.rotation.pairs.device
    return torch.device("cpu")

  def _required_keys_present(self) -> bool:
    for key in REQUIRED_STATE_SUFFIXES:
      if key not in self._loaded_suffixes:
        return False
      if _get_nested_buffer(self, key).numel() == 0:
        return False
    return True

  def _apply_marlin(self, x: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_awq_marlin_linear
    from vllm.scalar_type import scalar_types

    rotated = torch.ops.rotation.rotate(
      x,
      self.rotate_linear.rotation.pairs,
      self.rotate_linear.rotation.theta,
      self.rotate_linear.rotation.channel_scales,
      self.rotation_group_size,
    )
    return apply_awq_marlin_linear(
      input=rotated,
      weight=self.rotate_linear.qlinear.qweight,
      weight_scale=self.rotate_linear.qlinear.scales,
      weight_zp=self.rotate_linear.qlinear.qzeros,
      g_idx=self.rotate_linear.qlinear.g_idx,
      g_idx_sort_indices=self.rotate_linear.qlinear.g_idx_sort_indices,
      workspace=self.rotate_linear.qlinear.workspace,
      quant_type=scalar_types.uint4,
      output_size_per_partition=self.out_features,
      input_size_per_partition=self.in_features,
      input_global_scale=self.rotate_linear.qlinear.input_global_scale,
      input_dtype=self.input_dtype,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != self.in_features:
      raise ValueError(f"input last dimension must be {self.in_features}, got {x.shape[-1]}")
    if not self._required_keys_present():
      raise RuntimeError("native PARO tensors are not loaded")

    x_2d = x.reshape(-1, self.in_features)
    output = self._apply_marlin(x_2d)
    output = output.reshape(*x.shape[:-1], self.out_features)
    if self.bias is not None:
      output = output + self.bias
    return output


def _is_target_linear_path(path: str, target_names: Iterable[str]) -> bool:
  return any(path == name or path.endswith(f".{name}") for name in target_names)


def _get_parent_and_attr(root: nn.Module, module_path: str) -> tuple[nn.Module, str]:
  if "." not in module_path:
    return root, module_path
  cursor = root
  parts = module_path.split(".")
  for part in parts[:-1]:
    cursor = getattr(cursor, part)
  return cursor, parts[-1]


def iter_language_linear_modules(
  model: nn.Module,
  *,
  language_root: str,
  target_names: Iterable[str] = TARGET_LINEAR_NAMES,
) -> list[tuple[str, nn.Linear]]:
  matches: list[tuple[str, nn.Linear]] = []
  for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
      continue
    if language_root and not name.startswith(language_root):
      continue
    if _is_target_linear_path(name, target_names):
      matches.append((name, module))
  return matches


def replace_language_linear_modules(
  model: nn.Module,
  *,
  language_root: str,
  target_names: Iterable[str] = TARGET_LINEAR_NAMES,
  rotation_group_size: int = 128,
  require_cuda: bool = False,
) -> ReplaceLinearsSummary:
  targets = iter_language_linear_modules(
    model,
    language_root=language_root,
    target_names=target_names,
  )

  replaced: list[str] = []
  skipped: list[str] = []

  for module_path, linear in targets:
    try:
      parent, attr = _get_parent_and_attr(model, module_path)
      replacement = ParoNativeMarlinLinear.from_linear_module(linear, rotation_group_size=rotation_group_size)
      if require_cuda:
        replacement.cuda()
      setattr(parent, attr, replacement)
      replaced.append(module_path)
    except Exception as exc:
      skipped.append(f"{module_path}::{type(exc).__name__}:{exc}")

  return ReplaceLinearsSummary(
    replaced=len(replaced),
    skipped=len(skipped),
    language_root=language_root,
    target_names=tuple(target_names),
    replaced_names=tuple(replaced),
    skipped_reasons=tuple(skipped),
  )
