#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
DEFAULT_PARO_MODEL_PATH = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
DEFAULT_MODULE_PREFIX = "vlm.model.language_model.layers.0.self_attn.q_proj"
PARO_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MARLIN_WORKSPACE_SIZE = 188
ROTATION_GROUPS = 8
ROTATION_GROUP_SIZE = 128


def windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def normalize_path(raw: str | os.PathLike[str]) -> Path:
  return Path(os.path.expanduser(windows_to_wsl(str(raw))))


def dtype_from_name(name: str) -> torch.dtype:
  if name in ("fp16", "float16"):
    return torch.float16
  if name in ("bf16", "bfloat16"):
    return torch.bfloat16
  if name in ("fp32", "float32"):
    return torch.float32
  raise ValueError(f"unsupported dtype {name}")


def env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in ("1", "true", "yes", "on")


def _ensure_paro_rotation_op_available() -> None:
  if not hasattr(torch.ops, "rotation"):
    raise RuntimeError("PARO rotation namespace is unavailable in torch.ops")
  rotate_op = getattr(torch.ops.rotation, "rotate", None)
  if not callable(rotate_op):
    raise RuntimeError("PARO rotation op (torch.ops.rotation.rotate) is unavailable")


def _marlin_input_dtype(name: str) -> torch.dtype | None:
  if name == "none":
    return None
  if name == "int8":
    return torch.int8
  if name == "fp8":
    return torch.float8_e4m3fn
  raise ValueError(f"unsupported marlin input dtype mode {name}")


_MARLIN_RUNTIME: tuple[Any, Any] | None = None
_MARLIN_RUNTIME_ERROR: str | None = None
_PARO_PLUGIN_LOADED = False
_PARO_PLUGIN_ERROR: str | None = None


def ensure_paroquant_plugin() -> None:
  global _PARO_PLUGIN_LOADED, _PARO_PLUGIN_ERROR
  if _PARO_PLUGIN_LOADED:
    try:
      _ensure_paro_rotation_op_available()
    except Exception as exc:
      _PARO_PLUGIN_ERROR = f"{type(exc).__name__}: {exc}"
      raise RuntimeError(f"PARO rotation plugin unavailable: {_PARO_PLUGIN_ERROR}") from exc
    return
  try:
    import paroquant.inference.backends.vllm.plugin  # noqa: F401
    _ensure_paro_rotation_op_available()
  except Exception as exc:
    _PARO_PLUGIN_ERROR = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"PARO rotation plugin unavailable: {_PARO_PLUGIN_ERROR}") from exc
  _PARO_PLUGIN_LOADED = True
  _PARO_PLUGIN_ERROR = None


def ensure_native_paro_runtime() -> tuple[Any, Any]:
  return _load_marlin_runtime()


def _load_marlin_runtime() -> tuple[Any, Any]:
  global _MARLIN_RUNTIME, _MARLIN_RUNTIME_ERROR
  if _MARLIN_RUNTIME is not None:
    try:
      _ensure_paro_rotation_op_available()
    except Exception as exc:
      _MARLIN_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
      raise RuntimeError(f"PARO native runtime unavailable: {_MARLIN_RUNTIME_ERROR}") from exc
    _MARLIN_RUNTIME_ERROR = None
    return _MARLIN_RUNTIME
  try:
    ensure_paroquant_plugin()
  except Exception as exc:
    _MARLIN_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"PARO native runtime unavailable: {_MARLIN_RUNTIME_ERROR}") from exc
  try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_awq_marlin_linear
    from vllm.scalar_type import scalar_types
  except Exception as exc:
    _MARLIN_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"vLLM Marlin runtime unavailable: {_MARLIN_RUNTIME_ERROR}") from exc
  if not callable(apply_awq_marlin_linear):
    _MARLIN_RUNTIME_ERROR = f"apply_awq_marlin_linear is not callable ({type(apply_awq_marlin_linear)!r})"
    raise RuntimeError(f"vLLM Marlin runtime unavailable: {_MARLIN_RUNTIME_ERROR}")
  if not hasattr(scalar_types, "uint4"):
    _MARLIN_RUNTIME_ERROR = "scalar_types.uint4 is missing in vLLM build"
    raise RuntimeError(f"vLLM Marlin runtime unavailable: {_MARLIN_RUNTIME_ERROR}")
  _ensure_paro_rotation_op_available()
  _MARLIN_RUNTIME = (apply_awq_marlin_linear, scalar_types)
  _MARLIN_RUNTIME_ERROR = None
  return _MARLIN_RUNTIME


def _marlin_runtime_error() -> str | None:
  try:
    _load_marlin_runtime()
  except Exception:
    pass
  return _MARLIN_RUNTIME_ERROR


def _make_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
  return torch.empty(shape, dtype=dtype)


def _describe_tensor(tensor: torch.Tensor) -> dict[str, Any]:
  return {
    "shape": list(tensor.shape),
    "dtype": str(tensor.dtype),
    "device": str(tensor.device),
    "isCuda": bool(tensor.is_cuda),
    "isContiguous": bool(tensor.is_contiguous()),
  }


class ParoRotation(nn.Module):
  def __init__(self, in_features: int) -> None:
    super().__init__()
    if in_features % 2 != 0:
      raise ValueError(f"PARO rotation in_features must be even, got {in_features}")
    self.register_buffer(
      "theta",
      _make_empty((ROTATION_GROUPS, in_features // 2), torch.float16),
      persistent=True,
    )
    self.register_buffer(
      "pairs",
      _make_empty((ROTATION_GROUPS, in_features), torch.int16),
      persistent=True,
    )
    self.register_buffer(
      "channel_scales",
      _make_empty((1, in_features), torch.float16),
      persistent=True,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.rotation.rotate(
      x,
      self.pairs,
      self.theta,
      self.channel_scales,
      ROTATION_GROUP_SIZE,
    )


class ParoNativeQLinear(nn.Module):
  def __init__(self, in_features: int, out_features: int) -> None:
    super().__init__()
    if in_features % 128 != 0:
      raise ValueError(f"PARO qlinear in_features must be divisible by 128, got {in_features}")
    if out_features % 8 != 0:
      raise ValueError(f"PARO qlinear out_features must be divisible by 8, got {out_features}")
    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.register_buffer(
      "qweight",
      _make_empty((in_features // 16, out_features * 2), torch.int32),
      persistent=True,
    )
    self.register_buffer(
      "qzeros",
      _make_empty((in_features // 128, out_features // 8), torch.int32),
      persistent=True,
    )
    self.register_buffer(
      "scales",
      _make_empty((in_features // 128, out_features), torch.float16),
      persistent=True,
    )
    self.register_buffer("g_idx", _make_empty((0,), torch.int32), persistent=True)
    self.register_buffer("g_idx_sort_indices", _make_empty((0,), torch.int32), persistent=True)
    self.register_buffer("input_global_scale", _make_empty((), torch.float32), persistent=True)
    self.register_buffer(
      "workspace",
      _make_empty((MARLIN_WORKSPACE_SIZE,), torch.int32),
      persistent=True,
    )

  def forward(self, x: torch.Tensor, *, input_dtype: str = "int8") -> torch.Tensor:
    apply_awq_marlin_linear, scalar_types = _load_marlin_runtime()
    parsed_input_dtype = _marlin_input_dtype(str(input_dtype).lower())

    return apply_awq_marlin_linear(
      input=x,
      weight=self.qweight,
      weight_scale=self.scales,
      weight_zp=self.qzeros,
      g_idx=self.g_idx,
      g_idx_sort_indices=self.g_idx_sort_indices,
      workspace=self.workspace,
      quant_type=scalar_types.uint4,
      output_size_per_partition=self.out_features,
      input_size_per_partition=self.in_features,
      input_global_scale=self.input_global_scale,
      input_dtype=parsed_input_dtype,
    )


class ParoRotateLinear(nn.Module):
  def __init__(self, in_features: int, out_features: int) -> None:
    super().__init__()
    self.rotation = ParoRotation(in_features)
    self.qlinear = ParoNativeQLinear(in_features, out_features)

  def forward(self, x: torch.Tensor, *, input_dtype: str = "int8") -> torch.Tensor:
    return self.qlinear(self.rotation(x), input_dtype=input_dtype)


class NativeParoMarlinLinear(nn.Module):
  """HF-compatible PARO nested rotate+Marlin linear.

  The child names intentionally match checkpoint keys such as:
  q_proj.rotate_linear.rotation.theta and q_proj.rotate_linear.qlinear.qweight.
  """

  bias = None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    marlin_input_dtype: str = "int8",
    compute_dtype: torch.dtype = torch.float16,
    output_dtype: str = "input",
  ) -> None:
    super().__init__()
    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.marlin_input_dtype = str(marlin_input_dtype).lower()
    self._marlin_input_dtype = _marlin_input_dtype(self.marlin_input_dtype)
    self.compute_dtype = compute_dtype
    self.output_dtype = output_dtype
    self.rotate_linear = ParoRotateLinear(in_features, out_features)
    self._openpilot_paro_native = True
    self._openpilot_paro_activation_int8_ready = self._marlin_input_dtype == torch.int8
    self._openpilot_paro_marlin_runtime_ready = False
    self._openpilot_paro_fast_prefill_ready = False
    self._openpilot_paro_fast_prefill_reason = "not_checked"
    self._refresh_openpilot_fast_prefill_state()
    self._paro_name: str | None = None
    self._paro_target_device: torch.device | None = None
    self._paro_buffers_checked_device: torch.device | None = None

  def _refresh_openpilot_fast_prefill_state(self) -> None:
    activation_int8_ready = self._marlin_input_dtype == torch.int8
    self._openpilot_paro_activation_int8_requested = activation_int8_ready
    try:
      _load_marlin_runtime()
      self._openpilot_paro_marlin_runtime_ready = True
      self._openpilot_paro_activation_int8_ready = bool(activation_int8_ready)
      self._openpilot_paro_fast_prefill_ready = bool(activation_int8_ready)
      self._openpilot_paro_fast_prefill_reason = (
        "vllm_awq_marlin_int8_input" if activation_int8_ready else "marlin_input_dtype_not_int8"
      )
    except Exception:
      self._openpilot_paro_marlin_runtime_ready = False
      self._openpilot_paro_activation_int8_ready = False
      self._openpilot_paro_fast_prefill_ready = False
      self._openpilot_paro_fast_prefill_reason = _marlin_runtime_error() or "marlin_runtime_unavailable"

  @classmethod
  def from_linear(
    cls,
    module: nn.Module,
    *,
    marlin_input_dtype: str = "int8",
    compute_dtype: torch.dtype = torch.float16,
    output_dtype: str = "input",
  ) -> "NativeParoMarlinLinear":
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is None or out_features is None:
      weight = getattr(module, "weight", None)
      if weight is None or len(weight.shape) != 2:
        raise TypeError(f"cannot infer Linear shape from {module.__class__.__name__}")
      out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    return cls(
      int(in_features),
      int(out_features),
      marlin_input_dtype=marlin_input_dtype,
      compute_dtype=compute_dtype,
      output_dtype=output_dtype,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    apply_awq_marlin_linear, scalar_types = _load_marlin_runtime()
    self._refresh_openpilot_fast_prefill_state()

    original_dtype = x.dtype
    target_device = self._paro_target_device or self.rotate_linear.qlinear.qweight.device
    if self._paro_buffers_checked_device != target_device:
      buffers = (
        self.rotate_linear.rotation.pairs,
        self.rotate_linear.rotation.theta,
        self.rotate_linear.rotation.channel_scales,
        self.rotate_linear.qlinear.qweight,
        self.rotate_linear.qlinear.scales,
        self.rotate_linear.qlinear.qzeros,
        self.rotate_linear.qlinear.g_idx,
        self.rotate_linear.qlinear.g_idx_sort_indices,
        self.rotate_linear.qlinear.workspace,
        self.rotate_linear.qlinear.input_global_scale,
      )
      if any(buffer.device != target_device for buffer in buffers):
        self.to(device=target_device)
      self._paro_buffers_checked_device = target_device
    if x.device != target_device:
      x = x.to(device=target_device)
    if target_device.type == "cuda":
      target_index = target_device.index if target_device.index is not None else torch.cuda.current_device()
      if torch.cuda.current_device() != target_index:
        torch.cuda.set_device(target_index)
    original_shape = x.shape[:-1]
    x2d = x.reshape(-1, self.in_features)
    if x2d.dtype != self.compute_dtype:
      x2d = x2d.to(self.compute_dtype)
    rotated = torch.ops.rotation.rotate(
      x2d,
      self.rotate_linear.rotation.pairs,
      self.rotate_linear.rotation.theta,
      self.rotate_linear.rotation.channel_scales,
      ROTATION_GROUP_SIZE,
    )
    qlinear = self.rotate_linear.qlinear
    debug_state = {
      "module": self._paro_name,
      "targetDevice": str(target_device),
      "input": _describe_tensor(x),
      "input2d": _describe_tensor(x2d),
      "rotated": _describe_tensor(rotated),
      "qweight": _describe_tensor(qlinear.qweight),
      "qzeros": _describe_tensor(qlinear.qzeros),
      "scales": _describe_tensor(qlinear.scales),
      "workspace": _describe_tensor(qlinear.workspace),
      "inputGlobalScale": _describe_tensor(qlinear.input_global_scale),
        "marlinInputDtype": self.marlin_input_dtype,
    }
    if (
      not rotated.is_cuda
      or not qlinear.qweight.is_cuda
      or not qlinear.qzeros.is_cuda
      or not qlinear.scales.is_cuda
      or not qlinear.workspace.is_cuda
      or not qlinear.input_global_scale.is_cuda
    ):
      raise RuntimeError(f"PARO Marlin tensor device mismatch: {json.dumps(debug_state, sort_keys=True)}")
    try:
      out = apply_awq_marlin_linear(
        input=rotated,
        weight=qlinear.qweight,
        weight_scale=qlinear.scales,
        weight_zp=qlinear.qzeros,
        g_idx=qlinear.g_idx,
        g_idx_sort_indices=qlinear.g_idx_sort_indices,
        workspace=qlinear.workspace,
        quant_type=scalar_types.uint4,
        output_size_per_partition=self.out_features,
        input_size_per_partition=self.in_features,
        input_global_scale=qlinear.input_global_scale,
        input_dtype=self._marlin_input_dtype,
      )
    except Exception as exc:
      raise RuntimeError(
        f"PARO Marlin call failed in {self._paro_name}: "
        f"{type(exc).__name__}: {exc}; state={json.dumps(debug_state, sort_keys=True)}"
      ) from exc
    out = out.reshape(*original_shape, self.out_features)
    if self.output_dtype == "input" and out.dtype != original_dtype:
      return out.to(original_dtype)
    if self.output_dtype == "compute" and out.dtype != self.compute_dtype:
      return out.to(self.compute_dtype)
    return out

  def extra_repr(self) -> str:
    return (
      f"in_features={self.in_features}, out_features={self.out_features}, "
      f"marlin_input_dtype={self.marlin_input_dtype}, compute_dtype={self.compute_dtype}, "
      f"output_dtype={self.output_dtype}"
    )


def _device_from_hf_map(name: str, hf_device_map: dict[str, Any] | None) -> torch.device | None:
  if not hf_device_map:
    return None
  best_prefix = ""
  best_value: Any = None
  for prefix, value in hf_device_map.items():
    if name == prefix or name.startswith(f"{prefix}."):
      if len(prefix) > len(best_prefix):
        best_prefix = prefix
        best_value = value
  if best_value is None:
    return None
  if isinstance(best_value, int):
    return torch.device(f"cuda:{best_value}")
  if isinstance(best_value, str) and best_value not in ("cpu", "disk"):
    return torch.device(best_value)
  return None


def _paro_named_runtime_buffers(module: NativeParoMarlinLinear) -> dict[str, torch.Tensor]:
  return {
    "rotation.pairs": module.rotate_linear.rotation.pairs,
    "rotation.theta": module.rotate_linear.rotation.theta,
    "rotation.channel_scales": module.rotate_linear.rotation.channel_scales,
    "qlinear.qweight": module.rotate_linear.qlinear.qweight,
    "qlinear.scales": module.rotate_linear.qlinear.scales,
    "qlinear.qzeros": module.rotate_linear.qlinear.qzeros,
    "qlinear.g_idx": module.rotate_linear.qlinear.g_idx,
    "qlinear.g_idx_sort_indices": module.rotate_linear.qlinear.g_idx_sort_indices,
    "qlinear.workspace": module.rotate_linear.qlinear.workspace,
    "qlinear.input_global_scale": module.rotate_linear.qlinear.input_global_scale,
  }


def _paro_non_cuda_runtime_buffers(module: NativeParoMarlinLinear) -> list[str]:
  return [
    name
    for name, tensor in _paro_named_runtime_buffers(module).items()
    if not bool(tensor.is_cuda)
  ]


def finalize_native_paro_modules_for_device_map(model: nn.Module) -> list[dict[str, Any]]:
  """Remove Accelerate hooks from PARO modules and pin them to their layer device."""
  try:
    from accelerate.hooks import remove_hook_from_module
  except Exception:
    remove_hook_from_module = None

  require_cuda_modules = env_bool("ALPAMAYO_PARO_REQUIRE_CUDA_MODULES", False)
  require_fast_prefill = env_bool("ALPAMAYO_PARO_REQUIRE_FAST_PREFILL", False)
  hf_device_map = getattr(model, "hf_device_map", None)
  records: list[dict[str, Any]] = []
  for name, module in model.named_modules():
    if not isinstance(module, NativeParoMarlinLinear):
      continue
    if remove_hook_from_module is not None:
      remove_hook_from_module(module, recurse=True)
    target_device = _device_from_hf_map(name, hf_device_map)
    if target_device is not None:
      module.to(device=target_device)
      module._paro_target_device = target_device
    module._paro_name = name
    module._refresh_openpilot_fast_prefill_state()
    non_cuda_buffers = _paro_non_cuda_runtime_buffers(module)
    cuda_ready = not non_cuda_buffers and module.rotate_linear.qlinear.qweight.device.type == "cuda"
    records.append(
      {
        "name": name,
        "device": str(module.rotate_linear.qlinear.qweight.device),
        "targetDevice": str(target_device) if target_device is not None else None,
        "marlinInputDtype": str(module.marlin_input_dtype),
        "marlinRuntimeReady": bool(getattr(module, "_openpilot_paro_marlin_runtime_ready", False)),
        "activationInt8Ready": bool(getattr(module, "_openpilot_paro_activation_int8_ready", False)),
        "fastPrefillReady": bool(getattr(module, "_openpilot_paro_fast_prefill_ready", False)),
        "fastPrefillReason": str(getattr(module, "_openpilot_paro_fast_prefill_reason", "")),
        "cudaReady": bool(cuda_ready),
        "nonCudaBuffers": non_cuda_buffers,
      }
    )
  if require_cuda_modules:
    if not records:
      raise RuntimeError("ALPAMAYO_PARO_REQUIRE_CUDA_MODULES=1 but no NativeParoMarlinLinear modules were finalized")
    non_cuda_records = [
      {
        "name": record["name"],
        "device": record["device"],
        "targetDevice": record["targetDevice"],
        "nonCudaBuffers": record["nonCudaBuffers"],
      }
      for record in records
      if not bool(record["cudaReady"])
    ]
    if non_cuda_records:
      raise RuntimeError(
        "ALPAMAYO_PARO_REQUIRE_CUDA_MODULES=1 but some PARO modules are not fully CUDA resident: "
        f"{json.dumps(non_cuda_records[:16], sort_keys=True)}"
      )
  if require_fast_prefill:
    if not records:
      raise RuntimeError("ALPAMAYO_PARO_REQUIRE_FAST_PREFILL=1 but no NativeParoMarlinLinear modules were finalized")
    slow_records = [
      {
        "name": record["name"],
        "marlinRuntimeReady": record["marlinRuntimeReady"],
        "activationInt8Ready": record["activationInt8Ready"],
        "fastPrefillReady": record["fastPrefillReady"],
        "fastPrefillReason": record["fastPrefillReason"],
      }
      for record in records
      if not bool(record["fastPrefillReady"])
    ]
    if slow_records:
      raise RuntimeError(
        "ALPAMAYO_PARO_REQUIRE_FAST_PREFILL=1 but some PARO modules are not fast-prefill ready: "
        f"{json.dumps(slow_records[:16], sort_keys=True)}"
      )
  return records


@dataclass(frozen=True)
class ReplacementRecord:
  name: str
  in_features: int
  out_features: int


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
  parent_name, _, child_name = name.rpartition(".")
  parent = root.get_submodule(parent_name) if parent_name else root
  setattr(parent, child_name, module)


def iter_vlm_language_linear_names(model: nn.Module) -> list[str]:
  names: list[str] = []
  for name, module in model.named_modules():
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in PARO_LINEAR_NAMES:
      continue
    if ".language_model.layers." not in name:
      continue
    if not isinstance(module, nn.Linear):
      continue
    names.append(name)
  return names


def apply_native_paro_linear_replacements(
  model: nn.Module,
  *,
  marlin_input_dtype: str = "int8",
  compute_dtype: torch.dtype = torch.float16,
  output_dtype: str = "input",
) -> list[ReplacementRecord]:
  ensure_native_paro_runtime()
  records: list[ReplacementRecord] = []
  for name in iter_vlm_language_linear_names(model):
    original = model.get_submodule(name)
    replacement = NativeParoMarlinLinear.from_linear(
      original,
      marlin_input_dtype=marlin_input_dtype,
      compute_dtype=compute_dtype,
      output_dtype=output_dtype,
    )
    replacement._paro_name = name
    replacement._openpilot_paro_native = True
    replacement._openpilot_paro_activation_int8_ready = str(marlin_input_dtype).lower() == "int8"
    replacement._refresh_openpilot_fast_prefill_state()
    _set_submodule(model, name, replacement)
    records.append(
      ReplacementRecord(
        name=name,
        in_features=replacement.in_features,
        out_features=replacement.out_features,
      )
    )
  return records


def load_one_module_tensors(
  model_path: Path,
  module_prefix: str,
  module: NativeParoMarlinLinear,
  *,
  device: torch.device,
) -> None:
  from safetensors.torch import safe_open

  index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
  weight_map = index["weight_map"]
  state_names = tuple(module.state_dict().keys())
  shard_to_keys: dict[str, list[tuple[str, str]]] = {}
  for local_key in state_names:
    full_key = f"{module_prefix}.{local_key}"
    if full_key not in weight_map:
      raise KeyError(f"missing PARO key {full_key}")
    shard_to_keys.setdefault(weight_map[full_key], []).append((local_key, full_key))

  tensors: dict[str, torch.Tensor] = {}
  for shard_name, keys in shard_to_keys.items():
    with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as handle:
      for local_key, full_key in keys:
        tensors[local_key] = handle.get_tensor(full_key).to(device=device)
  missing, unexpected = module.load_state_dict(tensors, strict=True)
  if missing or unexpected:
    raise RuntimeError(f"load_state_dict mismatch missing={missing} unexpected={unexpected}")


def _run_one_module_probe(args: argparse.Namespace) -> dict[str, Any]:
  if args.torch_cuda_arch_list:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)

  model_path = normalize_path(args.model_path)
  device = torch.device(args.device)
  torch.cuda.set_device(device)
  report: dict[str, Any] = {
    "status": "started",
    "created_unix": time.time(),
    "model_path": str(model_path),
    "module_prefix": args.module_prefix,
    "device": str(device),
    "input_shape": [args.batch_tokens, args.in_features],
    "input_dtype": args.input_dtype,
    "marlin_input_dtype": args.marlin_input_dtype,
    "compute_dtype": args.compute_dtype,
    "output_dtype": args.output_dtype,
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "env": {
      "torch": torch.__version__,
      "torch_cuda": torch.version.cuda,
      "cuda_capability": list(torch.cuda.get_device_capability(device)),
      "device_name": torch.cuda.get_device_name(device),
    },
    "warmup_ms": [],
    "iter_ms": [],
  }

  ensure_native_paro_runtime()

  module = NativeParoMarlinLinear(
    args.in_features,
    args.out_features,
    marlin_input_dtype=args.marlin_input_dtype,
    compute_dtype=dtype_from_name(args.compute_dtype),
    output_dtype=args.output_dtype,
  ).to(device=device)
  load_one_module_tensors(model_path, args.module_prefix, module, device=device)
  module.eval()
  x = torch.randn(
    (args.batch_tokens, args.in_features),
    device=device,
    dtype=dtype_from_name(args.input_dtype),
  )

  with torch.inference_mode():
    for _ in range(max(args.warmup, 0)):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      out = module(x)
      torch.cuda.synchronize()
      report["warmup_ms"].append((time.perf_counter() - t0) * 1000.0)
    iter_ms = []
    out = None
    for _ in range(max(args.iters, 0)):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      out = module(x)
      torch.cuda.synchronize()
      iter_ms.append((time.perf_counter() - t0) * 1000.0)
  report["iter_ms"] = iter_ms
  report["mean_ms"] = sum(iter_ms) / len(iter_ms) if iter_ms else None
  report["output_shape"] = list(out.shape) if out is not None else None
  report["output_dtype_actual"] = str(out.dtype) if out is not None else None
  report["status"] = "ok"
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description="Native PARO rotate+Marlin helper/probe.")
  sub = parser.add_subparsers(dest="cmd", required=True)
  probe = sub.add_parser("one-module", help="Load and time one nested PARO linear module.")
  probe.add_argument("--model-path", default=DEFAULT_PARO_MODEL_PATH)
  probe.add_argument("--module-prefix", default=DEFAULT_MODULE_PREFIX)
  probe.add_argument("--device", default="cuda:0")
  probe.add_argument("--batch-tokens", type=int, default=1)
  probe.add_argument("--in-features", type=int, default=4096)
  probe.add_argument("--out-features", type=int, default=4096)
  probe.add_argument("--input-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  probe.add_argument("--compute-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  probe.add_argument("--output-dtype", choices=("input", "compute", "native"), default="input")
  probe.add_argument("--marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  probe.add_argument("--warmup", type=int, default=2)
  probe.add_argument("--iters", type=int, default=8)
  probe.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  probe.add_argument("--output", type=Path)
  args = parser.parse_args()

  try:
    report = _run_one_module_probe(args)
  except Exception as exc:
    report = {
      "status": "error",
      "type": type(exc).__name__,
      "message": str(exc),
      "created_unix": time.time(),
    }

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
