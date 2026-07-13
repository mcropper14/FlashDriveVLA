#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from paro_native_marlin import (
  apply_native_paro_linear_replacements,
  dtype_from_name,
  finalize_native_paro_modules_for_device_map,
)


_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
DEFAULT_PARO_MODEL_PATH = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
DEFAULT_ALPAMAYO_ROOT_WSL = Path("/mnt/g/alpamayo1.5")
DEFAULT_ALPAMAYO_ROOT_WIN = Path(r"G:\alpamayo1.5")


def windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def normalize_path(raw: str | os.PathLike[str]) -> Path:
  return Path(os.path.expanduser(windows_to_wsl(str(raw))))


def running_under_wsl() -> bool:
  return "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False


def default_alpamayo_root() -> Path:
  return DEFAULT_ALPAMAYO_ROOT_WSL if running_under_wsl() else DEFAULT_ALPAMAYO_ROOT_WIN


def insert_alpamayo_paths(root: Path) -> None:
  for path in (root, root / "src"):
    text = str(path)
    if text not in sys.path:
      sys.path.insert(0, text)


def build_split_device_map(split_index: int) -> dict[str, int]:
  device_map: dict[str, int] = {
    "vlm.model.visual": 0,
    "vlm.model.language_model.embed_tokens": 0,
    "vlm.model.language_model.norm": 1,
    "vlm.model.language_model.rotary_emb": 1,
    "vlm.lm_head": 1,
    "expert.norm": 1,
    "expert.rotary_emb": 1,
    "action_in_proj": 0,
    "action_out_proj": 1,
    "action_space": 0,
    "diffusion": 0,
  }
  for layer_idx in range(36):
    device = 0 if layer_idx < split_index else 1
    device_map[f"vlm.model.language_model.layers.{layer_idx}"] = device
    device_map[f"expert.layers.{layer_idx}"] = device
  return device_map


def gpu_only_max_memory(gpu_mem_gib: int) -> dict[int, str]:
  if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise RuntimeError("Alpamayo requires CUDA; CPU model offload is forbidden for latency benchmarks")
  return {idx: f"{gpu_mem_gib}GiB" for idx in range(torch.cuda.device_count())}


def assert_no_cpu_offload(model: Any) -> None:
  device_map = getattr(model, "hf_device_map", {}) or {}
  bad = {
    name: device
    for name, device in device_map.items()
    if str(device).lower() in ("cpu", "disk") or str(device).lower().startswith("cpu")
  }
  if bad:
    preview = dict(list(bad.items())[:12])
    raise RuntimeError(f"CPU/disk offload is forbidden for Alpamayo latency runs; offending device_map entries={preview}")


def patch_flattened_conv3d_loader() -> Any:
  original = torch.nn.Module.load_state_dict

  def patched(module: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(module, torch.nn.Conv3d) and "weight" in state_dict:
      weight = state_dict["weight"]
      target = module.weight
      if (
        getattr(weight, "ndim", None) == 2
        and tuple(target.shape)[0] == int(weight.shape[0])
        and int(weight.numel()) == int(target.numel())
      ):
        state_dict = dict(state_dict)
        state_dict["weight"] = weight.reshape_as(target)
    return original(module, state_dict, *args, **kwargs)

  torch.nn.Module.load_state_dict = patched

  def restore() -> None:
    torch.nn.Module.load_state_dict = original

  return restore


def patch_tie_weights_compat() -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  import alpamayo1_5.models.base_model as base_mod

  def tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
    if hasattr(self, "vlm") and hasattr(self.vlm, "tie_weights"):
      return self.vlm.tie_weights()
    return None

  base_mod.ReasoningVLA.tie_weights = tie_weights
  alpamayo_mod.Alpamayo1_5.tie_weights = tie_weights


def patch_alpamayo_init_for_paro(
  *,
  marlin_input_dtype: str,
  compute_dtype: torch.dtype,
  output_dtype: str,
) -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod

  if getattr(alpamayo_mod.Alpamayo1_5, "_openpilot_paro_patch_applied", False):
    return

  original_init = alpamayo_mod.Alpamayo1_5.__init__

  def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    original_init(self, *args, **kwargs)
    records = apply_native_paro_linear_replacements(
      self,
      marlin_input_dtype=marlin_input_dtype,
      compute_dtype=compute_dtype,
      output_dtype=output_dtype,
    )
    self._openpilot_paro_replacement_records = records

  alpamayo_mod.Alpamayo1_5.__init__ = patched_init
  alpamayo_mod.Alpamayo1_5._openpilot_paro_patch_applied = True


def set_attention_implementation(module: Any, attn_implementation: str) -> None:
  for submodule in module.modules():
    config = getattr(submodule, "config", None)
    if config is None:
      continue
    setattr(config, "_attn_implementation", attn_implementation)
    if hasattr(config, "attn_implementation"):
      setattr(config, "attn_implementation", attn_implementation)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
  return {
    "shape": list(tensor.shape),
    "dtype": str(tensor.dtype),
    "device": str(tensor.device),
  }


def repair_flattened_visual_patch_embed(model: Any, model_path: Path) -> dict[str, Any]:
  key = "vlm.model.visual.patch_embed.proj.weight"
  index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
  weight_map = index.get("weight_map", {})
  if key not in weight_map:
    return {"status": "skipped", "reason": "key_not_in_index", "key": key}

  target = model.vlm.model.visual.patch_embed.proj.weight
  with safe_open(str(model_path / weight_map[key]), framework="pt", device="cpu") as handle:
    weight = handle.get_tensor(key)
  if weight.ndim != 2 or int(weight.numel()) != int(target.numel()):
    return {
      "status": "skipped",
      "reason": "shape_not_compatible",
      "key": key,
      "source": tensor_summary(weight),
      "target": tensor_summary(target),
    }
  with torch.no_grad():
    target.copy_(weight.reshape_as(target).to(device=target.device, dtype=target.dtype))
  return {
    "status": "ok",
    "key": key,
    "source_shape": list(weight.shape),
    "target": tensor_summary(target),
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  if args.torch_cuda_arch_list:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)
  os.environ.setdefault("HF_HUB_OFFLINE", "1")
  os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

  alpamayo_root = normalize_path(args.alpamayo_root)
  model_path = normalize_path(args.model_path)
  insert_alpamayo_paths(alpamayo_root)
  patch_tie_weights_compat()
  patch_alpamayo_init_for_paro(
    marlin_input_dtype=args.marlin_input_dtype,
    compute_dtype=dtype_from_name(args.compute_dtype),
    output_dtype=args.output_dtype,
  )
  import paroquant.inference.backends.vllm.plugin  # noqa: F401

  from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

  if not model_path.exists():
    raise FileNotFoundError(f"PARO model path not found: {model_path}")

  if args.device_map_mode == "split":
    device_map: str | dict[str, int] | None = build_split_device_map(args.split_index)
  elif args.device_map_mode == "auto":
    device_map = "auto"
  elif args.device_map_mode == "none":
    device_map = None
  else:
    raise ValueError(f"unsupported device map mode {args.device_map_mode}")

  max_memory = None
  if device_map is not None:
    max_memory = gpu_only_max_memory(args.gpu_mem_gib)

  load_kwargs: dict[str, Any] = {
    "dtype": dtype_from_name(args.model_dtype),
    "attn_implementation": args.attn_implementation,
    "low_cpu_mem_usage": True,
    "ignore_mismatched_sizes": True,
  }
  if device_map is not None:
    load_kwargs["device_map"] = device_map
    load_kwargs["max_memory"] = max_memory

  report: dict[str, Any] = {
    "status": "started",
    "created_unix": time.time(),
    "model_path": str(model_path),
    "alpamayo_root": str(alpamayo_root),
    "env": {
      "torch": torch.__version__,
      "torch_cuda": torch.version.cuda,
      "cuda_available": torch.cuda.is_available(),
      "cuda_device_count": torch.cuda.device_count(),
      "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    },
    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    "load_kwargs": {
      key: (value if isinstance(value, (str, int, float, bool, type(None))) else str(value))
      for key, value in load_kwargs.items()
    },
  }
  if torch.cuda.is_available():
    report["env"]["devices"] = [
      {
        "index": idx,
        "name": torch.cuda.get_device_name(idx),
        "capability": list(torch.cuda.get_device_capability(idx)),
      }
      for idx in range(torch.cuda.device_count())
    ]

  restore_loader = patch_flattened_conv3d_loader()
  load_start = time.perf_counter()
  try:
    model = Alpamayo1_5.from_pretrained(str(model_path), **load_kwargs)
  finally:
    restore_loader()
  assert_no_cpu_offload(model)
  report["visual_patch_embed_repair"] = repair_flattened_visual_patch_embed(model, model_path)
  if args.expert_attn_implementation:
    set_attention_implementation(model.expert, args.expert_attn_implementation)
  report["paro_finalize"] = finalize_native_paro_modules_for_device_map(model)
  load_seconds = time.perf_counter() - load_start
  model.eval()
  if torch.cuda.is_available():
    torch.cuda.synchronize()

  records = getattr(model, "_openpilot_paro_replacement_records", [])
  report["load_seconds"] = load_seconds
  report["replacement_count"] = len(records)
  report["replacement_first"] = [record.__dict__ for record in records[:10]]
  report["model_device"] = str(getattr(model, "device", "unknown"))
  report["hf_device_map"] = getattr(model, "hf_device_map", None)

  first = model.get_submodule("vlm.model.language_model.layers.0.self_attn.q_proj")
  report["first_qproj"] = {
    "class": first.__class__.__name__,
    "in_features": getattr(first, "in_features", None),
    "out_features": getattr(first, "out_features", None),
    "qweight": tensor_summary(first.rotate_linear.qlinear.qweight)
    if hasattr(first, "rotate_linear")
    else None,
  }

  if args.forward_smoke:
    device = first.rotate_linear.qlinear.qweight.device
    x = torch.randn((args.forward_tokens, first.in_features), dtype=dtype_from_name(args.compute_dtype), device=device)
    with torch.inference_mode():
      warmup_ms: list[float] = []
      for _ in range(max(args.forward_warmup, 0)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = first(x)
        torch.cuda.synchronize()
        warmup_ms.append((time.perf_counter() - t0) * 1000.0)
      iter_ms: list[float] = []
      out = None
      for _ in range(max(args.forward_iters, 0)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = first(x)
        torch.cuda.synchronize()
        iter_ms.append((time.perf_counter() - t0) * 1000.0)
    report["forward_smoke"] = {
      "status": "ok",
      "warmup_ms": warmup_ms,
      "iter_ms": iter_ms,
      "mean_ms": sum(iter_ms) / len(iter_ms) if iter_ms else None,
      "output": tensor_summary(out) if out is not None else None,
    }

  report["status"] = "ok"
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description="Load Alpamayo PARO with native Marlin VLM linears installed before checkpoint load.")
  parser.add_argument("--model-path", default=DEFAULT_PARO_MODEL_PATH)
  parser.add_argument("--alpamayo-root", default=str(default_alpamayo_root()))
  parser.add_argument("--device-map-mode", choices=("split", "auto", "none"), default="split")
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--model-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  parser.add_argument("--compute-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  parser.add_argument("--output-dtype", choices=("input", "compute", "native"), default="native")
  parser.add_argument("--marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  parser.add_argument("--attn-implementation", default="sdpa")
  parser.add_argument("--expert-attn-implementation", default="eager")
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--forward-smoke", action="store_true")
  parser.add_argument("--forward-tokens", type=int, default=1)
  parser.add_argument("--forward-warmup", type=int, default=1)
  parser.add_argument("--forward-iters", type=int, default=2)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  try:
    report = run(args)
  except Exception as exc:
    report = {
      "status": "error",
      "created_unix": time.time(),
      "type": type(exc).__name__,
      "message": str(exc),
      "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
