#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open


_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_DEFAULT_MODEL_PATH = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
_DEFAULT_MODULE_PREFIX = "vlm.model.language_model.layers.0.self_attn.q_proj"
_REQUIRED_SUFFIXES = {
  "theta": "rotate_linear.rotation.theta",
  "pairs": "rotate_linear.rotation.pairs",
  "channel_scales": "rotate_linear.rotation.channel_scales",
  "qweight": "rotate_linear.qlinear.qweight",
  "qzeros": "rotate_linear.qlinear.qzeros",
  "scales": "rotate_linear.qlinear.scales",
  "g_idx": "rotate_linear.qlinear.g_idx",
  "g_idx_sort_indices": "rotate_linear.qlinear.g_idx_sort_indices",
  "input_global_scale": "rotate_linear.qlinear.input_global_scale",
  "workspace": "rotate_linear.qlinear.workspace",
}


def _windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def _normalize_path(raw: str) -> Path:
  return Path(os.path.expanduser(_windows_to_wsl(raw)))


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_tensors(model_path: Path, module_prefix: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  index = _read_json(model_path / "model.safetensors.index.json")
  weight_map = index.get("weight_map")
  if not isinstance(weight_map, dict):
    raise ValueError(f"missing weight_map in {model_path / 'model.safetensors.index.json'}")

  full_keys = {label: f"{module_prefix}.{suffix}" for label, suffix in _REQUIRED_SUFFIXES.items()}
  missing = [key for key in full_keys.values() if key not in weight_map]
  if missing:
    raise KeyError(f"missing keys in index: {missing[:8]}")

  shard_to_items: dict[str, list[tuple[str, str]]] = {}
  for label, key in full_keys.items():
    shard_to_items.setdefault(str(weight_map[key]), []).append((label, key))

  loaded: dict[str, torch.Tensor] = {}
  meta: dict[str, Any] = {"keys": full_keys, "shards": {}}
  for shard_name, items in sorted(shard_to_items.items()):
    shard_path = model_path / shard_name
    meta["shards"][shard_name] = [label for label, _ in items]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
      for label, key in items:
        loaded[label] = handle.get_tensor(key)

  return loaded, meta


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
  return {
    "shape": list(tensor.shape),
    "dtype": str(tensor.dtype),
    "device": str(tensor.device),
  }


def _dtype_from_name(name: str) -> torch.dtype:
  if name == "fp16":
    return torch.float16
  if name == "bf16":
    return torch.bfloat16
  if name == "fp32":
    return torch.float32
  raise ValueError(f"unsupported dtype {name}")


def _marlin_input_dtype(name: str) -> torch.dtype | None:
  if name == "none":
    return None
  if name == "int8":
    return torch.int8
  if name == "fp8":
    return torch.float8_e4m3fn
  raise ValueError(f"unsupported marlin input mode {name}")


def _run_mode(
  *,
  mode: str,
  x: torch.Tensor,
  pairs: torch.Tensor,
  theta: torch.Tensor,
  channel_scales: torch.Tensor,
  qweight: torch.Tensor,
  qzeros: torch.Tensor,
  scales: torch.Tensor,
  g_idx: torch.Tensor,
  g_idx_sort_indices: torch.Tensor,
  workspace: torch.Tensor,
  input_global_scale: torch.Tensor,
  out_features: int,
  in_features: int,
  warmup: int,
  iters: int,
) -> dict[str, Any]:
  from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_awq_marlin_linear
  from vllm.scalar_type import scalar_types

  input_dtype = _marlin_input_dtype(mode)
  result: dict[str, Any] = {
    "mode": mode,
    "status": "started",
    "warmup_ms": [],
    "iter_ms": [],
    "mean_ms": None,
    "output_shape": None,
    "error": None,
  }

  def _once() -> torch.Tensor:
    rotated = torch.ops.rotation.rotate(x, pairs, theta, channel_scales, 128)
    return apply_awq_marlin_linear(
      input=rotated,
      weight=qweight,
      weight_scale=scales,
      weight_zp=qzeros,
      g_idx=g_idx,
      g_idx_sort_indices=g_idx_sort_indices,
      workspace=workspace,
      quant_type=scalar_types.uint4,
      output_size_per_partition=out_features,
      input_size_per_partition=in_features,
      input_global_scale=input_global_scale,
      input_dtype=input_dtype,
    )

  try:
    for _ in range(max(warmup, 0)):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      out = _once()
      torch.cuda.synchronize()
      result["warmup_ms"].append((time.perf_counter() - t0) * 1000.0)

    iter_times: list[float] = []
    out = None
    for _ in range(max(iters, 0)):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      out = _once()
      torch.cuda.synchronize()
      iter_times.append((time.perf_counter() - t0) * 1000.0)

    result["iter_ms"] = iter_times
    result["mean_ms"] = sum(iter_times) / len(iter_times) if iter_times else None
    result["output_shape"] = list(out.shape) if out is not None else None
    result["status"] = "ok"
  except Exception as exc:
    result["status"] = "error"
    result["error"] = {"type": type(exc).__name__, "message": str(exc)}
  return result


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Probe one PARO native rotate+Marlin linear using vLLM kernels."
  )
  parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
  parser.add_argument("--module-prefix", default=_DEFAULT_MODULE_PREFIX)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-tokens", type=int, default=1)
  parser.add_argument("--input-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
  parser.add_argument("--modes", default="int8,none", help="Comma-separated marlin input modes: int8,fp8,none.")
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=8)
  parser.add_argument(
    "--torch-cuda-arch-list",
    default="8.9+PTX",
    help="Set before importing ParoQuant kernels; avoids nvcc 12.0 compute_120 failure on Blackwell.",
  )
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  if args.torch_cuda_arch_list:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)

  report: dict[str, Any] = {
    "status": "started",
    "created_unix": time.time(),
    "model_path": str(_normalize_path(args.model_path)),
    "module_prefix": args.module_prefix,
    "device": args.device,
    "batch_tokens": args.batch_tokens,
    "input_dtype": args.input_dtype,
    "modes": [mode.strip() for mode in args.modes.split(",") if mode.strip()],
    "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "env": {},
    "tensors": {},
    "shape_contract": {},
    "mode_results": [],
    "errors": [],
  }

  try:
    model_path = _normalize_path(args.model_path)
    tensors_cpu, load_meta = _load_tensors(model_path, args.module_prefix)
    report["load_meta"] = load_meta
  except Exception as exc:
    report["status"] = "error"
    report["errors"].append({"stage": "load_tensors", "type": type(exc).__name__, "message": str(exc)})
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1

  for label, tensor in tensors_cpu.items():
    report["tensors"][label] = _tensor_summary(tensor)

  if not torch.cuda.is_available():
    report["status"] = "error"
    report["errors"].append({"stage": "cuda", "type": "RuntimeError", "message": "torch.cuda.is_available() is false"})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1

  device = torch.device(args.device)
  torch.cuda.set_device(device)
  report["env"] = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_capability": list(torch.cuda.get_device_capability(device)),
    "device_name": torch.cuda.get_device_name(device),
  }

  try:
    import paroquant.inference.backends.vllm.plugin  # noqa: F401
  except Exception as exc:
    report["status"] = "error"
    report["errors"].append({"stage": "import_paroquant_vllm", "type": type(exc).__name__, "message": str(exc)})
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1

  dtype = _dtype_from_name(args.input_dtype)
  in_features = int(tensors_cpu["pairs"].shape[-1])
  out_features_from_scales = int(tensors_cpu["scales"].shape[-1])
  out_features_from_qweight = int(tensors_cpu["qweight"].shape[-1] // 2)
  report["shape_contract"] = {
    "in_features": in_features,
    "out_features_from_scales": out_features_from_scales,
    "out_features_from_qweight_marlin": out_features_from_qweight,
    "native_marlin_qweight_expected": [in_features // 16, out_features_from_scales * 2],
    "visible_paroquant_vllm_awq_expected": [in_features, out_features_from_scales // 8],
    "matches_native_marlin_qweight": list(tensors_cpu["qweight"].shape) == [in_features // 16, out_features_from_scales * 2],
    "matches_visible_vllm_plugin_awq_qweight": list(tensors_cpu["qweight"].shape) == [in_features, out_features_from_scales // 8],
  }

  x = torch.randn((args.batch_tokens, in_features), device=device, dtype=dtype)
  moved = {
    "pairs": tensors_cpu["pairs"].to(device=device, dtype=torch.int16),
    "theta": tensors_cpu["theta"].to(device=device, dtype=torch.float16),
    "channel_scales": tensors_cpu["channel_scales"].to(device=device, dtype=torch.float16),
    "qweight": tensors_cpu["qweight"].to(device=device, dtype=torch.int32),
    "qzeros": tensors_cpu["qzeros"].to(device=device, dtype=torch.int32),
    "scales": tensors_cpu["scales"].to(device=device, dtype=torch.float16),
    "g_idx": tensors_cpu["g_idx"].to(device=device, dtype=torch.int32),
    "g_idx_sort_indices": tensors_cpu["g_idx_sort_indices"].to(device=device, dtype=torch.int32),
    "workspace": tensors_cpu["workspace"].to(device=device, dtype=torch.int32),
    "input_global_scale": tensors_cpu["input_global_scale"].to(device=device, dtype=torch.float32),
  }

  try:
    rotated = torch.ops.rotation.rotate(
      x,
      moved["pairs"],
      moved["theta"],
      moved["channel_scales"],
      128,
    )
    torch.cuda.synchronize()
    report["rotation_smoke"] = {"status": "ok", "output_shape": list(rotated.shape)}
  except Exception as exc:
    report["rotation_smoke"] = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
    report["status"] = "error"
    report["errors"].append({"stage": "rotation", "type": type(exc).__name__, "message": str(exc)})

  if report.get("rotation_smoke", {}).get("status") == "ok":
    for mode in report["modes"]:
      report["mode_results"].append(
        _run_mode(
          mode=mode,
          x=x,
          pairs=moved["pairs"],
          theta=moved["theta"],
          channel_scales=moved["channel_scales"],
          qweight=moved["qweight"],
          qzeros=moved["qzeros"],
          scales=moved["scales"],
          g_idx=moved["g_idx"],
          g_idx_sort_indices=moved["g_idx_sort_indices"],
          workspace=moved["workspace"],
          input_global_scale=moved["input_global_scale"],
          out_features=out_features_from_scales,
          in_features=in_features,
          warmup=args.warmup,
          iters=args.iters,
        )
      )

  report["status"] = "ok" if any(item.get("status") == "ok" for item in report["mode_results"]) else report.get("status", "error")
  if report["status"] == "started":
    report["status"] = "error"

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
