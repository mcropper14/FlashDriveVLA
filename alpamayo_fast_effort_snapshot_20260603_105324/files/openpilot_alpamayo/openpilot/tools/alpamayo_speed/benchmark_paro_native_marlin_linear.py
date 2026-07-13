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

from paro_native_marlin_linear import (
  REQUIRED_STATE_SUFFIXES,
  OPTIONAL_STATE_SUFFIXES,
  TARGET_LINEAR_NAMES,
  ParoNativeMarlinLinear,
  iter_language_linear_modules,
  replace_language_linear_modules,
)


_DEFAULT_MODEL_PATH = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
_DEFAULT_MODULE_PREFIX = "vlm.model.language_model.layers.0.self_attn.q_proj"
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def _normalize_path(raw: str) -> Path:
  return Path(os.path.expanduser(_windows_to_wsl(raw)))


def _dtype_map(name: str) -> torch.dtype:
  if name == "fp16":
    return torch.float16
  if name == "bf16":
    return torch.bfloat16
  if name == "fp32":
    return torch.float32
  raise ValueError(f"unsupported dtype {name}")


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
  try:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    return data, None
  except Exception as exc:
    return {}, f"{type(exc).__name__}:{exc}"


def _load_nested_paro_tensors(
  model_path: Path,
  module_prefix: str,
  *,
  include_optional: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[str]]:
  index_path = model_path / "model.safetensors.index.json"
  index, index_error = _read_json(index_path)
  if index_error is not None:
    return {}, {}, [f"index_load_error::{index_error}"]

  weight_map = index.get("weight_map")
  if not isinstance(weight_map, dict):
    return {}, {}, ["index_missing_or_invalid_weight_map"]

  required = list(REQUIRED_STATE_SUFFIXES)
  if include_optional:
    required.extend(OPTIONAL_STATE_SUFFIXES)
  full_keys = [f"{module_prefix}.{suffix}" for suffix in required]
  missing = [key for key in full_keys if key not in weight_map]
  if missing:
    return {}, {}, [f"weight_map_missing::{len(missing)}::{missing[:8]}"]

  keys_by_shard: dict[str, list[tuple[str, str]]] = {}
  for key in full_keys:
    shard = str(weight_map[key])
    keys_by_shard.setdefault(shard, []).append((key, key))

  loaded: dict[str, torch.Tensor] = {}
  meta: dict[str, Any] = {
    "weight_map_entries": len(weight_map),
    "requested_entries": len(full_keys),
    "loaded_entries": 0,
    "shards": {},
  }
  for shard_name, entries in sorted(keys_by_shard.items()):
    shard_path = model_path / shard_name
    if not shard_path.exists():
      return {}, meta, [f"missing_shard::{shard_path}"]
    with safe_open(shard_path.as_posix(), framework="pt", device="cpu") as handle:
      for full_key, _ in entries:
        loaded[full_key] = handle.get_tensor(full_key)
        meta["loaded_entries"] += 1
        meta["shards"].setdefault(shard_name, 0)
        meta["shards"][shard_name] += 1
  return loaded, meta, []


def _shape_contract(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
  pairs = tensors["rotate_linear.rotation.pairs"]
  scales = tensors["rotate_linear.qlinear.scales"]
  return {
    "in_features": int(pairs.shape[-1]),
    "out_features": int(scales.shape[-1]),
    "input_dtype": str(pairs.dtype),
    "pairs_shape": list(pairs.shape),
    "scales_shape": list(scales.shape),
  }


def _timeit(fn) -> float:
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  out = fn()
  torch.cuda.synchronize()
  return out, (time.perf_counter() - t0) * 1000.0


def _run_smoke(
  *,
  model_path: Path,
  module_prefix: str,
  device: torch.device,
  dtype: torch.dtype,
  batch: int,
  warmup: int,
  iters: int,
) -> dict[str, Any]:
  report: dict[str, Any] = {
    "probe": "smoke",
    "module_prefix": module_prefix,
    "model_path": str(model_path),
    "requested_mode": "module",
  }

  tensors, meta, load_errors = _load_nested_paro_tensors(
    model_path,
    module_prefix,
    include_optional=True,
  )
  report["tensor_meta"] = meta
  if load_errors:
    report["status"] = "error"
    report["errors"] = load_errors
    return report

  if not torch.cuda.is_available():
    report["status"] = "error"
    report["errors"] = ["cuda_unavailable"]
    return report

  try:
    import paroquant.kernels.cuda  # noqa: F401
    import paroquant.inference.backends.vllm.plugin  # noqa: F401
  except Exception as exc:
    report["status"] = "error"
    report["errors"] = [f"runtime_import_failure::{type(exc).__name__}::{exc}"]
    return report

  local_tensors = {key[len(f"{module_prefix}."):]: value for key, value in tensors.items()}
  try:
    layer = ParoNativeMarlinLinear.from_nested_state_tensors(
      module_prefix="",
      tensors=local_tensors,
      rotation_group_size=128,
      strict=True,
    )
  except Exception as exc:
    report["status"] = "error"
    report["errors"] = [f"layer_build_failure::{type(exc).__name__}::{exc}"]
    return report

  layer = layer.to(device=device)
  x = torch.randn((batch, layer.in_features), device=device, dtype=dtype)
  contract = _shape_contract(local_tensors)
  report["shape_contract"] = contract

  try:
    _, warmup_ms = _timeit(lambda: layer(x))
    warmup_times: list[float] = [warmup_ms]
    for _ in range(max(warmup - 1, 0)):
      _, ms = _timeit(lambda: layer(x))
      warmup_times.append(ms)

    iter_times: list[float] = []
    out = None
    for _ in range(max(iters, 0)):
      out, ms = _timeit(lambda: layer(x))
      iter_times.append(ms)
    output_shape = list(out.shape) if out is not None else None

    report.update({
      "status": "ok",
      "output_shape": output_shape,
      "warmup_ms": warmup_times,
      "iter_ms": iter_times,
      "mean_ms": sum(iter_times) / len(iter_times) if iter_times else 0.0,
    })
  except Exception as exc:
    report["status"] = "error"
    report["errors"] = [f"forward_failure::{type(exc).__name__}::{exc}"]
  return report


def _run_patch_demo() -> dict[str, Any]:
  class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.self_attn = torch.nn.Module()
      self.mlp = torch.nn.Module()
      self.self_attn.q_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.self_attn.k_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.self_attn.v_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.self_attn.o_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.mlp.gate_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.mlp.up_proj = torch.nn.Linear(2048, 2048, bias=False)
      self.mlp.down_proj = torch.nn.Linear(2048, 2048, bias=False)

  class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.vlm = torch.nn.Module()
      self.vlm.model = torch.nn.Module()
      self.vlm.model.language_model = torch.nn.Module()
      self.vlm.model.language_model.layers = torch.nn.ModuleList([_TinyBlock()])
      self.action = torch.nn.Module()
      self.action.out_proj = torch.nn.Linear(64, 64, bias=False)

  model = _TinyModel()
  summary = replace_language_linear_modules(
    model,
    language_root="vlm.model.language_model.",
    rotation_group_size=128,
  )
  linears = iter_language_linear_modules(
    model,
    language_root="vlm.model.language_model.",
    target_names=TARGET_LINEAR_NAMES,
  )
  return {
    "status": "ok",
    "replacement": {
      "replaced": summary.replaced,
      "skipped": summary.skipped,
      "replaced_names": list(summary.replaced_names),
      "skipped_reasons": list(summary.skipped_reasons),
    },
    "post_replace_languages": [path for path, _ in linears],
  }


def main() -> int:
  parser = argparse.ArgumentParser(description="Benchmark a single PARO native nested linear using vLLM Marlin apply path.")
  parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
  parser.add_argument("--module-prefix", default=_DEFAULT_MODULE_PREFIX)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--iters", type=int, default=8)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--input-dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--patch-demo", action="store_true")
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  if args.torch_cuda_arch_list:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)

  report: dict[str, Any] = {
    "goal": "one_module_paro_native_marlin_probe",
    "created_unix": time.time(),
    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
  }

  if args.batch <= 0:
    report["status"] = "error"
    report["errors"] = ["batch must be >0"]
    _maybe_write_output(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1

  if args.patch_demo:
    report["patch_demo"] = _run_patch_demo()

  model_path = _normalize_path(args.model_path)
  report["smoke"] = _run_smoke(
    model_path=model_path,
    module_prefix=args.module_prefix,
    device=torch.device(args.device),
    dtype=_dtype_map(args.input_dtype),
    batch=args.batch,
    warmup=args.warmup,
    iters=args.iters,
  )

  if args.patch_demo and report["patch_demo"].get("status") != "ok":
    report["status"] = "error"
  elif report["smoke"].get("status") != "ok":
    report["status"] = "error"
  else:
    report["status"] = "ok"

  _maybe_write_output(args.output, report)
  print(json.dumps(report, indent=2, sort_keys=True))
  return 0 if report["status"] == "ok" else 1


def _maybe_write_output(output: Path | None, report: dict[str, Any]) -> None:
  if output is None:
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
  raise SystemExit(main())
