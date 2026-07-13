from __future__ import annotations

import argparse
import importlib.util
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
_AWQ_GEMM_PATH = (
  "/mnt/g/alpamayo1.5/a1_5_venv/lib/python3.12/site-packages/awq/modules/triton/gemm.py"
)
_AWQ_SPLIT_K_ITERS = 4
_ROTATE_GROUP_SIZE = 128
_AWQ_REQUIRED_KEYS = {
  "theta": "rotate_linear.rotation.theta",
  "pairs": "rotate_linear.rotation.pairs",
  "channel_scales": "rotate_linear.rotation.channel_scales",
  "qweight": "rotate_linear.qlinear.qweight",
  "qzeros": "rotate_linear.qlinear.qzeros",
  "scales": "rotate_linear.qlinear.scales",
}


def _windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None:
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def _normalize_path(raw: str) -> Path:
  normalized = _windows_to_wsl(raw) if os.name == "posix" else raw
  return Path(os.path.expanduser(normalized))


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
  try:
    return json.loads(path.read_text(encoding="utf-8")), None
  except FileNotFoundError:
    return None, f"missing::{path}"
  except json.JSONDecodeError as exc:
    return None, f"json_decode_error::{exc}"
  except OSError as exc:
    return None, f"read_error::{exc}"


def _collect_compatibility_issues(
  input_features: int,
  qweight: torch.Tensor,
  qzeros: torch.Tensor,
  scales: torch.Tensor,
) -> list[str]:
  issues: list[str] = []

  if qweight.ndim != 2:
    issues.append(f"qweight must be 2D, got {list(qweight.shape)}")
  if qzeros.ndim != 2:
    issues.append(f"qzeros must be 2D, got {list(qzeros.shape)}")
  if scales.ndim != 2:
    issues.append(f"scales must be 2D, got {list(scales.shape)}")
  if issues:
    return issues

  n_from_qweight = int(qweight.shape[1]) * 8
  group_size = -1
  if input_features <= 0:
    issues.append(f"input_features must be >0, got {input_features}")
  if qweight.shape[0] != input_features:
    issues.append(
      f"AWQ expects qweight.shape[0]==input_features. got input_features={input_features}, qweight.shape[0]={qweight.shape[0]}"
    )

  if qweight.shape[0] % 8 != 0:
    issues.append("qweight.shape[0] must be >=8 for AWQ dequant layout check")

  if qzeros.shape[1] != qweight.shape[1]:
    issues.append(
      f"AWQ expects qzeros.shape[1] == qweight.shape[1] (N/8). "
      f"got qweight.shape[1]={qweight.shape[1]}, qzeros.shape[1]={qzeros.shape[1]}"
    )

  if qweight.shape[0] <= 0 or qzeros.shape[0] <= 0:
    issues.append(
      f"AWQ requires qweight.shape[0] and qzeros.shape[0] > 0, got {qweight.shape[0]} and {qzeros.shape[0]}"
    )
  elif qweight.shape[0] % qzeros.shape[0] != 0:
    issues.append(
      f"AWQ expects K % qzeros.shape[0] == 0. got K={qweight.shape[0]}, qzeros.shape[0]={qzeros.shape[0]}"
    )
  else:
    group_size = qweight.shape[0] // qzeros.shape[0]
    if scales.shape[0] != qzeros.shape[0]:
      issues.append(
        f"AWQ expects scales.shape[0] == qzeros.shape[0]. got {scales.shape[0]} vs {qzeros.shape[0]}"
      )
    if scales.shape[1] != n_from_qweight:
      issues.append(
        f"AWQ expects scales.shape[1] == N (qweight.shape[1]*8). "
        f"expected {n_from_qweight}, got {scales.shape[1]}"
      )

  if qweight.dtype not in (torch.int32, torch.int16, torch.uint8):
    issues.append(f"qweight dtype usually int32 for AWQ, got {qweight.dtype}")
  if qzeros.dtype not in (torch.int32, torch.int16, torch.uint8):
    issues.append(f"qzeros dtype usually int32 for AWQ, got {qzeros.dtype}")
  if group_size not in (-1, 32, 64, 128) and group_size != qweight.shape[0]:
    issues.append(f"AWQ group_size={group_size} unsupported by observed kernel policy.")

  return issues


def _full_keys(prefix: str) -> dict[str, str]:
  return {name: f"{prefix}.{suffix}" for name, suffix in _AWQ_REQUIRED_KEYS.items()}


def _load_selected_tensors(
  model_path: Path,
  required: dict[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]], list[str], list[str], bool]:
  index_path = model_path / "model.safetensors.index.json"
  index, index_error = _read_json(index_path)
  if index is None:
    return {}, {}, [f"index_read_error::{index_error}"], [], False

  weight_map = index.get("weight_map")
  if not isinstance(weight_map, dict):
    return {}, {}, ["index_missing_weight_map"], [], False

  loaded: dict[str, torch.Tensor] = {}
  tensor_meta: dict[str, dict[str, str]] = {}
  shard_map: dict[str, list[tuple[str, str]]] = {}
  missing: list[str] = []
  notes: list[str] = []

  for label, key in required.items():
    shard = weight_map.get(key)
    if not isinstance(shard, str):
      missing.append(f"weight_map_missing::{key}")
      continue
    shard_map.setdefault(shard, []).append((label, key))

  if missing:
    return loaded, tensor_meta, missing, notes, False

  for shard_name, items in shard_map.items():
    shard_path = model_path / shard_name
    if not shard_path.exists():
      return loaded, tensor_meta, [f"missing_shard::{shard_path}"], notes, False
    try:
      with safe_open(shard_path.as_posix(), framework="torch", device="cpu") as f:
        for label, key in items:
          if key not in f.keys():
            missing.append(f"shard_missing_key::{key}")
            continue
          tensor = f.get_tensor(key)
          loaded[key] = tensor
          tensor_meta[label] = {"key": key, "shard": shard_name, "path": str(shard_path)}
    except Exception as exc:
      return loaded, tensor_meta, [f"safetensors_open_error::{shard_path}::{exc}"], notes, False

  if missing:
    return loaded, tensor_meta, missing, notes, False

  notes.append("loaded_selected_tensors_ok")
  return loaded, tensor_meta, [], notes, True


def _load_awq_module():
  path = Path(_AWQ_GEMM_PATH)
  if not path.exists():
    return None, [f"missing_awq_module::{path}"]

  spec = importlib.util.spec_from_file_location("probe_paro_awq_gemm", path.as_posix())
  if spec is None or spec.loader is None:
    return None, ["cannot_create_import_spec::probe_paro_awq_gemm"]

  module = importlib.util.module_from_spec(spec)
  try:
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, []
  except Exception as exc:
    return None, [f"awq_import_error::{type(exc).__name__}::{exc}"]


def _load_rotation_ops() -> tuple[bool, list[str]]:
  try:
    import paroquant.kernels.cuda  # noqa: F401
  except Exception as exc:
    return False, [f"paroquant_import_error::{type(exc).__name__}::{exc}"]

  try:
    _ = torch.ops.rotation.rotate
    return True, []
  except AttributeError as exc:
    return False, [f"missing_rotate_op::{type(exc).__name__}::{exc}"]


def _json_dump(payload: dict[str, Any], output: Path | None) -> None:
  text = json.dumps(payload, indent=2, sort_keys=True)
  print(text)
  if output is not None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _build_report() -> dict[str, Any]:
  return {
    "status": "ok",
    "status_reason": "start",
    "created_unix": time.time(),
    "tensors": {},
    "compatibility": {
      "status": "unknown",
      "checked_layout": "awq_triton_plain",
      "issues": [],
    },
    "timing": {
      "mean_ms": None,
      "warmup_ms": [],
      "iter_ms": [],
    },
    "output_shapes": {},
    "dtype_summary": {},
    "errors": [],
    "notes": [],
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Smoke-test one nested PARO linear shard with rotate + AWQ Triton GEMM."
  )
  parser.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
  parser.add_argument("--module-prefix", default=_DEFAULT_MODULE_PREFIX)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-tokens", type=int, default=1)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument(
    "--force-run",
    action="store_true",
    help="Attempt rotation/AWQ execution even if the static layout check fails.",
  )
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  report = _build_report()
  report["model_path"] = str(_normalize_path(args.model_path))
  report["module_prefix"] = args.module_prefix
  report["device"] = args.device
  report["batch_tokens"] = args.batch_tokens
  report["warmup"] = args.warmup
  report["iters"] = args.iters

  if args.batch_tokens <= 0:
    report["status"] = "error"
    report["status_reason"] = "invalid_batch_tokens"
    report["errors"].append(f"batch_tokens must be >=1, got {args.batch_tokens}")
    _json_dump(report, args.output)
    return 1

  model_path = _normalize_path(args.model_path)
  report["required_keys"] = _full_keys(args.module_prefix)
  selected, tensor_meta, load_errors, load_notes, ok = _load_selected_tensors(model_path, report["required_keys"])
  report["notes"].extend(load_notes)
  if load_errors:
    report["status"] = "error"
    report["status_reason"] = "tensor_load_error"
    report["errors"].extend(load_errors)
    report["tensors"] = {
      label: {"path": str(model_path / "unknown"), "loaded": False, "error": f"missing::{full_key}"}
      for label, full_key in report["required_keys"].items()
    }
    _json_dump(report, args.output)
    return 1

  for label, full_key in report["required_keys"].items():
    meta = tensor_meta.get(label, {})
    tensor = selected.get(full_key)
    if tensor is None:
      report["tensors"][label] = {
        "path": meta.get("path", ""),
        "loaded": False,
        "key": full_key,
      }
      continue

    report["tensors"][label] = {
      "path": meta.get("path", ""),
      "loaded": True,
      "key": full_key,
      "shape": list(tensor.shape),
      "dtype": str(tensor.dtype),
      "device": str(tensor.device),
    }
    report["dtype_summary"][label] = {
      "shape": list(tensor.shape),
      "dtype": str(tensor.dtype),
      "device": str(tensor.device),
    }

  if not ok:
    report["status"] = "error"
    report["status_reason"] = "tensor_load_error"
    report["errors"].append("did_not_load_all_required_tensors")
    _json_dump(report, args.output)
    return 1

  try:
    input_features = int(selected[report["required_keys"]["pairs"]].shape[-1])
    report["output_shapes"]["static_input"] = [args.batch_tokens, input_features]
    static_issues = _collect_compatibility_issues(
      input_features,
      selected[report["required_keys"]["qweight"]],
      selected[report["required_keys"]["qzeros"]],
      selected[report["required_keys"]["scales"]],
    )
  except Exception as exc:
    report["status"] = "error"
    report["status_reason"] = "static_layout_check_error"
    report["errors"].append(f"static_layout_check_error::{type(exc).__name__}::{exc}")
    _json_dump(report, args.output)
    return 1

  report["compatibility"]["issues"] = static_issues
  if static_issues and not args.force_run:
    report["compatibility"]["status"] = "incompatible"
    report["status"] = "error"
    report["status_reason"] = "shape_layout_mismatch"
    report["errors"].append("static AWQ plain triton layout check failed")
    _json_dump(report, args.output)
    return 1
  if static_issues:
    report["compatibility"]["status"] = "incompatible_forced"
    report["notes"].append("force_run enabled despite static layout mismatch")
  else:
    report["compatibility"]["status"] = "compatible"

  if not torch.cuda.is_available():
    report["status"] = "error"
    report["status_reason"] = "cuda_unavailable"
    report["errors"].append("torch.cuda.is_available()==False")
    _json_dump(report, args.output)
    return 1

  device = torch.device(args.device)
  if device.type != "cuda":
    report["status"] = "error"
    report["status_reason"] = "non_cuda_device"
    report["errors"].append(f"requested non-cuda device {args.device}")
    _json_dump(report, args.output)
    return 1

  rotate_ok, rotate_errors = _load_rotation_ops()
  if not rotate_ok:
    report["status"] = "error"
    report["status_reason"] = "rotation_import_error"
    report["errors"].extend(rotate_errors)
    _json_dump(report, args.output)
    return 1

  awq_module, awq_errors = _load_awq_module()
  if awq_module is None:
    report["status"] = "error"
    report["status_reason"] = "awq_import_error"
    report["errors"].extend(awq_errors)
    _json_dump(report, args.output)
    return 1

  try:
    theta = selected[report["required_keys"]["theta"]].to(device=device, dtype=torch.float16)
    pairs = selected[report["required_keys"]["pairs"]].to(device=device, dtype=torch.int16)
    channel_scales = selected[report["required_keys"]["channel_scales"]].to(device=device, dtype=torch.float16)
    qweight = selected[report["required_keys"]["qweight"]].to(device=device)
    qzeros = selected[report["required_keys"]["qzeros"]].to(device=device)
    scales = selected[report["required_keys"]["scales"]].to(device=device, dtype=torch.float16)
  except Exception as exc:
    report["status"] = "error"
    report["status_reason"] = "tensor_move_error"
    report["errors"].append(f"tensor_to_device_error::{type(exc).__name__}::{exc}")
    _json_dump(report, args.output)
    return 1

  in_features = int(pairs.shape[-1])
  if in_features <= 0:
    report["status"] = "error"
    report["status_reason"] = "invalid_in_features"
    report["errors"].append(f"rotation input features must be >0, got {in_features}")
    _json_dump(report, args.output)
    return 1

  x = torch.randn((args.batch_tokens, in_features), device=device, dtype=torch.float16)
  report["output_shapes"]["input"] = list(x.shape)

  try:
    rotated = torch.ops.rotation.rotate(x, pairs, theta, channel_scales, _ROTATE_GROUP_SIZE)
    report["output_shapes"]["rotated"] = list(rotated.shape)
  except Exception as exc:
    report["status"] = "error"
    report["status_reason"] = "rotate_run_error"
    report["errors"].append(f"rotate_error::{type(exc).__name__}::{exc}")
    _json_dump(report, args.output)
    return 1

  try:
    dequantized = awq_module.awq_dequantize_triton(qweight, scales, qzeros)
    report["output_shapes"]["awq_dequantized_weight"] = list(dequantized.shape)

    def _run_once() -> torch.Tensor:
      rotated_local = torch.ops.rotation.rotate(x, pairs, theta, channel_scales, _ROTATE_GROUP_SIZE)
      out = awq_module.awq_gemm_triton(
        rotated_local,
        qweight,
        scales,
        qzeros,
        _AWQ_SPLIT_K_ITERS,
      )
      return out.reshape(rotated_local.shape[0], -1)

    for _ in range(max(args.warmup, 0)):
      torch.cuda.synchronize(device=device)
      t0 = time.perf_counter()
      out = _run_once()
      torch.cuda.synchronize(device=device)
      report["timing"]["warmup_ms"].append((time.perf_counter() - t0) * 1000.0)

    iters = max(args.iters, 0)
    if iters:
      iter_times: list[float] = []
      for _ in range(iters):
        torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        out = _run_once()
        torch.cuda.synchronize(device=device)
        iter_times.append((time.perf_counter() - t0) * 1000.0)
      report["timing"]["iter_ms"] = iter_times
      report["timing"]["mean_ms"] = sum(iter_times) / len(iter_times) if iter_times else None
      report["output_shapes"]["awq_output_reshaped"] = list(out.shape)

    report["status_reason"] = "completed"
    _json_dump(report, args.output)
    return 0
  except Exception as exc:
    report["status"] = "error"
    message = str(exc)
    if (
      "shape" in message.lower()
      or "size" in message.lower()
      or "assertion" in message.lower()
      or "shape mismatch" in message.lower()
      or "shapes" in message.lower()
      or "qweight" in message.lower()
      or "qzeros" in message.lower()
      or "scales" in message.lower()
    ):
      report["status_reason"] = "shape_layout_mismatch"
      report["compatibility"]["status"] = "incompatible"
      report["compatibility"]["issues"].append(f"runtime_shape_layout_error::{type(exc).__name__}::{message}")
    else:
      report["status_reason"] = "awq_runtime_error"
    report["errors"].append(f"awq_runtime_error::{type(exc).__name__}::{message}")
    _json_dump(report, args.output)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
