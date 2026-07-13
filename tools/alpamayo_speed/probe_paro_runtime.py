from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _windows_to_wsl_path(value: str) -> str:
  match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
  if match and os.name == "posix":
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"
  return value


def _normalize_path(raw: str) -> Path:
  return Path(_windows_to_wsl_path(raw))


def _read_json(path: Path) -> Any:
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _read_model_config(model_path: Path) -> tuple[dict[str, Any] | None, str | None]:
  config_path = model_path / "config.json"
  if not config_path.exists():
    return None, f"missing_config::{config_path}"
  try:
    return _read_json(config_path), None
  except Exception as exc:  # pragma: no cover
    return None, f"{type(exc).__name__}: {exc}"


def _collect_shards(model_path: Path) -> dict[str, Any]:
  checks: list[str] = []
  shard_info = []

  for stem in ("model", "pytorch_model"):
    names = [f"{stem}-{idx:05d}-of-00003.safetensors" for idx in range(1, 4)]
    infos = []
    for name in names:
      path = model_path / name
      infos.append({
        "name": name,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
      })
    # Use this stem only if at least one of the expected shard names exists
    if any(item["exists"] for item in infos):
      checks.append(stem)
      shard_info = infos
      break

  if not checks:
    checks.append("model")
    shard_info = [
      {
        "name": f"model-{idx:05d}-of-00003.safetensors",
        "exists": False,
        "size_bytes": 0,
      }
      for idx in range(1, 4)
    ]

  return {
    "stem": checks[0],
    "files": shard_info,
    "all_present": all(item["exists"] for item in shard_info),
  }


def _contains_w4a8(value: Any) -> tuple[bool, Any]:
  if isinstance(value, str):
    return ("w4a8" in value.lower(), value)
  if isinstance(value, dict):
    for key in ("w4a8", "w4a8_format", "format", "weight_format", "quantization"):
      if key in value:
        detected, extracted = _contains_w4a8(value[key])
        if detected:
          return True, extracted
    for item in value.values():
      detected, extracted = _contains_w4a8(item)
      if detected:
        return True, extracted
  if isinstance(value, (list, tuple)):
    for item in value:
      detected, extracted = _contains_w4a8(item)
      if detected:
        return True, extracted
  return False, None


def _detect_w4a8_format(model_path: Path, config: dict[str, Any] | None) -> tuple[str | None, str]:
  w4a8_config_path = model_path / "w4a8_config.json"
  if w4a8_config_path.exists():
    try:
      w4a8_cfg = _read_json(w4a8_config_path)
      detected, extracted = _contains_w4a8(w4a8_cfg)
      if detected:
        if isinstance(extracted, str):
          return extracted, "w4a8_config_json"
        return str(extracted), "w4a8_config_json"
      return None, "w4a8_config_json"
    except Exception as exc:  # pragma: no cover
      return None, f"w4a8_config_json_parse_error:{type(exc).__name__}:{exc}"

  if not config:
    return None, "config_missing"

  detected, extracted = _contains_w4a8(config)
  if detected:
    if isinstance(extracted, str):
      return extracted, "config"
    return str(extracted), "config"
  return None, "not_detected"


def _try_import_paroquant(*, attempt_runtime_imports: bool) -> dict[str, Any]:
  status: dict[str, Any] = {
    "base_import": {"ok": False, "error": None, "method": "importlib.metadata.distribution"},
    "module_imports": {},
    "runtime_like_modules": [],
    "has_runtime_modules": False,
    "runtime_imports_attempted": attempt_runtime_imports,
  }

  try:
    distribution = importlib.metadata.distribution("paroquant")
  except Exception as exc:  # pragma: no cover
    status["base_import"]["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return status
  status["base_import"]["ok"] = True
  status["base_import"]["version"] = distribution.version

  files = [str(item).replace("\\", "/") for item in (distribution.files or [])]
  package_files = [item for item in files if item.startswith("paroquant/")]

  candidates = [
    "paroquant.transformers",
    "paroquant.transformers.runtime",
    "paroquant.runtime",
    "paroquant.inference",
    "paroquant.inference.backends.transformers",
    "paroquant.inference.backends.transformers.quantizer",
    "paroquant.inference.backends.transformers.modules",
    "paroquant.inference.backends.vllm.plugin",
    "transformers.integrations.paroquant",
    "transformers_integrations.paroquant",
    "paroquant.hooks",
    "paroquant.pytorch",
    "paroquant.models",
  ]
  for name in candidates:
    rel_parts = name.split(".")
    rel_dir = "/".join(rel_parts)
    rel_init = f"{rel_dir}/__init__.py"
    rel_file = f"{rel_dir}.py"
    found = any(item == rel_init or item == rel_file or item.startswith(f"{rel_dir}/") for item in package_files)
    item: dict[str, Any] = {
      "ok": found,
      "method": "distribution_files",
      "origin": rel_init if rel_init in package_files else (rel_file if rel_file in package_files else None),
    }
    try:
      if attempt_runtime_imports and found:
        try:
          importlib.import_module(name)
          item["runtime_import_ok"] = True
        except Exception as exc:  # pragma: no cover
          item["runtime_import_ok"] = False
          item["error"] = {"type": type(exc).__name__, "message": str(exc)}
      status["module_imports"][name] = item
    except Exception as exc:  # pragma: no cover
      status["module_imports"][name] = {
        "ok": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
      }

  try:
    top_level = set()
    for item in package_files:
      parts = item.split("/")
      if len(parts) >= 2 and parts[1] != "__pycache__":
        name = parts[1]
        if name.endswith(".py"):
          name = name[:-3]
        lname = name.lower()
        if "transf" in lname or "hook" in lname or "runtime" in lname or "quant" in lname or "inference" in lname:
          top_level.add(name)
    status["runtime_like_modules"] = sorted(top_level)
  except Exception as exc:  # pragma: no cover
    status["runtime_like_modules"].append(f"scan_error::{type(exc).__name__}:{exc}")

  status["has_runtime_modules"] = bool(status["runtime_like_modules"])
  return status


def _insert_alpamayo_paths(alpamayo_root: Path) -> None:
  if not alpamayo_root.exists():
    return
  sys.path.insert(0, str(alpamayo_root))
  sys.path.insert(0, str(alpamayo_root / "src"))


def _model_class_availability(alpamayo_root: Path, *, attempt_import: bool) -> dict[str, Any]:
  result = {
    "module_imported": False,
    "module_import_attempted": attempt_import,
    "source_scanned": False,
    "source_path": None,
    "source_error": None,
    "module_error": None,
    "Alpamayo1_5": {"available": False},
    "Alpamayo1_5FlashDrive": {"available": False},
  }
  source_path = alpamayo_root / "src" / "alpamayo1_5" / "models" / "alpamayo1_5.py"
  result["source_path"] = str(source_path)
  try:
    source = source_path.read_text(encoding="utf-8")
    result["source_scanned"] = True
    for name in ("Alpamayo1_5", "Alpamayo1_5FlashDrive"):
      result[name]["available"] = bool(re.search(rf"^class\s+{re.escape(name)}\b", source, flags=re.MULTILINE))
      if result[name]["available"]:
        result[name]["class_name"] = name
  except Exception as exc:  # pragma: no cover
    result["source_error"] = {"type": type(exc).__name__, "message": str(exc)}

  if not attempt_import:
    return result

  try:
    module = importlib.import_module("alpamayo1_5.models.alpamayo1_5")
    result["module_imported"] = True
    for name in ("Alpamayo1_5", "Alpamayo1_5FlashDrive"):
      obj = getattr(module, name, None)
      result[name]["available"] = obj is not None
      if obj is None:
        continue
      result[name]["class_name"] = getattr(obj, "__name__", str(obj))
  except Exception as exc:  # pragma: no cover
    result["module_error"] = {"type": type(exc).__name__, "message": str(exc)}
  return result


def _load_exception_dict(exc: BaseException) -> dict[str, str]:
  return {
    "type": type(exc).__name__,
    "message": str(exc),
  }


def _attempt_autoload(model_path: Path, *, label: str) -> dict[str, Any]:
  report = {
    "label": label,
    "ok": False,
    "exception": None,
  }
  load_kwargs = {
    "pretrained_model_name_or_path": str(model_path),
    "trust_remote_code": True,
    "local_files_only": True,
    "low_cpu_mem_usage": True,
    "device_map": "cpu",
    "torch_dtype": "auto",
    "use_safetensors": True,
  }
  try:
    if label == "AutoModel":
      from transformers import AutoModel

      model = AutoModel.from_pretrained(**load_kwargs)
    elif label == "Alpamayo1_5":
      from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

      model = Alpamayo1_5.from_pretrained(**load_kwargs)
    else:  # pragma: no cover
      raise RuntimeError(f"unknown model loader label: {label}")

    # Keep the object alive only long enough to prove constructor path completes.
    report["ok"] = True
    report["model_class"] = type(model).__name__
    del model
  except Exception as exc:  # pragma: no cover
    report["exception"] = _load_exception_dict(exc)
  return report


def _attempt_autoconfig(model_path: Path) -> dict[str, Any]:
  report = {
    "ok": False,
    "exception": None,
  }
  try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
      str(model_path),
      trust_remote_code=True,
      local_files_only=True,
    )
    report["ok"] = True
    report["config_class"] = type(cfg).__name__
    report["model_type"] = getattr(cfg, "model_type", None)
    report["architectures"] = getattr(cfg, "architectures", None)
  except Exception as exc:  # pragma: no cover
    report["exception"] = _load_exception_dict(exc)
  return report


def main() -> int:
  parser = argparse.ArgumentParser(description="Probe PARO model clone and runtime importability.")
  parser.add_argument(
    "--model-path",
    default=r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO",
    help="PARO model directory. Windows drive paths are converted to /mnt/<drive> on WSL.",
  )
  parser.add_argument("--alpamayo-root", default=r"/mnt/g/alpamayo1.5")
  parser.add_argument("--output", type=Path, default=None, help="Optional report JSON output path.")
  parser.add_argument(
    "--attempt-weight-load",
    action="store_true",
    help="Actually call from_pretrained on model weights. Defaults off because this can take minutes and many GiB.",
  )
  parser.add_argument(
    "--attempt-runtime-imports",
    action="store_true",
    help="Actually import ParoQuant runtime modules. Defaults off because some imports can compile CUDA extensions.",
  )
  parser.add_argument(
    "--attempt-alpamayo-imports",
    action="store_true",
    help="Actually import local Alpamayo Python modules. Defaults off because imports may initialize heavy CUDA/Torch paths.",
  )
  parser.add_argument(
    "--attempt-autoconfig",
    action="store_true",
    help="Actually call Transformers AutoConfig. Defaults off because trust_remote_code can import local modules.",
  )
  args = parser.parse_args()

  model_path = _normalize_path(args.model_path)
  alpamayo_root = _normalize_path(args.alpamayo_root)

  report: dict[str, Any] = {
    "status": "started",
    "model_path": str(model_path),
    "alpamayo_root": str(alpamayo_root),
    "created_at_unix": time.time(),
  }

  config, config_error = _read_model_config(model_path)
  report["config"] = {
    "loaded": config is not None,
    "path": str(model_path / "config.json"),
    "error": config_error,
    "architecture": config.get("architectures") if isinstance(config, dict) else None,
    "model_type": config.get("model_type") if isinstance(config, dict) else None,
  }

  w4a8_format, w4a8_source = _detect_w4a8_format(model_path, config)
  report["w4a8_format"] = w4a8_format
  report["w4a8_source"] = w4a8_source

  report["shards"] = _collect_shards(model_path)
  if not report["shards"]["all_present"]:
    report["status"] = "incomplete_clone"
    report["auto_model_probe"] = None
    report["alpamayo1_5_probe"] = None
    report["alpamayo_classes"] = {
      "Alpamayo1_5": {"available": False},
      "Alpamayo1_5FlashDrive": {"available": False},
      "module_imported": False,
      "module_error": {
        "type": "incomplete_clone",
        "message": "skipped until all three expected shards are present",
      },
    }
  else:
    _insert_alpamayo_paths(alpamayo_root)
    report["paroquant"] = _try_import_paroquant(attempt_runtime_imports=args.attempt_runtime_imports)
    report["alpamayo_classes"] = _model_class_availability(
      alpamayo_root,
      attempt_import=args.attempt_alpamayo_imports,
    )
    if args.attempt_autoconfig:
      report["auto_config_probe"] = _attempt_autoconfig(model_path)
    else:
      report["auto_config_probe"] = {
        "ok": False,
        "skipped": True,
        "exception": {
          "type": "skipped",
          "message": "pass --attempt-autoconfig to call Transformers AutoConfig",
        },
      }

    if args.attempt_weight_load and report["alpamayo_classes"].get("module_imported"):
      report["auto_model_probe"] = _attempt_autoload(model_path, label="AutoModel")
      report["alpamayo1_5_probe"] = _attempt_autoload(model_path, label="Alpamayo1_5")
    elif not args.attempt_weight_load:
      report["auto_model_probe"] = {
        "label": "AutoModel",
        "ok": False,
        "skipped": True,
        "exception": {
          "type": "skipped",
          "message": "pass --attempt-weight-load to call from_pretrained",
        },
      }
      report["alpamayo1_5_probe"] = {
        "label": "Alpamayo1_5",
        "ok": False,
        "skipped": True,
        "exception": {
          "type": "skipped",
          "message": "pass --attempt-weight-load to call from_pretrained",
        },
      }
    else:
      report["auto_model_probe"] = {
        "label": "AutoModel",
        "ok": False,
        "exception": {
          "type": "module_import_failed",
          "message": "alpamayo1_5.models.alpamayo1_5 import failed",
        },
      }
      report["alpamayo1_5_probe"] = {
        "label": "Alpamayo1_5",
        "ok": False,
        "exception": {
          "type": "module_import_failed",
          "message": "alpamayo1_5.models.alpamayo1_5 import failed",
        },
      }

    import_failures = [
      name
      for name, item in (report.get("paroquant", {}).get("module_imports") or {}).items()
      if not item.get("ok")
    ]
    class_ok = bool(report["alpamayo_classes"].get("Alpamayo1_5", {}).get("available"))
    flashdrive_ok = bool(report["alpamayo_classes"].get("Alpamayo1_5FlashDrive", {}).get("available"))
    load_ok = bool(report["auto_model_probe"].get("ok")) or bool(report["alpamayo1_5_probe"].get("ok"))
    if args.attempt_weight_load:
      report["status"] = "ok" if load_ok else "error"
    else:
      report["status"] = "ok_probe_only" if class_ok else "error"
    report["runtime_gaps"] = {
      "failed_paroquant_imports": import_failures,
      "missing_alpamayo1_5_flashdrive_class": not flashdrive_ok,
      "weight_load_attempted": bool(args.attempt_weight_load),
      "weight_load_succeeded": load_ok,
    }
  report["alpamayo1_5FlashDrive_available"] = (
    report.get("alpamayo_classes", {}).get("Alpamayo1_5FlashDrive", {}).get("available", False)
  )
  report["alpamayo1_5_available"] = (
    report.get("alpamayo_classes", {}).get("Alpamayo1_5", {}).get("available", False)
  )

  # Keep exact error detail where possible. If we have any exception fields, preserve the same strings.
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
