#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any


_DEFAULT_PARO_PATH_WIN = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
_DEFAULT_PARO_PATH_WSL = "/mnt/j/temp_alpamayo/Alpamayo-1.5-10B-finetuned-PARO"
_VIEW_SUFFIX = "-vllm-vlm-view"

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")

_KEEP_PREFIXES = (
  "vlm.model.visual.",
  "vlm.model.language_model.",
  "vlm.lm_head.",
)
_STRIP_PREFIXES = (
  ("vlm.model.visual.", "model.visual."),
  ("vlm.model.language_model.", "model.language_model."),
  ("vlm.lm_head.", "lm_head."),
)
_EXCLUDE_MARKERS = ("expert", "action", "traj", "trajectory", "tokenizer")
_VLLM_FLAT_SUFFIXES = (
  (".rotate_linear.rotation.theta", ".theta"),
  (".rotate_linear.rotation.pairs", ".pairs"),
  (".rotate_linear.rotation.channel_scales", ".channel_scales"),
  (".rotate_linear.qlinear.qweight", ".qweight"),
  (".rotate_linear.qlinear.qzeros", ".qzeros"),
  (".rotate_linear.qlinear.scales", ".scales"),
)
_VLLM_FLAT_EXCLUDE_SUFFIXES = (
  ".rotate_linear.qlinear.g_idx",
  ".rotate_linear.qlinear.g_idx_sort_indices",
  ".rotate_linear.qlinear.input_global_scale",
  ".rotate_linear.qlinear.workspace",
)

_QWEN3_CACHE_HINT_KEYWORDS = ("qwen3", "qwen3_vl", "qwen2-vl", "qwen2_vl")
_QWEN3_MODEL_TYPES = {
  "qwen3_vl",
  "qwen2_vl",
  "qwen2_vl_for_conditional_generation",
}


def _in_wsl() -> bool:
  if os.name != "posix":
    return False
  try:
    with Path("/proc/version").open("r", encoding="utf-8") as handle:
      return "microsoft" in handle.read().lower()
  except OSError:
    return False


def _windows_to_wsl(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if not match:
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def _normalize_path(raw: str) -> Path:
  expanded = os.path.expanduser(raw)
  return Path(_windows_to_wsl(expanded) if _in_wsl() else expanded)


def _default_paro_paths() -> tuple[Path, Path]:
  default_input = _DEFAULT_PARO_PATH_WSL if _in_wsl() else _DEFAULT_PARO_PATH_WIN
  input_path = _normalize_path(default_input)
  return input_path, input_path.with_name(f"{input_path.name}{_VIEW_SUFFIX}")


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _default_report_path() -> Path:
  return _repo_root() / "artifacts" / "alpamayo_speed" / (
    f"prepare_paro_vllm_view_{time.strftime('%Y%m%d_%H%M%S')}.json"
  )


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
  try:
    return json.loads(path.read_text(encoding="utf-8")), None
  except FileNotFoundError:
    return None, f"missing::{path}"
  except json.JSONDecodeError as exc:
    return None, f"json_decode_error::{exc}"
  except OSError as exc:
    return None, f"read_error::{type(exc).__name__}::{exc}"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
  merged = dict(base)
  for key, value in updates.items():
    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
      merged[key] = _deep_merge(merged[key], value)
    else:
      merged[key] = value
  return merged


def _extract_vocab_size(config: dict[str, Any] | None) -> int | None:
  if not isinstance(config, dict):
    return None
  vocab = config.get("vocab_size")
  if isinstance(vocab, int):
    return vocab
  text_config = config.get("text_config")
  if isinstance(text_config, dict) and isinstance(text_config.get("vocab_size"), int):
    return int(text_config["vocab_size"])
  return None


def _scan_cached_qwen3_config() -> tuple[dict[str, Any] | None, Path | None, list[str]]:
  cache_roots: list[Path] = []
  warnings: list[str] = []
  for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    value = os.getenv(key)
    if value:
      normalized = _normalize_path(value)
      cache_roots.append(normalized)
      if normalized.name != "hub":
        cache_roots.append(normalized / "hub")
  cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

  for cache_root in cache_roots:
    if not cache_root.exists():
      continue
    for model_dir in cache_root.glob("models--*"):
      if not model_dir.is_dir():
        continue
      if not any(token in model_dir.name.lower() for token in _QWEN3_CACHE_HINT_KEYWORDS):
        continue
      snapshots = model_dir / "snapshots"
      if not snapshots.exists():
        continue
      for snapshot in snapshots.glob("*"):
        cfg_path = snapshot / "config.json"
        if not cfg_path.exists():
          continue
        config, error = _read_json(cfg_path)
        if config is None:
          warnings.append(f"qwen3_cache_parse_error::{cfg_path}::{error}")
          continue
        model_type = str(config.get("model_type", "")).lower()
        architectures = config.get("architectures", [])
        has_qwen3 = (
          model_type in _QWEN3_MODEL_TYPES
          or any(isinstance(v, str) and v.lower().startswith("qwen3vl") for v in architectures)
        )
        if has_qwen3:
          return config, cfg_path, warnings
  return None, None, warnings


def _runtime_hints() -> dict[str, bool]:
  return {
    "paroquant_spec_available": importlib.util.find_spec("paroquant") is not None,
    "vllm_spec_available": importlib.util.find_spec("vllm") is not None,
    "transformers_spec_available": importlib.util.find_spec("transformers") is not None,
    "hf_cache_var_present": bool(
      os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("TRANSFORMERS_CACHE")
    ),
  }


def _strip_prefix(name: str) -> str:
  for old, new in _STRIP_PREFIXES:
    if name.startswith(old):
      return f"{new}{name[len(old):]}"
  return name


def _rewrite_for_key_layout(name: str, key_layout: str) -> str | None:
  if key_layout == "native":
    return name
  if key_layout != "vllm-flat":
    raise ValueError(f"unsupported key layout: {key_layout}")
  if name.endswith(_VLLM_FLAT_EXCLUDE_SUFFIXES):
    return None
  for old, new in _VLLM_FLAT_SUFFIXES:
    if name.endswith(old):
      return f"{name[:-len(old)]}{new}"
  return name


def _select_weight_map(
  weight_map: dict[str, str],
  key_layout: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, list[str]]]:
  included: dict[str, str] = {}
  excluded = {"non_vlm": 0, "excluded_markers": 0, "layout_excluded": 0, "layout_collision": 0}
  excluded_samples = {"non_vlm": [], "excluded_markers": [], "layout_excluded": [], "layout_collision": []}

  for key, shard in weight_map.items():
    key_str = str(key)
    if not any(key_str.startswith(prefix) for prefix in _KEEP_PREFIXES):
      excluded["non_vlm"] += 1
      if len(excluded_samples["non_vlm"]) < 16:
        excluded_samples["non_vlm"].append(key_str)
      continue
    lowered = key_str.lower()
    if any(marker in lowered for marker in _EXCLUDE_MARKERS):
      excluded["excluded_markers"] += 1
      if len(excluded_samples["excluded_markers"]) < 16:
        excluded_samples["excluded_markers"].append(key_str)
      continue
    stripped = _strip_prefix(key_str)
    rewritten = _rewrite_for_key_layout(stripped, key_layout)
    if rewritten is None:
      excluded["layout_excluded"] += 1
      if len(excluded_samples["layout_excluded"]) < 16:
        excluded_samples["layout_excluded"].append(key_str)
      continue
    if rewritten in included:
      excluded["layout_collision"] += 1
      if len(excluded_samples["layout_collision"]) < 16:
        excluded_samples["layout_collision"].append(key_str)
      continue
    included[rewritten] = str(shard)

  return included, excluded, excluded_samples


def _build_output_config(
  source_config: dict[str, Any] | None,
  cached_qwen3: dict[str, Any] | None,
  w4a8_config: dict[str, Any] | None,
  source_path: Path,
  runtime_hints: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, Any], list[str], bool, dict[str, Any]]:
  source_config = source_config or {}
  cached_qwen3 = cached_qwen3 or {}
  merged = _deep_merge(dict(cached_qwen3), source_config)

  source_vocab = _extract_vocab_size(source_config)
  cached_vocab = _extract_vocab_size(cached_qwen3)
  selected_vocab = source_vocab if source_vocab is not None else cached_vocab

  decisions: dict[str, Any] = {
    "source_vocab": source_vocab,
    "cached_vocab": cached_vocab,
    "selected_vocab": selected_vocab,
    "cached_qwen3_used": bool(cached_qwen3),
    "source_model_type": source_config.get("model_type"),
  }

  warnings: list[str] = []
  if selected_vocab is None:
    warnings.append("vocab_size_unresolved")
  else:
    merged["vocab_size"] = selected_vocab
    text_config = merged.get("text_config")
    if not isinstance(text_config, dict):
      text_config = {}
    text_config["vocab_size"] = selected_vocab
    merged["text_config"] = text_config

  merged["architectures"] = ["Qwen3VLForConditionalGeneration"]
  merged["model_type"] = "qwen3_vl"

  quantization_meta: dict[str, Any] = {
    "runtime_hints": runtime_hints,
    "w4a8_detected": w4a8_config is not None,
  }
  uncertain = False
  existing_quant = merged.get("quantization_config")
  if not isinstance(existing_quant, dict):
    existing_quant = {}

  if w4a8_config is not None:
    quantization_meta.update(
      {
        "source": "w4a8_config.json",
        "format": w4a8_config.get("format"),
        "bits": w4a8_config.get("bits", 4),
        "group_size": w4a8_config.get("group_size", 128),
        "krot": w4a8_config.get("krot", 8),
        "zero_point": w4a8_config.get("zero_point", True),
      }
    )
    existing_quant.update(
      {
        "quant_method": "paroquant",
        "bits": quantization_meta["bits"],
        "group_size": quantization_meta["group_size"],
        "krot": quantization_meta["krot"],
        "zero_point": quantization_meta["zero_point"],
        "approach": "w4a8",
        "w4a8": quantization_meta,
      }
    )
    warnings.append(
      "quantization_config uses ParoQuant defaults bits=4/group_size=128/krot=8/zero_point=true because w4a8_config.json does not spell them out"
    )
  else:
    uncertain = True
    warnings.append("w4a8_config_missing; quantization metadata kept conservative")
    existing_quant.update(
      {
        "approach": "unknown",
        "source": "conservative",
        "note": "w4a8_config.json not found",
      }
    )

  merged["quantization_config"] = existing_quant
  merged.setdefault("alpamayo_view_metadata", {})
  merged["alpamayo_view_metadata"].update(
    {
      "source_model_path": str(source_path),
      "created_unix": time.time(),
    }
  )

  return merged, quantization_meta, warnings, uncertain, decisions


def _rewrite_index(
  index: dict[str, Any],
  included: dict[str, str],
  source_path: Path,
) -> dict[str, Any]:
  rewritten = dict(index)
  rewritten["weight_map"] = dict(included)
  metadata = dict(rewritten.get("metadata", {}))
  metadata["alpamayo_vllm_view"] = {
    "source": str(source_path),
    "created_unix": time.time(),
  }
  rewritten["metadata"] = metadata
  return rewritten


def _attach_shard(
  source: Path,
  destination: Path,
  mode: str,
  force: bool,
  allow_copy: bool,
) -> tuple[bool, str]:
  if destination.exists():
    if not force:
      return False, f"skip_exists::{destination}"
    destination.unlink()

  if mode == "none":
    return True, f"skipped::{source}->{destination}"
  if mode == "copy":
    if not allow_copy:
      return False, "copy_requires --copy-shards"
    shutil.copy2(source, destination)
    return True, f"copied::{source}->{destination}"
  if mode == "symlink":
    try:
      destination.symlink_to(source)
      return True, f"symlink::{source}->{destination}"
    except OSError as exc:
      return False, f"symlink_error::{type(exc).__name__}:{exc}"
  if mode == "hardlink":
    try:
      os.link(source, destination)
      return True, f"hardlink::{source}->{destination}"
    except OSError as exc:
      return False, f"hardlink_error::{type(exc).__name__}:{exc}"
  return False, f"unsupported_mode::{mode}"


def _link_shards(
  input_path: Path,
  output_path: Path,
  included: dict[str, str],
  mode: str,
  force: bool,
  dry_run: bool,
  allow_copy: bool,
) -> tuple[dict[str, Any], list[str], bool]:
  result: dict[str, Any] = {
    "mode": mode,
    "required_shards": sorted(set(included.values())),
    "created": [],
    "failed": [],
  }
  warnings: list[str] = []
  had_error = False

  for shard_name in result["required_shards"]:
    source_shard = input_path / str(shard_name)
    destination_shard = output_path / str(shard_name)
    if not source_shard.exists():
      had_error = True
      result["failed"].append(f"missing_shard::{source_shard}")
      continue
    if dry_run:
      result["created"].append(f"would_{mode}::{source_shard}->{destination_shard}")
      continue
    ok, detail = _attach_shard(source_shard, destination_shard, mode, force, allow_copy)
    if ok:
      result["created"].append(detail)
    else:
      had_error = True
      result["failed"].append(detail)
      warnings.append(detail)
  return result, warnings, had_error


def _build_report(
  args: argparse.Namespace,
  input_path: Path,
  output_path: Path,
  report_path: Path,
  source_config: dict[str, Any] | None,
  source_err: dict[str, str | None],
  index: dict[str, Any] | None,
  index_error: str | None,
  w4a8_config: dict[str, Any] | None,
  cached_qwen3: dict[str, Any] | None,
  cached_qwen3_path: Path | None,
  scan_warnings: list[str],
  runtime_hints: dict[str, bool],
  excluded: dict[str, int],
  excluded_samples: dict[str, list[str]],
  included_count: int,
  original_count: int,
  output_config: dict[str, Any],
  config_warnings: list[str],
  quantization_summary: dict[str, Any],
  quantization_uncertain: bool,
  link_results: dict[str, Any] | None,
  link_warnings: list[str],
  status_error: bool,
) -> dict[str, Any]:
  report: dict[str, Any] = {
    "created_unix": time.time(),
    "status": "error" if status_error else "ok",
    "input_model_path": str(input_path),
    "output_model_path": str(output_path),
    "report_output_path": str(report_path),
    "dry_run": args.dry_run,
    "input_path_converted": str(input_path) != args.paro_path,
    "link_mode": args.link_mode,
    "copy_shards_enabled": args.copy_shards,
      "force": args.force,
      "key_layout": args.key_layout,
    "file_status": {
      "config_json": {
        "path": str(input_path / "config.json"),
        "found": source_config is not None,
        "error": source_err.get("config_json"),
      },
      "w4a8_config_json": {
        "path": str(input_path / "w4a8_config.json"),
        "found": w4a8_config is not None,
        "error": source_err.get("w4a8_config_json"),
      },
      "index_json": {
        "path": str(input_path / "model.safetensors.index.json"),
        "found": index is not None,
        "error": index_error,
      },
      "cached_qwen3_config_json": {
        "path": str(cached_qwen3_path) if cached_qwen3_path else None,
        "found": cached_qwen3 is not None,
      },
    },
    "counts": {
      "original_keys": original_count,
      "included_keys": included_count,
      "excluded_keys": original_count - included_count,
      "excluded_by_reason": excluded,
      "excluded_samples": excluded_samples,
    },
    "key_filter_rules": {
      "keep_prefixes": list(_KEEP_PREFIXES),
      "exclude_markers": list(_EXCLUDE_MARKERS),
      "strip_rules": [{"from": f, "to": t} for f, t in _STRIP_PREFIXES],
      "key_layout": args.key_layout,
      "vllm_flat_suffix_rules": [{"from": f, "to": t} for f, t in _VLLM_FLAT_SUFFIXES],
      "vllm_flat_excluded_suffixes": list(_VLLM_FLAT_EXCLUDE_SUFFIXES),
    },
    "config_decisions": {
      "architectures": output_config.get("architectures"),
      "model_type": output_config.get("model_type"),
      "selected_vocab": output_config.get("vocab_size"),
      "text_vocab": (
        output_config.get("text_config", {}).get("vocab_size")
        if isinstance(output_config.get("text_config"), dict)
        else None
      ),
      "runtime_hints": runtime_hints,
    },
    "quantization": {
      "uncertain": quantization_uncertain,
      "summary": quantization_summary,
    },
    "shard_link": link_results or {},
    "warnings": [],
  }

  if source_config is not None:
    report["source_input_snapshot"] = {
      "architectures": source_config.get("architectures"),
      "model_type": source_config.get("model_type"),
    }

  report["warnings"].extend(scan_warnings)
  report["warnings"].extend(config_warnings)
  report["warnings"].extend(link_warnings)

  if args.link_mode == "copy" and not args.copy_shards:
    report["status"] = "error"
    report["warnings"].append("link_mode_copy_without_copy_shards")
  if not status_error and not report["warnings"]:
    if report["status"] == "error":
      report["status"] = "ok"
  if report["warnings"] and report["status"] == "ok":
    report["status"] = "degraded"

  return report


def main() -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Create a lightweight PARO VLM-only view for Qwen3-VL/vLLM experiments without loading weights."
      " Reads config.json, w4a8_config.json, model.safetensors.index.json, and any local Qwen3-VL cached config."
      " Writes a rewritten config.json and model.safetensors.index.json."
    )
  )
  parser.add_argument(
    "--paro-path",
    default=str(_default_paro_paths()[0]),
    help="PARO source path (Windows drive path or WSL /mnt/<drive>/ path).",
  )
  parser.add_argument("--output", type=Path, default=None, help="Output directory for VLM-only view.")
  parser.add_argument("--report-output", type=Path, default=None, help="Optional JSON report path.")
  parser.add_argument("--dry-run", action="store_true", help="Validate and report only; do not write output files.")
  parser.add_argument("--force", action="store_true", help="Overwrite destination artifacts when present.")
  parser.add_argument(
    "--link-mode",
    choices=("hardlink", "symlink", "copy", "none"),
    default="hardlink",
    help="How to attach shard files.",
  )
  parser.add_argument(
    "--key-layout",
    choices=("vllm-flat", "native"),
    default="vllm-flat",
    help=(
      "How to rewrite nested PARO linear keys. vllm-flat maps rotate_linear.rotation/qlinear "
      "suffixes to the flat parameter names expected by the visible ParoQuant vLLM plugin; "
      "native only strips the outer Alpamayo vlm prefix."
    ),
  )
  parser.add_argument(
    "--copy-shards",
    action="store_true",
    help="Allow physical copy of shard files when --link-mode=copy.",
  )
  args = parser.parse_args()

  paro_default, output_default = _default_paro_paths()
  input_path = _normalize_path(args.paro_path)
  output_path = output_default if args.output is None else Path(args.output)
  report_output = args.report_output if args.report_output is not None else _default_report_path()

  status_error = False
  source_config, source_config_error = _read_json(input_path / "config.json")
  w4a8_config, w4a8_error = _read_json(input_path / "w4a8_config.json")
  source_errors = {
    "config_json": source_config_error,
    "w4a8_config_json": w4a8_error,
  }
  index, index_error = _read_json(input_path / "model.safetensors.index.json")

  if not input_path.exists():
    status_error = True
  if index is None:
    status_error = True
  if source_config is None and source_config_error is not None and not args.dry_run:
    status_error = True

  if not isinstance(index, dict):
    status_error = True
    weight_map = {}
  else:
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
      status_error = True
      weight_map = {}
    else:
      weight_map = {str(k): str(v) for k, v in raw_weight_map.items()}

  included_map, excluded_counts, excluded_samples = _select_weight_map(weight_map, args.key_layout)

  cached_qwen3, cached_qwen3_path, scan_warnings = _scan_cached_qwen3_config()
  runtime_hints = _runtime_hints()

  output_config, quant_report, config_warnings, quant_uncertain, config_decisions = _build_output_config(
    source_config,
    cached_qwen3,
    w4a8_config,
    input_path,
    runtime_hints,
  )

  rewritten_index = _rewrite_index(index if isinstance(index, dict) else {}, included_map, input_path)

  link_results: dict[str, Any] | None = None
  link_warnings: list[str] = []
  link_failed = False
  if not args.dry_run:
    if output_path.exists() and not output_path.is_dir():
      status_error = True
    elif output_path.exists() and list(output_path.iterdir()) and not args.force:
      status_error = True
    else:
      output_path.mkdir(parents=True, exist_ok=True)
      if args.link_mode in {"hardlink", "symlink", "copy", "none"}:
        link_results, link_warnings, link_failed = _link_shards(
          input_path,
          output_path,
          included_map,
          args.link_mode,
          args.force,
          args.dry_run,
          args.copy_shards,
        )

  if not args.dry_run and not status_error and not link_failed:
    (output_path / "config.json").write_text(
      json.dumps(output_config, indent=2, sort_keys=True),
      encoding="utf-8",
    )
    (output_path / "model.safetensors.index.json").write_text(
      json.dumps(rewritten_index, indent=2, sort_keys=True),
      encoding="utf-8",
    )

  if args.link_mode == "copy" and not args.copy_shards:
    link_failed = True

  report = _build_report(
    args=args,
    input_path=input_path,
    output_path=output_path,
    report_path=Path(report_output),
    source_config=source_config,
    source_err=source_errors,
    index=index if isinstance(index, dict) else None,
    index_error=index_error,
    w4a8_config=w4a8_config,
    cached_qwen3=cached_qwen3,
    cached_qwen3_path=cached_qwen3_path,
    scan_warnings=scan_warnings,
    runtime_hints=runtime_hints,
    excluded=excluded_counts,
    excluded_samples=excluded_samples,
    included_count=len(included_map),
    original_count=len(weight_map),
    output_config=output_config,
    config_warnings=config_warnings,
    quantization_summary=quant_report,
    quantization_uncertain=quant_uncertain,
    link_results=link_results,
    link_warnings=link_warnings,
    status_error=status_error or link_failed,
  )
  report["config_decisions"].update(config_decisions)
  report_output = Path(report_output)
  report_output.parent.mkdir(parents=True, exist_ok=True)
  report_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True))

  return 1 if report.get("status") == "error" else 0


if __name__ == "__main__":
  raise SystemExit(main())
