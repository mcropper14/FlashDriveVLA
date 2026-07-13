from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_VLM_LAYER_RE = re.compile(r"vlm\.model\.language_model\.layers\.(\d+)\.")
_DEFAULT_PARO_PATH = r"J:\temp_alpamayo\Alpamayo-1.5-10B-finetuned-PARO"
_VLM_PREFIX = "vlm.model.language_model."
_QUANT_MARKERS = ("qweight", "qzeros", "scales", "workspace")


def _in_wsl() -> bool:
  if os.name != "posix":
    return False

  try:
    with open("/proc/version", "r", encoding="utf-8") as handle:
      return "microsoft" in handle.read().lower()
  except OSError:
    return False


def _convert_windows_path(value: str) -> str:
  match = _WIN_DRIVE_RE.match(value)
  if match is None:
    return value
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def _normalize_path(value: str) -> Path:
  normalized = _convert_windows_path(value) if _in_wsl() else value
  return Path(os.path.expanduser(normalized))


def _read_json(path: Path) -> tuple[Any | None, str | None]:
  try:
    return json.loads(path.read_text(encoding="utf-8")), None
  except FileNotFoundError:
    return None, f"missing::{path}"
  except json.JSONDecodeError as exc:
    return None, f"json_decode_error::{exc}"
  except OSError as exc:
    return None, f"read_error::{exc}"


def _extract_quantized_layer_count(w4a8_config: Any) -> int | None:
  if isinstance(w4a8_config, dict):
    top_level = w4a8_config.get("num_quantized_layers")
    if isinstance(top_level, int):
      return top_level

    for key, value in w4a8_config.items():
      lower_key = str(key).lower()
      if lower_key in {"quantized_layers", "quantized_layer_ids", "quant_layers"} and isinstance(value, list):
        return len(value)
      if "quantized" in lower_key and "layer" in lower_key and isinstance(value, int):
        return value

      nested = _extract_quantized_layer_count(value)
      if nested is not None:
        return nested

  elif isinstance(w4a8_config, list):
    for item in w4a8_config:
      nested = _extract_quantized_layer_count(item)
      if nested is not None:
        return nested

  return None


def _extract_layer_index(key: str) -> int | None:
  match = _VLM_LAYER_RE.search(key)
  if not match:
    return None
  return int(match.group(1))


def _sample(items: list[str], limit: int) -> list[str]:
  return items[: max(limit, 0)]


def _analyze_index(weight_map: dict[str, Any], sample_limit: int) -> dict[str, Any]:
  all_keys = sorted(weight_map.keys())
  vlm_keys = [key for key in all_keys if key.startswith(_VLM_PREFIX)]
  quant_like_keys = [key for key in vlm_keys if any(marker in key for marker in _QUANT_MARKERS)]
  quant_qweight = [key for key in quant_like_keys if key.endswith(".qweight")]
  quant_qzeros = [key for key in quant_like_keys if key.endswith(".qzeros")]
  quant_scales = [key for key in quant_like_keys if key.endswith(".scales")]
  quant_workspace = [key for key in quant_like_keys if key.endswith(".workspace")]

  qweight_layers = Counter[int]()
  for key in quant_qweight:
    layer_index = _extract_layer_index(key)
    if layer_index is not None:
      qweight_layers[layer_index] += 1

  nested_rotate_qlinear = [key for key in quant_like_keys if ".rotate_linear.qlinear." in key]
  flat_qlinear = [key for key in quant_like_keys if ".qlinear." in key and ".rotate_linear.qlinear." not in key]

  layer_counts = [
    {"layer": layer, "qweight_modules": count}
    for layer, count in sorted(qweight_layers.items())
  ]

  compatible = {
    "status": "unknown",
    "likely_mismatch": False,
    "reason": "No visible qweight/qzeros/scales/workspace VLM quant keys matched.",
    "nested_rotate_linear_qlinear_count": len(nested_rotate_qlinear),
    "flat_qlinear_count": len(flat_qlinear),
    "sample_nested_rotate_linear": _sample(nested_rotate_qlinear, sample_limit),
    "sample_flat_qlinear": _sample(flat_qlinear, sample_limit),
  }
  if quant_qweight or quant_qzeros or quant_scales or quant_workspace:
    if nested_rotate_qlinear and not flat_qlinear:
      compatible["status"] = "likely_incompatible"
      compatible["likely_mismatch"] = True
      compatible["reason"] = (
        "Found rotate_linear.qlinear quantized keys but no flatter RotateQuantizedLinear keys. "
        "This layout is likely incompatible with the visible Transformers + paroquant expectation."
      )
    elif flat_qlinear:
      compatible["status"] = "likely_compatible"
      compatible["likely_mismatch"] = False
      compatible["reason"] = "Found flattened qlinear quantized keys without nested rotate_linear.qlinear usage."
    else:
      compatible["status"] = "unknown"
      compatible["reason"] = (
        "Quantized markers are present but no qlinear suffix pattern was detected."
      )

  return {
    "total_keys": len(all_keys),
    "vlm_language_keys": len(vlm_keys),
    "quant_like_vlm_language_keys": len(quant_like_keys),
    "qweight_modules_in_index": len(quant_qweight),
    "qweight_layer_counts": layer_counts,
    "qweight_layers_covered": len(layer_counts),
    "qweight_keys_per_layer": {
      "min": min(qweight_layers.values()) if qweight_layers else 0,
      "max": max(qweight_layers.values()) if qweight_layers else 0,
      "total": sum(qweight_layers.values()),
    },
    "sample": {
      "qweight": _sample(quant_qweight, sample_limit),
      "qzeros": _sample(quant_qzeros, sample_limit),
      "scales": _sample(quant_scales, sample_limit),
      "workspace": _sample(quant_workspace, sample_limit),
      "quant_like": _sample(quant_like_keys, sample_limit),
    },
    "paroquant_layout_compat": compatible,
    "weight_map_shards_for_quant_keys": {
      "qweight": sorted({weight_map[key] for key in quant_qweight}) if quant_qweight else [],
      "qzeros": sorted({weight_map[key] for key in quant_qzeros}) if quant_qzeros else [],
      "scales": sorted({weight_map[key] for key in quant_scales}) if quant_scales else [],
      "workspace": sorted({weight_map[key] for key in quant_workspace}) if quant_workspace else [],
    },
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Inspect PARO clone metadata/index key layout for VLM quantized modules."
  )
  parser.add_argument(
    "--model-path",
    default=_DEFAULT_PARO_PATH,
    help="PARO model path. Windows drive paths are converted to /mnt/<drive> under WSL.",
  )
  parser.add_argument(
    "--sample-limit",
    type=int,
    default=12,
    help="Max number of sampled keys per marker.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path to write JSON report.",
  )
  args = parser.parse_args()

  model_path = _normalize_path(args.model_path)
  sample_limit = max(args.sample_limit, 1)

  config_path = model_path / "config.json"
  w4a8_config_path = model_path / "w4a8_config.json"
  index_path = model_path / "model.safetensors.index.json"

  config, config_error = _read_json(config_path)
  w4a8_config, w4a8_error = _read_json(w4a8_config_path)
  index_content, index_error = _read_json(index_path)

  weight_map_error = None
  weight_map: dict[str, str] | None = None
  index_stats: dict[str, Any] | None = None
  if index_content is None:
    weight_map_error = index_error
  elif not isinstance(index_content, dict):
    weight_map_error = "invalid_index_format::expected object"
  else:
    maybe_map = index_content.get("weight_map")
    if not isinstance(maybe_map, dict):
      weight_map_error = "invalid_index_format::missing_or_non_dict_weight_map"
    else:
      weight_map = {str(k): str(v) for k, v in maybe_map.items()}
      index_stats = _analyze_index(weight_map, sample_limit)

  w4a8_quantized_layers = _extract_quantized_layer_count(w4a8_config)

  report: dict[str, Any] = {
    "model_path": str(model_path),
    "input_path_converted": str(model_path) != args.model_path,
    "status": "ok",
    "files": {
      "config_json": {
        "path": str(config_path),
        "loaded": config is not None,
        "error": config_error,
      },
      "w4a8_config_json": {
        "path": str(w4a8_config_path),
        "loaded": w4a8_config is not None,
        "error": w4a8_error,
      },
      "index_json": {
        "path": str(index_path),
        "loaded": index_content is not None,
        "error": index_error,
      },
    },
    "config_summary": {
      "architectures": config.get("architectures") if isinstance(config, dict) else None,
      "model_type": config.get("model_type") if isinstance(config, dict) else None,
      "transformers_version": config.get("transformers_version") if isinstance(config, dict) else None,
    },
    "w4a8_summary": {
      "num_quantized_layers": w4a8_quantized_layers,
      "format": w4a8_config.get("format") if isinstance(w4a8_config, dict) else None,
    },
    "index_summary": index_stats,
    "cross_check": {
      "qweight_modules_vs_w4a8_quantized_layers": None,
      "qweight_modules": index_stats["qweight_modules_in_index"] if index_stats else None,
    },
  }

  if weight_map_error is not None:
    report["status"] = "error"
    report["files"]["index_json"]["error"] = weight_map_error
  elif index_stats is not None and w4a8_quantized_layers is not None:
    report["cross_check"]["qweight_modules_vs_w4a8_quantized_layers"] = (
      "match" if report["cross_check"]["qweight_modules"] == w4a8_quantized_layers else "mismatch"
    )
    if report["cross_check"]["qweight_modules"] != w4a8_quantized_layers:
      report["cross_check"]["qweight_modules_difference"] = (
        report["cross_check"]["qweight_modules"] - w4a8_quantized_layers
      )

  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

  print(json.dumps(report, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
