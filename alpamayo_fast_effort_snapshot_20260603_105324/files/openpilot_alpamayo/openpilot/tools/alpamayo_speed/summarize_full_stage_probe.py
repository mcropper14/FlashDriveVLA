from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_STAGES = (
  "full_sample_sync",
  "vlm_generate_sync",
  "vlm_forward_sync",
  "vlm_visual_forward_sync",
  "vlm_language_forward_sync",
  "diffusion_sample_sync",
  "expert_forward_sync",
  "action_in_proj_sync",
  "action_out_proj_sync",
  "action_to_traj_sync",
  "extract_text_tokens_sync",
)


def _percentile(values: list[float], pct: float) -> float:
  if not values:
    raise ValueError("cannot compute percentile of an empty list")
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * pct
  low = math.floor(rank)
  high = math.ceil(rank)
  if low == high:
    return ordered[low]
  frac = rank - low
  return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _summarize(values: list[float], suffix: str = "_ms") -> dict[str, float | int]:
  if not values:
    return {"count": 0}
  def key(name: str) -> str:
    return f"{name}{suffix}"
  return {
    "count": len(values),
    key("mean"): round(sum(values) / len(values), 3),
    key("min"): round(min(values), 3),
    key("p50"): round(_percentile(values, 0.50), 3),
    key("p95"): round(_percentile(values, 0.95), 3),
    key("p99"): round(_percentile(values, 0.99), 3),
    key("max"): round(max(values), 3),
  }


def _stage_value(stage_summary: dict[str, Any], stage: str, metric: str) -> float | None:
  if stage not in stage_summary:
    return None
  value = stage_summary[stage].get(metric)
  if value is None:
    return None
  return float(value)


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
  reports = []
  for path in paths:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_source_path"] = str(path)
    reports.append(report)
  return reports


def build_summary(paths: list[Path], skip_first: bool, metric: str) -> dict[str, Any]:
  reports = _load_reports(paths)
  by_stage: dict[str, list[float]] = {stage: [] for stage in DEFAULT_STAGES}
  generated_sequence_lengths: list[int] = []
  input_tokens: list[int] = []
  cot_chars: list[int] = []
  inference_records: list[dict[str, Any]] = []

  for report in reports:
    source = report["_source_path"]
    shape = report.get("input_ids_shape")
    if isinstance(shape, list) and shape:
      input_tokens.append(int(shape[-1]))
    inferences = report.get("inferences", [])
    selected = inferences[1:] if skip_first and len(inferences) > 1 else inferences
    for inference in selected:
      stage_summary = inference.get("synced_stage_summary", {})
      record: dict[str, Any] = {
        "source": source,
        "infer_index": inference.get("infer_index"),
      }
      runtime_profile = inference.get("runtime_profile_unsynced", {})
      if "generated_sequence_length" in runtime_profile:
        generated_sequence_lengths.append(int(runtime_profile["generated_sequence_length"]))
        record["generated_sequence_length"] = int(runtime_profile["generated_sequence_length"])
      if "cot_chars" in inference:
        cot_chars.append(int(inference["cot_chars"]))
      for stage in DEFAULT_STAGES:
        value = _stage_value(stage_summary, stage, metric)
        if value is not None:
          by_stage[stage].append(value)
          record[stage] = value
      inference_records.append(record)

  stage_summary_out = {
    stage: _summarize(values)
    for stage, values in by_stage.items()
    if values
  }
  return {
    "status": "ok",
    "source_paths": [str(path) for path in paths],
    "skip_first": skip_first,
    "stage_metric": metric,
    "report_count": len(reports),
    "inference_count": len(inference_records),
    "input_tokens": _summarize([float(value) for value in input_tokens], suffix="") if input_tokens else {"count": 0},
    "generated_sequence_lengths": _summarize([float(value) for value in generated_sequence_lengths], suffix="") if generated_sequence_lengths else {"count": 0},
    "cot_chars": _summarize([float(value) for value in cot_chars], suffix="") if cot_chars else {"count": 0},
    "stages": stage_summary_out,
    "inferences": inference_records,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("reports", nargs="+", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--include-first", action="store_true", help="Include first/warmup inference in stage statistics.")
  parser.add_argument("--metric", choices=["mean_ms", "sum_ms", "max_ms", "min_ms"], default="mean_ms")
  args = parser.parse_args()

  summary = build_summary(args.reports, skip_first=not args.include_first, metric=args.metric)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
