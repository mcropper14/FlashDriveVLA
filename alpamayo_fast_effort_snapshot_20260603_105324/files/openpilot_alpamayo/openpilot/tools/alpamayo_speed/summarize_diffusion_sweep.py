from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STAGES = (
  "full_sample_sync",
  "vlm_generate_sync",
  "diffusion_sample_sync",
  "expert_forward_sync",
  "action_in_proj_sync",
  "action_out_proj_sync",
  "action_to_traj_sync",
  "extract_text_tokens_sync",
)


def _percentile(values: list[float], pct: float) -> float:
  if not values:
    raise ValueError("empty values")
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * pct
  low = math.floor(rank)
  high = math.ceil(rank)
  frac = rank - low
  return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _summary(values: list[float], suffix: str = "_ms") -> dict[str, float | int]:
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


def _stage_value(inference: dict[str, Any], stage: str, metric: str) -> float | None:
  stage_summary = inference.get("synced_stage_summary", {}).get(stage)
  if not stage_summary:
    return None
  value = stage_summary.get(metric)
  return float(value) if value is not None else None


def build_summary(report: dict[str, Any], skip_first_per_steps: bool, metric: str) -> dict[str, Any]:
  groups: dict[int, list[dict[str, Any]]] = {}
  for inference in report.get("inferences", []):
    steps = int(inference.get("diffusion_steps", report.get("args", {}).get("diffusion_steps", 0)))
    groups.setdefault(steps, []).append(inference)

  out_groups: dict[str, Any] = {}
  for steps, rows in sorted(groups.items(), reverse=True):
    selected = rows[1:] if skip_first_per_steps and len(rows) > 1 else rows
    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    min_ade = []
    generated_lengths = []
    expert_calls = []
    row_records = []
    for row in selected:
      record = {
        "infer_index": row.get("infer_index"),
        "repeat_index": row.get("repeat_index"),
      }
      if "min_ade_m" in row:
        min_ade.append(float(row["min_ade_m"]))
        record["min_ade_m"] = float(row["min_ade_m"])
      runtime = row.get("runtime_profile_unsynced", {})
      if "generated_sequence_length" in runtime:
        generated_lengths.append(float(runtime["generated_sequence_length"]))
      if "expert_step_calls" in runtime:
        expert_calls.append(float(runtime["expert_step_calls"]))
      for stage in STAGES:
        value = _stage_value(row, stage, metric)
        if value is not None:
          stage_values[stage].append(value)
          record[stage] = value
      row_records.append(record)

    out_groups[str(steps)] = {
      "raw_count": len(rows),
      "used_count": len(selected),
      "skipped_first": skip_first_per_steps and len(rows) > 1,
      "stages": {stage: _summary(values) for stage, values in stage_values.items() if values},
      "min_ade_m": _summary(min_ade, suffix="_m") if min_ade else {"count": 0},
      "generated_sequence_length": _summary(generated_lengths, suffix="") if generated_lengths else {"count": 0},
      "expert_step_calls": _summary(expert_calls, suffix="") if expert_calls else {"count": 0},
      "rows": row_records,
    }

  return {
    "status": "ok",
    "source_status": report.get("status"),
    "diffusion_steps_list": report.get("diffusion_steps_list"),
    "skip_first_per_steps": skip_first_per_steps,
    "stage_metric": metric,
    "groups": out_groups,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("report", type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--include-first-per-steps", action="store_true")
  parser.add_argument("--metric", choices=["sum_ms", "mean_ms", "max_ms", "min_ms"], default="sum_ms")
  args = parser.parse_args()

  report = json.loads(args.report.read_text(encoding="utf-8"))
  summary = build_summary(
    report,
    skip_first_per_steps=not args.include_first_per_steps,
    metric=args.metric,
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
