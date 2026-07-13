from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PCTS = ("p50_ms", "p95_ms", "p99_ms")


def _percentile(values: list[float], pct: float) -> float:
  if not values:
    raise ValueError("empty latency list")
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  rank = (len(ordered) - 1) * pct
  low = math.floor(rank)
  high = math.ceil(rank)
  frac = rank - low
  return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _latency_summary(values: list[float]) -> dict[str, float | int]:
  return {
    "count": len(values),
    "mean_ms": round(sum(values) / len(values), 3),
    "min_ms": round(min(values), 3),
    "p50_ms": round(_percentile(values, 0.50), 3),
    "p95_ms": round(_percentile(values, 0.95), 3),
    "p99_ms": round(_percentile(values, 0.99), 3),
    "max_ms": round(max(values), 3),
  }


def _stage(summary: dict[str, Any], name: str) -> dict[str, Any]:
  try:
    return summary["stages"][name]
  except KeyError as e:
    raise KeyError(f"summary missing stage {name}") from e


def _hz(ms: float) -> float:
  return 1000.0 / ms if ms > 0 else float("inf")


def _gap_multiplier(ms: float, target_ms: float) -> float:
  return ms / target_ms if target_ms > 0 else float("inf")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--full-summary", required=True, type=Path)
  parser.add_argument("--trt-text-report", required=True, type=Path)
  parser.add_argument("--deep-report", type=Path)
  parser.add_argument("--out", required=True, type=Path)
  args = parser.parse_args()

  full_summary = json.loads(args.full_summary.read_text(encoding="utf-8"))
  trt_text = json.loads(args.trt_text_report.read_text(encoding="utf-8"))
  trt_latencies = [float(value) for value in trt_text["latency_ms"]]
  trt_summary = _latency_summary(trt_latencies)

  full_sample = _stage(full_summary, "full_sample_sync")
  vlm_generate = _stage(full_summary, "vlm_generate_sync")
  diffusion = _stage(full_summary, "diffusion_sample_sync")
  expert = _stage(full_summary, "expert_forward_sync")
  action_to_traj = _stage(full_summary, "action_to_traj_sync")

  by_percentile: dict[str, dict[str, float]] = {}
  for pct in PCTS:
    full_ms = float(full_sample[pct])
    vlm_ms = float(vlm_generate[pct])
    text_ms = float(trt_summary[pct])
    non_vlm_ms = max(0.0, full_ms - vlm_ms)
    hybrid_text_only_ms = non_vlm_ms + text_ms
    by_percentile[pct] = {
      "full_pytorch_ms": round(full_ms, 3),
      "full_pytorch_hz": round(_hz(full_ms), 3),
      "pytorch_vlm_generate_ms": round(vlm_ms, 3),
      "non_vlm_remainder_ms": round(non_vlm_ms, 3),
      "trt_text17_ms": round(text_ms, 3),
      "optimistic_replace_vlm_with_text_trt_ms": round(hybrid_text_only_ms, 3),
      "optimistic_replace_vlm_with_text_trt_hz": round(_hz(hybrid_text_only_ms), 3),
      "perfect_vlm_zero_ms_remainder_hz": round(_hz(non_vlm_ms), 3),
      "path_a_gap_full_x": round(_gap_multiplier(full_ms, 50.0), 3),
      "path_b_gap_full_vs_200ms_x": round(_gap_multiplier(full_ms, 200.0), 3),
      "path_b_gap_hybrid_vs_200ms_x": round(_gap_multiplier(hybrid_text_only_ms, 200.0), 3),
      "max_plan_age_250_gap_hybrid_x": round(_gap_multiplier(hybrid_text_only_ms, 250.0), 3),
    }

  deep_payload: dict[str, Any] | None = None
  if args.deep_report is not None:
    deep = json.loads(args.deep_report.read_text(encoding="utf-8"))
    hot = deep["inferences"][-1]["synced_stage_summary"]
    deep_payload = {
      "source": str(args.deep_report),
      "hot_infer_index": deep["inferences"][-1]["infer_index"],
      "visual_forward_sum_ms": hot.get("vlm_visual_forward_sync", {}).get("sum_ms"),
      "language_forward_sum_ms": hot.get("vlm_language_forward_sync", {}).get("sum_ms"),
      "vlm_forward_sum_ms": hot.get("vlm_forward_sync", {}).get("sum_ms"),
      "vlm_forward_count": hot.get("vlm_forward_sync", {}).get("count"),
    }

  report = {
    "status": "ok",
    "sources": {
      "full_summary": str(args.full_summary),
      "trt_text_report": str(args.trt_text_report),
      "deep_report": str(args.deep_report) if args.deep_report is not None else None,
    },
    "caveats": [
      "TensorRT text17 timing is language-decoder-only and does not include real multimodal image tokens.",
      "Hybrid estimates subtract PyTorch VLM generation and add TensorRT text timing, so they are not a runnable integrated path.",
      "This analysis is a prioritization and infeasibility-evidence artifact, not proof that Path A or Path B is impossible.",
    ],
    "full_stage_input_tokens": full_summary.get("input_tokens"),
    "full_stage_generated_sequence_lengths": full_summary.get("generated_sequence_lengths"),
    "trt_text17_input_tokens": trt_text.get("input_tokens"),
    "trt_text17_output_tokens_requested": 17,
    "trt_text17_latency": trt_summary,
    "full_stage_key_stages": {
      "full_sample_sync": full_sample,
      "vlm_generate_sync": vlm_generate,
      "diffusion_sample_sync": diffusion,
      "expert_forward_sync_total": expert,
      "action_to_traj_sync": action_to_traj,
    },
    "deep_vlm_diagnostic": deep_payload,
    "hybrid_gap_by_percentile": by_percentile,
    "required_next_optimizations": [
      "Integrate real multimodal image-token path into TensorRT or equivalent optimized runtime.",
      "Accelerate or export the Alpamayo expert/diffusion path; current p99 diffusion is about 310 ms by itself.",
      "Measure C3X/PC streaming and control-loop plan-age only after the model runtime is near bounded-rate feasibility.",
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
