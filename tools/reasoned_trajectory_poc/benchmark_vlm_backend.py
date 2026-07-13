#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.pathsynth import BasePlan
from selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig
from selfdrive.controls.reasoned.rtp import parse_rtp
from selfdrive.controls.reasoned.scene_board import SceneBoardRenderer


def make_base_plan(frame_id: int, speed_mps: float) -> BasePlan:
  return BasePlan(
    frame_id=frame_id,
    model_log_mono_time_ns=frame_id * 50_000_000,
    t=tuple(i * 0.2 for i in range(17)),
    x=tuple(i * 5.0 for i in range(17)),
    y=tuple(0.0 for _ in range(17)),
    speeds=tuple(speed_mps for _ in range(17)),
    desired_curvature=0.0,
    v_ego=speed_mps,
  )


def percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return ordered[idx]


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark the configured RTP VLM backend against the 50 ms model-frame budget.")
  parser.add_argument("--frames", type=int, default=20)
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  parser.add_argument("--board-width", type=int, default=512)
  parser.add_argument("--board-height", type=int, default=384)
  parser.add_argument("--warmup-frames", type=int, default=1)
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "vlm_benchmark.json")
  args = parser.parse_args()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  planner = ReasonedPlanner(
    config=ReasonedPlannerConfig(deadline_ms=args.deadline_ms),
    renderer=SceneBoardRenderer(width=args.board_width, height=args.board_height),
  )
  for warmup_id in range(args.warmup_frames):
    planner.step(make_base_plan(-1 - warmup_id, 15.0), {"v_ego": 15.0})
  records = []
  for frame_id in range(args.frames):
    result = planner.step(make_base_plan(frame_id, 15.0), {"v_ego": 15.0})
    parsed = False
    try:
      parse_rtp(result.rtp_text)
      parsed = True
    except Exception:
      parsed = False
    records.append({
      "frame_id": frame_id,
      "should_publish": result.should_publish,
      "deadline_met": result.deadline_met,
      "valid": result.valid,
      "parsed": parsed,
      "total_ms": result.timings.publish_age_ms or result.timings.total_ms,
      "stage_total_ms": result.timings.total_ms,
      "vlm_prefill_ms": result.timings.scene_board_to_vlm_prefill_ms,
      "vlm_decode_ms": result.timings.vlm_decode_ms,
      "generated_token_count": result.generated_token_count,
      "vlm_backend": result.vlm_backend,
      "rtp_source_frame_id": result.rtp_source_frame_id,
      "rtp_age_frames": result.rtp_age_frames,
      "rtp_text": result.rtp_text,
      "invalid_reason": result.invalid_reason,
      "selected_candidate": None if result.synth is None else result.synth.selected_candidate,
    })

  totals = [r["total_ms"] for r in records]
  summary = {
    "frames": args.frames,
    "deadline_ms": args.deadline_ms,
    "board_width": args.board_width,
    "board_height": args.board_height,
    "publish_count": sum(1 for r in records if r["should_publish"]),
    "valid_count": sum(1 for r in records if r["valid"]),
    "parsed_count": sum(1 for r in records if r["parsed"]),
    "mean_ms": statistics.fmean(totals) if totals else 0.0,
    "p50_ms": percentile(totals, 50),
    "p90_ms": percentile(totals, 90),
    "p99_ms": percentile(totals, 99),
    "max_ms": max(totals) if totals else 0.0,
    "records": records,
  }
  args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
  print(f"benchmark={args.out}")


if __name__ == "__main__":
  main()
