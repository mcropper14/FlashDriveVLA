#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig
from selfdrive.controls.reasoned.vlm import StaticRtpEngine
from tools.reasoned_trajectory_poc.run_local_demo import SCENARIOS, make_base_plan


def mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else 0.0


def run(args: argparse.Namespace) -> dict:
  out_dir = args.out
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.engine == "static":
    engine = StaticRtpEngine(SCENARIOS[args.scenario])
  else:
    engine = None

  planner = ReasonedPlanner(
    config=ReasonedPlannerConfig(deadline_ms=args.deadline_ms),
    engine=engine,
  )
  for warmup_id in range(args.warmup_frames):
    planner.step(make_base_plan(-1 - warmup_id, args.speed_mps), {"v_ego": args.speed_mps})

  records = []
  for frame_id in range(args.frames):
    base_plan = make_base_plan(frame_id, args.speed_mps)
    result = planner.step(base_plan, {"v_ego": args.speed_mps})
    if frame_id < args.save_boards and result.board is not None:
      result.board.save(out_dir / f"{args.engine}_{args.scenario}_board_{frame_id:04d}.png")

    synth = result.synth
    records.append({
      "frame_id": frame_id,
      "stock_desired_curvature": base_plan.desired_curvature,
      "stock_speed_mps": base_plan.speeds[0],
      "stock_terminal_y_m": base_plan.y[-1],
      "reasoned_should_publish": result.should_publish,
      "reasoned_valid": result.valid,
      "reasoned_deadline_met": result.deadline_met,
      "reasoned_latency_ms": result.timings.publish_age_ms,
      "reasoned_stage_total_ms": result.timings.total_ms,
      "reasoned_selected_candidate": None if synth is None else synth.selected_candidate,
      "reasoned_desired_curvature": None if synth is None else synth.desired_curvature,
      "reasoned_path_delta_m": 0.0 if synth is None else synth.vlm_changed_path_meters,
      "reasoned_speed_delta_mps": 0.0 if synth is None else synth.vlm_changed_speed_mps,
      "invalid_reason": result.invalid_reason,
      "rtp_text": result.rtp_text,
    })

  published = [r for r in records if r["reasoned_should_publish"]]
  summary = {
    "engine": args.engine,
    "scenario": args.scenario,
    "frames": args.frames,
    "deadline_ms": args.deadline_ms,
    "publish_count": len(published),
    "invalid_count": sum(1 for r in records if not r["reasoned_valid"]),
    "deadline_miss_count": sum(1 for r in records if not r["reasoned_deadline_met"]),
    "mean_reasoned_latency_ms": mean([r["reasoned_latency_ms"] for r in records]),
    "mean_published_path_delta_m": mean([r["reasoned_path_delta_m"] for r in published]),
    "mean_published_speed_delta_mps": mean([r["reasoned_speed_delta_mps"] for r in published]),
    "stock_mean_desired_curvature": mean([r["stock_desired_curvature"] for r in records]),
    "reasoned_mean_desired_curvature": mean([r["reasoned_desired_curvature"] for r in published if r["reasoned_desired_curvature"] is not None]),
    "records": records,
  }
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare stock base plan against the reasoned RTP/PathSynth plan in a local deterministic sim harness.")
  parser.add_argument("--engine", choices=("static", "vlm"), default="static")
  parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="construction")
  parser.add_argument("--frames", type=int, default=20)
  parser.add_argument("--speed-mps", type=float, default=15.0)
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  parser.add_argument("--warmup-frames", type=int, default=1)
  parser.add_argument("--save-boards", type=int, default=3)
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "comparison")
  args = parser.parse_args()

  summary = run(args)
  out_path = args.out / f"stock_vs_reasoned_{args.engine}_{args.scenario}.json"
  out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
  print(f"comparison={out_path}")


if __name__ == "__main__":
  main()
