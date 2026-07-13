#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.pathsynth import BasePlan
from selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig
from selfdrive.controls.reasoned.vlm import StaticRtpEngine, detect_local_gpu


CONSTRUCTION_RTP = """RTPv1
scene=construction_merge
evidence=[cones_right_s22_45,lane_left_open]
meta=BIAS_LEFT_AND_SLOW
branch=base
lat_bias_m=1.25
speed_cap_mps=25%
stop_s=none
avoid=[right_edge_s8_48_margin1.25]
weights=[obs2.2,lane1.4,comfort1.0,base0.7,vlm1.0]
confidence=0.72"""

YIELD_RTP = """RTPv1
scene=crosswalk_yield
evidence=[pedestrian_right_s18,crosswalk_s20]
meta=YIELD
branch=base
lat_bias_m=0.0
speed_cap_mps=25%
stop_s=18.0
avoid=[pedestrian_s18_22_margin1.00]
weights=[obs3.0,lane1.0,comfort1.0,base0.8,vlm1.0]
confidence=0.80"""

SCENARIOS = {
  "construction": CONSTRUCTION_RTP,
  "yield": YIELD_RTP,
}


def make_base_plan(frame_id: int, speed_mps: float) -> BasePlan:
  xs = tuple(float(i * 5) for i in range(17))
  ys = tuple(0.08 * (i / 16.0) for i in range(17))
  ts = tuple(float(i) * 0.2 for i in range(17))
  return BasePlan(
    frame_id=frame_id,
    model_log_mono_time_ns=frame_id * 50_000_000,
    t=ts,
    x=xs,
    y=ys,
    speeds=tuple(speed_mps for _ in xs),
    desired_curvature=0.0,
    v_ego=speed_mps,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Run the local PC reasoned trajectory POC without vehicle hardware.")
  parser.add_argument("--frames", type=int, default=3)
  parser.add_argument("--speed-mps", type=float, default=15.0)
  parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="construction")
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc")
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  args = parser.parse_args()

  args.out.mkdir(parents=True, exist_ok=True)
  planner = ReasonedPlanner(
    config=ReasonedPlannerConfig(deadline_ms=args.deadline_ms),
    engine=StaticRtpEngine(SCENARIOS[args.scenario]),
  )
  trace_path = args.out / "trace.jsonl"

  with trace_path.open("w", encoding="utf-8") as trace:
    for frame_id in range(args.frames):
      base_plan = make_base_plan(frame_id, args.speed_mps)
      result = planner.step(base_plan, {"v_ego": args.speed_mps})
      if result.board is not None:
        result.board.save(args.out / f"scene_board_{frame_id:04d}.png")
      synth = result.synth
      trace.write(json.dumps({
        "frame_id": frame_id,
        "valid": result.valid,
        "should_publish": result.should_publish,
        "deadline_met": result.deadline_met,
        "selected_candidate": None if synth is None else synth.selected_candidate,
        "desired_curvature": None if synth is None else synth.desired_curvature,
        "vlm_changed_path_meters": None if synth is None else synth.vlm_changed_path_meters,
        "vlm_changed_speed_mps": None if synth is None else synth.vlm_changed_speed_mps,
        "latency_ms": result.timings.total_ms,
        "invalid_reason": result.invalid_reason,
      }) + "\n")

  print(f"local_gpu={detect_local_gpu()}")
  print(f"trace={trace_path}")
  print(f"scene_boards={args.out}")


if __name__ == "__main__":
  main()
