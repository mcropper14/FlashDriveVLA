#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.pathsynth import PathSynth
from selfdrive.controls.reasoned.rtp import RtpValidationError, parse_rtp
from selfdrive.controls.reasoned.ui_scene_board import UiSceneBoardRenderer
from selfdrive.controls.reasoned.vlm import PersistentRtpEngine
from tools.reasoned_trajectory_poc.run_metadrive_overlay_demo import (
  MetaDriveRouteFollower,
  camera_frame,
  make_base_plan_from_route,
  make_env,
  selected_lateral_offset_m,
  spawn_novel_scene,
  speed_mps,
)


DEFAULT_QWEN_COMMAND = "py -3.11 tools/reasoned_trajectory_poc/qwen_vlm_worker.py --model-dir models/vlm/qwen2_5_vl_3b_instruct --max-new-tokens 96"


def run_probe(args: argparse.Namespace) -> dict:
  os.environ["RTP_VLM_IMAGE_SIZE"] = str(args.vlm_image_size)
  if args.stderr_log:
    os.environ["RTP_VLM_STDERR_PATH"] = str(args.stderr_log)

  engine = PersistentRtpEngine(args.qwen_command)
  renderer = UiSceneBoardRenderer(args.board_width, args.board_height)
  follower = MetaDriveRouteFollower()
  synth_engine = PathSynth()
  out_dir = args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  results = []

  try:
    for scene in args.scenes:
      env_args = argparse.Namespace(**vars(args))
      env_args.novel_scene = scene
      env = make_env(env_args)
      try:
        spawned = spawn_novel_scene(env, scene)
        for _ in range(args.settle_frames):
          env.step([0.0, 0.0])

        current_speed = speed_mps(env)
        base_plan = make_base_plan_from_route(env, args.frame_id, current_speed)
        road = camera_frame(env)
        board = renderer.render(base_plan, {"v_ego": current_speed, "road_frame": road, "status": "PROBE"})
        image_path = out_dir / f"{scene}_vlm_input.png"
        board.save(image_path)

        start = time.perf_counter()
        rtp_result = engine.generate(args.frame_id, board, {"v_ego": current_speed}, args.deadline_ms)
        wall_ms = (time.perf_counter() - start) * 1000.0
        valid = False
        invalid_reason = ""
        program = None
        synth = None
        stock_steer, stock_gas, stock_debug = follower.action(env, args.speed_mps, 0.0)
        reasoned_steer = stock_steer
        reasoned_gas = stock_gas
        reasoned_debug = stock_debug
        lateral_offset_m = 0.0
        target_speed_mps = args.speed_mps

        try:
          program = parse_rtp(rtp_result.text)
          synth = synth_engine.compile(base_plan, program)
          valid = synth.valid
          if valid:
            lateral_offset_m = float(np.clip(selected_lateral_offset_m(synth), -0.65, 0.65))
            if getattr(program, "speed_scale", None) is not None:
              target_speed_mps = min(target_speed_mps, args.speed_mps * program.speed_scale)
            if program.speed_cap_mps is not None:
              target_speed_mps = min(target_speed_mps, program.speed_cap_mps)
            reasoned_steer, reasoned_gas, reasoned_debug = follower.action(env, target_speed_mps, lateral_offset_m)
        except RtpValidationError as exc:
          invalid_reason = str(exc)

        results.append({
          "scene": scene,
          "image_path": str(image_path),
          "spawned_scene": spawned,
          "rtp_text": rtp_result.text,
          "vlm_backend": rtp_result.backend,
          "vlm_wall_ms": wall_ms,
          "vlm_prefill_ms": rtp_result.prefill_ms,
          "vlm_decode_ms": rtp_result.decode_ms,
          "generated_token_count": rtp_result.generated_token_count,
          "valid": valid,
          "invalid_reason": invalid_reason,
          "program": None if program is None else {
            "scene": program.scene,
            "meta": program.meta,
            "branch": program.branch,
            "lat_bias_m": program.lat_bias_m,
            "speed_cap_mps": program.speed_cap_mps,
            "speed_scale": getattr(program, "speed_scale", None),
            "stop_s": program.stop_s,
            "evidence": list(program.evidence),
            "avoid": list(program.avoid),
            "confidence": program.confidence,
          },
          "synth": None if synth is None else {
            "valid": synth.valid,
            "selected_candidate": synth.selected_candidate,
            "desired_curvature": synth.desired_curvature,
            "path_delta_m": synth.vlm_changed_path_meters,
            "speed_delta_mps": synth.vlm_changed_speed_mps,
            "speed_cap_mps": synth.speed_cap_mps,
            "stop_s": synth.stop_s,
            "invalid_reason": synth.invalid_reason,
          },
          "control_delta": {
            "stock_steer": stock_steer,
            "reasoned_steer": reasoned_steer,
            "steer_delta": reasoned_steer - stock_steer,
            "stock_gas": stock_gas,
            "reasoned_gas": reasoned_gas,
            "gas_delta": reasoned_gas - stock_gas,
            "active_lateral_offset_m": lateral_offset_m,
            "stock_target_speed_mps": args.speed_mps,
            "reasoned_target_speed_mps": target_speed_mps,
            "target_speed_delta_mps": target_speed_mps - args.speed_mps,
            "stock_debug": stock_debug,
            "reasoned_debug": reasoned_debug,
          },
        })
      finally:
        env.close()
  finally:
    engine.close()

  summary = {
    "qwen_command": args.qwen_command,
    "scenes": args.scenes,
    "results": results,
  }
  out_path = out_dir / "qwen_novel_scene_probe.json"
  out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Probe whether Qwen emits control-relevant RTP for actual MetaDrive novel scenes.")
  parser.add_argument("--scenes", nargs="+", choices=("construction", "pedestrian", "stop_sign"), default=["construction", "pedestrian", "stop_sign"])
  parser.add_argument("--qwen-command", default=os.getenv("RTP_VLM_SERVER_COMMAND", DEFAULT_QWEN_COMMAND))
  parser.add_argument("--stderr-log", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_novel_probe_stderr.log")
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_novel_scene_probe")
  parser.add_argument("--frame-id", type=int, default=0)
  parser.add_argument("--deadline-ms", type=float, default=30_000.0)
  parser.add_argument("--settle-frames", type=int, default=2)
  parser.add_argument("--speed-mps", type=float, default=8.0)
  parser.add_argument("--map", default="3")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--camera-width", type=int, default=512)
  parser.add_argument("--camera-height", type=int, default=320)
  parser.add_argument("--board-width", type=int, default=512)
  parser.add_argument("--board-height", type=int, default=384)
  parser.add_argument("--vlm-image-size", type=int, default=384)
  args = parser.parse_args()

  summary = run_probe(args)
  print(json.dumps(summary, indent=2))
  print(f"artifacts={args.out}")


if __name__ == "__main__":
  main()
