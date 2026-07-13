#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.rtp import RtpValidationError, parse_rtp
from tools.reasoned_trajectory_poc.qwen_label_rtp_worker import DEFAULT_DURABLE_SCORE_LABELS
from tools.reasoned_trajectory_poc.qwen_trt_label_engine import (
  DEFAULT_ARTIFACT_DIR,
  DEFAULT_MODEL_DIR,
  QwenTrtRotatingLabelScorer,
  _parse_score_threshold_map,
  _score_groups,
  percentile,
)


DEFAULT_VIDEO = Path(__file__).resolve().parents[2].parent / "crashout.mp4"
DEFAULT_GROUPS = "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path"


def _summarize(values: Sequence[float]) -> dict[str, float]:
  if not values:
    return {"median": 0.0, "p90": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0, "min": 0.0}
  return {
    "median": float(statistics.median(values)),
    "p90": percentile(values, 90),
    "p99": percentile(values, 99),
    "p999": percentile(values, 99.9),
    "max": float(max(values)),
    "min": float(min(values)),
  }


def _would_change_lateral(rtp_text: str) -> tuple[bool, float, str, str]:
  try:
    program = parse_rtp(rtp_text)
  except RtpValidationError:
    return False, 0.0, "invalid", ""
  lateral = abs(program.lat_bias_m) > 1e-6 or program.meta in {
    "BIAS_LEFT",
    "BIAS_RIGHT",
    "BIAS_LEFT_AND_SLOW",
    "BIAS_RIGHT_AND_SLOW",
    "REJECT_BASE",
    "TAKE_LEFT_BRANCH",
    "TAKE_RIGHT_BRANCH",
  }
  return lateral, float(program.lat_bias_m), program.meta, program.scene


def _save_sample(image: Image.Image, out_dir: Path, frame_id: int, prefix: str) -> str:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{prefix}_{frame_id:06d}.jpg"
  image.save(path, quality=92)
  return str(path)


def _scorer_args(args: argparse.Namespace) -> argparse.Namespace:
  scorer_args = argparse.Namespace(
    model_dir=args.model_dir,
    artifact_dir=args.artifact_dir,
    image=args.warmup_image,
    image_mode="full",
    image_size=args.image_size,
    text_seq_len=args.text_seq_len,
    score_labels="construction_left,construction_right",
    score_label_groups=args.score_label_groups,
    score_rotate_groups=True,
    score_rotate_shared_engine=True,
    score_cache_ttl_frames=args.score_cache_ttl_frames,
    score_durable_labels=args.score_durable_labels,
    score_negative_clear_threshold=args.score_negative_clear_threshold,
    vehicle_state=args.vehicle_state,
    workspace_gb=args.workspace_gb,
    text_engine=args.text_engine,
    vision_engine=args.vision_engine,
    score_threshold=args.score_threshold,
    score_thresholds=args.score_thresholds,
    use_payload_vehicle_state=False,
    warmup=args.warmup,
    iters=0,
    out=None,
    manifest=args.manifest,
    write_manifest=False,
    require_manifest=args.require_manifest,
    label_keyed_text_engine=False,
  )
  scorer_args.score_thresholds_map = _parse_score_threshold_map(args.score_thresholds)
  return scorer_args


def evaluate(args: argparse.Namespace) -> dict:
  cap = cv2.VideoCapture(str(args.video))
  if not cap.isOpened():
    raise RuntimeError(f"failed to open video: {args.video}")

  fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

  out_dir = args.out
  sample_dir = out_dir / "input_samples"
  lateral_dir = out_dir / "lateral_samples"
  out_dir.mkdir(parents=True, exist_ok=True)

  scorer_args = _scorer_args(args)
  groups = _score_groups(args.score_label_groups)
  scorer = QwenTrtRotatingLabelScorer(scorer_args, groups)
  scorer.warmup(args.warmup)

  rows: list[dict] = []
  frame_id = 0
  scored_count = 0
  first_lateral: dict | None = None
  start_wall = time.perf_counter()
  while True:
    ok, bgr = cap.read()
    if not ok:
      break
    if frame_id < args.start_frame:
      frame_id += 1
      continue
    if args.end_frame >= 0 and frame_id > args.end_frame:
      break
    if (frame_id - args.start_frame) % args.stride != 0:
      frame_id += 1
      continue
    if args.max_frames > 0 and scored_count >= args.max_frames:
      break

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if scored_count < args.save_first:
      _save_sample(image, sample_dir, frame_id, "input")

    response = scorer.score(image, args.vehicle_state, frame_id)
    timings = response["timings_ms"]
    lateral, lat_bias_m, meta, scene = _would_change_lateral(response["rtp_text"])
    if lateral and first_lateral is None:
      first_lateral = {
        "frame_id": frame_id,
        "time_sec": frame_id / fps if fps > 0 else 0.0,
        "lat_bias_m": lat_bias_m,
        "meta": meta,
        "scene": scene,
        "labels": response["labels"],
        "labels_current_group": response.get("labels_current_group", []),
        "label_scores_cached": response.get("label_scores_cached", {}),
        "rtp_text": response["rtp_text"],
      }
    if lateral and len(list(lateral_dir.glob("lateral_*.jpg"))) < args.save_lateral:
      _save_sample(image, lateral_dir, frame_id, "lateral")

    rows.append({
      "frame_id": frame_id,
      "time_sec": frame_id / fps if fps > 0 else 0.0,
      "deadline_met": float(timings["total_ms"]) <= args.deadline_ms,
      "would_change_lateral": lateral,
      "lat_bias_m": lat_bias_m,
      "meta": meta,
      "scene": scene,
      "labels": response["labels"],
      "labels_scored_this_request": response.get("labels_scored_this_request", []),
      "labels_current_group": response.get("labels_current_group", []),
      "score_group_index": response.get("score_group_index"),
      "label_scores": response.get("label_scores", {}),
      "label_scores_cached": response.get("label_scores_cached", {}),
      "rtp_text": response["rtp_text"],
      "timings_ms": timings,
    })
    scored_count += 1
    frame_id += 1

  cap.release()

  elapsed_sec = time.perf_counter() - start_wall
  timing_keys = sorted(rows[0]["timings_ms"]) if rows else []
  timing_summary = {
    key: _summarize([float(row["timings_ms"][key]) for row in rows])
    for key in timing_keys
  }
  lateral_rows = [row for row in rows if row["would_change_lateral"]]
  deadline_misses = [row for row in rows if not row["deadline_met"]]
  result = {
    "kind": "qwen_trt_video_eval",
    "video": str(args.video),
    "video_frames": total_frames,
    "video_fps": fps,
    "video_width": width,
    "video_height": height,
    "sampled_frames": len(rows),
    "start_frame": args.start_frame,
    "end_frame": args.end_frame,
    "stride": args.stride,
    "deadline_ms": args.deadline_ms,
    "deadline_miss_count": len(deadline_misses),
    "deadline_miss_frames": [row["frame_id"] for row in deadline_misses[:50]],
    "would_change_lateral_count": len(lateral_rows),
    "first_lateral": first_lateral,
    "first_lateral_frame": None if first_lateral is None else first_lateral["frame_id"],
    "first_lateral_time_sec": None if first_lateral is None else first_lateral["time_sec"],
    "max_abs_lat_bias_m": max((abs(float(row["lat_bias_m"])) for row in rows), default=0.0),
    "timing_summary": timing_summary,
    "elapsed_sec": elapsed_sec,
    "scored_fps_wall": len(rows) / elapsed_sec if elapsed_sec > 0 else 0.0,
    "groups": groups,
    "score_threshold": args.score_threshold,
    "score_thresholds": scorer_args.score_thresholds_map,
    "records": rows,
  }
  (out_dir / "video_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate the TensorRT Qwen label scorer on a real video.")
  parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
  parser.add_argument("--out", type=Path, default=Path("artifacts") / "reasoned_trajectory_poc" / "crashout_qwen_trt_video_eval")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--artifact-dir", type=Path, default=Path("F:/qwen_trt_export") if Path("F:/qwen_trt_export").exists() else DEFAULT_ARTIFACT_DIR)
  parser.add_argument("--text-engine", type=Path, default=Path("F:/qwen_trt_export/nvfp4_trt/qwen_text_36layer_nvfp4_trt.engine") if Path("F:/qwen_trt_export/nvfp4_trt/qwen_text_36layer_nvfp4_trt.engine").exists() else None)
  parser.add_argument("--vision-engine", type=Path, default=None)
  parser.add_argument("--warmup-image", type=Path, default=Path("artifacts") / "reasoned_trajectory_poc" / "qwen_construction_loop_proof" / "vlm" / "vlm_input_0020.png")
  parser.add_argument("--image-size", type=int, default=168)
  parser.add_argument("--text-seq-len", type=int, default=220)
  parser.add_argument("--score-label-groups", default=DEFAULT_GROUPS)
  parser.add_argument("--score-cache-ttl-frames", type=int, default=60)
  parser.add_argument("--score-durable-labels", default=",".join(DEFAULT_DURABLE_SCORE_LABELS))
  parser.add_argument("--score-negative-clear-threshold", type=float, default=2.0)
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--score-thresholds", default="pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5")
  parser.add_argument("--vehicle-state", default="speed=5.0 mps")
  parser.add_argument("--workspace-gb", type=int, default=6)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--deadline-ms", type=float, default=50.0)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--end-frame", type=int, default=-1)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--max-frames", type=int, default=0)
  parser.add_argument("--save-first", type=int, default=6)
  parser.add_argument("--save-lateral", type=int, default=6)
  parser.add_argument("--manifest", type=Path, default=None)
  parser.add_argument("--require-manifest", action="store_true")
  args = parser.parse_args()
  if args.stride < 1:
    raise ValueError("--stride must be >= 1")
  result = evaluate(args)
  printable = {k: v for k, v in result.items() if k != "records"}
  print(json.dumps(printable, indent=2))
  print(f"artifacts={args.out}")


if __name__ == "__main__":
  main()
