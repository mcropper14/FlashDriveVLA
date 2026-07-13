#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.side_semantics import construction_hazard_side_from_labels

CONSTRUCTION_KIND_TOKENS = ("construction", "traffic_cone", "traffic_barrier")


def _as_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if result == result and result not in (float("inf"), float("-inf")) else None


def construction_requirement_from_record(record: dict[str, Any], *, horizon_m: float, intrusion_m: float) -> dict[str, Any]:
  ahead_m = _as_float(record.get("construction_nearest_ahead_m"))
  lateral_delta_m = _as_float(record.get("construction_nearest_lateral_delta_m"))
  if ahead_m is None or lateral_delta_m is None:
    ahead_m, lateral_delta_m = _nearest_construction_from_record(record)
  if ahead_m is None or lateral_delta_m is None:
    return {
      "required": False,
      "required_side": "none",
      "required_shift_label": "none",
      "ahead_m": ahead_m,
      "lateral_delta_m": lateral_delta_m,
    }

  required = 0.0 <= ahead_m <= horizon_m and abs(lateral_delta_m) <= intrusion_m
  if not required:
    return {
      "required": False,
      "required_side": "none",
      "required_shift_label": "none",
      "ahead_m": ahead_m,
      "lateral_delta_m": lateral_delta_m,
    }

  side = "right" if lateral_delta_m > 0.0 else "left"
  return {
    "required": True,
    "required_side": side,
    "required_shift_label": "construction_shift_left" if side == "right" else "construction_shift_right",
    "ahead_m": ahead_m,
    "lateral_delta_m": lateral_delta_m,
  }


def _nearest_construction_from_record(record: dict[str, Any]) -> tuple[float | None, float | None]:
  spawned = record.get("spawned_scene")
  if not isinstance(spawned, list):
    return None, None
  current_long_m = _as_float(record.get("route_longitudinal_m"))
  control_debug = record.get("control_debug") if isinstance(record.get("control_debug"), dict) else {}
  current_lateral_m = _as_float(control_debug.get("lane_lateral_m"))
  if current_long_m is None or current_lateral_m is None:
    return None, None

  nearest: tuple[float, float, float] | None = None
  for item in spawned:
    if not isinstance(item, dict):
      continue
    kind = str(item.get("kind", "")).lower()
    if not any(token in kind for token in CONSTRUCTION_KIND_TOKENS):
      continue
    route_s_m = _as_float(item.get("route_s_m"))
    lateral_m = _as_float(item.get("lateral_m"))
    if route_s_m is None or lateral_m is None:
      continue
    ahead_m = route_s_m - current_long_m
    lateral_delta_m = lateral_m - current_lateral_m
    distance_m = (ahead_m * ahead_m + lateral_delta_m * lateral_delta_m) ** 0.5
    if nearest is None or distance_m < nearest[0]:
      nearest = (distance_m, ahead_m, lateral_delta_m)
  if nearest is None:
    return None, None
  return nearest[1], nearest[2]


def qwen_construction_side(labels: list[str] | tuple[str, ...]) -> str:
  side = construction_hazard_side_from_labels(labels)
  return "none" if side == "unknown" else side


def consumed_construction_side(record: dict[str, Any]) -> str:
  sides: set[str] = set()
  details = record.get("durable_lateral_plan_details")
  if isinstance(details, list):
    for item in details:
      if not isinstance(item, dict):
        continue
      side = str(item.get("hazard_side", "unknown")).lower()
      if side in ("left", "right"):
        sides.add(side)
  if len(sides) == 1:
    return next(iter(sides))
  if len(sides) > 1:
    return "conflict"
  return rtp_construction_side(str(record.get("rtp_text") or ""))


def rtp_construction_side(rtp_text: str) -> str:
  for raw_line in rtp_text.splitlines():
    line = raw_line.strip()
    if line == "scene=construction_right":
      return "right"
    if line == "scene=construction_left":
      return "left"
    if line.startswith("avoid="):
      if "right_edge_" in line and "left_edge_" not in line:
        return "right"
      if "left_edge_" in line and "right_edge_" not in line:
        return "left"
  return "none"


def lateral_command_side(
  record: dict[str, Any],
  *,
  min_offset_m: float,
  keys: tuple[str, ...] = ("active_lateral_offset_m", "green_path_offset_metadrive_m", "durable_avoidance_offset_m", "desired_lateral_offset_m"),
) -> str:
  for key in keys:
    value = _as_float(record.get(key))
    if value is None or abs(value) < min_offset_m:
      continue
    # MetaDrive positive lateral is scene right; negative is scene left.
    return "right" if value > 0.0 else "left"
  return "none"


def planned_lateral_command_side(record: dict[str, Any], *, min_offset_m: float) -> str:
  return lateral_command_side(
    record,
    min_offset_m=min_offset_m,
    keys=("desired_lateral_offset_m", "durable_avoidance_offset_m", "active_lateral_offset_m"),
  )


def _crashed(record: dict[str, Any]) -> bool:
  flags = record.get("info_flags") if isinstance(record.get("info_flags"), dict) else {}
  return bool(flags.get("crash_object") or flags.get("crash_vehicle") or flags.get("crash_human"))


def evaluate_episode(
  episode: dict[str, Any],
  *,
  horizon_m: float = 14.0,
  intrusion_m: float = 1.65,
  min_offset_m: float = 0.10,
) -> dict[str, Any]:
  records = list(episode.get("records", []))
  rows: list[dict[str, Any]] = []
  required_count = 0
  qwen_side_correct = 0
  qwen_side_wrong = 0
  qwen_side_missing = 0
  not_required_count = 0
  false_construction_count = 0
  control_away_count = 0
  control_toward_count = 0
  control_neutral_count = 0
  planned_away_count = 0
  planned_toward_count = 0
  planned_neutral_count = 0
  consumed_side_correct = 0
  consumed_side_wrong = 0
  consumed_side_missing = 0
  collision_count = 0
  green_path_mismatch_count = 0

  for record in records:
    requirement = construction_requirement_from_record(record, horizon_m=horizon_m, intrusion_m=intrusion_m)
    labels = list(record.get("qwen_labels") or [])
    raw_labels = list(record.get("qwen_raw_labels") or [])
    qwen_side = qwen_construction_side(labels)
    raw_qwen_side = qwen_construction_side(raw_labels)
    consumed_side = consumed_construction_side(record)
    command_side = lateral_command_side(record, min_offset_m=min_offset_m)
    planned_command_side = planned_lateral_command_side(record, min_offset_m=min_offset_m)
    expected_command_side = "none"
    if requirement["required_side"] == "right":
      expected_command_side = "left"
    elif requirement["required_side"] == "left":
      expected_command_side = "right"

    if requirement["required"]:
      required_count += 1
      if qwen_side == requirement["required_side"]:
        qwen_side_correct += 1
      elif qwen_side == "none":
        qwen_side_missing += 1
      else:
        qwen_side_wrong += 1

      if consumed_side == requirement["required_side"]:
        consumed_side_correct += 1
      elif consumed_side == "none":
        consumed_side_missing += 1
      else:
        consumed_side_wrong += 1

      if command_side == expected_command_side:
        control_away_count += 1
      elif command_side == "none":
        control_neutral_count += 1
      else:
        control_toward_count += 1

      if planned_command_side == expected_command_side:
        planned_away_count += 1
      elif planned_command_side == "none":
        planned_neutral_count += 1
      else:
        planned_toward_count += 1
    else:
      not_required_count += 1
      if qwen_side != "none":
        false_construction_count += 1

    if _crashed(record):
      collision_count += 1
    if record.get("green_path_matches_tracked_path") is False:
      green_path_mismatch_count += 1

    rows.append({
      "frame_id": record.get("frame_id"),
      "model_frame_id": record.get("model_frame_id"),
      "rtp_source_frame_id": record.get("rtp_source_frame_id"),
      "required": requirement["required"],
      "required_side": requirement["required_side"],
      "required_shift_label": requirement["required_shift_label"],
      "ahead_m": requirement["ahead_m"],
      "lateral_delta_m": requirement["lateral_delta_m"],
      "qwen_labels": labels,
      "qwen_label_scores": record.get("qwen_label_scores"),
      "qwen_raw_labels": raw_labels,
      "qwen_raw_label_scores": record.get("qwen_raw_label_scores"),
      "qwen_labels_scored_this_request": record.get("qwen_labels_scored_this_request"),
      "qwen_score_group_index": record.get("qwen_score_group_index"),
      "qwen_label_state_debug": record.get("qwen_label_state_debug"),
      "qwen_side": qwen_side,
      "raw_qwen_side": raw_qwen_side,
      "consumed_construction_side": consumed_side,
      "expected_command_side": expected_command_side,
      "command_side": command_side,
      "planned_command_side": planned_command_side,
      "desired_lateral_offset_m": record.get("desired_lateral_offset_m"),
      "active_lateral_offset_m": record.get("active_lateral_offset_m"),
      "durable_avoidance_offset_m": record.get("durable_avoidance_offset_m"),
      "durable_lateral_plan_details": record.get("durable_lateral_plan_details"),
      "min_construction_route_clearance_m": record.get("min_construction_route_clearance_m"),
      "rtp_age_frames": record.get("rtp_age_frames"),
      "control_consumed_age_frames": record.get("control_consumed_age_frames"),
      "green_path_matches_tracked_path": record.get("green_path_matches_tracked_path"),
      "rtp_text": record.get("rtp_text"),
      "rtp_valid": record.get("rtp_valid"),
      "invalid_reason": record.get("invalid_reason"),
      "crashed": _crashed(record),
    })

  qwen_required_rate = None if required_count == 0 else qwen_side_correct / required_count
  consumed_required_rate = None if required_count == 0 else consumed_side_correct / required_count
  false_rate = None if not_required_count == 0 else false_construction_count / not_required_count
  control_away_rate = None if required_count == 0 else control_away_count / required_count
  planned_away_rate = None if required_count == 0 else planned_away_count / required_count
  rtp_ages = [int(row["rtp_age_frames"]) for row in rows if row["rtp_age_frames"] is not None]
  consumed_ages = [int(row["control_consumed_age_frames"]) for row in rows if row["control_consumed_age_frames"] is not None]
  first_side_frame = next((row["frame_id"] for row in rows if row["qwen_side"] != "none"), None)
  post_first_side_rows = [
    row for row in rows
    if first_side_frame is not None and row["frame_id"] is not None and row["frame_id"] >= first_side_frame
  ]
  post_first_side_required = [row for row in post_first_side_rows if row["required"]]
  post_first_side_not_required = [row for row in post_first_side_rows if not row["required"]]
  post_first_side_correct = sum(1 for row in post_first_side_required if row["qwen_side"] == row["required_side"])
  post_first_side_wrong = sum(1 for row in post_first_side_required if row["qwen_side"] not in ("none", row["required_side"]))
  post_first_side_missing = sum(1 for row in post_first_side_required if row["qwen_side"] == "none")
  post_first_side_false = sum(1 for row in post_first_side_not_required if row["qwen_side"] != "none")
  first_consumed_side_frame = next((row["frame_id"] for row in rows if row["consumed_construction_side"] != "none"), None)
  post_first_consumed_side_rows = [
    row for row in rows
    if first_consumed_side_frame is not None and row["frame_id"] is not None and row["frame_id"] >= first_consumed_side_frame
  ]
  post_first_consumed_side_required = [row for row in post_first_consumed_side_rows if row["required"]]
  post_first_consumed_side_correct = sum(1 for row in post_first_consumed_side_required if row["consumed_construction_side"] == row["required_side"])
  post_first_consumed_side_wrong = sum(1 for row in post_first_consumed_side_required if row["consumed_construction_side"] not in ("none", row["required_side"]))
  post_first_consumed_side_missing = sum(1 for row in post_first_consumed_side_required if row["consumed_construction_side"] == "none")
  startup_required_before_first_consumed_side = sum(
    1 for row in rows
    if row["required"]
    and first_consumed_side_frame is not None
    and row["frame_id"] is not None
    and row["frame_id"] < first_consumed_side_frame
  )
  return {
    "frames": len(records),
    "path_relevant_construction_frames": required_count,
    "qwen_side_correct": qwen_side_correct,
    "qwen_side_wrong": qwen_side_wrong,
    "qwen_side_missing": qwen_side_missing,
    "qwen_side_success_rate": qwen_required_rate,
    "consumed_plan_side_correct": consumed_side_correct,
    "consumed_plan_side_wrong": consumed_side_wrong,
    "consumed_plan_side_missing": consumed_side_missing,
    "consumed_plan_side_success_rate": consumed_required_rate,
    "not_path_relevant_frames": not_required_count,
    "false_construction_labels": false_construction_count,
    "false_construction_rate": false_rate,
    "control_away_frames": control_away_count,
    "control_toward_frames": control_toward_count,
    "control_neutral_frames": control_neutral_count,
    "control_away_rate": control_away_rate,
    "planned_away_frames": planned_away_count,
    "planned_toward_frames": planned_toward_count,
    "planned_neutral_frames": planned_neutral_count,
    "planned_away_rate": planned_away_rate,
    "collision_count": collision_count,
    "green_path_mismatch_count": green_path_mismatch_count,
    "min_construction_route_clearance_m": min((row["min_construction_route_clearance_m"] for row in rows if row["min_construction_route_clearance_m"] is not None), default=None),
    "max_rtp_age_frames": max(rtp_ages) if rtp_ages else 0,
    "max_consumed_age_frames": max(consumed_ages) if consumed_ages else 0,
    "first_qwen_side_frame": first_side_frame,
    "first_consumed_side_frame": first_consumed_side_frame,
    "startup_required_before_first_consumed_side_frames": startup_required_before_first_consumed_side,
    "post_first_side_path_relevant_construction_frames": len(post_first_side_required),
    "post_first_side_qwen_side_correct": post_first_side_correct,
    "post_first_side_qwen_side_wrong": post_first_side_wrong,
    "post_first_side_qwen_side_missing": post_first_side_missing,
    "post_first_side_qwen_side_success_rate": None if not post_first_side_required else post_first_side_correct / len(post_first_side_required),
    "post_first_side_not_path_relevant_frames": len(post_first_side_not_required),
    "post_first_side_false_construction_labels": post_first_side_false,
    "post_first_side_false_construction_rate": None if not post_first_side_not_required else post_first_side_false / len(post_first_side_not_required),
    "post_first_consumed_side_path_relevant_construction_frames": len(post_first_consumed_side_required),
    "post_first_consumed_side_correct": post_first_consumed_side_correct,
    "post_first_consumed_side_wrong": post_first_consumed_side_wrong,
    "post_first_consumed_side_missing": post_first_consumed_side_missing,
    "post_first_consumed_side_success_rate": None if not post_first_consumed_side_required else post_first_consumed_side_correct / len(post_first_consumed_side_required),
    "rows": rows,
  }


def _episode_path(path: Path, mode: str) -> Path:
  if path.is_file():
    return path
  candidate = path / mode / "episode.json"
  if candidate.exists():
    return candidate
  for fallback in ("vlm", "static", "stock"):
    candidate = path / fallback / "episode.json"
    if candidate.exists():
      return candidate
  raise FileNotFoundError(f"no episode.json found for {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate construction behavior from a MetaDrive RTP episode trace.")
  parser.add_argument("--episode", type=Path, required=True, help="Run directory or episode.json path.")
  parser.add_argument("--mode", default="vlm", help="Subdirectory to read when --episode is a run directory.")
  parser.add_argument("--horizon-m", type=float, default=14.0)
  parser.add_argument("--intrusion-m", type=float, default=1.65)
  parser.add_argument("--min-offset-m", type=float, default=0.10)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  episode_path = _episode_path(args.episode, args.mode)
  episode = json.loads(episode_path.read_text(encoding="utf-8"))
  result = evaluate_episode(
    episode,
    horizon_m=args.horizon_m,
    intrusion_m=args.intrusion_m,
    min_offset_m=args.min_offset_m,
  )
  result["episode"] = str(episode_path)
  result["horizon_m"] = float(args.horizon_m)
  result["intrusion_m"] = float(args.intrusion_m)
  result["min_offset_m"] = float(args.min_offset_m)

  out_path = args.out or (episode_path.parent / "construction_trace_evaluation.json")
  out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
  print(f"out={out_path}")


if __name__ == "__main__":
  main()
