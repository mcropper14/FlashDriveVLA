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


PEDESTRIAN_KIND_TOKENS = ("pedestrian", "person", "human")
PEDESTRIAN_LABELS = ("pedestrian_in_path", "pedestrian_entering_path")


def _as_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if result == result and result not in (float("inf"), float("-inf")) else None


def _kind_matches(item: dict[str, Any], tokens: tuple[str, ...]) -> bool:
  kind = str(item.get("kind", "")).lower()
  return any(token in kind for token in tokens)


def pedestrian_requirement_from_record(
  record: dict[str, Any],
  *,
  horizon_m: float,
  in_path_lateral_m: float,
  entering_lateral_m: float,
) -> dict[str, Any]:
  nearest = _nearest_pedestrian_from_record(record)
  ahead_m = nearest["ahead_m"]
  lateral_delta_m = nearest["lateral_delta_m"]
  target_lateral_delta_m = nearest["target_lateral_delta_m"]
  if ahead_m is None or lateral_delta_m is None:
    return {
      "required": False,
      "expected_label": "none",
      "expected_class": "none",
      "reason": "no_pedestrian_track",
      **nearest,
    }

  in_horizon = 0.0 <= ahead_m <= horizon_m
  in_path = in_horizon and abs(lateral_delta_m) <= in_path_lateral_m
  entering_path = False
  if in_horizon and target_lateral_delta_m is not None:
    moving_closer = abs(target_lateral_delta_m) < abs(lateral_delta_m)
    crosses_path = lateral_delta_m == 0.0 or lateral_delta_m * target_lateral_delta_m <= 0.0
    target_hits_path = abs(target_lateral_delta_m) <= in_path_lateral_m
    close_enough = abs(lateral_delta_m) <= entering_lateral_m
    entering_path = close_enough and moving_closer and (crosses_path or target_hits_path)

  if in_path:
    return {
      "required": True,
      "expected_label": "pedestrian_in_path",
      "expected_class": "in_path",
      "reason": "pedestrian_overlaps_path",
      **nearest,
    }
  if entering_path:
    return {
      "required": True,
      "expected_label": "pedestrian_entering_path",
      "expected_class": "entering_path",
      "reason": "pedestrian_entering_path",
      **nearest,
    }
  return {
    "required": False,
    "expected_label": "none",
    "expected_class": "none",
    "reason": "pedestrian_not_path_relevant",
    **nearest,
  }


def _nearest_pedestrian_from_record(record: dict[str, Any]) -> dict[str, Any]:
  from_spawned = _nearest_pedestrian_from_spawned(record)
  if from_spawned["ahead_m"] is not None and from_spawned["lateral_delta_m"] is not None:
    return from_spawned
  return {
    "ahead_m": _as_float(record.get("pedestrian_nearest_ahead_m")),
    "lateral_delta_m": _as_float(record.get("pedestrian_nearest_lateral_delta_m")),
    "target_lateral_delta_m": None,
    "route_distance_m": _as_float(record.get("min_pedestrian_route_clearance_m")),
    "object_lateral_m": _as_float(record.get("pedestrian_nearest_object_lateral_m")),
    "object_route_s_m": None,
    "kind": str(record.get("pedestrian_nearest_kind", "")),
    "speed_mps": None,
  }


def _nearest_pedestrian_from_spawned(record: dict[str, Any]) -> dict[str, Any]:
  spawned = record.get("spawned_scene")
  if not isinstance(spawned, list):
    return {
      "ahead_m": None,
      "lateral_delta_m": None,
      "target_lateral_delta_m": None,
      "route_distance_m": None,
      "object_lateral_m": None,
      "object_route_s_m": None,
      "kind": "",
      "speed_mps": None,
    }
  current_long_m = _as_float(record.get("route_longitudinal_m"))
  control_debug = record.get("control_debug") if isinstance(record.get("control_debug"), dict) else {}
  current_lateral_m = _as_float(control_debug.get("lane_lateral_m"))
  if current_long_m is None or current_lateral_m is None:
    return {
      "ahead_m": None,
      "lateral_delta_m": None,
      "target_lateral_delta_m": None,
      "route_distance_m": None,
      "object_lateral_m": None,
      "object_route_s_m": None,
      "kind": "",
      "speed_mps": None,
    }

  nearest: tuple[float, dict[str, Any], float, float] | None = None
  for item in spawned:
    if not isinstance(item, dict) or not _kind_matches(item, PEDESTRIAN_KIND_TOKENS):
      continue
    route_s_m = _as_float(item.get("route_s_m"))
    lateral_m = _as_float(item.get("lateral_m"))
    if route_s_m is None or lateral_m is None:
      continue
    ahead_m = route_s_m - current_long_m
    lateral_delta_m = lateral_m - current_lateral_m
    distance_m = (ahead_m * ahead_m + lateral_delta_m * lateral_delta_m) ** 0.5
    if nearest is None or distance_m < nearest[0]:
      nearest = (distance_m, item, ahead_m, lateral_delta_m)

  if nearest is None:
    return {
      "ahead_m": None,
      "lateral_delta_m": None,
      "target_lateral_delta_m": None,
      "route_distance_m": None,
      "object_lateral_m": None,
      "object_route_s_m": None,
      "kind": "",
      "speed_mps": None,
    }

  route_distance_m, item, ahead_m, lateral_delta_m = nearest
  target_lateral_m = _as_float(item.get("target_lateral_m"))
  return {
    "ahead_m": ahead_m,
    "lateral_delta_m": lateral_delta_m,
    "target_lateral_delta_m": None if target_lateral_m is None else target_lateral_m - current_lateral_m,
    "route_distance_m": route_distance_m,
    "object_lateral_m": _as_float(item.get("lateral_m")),
    "object_route_s_m": _as_float(item.get("route_s_m")),
    "kind": str(item.get("kind", "")),
    "speed_mps": _as_float(item.get("speed_mps")),
  }


def qwen_pedestrian_class(labels: list[str] | tuple[str, ...]) -> str:
  label_set = set(labels)
  has_in_path = "pedestrian_in_path" in label_set
  has_entering = "pedestrian_entering_path" in label_set
  if has_in_path:
    return "in_path"
  if has_entering:
    return "entering_path"
  return "none"


def consumed_path_agent(record: dict[str, Any]) -> bool:
  sources = record.get("durable_speed_plan_sources")
  if isinstance(sources, list) and any(str(source).startswith("corridor_object") for source in sources):
    return True
  text = str(record.get("rtp_text") or "")
  return "scene=path_conflict_agent" in text or "scene=mixed_agent_" in text or "corridor_object_" in text


def control_slowed_for_agent(record: dict[str, Any], *, slow_scale: float) -> bool:
  desired = _as_float(record.get("desired_speed_mps"))
  target = _as_float(record.get("target_speed_mps"))
  cap = _as_float(record.get("durable_speed_cap_mps"))
  if desired is None or desired <= 1e-3:
    return False
  threshold = desired * slow_scale
  if target is not None and target <= threshold:
    return True
  if cap is not None and cap <= threshold:
    return True
  return False


def control_response_ok_for_pedestrian(
  record: dict[str, Any],
  requirement: dict[str, Any],
  *,
  consumed_agent: bool,
  slow_scale: float,
  slow_distance_m: float,
) -> bool:
  if not requirement.get("required"):
    return False
  expected_class = str(requirement.get("expected_class") or "")
  ahead_m = _as_float(requirement.get("ahead_m"))
  slowed = control_slowed_for_agent(record, slow_scale=slow_scale)
  if expected_class == "in_path" and ahead_m is not None and ahead_m > slow_distance_m:
    return consumed_agent
  return slowed


def _crashed(record: dict[str, Any]) -> bool:
  flags = record.get("info_flags") if isinstance(record.get("info_flags"), dict) else {}
  return bool(flags.get("crash_human") or flags.get("crash_object") or flags.get("crash_vehicle"))


def evaluate_episode(
  episode: dict[str, Any],
  *,
  horizon_m: float = 30.0,
  in_path_lateral_m: float = 0.9,
  entering_lateral_m: float = 2.4,
  slow_scale: float = 0.75,
  slow_distance_m: float = 12.0,
) -> dict[str, Any]:
  records = list(episode.get("records", []))
  rows: list[dict[str, Any]] = []
  required_count = 0
  qwen_path_relevant = 0
  qwen_exact = 0
  qwen_wrong = 0
  qwen_missing = 0
  consumed_agent_count = 0
  control_success = 0
  not_required_count = 0
  false_qwen = 0
  false_consumed_agent = 0
  false_slow = 0
  collision_count = 0
  green_path_mismatch_count = 0

  for record in records:
    requirement = pedestrian_requirement_from_record(
      record,
      horizon_m=horizon_m,
      in_path_lateral_m=in_path_lateral_m,
      entering_lateral_m=entering_lateral_m,
    )
    labels = list(record.get("qwen_labels") or [])
    qwen_class = qwen_pedestrian_class(labels)
    consumed_agent = consumed_path_agent(record)
    slowed = control_slowed_for_agent(record, slow_scale=slow_scale)
    control_response_ok = control_response_ok_for_pedestrian(
      record,
      requirement,
      consumed_agent=consumed_agent,
      slow_scale=slow_scale,
      slow_distance_m=slow_distance_m,
    )
    if requirement["required"]:
      required_count += 1
      if qwen_class == requirement["expected_class"]:
        qwen_exact += 1
        qwen_path_relevant += 1
      elif qwen_class in ("in_path", "entering_path", "conflict"):
        qwen_path_relevant += 1
        qwen_wrong += 1
      elif qwen_class == "none":
        qwen_missing += 1
      else:
        qwen_wrong += 1
      if consumed_agent:
        consumed_agent_count += 1
      if control_response_ok:
        control_success += 1
    else:
      not_required_count += 1
      if qwen_class != "none":
        false_qwen += 1
      if consumed_agent:
        false_consumed_agent += 1
      if slowed:
        false_slow += 1

    if _crashed(record):
      collision_count += 1
    if record.get("green_path_matches_tracked_path") is False:
      green_path_mismatch_count += 1

    rows.append({
      "frame_id": record.get("frame_id"),
      "model_frame_id": record.get("model_frame_id"),
      "rtp_source_frame_id": record.get("rtp_source_frame_id"),
      "rtp_age_frames": record.get("rtp_age_frames"),
      "control_consumed_age_frames": record.get("control_consumed_age_frames"),
      "required": requirement["required"],
      "expected_class": requirement["expected_class"],
      "expected_label": requirement["expected_label"],
      "reason": requirement["reason"],
      "ahead_m": requirement["ahead_m"],
      "lateral_delta_m": requirement["lateral_delta_m"],
      "target_lateral_delta_m": requirement["target_lateral_delta_m"],
      "route_distance_m": requirement["route_distance_m"],
      "qwen_labels": labels,
      "qwen_label_scores": record.get("qwen_label_scores"),
      "qwen_pedestrian_class": qwen_class,
      "consumed_path_agent": consumed_agent,
      "control_slowed_for_agent": slowed,
      "control_response_ok_for_agent": control_response_ok,
      "target_speed_mps": record.get("target_speed_mps"),
      "desired_speed_mps": record.get("desired_speed_mps"),
      "durable_speed_cap_mps": record.get("durable_speed_cap_mps"),
      "durable_speed_plan_sources": record.get("durable_speed_plan_sources"),
      "min_pedestrian_route_clearance_m": record.get("min_pedestrian_route_clearance_m"),
      "green_path_matches_tracked_path": record.get("green_path_matches_tracked_path"),
      "rtp_text": record.get("rtp_text"),
      "rtp_valid": record.get("rtp_valid"),
      "invalid_reason": record.get("invalid_reason"),
      "crashed": _crashed(record),
    })

  required_rows = [row for row in rows if row["required"]]
  not_required_rows = [row for row in rows if not row["required"]]
  rtp_ages = [int(row["rtp_age_frames"]) for row in rows if row["rtp_age_frames"] is not None]
  consumed_ages = [int(row["control_consumed_age_frames"]) for row in rows if row["control_consumed_age_frames"] is not None]
  first_required_agent_frame = next((row["frame_id"] for row in required_rows if row["qwen_pedestrian_class"] != "none"), None)
  post_first_required = [
    row for row in required_rows
    if first_required_agent_frame is not None and row["frame_id"] is not None and row["frame_id"] >= first_required_agent_frame
  ]
  post_first_path_relevant = sum(1 for row in post_first_required if row["qwen_pedestrian_class"] in ("in_path", "entering_path", "conflict"))
  post_first_control = sum(1 for row in post_first_required if row["control_response_ok_for_agent"])

  return {
    "frames": len(records),
    "path_relevant_pedestrian_frames": required_count,
    "qwen_pedestrian_path_relevant": qwen_path_relevant,
    "qwen_pedestrian_exact": qwen_exact,
    "qwen_pedestrian_wrong": qwen_wrong,
    "qwen_pedestrian_missing": qwen_missing,
    "qwen_pedestrian_path_relevant_rate": None if required_count == 0 else qwen_path_relevant / required_count,
    "qwen_pedestrian_exact_rate": None if required_count == 0 else qwen_exact / required_count,
    "consumed_agent_frames": consumed_agent_count,
    "consumed_agent_rate": None if required_count == 0 else consumed_agent_count / required_count,
    "control_success_frames": control_success,
    "control_success_rate": None if required_count == 0 else control_success / required_count,
    "not_path_relevant_frames": not_required_count,
    "false_qwen_pedestrian_frames": false_qwen,
    "false_qwen_pedestrian_rate": None if not_required_count == 0 else false_qwen / not_required_count,
    "false_consumed_agent_frames": false_consumed_agent,
    "false_consumed_agent_rate": None if not_required_count == 0 else false_consumed_agent / not_required_count,
    "false_slow_frames": false_slow,
    "false_slow_rate": None if not_required_count == 0 else false_slow / not_required_count,
    "collision_count": collision_count,
    "green_path_mismatch_count": green_path_mismatch_count,
    "min_pedestrian_route_clearance_m": min((row["min_pedestrian_route_clearance_m"] for row in rows if row["min_pedestrian_route_clearance_m"] is not None), default=None),
    "max_rtp_age_frames": max(rtp_ages) if rtp_ages else 0,
    "max_consumed_age_frames": max(consumed_ages) if consumed_ages else 0,
    "first_required_agent_frame": first_required_agent_frame,
    "post_first_required_pedestrian_frames": len(post_first_required),
    "post_first_qwen_pedestrian_path_relevant": post_first_path_relevant,
    "post_first_qwen_pedestrian_path_relevant_rate": None if not post_first_required else post_first_path_relevant / len(post_first_required),
    "post_first_control_success_frames": post_first_control,
    "post_first_control_success_rate": None if not post_first_required else post_first_control / len(post_first_required),
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
  parser = argparse.ArgumentParser(description="Evaluate pedestrian path-conflict behavior from a MetaDrive RTP episode trace.")
  parser.add_argument("--episode", type=Path, required=True, help="Run directory or episode.json path.")
  parser.add_argument("--mode", default="vlm", help="Subdirectory to read when --episode is a run directory.")
  parser.add_argument("--horizon-m", type=float, default=30.0)
  parser.add_argument("--in-path-lateral-m", type=float, default=0.9)
  parser.add_argument("--entering-lateral-m", type=float, default=2.4)
  parser.add_argument("--slow-scale", type=float, default=0.75)
  parser.add_argument("--slow-distance-m", type=float, default=12.0)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  episode_path = _episode_path(args.episode, args.mode)
  episode = json.loads(episode_path.read_text(encoding="utf-8"))
  result = evaluate_episode(
    episode,
    horizon_m=args.horizon_m,
    in_path_lateral_m=args.in_path_lateral_m,
    entering_lateral_m=args.entering_lateral_m,
    slow_scale=args.slow_scale,
    slow_distance_m=args.slow_distance_m,
  )
  result["episode"] = str(episode_path)
  result["horizon_m"] = float(args.horizon_m)
  result["in_path_lateral_m"] = float(args.in_path_lateral_m)
  result["entering_lateral_m"] = float(args.entering_lateral_m)
  result["slow_scale"] = float(args.slow_scale)
  result["slow_distance_m"] = float(args.slow_distance_m)

  out_path = args.out or (episode_path.parent / "pedestrian_trace_evaluation.json")
  out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
  print(f"out={out_path}")


if __name__ == "__main__":
  main()
