#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.lead_semantics import (
  NO_SLOW_LEAD_CLASSES,
  REQUIRED_LEAD_CLASSES,
  canonical_lead_class,
  classify_lead_track,
  qwen_lead_class_from_labels,
)


VEHICLE_KIND_TOKENS = ("lead_vehicle", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")


def _as_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if result == result and result not in (float("inf"), float("-inf")) else None


def _as_int(value: Any) -> int | None:
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def qwen_lead_class(record: dict[str, Any]) -> str:
  direct = canonical_lead_class(record.get("lead_class"))
  if direct:
    return direct
  return qwen_lead_class_from_labels(record.get("qwen_labels"))


def lead_requirement_from_record(
  record: dict[str, Any],
  *,
  horizon_m: float = 60.0,
  path_lateral_m: float = 1.35,
  stopped_speed_mps: float = 0.35,
  slower_closing_mps: float = 0.45,
  slower_rel_loss_mps: float = 0.45,
  slower_desired_loss_mps: float = 0.45,
  braking_closing_mps: float = 1.40,
  braking_accel_mps2: float = -0.80,
  cut_in_lateral_rate_mps: float = 0.25,
  crossing_lateral_rate_mps: float = 0.60,
  crossing_max_longitudinal_speed_mps: float = 1.00,
) -> dict[str, Any]:
  return classify_lead_track(
    record,
    horizon_m=horizon_m,
    path_lateral_m=path_lateral_m,
    stopped_speed_mps=stopped_speed_mps,
    slower_closing_mps=slower_closing_mps,
    slower_rel_loss_mps=slower_rel_loss_mps,
    slower_desired_loss_mps=slower_desired_loss_mps,
    braking_closing_mps=braking_closing_mps,
    braking_accel_mps2=braking_accel_mps2,
    cut_in_lateral_rate_mps=cut_in_lateral_rate_mps,
    crossing_lateral_rate_mps=crossing_lateral_rate_mps,
    crossing_max_longitudinal_speed_mps=crossing_max_longitudinal_speed_mps,
  )


def _nominal_speed_mps(episode: dict[str, Any]) -> float:
  for key in ("nominal_speed_mps", "desired_speed_mps", "speed_mps"):
    value = _as_float(episode.get(key))
    if value is not None:
      return value
  records = episode.get("records")
  if not isinstance(records, list):
    return 0.0
  targets = [_as_float(record.get("target_speed_mps")) for record in records if isinstance(record, dict)]
  targets = [value for value in targets if value is not None]
  if targets:
    return max(targets)
  speeds = [_as_float(record.get("speed_mps")) for record in records if isinstance(record, dict)]
  speeds = [value for value in speeds if value is not None]
  return max(speeds, default=0.0)


def _speed_response(record: dict[str, Any], nominal_speed_mps: float, *, min_slow_delta_mps: float = 0.25) -> dict[str, Any]:
  target = _as_float(record.get("target_speed_mps"))
  cap = _as_float(record.get("durable_speed_cap_mps"))
  effective = target if target is not None else cap
  if effective is None:
    return {"target_speed_mps": target, "effective_cap_mps": cap, "slowed": False, "strong_slow": False, "stopped": False}
  slowed = effective <= nominal_speed_mps - min_slow_delta_mps and effective <= nominal_speed_mps * 0.95
  return {
    "target_speed_mps": target,
    "effective_cap_mps": cap,
    "slowed": bool(slowed),
    "strong_slow": bool(effective <= nominal_speed_mps * 0.60),
    "stopped": bool(effective <= 0.50),
  }


def _vehicle_route_clearance_ok(record: dict[str, Any], min_vehicle_clearance_m: float) -> bool:
  clearance = _as_float(record.get("min_vehicle_route_clearance_m"))
  return clearance is not None and clearance >= min_vehicle_clearance_m


def _control_satisfies(action: str, speed_response: dict[str, Any], record: dict[str, Any], *, min_vehicle_clearance_m: float) -> bool:
  if action == "none":
    return not bool(speed_response["slowed"])
  if action == "slow":
    return bool(speed_response["slowed"]) and not bool(speed_response["stopped"])
  if action == "strong_slow":
    return bool(speed_response["strong_slow"])
  if action == "stop":
    return bool(speed_response["stopped"] or _vehicle_route_clearance_ok(record, min_vehicle_clearance_m))
  if action == "yield":
    return bool(speed_response["strong_slow"] or speed_response["stopped"] or _vehicle_route_clearance_ok(record, min_vehicle_clearance_m))
  return False


def _crashed(record: dict[str, Any]) -> bool:
  flags = record.get("info_flags") if isinstance(record.get("info_flags"), dict) else {}
  return bool(flags.get("crash_object") or flags.get("crash_vehicle") or flags.get("crash_human"))


def _angle_delta_rad(actual_heading: float, route_heading: float) -> float:
  return (float(actual_heading) - float(route_heading) + 3.141592653589793) % (2.0 * 3.141592653589793) - 3.141592653589793


def _vehicle_kind_matches(kind: str) -> bool:
  normalized = str(kind).lower()
  return any(token in normalized for token in VEHICLE_KIND_TOKENS)


def _record_vehicle_heading_checks(record: dict[str, Any]) -> tuple[int, int, int, float | None]:
  """Return checked, missing, violations, min alignment cosine for vehicle heading evidence."""
  checked = 0
  missing = 0
  violations = 0
  alignments: list[float] = []

  top_route = _as_float(record.get("lead_route_heading_theta"))
  top_actual = _as_float(record.get("lead_actual_heading_theta"))
  if top_actual is None:
    top_actual = _as_float(record.get("lead_visual_heading_theta"))
  if top_route is not None and top_actual is not None:
    checked += 1
    alignment = float(math.cos(_angle_delta_rad(top_actual, top_route)))
    alignments.append(alignment)
    violations += int(alignment < 0.0)

  spawned = record.get("spawned_scene")
  if isinstance(spawned, list):
    for item in spawned:
      if not isinstance(item, dict) or not _vehicle_kind_matches(str(item.get("kind", ""))):
        continue
      route_heading = _as_float(item.get("route_heading_theta"))
      actual_heading = _as_float(item.get("actual_heading_theta"))
      if actual_heading is None:
        actual_heading = _as_float(item.get("visual_heading_theta"))
      if route_heading is None or actual_heading is None:
        missing += 1
        continue
      checked += 1
      alignment = float(math.cos(_angle_delta_rad(actual_heading, route_heading)))
      alignments.append(alignment)
      violations += int(alignment < 0.0)

  if checked == 0 and _as_int(record.get("lead_present")) == 1:
    missing += 1

  return checked, missing, violations, (min(alignments) if alignments else None)


def evaluate_episode(
  episode: dict[str, Any],
  *,
  horizon_m: float = 60.0,
  path_lateral_m: float = 1.35,
  min_vehicle_clearance_m: float = 2.0,
  max_allowed_age_frames: int | None = None,
) -> dict[str, Any]:
  records = [record for record in episode.get("records", []) if isinstance(record, dict)]
  nominal_speed = _nominal_speed_mps(episode)
  rows: list[dict[str, Any]] = []
  per_class: dict[str, dict[str, int]] = {}
  false_slow_total = 0
  false_slow_count = 0
  required_total = 0
  required_qwen_correct = 0
  required_control_ok = 0
  max_age = 0
  age_violation_count = 0
  collision_count = 0
  vehicle_heading_checked_frames = 0
  vehicle_heading_missing_frames = 0
  vehicle_heading_violation_count = 0
  min_vehicle_heading_alignment_cos: float | None = None

  for record in records:
    semantic_record = dict(record)
    semantic_record.setdefault("desired_speed_mps", nominal_speed)
    requirement = lead_requirement_from_record(semantic_record, horizon_m=horizon_m, path_lateral_m=path_lateral_m)
    expected_class = str(requirement["expected_class"])
    qwen_class = qwen_lead_class(record)
    speed_response = _speed_response(record, nominal_speed)
    action = str(requirement["action"])
    vehicle_clearance_ok = _vehicle_route_clearance_ok(record, min_vehicle_clearance_m)
    control_ok = _control_satisfies(action, speed_response, record, min_vehicle_clearance_m=min_vehicle_clearance_m)
    qwen_correct = qwen_class == expected_class or (expected_class == "none" and qwen_class == "")
    required = expected_class in REQUIRED_LEAD_CLASSES
    no_slow = expected_class in NO_SLOW_LEAD_CLASSES

    bucket = per_class.setdefault(expected_class, {"frames": 0, "qwen_correct": 0, "control_ok": 0, "false_slow": 0})
    bucket["frames"] += 1
    bucket["qwen_correct"] += int(qwen_correct)
    bucket["control_ok"] += int(control_ok)
    if no_slow:
      false_slow_total += 1
      if speed_response["slowed"]:
        false_slow_count += 1
        bucket["false_slow"] += 1
    if required:
      required_total += 1
      required_qwen_correct += int(qwen_correct)
      required_control_ok += int(control_ok)

    age = _as_int(record.get("control_consumed_age_frames"))
    if age is None:
      age = _as_int(record.get("rtp_age_frames"))
    if age is not None:
      max_age = max(max_age, age)
      if max_allowed_age_frames is not None and age > max_allowed_age_frames:
        age_violation_count += 1
    if _crashed(record):
      collision_count += 1
    heading_checked, heading_missing, heading_violations, min_heading_alignment = _record_vehicle_heading_checks(record)
    if heading_checked > 0:
      vehicle_heading_checked_frames += 1
    if heading_missing > 0:
      vehicle_heading_missing_frames += 1
    if heading_violations > 0:
      vehicle_heading_violation_count += 1
    if min_heading_alignment is not None:
      min_vehicle_heading_alignment_cos = (
        min_heading_alignment if min_vehicle_heading_alignment_cos is None
        else min(min_vehicle_heading_alignment_cos, min_heading_alignment)
      )

    rows.append({
      "frame_id": record.get("frame_id"),
      "expected_class": expected_class,
      "expected_action": action,
      "requirement_reason": requirement["reason"],
      "scenario_expected_lead_class": record.get("expected_lead_class"),
      "qwen_class": qwen_class,
      "qwen_correct": qwen_correct,
      "control_ok": control_ok,
      "lead_present": record.get("lead_present"),
      "lead_distance_m": record.get("lead_distance_m"),
      "lead_lateral_m": record.get("lead_lateral_m"),
      "lead_speed_mps": record.get("lead_speed_mps"),
      "lead_rel_speed_mps": record.get("lead_rel_speed_mps"),
      "lead_closing_mps": record.get("lead_closing_mps"),
      "lead_accel_mps2": record.get("lead_accel_mps2"),
      "lead_lateral_velocity_mps": record.get("lead_lateral_velocity_mps"),
      "lead_route_heading_theta": record.get("lead_route_heading_theta"),
      "lead_actual_heading_theta": record.get("lead_actual_heading_theta"),
      "lead_heading_error_rad": record.get("lead_heading_error_rad"),
      "lead_heading_alignment_cos": record.get("lead_heading_alignment_cos"),
      "lead_heading_same_direction": record.get("lead_heading_same_direction"),
      "vehicle_heading_checked": heading_checked,
      "vehicle_heading_missing": heading_missing,
      "vehicle_heading_violations": heading_violations,
      "vehicle_heading_min_alignment_cos": min_heading_alignment,
      "target_speed_mps": speed_response["target_speed_mps"],
      "durable_speed_cap_mps": speed_response["effective_cap_mps"],
      "slowed": speed_response["slowed"],
      "strong_slow": speed_response["strong_slow"],
      "stopped": speed_response["stopped"],
      "min_vehicle_route_clearance_m": record.get("min_vehicle_route_clearance_m"),
      "vehicle_route_clearance_ok": vehicle_clearance_ok,
      "lead_speed_guard_clear_reason": record.get("lead_speed_guard_clear_reason"),
      "lead_speed_guard_cleared_sources": record.get("lead_speed_guard_cleared_sources"),
      "rtp_age_frames": record.get("rtp_age_frames"),
      "control_consumed_age_frames": record.get("control_consumed_age_frames"),
      "crashed": _crashed(record),
    })

  per_class_rates = {
    key: {
      **value,
      "qwen_success_rate": None if value["frames"] == 0 else value["qwen_correct"] / value["frames"],
      "control_success_rate": None if value["frames"] == 0 else value["control_ok"] / value["frames"],
      "false_slow_rate": None if value["frames"] == 0 else value["false_slow"] / value["frames"],
    }
    for key, value in sorted(per_class.items())
  }
  return {
    "frames": len(records),
    "nominal_speed_mps": nominal_speed,
    "required_lead_frames": required_total,
    "required_qwen_correct": required_qwen_correct,
    "required_qwen_success_rate": None if required_total == 0 else required_qwen_correct / required_total,
    "required_control_ok": required_control_ok,
    "required_control_success_rate": None if required_total == 0 else required_control_ok / required_total,
    "no_slow_frames": false_slow_total,
    "false_slow_frames": false_slow_count,
    "false_slow_rate": None if false_slow_total == 0 else false_slow_count / false_slow_total,
    "max_consumed_age_frames": max_age,
    "age_violation_count": age_violation_count,
    "collision_count": collision_count,
    "vehicle_heading_checked_frames": vehicle_heading_checked_frames,
    "vehicle_heading_missing_frames": vehicle_heading_missing_frames,
    "vehicle_heading_violation_count": vehicle_heading_violation_count,
    "min_vehicle_heading_alignment_cos": min_vehicle_heading_alignment_cos,
    "per_class": per_class_rates,
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
  parser = argparse.ArgumentParser(description="Evaluate lead behavior from a MetaDrive RTP episode trace.")
  parser.add_argument("--episode", type=Path, required=True, help="Run directory or episode.json path.")
  parser.add_argument("--mode", default="vlm", help="Subdirectory to read when --episode is a run directory.")
  parser.add_argument("--horizon-m", type=float, default=60.0)
  parser.add_argument("--path-lateral-m", type=float, default=1.35)
  parser.add_argument("--min-vehicle-clearance-m", type=float, default=2.0)
  parser.add_argument("--max-allowed-age-frames", type=int, default=None)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  episode_path = _episode_path(args.episode, args.mode)
  episode = json.loads(episode_path.read_text(encoding="utf-8"))
  result = evaluate_episode(
    episode,
    horizon_m=args.horizon_m,
    path_lateral_m=args.path_lateral_m,
    min_vehicle_clearance_m=args.min_vehicle_clearance_m,
    max_allowed_age_frames=args.max_allowed_age_frames,
  )
  result["episode"] = str(episode_path)
  result["horizon_m"] = float(args.horizon_m)
  result["path_lateral_m"] = float(args.path_lateral_m)
  result["min_vehicle_clearance_m"] = float(args.min_vehicle_clearance_m)

  out_path = args.out or (episode_path.parent / "lead_trace_evaluation.json")
  out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
  print(f"out={out_path}")


if __name__ == "__main__":
  main()
