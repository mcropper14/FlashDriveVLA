from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


LEAD_CLASS_ALIASES = {
  "true_moving_lead": "true_moving_lead",
  "slower_lead": "slower_lead",
  "slower_lead_closing": "slower_lead",
  "braking_lead": "braking_lead",
  "braking_lead_closing": "braking_lead",
  "stopped_lead": "stopped_lead",
  "stopped_lead_in_path": "stopped_lead",
  "cut_in_vehicle": "cut_in_vehicle",
  "cut_in_vehicle_entering_path": "cut_in_vehicle",
  "crossing_vehicle": "crossing_vehicle",
  "crossing_vehicle_conflict": "crossing_vehicle",
  "irrelevant_vehicle": "irrelevant_vehicle",
}
LEAD_CLASS_PRIORITY = (
  "stopped_lead",
  "braking_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
  "slower_lead",
  "true_moving_lead",
  "irrelevant_vehicle",
)
REQUIRED_LEAD_CLASSES = {"slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle"}
NO_SLOW_LEAD_CLASSES = {"none", "true_moving_lead", "irrelevant_vehicle"}
LEAD_CHOICE_WORD_BY_CLASS = {
  "true_moving_lead": "moving",
  "slower_lead": "slower",
  "braking_lead": "braking",
  "stopped_lead": "stopped",
  "cut_in_vehicle": "merge",
  "crossing_vehicle": "crossing",
  "irrelevant_vehicle": "irrelevant",
}


def _as_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def canonical_lead_class(value: Any) -> str:
  return LEAD_CLASS_ALIASES.get(str(value or "").strip(), "")


def qwen_lead_class_from_labels(labels: Sequence[str] | None) -> str:
  if not isinstance(labels, (list, tuple)):
    return ""
  label_set = set(str(label) for label in labels)
  for label in LEAD_CLASS_PRIORITY:
    if label in label_set:
      return label
  return ""


def lead_choice_word(lead_class: str) -> str | None:
  return LEAD_CHOICE_WORD_BY_CLASS.get(canonical_lead_class(lead_class) or lead_class)


def lead_track_metrics(record: Mapping[str, Any]) -> dict[str, float]:
  lead_present = _as_float(record.get("lead_present"))
  desired_speed_mps = _as_float(record.get("desired_speed_mps"))
  lead_distance_m = _as_float(record.get("lead_distance_m"))
  lead_lateral_m = _as_float(record.get("lead_lateral_m"))
  lead_speed_mps = _as_float(record.get("lead_speed_mps"))
  lead_rel_speed_mps = _as_float(record.get("lead_rel_speed_mps"))
  lead_closing_mps = _as_float(record.get("lead_closing_mps"))
  lead_accel_mps2 = _as_float(record.get("lead_accel_mps2"))
  lead_lateral_velocity_mps = _as_float(record.get("lead_lateral_velocity_mps"))
  return {
    "lead_present": 0.0 if lead_present is None else lead_present,
    "desired_speed_mps": math.nan if desired_speed_mps is None else desired_speed_mps,
    "lead_distance_m": math.nan if lead_distance_m is None else lead_distance_m,
    "lead_lateral_m": math.nan if lead_lateral_m is None else lead_lateral_m,
    "lead_speed_mps": math.nan if lead_speed_mps is None else lead_speed_mps,
    "lead_rel_speed_mps": math.nan if lead_rel_speed_mps is None else lead_rel_speed_mps,
    "lead_closing_mps": math.nan if lead_closing_mps is None else lead_closing_mps,
    "lead_accel_mps2": math.nan if lead_accel_mps2 is None else lead_accel_mps2,
    "lead_lateral_velocity_mps": math.nan if lead_lateral_velocity_mps is None else lead_lateral_velocity_mps,
  }


def _moving_toward_path(lateral_m: float, lateral_velocity_mps: float, path_lateral_m: float, min_rate_mps: float) -> bool:
  if abs(lateral_m) <= path_lateral_m:
    return True
  if lateral_m > path_lateral_m:
    return lateral_velocity_mps < -abs(min_rate_mps)
  if lateral_m < -path_lateral_m:
    return lateral_velocity_mps > abs(min_rate_mps)
  return False


def classify_lead_track(
  record: Mapping[str, Any],
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
  lead_present = _as_int(record.get("lead_present"))
  if lead_present != 1:
    return {"expected_class": "none", "action": "none", "reason": "no_lead_track", "required": False}

  distance_m = _as_float(record.get("lead_distance_m"))
  lateral_m = _as_float(record.get("lead_lateral_m"))
  desired_speed_mps = _as_float(record.get("desired_speed_mps"))
  lead_speed_mps = _as_float(record.get("lead_speed_mps"))
  rel_speed_mps = _as_float(record.get("lead_rel_speed_mps"))
  closing_mps = _as_float(record.get("lead_closing_mps"))
  accel_mps2 = _as_float(record.get("lead_accel_mps2"))
  lateral_velocity_mps = _as_float(record.get("lead_lateral_velocity_mps"))
  if None in (distance_m, lateral_m, lead_speed_mps, rel_speed_mps, closing_mps, accel_mps2, lateral_velocity_mps):
    return {"expected_class": "none", "action": "none", "reason": "incomplete_track", "required": False}

  assert distance_m is not None and lateral_m is not None and lead_speed_mps is not None
  assert rel_speed_mps is not None and closing_mps is not None and accel_mps2 is not None and lateral_velocity_mps is not None

  if distance_m < -0.5 or distance_m > horizon_m:
    return {"expected_class": "irrelevant_vehicle", "action": "none", "reason": "not_ahead_in_horizon", "required": False}

  in_path = abs(lateral_m) <= path_lateral_m
  moving_toward = _moving_toward_path(lateral_m, lateral_velocity_mps, path_lateral_m, cut_in_lateral_rate_mps)
  if not in_path:
    if not moving_toward:
      return {"expected_class": "irrelevant_vehicle", "action": "none", "reason": "outside_path_not_entering", "required": False}
    if abs(lateral_velocity_mps) >= crossing_lateral_rate_mps and lead_speed_mps <= crossing_max_longitudinal_speed_mps:
      return {"expected_class": "crossing_vehicle", "action": "yield", "reason": "lateral_crossing_conflict", "required": True}
    return {"expected_class": "cut_in_vehicle", "action": "yield", "reason": "vehicle_entering_path", "required": True}

  if lead_speed_mps <= stopped_speed_mps:
    return {"expected_class": "stopped_lead", "action": "stop", "reason": "stopped_in_path", "required": True}
  if accel_mps2 <= braking_accel_mps2 or closing_mps >= braking_closing_mps:
    return {"expected_class": "braking_lead", "action": "strong_slow", "reason": "braking_or_rapid_closing", "required": True}
  desired_loss_mps = None if desired_speed_mps is None else desired_speed_mps - lead_speed_mps
  if (
    closing_mps >= slower_closing_mps or
    rel_speed_mps <= -abs(slower_rel_loss_mps) or
    (desired_loss_mps is not None and desired_loss_mps >= slower_desired_loss_mps)
  ):
    return {"expected_class": "slower_lead", "action": "slow", "reason": "closing_on_slower_lead", "required": True}
  return {"expected_class": "true_moving_lead", "action": "none", "reason": "stable_or_opening_lead", "required": False}
