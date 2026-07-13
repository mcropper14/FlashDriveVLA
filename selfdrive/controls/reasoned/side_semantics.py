from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


LEFT = "left"
RIGHT = "right"
UNKNOWN = "unknown"
NONE = "none"

CONSTRUCTION_SIDE_LABELS = ("construction_left", "construction_right")
CONSTRUCTION_EDGE_LABELS = ("construction_blue_edge", "construction_purple_edge")
CONSTRUCTION_ACTION_LABELS = ("construction_drive_left", "construction_drive_right")
CONSTRUCTION_SHIFT_LABELS = ("construction_shift_left", "construction_shift_right")
CONSTRUCTION_CANDIDATE_LABELS = ("construction_blocks_left_candidate", "construction_blocks_right_candidate")

CONSTRUCTION_PAIRED_LABELS = (
  CONSTRUCTION_ACTION_LABELS,
  CONSTRUCTION_SHIFT_LABELS,
  CONSTRUCTION_EDGE_LABELS,
  CONSTRUCTION_SIDE_LABELS,
  CONSTRUCTION_CANDIDATE_LABELS,
)

CONSTRUCTION_LABEL_TO_HAZARD_SIDE = {
  # Action/shift labels describe the safe maneuver, so their hazard side is opposite.
  "construction_drive_left": RIGHT,
  "construction_drive_right": LEFT,
  "construction_shift_left": RIGHT,
  "construction_shift_right": LEFT,
  # Scene-board edge labels are rendered in openpilot/driver coordinates.
  "construction_blue_edge": LEFT,
  "construction_purple_edge": RIGHT,
  # Semantic and candidate labels name the obstructed/hazard side directly.
  "construction_left": LEFT,
  "construction_right": RIGHT,
  "construction_blocks_left_candidate": LEFT,
  "construction_blocks_right_candidate": RIGHT,
}


@dataclass(frozen=True)
class ConstructionRtpFields:
  scene: str
  evidence: str
  meta: str
  lat_bias_m: float
  avoid_token: str


def lateral_side_openpilot(lateral_m: float | None, deadband_m: float = 1e-3) -> str:
  if lateral_m is None:
    return NONE
  lateral = float(lateral_m)
  if lateral > deadband_m:
    return LEFT
  if lateral < -deadband_m:
    return RIGHT
  return NONE


def openpilot_to_metadrive_lateral_m(lateral_m: float) -> float:
  # Openpilot/PathSynth convention: positive lateral is ego-left.
  # This POC's MetaDrive route adapter convention: positive lane lateral is
  # ego-right, matching observed route spawning and local_coordinates behavior.
  return -float(lateral_m)


def metadrive_to_openpilot_lateral_m(lateral_m: float) -> float:
  return -float(lateral_m)


def lateral_side_metadrive(lateral_m: float | None, deadband_m: float = 1e-3) -> str:
  return lateral_side_openpilot(None if lateral_m is None else metadrive_to_openpilot_lateral_m(lateral_m), deadband_m)


def normalized_lateral_sign_to_openpilot(sign: float | None, default: float = -1.0) -> float:
  try:
    value = float(default if sign is None else sign)
  except (TypeError, ValueError):
    value = float(default)
  return 1.0 if value >= 0.0 else -1.0


def route_to_openpilot_lateral_m(route_lateral_m: float, route_lateral_sign_to_openpilot: float) -> float:
  return normalized_lateral_sign_to_openpilot(route_lateral_sign_to_openpilot) * float(route_lateral_m)


def openpilot_to_route_lateral_m(openpilot_lateral_m: float, route_lateral_sign_to_openpilot: float) -> float:
  return normalized_lateral_sign_to_openpilot(route_lateral_sign_to_openpilot) * float(openpilot_lateral_m)


def lateral_side_route(route_lateral_m: float | None, route_lateral_sign_to_openpilot: float, deadband_m: float = 1e-3) -> str:
  return lateral_side_openpilot(
    None if route_lateral_m is None else route_to_openpilot_lateral_m(route_lateral_m, route_lateral_sign_to_openpilot),
    deadband_m,
  )


def route_lateral_for_openpilot_side(side: str, magnitude_m: float, route_lateral_sign_to_openpilot: float) -> float:
  magnitude = abs(float(magnitude_m))
  if side == LEFT:
    return openpilot_to_route_lateral_m(magnitude, route_lateral_sign_to_openpilot)
  if side == RIGHT:
    return openpilot_to_route_lateral_m(-magnitude, route_lateral_sign_to_openpilot)
  return 0.0


def metadrive_lateral_for_side(side: str, magnitude_m: float) -> float:
  magnitude = abs(float(magnitude_m))
  if side == LEFT:
    return -magnitude
  if side == RIGHT:
    return magnitude
  return 0.0


def avoidance_openpilot_lateral_for_hazard_side(hazard_side: str, magnitude_m: float) -> float:
  magnitude = abs(float(magnitude_m))
  if hazard_side == RIGHT:
    return magnitude
  if hazard_side == LEFT:
    return -magnitude
  return 0.0


def construction_hazard_side_from_labels(labels: Iterable[str]) -> str:
  label_set = set(labels)
  for left_label, right_label in CONSTRUCTION_PAIRED_LABELS:
    has_left = left_label in label_set
    has_right = right_label in label_set
    if has_left and not has_right:
      return CONSTRUCTION_LABEL_TO_HAZARD_SIDE[left_label]
    if has_right and not has_left:
      return CONSTRUCTION_LABEL_TO_HAZARD_SIDE[right_label]
  return UNKNOWN


def construction_rtp_fields_for_hazard_side(hazard_side: str) -> ConstructionRtpFields:
  if hazard_side == LEFT:
    return ConstructionRtpFields(
      scene="construction_left",
      evidence="cones_barrier_left_edge",
      meta="BIAS_RIGHT",
      lat_bias_m=avoidance_openpilot_lateral_for_hazard_side(LEFT, 1.25),
      avoid_token="left_edge_s8_48_margin1.25",
    )
  if hazard_side == RIGHT:
    return ConstructionRtpFields(
      scene="construction_right",
      evidence="cones_barrier_right_edge",
      meta="BIAS_LEFT",
      lat_bias_m=avoidance_openpilot_lateral_for_hazard_side(RIGHT, 1.25),
      avoid_token="right_edge_s8_48_margin1.25",
    )
  return ConstructionRtpFields(
    scene="construction_presence_unknown",
    evidence="cones_barrier_side_unknown",
    meta="BASE",
    lat_bias_m=0.0,
    avoid_token="",
  )


def construction_hazard_side_from_avoid_token(source_token: str) -> str:
  if source_token.startswith("right_edge_"):
    return RIGHT
  if source_token.startswith("left_edge_"):
    return LEFT
  return NONE


def construction_avoidance_openpilot_side_valid(source_token: str, openpilot_lateral_m: float | None) -> bool:
  hazard_side = construction_hazard_side_from_avoid_token(source_token)
  target_side = lateral_side_openpilot(openpilot_lateral_m)
  if hazard_side == RIGHT:
    return target_side == LEFT
  if hazard_side == LEFT:
    return target_side == RIGHT
  return True


def construction_hazard_metadrive_lateral_for_side(hazard_side: str, magnitude_m: float) -> float:
  return metadrive_lateral_for_side(hazard_side, magnitude_m)


def construction_avoidance_metadrive_lateral_for_hazard_side(hazard_side: str, magnitude_m: float) -> float:
  return openpilot_to_metadrive_lateral_m(
    avoidance_openpilot_lateral_for_hazard_side(hazard_side, magnitude_m)
  )


def construction_avoidance_metadrive_side_valid(source_token: str, metadrive_lateral_m: float | None) -> bool:
  return construction_avoidance_openpilot_side_valid(
    source_token,
    None if metadrive_lateral_m is None else metadrive_to_openpilot_lateral_m(metadrive_lateral_m),
  )


def construction_avoidance_route_side_valid(source_token: str, route_lateral_m: float | None, route_lateral_sign_to_openpilot: float) -> bool:
  return construction_avoidance_openpilot_side_valid(
    source_token,
    None if route_lateral_m is None else route_to_openpilot_lateral_m(route_lateral_m, route_lateral_sign_to_openpilot),
  )
