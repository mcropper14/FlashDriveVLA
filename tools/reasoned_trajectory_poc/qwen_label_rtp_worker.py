#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from selfdrive.controls.reasoned.side_semantics import (
  CONSTRUCTION_ACTION_LABELS,
  CONSTRUCTION_CANDIDATE_LABELS,
  CONSTRUCTION_EDGE_LABELS,
  CONSTRUCTION_LABEL_TO_HAZARD_SIDE,
  CONSTRUCTION_SHIFT_LABELS,
  CONSTRUCTION_SIDE_LABELS,
  construction_avoidance_openpilot_side_valid,
  construction_hazard_side_from_labels,
  construction_rtp_fields_for_hazard_side,
)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "vlm" / "qwen2_5_vl_3b_instruct"
DEFAULT_SIGNAL_STOP_S = 18.0


def _signal_stop_s_from_env() -> float:
  try:
    return float(os.getenv("RTP_SIGNAL_STOP_S", str(DEFAULT_SIGNAL_STOP_S)))
  except ValueError:
    return DEFAULT_SIGNAL_STOP_S

LABEL_PROMPT = """You are inspecting a driving scene board for control-relevant hazards.
You receive multiple views of the same scene:
image 1: full driver UI overlay scene board.
image 2: zoomed center planned corridor.
image 3: zoomed lower/mid road region.
image 4: zoomed forward signal/object region.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,construction_drive_left,construction_drive_right,construction_shift_left,construction_shift_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,true_moving_lead,slower_lead,braking_lead,stopped_lead,cut_in_vehicle,crossing_vehicle,irrelevant_vehicle,animal_in_path,animal_entering_path,red_stop_light,green_go_light,stop_sign,none

Rules:
- Only consider hazards that overlap the green planned path, intrude into the green corridor, narrow/block the path ahead, or are moving/imminently moving into the green corridor.
- Ignore obstacles that are merely visible beside the road, behind lane lines, on shoulders, along walls, darkened outside the corridor focus band, or far outside the planned path.
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers overlap, intrude into, narrow, or block the green planned path.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel overlaps, intrudes into, narrows, or blocks the green planned path.
- Include construction_left only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the blue / ego-left edge of the green planned path.
- Include construction_right only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the purple / ego-right edge of the green planned path.
- Include construction_drive_left only if the safe bounded driving response is to drive left, toward image-left / driver-left, away from construction on the image-right / driver-right side.
- Include construction_drive_right only if the safe bounded driving response is to drive right, toward image-right / driver-right, away from construction on the image-left / driver-left side.
- Include construction_shift_left only if the safe bounded path response is to shift left away from a construction hazard.
- Include construction_shift_right only if the safe bounded path response is to shift right away from a construction hazard.
- Ignore tiny distant cone rows, horizon dots, walls, UI text, colored overlay lines/text, lane paint, and decorative/off-path construction objects that do not affect the green path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel and does not fit a more specific lead/cut-in/crossing label.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel and does not fit a more specific cut_in_vehicle or crossing_vehicle label.
- Include true_moving_lead only for a vehicle ahead in the green planned path moving near ego speed with stable spacing; this means no extra slowdown or yield.
- Include slower_lead only for a vehicle ahead in the green planned path moving slower than ego or getting closer; this means proportional slowdown.
- Include braking_lead only for a lead vehicle ahead visibly braking, showing brake lights, or closing rapidly; this means stronger slowdown.
- Include stopped_lead only for a stopped or nearly stopped vehicle ahead blocking the green planned path; this means stop, creep, or route response.
- Include cut_in_vehicle only for a vehicle from an adjacent lane entering or about to enter the green planned path ahead.
- Include crossing_vehicle only for a vehicle crossing the green planned path from the side with a conflict risk.
- Include irrelevant_vehicle only for visible vehicles outside the green planned path or moving away without affecting ego path.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red traffic light or red arrow controls the ego lane, ego road, or green planned path, even if the signal head appears high or off-center in the image.
- Include green_go_light if a green traffic light or green arrow controls the ego lane, ego road, or green planned path and authorizes proceeding.
- Include stop_sign if a STOP sign faces or controls the ego lane, ego road, or green planned path, even if the sign is beside the lane rather than inside the green corridor.
- Do not include red_stop_light, green_go_light, or stop_sign for signals/signs that control cross traffic, side roads, parking lots, or another lane/path.
- Output none only if no listed hazard is visible.
- No prose."""

COMPOSITE_LABEL_PROMPT = """You are inspecting one composite driving scene board for control-relevant hazards.
The image is a 2x2 composite:
top-left: full driver UI overlay scene board.
top-right: zoomed center planned corridor.
bottom-left: zoomed lower/mid road region.
bottom-right: zoomed forward signal/object region.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,construction_drive_left,construction_drive_right,construction_shift_left,construction_shift_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,true_moving_lead,slower_lead,braking_lead,stopped_lead,cut_in_vehicle,crossing_vehicle,irrelevant_vehicle,animal_in_path,animal_entering_path,red_stop_light,green_go_light,stop_sign,none

Rules:
- Only consider hazards that overlap the green planned path, intrude into the green corridor, narrow/block the path ahead, or are moving/imminently moving into the green corridor.
- Ignore obstacles that are merely visible beside the road, behind lane lines, on shoulders, along walls, darkened outside the corridor focus band, or far outside the planned path.
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers overlap, intrude into, narrow, or block the green planned path.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel overlaps, intrudes into, narrows, or blocks the green planned path.
- Include construction_left only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the blue / ego-left edge of the green planned path.
- Include construction_right only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the purple / ego-right edge of the green planned path.
- Include construction_drive_left only if the safe bounded driving response is to drive left, toward image-left / driver-left, away from construction on the image-right / driver-right side.
- Include construction_drive_right only if the safe bounded driving response is to drive right, toward image-right / driver-right, away from construction on the image-left / driver-left side.
- Include construction_shift_left only if the safe bounded path response is to shift left away from a construction hazard.
- Include construction_shift_right only if the safe bounded path response is to shift right away from a construction hazard.
- Ignore tiny distant cone rows, horizon dots, walls, UI text, colored overlay lines/text, lane paint, and decorative/off-path construction objects that do not affect the green path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel and does not fit a more specific lead/cut-in/crossing label.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel and does not fit a more specific cut_in_vehicle or crossing_vehicle label.
- Include true_moving_lead only for a vehicle ahead in the green planned path moving near ego speed with stable spacing; this means no extra slowdown or yield.
- Include slower_lead only for a vehicle ahead in the green planned path moving slower than ego or getting closer; this means proportional slowdown.
- Include braking_lead only for a lead vehicle ahead visibly braking, showing brake lights, or closing rapidly; this means stronger slowdown.
- Include stopped_lead only for a stopped or nearly stopped vehicle ahead blocking the green planned path; this means stop, creep, or route response.
- Include cut_in_vehicle only for a vehicle from an adjacent lane entering or about to enter the green planned path ahead.
- Include crossing_vehicle only for a vehicle crossing the green planned path from the side with a conflict risk.
- Include irrelevant_vehicle only for visible vehicles outside the green planned path or moving away without affecting ego path.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red traffic light or red arrow controls the ego lane, ego road, or green planned path, even if the signal head appears high or off-center in the image.
- Include green_go_light if a green traffic light or green arrow controls the ego lane, ego road, or green planned path and authorizes proceeding.
- Include stop_sign if a STOP sign faces or controls the ego lane, ego road, or green planned path, even if the sign is beside the lane rather than inside the green corridor.
- Do not include red_stop_light, green_go_light, or stop_sign for signals/signs that control cross traffic, side roads, parking lots, or another lane/path.
- Output none only if no listed hazard is visible.
- No prose."""

FULL_LABEL_PROMPT = """You are inspecting one full driver UI overlay scene board for control-relevant hazards.
The green overlay is the planned path of travel. The blue edge is ego-left and the purple edge is ego-right.

Output exactly a comma-separated subset of these labels:
cones,barrier,construction_left,construction_right,construction_drive_left,construction_drive_right,construction_shift_left,construction_shift_right,pedestrian_in_path,pedestrian_entering_path,vehicle_in_path,vehicle_entering_path,true_moving_lead,slower_lead,braking_lead,stopped_lead,cut_in_vehicle,crossing_vehicle,irrelevant_vehicle,animal_in_path,animal_entering_path,red_stop_light,green_go_light,stop_sign,none

Rules:
- Only consider hazards that overlap the green planned path, intrude into the green corridor, narrow/block the path ahead, or are moving/imminently moving into the green corridor.
- Ignore obstacles that are merely visible beside the road, behind lane lines, on shoulders, along walls, darkened outside the corridor focus band, or far outside the planned path.
- Include cones if traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers overlap, intrude into, narrow, or block the green planned path.
- Include barrier if a road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel overlaps, intrudes into, narrows, or blocks the green planned path.
- Include construction_left only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the blue / ego-left edge of the green planned path.
- Include construction_right only if a control-relevant cone/barrier hazard touches, overlaps, or narrows the purple / ego-right edge of the green planned path.
- Include construction_drive_left only if the safe bounded driving response is to drive left, toward image-left / driver-left, away from construction on the image-right / driver-right side.
- Include construction_drive_right only if the safe bounded driving response is to drive right, toward image-right / driver-right, away from construction on the image-left / driver-left side.
- Include construction_shift_left only if the safe bounded path response is to shift left away from a construction hazard.
- Include construction_shift_right only if the safe bounded path response is to shift right away from a construction hazard.
- Ignore tiny distant cone rows, horizon dots, walls, UI text, colored overlay lines/text, lane paint, and decorative/off-path construction objects that do not affect the green path.
- Include pedestrian_in_path only for a visible upright human figure, person, pedestrian, or human body currently overlapping the green planned path or standing/walking directly inside the ego lane corridor ahead. Count small, partially transparent, or green-tinted people if the overlay covers them.
- Include pedestrian_entering_path only for a visible upright human figure, person, pedestrian, or human body beside the green path that is clearly crossing, walking, stepping, or moving into the green planned path soon.
- Include vehicle_in_path only if a car, truck, bicycle, motorcycle, or similar road user is currently in the planned path of travel and does not fit a more specific lead/cut-in/crossing label.
- Include vehicle_entering_path only if such a road user is visibly about to move into the planned path of travel and does not fit a more specific cut_in_vehicle or crossing_vehicle label.
- Include true_moving_lead only for a vehicle ahead in the green planned path moving near ego speed with stable spacing; this means no extra slowdown or yield.
- Include slower_lead only for a vehicle ahead in the green planned path moving slower than ego or getting closer; this means proportional slowdown.
- Include braking_lead only for a lead vehicle ahead visibly braking, showing brake lights, or closing rapidly; this means stronger slowdown.
- Include stopped_lead only for a stopped or nearly stopped vehicle ahead blocking the green planned path; this means stop, creep, or route response.
- Include cut_in_vehicle only for a vehicle from an adjacent lane entering or about to enter the green planned path ahead.
- Include crossing_vehicle only for a vehicle crossing the green planned path from the side with a conflict risk.
- Include irrelevant_vehicle only for visible vehicles outside the green planned path or moving away without affecting ego path.
- Include animal_in_path only if an animal is currently in the planned path of travel.
- Include animal_entering_path only if an animal is visibly about to move into the planned path of travel.
- Pedestrian, vehicle, and animal path-conflict labels mean slow or stop in lane. They must not imply steering around the agent.
- Construction labels are the only labels that may imply a lateral path bias.
- Do not output any path-conflict label for cones, barriers, poles, signs, shadows, lane markings, sidewalks, or uncertain dark blobs.
- Do not output pedestrian labels for cones, barriers, signs, lane markers, traffic posts, poles, route arrows, UI text, or shadows.
- Do not yield for a person, vehicle, or animal that is visible but clearly outside the path of travel and not moving into it.
- Include red_stop_light if a red traffic light or red arrow controls the ego lane, ego road, or green planned path, even if the signal head appears high or off-center in the image.
- Include green_go_light if a green traffic light or green arrow controls the ego lane, ego road, or green planned path and authorizes proceeding.
- Include stop_sign if a STOP sign faces or controls the ego lane, ego road, or green planned path, even if the sign is beside the lane rather than inside the green corridor.
- Do not include red_stop_light, green_go_light, or stop_sign for signals/signs that control cross traffic, side roads, parking lots, or another lane/path.
- Output none only if no listed hazard is visible.
- No prose."""

SCORE_PROMPT = """Score one driving-scene label from the driver-view image.
The green overlay is the ego planned path.
Answer exactly yes or no.
Answer yes only when the question condition is directly visible.
Only consider hazards that overlap the green path, intrude into the green corridor, narrow/block the path ahead, or are moving/imminently moving into the green corridor.
Ignore objects that are merely visible beside the road, on shoulders, along walls, behind lane lines, darkened outside the corridor focus band, or far outside the planned path.
For pedestrian/vehicle/animal path questions, the agent must overlap the green path or clearly be entering it based on inferred path of travel.
For construction side questions, consider only control-relevant cones, pylons, bollards, barricades, checker panels, or blocked-lane panels that intrude into, narrow, or block the green path.
For traffic-light and stop-sign questions, the signal/sign does not need to overlap the green path, but it must face or control the ego lane, ego road, or green planned path.
Ignore traffic lights and stop signs that control cross traffic, side roads, parking lots, turn lanes not on the green path, or any other lane/path.
Ignore tiny distant cone rows, horizon dots, walls, lane paint, UI text, colored overlay lines/text, and decorative/off-path construction objects.
When the vehicle state says lead present yes, use distance, lateral offset, lead speed, relative speed, closing, acceleration, and lateral velocity as production lead-track evidence together with the image.
If lead present yes and absolute lateral offset is small, choose among true_moving_lead, slower_lead, braking_lead, or stopped_lead, not cut_in_vehicle, crossing_vehicle, irrelevant_vehicle, or none.
Use stopped_lead only when lead speed is near zero. Use braking_lead only when acceleration is clearly negative, brake lights are visible, or closing speed is rapid.
Use slower_lead when lead speed is above zero but relative speed is negative or closing is positive. Use true_moving_lead when relative speed is near zero or positive and spacing is stable or opening.
Use cut_in_vehicle or crossing_vehicle only when the vehicle is laterally outside the corridor and lateral velocity shows it is moving into/across the path.
Construction left/right means the side of the green path in the driver-view image, not simulator lane-coordinate sign.
If the relevant construction hazard intrudes from image-left / driver-left into the green path, answer yes only to construction_left.
If the relevant construction hazard intrudes from image-right / driver-right into the green path, answer yes only to construction_right.
Do not treat UI text, lane lines, route arrows, shadows, poles, or signs as pedestrians.
For red_stop_light, answer yes only for a red traffic light, red arrow, or red traffic-signal icon/marker that applies to the ego path.
For green_go_light, answer yes only for a green traffic light, green arrow, or green traffic-signal icon/marker that applies to the ego path and means proceed."""

LEAD_LABELS = {
  "true_moving_lead",
  "slower_lead",
  "braking_lead",
  "stopped_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
  "irrelevant_vehicle",
}
ACTIVE_LEAD_LABELS = {
  "slower_lead",
  "braking_lead",
  "stopped_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
}
PATH_CONFLICT_LABELS = {
  "pedestrian_in_path",
  "pedestrian_entering_path",
  "vehicle_in_path",
  "vehicle_entering_path",
  "animal_in_path",
  "animal_entering_path",
  "cut_in_vehicle",
  "crossing_vehicle",
}
PATH_BLOCKING_AGENT_LABELS = frozenset((
  "pedestrian_in_path",
  "vehicle_in_path",
  "animal_in_path",
))
PATH_ENTERING_AGENT_LABELS = frozenset((
  "pedestrian_entering_path",
  "vehicle_entering_path",
  "animal_entering_path",
  "cut_in_vehicle",
  "crossing_vehicle",
))
PATH_AGENT_RTP_VERSION = 2
PATH_AGENT_STOP_S = 18.0
PATH_AGENT_STOP_AVOID = "corridor_object_s18_28"
PATH_AGENT_YIELD_SPEED_CAP = "50%"
GREEN_SIGNAL_RTP_CONFIDENCE = 0.82
SCORE_LABELS = (
  "cones",
  "barrier",
  "construction_left",
  "construction_right",
  "construction_blue_edge",
  "construction_purple_edge",
  "construction_drive_left",
  "construction_drive_right",
  "construction_shift_left",
  "construction_shift_right",
  "construction_blocks_left_candidate",
  "construction_blocks_right_candidate",
  "pedestrian_in_path",
  "pedestrian_entering_path",
  "vehicle_in_path",
  "vehicle_entering_path",
  "true_moving_lead",
  "slower_lead",
  "braking_lead",
  "stopped_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
  "irrelevant_vehicle",
  "animal_in_path",
  "animal_entering_path",
  "red_stop_light",
  "green_go_light",
  "stop_sign",
)
DEFAULT_SCORE_LABEL_GROUPS = (
  ("red_stop_light", "green_go_light"),
  ("stop_sign", "green_go_light"),
  ("cones", "barrier"),
  ("construction_left", "construction_right"),
  ("construction_blue_edge", "construction_purple_edge"),
  ("pedestrian_in_path", "pedestrian_entering_path"),
  ("vehicle_in_path", "vehicle_entering_path"),
  ("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle"),
  ("animal_in_path", "animal_entering_path"),
)
DEFAULT_DURABLE_SCORE_LABELS = SCORE_LABELS
EXCLUSIVE_LABEL_GROUPS = (
  frozenset(("construction_left", "construction_right")),
  frozenset(("construction_blue_edge", "construction_purple_edge")),
  frozenset(("construction_drive_left", "construction_drive_right")),
  frozenset(("construction_shift_left", "construction_shift_right")),
  frozenset(("construction_blocks_left_candidate", "construction_blocks_right_candidate")),
  frozenset(("red_stop_light", "green_go_light")),
  frozenset(("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")),
)
CONSTRUCTION_PRESENCE_LABELS = frozenset(("cones", "barrier"))
CONSTRUCTION_SEMANTIC_SIDE_LABELS = frozenset(CONSTRUCTION_SIDE_LABELS)
CONSTRUCTION_EDGE_SIDE_LABELS = frozenset(CONSTRUCTION_EDGE_LABELS)
CONSTRUCTION_PRESENCE_DEPENDENT_SIDE_LABELS = CONSTRUCTION_SEMANTIC_SIDE_LABELS | CONSTRUCTION_EDGE_SIDE_LABELS
CONSTRUCTION_SELF_EVIDENCING_LABELS = frozenset(CONSTRUCTION_ACTION_LABELS) | frozenset(CONSTRUCTION_SHIFT_LABELS) | frozenset(CONSTRUCTION_CANDIDATE_LABELS)
CONSTRUCTION_CONTROL_LABELS = CONSTRUCTION_PRESENCE_DEPENDENT_SIDE_LABELS | CONSTRUCTION_SELF_EVIDENCING_LABELS
CONSTRUCTION_SIDE_BY_LABEL = {
  label: CONSTRUCTION_LABEL_TO_HAZARD_SIDE[label]
  for label in CONSTRUCTION_CONTROL_LABELS
}
CONSTRUCTION_SIDE_LOCK_CONFIRM_FRAMES = 3
CONSTRUCTION_SIDE_CONFLICT_CONFIRM_FRAMES = 3
CONSTRUCTION_PRESENCE_HOLD_FRAMES = 10
CONSTRUCTION_SIDE_IMMEDIATE_OVERRIDE_SCORE = 3.0
CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE = 2.3
CONSTRUCTION_SIDE_IMMEDIATE_LOCK_SCORE = 2.0
CONSTRUCTION_SIDE_IMMEDIATE_LOCK_MARGIN = 1.75
CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX = 0.05
CONSTRUCTION_EDGE_BOOTSTRAP_SCORE = 1.9
CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_MARGIN
CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX
CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE = 2.2
CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN = 0.8
CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE = 2.0
CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN = 0.5
CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES = 8
CONSTRUCTION_DIRECT_SEMANTIC_SCORE = 1.2
CONSTRUCTION_DIRECT_SEMANTIC_MARGIN = 0.5
CONSTRUCTION_DIRECT_EDGE_SCORE = 1.5
CONSTRUCTION_DIRECT_EDGE_MARGIN = 1.0
CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE = 2.5
CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN = 1.5
CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE = 2.2
CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN = 1.5
CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE = 2.3
CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_MARGIN
CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX
CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M = 0.12
CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES = 12
CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M = CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M
CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M = 0.25
CONSTRUCTION_ACTION_IMMEDIATE_SCORE = 2.5
CONSTRUCTION_ACTION_IMMEDIATE_MARGIN = 1.0
CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE = CONSTRUCTION_SIDE_IMMEDIATE_OVERRIDE_SCORE
CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_MARGIN
CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION = True
CONSTRUCTION_SIDE_CLEAR_MAX_SCORE = 0.05
CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M = 0.8
CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M = 0.35
CONSTRUCTION_PRESENCE_CLEAR_CONFIRM_FRAMES = 1
CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE = 2.0
CONSTRUCTION_CLEAR_RTP_CONFIDENCE = 0.74
CONSTRUCTION_STATE_MACHINE_VERSION = 31
CONSTRUCTION_LOCKED_SIDE_LABEL = {
  "left": "construction_blue_edge",
  "right": "construction_purple_edge",
}
EXCLUSIVE_LABEL_MIN_MARGIN = 0.05
SCORE_QUESTIONS = {
  "cones": "Are any traffic cones, blue cones, construction cones, pylons, bollards, or cone-shaped lane markers overlapping, intruding into, narrowing, or blocking the green planned path?",
  "barrier": "Is any road barrier, barricade, blue-white checker barrier, construction barrier, or blocked-lane panel overlapping, intruding into, narrowing, or blocking the green planned path?",
  "construction_left": "Ignoring traffic lights, signal icons, poles, lane lines, colored overlay lines/text, road-edge dotted markers, wall markers, and tiny orange/red horizon dots, is there a real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel intruding from image-left / driver-left into the green planned corridor?",
  "construction_right": "Ignoring traffic lights, signal icons, poles, lane lines, colored overlay lines/text, road-edge dotted markers, wall markers, and tiny orange/red horizon dots, is there a real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel intruding from image-right / driver-right into the green planned corridor?",
  "construction_blue_edge": "Ignoring traffic lights, signal icons, poles, lane lines, colored overlay text, road-edge dotted markers, wall markers, and tiny orange/red horizon dots, is the relevant real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel touching, overlapping, or narrowing the blue edge of the green planned corridor?",
  "construction_purple_edge": "Ignoring traffic lights, signal icons, poles, lane lines, colored overlay text, road-edge dotted markers, wall markers, and tiny orange/red horizon dots, is the relevant real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel touching, overlapping, or narrowing the purple edge of the green planned corridor?",
  "construction_drive_left": "Should the ego vehicle drive left, toward image-left / driver-left, to keep the green planned corridor clear of a construction cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel?",
  "construction_drive_right": "Should the ego vehicle drive right, toward image-right / driver-right, to keep the green planned corridor clear of a construction cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel?",
  "construction_shift_left": "Should the green planned path shift left within the lane to clear a construction cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel that affects the corridor?",
  "construction_shift_right": "Should the green planned path shift right within the lane to clear a construction cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel that affects the corridor?",
  "construction_blocks_left_candidate": "In the candidate-left auxiliary scene board, does a real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel touch, overlap, narrow, or block the green candidate corridor?",
  "construction_blocks_right_candidate": "In the candidate-right auxiliary scene board, does a real cone, barrel, pylon, bollard, barricade, barrier, or blocked-lane panel touch, overlap, narrow, or block the green candidate corridor?",
  "pedestrian_in_path": "Is any visible upright human figure, person, pedestrian, or human body partly or fully inside the green planned path or directly blocking the ego lane ahead, even if small or partially transparent?",
  "pedestrian_entering_path": "Is any visible upright human figure, person, pedestrian, or human body next to the green planned path and clearly crossing, walking, stepping, or moving into that green path soon?",
  "vehicle_in_path": "Is a car, truck, bicycle, motorcycle, or similar road user currently overlapping the green planned path or directly blocking the ego lane corridor ahead?",
  "vehicle_entering_path": "Is a car, truck, bicycle, motorcycle, or similar road user beside the green planned path and clearly moving into the ego lane corridor soon?",
  "true_moving_lead": "Is there a vehicle ahead in the green planned path moving near ego speed with stable spacing, so no extra slowdown or yield is needed?",
  "slower_lead": "Is there a vehicle ahead in the green planned path moving slower than ego speed or getting closer, requiring proportional slowdown but not a full stop?",
  "braking_lead": "Is there a lead vehicle ahead in the green planned path visibly braking, showing brake lights, or closing rapidly enough to require stronger slowdown?",
  "stopped_lead": "Is there a stopped or nearly stopped vehicle ahead blocking the green planned path, requiring stop, creep, or route response?",
  "cut_in_vehicle": "Is a vehicle from an adjacent lane visibly cutting into or about to enter the green planned path ahead, requiring yield or slowdown?",
  "crossing_vehicle": "Is a vehicle crossing the green planned path from the side with a conflict risk, requiring yield or stop?",
  "irrelevant_vehicle": "Are visible vehicles clearly outside the green planned path or moving away without affecting ego path, so no slowdown or yield is needed?",
  "animal_in_path": "Is an animal currently overlapping the green planned path or directly blocking the ego lane corridor ahead?",
  "animal_entering_path": "Is an animal beside the green planned path and clearly moving into the ego lane corridor soon?",
  "red_stop_light": "Is a red traffic light, red arrow, or red traffic-signal icon/marker visible anywhere in the image that faces or controls the ego lane, ego road, or green planned path?",
  "green_go_light": "Is a green traffic light, green arrow, or green traffic-signal icon/marker visible anywhere in the image that faces or controls the ego lane, ego road, or green planned path and authorizes proceeding?",
  "stop_sign": "Is a STOP sign visible anywhere in the image that faces or controls the ego lane, ego road, or green planned path?",
}


class RotatingScoreState:
  def __init__(
    self,
    groups: Sequence[Sequence[str]],
    cache_ttl_frames: int,
    durable_labels: Sequence[str] = DEFAULT_DURABLE_SCORE_LABELS,
    negative_clear_threshold: float = 2.0,
  ):
    self.groups = tuple(tuple(group) for group in groups)
    self.cache_ttl_frames = max(0, cache_ttl_frames)
    self._construction_presence_ttl_frames = max(self.cache_ttl_frames, CONSTRUCTION_PRESENCE_HOLD_FRAMES)
    self.durable_labels = frozenset(durable_labels)
    self.negative_clear_threshold = max(0.0, negative_clear_threshold)
    self.next_group_idx = 0
    self._positive_frame: dict[str, int] = {}
    self._scores: dict[str, float] = {}
    self._score_frame: dict[str, int] = {}
    self._construction_presence_anchor_frame: int | None = None
    group_sets = tuple(frozenset(group) for group in self.groups)
    self._construction_side_consensus_required = (
      any(group & CONSTRUCTION_SEMANTIC_SIDE_LABELS for group in group_sets)
      and any(group & CONSTRUCTION_EDGE_SIDE_LABELS for group in group_sets)
    )
    self._construction_locked_side: str | None = None
    self._construction_locked_frame: int | None = None
    self._construction_lock_candidate_side: str | None = None
    self._construction_lock_candidate_count = 0
    self._construction_lock_candidate_frame: int | None = None
    self._construction_pending_side: str | None = None
    self._construction_pending_count = 0
    self._construction_cleared_side: str | None = None
    self._construction_clear_candidate_count = 0
    self._construction_clear_candidate_frame: int | None = None
    self._construction_lateral_clear_frame: int | None = None
    self._last_vehicle_state_text = ""

  def next_group(self) -> tuple[int, tuple[str, ...]]:
    if not self.groups:
      return 0, ()
    idx = self.next_group_idx
    self.next_group_idx = (self.next_group_idx + 1) % len(self.groups)
    return idx, self.groups[idx]

  def update(
    self,
    group: Sequence[str],
    labels: Sequence[str],
    scores: dict[str, float],
    frame_id: int,
    vehicle_state_text: str = "",
  ) -> tuple[str, ...]:
    self._last_vehicle_state_text = vehicle_state_text
    label_set = _resolve_exclusive_labels(set(labels), scores)
    if label_set & CONSTRUCTION_PRESENCE_LABELS and not self._has_active_construction_presence(frame_id):
      self._construction_presence_anchor_frame = frame_id
      if not self._construction_lock_survives_presence_reacquire(frame_id):
        self._construction_locked_side = None
        self._reset_construction_lock_candidate()
        self._reset_construction_pending_side()
      self._construction_cleared_side = None
    label_set = self._construction_side_locked_labels(label_set, scores, frame_id)
    for label in group:
      if label in scores:
        self._scores[label] = scores[label]
        self._score_frame[label] = frame_id
    for label in group:
      if label in label_set:
        self._clear_exclusive_conflicts(label)
        self._positive_frame[label] = frame_id
      else:
        self._clear_or_hold_negative(label, scores, frame_id)
    self._apply_construction_presence_clear_if_observed(group, label_set, scores, frame_id)
    self._apply_construction_side_clear_if_observed(group, label_set, scores)
    if not self._has_active_construction_presence(frame_id):
      self._construction_presence_anchor_frame = None
      self._construction_locked_side = None
      self._construction_locked_frame = None
      self._construction_cleared_side = None
      self._reset_construction_lock_candidate()
      self._reset_construction_pending_side()
      self._reset_construction_clear_candidate()
    return self.active_labels(frame_id)

  def _clear_exclusive_conflicts(self, label: str) -> None:
    for group in EXCLUSIVE_LABEL_GROUPS:
      if label not in group:
        continue
      for other in group:
        if other != label:
          self._positive_frame.pop(other, None)
      return

  def _clear_or_hold_negative(self, label: str, scores: dict[str, float], frame_id: int) -> None:
    if label in self.durable_labels:
      last_frame = self._positive_frame.get(label)
      score = scores.get(label)
      if (
        last_frame is not None and
        frame_id - last_frame <= self._positive_ttl_frames(label) and
        score is not None and
        score > -self.negative_clear_threshold
      ):
        # Positive scene evidence should survive a brief occlusion or splash.
        # It still expires by TTL, or clears on a strong negative score.
        return
    self._positive_frame.pop(label, None)

  def active_labels(self, frame_id: int) -> tuple[str, ...]:
    active = []
    for label in SCORE_LABELS:
      last_frame = self._positive_frame.get(label)
      if last_frame is None:
        continue
      if frame_id - last_frame <= self._positive_ttl_frames(label):
        active.append(label)
      else:
        self._positive_frame.pop(label, None)
    self._expire_old_scores(frame_id)
    active = list(_resolve_exclusive_labels(set(active), self._scores))
    active.sort(key=lambda label: SCORE_LABELS.index(label) if label in SCORE_LABELS else len(SCORE_LABELS))
    active = self._construction_presence_gated(active, frame_id)
    return tuple(active) if active else ("none",)

  def active_scores(self, frame_id: int) -> dict[str, float]:
    active = set(self.active_labels(frame_id))
    score_labels = set(active)
    for group in EXCLUSIVE_LABEL_GROUPS:
      if active & group:
        score_labels.update(group)
    self._expire_old_scores(frame_id)
    return {label: score for label, score in self._scores.items() if label in score_labels}

  def _expire_old_scores(self, frame_id: int) -> None:
    for label, last_frame in list(self._score_frame.items()):
      if frame_id - last_frame <= self._score_ttl_frames(label):
        continue
      self._score_frame.pop(label, None)
      self._scores.pop(label, None)

  def _positive_ttl_frames(self, label: str) -> int:
    if label in CONSTRUCTION_PRESENCE_LABELS:
      return self._construction_presence_ttl_frames
    if label in CONSTRUCTION_ACTION_LABELS:
      return max(self.cache_ttl_frames, CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES)
    return self.cache_ttl_frames

  def _score_ttl_frames(self, label: str) -> int:
    if label in CONSTRUCTION_PRESENCE_LABELS:
      return self._construction_presence_ttl_frames
    if label in (CONSTRUCTION_SEMANTIC_SIDE_LABELS | CONSTRUCTION_EDGE_SIDE_LABELS | frozenset(CONSTRUCTION_ACTION_LABELS)):
      return max(self.cache_ttl_frames, CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES)
    return self.cache_ttl_frames

  def _has_active_construction_presence(self, frame_id: int) -> bool:
    for label in CONSTRUCTION_PRESENCE_LABELS:
      last_frame = self._positive_frame.get(label)
      if last_frame is not None and frame_id - last_frame <= self._construction_presence_ttl_frames:
        return True
    return False

  def _construction_presence_gated(self, labels: list[str], frame_id: int) -> list[str]:
    presence_active = self._has_active_construction_presence(frame_id)
    if not presence_active:
      self._construction_presence_anchor_frame = None
    anchor_frame = self._construction_presence_anchor_frame
    direct_consensus_labels = self._construction_recent_direct_consensus_labels(frame_id, anchor_frame) if presence_active else ()
    direct_consensus_side = self._construction_label_side(set(direct_consensus_labels))
    eligible_side_labels = []
    if presence_active and anchor_frame is not None:
      for label in labels:
        if label not in CONSTRUCTION_PRESENCE_DEPENDENT_SIDE_LABELS:
          continue
        side_frame = self._positive_frame.get(label)
        if side_frame is not None and side_frame >= anchor_frame:
          eligible_side_labels.append(label)
    consensus_side = self._construction_side_consensus(eligible_side_labels) if self._construction_side_consensus_required else None
    gated = []
    for label in labels:
      if self._construction_control_suppressed_after_clear(label, frame_id):
        continue
      if label in CONSTRUCTION_SELF_EVIDENCING_LABELS:
        if label in CONSTRUCTION_ACTION_LABELS and not self._construction_action_allowed(label, direct_consensus_side, frame_id):
          self._positive_frame.pop(label, None)
          continue
        if direct_consensus_side is not None and CONSTRUCTION_SIDE_BY_LABEL.get(label) != direct_consensus_side:
          self._positive_frame.pop(label, None)
          continue
        gated.append(label)
        continue
      if label not in CONSTRUCTION_PRESENCE_DEPENDENT_SIDE_LABELS:
        gated.append(label)
        continue
      side_frame = self._positive_frame.get(label)
      recent_action_side = self._construction_recent_action_side(frame_id)
      label_side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
      if recent_action_side is not None and label_side is not None and label_side != recent_action_side and direct_consensus_side != label_side:
        continue
      if self._construction_edge_bootstrap_allowed(label, side_frame, frame_id):
        gated.append(label)
        self._observe_construction_side_for_lock(label, side_frame if side_frame is not None else frame_id)
        continue
      if self._construction_edge_toward_path_override_allowed(label, side_frame, frame_id):
        gated.append(label)
        self._observe_construction_side_for_lock(label, side_frame if side_frame is not None else frame_id)
        continue
      if self._construction_semantic_neutral_bootstrap_allowed(label, side_frame, frame_id):
        gated.append(label)
        self._observe_construction_side_for_lock(label, side_frame if side_frame is not None else frame_id)
        continue
      if self._construction_semantic_toward_path_override_allowed(label, side_frame, frame_id):
        gated.append(label)
        self._observe_construction_side_for_lock(label, side_frame if side_frame is not None else frame_id)
        continue
      if not (presence_active and anchor_frame is not None and side_frame is not None and side_frame >= anchor_frame):
        continue
      if self._construction_side_consensus_required and CONSTRUCTION_SIDE_BY_LABEL[label] != consensus_side:
        continue
      if self._construction_side_suppressed_after_clear(label):
        continue
      gated.append(label)
      if label in CONSTRUCTION_SIDE_BY_LABEL:
        self._observe_construction_side_for_lock(label, side_frame)
    for label in direct_consensus_labels:
      if self._construction_control_suppressed_after_clear(label, frame_id):
        continue
      if self._construction_side_suppressed_after_clear(label):
        continue
      if label not in gated:
        gated.append(label)
      side_frame = self._score_frame.get(label)
      self._observe_construction_side_for_lock(label, side_frame if side_frame is not None else frame_id)
    if direct_consensus_side is not None:
      for label in tuple(CONSTRUCTION_SELF_EVIDENCING_LABELS):
        if CONSTRUCTION_SIDE_BY_LABEL.get(label) != direct_consensus_side:
          self._positive_frame.pop(label, None)
    if presence_active and anchor_frame is not None and self._construction_locked_side is not None:
      if not any(label in CONSTRUCTION_PRESENCE_DEPENDENT_SIDE_LABELS for label in gated):
        locked_label = self._locked_construction_side_label()
        if locked_label is not None:
          gated.append(locked_label)
    gated = list(dict.fromkeys(gated))
    gated.sort(key=lambda label: SCORE_LABELS.index(label) if label in SCORE_LABELS else len(SCORE_LABELS))
    return gated

  def _locked_construction_side_label(self) -> str | None:
    if self._construction_locked_side is None:
      return None
    label = CONSTRUCTION_LOCKED_SIDE_LABEL.get(self._construction_locked_side)
    if label is None:
      return None
    if any(label in group for group in self.groups):
      return label
    return None

  def _recent_unclosed_construction_lock(self, frame_id: int) -> bool:
    if self._construction_lateral_clear_frame is not None:
      return False
    if self._construction_locked_side is None:
      return False
    max_age_frames = max(self.cache_ttl_frames, CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES)
    for label in CONSTRUCTION_CONTROL_LABELS:
      if CONSTRUCTION_SIDE_BY_LABEL.get(label) != self._construction_locked_side:
        continue
      side_frame = self._positive_frame.get(label)
      if side_frame is not None and frame_id - side_frame <= max_age_frames:
        return True
    return False

  def _construction_lock_survives_presence_reacquire(self, frame_id: int) -> bool:
    if self._construction_locked_side is None:
      return False
    if self._recent_unclosed_construction_lock(frame_id):
      return True
    if self._construction_locked_frame is not None and frame_id - self._construction_locked_frame <= CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES:
      return True
    return self._tracked_path_is_away_from_construction_side_with_threshold(
      self._construction_locked_side,
      CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M,
    )

  def _reset_construction_lock_candidate(self) -> None:
    self._construction_lock_candidate_side = None
    self._construction_lock_candidate_count = 0
    self._construction_lock_candidate_frame = None

  def _reset_construction_pending_side(self) -> None:
    self._construction_pending_side = None
    self._construction_pending_count = 0

  def _reset_construction_clear_candidate(self) -> None:
    self._construction_clear_candidate_count = 0
    self._construction_clear_candidate_frame = None

  def construction_clear_active(self, frame_id: int) -> bool:
    return (
      self._construction_lateral_clear_frame is not None and
      frame_id - self._construction_lateral_clear_frame <= self.cache_ttl_frames
    )

  def debug_state(self, frame_id: int) -> dict[str, object]:
    return {
      "construction_presence_active": self._has_active_construction_presence(frame_id),
      "construction_presence_anchor_frame": self._construction_presence_anchor_frame,
      "construction_locked_side": self._construction_locked_side,
      "construction_locked_frame": self._construction_locked_frame,
      "construction_lock_candidate_side": self._construction_lock_candidate_side,
      "construction_lock_candidate_count": self._construction_lock_candidate_count,
      "construction_pending_side": self._construction_pending_side,
      "construction_pending_count": self._construction_pending_count,
      "construction_cleared_side": self._construction_cleared_side,
      "construction_clear_candidate_count": self._construction_clear_candidate_count,
      "construction_lateral_clear_frame": self._construction_lateral_clear_frame,
      "construction_direct_consensus_labels": list(self._construction_recent_direct_consensus_labels(frame_id, self._construction_presence_anchor_frame)),
      "last_vehicle_state_text": self._last_vehicle_state_text,
    }

  def _construction_recent_direct_consensus_labels(self, frame_id: int, min_frame: int | None = None) -> tuple[str, ...]:
    semantic = self._construction_recent_family_best(
      ("construction_left", "construction_right"),
      frame_id,
      CONSTRUCTION_DIRECT_SEMANTIC_SCORE,
      CONSTRUCTION_DIRECT_SEMANTIC_MARGIN,
      min_frame,
    )
    edge = self._construction_recent_family_best(
      ("construction_blue_edge", "construction_purple_edge"),
      frame_id,
      CONSTRUCTION_DIRECT_EDGE_SCORE,
      CONSTRUCTION_DIRECT_EDGE_MARGIN,
      min_frame,
    )
    if semantic is None or edge is None:
      return ()
    semantic_label, semantic_side = semantic
    edge_label, edge_side = edge
    if semantic_side != edge_side:
      return ()
    return (semantic_label, edge_label)

  def _construction_recent_family_best(
    self,
    labels: tuple[str, str],
    frame_id: int,
    min_score: float,
    min_margin: float,
    min_frame: int | None = None,
  ) -> tuple[str, str] | None:
    scored: list[tuple[float, str, int]] = []
    for label in labels:
      score = self._scores.get(label)
      score_frame = self._score_frame.get(label)
      if score is None or score_frame is None:
        return None
      if min_frame is not None and score_frame < min_frame:
        return None
      if frame_id - score_frame > CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES:
        return None
      score_f = float(score)
      if not math.isfinite(score_f):
        return None
      scored.append((score_f, label, score_frame))
    scored.sort(reverse=True)
    best_score, best_label, _best_frame = scored[0]
    runner_up_score = scored[1][0]
    if best_score < min_score or best_score - runner_up_score < min_margin:
      return None
    side = CONSTRUCTION_SIDE_BY_LABEL.get(best_label)
    if side is None:
      return None
    return best_label, side

  def _apply_construction_presence_clear_if_observed(self, group: Sequence[str], labels: set[str], scores: dict[str, float], frame_id: int) -> None:
    group_set = frozenset(group)
    if not group_set or not group_set.issubset(CONSTRUCTION_PRESENCE_LABELS):
      return
    positive_presence_labels = self._positive_scored_labels(labels, scores, CONSTRUCTION_PRESENCE_LABELS)
    if positive_presence_labels:
      self._reset_construction_clear_candidate()
      return
    if not self._tracked_path_is_shifted_for_construction_clear():
      self._reset_construction_clear_candidate()
      return
    presence_scores = [float(scores.get(label, float("inf"))) for label in group_set]
    if not presence_scores or any(not math.isfinite(score) for score in presence_scores):
      self._reset_construction_clear_candidate()
      return
    if max(presence_scores) > CONSTRUCTION_SIDE_CLEAR_MAX_SCORE:
      self._reset_construction_clear_candidate()
      return
    if self._construction_clear_candidate_frame != frame_id:
      self._construction_clear_candidate_count += 1
      self._construction_clear_candidate_frame = frame_id
    if self._construction_clear_candidate_count < CONSTRUCTION_PRESENCE_CLEAR_CONFIRM_FRAMES:
      return
    self._clear_all_construction_state(frame_id)

  def _apply_construction_side_clear_if_observed(self, group: Sequence[str], labels: set[str], scores: dict[str, float]) -> None:
    if self._construction_locked_side is None:
      return
    group_set = frozenset(group)
    if not group_set or not group_set.issubset(CONSTRUCTION_EDGE_SIDE_LABELS):
      return
    positive_edge_labels = self._positive_scored_labels(labels, scores, CONSTRUCTION_EDGE_SIDE_LABELS)
    if positive_edge_labels:
      return
    if not self._tracked_path_is_away_from_construction_side(self._construction_locked_side):
      return
    edge_scores = [float(scores.get(label, float("inf"))) for label in group_set]
    if not edge_scores or any(not math.isfinite(score) for score in edge_scores):
      return
    if max(edge_scores) > CONSTRUCTION_SIDE_CLEAR_MAX_SCORE:
      return
    cleared_side = self._construction_locked_side
    self._clear_construction_sides(cleared_side)
    for label in tuple(CONSTRUCTION_CONTROL_LABELS):
      if CONSTRUCTION_SIDE_BY_LABEL.get(label) == cleared_side:
        self._positive_frame.pop(label, None)

  def _construction_side_suppressed_after_clear(self, label: str) -> bool:
    side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
    if side is None or side != self._construction_cleared_side:
      return False
    if not self._tracked_path_is_away_from_construction_side(side):
      self._construction_cleared_side = None
      return False
    return True

  def _construction_control_suppressed_after_clear(self, label: str, frame_id: int) -> bool:
    if label not in CONSTRUCTION_CONTROL_LABELS:
      return False
    if self._construction_lateral_clear_frame is None:
      return False
    if not self._tracked_path_is_shifted_for_construction_clear():
      self._construction_lateral_clear_frame = None
      return False
    if self._fresh_construction_presence_after_clear(frame_id):
      self._construction_lateral_clear_frame = None
      self._construction_cleared_side = None
      return False
    return True

  def _fresh_construction_presence_after_clear(self, frame_id: int) -> bool:
    if self._construction_lateral_clear_frame is None:
      return False
    for label in CONSTRUCTION_PRESENCE_LABELS:
      score_frame = self._score_frame.get(label)
      positive_frame = self._positive_frame.get(label)
      if score_frame is None or positive_frame is None:
        continue
      if score_frame <= self._construction_lateral_clear_frame or positive_frame <= self._construction_lateral_clear_frame:
        continue
      if frame_id - score_frame > self.cache_ttl_frames:
        continue
      score = self._scores.get(label)
      if score is None:
        continue
      score_f = float(score)
      if math.isfinite(score_f) and score_f >= CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE:
        return True
    return False

  def _construction_edge_bootstrap_allowed(self, label: str, side_frame: int | None, frame_id: int) -> bool:
    if label not in CONSTRUCTION_EDGE_SIDE_LABELS:
      return False
    if self._construction_side_consensus_required:
      if self._has_active_construction_presence(frame_id):
        return False
      if not self._tracked_path_is_near_neutral_for_construction_reversal():
        return False
      if side_frame is None or frame_id - side_frame > self.cache_ttl_frames:
        return False
      same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
      return (
        math.isfinite(same_score) and
        math.isfinite(margin) and
        same_score >= CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE and
        margin >= CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN
      )
    if self._has_active_construction_presence(frame_id):
      return False
    if side_frame is None or frame_id - side_frame > self.cache_ttl_frames:
      return False
    same_score, opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      math.isfinite(same_score) and
      math.isfinite(opposite_score) and
      same_score >= CONSTRUCTION_EDGE_BOOTSTRAP_SCORE and
      opposite_score <= CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX and
      margin >= CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN
    )

  def _construction_edge_toward_path_override_allowed(self, label: str, side_frame: int | None, frame_id: int) -> bool:
    if label not in CONSTRUCTION_EDGE_SIDE_LABELS:
      return False
    if side_frame is None or frame_id - side_frame > self.cache_ttl_frames:
      return False
    side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
    if side is None or not self._tracked_path_is_toward_construction_side(side):
      return False
    same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      math.isfinite(same_score) and
      math.isfinite(margin) and
      same_score >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE and
      margin >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN
    )

  def _construction_semantic_toward_path_override_allowed(self, label: str, side_frame: int | None, frame_id: int) -> bool:
    if label not in CONSTRUCTION_SEMANTIC_SIDE_LABELS:
      return False
    if side_frame is None or frame_id - side_frame > self.cache_ttl_frames:
      return False
    side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
    if side is None or not self._tracked_path_is_toward_construction_side(side):
      return False
    same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      math.isfinite(same_score) and
      math.isfinite(margin) and
      same_score >= CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE and
      margin >= CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN
    )

  def _construction_semantic_neutral_bootstrap_allowed(self, label: str, side_frame: int | None, frame_id: int) -> bool:
    if label not in CONSTRUCTION_SEMANTIC_SIDE_LABELS:
      return False
    if not self._has_active_construction_presence(frame_id):
      return False
    if not self._tracked_path_is_near_neutral_for_construction_reversal():
      return False
    if side_frame is None or frame_id - side_frame > self.cache_ttl_frames:
      return False
    same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      math.isfinite(same_score) and
      math.isfinite(margin) and
      same_score >= CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE and
      margin >= CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN
    )

  def _construction_action_allowed(self, label: str, direct_consensus_side: str | None, frame_id: int) -> bool:
    side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
    if side is None:
      return False
    if not self._construction_action_fresh_or_strong(label, frame_id):
      return False
    if direct_consensus_side is not None:
      return side == direct_consensus_side
    if self._construction_locked_side is not None and side == self._construction_locked_side:
      return True
    if self._construction_locked_side is not None and side != self._construction_locked_side:
      same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
      return (
        self._tracked_path_is_toward_construction_side(side) and
        math.isfinite(same_score) and
        math.isfinite(margin) and
        same_score >= CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE and
        margin >= CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN
      )
    if self._tracked_path_is_near_neutral_for_construction_action():
      return self._construction_action_score_strong_enough(label)
    if self._tracked_path_is_supporting_construction_action(side):
      return self._construction_action_score_strong_enough(label)
    same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      self._tracked_path_is_toward_construction_side(side) and
      math.isfinite(same_score) and
      math.isfinite(margin) and
      same_score >= CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE and
      margin >= CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN
    )

  def _construction_action_score_strong_enough(self, label: str) -> bool:
    same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
    return (
      math.isfinite(same_score) and
      math.isfinite(margin) and
      same_score >= CONSTRUCTION_ACTION_IMMEDIATE_SCORE and
      margin >= CONSTRUCTION_ACTION_IMMEDIATE_MARGIN
    )

  def _construction_action_fresh_or_strong(self, label: str, frame_id: int) -> bool:
    positive_frame = self._positive_frame.get(label)
    if positive_frame is None or frame_id - positive_frame <= self.cache_ttl_frames:
      return True
    return self._construction_action_score_strong_enough(label)

  def _tracked_path_is_supporting_construction_action(self, side: str) -> bool:
    return self._tracked_path_is_away_from_construction_side_with_threshold(
      side,
      CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M,
    )

  def _tracked_path_is_away_from_construction_side(self, side: str) -> bool:
    return self._tracked_path_is_away_from_construction_side_with_threshold(
      side,
      CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M,
    )

  def _tracked_path_is_committed_away_from_construction_side(self, side: str) -> bool:
    return self._tracked_path_is_away_from_construction_side_with_threshold(
      side,
      CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M,
    )

  def _tracked_path_is_toward_construction_side(self, side: str) -> bool:
    match = re.search(r"\btracked_path_lat=(-?\d+(?:\.\d+)?)m\b", self._last_vehicle_state_text)
    if match is None:
      return False
    tracked_path_lat_m = float(match.group(1))
    if abs(tracked_path_lat_m) < CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M:
      return False
    avoid_token = "right_edge_s0_1" if side == "right" else "left_edge_s0_1"
    return not construction_avoidance_openpilot_side_valid(avoid_token, tracked_path_lat_m)

  def _tracked_path_is_near_neutral_for_construction_reversal(self) -> bool:
    match = re.search(r"\btracked_path_lat=(-?\d+(?:\.\d+)?)m\b", self._last_vehicle_state_text)
    if match is None:
      return False
    return abs(float(match.group(1))) <= CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M

  def _tracked_path_is_near_neutral_for_construction_action(self) -> bool:
    match = re.search(r"\btracked_path_lat=(-?\d+(?:\.\d+)?)m\b", self._last_vehicle_state_text)
    if match is None:
      return True
    return abs(float(match.group(1))) <= CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M

  def _tracked_path_is_away_from_construction_side_with_threshold(self, side: str, min_offset_m: float) -> bool:
    match = re.search(r"\btracked_path_lat=(-?\d+(?:\.\d+)?)m\b", self._last_vehicle_state_text)
    if match is None:
      return False
    tracked_path_lat_m = float(match.group(1))
    if abs(tracked_path_lat_m) < min_offset_m:
      return False
    avoid_token = "right_edge_s0_1" if side == "right" else "left_edge_s0_1"
    return construction_avoidance_openpilot_side_valid(avoid_token, tracked_path_lat_m)

  def _tracked_path_is_shifted_for_construction_clear(self) -> bool:
    match = re.search(r"\btracked_path_lat=(-?\d+(?:\.\d+)?)m\b", self._last_vehicle_state_text)
    if match is None:
      return False
    return abs(float(match.group(1))) >= CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M

  def _clear_construction_sides(self, cleared_side: str | None) -> None:
    self._construction_locked_side = None
    self._construction_locked_frame = None
    self._construction_cleared_side = cleared_side
    self._reset_construction_lock_candidate()
    self._reset_construction_pending_side()
    self._reset_construction_clear_candidate()

  def _clear_all_construction_state(self, frame_id: int) -> None:
    self._clear_construction_sides(self._construction_locked_side)
    self._construction_presence_anchor_frame = None
    self._construction_lateral_clear_frame = frame_id
    for label in tuple(CONSTRUCTION_PRESENCE_LABELS | CONSTRUCTION_CONTROL_LABELS):
      self._positive_frame.pop(label, None)

  @staticmethod
  def _positive_scored_labels(labels: set[str], scores: dict[str, float], candidates: frozenset[str]) -> set[str]:
    positives = set()
    for label in labels & candidates:
      score = scores.get(label)
      if score is None:
        positives.add(label)
        continue
      score_f = float(score)
      if math.isfinite(score_f) and score_f > CONSTRUCTION_SIDE_CLEAR_MAX_SCORE:
        positives.add(label)
    return positives

  def _observe_construction_side_for_lock(self, label: str, side_frame: int) -> None:
    if self._construction_locked_side is not None:
      return
    side = CONSTRUCTION_SIDE_BY_LABEL[label]
    same_score, opposite_score, margin = self._construction_label_family_score_stats(label)
    immediate_score = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_SCORE
    immediate_margin = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_MARGIN
    immediate_opposite_max = CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX
    if label in CONSTRUCTION_EDGE_SIDE_LABELS:
      immediate_score = CONSTRUCTION_EDGE_BOOTSTRAP_SCORE
      immediate_margin = CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN
      immediate_opposite_max = CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX
    if (
      math.isfinite(same_score) and
      math.isfinite(opposite_score) and
      same_score >= immediate_score and
      opposite_score <= immediate_opposite_max and
      margin >= immediate_margin
    ):
      self._construction_locked_side = side
      self._construction_locked_frame = side_frame
      self._reset_construction_lock_candidate()
      self._reset_construction_pending_side()
      return
    if self._construction_lock_candidate_side == side:
      if self._construction_lock_candidate_frame != side_frame:
        self._construction_lock_candidate_count += 1
        self._construction_lock_candidate_frame = side_frame
    else:
      self._construction_lock_candidate_side = side
      self._construction_lock_candidate_count = 1
      self._construction_lock_candidate_frame = side_frame
    if self._construction_lock_candidate_count >= CONSTRUCTION_SIDE_LOCK_CONFIRM_FRAMES:
      self._construction_locked_side = side
      self._construction_locked_frame = side_frame
      self._reset_construction_lock_candidate()
      self._reset_construction_pending_side()

  def _construction_label_family_score_stats(self, label: str) -> tuple[float, float, float]:
    side = CONSTRUCTION_SIDE_BY_LABEL[label]
    group = next((group for group in EXCLUSIVE_LABEL_GROUPS if label in group), frozenset((label,)))
    same_scores = []
    opposite_scores = []
    for candidate in group:
      candidate_side = CONSTRUCTION_SIDE_BY_LABEL.get(candidate)
      if candidate_side is None:
        continue
      score = self._scores.get(candidate)
      if score is None:
        continue
      if candidate_side == side:
        same_scores.append(float(score))
      else:
        opposite_scores.append(float(score))
    same_score = max(same_scores, default=float("-inf"))
    opposite_score = max(opposite_scores, default=float("-inf"))
    return same_score, opposite_score, same_score - opposite_score

  @staticmethod
  def _construction_label_side(labels: set[str]) -> str | None:
    side = construction_hazard_side_from_labels(labels)
    return side if side != "unknown" else None

  def _construction_side_locked_labels(self, labels: set[str], scores: dict[str, float], frame_id: int) -> set[str]:
    if self._construction_locked_side is None:
      return labels
    requested_side = self._construction_label_side(labels)
    if requested_side is None or requested_side == self._construction_locked_side:
      self._reset_construction_pending_side()
      return labels

    conflict_scores = [
      float(scores.get(label, float("-inf")))
      for label in labels
      if CONSTRUCTION_SIDE_BY_LABEL.get(label) == requested_side
    ]
    conflict_score = max(conflict_scores, default=float("-inf"))
    locked_side_scores = [
      float(scores.get(label, float("-inf")))
      for label in scores
      if CONSTRUCTION_SIDE_BY_LABEL.get(label) == self._construction_locked_side
    ]
    locked_side_score = max(locked_side_scores, default=float("-inf"))
    conflict_margin = conflict_score - locked_side_score
    early_reversal_age = None if self._construction_locked_frame is None else frame_id - self._construction_locked_frame
    early_reversal_allowed = (
      early_reversal_age is not None and
      0 <= early_reversal_age <= CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES and
      self._tracked_path_is_near_neutral_for_construction_reversal() and
      conflict_score >= CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE and
      locked_side_score <= CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX and
      conflict_margin >= CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN
    )
    if early_reversal_allowed:
      self._construction_locked_side = requested_side
      self._construction_locked_frame = frame_id
      self._reset_construction_pending_side()
      return labels

    locked_side_already_committed_away = self._tracked_path_is_committed_away_from_construction_side(self._construction_locked_side)
    tracked_path_toward_requested_side = self._tracked_path_is_toward_construction_side(requested_side)
    if (
      CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION and
      locked_side_already_committed_away and
      not (labels & set(CONSTRUCTION_ACTION_LABELS)) and
      self._construction_recent_action_side(frame_id) != requested_side
    ):
      return {
        label for label in labels
        if CONSTRUCTION_SIDE_BY_LABEL.get(label, self._construction_locked_side) == self._construction_locked_side
      }
    if conflict_score >= CONSTRUCTION_SIDE_IMMEDIATE_OVERRIDE_SCORE and (not locked_side_already_committed_away or tracked_path_toward_requested_side):
      self._construction_locked_side = requested_side
      self._construction_locked_frame = frame_id
      self._reset_construction_pending_side()
      return labels
    if conflict_score >= CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE and tracked_path_toward_requested_side:
      self._construction_locked_side = requested_side
      self._construction_locked_frame = frame_id
      self._reset_construction_pending_side()
      return labels

    if self._construction_pending_side == requested_side:
      self._construction_pending_count += 1
    else:
      self._construction_pending_side = requested_side
      self._construction_pending_count = 1

    if self._construction_pending_count >= CONSTRUCTION_SIDE_CONFLICT_CONFIRM_FRAMES:
      self._construction_locked_side = requested_side
      self._construction_locked_frame = frame_id
      self._reset_construction_pending_side()
      return labels

    return {
      label for label in labels
      if CONSTRUCTION_SIDE_BY_LABEL.get(label, self._construction_locked_side) == self._construction_locked_side
    }

  @staticmethod
  def _family_side(labels: set[str], left_label: str, right_label: str) -> str | None:
    side = construction_hazard_side_from_labels((label for label in (left_label, right_label) if label in labels))
    return side if side != "unknown" else None

  def _construction_recent_action_side(self, frame_id: int) -> str | None:
    sides = set()
    for label in CONSTRUCTION_ACTION_LABELS:
      positive_frame = self._positive_frame.get(label)
      if positive_frame is None or frame_id - positive_frame > CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES:
        continue
      score = self._scores.get(label)
      if score is not None:
        score_f = float(score)
        if not math.isfinite(score_f) or score_f <= CONSTRUCTION_SIDE_CLEAR_MAX_SCORE:
          continue
      same_score, _opposite_score, margin = self._construction_label_family_score_stats(label)
      if (
        not math.isfinite(same_score) or
        not math.isfinite(margin) or
        same_score < CONSTRUCTION_ACTION_IMMEDIATE_SCORE or
        margin < CONSTRUCTION_ACTION_IMMEDIATE_MARGIN
      ):
        continue
      side = CONSTRUCTION_SIDE_BY_LABEL.get(label)
      if side is not None:
        sides.add(side)
    return next(iter(sides)) if len(sides) == 1 else None

  @classmethod
  def _construction_side_consensus(cls, labels: list[str]) -> str | None:
    label_set = set(labels)
    semantic_side = cls._family_side(label_set, "construction_left", "construction_right")
    edge_side = cls._family_side(label_set, "construction_blue_edge", "construction_purple_edge")
    if semantic_side is not None and edge_side is not None and semantic_side == edge_side:
      return semantic_side
    return None


def _resolve_exclusive_labels(labels: set[str], scores: dict[str, float]) -> set[str]:
  resolved = set(labels)
  for group in EXCLUSIVE_LABEL_GROUPS:
    active = resolved & group
    if len(active) <= 1:
      continue
    ordered = sorted(active, key=lambda label: scores.get(label, float("-inf")), reverse=True)
    resolved.difference_update(group)
    best = ordered[0]
    runner_up = ordered[1]
    if scores.get(best, float("-inf")) - scores.get(runner_up, float("-inf")) >= EXCLUSIVE_LABEL_MIN_MARGIN:
      resolved.add(best)
  return resolved


def _load(model_dir: Path):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
  )
  model.eval()
  return processor, model


def _image_from_payload(payload: dict) -> Image.Image:
  data = base64.b64decode(payload["scene_board_image_b64"])
  return Image.open(BytesIO(data)).convert("RGB")


def _thumb(image: Image.Image, max_side: int) -> Image.Image:
  out = image.copy()
  out.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
  return out


def _crop_resize(image: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
  return image.crop(box).resize((size, size), Image.Resampling.BICUBIC)


def _probe_images(image: Image.Image, size: int) -> list[Image.Image]:
  w, h = image.size
  x0 = int(w * 0.29)
  x1 = int(w * 0.71)
  return [
    _thumb(image, size),
    _crop_resize(image, (x0, int(h * 0.22), x1, int(h * 0.82)), size),
    _crop_resize(image, (0, int(h * 0.24), w, int(h * 0.86)), size),
    _crop_resize(image, (int(w * 0.31), int(h * 0.10), int(w * 0.69), int(h * 0.66)), size),
  ]


def _composite_probe_image(image: Image.Image, size: int) -> Image.Image:
  probes = _probe_images(image, max(64, size // 2))
  half = size // 2
  composite = Image.new("RGB", (size, size), (8, 10, 12))
  slots = ((0, 0), (half, 0), (0, half), (half, half))
  labels = ("full", "corridor", "road", "forward")
  for probe, xy, label in zip(probes, slots, labels, strict=True):
    tile = probe.resize((half, half), Image.Resampling.BILINEAR)
    composite.paste(tile, xy)
    draw = ImageDraw.Draw(composite)
    draw.rectangle((xy[0], xy[1], xy[0] + 52, xy[1] + 14), fill=(0, 0, 0))
    draw.text((xy[0] + 3, xy[1] + 2), label, fill=(255, 255, 255))
  draw = ImageDraw.Draw(composite)
  draw.line((half, 0, half, size), fill=(0, 0, 0), width=2)
  draw.line((0, half, size, half), fill=(0, 0, 0), width=2)
  return composite


def _inference_images(image: Image.Image, image_mode: str, size: int) -> tuple[list[Image.Image], str]:
  if image_mode == "full":
    return [_thumb(image, size)], FULL_LABEL_PROMPT
  if image_mode == "composite":
    return [_composite_probe_image(image, size)], COMPOSITE_LABEL_PROMPT
  return _probe_images(image, size), LABEL_PROMPT


def _extract_new_text(processor, inputs, output_ids) -> str:
  trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
  return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _normalize_labels(raw: str) -> tuple[str, ...]:
  allowed = {
    "cones",
    "barrier",
    "construction_left",
    "construction_right",
    *PATH_CONFLICT_LABELS,
    "red_stop_light",
    "green_go_light",
    "stop_sign",
    "none",
  }
  text = raw.lower()
  text = text.replace("traffic cones", "cones")
  text = text.replace("traffic cone", "cones")
  text = text.replace("barricade", "barrier")
  text = text.replace("construction left", "construction_left")
  text = text.replace("construction on left", "construction_left")
  text = text.replace("construction on the left", "construction_left")
  text = text.replace("cones left", "construction_left")
  text = text.replace("cones on left", "construction_left")
  text = text.replace("cones on the left", "construction_left")
  text = text.replace("barrier left", "construction_left")
  text = text.replace("barrier on left", "construction_left")
  text = text.replace("barrier on the left", "construction_left")
  text = text.replace("construction right", "construction_right")
  text = text.replace("construction on right", "construction_right")
  text = text.replace("construction on the right", "construction_right")
  text = text.replace("cones right", "construction_right")
  text = text.replace("cones on right", "construction_right")
  text = text.replace("cones on the right", "construction_right")
  text = text.replace("barrier right", "construction_right")
  text = text.replace("barrier on right", "construction_right")
  text = text.replace("barrier on the right", "construction_right")
  text = text.replace("pedestrian in path", "pedestrian_in_path")
  text = text.replace("pedestrian entering path", "pedestrian_entering_path")
  text = text.replace("pedestrian entering the path", "pedestrian_entering_path")
  text = text.replace("person in path", "pedestrian_in_path")
  text = text.replace("person entering path", "pedestrian_entering_path")
  text = text.replace("person entering the path", "pedestrian_entering_path")
  text = text.replace("car in path", "vehicle_in_path")
  text = text.replace("car entering path", "vehicle_entering_path")
  text = text.replace("car entering the path", "vehicle_entering_path")
  text = text.replace("vehicle in path", "vehicle_in_path")
  text = text.replace("vehicle entering path", "vehicle_entering_path")
  text = text.replace("vehicle entering the path", "vehicle_entering_path")
  text = text.replace("true moving lead", "true_moving_lead")
  text = text.replace("moving lead", "true_moving_lead")
  text = text.replace("slower lead", "slower_lead")
  text = text.replace("slow lead", "slower_lead")
  text = text.replace("braking lead", "braking_lead")
  text = text.replace("stopped lead", "stopped_lead")
  text = text.replace("stopped vehicle ahead", "stopped_lead")
  text = text.replace("cut in vehicle", "cut_in_vehicle")
  text = text.replace("cut-in vehicle", "cut_in_vehicle")
  text = text.replace("crossing vehicle", "crossing_vehicle")
  text = text.replace("irrelevant vehicle", "irrelevant_vehicle")
  text = text.replace("animal in path", "animal_in_path")
  text = text.replace("animal entering path", "animal_entering_path")
  text = text.replace("animal entering the path", "animal_entering_path")
  text = text.replace("red stop light", "red_stop_light")
  text = text.replace("red light", "red_stop_light")
  text = text.replace("green go light", "green_go_light")
  text = text.replace("green stop light", "green_go_light")
  text = text.replace("green traffic light", "green_go_light")
  text = text.replace("green light", "green_go_light")
  text = text.replace("stop sign", "stop_sign")
  aliases = {
    "cone": "cones",
    "cones": "cones",
    "barriers": "barrier",
    "barricades": "barrier",
  }
  found = []
  for token in re.split(r"[^a-z0-9_]+", text):
    token = aliases.get(token, token)
    if token in allowed and token not in found:
      found.append(token)
  if not found:
    return ("none",)
  if len(found) > 1 and "none" in found:
    found.remove("none")
  return tuple(found)


def _with_visual_fallbacks(
  image: Image.Image,
  labels: tuple[str, ...],
  *,
  enable_signal: bool = False,
  enable_construction: bool = False,
  enable_stop: bool = False,
) -> tuple[str, ...]:
  found = [label for label in labels if label != "none"]
  label_set = set(found)
  if enable_signal or os.getenv("RTP_VLM_ENABLE_VISUAL_SIGNAL_FALLBACK") == "1":
    visual_signal = _visual_traffic_signal_label(image)
    if visual_signal is not None:
      found = [label for label in found if label not in {"red_stop_light", "green_go_light"}]
      found.append(visual_signal)
      label_set = set(found)

  if enable_construction or os.getenv("RTP_VLM_ENABLE_VISUAL_CONSTRUCTION_FALLBACK") == "1":
    visual_construction = _visual_construction_side_label(image)
    if visual_construction is not None:
      found = [label for label in found if label not in {"construction_left", "construction_right"}]
      found.append(visual_construction)
      label_set = set(found)

  if (
    (enable_stop or os.getenv("RTP_VLM_ENABLE_VISUAL_STOP_FALLBACK") == "1") and
    "red_stop_light" not in label_set and
    "green_go_light" not in label_set and
    "stop_sign" not in label_set and
    _has_forward_stop_cue(image)
  ):
    found.append("stop_sign")
    label_set.add("stop_sign")

  return tuple(found) if found else ("none",)


def _visual_traffic_signal_label(image: Image.Image) -> str | None:
  arr = np.asarray(image.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  roi = arr[int(h * 0.04):int(h * 0.50), int(w * 0.30):int(w * 0.74)]
  if roi.size == 0:
    return None
  red = (roi[:, :, 0] > 185) & (roi[:, :, 1] < 95) & (roi[:, :, 2] < 95)
  green = (roi[:, :, 1] > 170) & (roi[:, :, 0] < 95) & (roi[:, :, 2] < 140)
  red_count = int(red.sum())
  green_count = int(green.sum())
  min_pixels = 70
  if red_count >= min_pixels and red_count > green_count * 1.35:
    return "red_stop_light"
  if green_count >= min_pixels and green_count > red_count * 1.35:
    return "green_go_light"
  return None


def _visual_construction_side_label(image: Image.Image) -> str | None:
  arr = np.asarray(image.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  y0 = int(h * 0.42)
  y1 = int(h * 0.96)
  if y1 <= y0:
    return None

  roi = arr[y0:y1, :, :]
  r = roi[:, :, 0]
  g = roi[:, :, 1]
  b = roi[:, :, 2]
  green_path = (g > 85) & (r < 95) & (b < 145)
  blue_construction = (b > 120) & (r < 120) & (g > 35) & (g < 130)
  orange_construction = (r > 155) & (g > 45) & (g < 185) & (b < 125)
  construction = (blue_construction | orange_construction) & ~green_path

  count = int(construction.sum())
  if count < 80:
    return None

  _, path_xs = np.nonzero(green_path)
  path_center_x = float(np.median(path_xs)) if len(path_xs) >= 40 else w * 0.5
  _, construction_xs = np.nonzero(construction)
  construction_center_x = float(np.median(construction_xs))
  left_count = int((construction_xs < path_center_x - max(8.0, w * 0.018)).sum())
  right_count = int((construction_xs > path_center_x + max(8.0, w * 0.018)).sum())

  if right_count > max(50, int(left_count * 1.20)) or construction_center_x > path_center_x + w * 0.035:
    return "construction_right"
  if left_count > max(50, int(right_count * 1.20)) or construction_center_x < path_center_x - w * 0.035:
    return "construction_left"
  return None


def _has_forward_stop_cue(image: Image.Image) -> bool:
  arr = np.asarray(image.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  roi = arr[int(h * 0.10):int(h * 0.42), int(w * 0.35):int(w * 0.65)]
  if roi.size == 0:
    return False
  red = (roi[:, :, 0] > 130) & (roi[:, :, 1] < 90) & (roi[:, :, 2] < 90)
  blue = (roi[:, :, 2] > 130) & (roi[:, :, 0] < 90) & (roi[:, :, 1] < 120)
  return int(red.sum() + blue.sum()) >= 12


def _has_center_dark_upright_obstacle(image: Image.Image) -> bool:
  w, h = image.size
  crop = _crop_resize(
    image,
    (int(w * 0.29), int(h * 0.22), int(w * 0.71), int(h * 0.82)),
    384,
  )
  arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
  h, w, _ = arr.shape
  roi = arr[int(h * 0.10):int(h * 0.80), int(w * 0.35):int(w * 0.65)]
  if roi.size == 0:
    return False
  dark = (roi[:, :, 0] < 100) & (roi[:, :, 1] < 110) & (roi[:, :, 2] < 110)
  ys, xs = np.nonzero(dark)
  if len(xs) < 18:
    return False

  # One narrow upright blob in the planned corridor is enough for a cautious yield.
  x0, x1 = int(xs.min()), int(xs.max())
  y0, y1 = int(ys.min()), int(ys.max())
  width = max(1, x1 - x0 + 1)
  height = max(1, y1 - y0 + 1)
  area = int(len(xs))
  aspect = height / width
  fill = area / float(width * height)
  return 2.0 <= aspect <= 12.0 and 40 <= area <= 2200 and width <= 45 and fill >= 0.08


def _score_label_ids(processor) -> tuple[tuple[int, ...], tuple[int, ...]]:
  tokenizer = processor.tokenizer
  yes_ids = _single_token_ids(tokenizer, ("yes", "Yes", " yes", " Yes"))
  no_ids = _single_token_ids(tokenizer, ("no", "No", " no", " No"))
  if not yes_ids or not no_ids:
    raise RuntimeError(f"failed to find single-token yes/no ids: yes={yes_ids} no={no_ids}")
  return yes_ids, no_ids


def _single_token_ids(tokenizer, variants: Sequence[str]) -> tuple[int, ...]:
  ids: set[int] = set()
  for variant in variants:
    encoded = tokenizer(variant, add_special_tokens=False).input_ids
    if len(encoded) == 1:
      ids.add(int(encoded[0]))
  return tuple(sorted(ids))


def _parse_score_label_groups(raw: str) -> tuple[tuple[str, ...], ...]:
  groups: list[tuple[str, ...]] = []
  for group_raw in raw.split(";"):
    labels = tuple(label.strip() for label in group_raw.split(",") if label.strip())
    if labels:
      groups.append(labels)
  return tuple(groups)


def _validate_score_labels(labels: Sequence[str], parser: argparse.ArgumentParser, field_name: str) -> None:
  unknown = sorted(set(labels) - set(SCORE_LABELS))
  if unknown:
    parser.error(f"unknown {field_name} entries: {unknown}")


def _parse_score_thresholds(raw: str, parser: argparse.ArgumentParser) -> dict[str, float]:
  thresholds: dict[str, float] = {}
  if not raw.strip():
    return thresholds
  for item in raw.split(","):
    if not item.strip():
      continue
    if ":" not in item:
      parser.error(f"invalid --score-thresholds item: {item}")
    label, value_raw = item.split(":", 1)
    label = label.strip()
    if label not in SCORE_LABELS:
      parser.error(f"unknown --score-thresholds label: {label}")
    try:
      thresholds[label] = float(value_raw)
    except ValueError:
      parser.error(f"invalid --score-thresholds value for {label}: {value_raw}")
  return thresholds


def _score_labels(
  processor,
  model,
  images: list[Image.Image],
  image_prompt: str,
  vehicle_state_text: str,
  score_labels: Sequence[str],
  score_threshold: float,
  score_thresholds: dict[str, float] | None = None,
) -> tuple[str, tuple[str, ...], float, float, dict[str, float]]:
  if not score_labels:
    return "none", ("none",), 0.0, 0.0, {}

  prompts: list[str] = []
  batch_images: list[Image.Image] = []
  for label in score_labels:
    question = SCORE_QUESTIONS[label]
    content = [{"type": "image", "image": image} for image in images]
    content.append({
      "type": "text",
      "text": (
        f"{SCORE_PROMPT}\nVehicle state: {vehicle_state_text}\n"
        f"Question: {question}"
      ),
    })
    prompts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    batch_images.extend(images)

  prefill_start = time.perf_counter()
  inputs = processor(text=prompts, images=batch_images, padding=True, return_tensors="pt").to("cuda")
  torch.cuda.synchronize()
  prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

  yes_ids, no_ids = _score_label_ids(processor)
  decode_start = time.perf_counter()
  with torch.inference_mode():
    outputs = model(**inputs)
  torch.cuda.synchronize()
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  next_logits = outputs.logits[:, -1, :]

  scores: dict[str, float] = {}
  selected: list[str] = []
  for idx, label in enumerate(score_labels):
    yes_score = float(torch.max(next_logits[idx, list(yes_ids)]).detach().cpu())
    no_score = float(torch.max(next_logits[idx, list(no_ids)]).detach().cpu())
    score = yes_score - no_score
    scores[label] = score
    threshold = score_threshold if score_thresholds is None else score_thresholds.get(label, score_threshold)
    if score >= threshold:
      selected.append(label)

  labels = tuple(selected) if selected else ("none",)
  score_text = ",".join(f"{label}:{scores[label]:.3f}" for label in score_labels)
  return score_text, labels, prefill_ms, decode_ms, scores


def _construction_side(label_set: set[str]) -> str:
  return construction_hazard_side_from_labels(label_set)


def _construction_rtp_fields(side: str) -> tuple[str, str, str, float, str]:
  fields = construction_rtp_fields_for_hazard_side(side)
  return fields.scene, fields.evidence, fields.meta, fields.lat_bias_m, fields.avoid_token


def _lead_class(label_set: set[str]) -> str:
  for label in (
    "stopped_lead",
    "braking_lead",
    "cut_in_vehicle",
    "crossing_vehicle",
    "slower_lead",
    "true_moving_lead",
    "irrelevant_vehicle",
  ):
    if label in label_set:
      return label
  return ""


def _lead_rtp_fields(lead_class: str) -> tuple[str, str, str, str, str, str, float]:
  if lead_class == "true_moving_lead":
    return "true_moving_lead", "true_moving_lead", "BASE", "none", "none", "", 0.78
  if lead_class == "irrelevant_vehicle":
    return "irrelevant_vehicle", "irrelevant_vehicle", "BASE", "none", "none", "", 0.78
  if lead_class == "slower_lead":
    return "slower_lead", "slower_lead_closing", "SLOW", "80%", "none", "lead_vehicle_s12_45", 0.72
  if lead_class == "braking_lead":
    return "braking_lead", "braking_lead_closing", "YIELD", "45%", "none", "lead_vehicle_s8_45", 0.74
  if lead_class == "stopped_lead":
    return "stopped_lead", "stopped_lead_in_path", "STOP", "0.0", "18.0", "lead_vehicle_s8_35", 0.76
  if lead_class == "cut_in_vehicle":
    return "cut_in_vehicle", "cut_in_vehicle_entering_path", "YIELD", "50%", "none", "cut_in_vehicle_s8_30", 0.73
  if lead_class == "crossing_vehicle":
    return "crossing_vehicle", "crossing_vehicle_conflict", "YIELD", "35%", "none", "crossing_vehicle_s8_30", 0.73
  return "", "", "BASE", "none", "none", "", 0.70


def _path_agent_rtp_fields(label_set: set[str]) -> tuple[str, str, str, str, str, str, float]:
  if label_set & PATH_BLOCKING_AGENT_LABELS:
    return (
      "path_blocking_agent",
      "agent_blocking_path",
      "STOP",
      "0.0",
      f"{PATH_AGENT_STOP_S:.1f}",
      PATH_AGENT_STOP_AVOID,
      0.76,
    )
  if label_set & PATH_ENTERING_AGENT_LABELS:
    return (
      "path_conflict_agent",
      "agent_entering_path",
      "YIELD",
      PATH_AGENT_YIELD_SPEED_CAP,
      "none",
      PATH_AGENT_STOP_AVOID,
      0.70,
    )
  return "path_conflict_agent", "agent_in_or_entering_path", "YIELD", PATH_AGENT_YIELD_SPEED_CAP, "none", PATH_AGENT_STOP_AVOID, 0.70


def _labels_to_rtp(labels: tuple[str, ...]) -> str:
  label_set = set(labels)
  has_red_stop = "red_stop_light" in label_set
  has_stop_sign = "stop_sign" in label_set
  has_stop = has_red_stop or has_stop_sign
  has_green_go = "green_go_light" in label_set
  has_path_conflict_agent = bool(PATH_CONFLICT_LABELS & label_set)
  lead_class = _lead_class(label_set)
  has_active_lead = lead_class in ACTIVE_LEAD_LABELS
  has_construction = bool({
    "cones",
    "barrier",
    "construction_left",
    "construction_right",
    "construction_blue_edge",
    "construction_purple_edge",
    "construction_drive_left",
    "construction_drive_right",
    "construction_shift_left",
    "construction_shift_right",
    "construction_blocks_left_candidate",
    "construction_blocks_right_candidate",
  } & label_set)
  construction_side = _construction_side(label_set)
  construction_scene, construction_evidence, construction_meta, construction_lat_bias, construction_avoid = _construction_rtp_fields(construction_side)
  lead_scene, lead_evidence, lead_meta, lead_speed_cap, lead_stop_s, lead_avoid, lead_confidence = _lead_rtp_fields(lead_class)
  agent_scene, agent_evidence, agent_meta, agent_speed_cap, agent_stop_s, agent_avoid, agent_confidence = _path_agent_rtp_fields(label_set)
  signal_stop_s = _signal_stop_s_from_env()
  signal_stop_token = f"stop_line_s{signal_stop_s:.1f}"

  if has_stop and has_construction:
    avoid = f"[{signal_stop_token}]" if not construction_avoid else f"[{construction_avoid},{signal_stop_token}]"
    stop_speed_cap = "0.0" if has_stop_sign else "none"
    stop_evidence = "stop_sign_for_path" if has_stop_sign else "red_signal_for_path"
    return "\n".join((
      "RTPv1",
      f"scene=mixed_stop_{construction_scene}",
      f"evidence=[{stop_evidence},{construction_evidence}]",
      "meta=STOP",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      f"speed_cap_mps={stop_speed_cap}",
      f"stop_s={signal_stop_s:.1f}",
      f"avoid={avoid}",
      "weights=[obs3.0,lane1.4,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.76",
    ))
  if not has_stop and has_active_lead and has_construction:
    avoid_items = [item for item in (construction_avoid, lead_avoid) if item]
    evidence = f"{lead_evidence},{construction_evidence}"
    return "\n".join((
      "RTPv1",
      f"scene=mixed_{lead_scene}_{construction_scene}",
      f"evidence=[{evidence}]",
      f"meta={lead_meta}",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      f"speed_cap_mps={lead_speed_cap}",
      f"stop_s={lead_stop_s}",
      f"avoid=[{','.join(avoid_items)}]",
      "weights=[obs3.0,lane1.4,comfort1.0,base0.7,vlm1.0]",
      f"confidence={max(lead_confidence, 0.72):.2f}",
    ))
  if has_path_conflict_agent and has_construction:
    avoid = f"[{agent_avoid}]" if not construction_avoid else f"[{construction_avoid},{agent_avoid}]"
    evidence_items = ["green_signal_for_path"] if has_green_go else []
    evidence_items.extend((agent_evidence, construction_evidence))
    return "\n".join((
      "RTPv1",
      f"scene=mixed_agent_{construction_scene}",
      f"evidence=[{','.join(evidence_items)}]",
      f"meta={agent_meta}",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      f"speed_cap_mps={agent_speed_cap}",
      f"stop_s={agent_stop_s}",
      f"avoid={avoid}",
      "weights=[obs3.0,lane1.4,comfort1.0,base0.7,vlm1.0]",
      f"confidence={max(agent_confidence, 0.72, GREEN_SIGNAL_RTP_CONFIDENCE if has_green_go else 0.0):.2f}",
    ))
  if not has_stop and lead_class:
    avoid = "[]" if not lead_avoid else f"[{lead_avoid}]"
    return "\n".join((
      "RTPv1",
      f"scene={lead_scene}",
      f"evidence=[{lead_evidence}]",
      f"meta={lead_meta}",
      "branch=base",
      "lat_bias_m=0.0",
      f"speed_cap_mps={lead_speed_cap}",
      f"stop_s={lead_stop_s}",
      f"avoid={avoid}",
      "weights=[obs2.4,lane1.2,comfort1.0,base0.9,vlm1.0]",
      f"confidence={lead_confidence:.2f}",
    ))
  if has_stop:
    stop_scene = "stop_sign" if has_stop_sign else "red_traffic_light"
    stop_evidence = "stop_sign_for_path" if has_stop_sign else "red_signal_for_path"
    stop_speed_cap = "0.0" if has_stop_sign else "none"
    return "\n".join((
      "RTPv1",
      f"scene={stop_scene}",
      f"evidence=[{stop_evidence}]",
      "meta=STOP",
      "branch=base",
      "lat_bias_m=0.0",
      f"speed_cap_mps={stop_speed_cap}",
      f"stop_s={signal_stop_s:.1f}",
      f"avoid=[{signal_stop_token}]",
      "weights=[obs3.0,lane1.2,comfort1.0,base0.7,vlm1.0]",
      "confidence=0.74",
    ))
  if has_path_conflict_agent:
    evidence = f"green_signal_for_path,{agent_evidence}" if has_green_go else agent_evidence
    return "\n".join((
      "RTPv1",
      f"scene={agent_scene}",
      f"evidence=[{evidence}]",
      f"meta={agent_meta}",
      "branch=base",
      "lat_bias_m=0.0",
      f"speed_cap_mps={agent_speed_cap}",
      f"stop_s={agent_stop_s}",
      f"avoid=[{agent_avoid}]",
      "weights=[obs3.0,lane1.2,comfort1.0,base0.7,vlm1.0]",
      f"confidence={max(agent_confidence, GREEN_SIGNAL_RTP_CONFIDENCE if has_green_go else 0.0):.2f}",
    ))
  if has_construction:
    avoid = "[]" if not construction_avoid else f"[{construction_avoid}]"
    evidence = f"green_signal_for_path,{construction_evidence}" if has_green_go else construction_evidence
    return "\n".join((
      "RTPv1",
      f"scene={construction_scene}",
      f"evidence=[{evidence}]",
      f"meta={construction_meta}",
      "branch=base",
      f"lat_bias_m={construction_lat_bias}",
      "speed_cap_mps=none",
      "stop_s=none",
      f"avoid={avoid}",
      "weights=[obs2.5,lane1.4,comfort1.0,base0.7,vlm1.0]",
      f"confidence={max(0.72, GREEN_SIGNAL_RTP_CONFIDENCE if has_green_go else 0.0):.2f}",
    ))
  if has_green_go:
    return "\n".join((
      "RTPv1",
      "scene=green_traffic_light",
      "evidence=[green_signal_for_path]",
      "meta=BASE",
      "branch=base",
      "lat_bias_m=0.0",
      "speed_cap_mps=none",
      "stop_s=none",
      "avoid=[]",
      "weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]",
      "confidence=0.82",
    ))
  return "\n".join((
    "RTPv1",
    "scene=nominal",
    "evidence=[open_lane]",
    "meta=BASE",
    "branch=base",
    "lat_bias_m=0.0",
    "speed_cap_mps=none",
    "stop_s=none",
    "avoid=[]",
    "weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]",
    "confidence=0.70",
  ))


def generate(
  processor,
  model,
  payload: dict,
  max_new_tokens: int,
  image_mode: str = "multi",
  label_mode: str = "generate",
  score_threshold: float = 0.0,
  score_thresholds: dict[str, float] | None = None,
  score_labels: Sequence[str] = SCORE_LABELS,
  enable_visual_signal_fallback: bool = False,
  enable_visual_construction_fallback: bool = False,
  enable_visual_stop_fallback: bool = False,
) -> dict:
  source = _image_from_payload(payload)
  image_size = int(os.getenv("RTP_VLM_IMAGE_SIZE", "384"))
  images, image_prompt = _inference_images(source, image_mode, image_size)
  scores: dict[str, float] = {}
  generated_token_count = 0

  if label_mode == "score":
    labels_text, labels, prefill_ms, decode_ms, scores = _score_labels(
      processor,
      model,
      images,
      image_prompt,
      str(payload.get("scene_board_state_text", "")),
      score_labels,
      score_threshold,
      score_thresholds,
    )
  else:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"{image_prompt}\nVehicle state: {payload.get('scene_board_state_text', '')}"})
    messages = [{"role": "user", "content": content}]

    prefill_start = time.perf_counter()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    decode_start = time.perf_counter()
    with torch.inference_mode():
      output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
      )
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    labels_text = _extract_new_text(processor, inputs, output_ids)
    labels = _normalize_labels(labels_text)
    generated_token_count = int(output_ids.shape[-1] - inputs.input_ids.shape[-1])

  labels = _with_visual_fallbacks(
    source,
    labels,
    enable_signal=enable_visual_signal_fallback,
    enable_construction=enable_visual_construction_fallback,
    enable_stop=enable_visual_stop_fallback,
  )
  rtp_text = _labels_to_rtp(labels)
  return {
    "text": rtp_text,
    "rtp_text": rtp_text,
    "labels_text": labels_text,
    "labels": list(labels),
    "label_mode": label_mode,
    "image_mode": image_mode,
    "label_scores": scores,
    "generated_token_count": generated_token_count,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "backend": f"qwen2.5-vl-3b-label-rtp-{image_mode}-{label_mode}",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent Qwen label-to-RTP worker. Reads JSONL on stdin, writes JSONL on stdout.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--max-new-tokens", type=int, default=24)
  parser.add_argument("--image-mode", choices=("multi", "composite", "full"), default="full")
  parser.add_argument("--label-mode", choices=("generate", "score"), default="score")
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--score-thresholds", default="")
  parser.add_argument("--score-labels", default=",".join(SCORE_LABELS))
  parser.add_argument("--score-rotate-groups", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--score-label-groups", default=";".join(",".join(group) for group in DEFAULT_SCORE_LABEL_GROUPS))
  parser.add_argument("--score-cache-ttl-frames", type=int, default=3)
  parser.add_argument("--score-durable-labels", default=",".join(DEFAULT_DURABLE_SCORE_LABELS))
  parser.add_argument("--score-negative-clear-threshold", type=float, default=2.0)
  parser.add_argument("--enable-visual-fallbacks", action="store_true", help="Demo-only pixel fallback for synthetic scene boards. Do not use for production road runs.")
  parser.add_argument("--enable-visual-signal-fallback", action="store_true", help="Demo-only red/green pixel fallback for synthetic traffic-light overlays.")
  parser.add_argument("--enable-visual-construction-fallback", action="store_true", help="Demo-only cone/barrier color fallback for synthetic MetaDrive boards.")
  parser.add_argument("--enable-visual-stop-fallback", action="store_true", help="Demo-only stop-sign pixel fallback for synthetic boards.")
  args = parser.parse_args()
  if args.score_rotate_groups is None:
    args.score_rotate_groups = args.label_mode == "score"
  score_labels = tuple(label.strip() for label in args.score_labels.split(",") if label.strip())
  _validate_score_labels(score_labels, parser, "--score-labels")
  durable_score_labels = tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip())
  _validate_score_labels(durable_score_labels, parser, "--score-durable-labels")
  score_thresholds = _parse_score_thresholds(args.score_thresholds, parser)
  score_groups = _parse_score_label_groups(args.score_label_groups)
  for group in score_groups:
    _validate_score_labels(group, parser, "--score-label-groups")
  if args.score_rotate_groups and args.label_mode != "score":
    parser.error("--score-rotate-groups requires --label-mode score")
  if args.score_rotate_groups and not score_groups:
    parser.error("--score-rotate-groups requires at least one --score-label-groups group")
  rotating_state = (
    RotatingScoreState(score_groups, args.score_cache_ttl_frames, durable_score_labels, args.score_negative_clear_threshold)
    if args.score_rotate_groups else None
  )

  processor, model = _load(args.model_dir)
  warm = Image.new("RGB", (384, 384), (20, 20, 20))
  buf = BytesIO()
  warm.save(buf, format="PNG")
  generate(
    processor,
    model,
    {"scene_board_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "scene_board_state_text": "warmup"},
    4,
    image_mode=args.image_mode,
    label_mode=args.label_mode,
    score_threshold=args.score_threshold,
    score_thresholds=score_thresholds,
    score_labels=score_groups[0] if args.score_rotate_groups else score_labels,
    enable_visual_signal_fallback=args.enable_visual_fallbacks or args.enable_visual_signal_fallback,
    enable_visual_construction_fallback=args.enable_visual_fallbacks or args.enable_visual_construction_fallback,
    enable_visual_stop_fallback=args.enable_visual_fallbacks or args.enable_visual_stop_fallback,
  )

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      if payload.get("control") == "reset_runtime_state":
        if args.score_rotate_groups:
          rotating_state = RotatingScoreState(score_groups, args.score_cache_ttl_frames, durable_score_labels, args.score_negative_clear_threshold)
        response = {"ok": True, "control": "reset_runtime_state"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        continue
      request_score_labels = score_labels
      score_group_idx = None
      if rotating_state is not None:
        score_group_idx, request_score_labels = rotating_state.next_group()
      response = generate(
        processor,
        model,
        payload,
        args.max_new_tokens,
        image_mode=args.image_mode,
        label_mode=args.label_mode,
        score_threshold=args.score_threshold,
        score_thresholds=score_thresholds,
        score_labels=request_score_labels,
        enable_visual_signal_fallback=args.enable_visual_fallbacks or args.enable_visual_signal_fallback,
        enable_visual_construction_fallback=args.enable_visual_fallbacks or args.enable_visual_construction_fallback,
        enable_visual_stop_fallback=args.enable_visual_fallbacks or args.enable_visual_stop_fallback,
      )
      if rotating_state is not None:
        frame_id = int(payload.get("frame_id", 0))
        cached_labels = rotating_state.update(
          request_score_labels,
          response["labels"],
          response["label_scores"],
          frame_id,
          str(payload.get("scene_board_state_text", "")),
        )
        cached_labels = _with_visual_fallbacks(
          _image_from_payload(payload),
          cached_labels,
          enable_signal=args.enable_visual_fallbacks or args.enable_visual_signal_fallback,
          enable_construction=args.enable_visual_fallbacks or args.enable_visual_construction_fallback,
          enable_stop=args.enable_visual_fallbacks or args.enable_visual_stop_fallback,
        )
        rtp_text = _labels_to_rtp(cached_labels)
        if rotating_state.construction_clear_active(frame_id):
          rtp_text = _with_rtp_confidence(rtp_text, CONSTRUCTION_CLEAR_RTP_CONFIDENCE)
        response["labels_scored_this_request"] = list(request_score_labels)
        response["score_group_index"] = score_group_idx
        response["labels_current_group"] = response["labels"]
        response["labels"] = list(cached_labels)
        response["label_scores_cached"] = rotating_state.active_scores(frame_id)
        response["label_state_debug"] = rotating_state.debug_state(frame_id)
        response["rtp_text"] = rtp_text
        response["text"] = rtp_text
        response["backend"] = f"{response['backend']}-rotating"
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
  main()
