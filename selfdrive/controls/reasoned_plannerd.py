#!/usr/bin/env python3
from __future__ import annotations

import math
import os

import cereal.messaging as messaging
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.reasoned.pathsynth import BasePlan
from openpilot.selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig, ReasonedStepResult


def _vehicle_state(sm: messaging.SubMaster) -> dict[str, float]:
  car_state = sm["carState"]
  return {
    "v_ego": float(getattr(car_state, "vEgo", 0.0)),
    "a_ego": float(getattr(car_state, "aEgo", 0.0)),
    "steering_angle_deg": float(getattr(car_state, "steeringAngleDeg", 0.0)),
    "left_blinker": float(bool(getattr(car_state, "leftBlinker", False))),
    "right_blinker": float(bool(getattr(car_state, "rightBlinker", False))),
  }


def _safe_float(value: float, default: float = -1.0) -> float:
  return float(value) if math.isfinite(value) else default


def _publish_lateral(pm: messaging.PubMaster, result: ReasonedStepResult) -> None:
  if result.synth is None or not result.should_publish:
    return
  msg = messaging.new_message("lateralManeuverPlan")
  msg.valid = True
  msg.lateralManeuverPlan.desiredCurvature = float(result.synth.desired_curvature)
  pm.send("lateralManeuverPlan", msg)


def _publish_audit(pm: messaging.PubMaster, planner: ReasonedPlanner, result: ReasonedStepResult) -> None:
  msg = messaging.new_message("reasonedTrajectoryPlan")
  msg.valid = bool(result.should_publish)
  plan = msg.reasonedTrajectoryPlan
  plan.frameId = int(result.frame_id)
  plan.modelMonoTime = int(result.model_log_mono_time_ns)
  plan.planValid = bool(result.should_publish)
  plan.generatedTokenCount = int(min(result.generated_token_count, 65535))
  plan.cameraToSceneBoardMs = _safe_float(result.timings.camera_to_scene_board_ms)
  plan.sceneBoardToVlmPrefillMs = _safe_float(result.timings.scene_board_to_vlm_prefill_ms)
  plan.vlmDecodeMs = _safe_float(result.timings.vlm_decode_ms)
  plan.rtpParseMs = _safe_float(result.timings.rtp_parse_ms)
  plan.pathSynthMs = _safe_float(result.timings.path_synth_ms)
  plan.publishAgeMs = _safe_float(result.timings.publish_age_ms)
  plan.controlConsumedAgeMs = -1.0
  plan.deadlineMissCount = int(planner.deadline_miss_count)
  plan.invalidRtpCount = int(planner.invalid_rtp_count)
  plan.vlmBackend = result.vlm_backend
  plan.rtpText = result.rtp_text
  plan.invalidReason = result.invalid_reason
  plan.rtpSourceFrameId = -1 if result.rtp_source_frame_id is None else int(result.rtp_source_frame_id)
  plan.rtpAgeFrames = -1 if result.rtp_age_frames is None else int(result.rtp_age_frames)

  if result.program is not None:
    plan.scene = result.program.scene
    plan.evidence = ",".join(result.program.evidence)
    plan.meta = result.program.meta
    plan.branch = result.program.branch
    plan.latBiasM = float(result.program.lat_bias_m)
    if result.program.speed_cap_mps is not None:
      plan.speedCapMps = float(result.program.speed_cap_mps)
    elif result.program.speed_scale is not None and result.synth is not None:
      plan.speedCapMps = float(result.synth.speed_cap_mps)
    else:
      plan.speedCapMps = -1.0
    plan.stopS = -1.0 if result.program.stop_s is None else float(result.program.stop_s)
    plan.avoid = ",".join(result.program.avoid)
    plan.weights = ",".join(f"{k}{v:.3g}" for k, v in result.program.weights.items())
    plan.confidence = float(result.program.confidence)

  if result.synth is not None:
    plan.selectedCandidate = result.synth.selected_candidate
    plan.desiredCurvature = float(result.synth.desired_curvature)
    plan.vlmChangedPathMeters = float(result.synth.vlm_changed_path_meters)
    plan.vlmChangedSpeedMps = float(result.synth.vlm_changed_speed_mps)

  pm.send("reasonedTrajectoryPlan", msg)


def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)
  cloudlog.info("reasoned_plannerd starting local PC RTP POC")

  planner = ReasonedPlanner(ReasonedPlannerConfig(
    allow_async_rtp=os.getenv("RTP_VLM_ASYNC") == "1",
    max_async_age_frames=int(os.getenv("RTP_VLM_ASYNC_MAX_AGE_FRAMES", "6")),
  ))
  pm = messaging.PubMaster(["lateralManeuverPlan", "reasonedTrajectoryPlan"])
  sm = messaging.SubMaster(
    ["modelV2", "carState", "selfdriveState"],
    poll="modelV2",
    ignore_alive=["carState", "selfdriveState"],
    ignore_avg_freq=["carState", "selfdriveState"],
  )

  while True:
    sm.update()
    if not sm.updated["modelV2"]:
      continue

    state = _vehicle_state(sm)
    base_plan = BasePlan.from_model(sm["modelV2"], sm.logMonoTime["modelV2"], v_ego=state["v_ego"])
    result = planner.step(base_plan, state)
    result.timings.publish_age_ms = result.timings.total_ms
    _publish_lateral(pm, result)
    _publish_audit(pm, planner, result)


if __name__ == "__main__":
  main()
