#!/usr/bin/env python3
import os
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests

import cereal.messaging as messaging
from cereal import car, log
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler
from openpilot.selfdrive.alpamayo.protocol import (
  FRAME_ENCODING_JPEG_BGR,
  PROTOCOL_VERSION,
  REQUEST_CONTENT_TYPE,
  RESPONSE_CONTENT_TYPE,
  builder_to_xyzt,
  decode_payload,
  encode_payload,
  parse_xyzt_dict,
  serialize_jpeg_bgr_frame,
  serialize_nv12_frame,
  xyzt_to_dict,
)
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyzt


SEMANTIC_PLAN_FREQ = 4
PUBLISH_DECIMATION = max(1, ModelConstants.MODEL_RUN_FREQ // SEMANTIC_PLAN_FREQ)
REMOTE_CONNECT_TIMEOUT_S = 0.15
REMOTE_READ_TIMEOUT_S = 2.0
REMOTE_CACHE_MAX_AGE_S = 1.0
REMOTE_INIT_RETRY_S = 5.0
FRAME_SYNC_TOLERANCE_NS = int(0.05 * 1e9)
FRAME_HISTORY_TOLERANCE_NS = int(0.065 * 1e9)
FRAME_BUNDLE_STEP_NS = int(0.1 * 1e9)
FRAME_BUNDLE_COUNT = 2
REMOTE_TRANSPORT_PIXELS = 65536
REMOTE_TRANSPORT_JPEG_QUALITY = 80
POSE_HISTORY_DT_S = 0.1
POSE_HISTORY_STEPS = 16
POSE_EXTRAPOLATION_LIMIT_S = 0.15
@dataclass
class SemanticPlanData:
  t: np.ndarray
  position: np.ndarray
  orientation: np.ndarray
  velocity: np.ndarray
  orientation_rate: np.ndarray
  acceleration: np.ndarray
  confidence: float
  consistency: float
  desired_curvature: float
  desired_acceleration: float
  should_stop: bool
  blend_hint: float
  source: int
  status: int
  age: float = 0.0


@dataclass
class CapturedFrame:
  stream: str
  frame_id: int
  timestamp_sof: int
  timestamp_eof: int
  width: int
  height: int
  stride: int
  uv_offset: int
  data: bytes


@dataclass
class PoseHistorySample:
  timestamp_s: float
  position_ned: np.ndarray
  rotation_ned_from_calib: np.ndarray
  velocity_ned: np.ndarray
  velocity_calib: np.ndarray
  acceleration_calib: np.ndarray
  angular_velocity_calib: np.ndarray


@dataclass
class ProviderContext:
  model_msg: Any
  model_mono_time: int
  car_state: Any
  selfdrive_state: Any
  nav_instruction: Any
  device_state: Any
  road_camera_state: Any
  live_calibration: Any
  driver_monitoring_state: Any
  frame_bundle: dict[str, list[CapturedFrame]]
  ego_history_xyz: np.ndarray | None
  ego_history_rot: np.ndarray | None
  frame_t0_s: float | None


def _xyzt_to_numpy(builder) -> tuple[np.ndarray, np.ndarray] | None:
  xyzt = builder_to_xyzt(builder)
  if xyzt is None:
    return None
  return parse_xyzt_dict(xyzt, xyzt["t"])


def _resample_component(t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
  if np.array_equal(t, target_t):
    return values.astype(np.float32)
  return np.column_stack([
    np.interp(target_t, t, values[:, 0]),
    np.interp(target_t, t, values[:, 1]),
    np.interp(target_t, t, values[:, 2]),
  ]).astype(np.float32)


def _extract_model_components(model_msg) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
  components = {
    "position": _xyzt_to_numpy(model_msg.position),
    "orientation": _xyzt_to_numpy(model_msg.orientation),
    "velocity": _xyzt_to_numpy(model_msg.velocity),
    "orientation_rate": _xyzt_to_numpy(model_msg.orientationRate),
    "acceleration": _xyzt_to_numpy(model_msg.acceleration),
  }
  if any(v is None for v in components.values()):
    return None
  return components


def _clone_semantic_plan(plan: SemanticPlanData, age: float | None = None) -> SemanticPlanData:
  return SemanticPlanData(
    t=plan.t.copy(),
    position=plan.position.copy(),
    orientation=plan.orientation.copy(),
    velocity=plan.velocity.copy(),
    orientation_rate=plan.orientation_rate.copy(),
    acceleration=plan.acceleration.copy(),
    confidence=plan.confidence,
    consistency=plan.consistency,
    desired_curvature=plan.desired_curvature,
    desired_acceleration=plan.desired_acceleration,
    should_stop=plan.should_stop,
    blend_hint=plan.blend_hint,
    source=plan.source,
    status=plan.status,
    age=plan.age if age is None else age,
  )


def _enum_name(value: Any) -> str:
  return str(value)


def _response_source(source: Any) -> int:
  if isinstance(source, int):
    return source
  normalized = str(source).replace("_", "").replace("-", "").lower()
  if normalized == "localegpu":
    return log.SemanticPlan.Source.localEgpu
  if normalized == "stockmirror":
    return log.SemanticPlan.Source.stockMirror
  return log.SemanticPlan.Source.remoteServer


def _response_status(status: Any) -> int:
  if isinstance(status, int):
    return status
  normalized = str(status).replace("_", "").replace("-", "").lower()
  mapping = {
    "valid": log.SemanticPlan.Status.valid,
    "stale": log.SemanticPlan.Status.stale,
    "error": log.SemanticPlan.Status.error,
    "unavailable": log.SemanticPlan.Status.unavailable,
  }
  return mapping.get(normalized, log.SemanticPlan.Status.valid)


def _safe_float(value: Any, default: float) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _clamped_float(value: Any, default: float, lower: float = 0.0, upper: float = 1.0) -> float:
  return float(np.clip(_safe_float(value, default), lower, upper))


def _xyzt_first_component(builder: Any, default: float = 0.0) -> float:
  try:
    values = getattr(builder, "x")
    if len(values):
      return float(values[0])
  except (AttributeError, IndexError, TypeError, ValueError):
    pass
  return default


def _approx_vehicle_state(model_msg: Any) -> Any:
  v_ego = _xyzt_first_component(getattr(model_msg, "velocity", None), 0.0)
  a_ego = _xyzt_first_component(getattr(model_msg, "acceleration", None), 0.0)
  return SimpleNamespace(
    vEgo=v_ego,
    aEgo=a_ego,
    standstill=abs(v_ego) < 0.1,
    steeringAngleDeg=0.0,
    gasPressed=False,
    brakePressed=False,
  )


def _approx_selfdrive_state() -> Any:
  return SimpleNamespace(
    enabled=False,
    active=False,
    experimentalMode=False,
  )


def _approx_driver_monitoring_state() -> Any:
  return SimpleNamespace(
    isRHD=False,
  )


def _fill_xyzt_safe(builder: Any, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
  # Cap'n Proto accepts plain Python lists for assignment, while fill_xyzt assumes numpy arrays for x/y/z.
  builder.t = np.asarray(t, dtype=np.float32).tolist()
  builder.x = np.asarray(x, dtype=np.float32).tolist()
  builder.y = np.asarray(y, dtype=np.float32).tolist()
  builder.z = np.asarray(z, dtype=np.float32).tolist()


def _transport_frame_dims(width: int, height: int, max_pixels: int = REMOTE_TRANSPORT_PIXELS) -> tuple[int, int]:
  aspect = float(width) / float(height)
  target_width = int(round(np.sqrt(max_pixels * aspect)))
  target_height = int(round(target_width / aspect))
  target_width = max(64, target_width - (target_width % 2))
  target_height = max(64, target_height - (target_height % 2))

  while target_width * target_height > max_pixels:
    if target_width >= target_height:
      target_width -= 2
    else:
      target_height -= 2
  return target_width, target_height


def _encode_transport_frame(frame: CapturedFrame) -> dict[str, Any]:
  import cv2

  target_width, target_height = _transport_frame_dims(frame.width, frame.height)
  expected_bytes = frame.stride * frame.height * 3 // 2
  payload = frame.data[:expected_bytes]
  if len(payload) < expected_bytes:
    return serialize_nv12_frame(
      frame.stream,
      frame.data,
      frame.width,
      frame.height,
      frame.stride,
      frame.uv_offset,
      frame.frame_id,
      frame.timestamp_sof,
      frame.timestamp_eof,
    )

  nv12 = np.frombuffer(payload, dtype=np.uint8).reshape((frame.height + frame.height // 2, frame.stride))
  bgr = cv2.cvtColor(nv12[:, :frame.width], cv2.COLOR_YUV2BGR_NV12)
  if (target_width, target_height) != (frame.width, frame.height):
    bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
  ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), REMOTE_TRANSPORT_JPEG_QUALITY])
  if not ok:
    return serialize_nv12_frame(
      frame.stream,
      frame.data,
      frame.width,
      frame.height,
      frame.stride,
      frame.uv_offset,
      frame.frame_id,
      frame.timestamp_sof,
      frame.timestamp_eof,
    )

  return serialize_jpeg_bgr_frame(
    frame.stream,
    encoded.tobytes(),
    target_width,
    target_height,
    frame.frame_id,
    frame.timestamp_sof,
    frame.timestamp_eof,
  )


def _camera_context(ctx: ProviderContext) -> dict[str, Any]:
  sensor = _enum_name(getattr(ctx.road_camera_state, "sensor", "unknown"))
  device_type = _enum_name(getattr(ctx.device_state, "deviceType", "unknown"))
  dc = DEVICE_CAMERAS.get((device_type, sensor)) or DEVICE_CAMERAS.get(("unknown", sensor)) or DEVICE_CAMERAS[("pc", "unknown")]

  calib_rpy = np.zeros(3, dtype=np.float32)
  live_calib = getattr(ctx.live_calibration, "rpyCalib", [])
  if len(live_calib) == 3:
    calib_rpy[:] = np.asarray(live_calib, dtype=np.float32)

  return {
    "deviceType": device_type,
    "sensor": sensor,
    "calibrationRpy": calib_rpy.tolist(),
    "roadIntrinsics": dc.fcam.intrinsics.astype(np.float32).tolist(),
    "wideIntrinsics": dc.ecam.intrinsics.astype(np.float32).tolist(),
    "roadWarpMatrix": get_warp_matrix(calib_rpy, dc.fcam.intrinsics, False).astype(np.float32).tolist(),
    "wideWarpMatrix": get_warp_matrix(calib_rpy, dc.ecam.intrinsics, True).astype(np.float32).tolist(),
    "frameSyncToleranceNs": FRAME_SYNC_TOLERANCE_NS,
  }


def _nav_context(nav_instruction, nav_present: bool) -> dict[str, Any] | None:
  if not nav_present:
    return None

  primary = getattr(nav_instruction, "maneuverPrimaryText", "").strip()
  secondary = getattr(nav_instruction, "maneuverSecondaryText", "").strip()
  distance_m = float(getattr(nav_instruction, "maneuverDistance", 0.0))
  nav_text_parts = [part for part in (primary, secondary) if part]
  if distance_m > 0.0:
    nav_text_parts.append(f"in {int(round(distance_m))}m")

  return {
    "maneuverPrimaryText": primary,
    "maneuverSecondaryText": secondary,
    "maneuverDistance": distance_m,
    "maneuverType": getattr(nav_instruction, "maneuverType", ""),
    "maneuverModifier": getattr(nav_instruction, "maneuverModifier", ""),
    "distanceRemaining": float(getattr(nav_instruction, "distanceRemaining", 0.0)),
    "timeRemaining": float(getattr(nav_instruction, "timeRemaining", 0.0)),
    "timeRemainingTypical": float(getattr(nav_instruction, "timeRemainingTypical", 0.0)),
    "showFull": bool(getattr(nav_instruction, "showFull", False)),
    "speedLimit": float(getattr(nav_instruction, "speedLimit", 0.0)),
    "speedLimitSign": _enum_name(getattr(nav_instruction, "speedLimitSign", "")),
    "text": " ".join(nav_text_parts),
  }


class StockMirrorProvider:
  name = "stock-mirror"

  def build(self, ctx: ProviderContext) -> SemanticPlanData | None:
    components = _extract_model_components(ctx.model_msg)
    if components is None:
      return None

    t = components["position"][0]
    return SemanticPlanData(
      t=t,
      position=components["position"][1],
      orientation=components["orientation"][1],
      velocity=components["velocity"][1],
      orientation_rate=components["orientation_rate"][1],
      acceleration=components["acceleration"][1],
      confidence=0.35,
      consistency=1.0,
      desired_curvature=float(ctx.model_msg.action.desiredCurvature),
      desired_acceleration=float(ctx.model_msg.action.desiredAcceleration),
      should_stop=bool(ctx.model_msg.action.shouldStop),
      blend_hint=0.25,
      source=log.SemanticPlan.Source.stockMirror,
      status=log.SemanticPlan.Status.valid,
    )


class PoseHistoryBuffer:
  def __init__(self, max_seconds: float = 4.0):
    self.samples: deque[PoseHistorySample] = deque(maxlen=int(max_seconds * ModelConstants.MODEL_RUN_FREQ) + 16)
    self.calibrator = PoseCalibrator()

  @staticmethod
  def _measurement_valid(live_pose, field: str) -> bool:
    return bool(getattr(getattr(live_pose, field), "valid", False))

  def update_calibration(self, live_calibration) -> None:
    if len(getattr(live_calibration, "rpyCalib", [])) == 3:
      self.calibrator.feed_live_calib(live_calibration)

  def feed(self, live_pose, timestamp_ns: int | None = None) -> None:
    if not (self._measurement_valid(live_pose, "orientationNED") and self._measurement_valid(live_pose, "velocityDevice")):
      return

    if timestamp_ns is None:
      timestamp_ns = int(getattr(live_pose, "timestamp", 0))
    if timestamp_ns <= 0:
      return

    t_s = timestamp_ns * 1e-9
    if self.samples and t_s <= self.samples[-1].timestamp_s:
      return

    pose = Pose.from_live_pose(live_pose)
    calibrated_pose = self.calibrator.build_calibrated_pose(pose)
    rotation_ned_from_calib = rot_from_euler(calibrated_pose.orientation.xyz).astype(np.float32)
    velocity_calib = calibrated_pose.velocity.xyz.astype(np.float32)
    velocity_ned = (rotation_ned_from_calib @ velocity_calib).astype(np.float32)
    acceleration_calib = calibrated_pose.acceleration.xyz.astype(np.float32)
    angular_velocity_calib = calibrated_pose.angular_velocity.xyz.astype(np.float32)

    if self.samples:
      prev = self.samples[-1]
      dt = max(t_s - prev.timestamp_s, 0.0)
      position_ned = prev.position_ned + 0.5 * (prev.velocity_ned + velocity_ned) * dt
    else:
      position_ned = np.zeros(3, dtype=np.float32)

    self.samples.append(PoseHistorySample(
      timestamp_s=t_s,
      position_ned=position_ned.astype(np.float32),
      rotation_ned_from_calib=rotation_ned_from_calib,
      velocity_ned=velocity_ned,
      velocity_calib=velocity_calib,
      acceleration_calib=acceleration_calib,
      angular_velocity_calib=angular_velocity_calib,
    ))

  def build(self, t0_s: float, num_steps: int = POSE_HISTORY_STEPS, dt_s: float = POSE_HISTORY_DT_S) -> tuple[np.ndarray, np.ndarray] | None:
    if len(self.samples) < 2:
      return None

    timestamps = np.asarray([sample.timestamp_s for sample in self.samples], dtype=np.float64)
    positions = np.asarray([sample.position_ned for sample in self.samples], dtype=np.float32)
    rotations = np.asarray([sample.rotation_ned_from_calib for sample in self.samples], dtype=np.float32)

    if t0_s > timestamps[-1]:
      extrapolation_dt = t0_s - timestamps[-1]
      if extrapolation_dt > POSE_EXTRAPOLATION_LIMIT_S:
        return None

      last = self.samples[-1]
      timestamps = np.concatenate([timestamps, [t0_s]])
      positions = np.vstack([positions, last.position_ned + last.velocity_ned * extrapolation_dt])
      rotations = np.concatenate([rotations, last.rotation_ned_from_calib[None]], axis=0)

    target_times = t0_s + np.arange(-(num_steps - 1), 1, dtype=np.float64) * dt_s
    if target_times[0] < timestamps[0] or target_times[-1] > timestamps[-1]:
      return None

    eulers = np.asarray([euler_from_rot(rot) for rot in rotations], dtype=np.float32)
    eulers[:, 2] = np.unwrap(eulers[:, 2])

    interp_positions = np.column_stack([
      np.interp(target_times, timestamps, positions[:, 0]),
      np.interp(target_times, timestamps, positions[:, 1]),
      np.interp(target_times, timestamps, positions[:, 2]),
    ]).astype(np.float32)
    interp_eulers = np.column_stack([
      np.interp(target_times, timestamps, eulers[:, 0]),
      np.interp(target_times, timestamps, eulers[:, 1]),
      np.interp(target_times, timestamps, eulers[:, 2]),
    ]).astype(np.float32)
    interp_rotations = np.asarray([rot_from_euler(euler).astype(np.float32) for euler in interp_eulers], dtype=np.float32)

    t0_rotation = interp_rotations[-1]
    local_positions = (t0_rotation.T @ (interp_positions - interp_positions[-1]).T).T.astype(np.float32)
    local_rotations = np.asarray([t0_rotation.T @ rotation for rotation in interp_rotations], dtype=np.float32)
    return local_positions, local_rotations


class VisionStreamManager:
  STREAMS = {
    "road": VisionStreamType.VISION_STREAM_ROAD,
    "wideRoad": VisionStreamType.VISION_STREAM_WIDE_ROAD,
  }
  REQUIRED_STREAMS = ("wideRoad", "road")

  def __init__(self):
    self.clients: dict[str, VisionIpcClient] = {}
    self.history: dict[str, deque[CapturedFrame]] = {}

    while True:
      available_streams = VisionIpcClient.available_streams("camerad", block=False)
      if available_streams:
        break
      time.sleep(0.1)

    missing_streams = [name for name, stream_type in self.STREAMS.items() if stream_type not in available_streams]
    if missing_streams:
      raise RuntimeError(f"alpamayod remote provider requires narrow+wide streams, missing={missing_streams}")

    for name, stream_type in self.STREAMS.items():
      client = VisionIpcClient("camerad", stream_type, True)
      while not client.connect(False):
        time.sleep(0.1)
      self.clients[name] = client
      self.history[name] = deque(maxlen=12)
      cloudlog.info("alpamayod connected %s stream: %dx%d", name, client.width, client.height)

  def poll(self) -> None:
    for name, client in self.clients.items():
      while True:
        buf = client.recv(1)
        if buf is None:
          break
        self.history[name].append(CapturedFrame(
          stream=name,
          frame_id=int(client.frame_id),
          timestamp_sof=int(client.timestamp_sof),
          timestamp_eof=int(client.timestamp_eof),
          width=int(buf.width),
          height=int(buf.height),
          stride=int(buf.stride),
          uv_offset=int(buf.uv_offset),
          data=np.asarray(buf.data, dtype=np.uint8).tobytes(),
        ))

  @staticmethod
  def _select_frames(frames: list[CapturedFrame], target_times: list[int]) -> list[CapturedFrame] | None:
    selected: list[CapturedFrame] = []
    used_frame_ids: set[int] = set()
    for target_time in target_times:
      candidates = [frame for frame in frames if frame.frame_id not in used_frame_ids]
      if not candidates:
        return None
      best = min(candidates, key=lambda frame: abs(frame.timestamp_eof - target_time))
      if abs(best.timestamp_eof - target_time) > FRAME_HISTORY_TOLERANCE_NS:
        return None
      used_frame_ids.add(best.frame_id)
      selected.append(best)
    selected.sort(key=lambda frame: frame.timestamp_eof)
    return selected

  def get_frame_bundle(self, num_frames: int = FRAME_BUNDLE_COUNT, step_ns: int = FRAME_BUNDLE_STEP_NS) -> dict[str, list[CapturedFrame]]:
    self.poll()
    if any(len(self.history.get(name, ())) == 0 for name in self.REQUIRED_STREAMS):
      return {}

    latest_t0_ns = min(self.history[name][-1].timestamp_eof for name in self.REQUIRED_STREAMS)
    target_times = [latest_t0_ns - (num_frames - 1 - idx) * step_ns for idx in range(num_frames)]

    selected: dict[str, list[CapturedFrame]] = {}
    for name in self.REQUIRED_STREAMS:
      picked = self._select_frames(list(self.history[name]), target_times)
      if picked is None:
        return {}
      selected[name] = picked
    return selected


class RemoteServerProvider:
  name = "remote-server"

  def __init__(self, endpoint: str):
    self.endpoint = endpoint
    self.session = requests.Session()
    self.last_valid_plan: SemanticPlanData | None = None
    self.last_valid_monotonic = 0.0

  def _build_request(self, ctx: ProviderContext) -> dict[str, Any] | None:
    stock_components = _extract_model_components(ctx.model_msg)
    if stock_components is None or ctx.ego_history_xyz is None or ctx.ego_history_rot is None:
      return None
    if any(name not in ctx.frame_bundle for name in VisionStreamManager.REQUIRED_STREAMS):
      return None

    ordered_frames = [
      frame
      for stream_name in VisionStreamManager.REQUIRED_STREAMS
      for frame in ctx.frame_bundle[stream_name]
    ]
    if len(ordered_frames) != len(VisionStreamManager.REQUIRED_STREAMS) * FRAME_BUNDLE_COUNT:
      return None

    frame_payloads = [_encode_transport_frame(frame) for frame in ordered_frames]

    latest_road = ctx.frame_bundle["road"][-1]
    latest_wide = ctx.frame_bundle["wideRoad"][-1]
    frame_skew_ms = abs(latest_road.timestamp_sof - latest_wide.timestamp_sof) / 1e6

    return {
      "protocolVersion": PROTOCOL_VERSION,
      "sentMonoTime": time.monotonic_ns(),
      "modelMonoTime": int(ctx.model_mono_time),
      "semanticPlanHz": SEMANTIC_PLAN_FREQ,
      "camera": _camera_context(ctx),
      "cameraBundle": {
        "streamOrder": list(VisionStreamManager.REQUIRED_STREAMS),
        "framesPerCamera": FRAME_BUNDLE_COUNT,
        "frameStepNs": FRAME_BUNDLE_STEP_NS,
      },
      "frames": frame_payloads,
      "frameSkewMs": frame_skew_ms,
      "frameT0Ns": int(ctx.frame_t0_s * 1e9) if ctx.frame_t0_s is not None else None,
      "egoHistory": {
        "dtS": POSE_HISTORY_DT_S,
        "xyz": ctx.ego_history_xyz.tolist(),
        "rot": ctx.ego_history_rot.tolist(),
      },
      "runtimeConfig": {
        "cameraMode": "front2",
        "numFrames": FRAME_BUNDLE_COUNT,
        "minPixels": REMOTE_TRANSPORT_PIXELS,
        "maxPixels": REMOTE_TRANSPORT_PIXELS,
        "reasoningMode": "prefill_future_start",
        "diffusionSteps": 6,
        "numTrajSamples": 1,
        "deviceMapMode": "manual_split",
        "splitIndex": 16,
        "attnImplementation": "flash_attention_2",
        "expertAttnImplementation": "eager",
        "transportEncoding": FRAME_ENCODING_JPEG_BGR,
        "transportJpegQuality": REMOTE_TRANSPORT_JPEG_QUALITY,
      },
      "vehicleState": {
        "vEgo": float(getattr(ctx.car_state, "vEgo", 0.0)),
        "aEgo": float(getattr(ctx.car_state, "aEgo", 0.0)),
        "standstill": bool(getattr(ctx.car_state, "standstill", False)),
        "steeringAngleDeg": float(getattr(ctx.car_state, "steeringAngleDeg", 0.0)),
        "gasPressed": bool(getattr(ctx.car_state, "gasPressed", False)),
        "brakePressed": bool(getattr(ctx.car_state, "brakePressed", False)),
      },
      "selfdriveState": {
        "enabled": bool(getattr(ctx.selfdrive_state, "enabled", False)),
        "active": bool(getattr(ctx.selfdrive_state, "active", False)),
        "experimentalMode": bool(getattr(ctx.selfdrive_state, "experimentalMode", False)),
        "isRhd": bool(getattr(ctx.driver_monitoring_state, "isRHD", False)),
      },
      "navigation": _nav_context(ctx.nav_instruction, ctx.nav_instruction is not None),
      "stockPlan": {
        "position": xyzt_to_dict(*stock_components["position"]),
        "orientation": xyzt_to_dict(*stock_components["orientation"]),
        "velocity": xyzt_to_dict(*stock_components["velocity"]),
        "orientationRate": xyzt_to_dict(*stock_components["orientation_rate"]),
        "acceleration": xyzt_to_dict(*stock_components["acceleration"]),
        "action": {
          "desiredCurvature": float(ctx.model_msg.action.desiredCurvature),
          "desiredAcceleration": float(ctx.model_msg.action.desiredAcceleration),
          "shouldStop": bool(ctx.model_msg.action.shouldStop),
        },
      },
    }

  def _response_component(self, payload: dict[str, Any] | None, key: str,
                          fallback_t: np.ndarray, fallback_values: np.ndarray,
                          shared_t: np.ndarray) -> np.ndarray:
    if payload is None or key not in payload:
      return _resample_component(fallback_t, fallback_values, shared_t)
    parsed = parse_xyzt_dict(payload[key], payload.get("t", fallback_t))
    if parsed is None:
      return _resample_component(fallback_t, fallback_values, shared_t)
    component_t, component_values = parsed
    return _resample_component(component_t, component_values, shared_t)

  def _response_to_plan(self, response: dict[str, Any], ctx: ProviderContext) -> SemanticPlanData | None:
    if not isinstance(response, dict):
      return None

    semantic = response.get("semanticPlan", response)
    if not isinstance(semantic, dict):
      return None

    stock_components = _extract_model_components(ctx.model_msg)
    if stock_components is None:
      return None

    stock_t = stock_components["position"][0]
    trajectory = semantic.get("trajectory", semantic)
    if not isinstance(trajectory, dict):
      trajectory = semantic

    position_payload = trajectory.get("position")
    shared_t = stock_t
    if isinstance(position_payload, dict):
      parsed_position = parse_xyzt_dict(position_payload, trajectory.get("t", stock_t))
      if parsed_position is not None:
        shared_t = parsed_position[0]

    position = self._response_component(trajectory, "position", *stock_components["position"], shared_t)
    orientation = self._response_component(trajectory, "orientation", *stock_components["orientation"], shared_t)
    velocity = self._response_component(trajectory, "velocity", *stock_components["velocity"], shared_t)
    orientation_rate = self._response_component(trajectory, "orientationRate", *stock_components["orientation_rate"], shared_t)
    acceleration = self._response_component(trajectory, "acceleration", *stock_components["acceleration"], shared_t)

    status = _response_status(semantic.get("status", "valid"))
    return SemanticPlanData(
      t=shared_t.astype(np.float32),
      position=position,
      orientation=orientation,
      velocity=velocity,
      orientation_rate=orientation_rate,
      acceleration=acceleration,
      confidence=_clamped_float(semantic.get("confidence", 0.0), 0.0),
      consistency=_clamped_float(semantic.get("consistency", 1.0), 1.0),
      desired_curvature=_safe_float(semantic.get("desiredCurvature", ctx.model_msg.action.desiredCurvature), float(ctx.model_msg.action.desiredCurvature)),
      desired_acceleration=_safe_float(semantic.get("desiredAcceleration", ctx.model_msg.action.desiredAcceleration), float(ctx.model_msg.action.desiredAcceleration)),
      should_stop=bool(semantic.get("shouldStop", ctx.model_msg.action.shouldStop)),
      blend_hint=_clamped_float(semantic.get("blendHint", 1.0), 1.0),
      source=_response_source(semantic.get("source", "remoteServer")),
      status=status,
      age=max(_safe_float(semantic.get("age", 0.0), 0.0), 0.0),
    )

  def build(self, ctx: ProviderContext) -> SemanticPlanData | None:
    request_payload = self._build_request(ctx)
    if request_payload is None:
      return None

    request_start = time.monotonic()
    try:
      response = self.session.post(
        self.endpoint,
        data=encode_payload(request_payload),
        headers={
          "Content-Type": REQUEST_CONTENT_TYPE,
          "Accept": RESPONSE_CONTENT_TYPE,
        },
        timeout=(REMOTE_CONNECT_TIMEOUT_S, REMOTE_READ_TIMEOUT_S),
      )
      response.raise_for_status()
      plan = self._response_to_plan(decode_payload(response.content), ctx)
      if plan is None:
        return None

      plan.age = max(plan.age, time.monotonic() - request_start)
      if plan.status == log.SemanticPlan.Status.valid:
        self.last_valid_plan = _clone_semantic_plan(plan)
        self.last_valid_monotonic = time.monotonic() - plan.age
      return plan
    except Exception as exc:
      age = time.monotonic() - self.last_valid_monotonic
      if self.last_valid_plan is not None and age < REMOTE_CACHE_MAX_AGE_S:
        cloudlog.warning("alpamayod remote request failed, using cached semantic plan: %s", exc)
        return _clone_semantic_plan(self.last_valid_plan, age=age)

      cloudlog.warning("alpamayod remote request failed without usable cache: %s", exc)
      return None


def main():
  # This sidecar must never compete with the stock control/model daemons for realtime scheduling.
  try:
    os.nice(10)
  except OSError:
    pass
  params = Params()

  cloudlog.info("alpamayod is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("alpamayod got CarParams: %s", CP.brand)

  endpoint_value = params.get("AlpamayoServerEndpoint")
  if isinstance(endpoint_value, bytes):
    endpoint = endpoint_value.decode("utf-8")
  else:
    endpoint = endpoint_value or None
  remote_enabled = endpoint is not None and len(endpoint) > 0
  remote_provider: RemoteServerProvider | None = None
  vision_streams: VisionStreamManager | None = None
  next_remote_init_attempt = 0.0
  pose_history = PoseHistoryBuffer()

  sm = messaging.SubMaster(
    ['modelV2', 'navInstruction', 'deviceState', 'roadCameraState', 'liveCalibration', 'livePose'],
    poll='modelV2',
    ignore_alive=['navInstruction'],
    ignore_avg_freq=['navInstruction'],
    ignore_valid=['navInstruction'],
  )
  pm = messaging.PubMaster(['semanticPlan'])
  consecutive_valid = 0
  publish_count = 0

  cloudlog.info("alpamayod booted with remote_enabled=%s", remote_enabled)

  while True:
    try:
      sm.update()
      now = time.monotonic()

      if sm.updated['liveCalibration']:
        pose_history.update_calibration(sm['liveCalibration'])
      if sm.updated['livePose']:
        pose_history.feed(sm['livePose'], int(sm.logMonoTime['livePose']))

      # Match the working manual probe: keep VisionIPC history warm continuously,
      # not only on the decimated semantic publish ticks.
      if vision_streams is not None:
        vision_streams.poll()

      if not sm.updated['modelV2']:
        continue

      publish_count += 1
      if publish_count % PUBLISH_DECIMATION != 0:
        continue

      plan_send = messaging.new_message('semanticPlan')
      plan_send.valid = sm.seen['modelV2'] and sm.valid['modelV2']
      sp = plan_send.semanticPlan
      sp.frameId = sm['modelV2'].frameId
      sp.frameIdExtra = sm['modelV2'].frameIdExtra
      sp.timestampEof = sm['modelV2'].timestampEof
      sp.modelMonoTime = sm.logMonoTime['modelV2']
      sp.source = log.SemanticPlan.Source.none
      sp.status = log.SemanticPlan.Status.unavailable

      if remote_enabled and remote_provider is None and now >= next_remote_init_attempt:
        try:
          remote_provider = RemoteServerProvider(endpoint)
          vision_streams = VisionStreamManager()
          cloudlog.info("alpamayod remote provider initialized")
        except Exception:
          remote_provider = None
          vision_streams = None
          next_remote_init_attempt = now + REMOTE_INIT_RETRY_S
          cloudlog.exception("alpamayod remote path init failed; staying stock-mirror until retry")

      use_remote = remote_provider is not None and vision_streams is not None
      frame_bundle = vision_streams.get_frame_bundle() if use_remote else {}
      frame_t0_s = None
      ego_history_xyz = ego_history_rot = None
      if frame_bundle:
        frame_t0_ns = min(frame_bundle[name][-1].timestamp_eof for name in VisionStreamManager.REQUIRED_STREAMS)
        frame_t0_s = frame_t0_ns * 1e-9
        history = pose_history.build(frame_t0_s)
        if history is not None:
          ego_history_xyz, ego_history_rot = history

      nav_instruction = sm['navInstruction'] if sm.seen['navInstruction'] else None
      ctx = ProviderContext(
        model_msg=sm['modelV2'],
        model_mono_time=sm.logMonoTime['modelV2'],
        car_state=_approx_vehicle_state(sm['modelV2']),
        selfdrive_state=_approx_selfdrive_state(),
        nav_instruction=nav_instruction,
        device_state=sm['deviceState'],
        road_camera_state=sm['roadCameraState'],
        live_calibration=sm['liveCalibration'],
        driver_monitoring_state=_approx_driver_monitoring_state(),
        frame_bundle=frame_bundle,
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        frame_t0_s=frame_t0_s,
      )

      generation_start = time.monotonic()
      semantic_plan = None
      if use_remote and ego_history_xyz is not None and ego_history_rot is not None:
        semantic_plan = remote_provider.build(ctx)
      generation_time = time.monotonic() - generation_start

      if semantic_plan is None:
        consecutive_valid = 0
        sp.generationExecutionTime = float(generation_time)
        sp.age = 0.0
        sp.confidence = 0.0
        sp.consistency = 0.0
        sp.consecutiveValid = 0
        sp.navInstructionPresent = nav_instruction is not None
        sp.blendHint = 0.0
        pm.send('semanticPlan', plan_send)
        continue

      if semantic_plan.status == log.SemanticPlan.Status.valid:
        consecutive_valid = min(consecutive_valid + 1, 255)
      else:
        consecutive_valid = 0

      sp.generationExecutionTime = float(generation_time)
      sp.age = float(semantic_plan.age)
      sp.confidence = semantic_plan.confidence
      sp.consistency = semantic_plan.consistency
      sp.desiredCurvature = semantic_plan.desired_curvature
      sp.desiredAcceleration = semantic_plan.desired_acceleration
      sp.shouldStop = semantic_plan.should_stop
      sp.source = semantic_plan.source
      sp.status = semantic_plan.status
      sp.consecutiveValid = consecutive_valid
      sp.navInstructionPresent = nav_instruction is not None
      sp.blendHint = semantic_plan.blend_hint

      _fill_xyzt_safe(sp.position, semantic_plan.t, *semantic_plan.position.T)
      _fill_xyzt_safe(sp.orientation, semantic_plan.t, *semantic_plan.orientation.T)
      _fill_xyzt_safe(sp.velocity, semantic_plan.t, *semantic_plan.velocity.T)
      _fill_xyzt_safe(sp.orientationRate, semantic_plan.t, *semantic_plan.orientation_rate.T)
      _fill_xyzt_safe(sp.acceleration, semantic_plan.t, *semantic_plan.acceleration.T)
      pm.send('semanticPlan', plan_send)
    except Exception:
      cloudlog.exception("alpamayod iteration failed; publishing unavailable semantic plan")
      consecutive_valid = 0
      fallback_send = messaging.new_message('semanticPlan')
      if sm.seen['modelV2']:
        fallback_send.semanticPlan.frameId = sm['modelV2'].frameId
        fallback_send.semanticPlan.frameIdExtra = sm['modelV2'].frameIdExtra
        fallback_send.semanticPlan.timestampEof = sm['modelV2'].timestampEof
        fallback_send.semanticPlan.modelMonoTime = sm.logMonoTime['modelV2']
      fallback_send.semanticPlan.source = log.SemanticPlan.Source.none
      fallback_send.semanticPlan.status = log.SemanticPlan.Status.unavailable
      pm.send('semanticPlan', fallback_send)


if __name__ == "__main__":
  main()
