#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import requests
import zstandard as zstd


FRAME_DT_S = 0.05
MODEL_STEP_SEC = 0.1
HISTORY_STEPS = 16
STREAMS = ("wideRoad", "road")


def wrap_angle(angle: float) -> float:
  return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def percentile(values: list[float], pct: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return ordered[idx]


def speed_mps(env) -> float:
  return float(np.linalg.norm(np.asarray(env.vehicle.velocity, dtype=np.float32)))


def route_lane_for_vehicle(env):
  vehicle = env.vehicle
  nav = getattr(vehicle, "navigation", None)
  ref_lanes = getattr(nav, "current_ref_lanes", None)
  if ref_lanes:
    if getattr(vehicle, "lane", None) in ref_lanes:
      return vehicle.lane
    lane_id = getattr(getattr(vehicle, "lane", None), "index", (None, None, 0))[2]
    if isinstance(lane_id, int) and 0 <= lane_id < len(ref_lanes):
      return ref_lanes[lane_id]
    return ref_lanes[0]
  return vehicle.lane


def next_route_lane(env, current_lane):
  nav = getattr(env.vehicle, "navigation", None)
  next_lanes = getattr(nav, "next_ref_lanes", None)
  if not next_lanes:
    return None
  lane_id = getattr(current_lane, "index", (None, None, 0))[2]
  if isinstance(lane_id, int) and 0 <= lane_id < len(next_lanes):
    return next_lanes[lane_id]
  return next_lanes[0]


def route_world_point(env, ahead_m: float, lateral_offset_m: float) -> np.ndarray:
  lane = route_lane_for_vehicle(env)
  long_m, _ = lane.local_coordinates(env.vehicle.position)
  target_long = float(long_m) + float(ahead_m)
  target_lane = lane
  if target_long > lane.length:
    overflow = target_long - lane.length
    next_lane = next_route_lane(env, lane)
    if next_lane is not None:
      target_lane = next_lane
      target_long = min(max(0.0, overflow), next_lane.length)
    else:
      target_long = lane.length
  else:
    target_long = min(max(0.0, target_long), lane.length)

  half_width = max(0.1, float(target_lane.width_at(target_long)) * 0.5 - 0.45)
  lateral = float(np.clip(lateral_offset_m, -half_width, half_width))
  return np.asarray(target_lane.position(target_long, lateral), dtype=np.float32)


def _slew(prev: float, target: float, max_delta: float) -> float:
  return float(prev + np.clip(target - prev, -abs(max_delta), abs(max_delta)))


class MetaDriveRouteFollower:
  def __init__(self, max_steer: float = 0.75, max_steer_rate_per_s: float = 0.9, steer_smoothing_alpha: float = 0.35):
    self.max_steer = float(max_steer)
    self.max_steer_rate_per_s = float(max_steer_rate_per_s)
    self.steer_smoothing_alpha = float(steer_smoothing_alpha)
    self.last_steer = 0.0

  def action(self, env, target_speed_mps: float, lateral_offset_m: float, dt: float) -> tuple[float, float, dict[str, float]]:
    vehicle = env.vehicle
    speed = speed_mps(env)
    lane = route_lane_for_vehicle(env)
    long_m, current_lateral_m = lane.local_coordinates(vehicle.position)
    lookahead_m = float(np.clip(6.0 + speed * 1.65, 8.0, 28.0))
    target = route_world_point(env, lookahead_m, lateral_offset_m)
    dx = float(target[0] - vehicle.position[0])
    dy = float(target[1] - vehicle.position[1])
    desired_heading = math.atan2(dy, dx)
    heading_error = wrap_angle(desired_heading - float(vehicle.heading_theta))
    lateral_error = float(current_lateral_m - lateral_offset_m)

    raw_steer = float(np.clip(1.45 * heading_error + 0.04 * lateral_error, -self.max_steer, self.max_steer))
    filtered_steer = self.last_steer + self.steer_smoothing_alpha * (raw_steer - self.last_steer)
    steer = _slew(self.last_steer, filtered_steer, self.max_steer_rate_per_s * max(dt, 1e-3))
    steer = float(np.clip(steer, -self.max_steer, self.max_steer))
    self.last_steer = steer

    speed_error = float(target_speed_mps) - speed
    speed_gain = 0.45 if target_speed_mps <= 0.5 and speed_error < 0.0 else 0.16
    gas = float(np.clip(speed_gain * speed_error, -0.65, 0.55))
    return steer, gas, {
      "route_longitudinal_m": float(long_m),
      "route_lateral_m": float(current_lateral_m),
      "target_lateral_m": float(lateral_offset_m),
      "lookahead_m": lookahead_m,
      "heading_error_rad": float(heading_error),
      "raw_steer": raw_steer,
    }


class AlpamayoTrajectoryController:
  def __init__(self, max_steer: float = 0.75, max_steer_rate_per_s: float = 0.9):
    self.max_steer = float(max_steer)
    self.max_steer_rate_per_s = float(max_steer_rate_per_s)
    self.last_steer = 0.0

  def action(self, semantic_plan: dict[str, Any], current_speed_mps: float, dt: float) -> tuple[float, float, dict[str, float]]:
    trajectory = semantic_plan.get("trajectory", {})
    position = trajectory.get("position", {})
    velocity = trajectory.get("velocity", {})
    xs = np.asarray(position.get("x", []), dtype=np.float32)
    ys = np.asarray(position.get("y", []), dtype=np.float32)
    vx = np.asarray(velocity.get("x", []), dtype=np.float32)
    vy = np.asarray(velocity.get("y", []), dtype=np.float32)
    if len(xs) < 2 or len(xs) != len(ys):
      raise ValueError("Alpamayo trajectory.position is missing or malformed")

    forward = np.maximum(xs, 0.1)
    valid = np.where(forward >= 6.0)[0]
    idx = int(valid[0]) if len(valid) else min(len(xs) - 1, max(1, len(xs) // 4))
    heading_error = float(math.atan2(float(ys[idx]), max(float(forward[idx]), 0.1)))
    raw_steer = float(np.clip(1.65 * heading_error, -self.max_steer, self.max_steer))
    steer = _slew(self.last_steer, raw_steer, self.max_steer_rate_per_s * max(dt, 1e-3))
    steer = float(np.clip(steer, -self.max_steer, self.max_steer))
    self.last_steer = steer

    if len(vx) == len(xs) and len(vy) == len(xs):
      target_speed = float(np.clip(math.hypot(float(vx[idx]), float(vy[idx])), 0.0, 18.0))
    else:
      target_speed = float(current_speed_mps)
    desired_accel = float(semantic_plan.get("desiredAcceleration", 0.0) or 0.0)
    gas = float(np.clip(0.13 * (target_speed - current_speed_mps) + 0.10 * desired_accel, -0.65, 0.55))
    return steer, gas, {
      "alpamayo_lookahead_index": float(idx),
      "alpamayo_lookahead_x_m": float(xs[idx]),
      "alpamayo_lookahead_y_m": float(ys[idx]),
      "alpamayo_heading_error_rad": heading_error,
      "alpamayo_target_speed_mps": target_speed,
      "alpamayo_desired_accel_mps2": desired_accel,
      "alpamayo_raw_steer": raw_steer,
    }


@dataclass
class EgoSample:
  position: np.ndarray
  heading: float


class EgoHistory:
  def __init__(self, steps: int = HISTORY_STEPS, sample_dt_s: float = MODEL_STEP_SEC):
    self.steps = int(steps)
    self.sample_dt_s = float(sample_dt_s)
    self.samples: deque[EgoSample] = deque(maxlen=self.steps)
    self.last_sample_s: float | None = None

  def append_from_env(self, env) -> None:
    pos = np.asarray([float(env.vehicle.position[0]), float(env.vehicle.position[1]), 0.0], dtype=np.float32)
    self.samples.append(EgoSample(pos, float(env.vehicle.heading_theta)))

  def append_from_env_if_due(self, env, timestamp_s: float) -> None:
    if self.last_sample_s is None or timestamp_s - self.last_sample_s >= self.sample_dt_s - 1e-9:
      self.append_from_env(env)
      self.last_sample_s = timestamp_s

  def has_sufficient_samples(self, allow_backfill: bool = False) -> bool:
    return bool(self.samples) and (allow_backfill or len(self.samples) >= self.steps)

  def payload(self, allow_backfill: bool = False) -> dict[str, Any]:
    if not self.samples:
      raise ValueError("ego history is empty")
    samples = list(self.samples)
    if len(samples) < self.steps:
      if not allow_backfill:
        raise ValueError("ego history is not fully collected yet")
    while len(samples) < self.steps:
      samples.insert(0, samples[0])
    samples = samples[-self.steps:]

    current = samples[-1]
    cos_h = math.cos(current.heading)
    sin_h = math.sin(current.heading)
    xyz = []
    rot = []
    for sample in samples:
      delta = sample.position - current.position
      forward = float(delta[0] * cos_h + delta[1] * sin_h)
      left = float(-delta[0] * sin_h + delta[1] * cos_h)
      yaw = wrap_angle(sample.heading - current.heading)
      cy = math.cos(yaw)
      sy = math.sin(yaw)
      xyz.append([forward, left, float(delta[2])])
      rot.append([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
      ])
    return {"xyz": xyz, "rot": rot}


def make_env(args: argparse.Namespace):
  from panda3d.core import GraphicsOutput, Texture, Vec3
  from metadrive.component.sensors.rgb_camera import RGBCamera
  from metadrive.envs.metadrive_env import MetaDriveEnv

  class CopyRamRGBCamera(RGBCamera):
    def __init__(self, *cam_args, **cam_kwargs):
      super().__init__(*cam_args, **cam_kwargs)
      self.cpu_texture = Texture()
      self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

    def get_rgb_array_cpu(self) -> np.ndarray:
      origin_img = self.cpu_texture
      img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
      img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
      img = img[:, :, :3]
      return img[::-1].copy()

  class RGBCameraWide(CopyRamRGBCamera):
    def __init__(self, *cam_args, **cam_kwargs):
      super().__init__(*cam_args, **cam_kwargs)
      lens = self.get_lens()
      lens.setFov(120)
      lens.setNear(0.1)

  class RGBCameraRoad(CopyRamRGBCamera):
    def __init__(self, *cam_args, **cam_kwargs):
      super().__init__(*cam_args, **cam_kwargs)
      lens = self.get_lens()
      lens.setFov(40)
      lens.setNear(0.1)

  map_arg: int | str = int(args.map) if str(args.map).isdigit() else str(args.map)
  config = {
    "use_render": False,
    "image_observation": True,
    "sensors": {
      "rgb_wide": (RGBCameraWide, args.camera_width, args.camera_height),
      "rgb_road": (RGBCameraRoad, args.camera_width, args.camera_height),
    },
    "vehicle_config": {
      "enable_reverse": False,
      "image_source": "rgb_road",
    },
    "interface_panel": [],
    "traffic_density": 0.0,
    "manual_control": False,
    "show_logo": False,
    "show_fps": False,
    "start_seed": args.seed,
    "num_scenarios": 1,
    "map": map_arg,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "out_of_route_done": False,
    "on_continuous_line_done": False,
    "decision_repeat": 1,
    "physics_world_step_size": args.tick_sec,
    "preload_models": False,
  }
  env = MetaDriveEnv(config)
  env.reset(seed=args.seed)
  for sensor_name in ("rgb_wide", "rgb_road"):
    cam = env.engine.sensors[sensor_name]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(Vec3(0.0, 0.0, 1.22))
    cam.get_cam().setHpr(Vec3(0.0, 0.0, 0.0))
  return env


def camera_rgb(env, sensor_name: str) -> np.ndarray:
  cam = env.engine.sensors[sensor_name]
  if hasattr(cam, "get_rgb_array_cpu"):
    frame = cam.get_rgb_array_cpu()
  else:
    frame = cam.perceive(to_float=False)
    if not isinstance(frame, np.ndarray):
      frame = frame.get()
    frame = frame[:, :, :3][:, :, ::-1]
  return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def jpeg_b64_from_rgb(frame_rgb: np.ndarray, quality: int) -> str:
  buf = BytesIO()
  Image.fromarray(frame_rgb).save(buf, format="JPEG", quality=int(quality))
  return base64.b64encode(buf.getvalue()).decode("ascii")


def capture_frames(env, frame_id: int, timestamp_eof_ns: int, quality: int, tick_sec: float) -> dict[str, dict[str, Any]]:
  wide_rgb = camera_rgb(env, "rgb_wide")
  road_rgb = camera_rgb(env, "rgb_road")
  height, width = road_rgb.shape[:2]
  timestamp_sof_ns = max(0, timestamp_eof_ns - int(tick_sec * 1e9))
  return {
    "wideRoad": {
      "stream": "wideRoad",
      "encoding": "jpeg_bgr",
      "frameId": frame_id,
      "timestampSof": timestamp_sof_ns,
      "timestampEof": timestamp_eof_ns,
      "width": width,
      "height": height,
      "dataBase64": jpeg_b64_from_rgb(wide_rgb, quality),
      "_rgb": wide_rgb,
    },
    "road": {
      "stream": "road",
      "encoding": "jpeg_bgr",
      "frameId": frame_id,
      "timestampSof": timestamp_sof_ns,
      "timestampEof": timestamp_eof_ns,
      "width": width,
      "height": height,
      "dataBase64": jpeg_b64_from_rgb(road_rgb, quality),
      "_rgb": road_rgb,
    },
  }


def frame_for_payload(frame: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in frame.items() if not key.startswith("_")}


def select_nearest_frames(
  stream_frames: list[dict[str, Any]],
  target_times_ns: list[int],
) -> list[dict[str, Any]] | None:
  selected: list[dict[str, Any]] = []
  for target_time_ns in target_times_ns:
    candidates = list(stream_frames)
    if not candidates:
      return None
    best = min(candidates, key=lambda frame: abs(int(frame["timestampEof"]) - target_time_ns))
    selected.append(best)
  selected.sort(key=lambda frame: int(frame["timestampEof"]))
  return selected


def build_request(
  frame_buffers: dict[str, deque[dict[str, Any]]],
  ego_history: EgoHistory,
  vehicle_state: dict[str, Any],
  frame_id: int,
  frame_step_ns: int,
  num_frames: int,
  allow_backfill: bool,
  target_t0_ns: int | None = None,
) -> tuple[dict[str, Any], int, int] | None:
  if not all(len(frame_buffers[stream]) for stream in STREAMS):
    return None
  latest_t0_ns = target_t0_ns
  if latest_t0_ns is None:
    latest_t0_ns = min(int(frame_buffers[name][-1]["timestampEof"]) for name in STREAMS)
  target_times_ns = [latest_t0_ns - (num_frames - 1 - idx) * frame_step_ns for idx in range(num_frames)]
  frames: list[dict[str, Any]] = []
  selected_frame_ids: set[int] = set()
  for stream in STREAMS:
    picked = select_nearest_frames(list(frame_buffers[stream]), target_times_ns)
    if picked is None:
      return None
    selected_frame_ids.update(int(frame["frameId"]) for frame in picked)
    frames.extend(frame_for_payload(frame) for frame in picked)

  if not frames:
    return None

  try:
    history_payload = ego_history.payload(allow_backfill=allow_backfill)
  except ValueError:
    return None

  request_frame_id = max(selected_frame_ids) if selected_frame_ids else frame_id
  return ({
    "protocolVersion": 1,
    "requestId": f"metadrive-{frame_id:06d}",
    "createdMonoTime": int(time.monotonic_ns()),
    "semanticPlanHz": 10,
    "cameraBundle": {
      "streamOrder": list(STREAMS),
      "framesPerCamera": num_frames,
      "frameStepNs": frame_step_ns,
    },
    "vehicleState": vehicle_state,
    "frames": frames,
    "egoHistory": history_payload,
    "navigation": {"text": "Follow the lane and continue safely."},
    "runtimeConfig": {
      "reasoningMode": "full",
      "transportEncoding": "jpeg_bgr",
      "cameraStreams": list(STREAMS),
    },
  }, request_frame_id, int(latest_t0_ns))


def decode_endpoint_body(body: bytes) -> dict[str, Any]:
  if body[:1] in (b"{", b"["):
    return json.loads(body.decode("utf-8"))
  return json.loads(zstd.ZstdDecompressor().decompress(body).decode("utf-8"))


def post_alpamayo(endpoint_url: str, request_payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any], float]:
  start = time.perf_counter()
  response = requests.post(
    endpoint_url,
    data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    timeout=timeout_s,
  )
  elapsed_ms = (time.perf_counter() - start) * 1000.0
  payload = decode_endpoint_body(response.content)
  return int(response.status_code), payload, elapsed_ms


def semantic_status(payload: dict[str, Any]) -> str:
  semantic = payload.get("semanticPlan", {})
  return str(semantic.get("status", "missing"))


def semantic_valid(payload: dict[str, Any]) -> bool:
  semantic = payload.get("semanticPlan", {})
  trajectory = semantic.get("trajectory", {})
  position = trajectory.get("position", {})
  return semantic.get("status") in ("valid", 1) and len(position.get("x", [])) >= 2


def draw_label_block(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
  import cv2

  out = np.ascontiguousarray(frame_rgb.copy())
  font = cv2.FONT_HERSHEY_SIMPLEX
  frame_w, frame_h = out.shape[1], out.shape[0]
  pad = max(6, round(frame_w * 0.015))
  max_box_w = frame_w - pad * 2
  # Leave most of the frame visible -- the debug panel is a reference
  # overlay, not the point of the shot.
  max_box_h = round(frame_h * 0.55)

  # Pick the largest scale (within a sane range) whose widest line still
  # fits max_box_w, instead of a fixed scale that silently clips long
  # lines (e.g. the wrapped "reason ..." text) off the edge of the frame.
  thickness = 1
  scale = max(0.32, min(0.62, frame_w / 820.0))
  for _ in range(6):
    width = max((cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines), default=0)
    if width + pad * 2 <= max_box_w or scale <= 0.30:
      break
    scale = round(scale * 0.9, 3)

  line_h = round(scale * 34) + 4
  visible_lines = lines
  box_h_needed = pad * 2 + line_h * len(lines)
  if box_h_needed > max_box_h and line_h > 0:
    max_lines = max(1, (max_box_h - pad * 2) // line_h)
    visible_lines = lines[:max_lines]

  width = max((cv2.getTextSize(line, font, scale, thickness)[0][0] for line in visible_lines), default=0)
  box_w = min(max_box_w, width + pad * 2)
  box_h = min(max_box_h, pad * 2 + line_h * len(visible_lines))
  overlay = out.copy()
  cv2.rectangle(overlay, (0, 0), (box_w, box_h), (16, 18, 22), -1)
  out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
  for idx, line in enumerate(visible_lines):
    y = pad + round(line_h * 0.72) + idx * line_h
    cv2.putText(out, line, (pad, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
  return out


def write_side_by_side_video(stock_frames: list[np.ndarray], alpamayo_frames: list[np.ndarray], output_path: Path, fps: float) -> None:
  import cv2

  if not stock_frames or not alpamayo_frames:
    raise ValueError("both stock and Alpamayo frames are required for side-by-side video")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  frame_count = min(len(stock_frames), len(alpamayo_frames))
  h = min(stock_frames[0].shape[0], alpamayo_frames[0].shape[0])
  w = min(stock_frames[0].shape[1], alpamayo_frames[0].shape[1])
  writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    float(fps),
    (w * 2, h),
  )
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer: {output_path}")
  try:
    for idx in range(frame_count):
      left = stock_frames[idx]
      right = alpamayo_frames[idx]
      if left.shape[:2] != (h, w):
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
      if right.shape[:2] != (h, w):
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
      combined_rgb = np.concatenate([left[:, :, :3], right[:, :, :3]], axis=1)
      writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
  finally:
    writer.release()


def write_single_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
  import cv2

  if not frames:
    raise ValueError("at least one frame is required for single-episode video")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  h, w = frames[0].shape[:2]
  writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    float(fps),
    (w, h),
  )
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer: {output_path}")
  try:
    for frame in frames:
      frame_rgb = frame
      if frame_rgb.shape[:2] != (h, w):
        frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
      writer.write(cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2BGR))
  finally:
    writer.release()


def run_episode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
  env = make_env(args)
  route_controller = MetaDriveRouteFollower(args.max_steer, args.max_steer_rate_per_s, args.steer_smoothing_alpha)
  alpamayo_controller = AlpamayoTrajectoryController(args.max_steer, args.max_steer_rate_per_s)
  sample_dt_s = max(float(args.model_step_sec), 1e-6)
  query_interval_sec = sample_dt_s * max(1, args.query_every)
  frame_step_ns = int(sample_dt_s * 1e9)
  frame_buffer_size = max(
    args.num_frames + 4,
    int(math.ceil((sample_dt_s * max(1, args.num_frames - 1)) / max(args.tick_sec, 1e-6)) + 4),
    args.warmup_stock_frames,
  )
  if args.async_endpoint and mode == "alpamayo":
    frame_buffer_size = max(frame_buffer_size, args.frames + args.num_frames + 4)
  ego_history = EgoHistory(HISTORY_STEPS, sample_dt_s=sample_dt_s)
  frame_buffers = {stream: deque(maxlen=frame_buffer_size) for stream in STREAMS}
  records: list[dict[str, Any]] = []
  video_frames: list[np.ndarray] = []
  endpoint_latencies: list[float] = []
  endpoint_errors: list[str] = []
  endpoint_attempts: int = 0
  endpoint_successes: int = 0
  latest_plan_frame_id: int | None = None
  latest_plan_latency_ms: float | None = None
  last_semantic: dict[str, Any] | None = None
  last_response_payload: dict[str, Any] | None = None
  latest_response_status: str = ""
  latest_response_error: str = ""
  latest_endpoint_status_code: int | None = None
  latest_endpoint_latency_ms: float | None = None
  endpoint_executor: concurrent.futures.ThreadPoolExecutor | None = None
  endpoint_future: concurrent.futures.Future[tuple[int, dict[str, Any], float]] | None = None
  endpoint_request_frame_id: int | None = None
  last_request_t0_ns: int | None = None
  prev_speed: float | None = None
  next_query_time_s = args.tick_sec * max(0, args.warmup_stock_frames)
  terminated = False
  truncated = False
  episode_wall_start = time.perf_counter()
  captured_request: dict[str, Any] | None = None

  if args.async_endpoint and mode == "alpamayo":
    endpoint_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

  def consume_endpoint_result(
    status_code: int,
    payload: dict[str, Any],
    latency_ms: float,
    request_frame_id_for_plan: int | None,
  ) -> None:
    nonlocal latest_endpoint_status_code, latest_endpoint_latency_ms, latest_response_status, latest_response_error
    nonlocal latest_plan_frame_id, latest_plan_latency_ms, last_semantic, last_response_payload
    nonlocal endpoint_successes
    latest_endpoint_status_code = status_code
    latest_endpoint_latency_ms = latency_ms
    latest_response_status = semantic_status(payload)
    last_response_payload = payload
    endpoint_latencies.append(latency_ms)
    endpoint_successes += 1
    if semantic_valid(payload):
      latest_response_error = ""
      last_semantic = payload.get("semanticPlan", {})
      if request_frame_id_for_plan is not None:
        latest_plan_frame_id = request_frame_id_for_plan
        latest_plan_latency_ms = latency_ms
    else:
      latest_response_error = str(payload.get("semanticPlan", {}).get("error", "invalid semanticPlan"))
      endpoint_errors.append(latest_response_error)

  try:
    for frame_id in range(args.frames):
      frame_start = time.perf_counter()
      timestamp_s = float(frame_id) * args.tick_sec
      timestamp_eof_ns = int(round(timestamp_s * 1e9))
      if mode == "alpamayo":
        ego_history.append_from_env_if_due(env, timestamp_s)

      current_speed_before_step = speed_mps(env)
      a_ego = 0.0 if prev_speed is None else (current_speed_before_step - prev_speed) / max(args.tick_sec, 1e-3)
      vehicle_state = {
        "vEgo": current_speed_before_step,
        "aEgo": a_ego,
        "standstill": current_speed_before_step < 0.01,
        "steeringAngleDeg": float(getattr(env.vehicle, "steering", 0.0) * getattr(env.vehicle, "MAX_STEERING", 1.0)),
      }
      captured = capture_frames(env, frame_id, timestamp_eof_ns, args.jpeg_quality, args.tick_sec)
      for stream, frame in captured.items():
        frame_buffers[stream].append(frame)

      in_flight = endpoint_future is not None and not endpoint_future.done()
      request_frame_id = None

      if endpoint_future is not None and endpoint_future.done():
        completed_request_frame_id = endpoint_request_frame_id
        endpoint_request_frame_id = None
        try:
          status_code, payload, latency_ms = endpoint_future.result()
          consume_endpoint_result(status_code, payload, latency_ms, completed_request_frame_id)
        except Exception as exc:
          latest_response_status = "error"
          latest_response_error = f"{type(exc).__name__}: {exc}"
          endpoint_errors.append(latest_response_error)
          latest_endpoint_status_code = None
          latest_endpoint_latency_ms = None
        endpoint_future = None
        in_flight = False

      can_query = (
        mode == "alpamayo"
        and frame_id >= args.warmup_stock_frames
        and (timestamp_s + 1e-9) >= next_query_time_s
        and not in_flight
      )

      if can_query and ego_history.has_sufficient_samples(allow_backfill=args.backfill_ego_history):
        latest_available_t0_ns = min(int(frame_buffers[name][-1]["timestampEof"]) for name in STREAMS)
        target_t0_ns = None
        if args.async_endpoint and last_request_t0_ns is not None and last_request_t0_ns < latest_available_t0_ns:
          target_t0_ns = min(
            latest_available_t0_ns,
            last_request_t0_ns + frame_step_ns * max(1, int(args.catchup_stride_steps)),
          )
        request_pair = build_request(
          frame_buffers,
          ego_history,
          vehicle_state,
          frame_id,
          frame_step_ns=frame_step_ns,
          num_frames=args.num_frames,
          allow_backfill=args.backfill_ego_history,
          target_t0_ns=target_t0_ns,
        )
        if request_pair is not None:
          request_payload, request_frame_id, request_t0_ns = request_pair
          last_request_t0_ns = request_t0_ns
          endpoint_attempts += 1
          if args.capture_request_output is not None and captured_request is None:
            match_request = args.capture_request_frame_id is None or (
              request_frame_id == args.capture_request_frame_id or frame_id == args.capture_request_frame_id
            )
            if match_request:
              captured_request = {
                "metadata": {
                  "sim_frame_id": frame_id,
                  "request_frame_id": request_frame_id,
                  "mode": mode,
                  "model_step_sec": args.model_step_sec,
                  "tick_sec": args.tick_sec,
                  "warmup_stock_frames": args.warmup_stock_frames,
                },
                "request_payload": request_payload,
              }
          next_query_time_s += query_interval_sec
          if args.async_endpoint:
            endpoint_future = endpoint_executor.submit(post_alpamayo, args.endpoint_url, request_payload, args.endpoint_timeout_s)
            endpoint_request_frame_id = request_frame_id
            in_flight = True
          else:
            try:
              status_code, payload, latency_ms = post_alpamayo(args.endpoint_url, request_payload, args.endpoint_timeout_s)
              consume_endpoint_result(status_code, payload, latency_ms, request_frame_id)
            except Exception as exc:
              latest_response_status = "error"
              latest_response_error = f"{type(exc).__name__}: {exc}"
              endpoint_errors.append(latest_response_error)
              latest_endpoint_status_code = None
              latest_endpoint_latency_ms = None

      stock_steer, stock_gas, stock_debug = route_controller.action(env, args.speed_mps, 0.0, args.tick_sec)
      steer = stock_steer
      gas = stock_gas
      control_source = "stock_route_follower"
      alpamayo_debug: dict[str, float] = {}
      if mode == "alpamayo" and last_semantic is not None:
        try:
          steer, gas, alpamayo_debug = alpamayo_controller.action(last_semantic, speed_mps(env), args.tick_sec)
          control_source = "alpamayo_trajectory"
        except Exception as exc:
          latest_response_error = latest_response_error or f"Alpamayo control decode failed: {exc}"
          endpoint_errors.append(latest_response_error)

      endpoint_status_code = latest_endpoint_status_code
      endpoint_latency_ms = latest_endpoint_latency_ms
      response_status = latest_response_status
      response_error = latest_response_error
      latest_plan_age_frames = frame_id - latest_plan_frame_id if latest_plan_frame_id is not None else None

      step_start = time.perf_counter()
      _, reward, terminated, truncated, info = env.step([steer, gas])
      sim_step_ms = (time.perf_counter() - step_start) * 1000.0
      lane = route_lane_for_vehicle(env)
      route_long_m, route_lateral_m = lane.local_coordinates(env.vehicle.position)
      route_long_m = float(route_long_m)
      route_lateral_m = float(route_lateral_m)
      current_speed = speed_mps(env)
      prev_speed = current_speed
      semantic = last_semantic or {}
      trajectory = semantic.get("trajectory", {})
      position = trajectory.get("position", {})
      record = {
        "frame_id": frame_id,
        "mode": mode,
        "control_source": control_source,
        "speed_mps": current_speed,
        "route_longitudinal_m": route_long_m,
        "route_lateral_m": route_lateral_m,
        "steer": float(steer),
        "gas": float(gas),
        "stock_steer": float(stock_steer),
        "stock_gas": float(stock_gas),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "sim_step_ms": sim_step_ms,
        "frame_wall_ms": (time.perf_counter() - frame_start) * 1000.0,
        "endpoint_status_code": endpoint_status_code,
        "endpoint_latency_ms": endpoint_latency_ms,
        "semantic_status": response_status,
        "semantic_error": response_error,
        "semantic_source": semantic.get("source", ""),
        "trajectory_len": len(position.get("x", [])),
        "desired_curvature": semantic.get("desiredCurvature"),
        "desired_acceleration": semantic.get("desiredAcceleration"),
        "cot_preview": str(semantic.get("cot", ""))[:160],
        "stock_debug": stock_debug,
        "alpamayo_debug": alpamayo_debug,
        "model_step_sec": sample_dt_s,
        "async_endpoint": bool(args.async_endpoint and mode == "alpamayo"),
      "warmup_stock_frames": args.warmup_stock_frames,
      "frame_buffer_size": frame_buffer_size,
      "request_frame_id": request_frame_id,
        "latest_plan_frame_id": latest_plan_frame_id,
        "latest_plan_age_frames": latest_plan_age_frames,
        "latest_plan_latency_ms": latest_plan_latency_ms,
        "in_flight": endpoint_future is not None and not endpoint_future.done(),
        "realtime": bool(args.realtime),
      }
      records.append(record)
      if not args.no_video:
        video_lines = [
          f"{mode.upper()} frame {frame_id}",
          f"src {control_source}",
          f"v {current_speed:.2f} m/s lat {route_lateral_m:.2f} m",
          f"steer {float(steer):+.3f} gas {float(gas):+.3f}",
        ]
        if mode == "alpamayo":
          if response_status:
            video_lines.append(f"endpoint {endpoint_status_code or '-'} {response_status}")
          if endpoint_latency_ms is not None:
            video_lines[-1] += f" {endpoint_latency_ms:.0f}ms"
          if response_error:
            video_lines.append(response_error[:54])
          if latest_plan_frame_id is not None:
            video_lines.append(f"plan id {latest_plan_frame_id} age {latest_plan_age_frames} lat {latest_plan_latency_ms or 0:.0f}ms")
          if last_semantic is not None:
            video_lines.append(str(last_semantic.get("cot", ""))[:54])
        video_frames.append(draw_label_block(captured["road"]["_rgb"], video_lines))
      if terminated or truncated:
        break
      if args.realtime:
        target_elapsed = ((frame_id + 1) * args.tick_sec) / max(args.realtime_speed, 1e-6)
        sleep_s = episode_wall_start + target_elapsed - time.perf_counter()
        if sleep_s > 0.0:
          time.sleep(sleep_s)
  finally:
    env.close()
    if endpoint_future is not None and not endpoint_future.done():
      endpoint_future.cancel()
    if endpoint_executor is not None:
      endpoint_executor.shutdown(wait=False, cancel_futures=True)

  lane_abs = [abs(float(r["route_lateral_m"])) for r in records]
  speeds = [float(r["speed_mps"]) for r in records]
  route_distance = 0.0
  if records:
    route_distance = float(records[-1]["route_longitudinal_m"] - records[0]["route_longitudinal_m"])
  valid_responses = sum(1 for r in records if r.get("semantic_status") in ("valid", 1))
  return {
    "mode": mode,
    "frames": len(records),
    "endpoint_calls": endpoint_successes,
    "endpoint_attempts": endpoint_attempts,
    "valid_endpoint_responses": valid_responses,
    "endpoint_error_count": len(endpoint_errors),
    "endpoint_errors": endpoint_errors[:10],
    "endpoint_latency_ms": {
      "mean": statistics.fmean(endpoint_latencies) if endpoint_latencies else None,
      "p95": percentile(endpoint_latencies, 95),
      "p99": percentile(endpoint_latencies, 99),
      "max": max(endpoint_latencies) if endpoint_latencies else None,
    },
    "mean_speed_mps": statistics.fmean(speeds) if speeds else 0.0,
    "final_speed_mps": speeds[-1] if speeds else 0.0,
    "route_distance_m": route_distance,
    "max_abs_route_lateral_m": max(lane_abs) if lane_abs else 0.0,
    "terminated": bool(terminated),
    "truncated": bool(truncated),
    "records": records,
    "last_response_payload": last_response_payload,
    "captured_request_payload": captured_request,
    "_video_frames": video_frames,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Benchmark local Alpamayo contract inference on real MetaDrive camera frames.")
  parser.add_argument("--mode", choices=("stock", "alpamayo", "both"), default="both")
  parser.add_argument("--endpoint-url", default="http://127.0.0.1:8765/alpamayo")
  parser.add_argument("--endpoint-timeout-s", type=float, default=240.0)
  parser.add_argument("--output", type=Path, default=Path("artifacts/alpamayo_speed/alpamayo_metadrive_contract_benchmark.json"))
  parser.add_argument("--capture-request-output", type=Path, help="Optional path to write the captured Alpamayo request payload.")
  parser.add_argument("--capture-request-frame-id", type=int, help="Capture when request_frame_id or sim frame_id matches this value.")
  parser.add_argument("--frames", type=int, default=4)
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--query-every", type=int, default=1)
  parser.add_argument("--catchup-stride-steps", type=int, default=1)
  parser.add_argument("--speed-mps", type=float, default=4.0)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--map", default="S")
  parser.add_argument("--model-step-sec", type=float, default=MODEL_STEP_SEC)
  parser.add_argument("--warmup-stock-frames", type=int, default=None)
  parser.add_argument("--backfill-ego-history", action="store_true", help="Allow duplicated ego history points when not enough model-step samples have been collected yet.")
  endpoint_mode = parser.add_mutually_exclusive_group()
  endpoint_mode.add_argument("--async-endpoint", action="store_true", help="Enable async endpoint request mode.")
  endpoint_mode.add_argument("--sync-endpoint", action="store_true", help="Disable async mode and issue endpoint calls synchronously.")
  parser.add_argument("--camera-width", type=int, default=512)
  parser.add_argument("--camera-height", type=int, default=384)
  parser.add_argument("--jpeg-quality", type=int, default=85)
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument("--video-output", type=Path, default=Path("artifacts/alpamayo_speed/alpamayo_metadrive_side_by_side.mp4"))
  parser.add_argument("--video-fps", type=float, default=10.0)
  parser.add_argument("--tick-sec", type=float, default=FRAME_DT_S)
  parser.add_argument("--realtime", action="store_true", help="Pace the simulator loop to wall-clock time instead of running as fast as possible.")
  parser.add_argument("--realtime-speed", type=float, default=1.0, help="Wall-clock pacing multiplier used with --realtime; 1.0 means simulated time equals wall time.")
  parser.add_argument("--max-steer", type=float, default=0.75)
  parser.add_argument("--max-steer-rate-per-s", type=float, default=0.9)
  parser.add_argument("--steer-smoothing-alpha", type=float, default=0.35)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.tick_sec <= 0:
    raise ValueError("--tick-sec must be positive")
  if args.model_step_sec <= 0:
    raise ValueError("--model-step-sec must be positive")
  if args.realtime_speed <= 0:
    raise ValueError("--realtime-speed must be positive")
  if args.warmup_stock_frames is None:
    args.warmup_stock_frames = int(math.ceil(HISTORY_STEPS * args.model_step_sec / max(args.tick_sec, 1e-6)))
  if args.warmup_stock_frames < 0:
    raise ValueError("--warmup-stock-frames must be zero or greater")
  if args.query_every < 1:
    raise ValueError("--query-every must be >= 1")
  if args.catchup_stride_steps < 1:
    raise ValueError("--catchup-stride-steps must be >= 1")
  if args.mode == "stock":
    args.async_endpoint = False
  else:
    args.async_endpoint = not args.sync_endpoint
  args.output.parent.mkdir(parents=True, exist_ok=True)
  start = time.perf_counter()
  modes = ("stock", "alpamayo") if args.mode == "both" else (args.mode,)
  result = {
    "created_at_unix": time.time(),
    "config": {
      "frames": args.frames,
      "num_frames": args.num_frames,
      "query_every": args.query_every,
      "catchup_stride_steps": args.catchup_stride_steps,
      "model_step_sec": args.model_step_sec,
      "async_endpoint": args.async_endpoint,
      "warmup_stock_frames": args.warmup_stock_frames,
      "backfill_ego_history": args.backfill_ego_history,
      "frame_buffer_size": max(
        args.num_frames + 4,
        int(math.ceil((max(float(args.model_step_sec), 1e-6) * max(1, args.num_frames - 1)) / max(args.tick_sec, 1e-6)) + 4),
        args.warmup_stock_frames,
        args.frames + args.num_frames + 4 if args.async_endpoint else 0,
      ),
      "speed_mps": args.speed_mps,
      "seed": args.seed,
      "map": args.map,
      "camera_width": args.camera_width,
      "camera_height": args.camera_height,
      "realtime": args.realtime,
      "realtime_speed": args.realtime_speed,
      "endpoint_url": args.endpoint_url if "alpamayo" in modes else "",
    },
    "episodes": {},
  }
  for mode in modes:
    episode = run_episode(args, mode)
    result["episodes"][mode] = episode
  if args.capture_request_output is not None:
    captured_request = None
    for mode in modes:
      captured_request = result["episodes"][mode].get("captured_request_payload")
      if captured_request is not None:
        break
    for mode in modes:
      result["episodes"][mode].pop("captured_request_payload", None)
    if captured_request is None:
      raise ValueError("No Alpamayo request payload was built for --capture-request-output")
    args.capture_request_output.parent.mkdir(parents=True, exist_ok=True)
    args.capture_request_output.write_text(json.dumps(captured_request, indent=2), encoding="utf-8")
  video_output = None
  if not args.no_video and "stock" in result["episodes"] and "alpamayo" in result["episodes"]:
    write_side_by_side_video(
      result["episodes"]["stock"].get("_video_frames", []),
      result["episodes"]["alpamayo"].get("_video_frames", []),
      args.video_output,
      args.video_fps,
    )
    video_output = str(args.video_output)
  elif not args.no_video and len(result["episodes"]) == 1:
    episode = next(iter(result["episodes"].values()))
    write_single_video(episode.get("_video_frames", []), args.video_output, args.video_fps)
    video_output = str(args.video_output)
  for episode in result["episodes"].values():
    episode.pop("_video_frames", None)
  result["wall_seconds"] = time.perf_counter() - start
  result["video_output"] = video_output
  args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({
    "output": str(args.output),
    "video_output": video_output,
    "wall_seconds": result["wall_seconds"],
    "episodes": {
      mode: {
        "frames": episode["frames"],
        "endpoint_calls": episode["endpoint_calls"],
        "valid_endpoint_responses": episode["valid_endpoint_responses"],
        "mean_endpoint_latency_ms": episode["endpoint_latency_ms"]["mean"],
        "route_distance_m": episode["route_distance_m"],
        "max_abs_route_lateral_m": episode["max_abs_route_lateral_m"],
      }
      for mode, episode in result["episodes"].items()
    },
  }, indent=2))


if __name__ == "__main__":
  main()
