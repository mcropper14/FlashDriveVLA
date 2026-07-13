#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np

from openpilot.selfdrive.alpamayo.local_adapter import LocalAlpamayoAdapter
from openpilot.selfdrive.alpamayo.pc_endpoint import AlpamayoPcEndpoint, PcTraceLogger
from openpilot.selfdrive.alpamayo.protocol import decode_payload, encode_payload


def _synthetic_bgr(width: int, height: int, frame_idx: int, camera_idx: int) -> np.ndarray:
  y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
  x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
  b = np.broadcast_to((x + frame_idx * 7 + camera_idx * 17).astype(np.uint8), (height, width))
  g = np.broadcast_to((y + frame_idx * 5 + camera_idx * 11).astype(np.uint8), (height, width))
  r = np.full((height, width), 80 + camera_idx * 40, dtype=np.uint8)
  return np.stack([b, g, r], axis=2)


def _jpeg_bgr_frame(stream: str, width: int, height: int, frame_idx: int, camera_idx: int) -> dict:
  import cv2

  bgr = _synthetic_bgr(width, height, frame_idx, camera_idx)
  ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
  if not ok:
    raise RuntimeError("failed to encode synthetic frame")
  timestamp_ns = int((frame_idx + 1) * 100_000_000)
  return {
    "stream": stream,
    "encoding": "jpeg_bgr",
    "frameId": frame_idx,
    "timestampSof": timestamp_ns - 50_000_000,
    "timestampEof": timestamp_ns,
    "width": width,
    "height": height,
    "dataBase64": base64.b64encode(encoded.tobytes()).decode("ascii"),
  }


def _request(width: int, height: int, num_frames: int) -> dict:
  frames = []
  for camera_idx, stream in enumerate(("wideRoad", "road")):
    for frame_idx in range(num_frames):
      frames.append(_jpeg_bgr_frame(stream, width, height, frame_idx, camera_idx))

  t = np.arange(16, dtype=np.float32) * 0.1
  ego_xyz = np.column_stack([t * 2.0, np.zeros_like(t), np.zeros_like(t)]).astype(np.float32)
  ego_rot = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
  return {
    "protocolVersion": 1,
    "sentMonoTime": time.monotonic_ns(),
    "modelMonoTime": time.monotonic_ns(),
    "semanticPlanHz": 10,
    "cameraBundle": {
      "streamOrder": ["wideRoad", "road"],
      "framesPerCamera": num_frames,
      "frameStepNs": 100_000_000,
    },
    "frames": frames,
    "frameSkewMs": 0.0,
    "frameT0Ns": int(num_frames * 100_000_000),
    "egoHistory": {
      "dtS": 0.1,
      "xyz": ego_xyz.tolist(),
      "rot": ego_rot.tolist(),
    },
    "runtimeConfig": {
      "cameraMode": "front2",
      "numFrames": num_frames,
      "minPixels": 65536,
      "maxPixels": 65536,
      "reasoningMode": "full",
      "diffusionSteps": 6,
      "numTrajSamples": 1,
      "transportEncoding": "jpeg_bgr",
      "transportJpegQuality": 85,
    },
    "vehicleState": {
      "vEgo": 2.0,
      "aEgo": 0.0,
      "standstill": False,
      "steeringAngleDeg": 0.0,
      "gasPressed": False,
      "brakePressed": False,
    },
    "selfdriveState": {
      "enabled": True,
      "active": True,
      "experimentalMode": False,
      "isRhd": False,
    },
    "navigation": None,
  }


def _response_summary(status: int, response: dict, elapsed_ms: float) -> dict:
  semantic = response.get("semanticPlan", {})
  trajectory = semantic.get("trajectory", {})
  return {
    "status_code": status,
    "elapsed_ms": round(elapsed_ms, 3),
    "semantic_status": semantic.get("status"),
    "semantic_source": semantic.get("source"),
    "desired_curvature": semantic.get("desiredCurvature"),
    "desired_acceleration": semantic.get("desiredAcceleration"),
    "trajectory_lengths": {
      key: len(value.get("t", [])) if isinstance(value, dict) else 0
      for key, value in trajectory.items()
    },
    "debug": semantic.get("debug", {}),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Run local Alpamayo adapter through the strict PC endpoint contract.")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--trace-path", type=Path)
  parser.add_argument("--width", type=int, default=512)
  parser.add_argument("--height", type=int, default=384)
  parser.add_argument("--num-frames", type=int, default=2)
  args = parser.parse_args()

  endpoint = AlpamayoPcEndpoint(LocalAlpamayoAdapter(), PcTraceLogger(args.trace_path))
  request = _request(args.width, args.height, args.num_frames)
  start = time.perf_counter()
  status, body, content_type = endpoint.handle_payload(encode_payload(request))
  elapsed_ms = (time.perf_counter() - start) * 1000.0
  response = decode_payload(body)
  report = {
    "created_at_unix": time.time(),
    "content_type": content_type,
    "request": {
      "frame_count": len(request["frames"]),
      "width": args.width,
      "height": args.height,
      "num_frames": args.num_frames,
      "ego_history_shape": [len(request["egoHistory"]["xyz"]), len(request["egoHistory"]["xyz"][0])],
    },
    "response_summary": _response_summary(status, response, elapsed_ms),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()

