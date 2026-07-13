from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import math
import numbers
import os
import shlex
import subprocess
import threading
import time
from typing import Optional

try:
  from openpilot.selfdrive.controls.reasoned.scene_board import SceneBoard
  from openpilot.selfdrive.controls.reasoned.rtp import parse_rtp
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.scene_board import SceneBoard
  from selfdrive.controls.reasoned.rtp import parse_rtp


FIXED_RTP_PROMPT = """You are compiling a real-time driving trajectory program.
Use only visible evidence and supplied vehicle state.
Output exactly RTPv1. No prose.
Choose a maneuver and constraints that modify the candidate path.
Never output raw steering. Never output CAN commands.
Prefer the base path unless visual evidence requires a constraint."""

DEFAULT_STATIC_RTP = """RTPv1
scene=nominal
evidence=[base_path_visible]
meta=BASE
branch=base
lat_bias_m=0.0
speed_cap_mps=none
stop_s=none
avoid=[]
weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]
confidence=0.90"""


class VlmError(RuntimeError):
  pass


class VlmTimeout(VlmError):
  pass


@dataclass(frozen=True)
class RtpEngineResult:
  text: str
  generated_token_count: int
  prefill_ms: float
  decode_ms: float
  backend: str
  source_frame_id: int | None = None
  labels: tuple[str, ...] = ()
  label_scores: dict[str, float] | None = None
  raw_labels: tuple[str, ...] = ()
  raw_label_scores: dict[str, float] | None = None
  labels_scored_this_request: tuple[str, ...] = ()
  score_group_index: int | None = None
  label_state_debug: dict | None = None
  choice: dict | None = None


class RtpEngine:
  backend = "base"

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    raise NotImplementedError

  def close(self) -> None:
    pass


class StaticRtpEngine(RtpEngine):
  backend = "static"

  def __init__(self, static_program: Optional[str] = None):
    self.static_program = static_program or os.getenv("RTP_STATIC_PROGRAM") or DEFAULT_STATIC_RTP

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    start = time.perf_counter()
    text = self.static_program.strip()
    return RtpEngineResult(
      text=text,
      generated_token_count=len([part for part in text.split() if part]),
      prefill_ms=0.0,
      decode_ms=(time.perf_counter() - start) * 1000.0,
      backend=self.backend,
      source_frame_id=frame_id,
    )


class ExternalRtpEngine(RtpEngine):
  backend = "external_gpu"

  def __init__(self, command: str):
    if not command:
      raise VlmError("RTP_VLM_COMMAND is empty")
    self.command = _split_command(command)

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    start = time.perf_counter()
    image_bytes = _board_image_bytes(board)
    payload = {
      "frame_id": frame_id,
      "prompt": FIXED_RTP_PROMPT,
      "vehicle_state": _json_vehicle_state(vehicle_state),
      "scene_board_state_text": board.state_text,
      "scene_board_image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    if board.aux_pngs:
      payload["scene_board_aux_images_b64"] = {
        str(name): base64.b64encode(data).decode("ascii")
        for name, data in board.aux_pngs.items()
      }
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("RTP_CUDA_DEVICE", "0"))
    _sanitize_python_env(env)
    try:
      proc = subprocess.run(
        self.command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=max(0.001, deadline_ms / 1000.0),
        env=env,
        check=False,
      )
    except subprocess.TimeoutExpired as exc:
      raise VlmTimeout("external VLM exceeded deadline") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
      raise VlmError(proc.stderr.strip() or f"external VLM exited {proc.returncode}")

    text = proc.stdout.strip()
    return RtpEngineResult(
      text=text,
      generated_token_count=len([part for part in text.split() if part]),
      prefill_ms=0.0,
      decode_ms=elapsed_ms,
      backend=self.backend,
      source_frame_id=frame_id,
    )


class PersistentRtpEngine(RtpEngine):
  backend = "persistent_gpu"

  def __init__(self, command: str):
    if not command:
      raise VlmError("RTP_VLM_SERVER_COMMAND is empty")
    self.command = _split_command(command)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("RTP_CUDA_DEVICE", "0"))
    _sanitize_python_env(env)
    stderr_target = subprocess.DEVNULL
    self._stderr_file = None
    if stderr_path := os.getenv("RTP_VLM_STDERR_PATH"):
      self._stderr_file = open(stderr_path, "a", encoding="utf-8")
      stderr_target = self._stderr_file
    self.proc = subprocess.Popen(
      self.command,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=stderr_target,
      text=True,
      bufsize=1,
      env=env,
    )
    if os.getenv("RTP_VLM_WAIT_READY") == "1" or "--ready-jsonl" in self.command:
      self._wait_ready()

  def _wait_ready(self) -> None:
    if self.proc.stdout is None:
      raise VlmError("persistent VLM stdout is not available")
    line = self.proc.stdout.readline()
    if not line:
      raise VlmError(f"persistent VLM exited before ready marker ({self.proc.poll()})")
    try:
      response = json.loads(line)
    except json.JSONDecodeError as exc:
      raise VlmError(f"persistent VLM returned non-JSON ready marker: {line[:160]}") from exc
    if response.get("error"):
      raise VlmError(str(response["error"]))
    if response.get("ready") is not True:
      raise VlmError(f"persistent VLM returned unexpected ready marker: {line[:160]}")

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    if self.proc.poll() is not None:
      raise VlmError(f"persistent VLM exited {self.proc.returncode}")
    if self.proc.stdin is None or self.proc.stdout is None:
      raise VlmError("persistent VLM pipes are not available")

    start = time.perf_counter()
    image_bytes = _board_image_bytes(board)
    payload = {
      "frame_id": frame_id,
      "deadline_ms": deadline_ms,
      "prompt": FIXED_RTP_PROMPT,
      "vehicle_state": _json_vehicle_state(vehicle_state),
      "scene_board_state_text": board.state_text,
      "scene_board_image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }
    if board.aux_pngs:
      payload["scene_board_aux_images_b64"] = {
        str(name): base64.b64encode(data).decode("ascii")
        for name, data in board.aux_pngs.items()
      }
    self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    self.proc.stdin.flush()
    response = self._read_response_line()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if response.get("error"):
      raise VlmError(str(response["error"]))
    text = str(response.get("rtp_text", response.get("text", ""))).strip()
    raw_scores = response.get("label_scores_cached") or response.get("label_scores") or {}
    raw_label_scores = response.get("label_scores") or {}
    if not isinstance(raw_scores, dict):
      raw_scores = {}
    if not isinstance(raw_label_scores, dict):
      raw_label_scores = {}
    raw_labels = response.get("labels") or ()
    raw_labels_current_group = response.get("labels_current_group") or raw_labels
    labels_scored_this_request = response.get("labels_scored_this_request") or ()
    score_group_index_raw = response.get("score_group_index")
    try:
      score_group_index = None if score_group_index_raw is None else int(score_group_index_raw)
    except (TypeError, ValueError):
      score_group_index = None
    label_state_debug = response.get("label_state_debug")
    if not isinstance(label_state_debug, dict):
      label_state_debug = None
    return RtpEngineResult(
      text=text,
      generated_token_count=int(response.get("generated_token_count", 0)),
      prefill_ms=float(response.get("prefill_ms", 0.0)),
      decode_ms=float(response.get("decode_ms", elapsed_ms)),
      backend=str(response.get("backend", self.backend)),
      source_frame_id=int(response.get("source_frame_id", response.get("frame_id", frame_id))),
      labels=tuple(str(label) for label in raw_labels),
      label_scores={str(key): float(value) for key, value in raw_scores.items()},
      raw_labels=tuple(str(label) for label in raw_labels_current_group),
      raw_label_scores={str(key): float(value) for key, value in raw_label_scores.items()},
      labels_scored_this_request=tuple(str(label) for label in labels_scored_this_request),
      score_group_index=score_group_index,
      label_state_debug=label_state_debug,
      choice=response.get("choice"),
    )

  def _read_response_line(self) -> dict:
    if self.proc.stdout is None:
      raise VlmError("persistent VLM stdout is not available")
    while True:
      line = self.proc.stdout.readline()
      if not line:
        raise VlmError("persistent VLM closed stdout")
      try:
        response = json.loads(line)
      except json.JSONDecodeError as exc:
        raise VlmError(f"persistent VLM returned non-JSON: {line[:160]}") from exc
      if response.get("ready") is True and "rtp_text" not in response and "text" not in response:
        continue
      return response

  def reset_runtime_state(self) -> None:
    if self.proc.poll() is not None:
      raise VlmError(f"persistent VLM exited {self.proc.returncode}")
    if self.proc.stdin is None or self.proc.stdout is None:
      raise VlmError("persistent VLM pipes are not available")
    self.proc.stdin.write(json.dumps({"control": "reset_runtime_state"}, separators=(",", ":")) + "\n")
    self.proc.stdin.flush()
    response = self._read_response_line()
    if response.get("error"):
      raise VlmError(str(response["error"]))
    if response.get("ok") is not True:
      raise VlmError(f"persistent VLM reset returned unexpected response: {response}")

  def close(self) -> None:
    try:
      if self.proc.stdin is not None:
        self.proc.stdin.close()
      self.proc.wait(timeout=1.0)
    except Exception:
      if self.proc.poll() is None:
        self.proc.terminate()
        try:
          self.proc.wait(timeout=2.0)
        except Exception:
          self.proc.kill()
    finally:
      if self.proc.stdout is not None:
        self.proc.stdout.close()
      if self._stderr_file is not None:
        self._stderr_file.close()


class AsyncRtpEngine(RtpEngine):
  backend = "async_gpu"

  def __init__(
    self,
    inner: RtpEngine,
    update_period_frames: int = 2,
    max_age_frames: int = 6,
    latest_only: bool = False,
    drop_stale_results: bool = False,
    max_result_age_frames: int = 0,
  ):
    self.inner = inner
    self.update_period_frames = max(1, update_period_frames)
    self.max_age_frames = max(0, max_age_frames)
    self.latest_only = latest_only
    self.drop_stale_results = drop_stale_results
    self.max_result_age_frames = max(0, max_result_age_frames)
    self._lock = threading.Lock()
    self._in_flight = False
    self._last_submitted_frame: int | None = None
    self._last_request_frame: int | None = None
    self._pending: tuple[int, SceneBoard, dict[str, float]] | None = None
    self._latest: RtpEngineResult | None = None
    self._last_error = ""
    self._log_path = os.getenv("RTP_VLM_ASYNC_LOG_PATH", "")
    self._epoch = 0

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    with self._lock:
      self._last_request_frame = frame_id
    self._maybe_submit(frame_id, board, vehicle_state)
    with self._lock:
      latest = self._latest
      last_error = self._last_error

    if latest is None or latest.source_frame_id is None:
      raise VlmTimeout(f"async VLM has no RTP yet: {last_error}")

    age_frames = frame_id - latest.source_frame_id
    if age_frames < 0 or age_frames > self.max_age_frames:
      raise VlmTimeout(f"async VLM RTP stale: age_frames={age_frames} max={self.max_age_frames}")

    return RtpEngineResult(
      text=latest.text,
      generated_token_count=latest.generated_token_count,
      prefill_ms=0.0,
      decode_ms=0.0,
      backend=f"async({latest.backend})",
      source_frame_id=latest.source_frame_id,
      labels=latest.labels,
      label_scores=latest.label_scores,
      raw_labels=latest.raw_labels,
      raw_label_scores=latest.raw_label_scores,
      labels_scored_this_request=latest.labels_scored_this_request,
      score_group_index=latest.score_group_index,
      label_state_debug=latest.label_state_debug,
      choice=latest.choice,
    )

  def _maybe_submit(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float]) -> None:
    board_snapshot = SceneBoard(board.width, board.height, bytearray(board.pixels), board.state_text)
    state_snapshot = dict(vehicle_state)
    state_snapshot.pop("road_frame", None)

    with self._lock:
      if self._in_flight:
        if self.latest_only:
          self._pending = (frame_id, board_snapshot, state_snapshot)
        return
      if self._last_submitted_frame is not None and frame_id - self._last_submitted_frame < self.update_period_frames:
        return
      self._in_flight = True
      self._last_submitted_frame = frame_id
      epoch = self._epoch

    thread = threading.Thread(
      target=self._worker,
      args=(frame_id, board_snapshot, state_snapshot, epoch),
      name=f"async-rtp-{frame_id}",
      daemon=True,
    )
    thread.start()

  def _worker(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], epoch: int) -> None:
    current_frame_id = frame_id
    current_board = board
    current_state = vehicle_state
    while True:
      try:
        result = self.inner.generate(current_frame_id, current_board, current_state, 10_000.0)
        parse_rtp(result.text)
        if result.source_frame_id is None:
          result = RtpEngineResult(
            text=result.text,
            generated_token_count=result.generated_token_count,
            prefill_ms=result.prefill_ms,
            decode_ms=result.decode_ms,
            backend=result.backend,
            source_frame_id=current_frame_id,
            labels=result.labels,
            label_scores=result.label_scores,
            raw_labels=result.raw_labels,
            raw_label_scores=result.raw_label_scores,
            labels_scored_this_request=result.labels_scored_this_request,
            score_group_index=result.score_group_index,
            label_state_debug=result.label_state_debug,
            choice=result.choice,
          )
        with self._lock:
          if epoch != self._epoch:
            self._log_async_event("dropped_reset_epoch", frame_id=current_frame_id, source_frame_id=result.source_frame_id)
            return
          last_request_frame = self._last_request_frame
          completion_age = 0 if last_request_frame is None or result.source_frame_id is None else last_request_frame - result.source_frame_id
          if not self.drop_stale_results or completion_age <= self.max_result_age_frames:
            self._latest = result
            self._last_error = ""
            self._log_async_event(
              "accepted",
              frame_id=current_frame_id,
              source_frame_id=result.source_frame_id,
              last_request_frame=last_request_frame,
              completion_age_frames=completion_age,
              labels=list(result.labels),
              raw_labels=list(result.raw_labels),
              labels_scored_this_request=list(result.labels_scored_this_request),
              score_group_index=result.score_group_index,
              label_state_debug=result.label_state_debug,
              backend=result.backend,
            )
          else:
            self._last_error = f"dropped stale async RTP: completion_age_frames={completion_age} max={self.max_result_age_frames}"
            self._log_async_event(
              "dropped_stale",
              frame_id=current_frame_id,
              source_frame_id=result.source_frame_id,
              last_request_frame=last_request_frame,
              completion_age_frames=completion_age,
              max_result_age_frames=self.max_result_age_frames,
              labels=list(result.labels),
              raw_labels=list(result.raw_labels),
              labels_scored_this_request=list(result.labels_scored_this_request),
              score_group_index=result.score_group_index,
              label_state_debug=result.label_state_debug,
              backend=result.backend,
            )
          pending = self._pending
          self._pending = None
          if self.latest_only and pending is not None:
            current_frame_id, current_board, current_state = pending
            self._last_submitted_frame = current_frame_id
            self._log_async_event("consume_pending", frame_id=current_frame_id)
            continue
          self._in_flight = False
          return
      except Exception as exc:
        with self._lock:
          if epoch != self._epoch:
            self._log_async_event("dropped_reset_epoch_error", frame_id=current_frame_id, error=repr(exc))
            return
          self._last_error = str(exc)
          self._log_async_event("error", frame_id=current_frame_id, error=repr(exc))
          pending = self._pending
          self._pending = None
          if self.latest_only and pending is not None:
            current_frame_id, current_board, current_state = pending
            self._last_submitted_frame = current_frame_id
            self._log_async_event("consume_pending_after_error", frame_id=current_frame_id)
            continue
          self._in_flight = False
          return

  def wait_idle(self, timeout_s: float = 5.0) -> bool:
    deadline = time.perf_counter() + max(0.0, timeout_s)
    while True:
      with self._lock:
        if not self._in_flight:
          return True
      if time.perf_counter() >= deadline:
        return False
      time.sleep(0.005)

  def reset_runtime_state(self) -> None:
    with self._lock:
      self._epoch += 1
      self._in_flight = False
      self._last_submitted_frame = None
      self._last_request_frame = None
      self._pending = None
      self._latest = None
      self._last_error = ""
      self._log_async_event("reset_runtime_state", epoch=self._epoch)
    reset_inner = getattr(self.inner, "reset_runtime_state", None)
    if callable(reset_inner):
      reset_inner()

  def close(self) -> None:
    close = getattr(self.inner, "close", None)
    if close is not None:
      close()

  def _log_async_event(self, event: str, **fields) -> None:
    if not self._log_path:
      return
    row = {
      "event": event,
      "mono_time": time.monotonic(),
      **fields,
    }
    try:
      with open(self._log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except Exception:
      pass


def build_rtp_engine() -> RtpEngine:
  server_command = os.getenv("RTP_VLM_SERVER_COMMAND")
  if server_command:
    engine: RtpEngine = PersistentRtpEngine(server_command)
    if os.getenv("RTP_VLM_ASYNC") == "1":
      return AsyncRtpEngine(
        engine,
        update_period_frames=int(os.getenv("RTP_VLM_ASYNC_PERIOD_FRAMES", "2")),
        max_age_frames=int(os.getenv("RTP_VLM_ASYNC_MAX_AGE_FRAMES", "6")),
        latest_only=os.getenv("RTP_VLM_ASYNC_LATEST_ONLY") == "1",
        drop_stale_results=os.getenv("RTP_VLM_ASYNC_DROP_STALE_RESULTS") == "1",
        max_result_age_frames=int(os.getenv("RTP_VLM_ASYNC_MAX_RESULT_AGE_FRAMES", os.getenv("RTP_VLM_ASYNC_MAX_AGE_FRAMES", "6"))),
      )
    return engine
  command = os.getenv("RTP_VLM_COMMAND")
  if command:
    engine = ExternalRtpEngine(command)
    if os.getenv("RTP_VLM_ASYNC") == "1":
      return AsyncRtpEngine(
        engine,
        update_period_frames=int(os.getenv("RTP_VLM_ASYNC_PERIOD_FRAMES", "2")),
        max_age_frames=int(os.getenv("RTP_VLM_ASYNC_MAX_AGE_FRAMES", "6")),
        latest_only=os.getenv("RTP_VLM_ASYNC_LATEST_ONLY") == "1",
        drop_stale_results=os.getenv("RTP_VLM_ASYNC_DROP_STALE_RESULTS") == "1",
        max_result_age_frames=int(os.getenv("RTP_VLM_ASYNC_MAX_RESULT_AGE_FRAMES", os.getenv("RTP_VLM_ASYNC_MAX_AGE_FRAMES", "6"))),
      )
    return engine
  return StaticRtpEngine()


def _sanitize_python_env(env: dict[str, str]) -> None:
  if env.get("PYTHONUTF8") not in (None, "0", "1"):
    env.pop("PYTHONUTF8", None)


def _board_image_bytes(board: SceneBoard) -> bytes:
  image_format = os.getenv("RTP_VLM_IMAGE_FORMAT", "ppm").strip().lower()
  if image_format in ("jpg", "jpeg"):
    quality = int(os.getenv("RTP_VLM_JPEG_QUALITY", "85"))
    return board.to_jpeg_bytes(quality) or board.to_ppm_bytes()
  if image_format == "ppm":
    return board.to_ppm_bytes()
  return board.to_png_bytes() or board.to_ppm_bytes()


def _json_vehicle_state(vehicle_state: dict[str, object]) -> dict[str, bool | float | int | str]:
  out: dict[str, bool | float | int | str] = {}
  for key, value in vehicle_state.items():
    if isinstance(value, bool):
      out[str(key)] = value
    elif isinstance(value, numbers.Integral):
      out[str(key)] = int(value)
    elif isinstance(value, numbers.Real):
      real_value = float(value)
      if math.isfinite(real_value):
        out[str(key)] = real_value
    elif isinstance(value, str):
      out[str(key)] = value
  return out


def _split_command(command: str) -> list[str]:
  if os.name != "nt":
    return shlex.split(command)

  import ctypes

  argc = ctypes.c_int()
  shell32 = ctypes.windll.shell32
  kernel32 = ctypes.windll.kernel32
  shell32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
  shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
  kernel32.LocalFree.argtypes = [ctypes.c_void_p]
  kernel32.LocalFree.restype = ctypes.c_void_p

  argv = shell32.CommandLineToArgvW(command, ctypes.byref(argc))
  if not argv:
    raise VlmError(f"failed to parse command: {command}")
  try:
    return [argv[i] for i in range(argc.value)]
  finally:
    kernel32.LocalFree(argv)


def detect_local_gpu() -> str:
  try:
    proc = subprocess.run(
      ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
      text=True,
      capture_output=True,
      timeout=2.0,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired):
    return "nvidia-smi unavailable"
  if proc.returncode != 0:
    return proc.stderr.strip() or "nvidia-smi failed"
  return proc.stdout.strip() or "no NVIDIA GPU reported"
