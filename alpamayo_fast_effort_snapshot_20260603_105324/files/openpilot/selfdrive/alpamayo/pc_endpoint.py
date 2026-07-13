#!/usr/bin/env python3
import argparse
from collections import deque
import copy
import importlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from openpilot.selfdrive.alpamayo.protocol import (
  PROTOCOL_VERSION,
REQUEST_CONTENT_TYPE,
RESPONSE_CONTENT_TYPE,
decode_payload,
encode_payload,
parse_xyzt_dict,
)


PC_TRACE_ENV = "ALPAMAYO_PC_TRACE_PATH"
REQUIRED_REASONING_MODE = "full"
REQUIRED_TRAJECTORY_FIELDS = ("position", "orientation", "velocity", "orientationRate", "acceleration")
PRIMARY_SOURCES = ("remoteServer", "localEgpu", 2, 3)
DEFAULT_ADAPTER_SPEC = "openpilot.selfdrive.alpamayo.local_adapter:LocalAlpamayoAdapter"
VALID_STATUSES = ("valid", 1)


def _env_bool(name: str, default: bool = False) -> bool:
  value = os.getenv(name)
  if value is None:
    return bool(default)
  return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.getenv(name, str(default)))
  except (TypeError, ValueError):
    return int(default)


def _reasoning_generation_disabled() -> bool:
  return _env_bool("ALPAMAYO_DISABLE_REASONING_GENERATION", False)


def _async_fast_recompute_enabled() -> bool:
  return _env_bool("ALPAMAYO_PC_ASYNC_FAST_RECOMPUTE", True)


def _is_truthy_skip_flag(value: Any) -> bool:
  if isinstance(value, str):
    return value.strip().lower() in ("1", "true", "yes", "on")
  return str(value).strip().lower() in ("1", "true", "yes", "on")


class AlpamayoAdapter(Protocol):
  def infer(self, request: dict[str, Any]) -> dict[str, Any]:
    ...


class ContractError(ValueError):
  pass


class NoAdapter:
  def infer(self, request: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("no Alpamayo adapter configured")


def _coerce_positive_shape(shape_container: dict[str, Any], label: str) -> list[int]:
  if label not in shape_container:
    raise ContractError(f"debug.deepstackInputShapes.{label} missing")
  shape = shape_container.get(label)
  if not isinstance(shape, (list, tuple)):
    raise ContractError(f"debug.deepstackInputShapes.{label} must be an array of positive integers")
  parsed: list[int] = []
  for dim in shape:
    if isinstance(dim, bool):
      raise ContractError(f"debug.deepstackInputShapes.{label} must be an array of positive integers")
    try:
      parsed_dim = int(dim)
    except (TypeError, ValueError):
      raise ContractError(f"debug.deepstackInputShapes.{label} must be an array of positive integers")
    if parsed_dim <= 0:
      raise ContractError(f"debug.deepstackInputShapes.{label} must contain positive integers")
    parsed.append(parsed_dim)
  return parsed


def _jsonable(value: Any) -> Any:
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return str(value)


class PcTraceLogger:
  def __init__(self, path: str | Path | None = None):
    env_path = os.getenv(PC_TRACE_ENV)
    self.path = Path(path or env_path) if path or env_path else None
    self.disabled = self.path is None

  def log(self, event: str, **fields: Any) -> None:
    if self.disabled or self.path is None:
      return
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      payload = {"event": event, "monoTimeNs": time.monotonic_ns()}
      payload.update({key: _jsonable(value) for key, value in fields.items()})
      with self.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
      self.disabled = True


def _error_response(status: str, reason: str) -> dict[str, Any]:
  return {
    "semanticPlan": {
      "source": "remoteServer",
      "status": status,
      "confidence": 0.0,
      "consistency": 0.0,
      "age": 0.0,
      "error": reason,
    }
  }


def _latest_frame_id(request: dict[str, Any]) -> int:
  frames = request.get("frames", [])
  frame_ids: list[int] = []
  if isinstance(frames, list):
    for frame in frames:
      if not isinstance(frame, dict):
        continue
      try:
        frame_ids.append(int(frame.get("frameId", -1)))
      except (TypeError, ValueError):
        continue
  return max(frame_ids) if frame_ids else -1


def _trace_response_debug_fields(response: dict[str, Any]) -> dict[str, Any]:
  semantic = response.get("semanticPlan", response)
  if not isinstance(semantic, dict):
    return {}
  debug = semantic.get("debug")
  if not isinstance(debug, dict):
    return {}
  runtime_profile = debug.get("runtimeProfile")
  if not isinstance(runtime_profile, dict):
    runtime_profile = {}
  frame_cache = debug.get("frameCacheStats")
  if not isinstance(frame_cache, dict):
    frame_cache = {}
  vision_cache = frame_cache.get("streamingVisionCache")
  if not isinstance(vision_cache, dict):
    vision_cache = {}
  prefix_cache = frame_cache.get("vlmPrefixCache")
  if not isinstance(prefix_cache, dict):
    prefix_cache = {}
  shifted_kv_plan = prefix_cache.get("shiftedPromptKvReusePlan")
  if not isinstance(shifted_kv_plan, dict):
    shifted_kv_plan = {}
  image_vision_runtime = frame_cache.get("streaming_image_vision_runtime")
  if not isinstance(image_vision_runtime, dict):
    image_vision_runtime = {}
  return {
    "adapterLatencyMs": debug.get("adapterLatencyMs"),
    "numFrames": debug.get("numFrames"),
    "streamingVisionAttentionMaskEnabled": debug.get("streamingVisionAttentionMaskEnabled"),
    "vlmGenerateSeconds": runtime_profile.get("vlm_generate_seconds"),
    "diffusionSeconds": runtime_profile.get("diffusion_seconds"),
    "actionToTrajSeconds": runtime_profile.get("action_to_traj_seconds"),
    "expertStepCalls": runtime_profile.get("expert_step_calls"),
    "runtimeTotalSeconds": runtime_profile.get("total_seconds"),
    "fullGenerationCacheHit": runtime_profile.get("vlm_full_generation_cache_hit"),
    "trustedReplayAllowed": runtime_profile.get("vlm_full_generation_cache_trusted_replay_allowed"),
    "trustedReplayRequested": prefix_cache.get("trustedReplayRequested"),
    "trustedReplayDisabledForDiffusionFreshness": prefix_cache.get("trustedReplayDisabledForDiffusionFreshness"),
    "dflashTrustedReplayAllowed": runtime_profile.get("dflash_full_generation_cache_trusted_replay_allowed"),
    "dflashFullGenerationCacheHit": runtime_profile.get("dflash_full_generation_cache_hit"),
    "fullGenerationWindowSignatureMatch": runtime_profile.get("vlm_full_generation_cache_window_signature_match"),
    "fullGenerationPromptCacheContextExact": runtime_profile.get("vlm_full_generation_prompt_cache_context_exact"),
    "fullGenerationPromptCacheContextBlocked": runtime_profile.get(
      "vlm_full_generation_cache_disabled_without_exact_prompt_cache_context"
    ),
    "dflashFullGenerationWindowSignatureMatch": runtime_profile.get(
      "dflash_full_generation_cache_window_signature_match"
    ),
    "dflashFullGenerationPromptCacheContextExact": runtime_profile.get(
      "dflash_full_generation_prompt_cache_context_exact"
    ),
    "dflashFullGenerationPromptCacheContextBlocked": runtime_profile.get(
      "dflash_full_generation_cache_disabled_without_exact_prompt_cache_context"
    ),
    "sourceCachePromptCacheBlocked": runtime_profile.get(
      "streaming_vlm_draft_verify_source_cache_prompt_cache_blocked"
    ),
    "servedFromLastValidCache": debug.get("servedFromLastValidCache"),
    "forceVlmRefresh": debug.get("forceVlmRefresh"),
    "adaptiveFlowCacheHit": runtime_profile.get("adaptive_flow_cache_hit"),
    "adaptiveFlowCacheMiss": runtime_profile.get("adaptive_flow_cache_miss"),
    "adaptiveFlowCacheKeyMode": runtime_profile.get("adaptive_flow_cache_key_mode"),
    "adaptiveFlowMiddleVelocityReused": runtime_profile.get("adaptive_flow_middle_velocity_reused"),
    "adaptiveFlowInitialNoiseReused": runtime_profile.get("adaptive_flow_initial_noise_reused"),
    "adaptiveFlowMiddleVelocityExpertStepSkipped": runtime_profile.get(
      "adaptive_flow_middle_velocity_expert_step_skipped"
    ),
    "adaptiveFlowActionCacheHit": runtime_profile.get("adaptive_flow_action_cache_hit"),
    "adaptiveFlowActionCacheMiss": runtime_profile.get("adaptive_flow_action_cache_miss"),
    "adaptiveFlowSelectedSteps": runtime_profile.get("adaptive_flow_selected_steps"),
    "adaptiveFlowMode": runtime_profile.get("adaptive_flow_mode"),
    "adaptiveFlowGraphableOneStep": runtime_profile.get("adaptive_flow_graphable_one_step"),
    "graphActionStageRequested": runtime_profile.get("graph_action_stage_requested"),
    "graphActionStageMode": runtime_profile.get("graph_action_stage_mode"),
    "graphActionStageReady": runtime_profile.get("graph_action_stage_ready"),
    "graphActionDiffusionStepGraphMode": runtime_profile.get("graph_action_diffusion_step_graph_mode"),
    "graphActionDiffusionStepGraphReady": runtime_profile.get("graph_action_diffusion_step_graph_ready"),
    "graphActionDiffusionStepGraphCacheHit": runtime_profile.get("graph_action_diffusion_step_graph_cache_hit"),
    "streamOverlapRatio": frame_cache.get("stream_overlap_ratio"),
    "streamingFrameHits": vision_cache.get("frame_hits"),
    "streamingFrameMisses": vision_cache.get("frame_misses"),
    "streamingPreRopeLayerSlots": vision_cache.get("pre_rope_kv_layer_slots"),
    "visionMaskShortcut": frame_cache.get("streaming_image_vision_attention_mask_unverified_shortcut"),
    "visionQkvCachedBlocks": image_vision_runtime.get("streaming_vision_qkv_split_projection_cached_blocks"),
    "visionQkvMissTokens": image_vision_runtime.get("streaming_vision_qkv_split_projection_miss_tokens"),
    "prefixHit": prefix_cache.get("hit"),
    "prefixReason": prefix_cache.get("reason"),
    "streamingReuseMode": prefix_cache.get("streamingReuseMode"),
    "streamingReuseUnverified": prefix_cache.get("streamingReuseUnverified"),
    "refreshDue": prefix_cache.get("refreshDue"),
    "streamingReuseOverlapRatio": prefix_cache.get("streamingReuseOverlapRatio"),
    "streamingReuseOverlapSource": prefix_cache.get("streamingReuseOverlapSource"),
    "languageVisualTokenSpans": prefix_cache.get("languageVisualTokenSpans"),
    "shiftedPromptKvRangeCount": shifted_kv_plan.get("range_count"),
    "shiftedPromptKvRetainedTokens": shifted_kv_plan.get("retained_language_tokens"),
    "shiftedPromptKvCurrentTokens": shifted_kv_plan.get("current_visual_language_tokens"),
    "shiftedPromptKvRetainedRatio": shifted_kv_plan.get("retained_ratio"),
  }


def validate_request(request: dict[str, Any]) -> None:
  if not isinstance(request, dict):
    raise ContractError("request is not an object")
  if int(request.get("protocolVersion", -1)) != PROTOCOL_VERSION:
    raise ContractError("protocolVersion mismatch")
  if "stockPlan" in request:
    raise ContractError("stockPlan is forbidden for pure Alpamayo")

  frames = request.get("frames")
  if not isinstance(frames, list) or not frames:
    raise ContractError("frames missing")
  for frame in frames:
    if not isinstance(frame, dict):
      raise ContractError("frame is not an object")
    for key in ("stream", "encoding", "frameId", "timestampSof", "timestampEof", "width", "height", "dataBase64"):
      if key not in frame:
        raise ContractError(f"frame.{key} missing")

  vehicle_state = request.get("vehicleState")
  if not isinstance(vehicle_state, dict):
    raise ContractError("vehicleState missing")
  for key in ("vEgo", "aEgo", "standstill", "steeringAngleDeg"):
    if key not in vehicle_state:
      raise ContractError(f"vehicleState.{key} missing")

  camera_bundle = request.get("cameraBundle")
  if not isinstance(camera_bundle, dict):
    raise ContractError("cameraBundle missing")
  for key in ("streamOrder", "framesPerCamera", "frameStepNs"):
    if key not in camera_bundle:
      raise ContractError(f"cameraBundle.{key} missing")

  runtime_config = request.get("runtimeConfig")
  if not isinstance(runtime_config, dict):
    raise ContractError("runtimeConfig missing")
  reasoning_mode = runtime_config.get("reasoningMode")
  if reasoning_mode is None:
    raise ContractError("runtimeConfig.reasoningMode missing")
  if str(reasoning_mode) != REQUIRED_REASONING_MODE:
    raise ContractError(f"runtimeConfig.reasoningMode must be '{REQUIRED_REASONING_MODE}'")
  if _is_truthy_skip_flag(runtime_config.get("skipVlmGeneration")):
    raise ContractError("runtimeConfig.skipVlmGeneration is forbidden for production Alpamayo")

  ego_history = request.get("egoHistory")
  if not isinstance(ego_history, dict):
    raise ContractError("egoHistory missing")
  if "xyz" not in ego_history:
    raise ContractError("egoHistory.xyz missing")
  if "rot" not in ego_history:
    raise ContractError("egoHistory.rot missing")


def _semantic_response(response: dict[str, Any]) -> dict[str, Any]:
  semantic = response.get("semanticPlan", response)
  if not isinstance(semantic, dict):
    raise ContractError("semanticPlan missing")
  return semantic


def _assert_deepstack_reasoning_debug(debug: dict[str, Any]) -> None:
  reasoning_mode = debug.get("reasoningMode")
  if reasoning_mode is None:
    raise ContractError("debug.reasoningMode missing")
  if str(reasoning_mode) != REQUIRED_REASONING_MODE:
    raise ContractError(f"debug.reasoningMode must be '{REQUIRED_REASONING_MODE}'")

  if _is_truthy_skip_flag(debug.get("skipVlmGeneration")):
    raise ContractError("debug.skipVlmGeneration is forbidden for production Alpamayo")

  reasoning_tokens = debug.get("reasoningGeneratedTokens")
  if not isinstance(reasoning_tokens, (int, float)):
    raise ContractError("debug.reasoningGeneratedTokens must be positive for production Alpamayo")
  if int(reasoning_tokens) <= 0:
    raise ContractError("debug.reasoningGeneratedTokens must be positive for production Alpamayo")

  deepstack_shapes = debug.get("deepstackInputShapes")
  if not isinstance(deepstack_shapes, dict):
    raise ContractError("debug.deepstackInputShapes missing")

  required_shape_keys = ("inputIdsShape", "pixelValuesShape", "imageGridThwShape")
  deepstack_shape_values = {
    key: _coerce_positive_shape(deepstack_shapes, key)
    for key in required_shape_keys
  }

  deepstack_shapes.update(deepstack_shape_values)

  if len(deepstack_shape_values["inputIdsShape"]) != 2:
    raise ContractError("debug.deepstackInputShapes.inputIdsShape must have 2 dimensions")
  if len(deepstack_shape_values["pixelValuesShape"]) not in (2, 4):
    raise ContractError("debug.deepstackInputShapes.pixelValuesShape must have 2 flattened-patch dimensions or 4 NCHW dimensions")
  if len(deepstack_shape_values["pixelValuesShape"]) == 4 and deepstack_shape_values["pixelValuesShape"][1] != 3:
    raise ContractError("debug.deepstackInputShapes.pixelValuesShape channel dimension must be 3")
  if len(deepstack_shape_values["imageGridThwShape"]) != 2 or deepstack_shape_values["imageGridThwShape"][1] != 3:
    raise ContractError("debug.deepstackInputShapes.imageGridThwShape must be [N,3]")


def validate_response(response: dict[str, Any], *, require_deepstack_reasoning: bool = False) -> None:
  if not isinstance(response, dict):
    raise ContractError("response is not an object")

  semantic = _semantic_response(response)
  if semantic.get("source", "remoteServer") not in PRIMARY_SOURCES:
    raise ContractError("semantic source is not Alpamayo primary")
  if semantic.get("status", "valid") not in VALID_STATUSES:
    raise ContractError("semantic status is not valid")

  trajectory = semantic.get("trajectory", semantic)
  if not isinstance(trajectory, dict):
    raise ContractError("trajectory missing")

  position_payload = trajectory.get("position")
  if not isinstance(position_payload, dict) or "t" not in position_payload:
    raise ContractError("trajectory.position.t missing")
  parsed_position = parse_xyzt_dict(position_payload, None)
  if parsed_position is None:
    raise ContractError("trajectory.position invalid")
  shared_t = parsed_position[0]

  for field in REQUIRED_TRAJECTORY_FIELDS:
    payload = trajectory.get(field)
    if parse_xyzt_dict(payload, shared_t) is None:
      raise ContractError(f"trajectory.{field} invalid")

  if require_deepstack_reasoning:
    debug = semantic.get("debug")
    if not isinstance(debug, dict):
      raise ContractError("debug missing")
    _assert_deepstack_reasoning_debug(debug)


def load_adapter(spec: str | None) -> AlpamayoAdapter:
  if not spec:
    return NoAdapter()
  module_name, sep, attr_name = spec.partition(":")
  if not sep or not module_name or not attr_name:
    raise ValueError("adapter spec must be module:attribute")

  module = importlib.import_module(module_name)
  target = getattr(module, attr_name)
  adapter = target() if callable(target) else target
  infer = getattr(adapter, "infer", None)
  if not callable(infer):
    raise TypeError("adapter must expose infer(request)")
  return adapter


class AlpamayoPcEndpoint:
  def __init__(self, adapter: AlpamayoAdapter | None = None, trace_logger: PcTraceLogger | None = None):
    self.adapter = adapter or NoAdapter()
    self.trace_logger = trace_logger or PcTraceLogger()
    self.adapter_lock = threading.Lock()
    self.last_valid_lock = threading.Lock()
    self.last_valid_response: dict[str, Any] | None = None
    self.last_valid_latest_frame_id = -1
    self.refresh_lock = threading.Lock()
    self.refresh_thread: threading.Thread | None = None
    self.pending_fast_refresh_requests: deque[dict[str, Any]] = deque(
      maxlen=max(1, _env_int("ALPAMAYO_PC_ASYNC_FAST_QUEUE_MAX", 96))
    )

  def _store_last_valid(self, response: dict[str, Any], latest_frame_id: int | None = None) -> None:
    with self.last_valid_lock:
      self.last_valid_response = copy.deepcopy(response)
      if latest_frame_id is not None:
        self.last_valid_latest_frame_id = int(latest_frame_id)

  def _last_valid_frame_gap(self, latest_frame_id: int) -> int | None:
    with self.last_valid_lock:
      if self.last_valid_response is None or self.last_valid_latest_frame_id < 0:
        return None
      return int(latest_frame_id) - int(self.last_valid_latest_frame_id)

  def _cached_last_valid_response(self, latest_frame_id: int) -> dict[str, Any] | None:
    with self.last_valid_lock:
      if self.last_valid_response is None:
        return None
      response = copy.deepcopy(self.last_valid_response)
      source_frame_id = int(self.last_valid_latest_frame_id)
    semantic = response.get("semanticPlan")
    if isinstance(semantic, dict):
      debug = semantic.setdefault("debug", {})
      if isinstance(debug, dict):
        debug["servedFromLastValidCache"] = True
        debug["servedFromLastValidCacheLatestFrameId"] = source_frame_id
        debug["servedFromLastValidCacheRequestedFrameId"] = int(latest_frame_id)
        debug["servedFromLastValidCacheAgeFrames"] = max(0, int(latest_frame_id) - source_frame_id)
        runtime_profile = debug.setdefault("runtimeProfile", {})
        if isinstance(runtime_profile, dict):
          runtime_profile["pc_endpoint_served_from_last_valid_cache"] = 1
      semantic["age"] = 0.0
    return response

  @staticmethod
  def _response_requests_background_refresh(response: dict[str, Any]) -> bool:
    semantic = response.get("semanticPlan", response)
    if not isinstance(semantic, dict):
      return False
    debug = semantic.get("debug")
    if not isinstance(debug, dict):
      return False
    frame_cache = debug.get("frameCacheStats")
    if not isinstance(frame_cache, dict):
      return False
    prefix_cache = frame_cache.get("vlmPrefixCache")
    if not isinstance(prefix_cache, dict) or not bool(prefix_cache.get("refreshDue")):
      return False
    if _reasoning_generation_disabled():
      streaming_reuse_mode = str(prefix_cache.get("streamingReuseMode", ""))
      if bool(prefix_cache.get("trustedReplayAllowed")) and streaming_reuse_mode == "trusted_full_replay_no_reasoning":
        return False
      if streaming_reuse_mode == "shifted_kv_current_state_suffix":
        return False
    return True

  @staticmethod
  def _background_request(request: dict[str, Any], *, force_vlm_refresh: bool, fast_no_prefill: bool) -> dict[str, Any]:
    refresh_request = copy.deepcopy(request)
    runtime_config = refresh_request.setdefault("runtimeConfig", {})
    if isinstance(runtime_config, dict):
      if force_vlm_refresh:
        runtime_config["alpamayoForceVlmRefresh"] = True
      else:
        runtime_config.pop("alpamayoForceVlmRefresh", None)
        runtime_config.pop("forceVlmRefresh", None)
      if fast_no_prefill:
        runtime_config["alpamayoDeferStateFreshNoReasoning"] = True
        runtime_config["alpamayoRequireFastNoPrefill"] = True
      else:
        runtime_config.pop("alpamayoDeferStateFreshNoReasoning", None)
        runtime_config.pop("alpamayoRequireFastNoPrefill", None)
    return refresh_request

  def _start_background_refresh_request(self, request: dict[str, Any]) -> None:
    self._start_background_worker(
      self._background_request(request, force_vlm_refresh=True, fast_no_prefill=False),
      "background_refresh",
    )

  def _start_background_fast_refresh_request(self, request: dict[str, Any]) -> None:
    self._start_background_worker(
      self._background_request(request, force_vlm_refresh=False, fast_no_prefill=True),
      "background_fast_refresh",
    )

  def _start_background_worker(self, request: dict[str, Any], refresh_kind: str) -> None:
    with self.refresh_lock:
      if self.refresh_thread is not None and self.refresh_thread.is_alive():
        if refresh_kind == "background_fast_refresh":
          self.pending_fast_refresh_requests.append(request)
        return
      self.refresh_thread = threading.Thread(
        target=self._run_background_refresh_loop,
        args=(request, refresh_kind),
        daemon=True,
      )
      self.refresh_thread.start()

  def _start_background_refresh_if_needed(self, request: dict[str, Any], response: dict[str, Any]) -> None:
    if not self._response_requests_background_refresh(response):
      return
    self._start_background_refresh_request(request)

  def _run_background_refresh_loop(self, request: dict[str, Any], refresh_kind: str) -> None:
    next_request: dict[str, Any] | None = request
    next_kind = refresh_kind
    while next_request is not None:
      self._run_background_refresh(next_request, next_kind)
      with self.refresh_lock:
        if self.pending_fast_refresh_requests:
          next_request = self.pending_fast_refresh_requests.popleft()
          next_kind = "background_fast_refresh"
        else:
          self.refresh_thread = None
          next_request = None

  def _run_background_refresh(self, request: dict[str, Any], refresh_kind: str = "background_refresh") -> None:
    start_ns = time.monotonic_ns()
    latest_frame_id = _latest_frame_id(request)
    outcome = f"{refresh_kind}_valid"
    reason = ""
    response_payload: dict[str, Any] | None = None
    try:
      with self.adapter_lock:
        try:
          response_payload = self.adapter.infer(request)
        except Exception as exc:
          if refresh_kind != "background_fast_refresh" or not str(exc).startswith("alpamayo_fast_no_prefill_required:"):
            raise
          reason = str(exc)
          fallback_request = self._background_request(request, force_vlm_refresh=False, fast_no_prefill=False)
          response_payload = self.adapter.infer(fallback_request)
          outcome = "background_fast_refresh_fallback_valid"
      validate_response(response_payload, require_deepstack_reasoning=True)
      self._store_last_valid(response_payload, latest_frame_id)
    except Exception as exc:
      outcome = f"{refresh_kind}_error"
      reason = str(exc)
      response_payload = _error_response("unavailable", reason)
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    self.trace_logger.log(
      "pc_alpamayo_background_refresh",
      statusCode=int(HTTPStatus.OK if not reason else HTTPStatus.SERVICE_UNAVAILABLE),
      outcome=outcome,
      reason=reason,
      latestFrameId=latest_frame_id,
      refreshKind=refresh_kind,
      latencyMs=elapsed_ms,
      **_trace_response_debug_fields(response_payload or {}),
    )

  def handle_payload(self, body: bytes, content_type: str = REQUEST_CONTENT_TYPE) -> tuple[int, bytes, str]:
    start_ns = time.monotonic_ns()
    latest_frame_id = -1
    status_code = HTTPStatus.OK
    outcome = "valid"
    reason = ""
    response_payload: dict[str, Any]

    try:
      if content_type and content_type.split(";", 1)[0].strip() not in ("", REQUEST_CONTENT_TYPE, "application/json"):
        raise ContractError("unsupported content type")
      request = decode_payload(body)
      latest_frame_id = _latest_frame_id(request)
      validate_request(request)
      async_fast_cached = False
      if _async_fast_recompute_enabled():
        cached_response = self._cached_last_valid_response(latest_frame_id)
        if cached_response is not None:
          response_payload = cached_response
          outcome = "async_fast_cached_last_valid_refresh_queued"
          async_fast_cached = True
          self._start_background_fast_refresh_request(request)
      acquired = False
      cached_for_gap = False
      if not async_fast_cached:
        last_valid_gap = self._last_valid_frame_gap(latest_frame_id)
        if last_valid_gap is not None and last_valid_gap > 12:
          cached_response = self._cached_last_valid_response(latest_frame_id)
          if cached_response is not None:
            response_payload = cached_response
            outcome = "stale_gap_cached_last_valid"
            cached_for_gap = True
            self._start_background_refresh_request(request)
      if not async_fast_cached and not cached_for_gap:
        acquired = self.adapter_lock.acquire(blocking=False)
        if not acquired:
          cached_response = self._cached_last_valid_response(latest_frame_id)
          if cached_response is None:
            self.adapter_lock.acquire()
            acquired = True
          else:
            response_payload = cached_response
            outcome = "adapter_busy_cached_last_valid"
        if acquired:
          try:
            response_payload = self.adapter.infer(request)
          finally:
            self.adapter_lock.release()
      validate_response(response_payload, require_deepstack_reasoning=True)
      if outcome not in (
        "adapter_busy_cached_last_valid",
        "stale_gap_cached_last_valid",
        "async_fast_cached_last_valid_refresh_queued",
      ):
        self._store_last_valid(response_payload, latest_frame_id)
      self._start_background_refresh_if_needed(request, response_payload)
    except ContractError as exc:
      status_code = HTTPStatus.BAD_REQUEST
      outcome = "contract_error"
      reason = str(exc)
      response_payload = _error_response("error", reason)
    except Exception as exc:
      status_code = HTTPStatus.SERVICE_UNAVAILABLE
      outcome = "adapter_unavailable"
      reason = str(exc)
      response_payload = _error_response("unavailable", reason)

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    self.trace_logger.log(
      "pc_alpamayo_request",
      statusCode=int(status_code),
      outcome=outcome,
      reason=reason,
      latestFrameId=latest_frame_id,
      latencyMs=elapsed_ms,
      **_trace_response_debug_fields(response_payload),
    )
    return int(status_code), encode_payload(response_payload), RESPONSE_CONTENT_TYPE


class AlpamayoRequestHandler(BaseHTTPRequestHandler):
  endpoint: AlpamayoPcEndpoint

  def do_GET(self) -> None:
    if self.path != "/health":
      self.send_error(HTTPStatus.NOT_FOUND)
      return
    payload = b'{"ok":true,"service":"alpamayo-pc-endpoint"}'
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

  def do_POST(self) -> None:
    if self.path not in ("/", "/alpamayo"):
      self.send_error(HTTPStatus.NOT_FOUND)
      return
    try:
      content_length = int(self.headers.get("Content-Length", "0"))
    except ValueError:
      content_length = 0
    body = self.rfile.read(content_length)
    status, response_body, response_content_type = self.endpoint.handle_payload(body, self.headers.get("Content-Type", ""))

    self.send_response(status)
    self.send_header("Content-Type", response_content_type)
    self.send_header("Content-Length", str(len(response_body)))
    self.end_headers()
    self.wfile.write(response_body)

  def log_message(self, fmt: str, *args: Any) -> None:
    return


def make_server(host: str, port: int, adapter: AlpamayoAdapter, trace_logger: PcTraceLogger | None = None) -> ThreadingHTTPServer:
  endpoint = AlpamayoPcEndpoint(adapter, trace_logger)

  class Handler(AlpamayoRequestHandler):
    pass

  Handler.endpoint = endpoint
  return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
  parser = argparse.ArgumentParser(description="Strict PC-side Alpamayo inference endpoint.")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--adapter", default=DEFAULT_ADAPTER_SPEC, help="Python adapter as module:attribute exposing infer(request).")
  parser.add_argument("--trace-path", default=None)
  args = parser.parse_args()

  adapter = load_adapter(args.adapter)
  server = make_server(args.host, args.port, adapter, PcTraceLogger(args.trace_path))
  print(f"alpamayo PC endpoint listening on http://{args.host}:{args.port}/alpamayo", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
