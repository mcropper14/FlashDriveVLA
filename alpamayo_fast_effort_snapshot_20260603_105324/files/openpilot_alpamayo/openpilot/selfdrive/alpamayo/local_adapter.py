#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import math
import threading
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.selfdrive.alpamayo.protocol import FRAME_ENCODING_JPEG_BGR, FRAME_ENCODING_NV12, xyzt_to_dict


STREAM_TO_ALPAMAYO_CAMERA_INDEX = {
  "wideRoad": 1,  # Alpamayo/Qwen prompt name: Front camera
  "road": 6,      # Alpamayo/Qwen prompt name: Front telephoto camera
}
DEFAULT_NUM_HISTORY_STEPS = 16
DEFAULT_HISTORY_DT_S = 0.1
DEFAULT_MAX_GENERATION_LENGTH = 16
DEFAULT_STATIC_GRAPH_MAX_PROMPT_TOKENS = 4096
DEFAULT_STATIC_GRAPH_MAX_VISUAL_TOKENS = 4096
DEFAULT_TARGET_MODEL_WIN = Path(r"E:\ture_opamayo\openpilot_alpamayo\Alpamayo-1.5-10B-finetuned")
DEFAULT_ALPAMAYO_ROOT_WIN = Path(r"G:\alpamayo1.5")
DEFAULT_DFLASH_DRAFT_MODEL_WIN = Path(r"E:\ture_opamayo\openpilot_alpamayo\Alpamayo-1.5-DFlash")
DEFAULT_DFLASH_PACKAGE_ROOT_WIN = Path(r"E:\ture_opamayo\openpilot_alpamayo\dflash")
DEFAULT_TARGET_MODEL_WSL = Path("/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-10B-finetuned")
DEFAULT_ALPAMAYO_ROOT_WSL = Path("/mnt/g/alpamayo1.5")
DEFAULT_DFLASH_DRAFT_MODEL_WSL = Path("/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-DFlash")
DEFAULT_DFLASH_PACKAGE_ROOT_WSL = Path("/mnt/e/ture_opamayo/openpilot_alpamayo/dflash")
REQUIRED_REASONING_MODE = "full"
FLASHVLA_TARGET_TERMS = (
  "flashvla",
  "flash-vla",
  "flash_vla",
  "flashdrivevla",
  "flashdrive-vla",
  "flashdrive_vla",
  "flashdrive finetuned",
)
FLASHDRIVEVLA_TARGET_MODEL_NAMES = (
  "alpamayo-1.5-10b-finetuned",
  "alpamayo-1.5-10b-finetuned-paro",
  "alpamayo-r1-10b-finetuned",
  "alpamayo-r1-10b-finetuned-paro",
)
OPENPILOT_TOKENIZED_METADATA_KEYS = (
  "_openpilot_prefix_semantic_signature",
  "_openpilot_visual_token_count",
  "_openpilot_fused_input_ids_signature",
)


def _is_truthy_skip_flag(value: Any) -> bool:
  if isinstance(value, str):
    return value.strip().lower() in ("1", "true", "yes", "on")
  return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LocalAlpamayoConfig:
  alpamayo_root: Path
  target_model: Path
  camera_streams: tuple[str, ...] = ("wideRoad", "road")
  min_pixels: int = 65536
  max_pixels: int = 65536
  num_frames: int = 4
  diffusion_steps: int = 6
  adaptive_flow_enabled: bool = True
  adaptive_flow_min_steps: int = 1
  adaptive_flow_schedule: str = "cosine_ease"
  adaptive_flow_overlap_threshold: float = 0.5
  adaptive_flow_reuse_middle_velocity: bool = True
  adaptive_flow_reuse_initial_noise: bool = True
  adaptive_flow_action_cache_reuse: bool = True
  adaptive_flow_cache_max_entries: int = 4
  max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH
  gpu_mem_gib: int = 15
  cpu_mem_gib: int = 96
  split_index: int = 16
  attn_implementation: str = "flash_attention_2"
  expert_attn_implementation: str = "eager"
  greedy: bool = True
  manual_generation: bool = True
  skip_vlm_generation: bool = False
  disable_reasoning_generation: bool = False
  no_reasoning_trust_shifted_prompt_cache: bool = False
  require_state_fresh_no_reasoning: bool = True
  device_map_mode: str = "current_split"
  model_dtype: str = "bfloat16"
  autocast_dtype: str = "bfloat16"
  paro_native: bool = False
  paro_compute_dtype: str = "float16"
  paro_output_dtype: str = "native"
  paro_marlin_input_dtype: str = "int8"
  paro_require_cuda_modules: bool = False
  paro_activation_int8: bool = False
  paro_fast_prefill: bool = False
  paro_require_fast_prefill: bool = False
  torch_cuda_arch_list: str = "8.9+PTX"
  cuda_graphs: bool = True
  cuda_graph_cache_size: int = 4
  cuda_graph_capture_vlm_decode: bool = False
  static_graph_strict_shapes: bool = True
  static_graph_max_prompt_tokens: int = DEFAULT_STATIC_GRAPH_MAX_PROMPT_TOKENS
  static_graph_max_visual_tokens: int = DEFAULT_STATIC_GRAPH_MAX_VISUAL_TOKENS
  graph_visual_stage: bool = True
  graph_prefill_stage: bool = True
  graph_standard_prefill_stage: bool = False
  graph_draft_verify_prefill_stage: bool = True
  graph_decode_stage: bool = True
  graph_action_stage: bool = True
  streaming_vision_cache: bool = True
  streaming_vision_attention_mask: bool = False
  vlm_runtime_backend: str = "torch"
  persistent_vlm_prefix_cache: bool = True
  vlm_prefix_cache_max_entries: int = 4
  streaming_vlm_prefix_reuse: bool = True
  streaming_vlm_trust_shifted_draft: bool = True
  streaming_vlm_source_cache_draft_verify_unverified: bool = False
  streaming_vlm_trusted_replay_refresh_interval: int = 24
  streaming_vlm_prefix_reuse_min_overlap: float = 0.5
  streaming_vlm_prefix_reuse_max_chain: int = 128
  dflash_enabled: bool = True
  dflash_draft_model: Path = DEFAULT_DFLASH_DRAFT_MODEL_WIN
  dflash_package_root: Path = DEFAULT_DFLASH_PACKAGE_ROOT_WIN
  dflash_draft_device: str = "lm_head"
  dflash_attn_implementation: str = "flash_attention_2"
  dflash_min_acceptance_rate: float = 0.0
  dflash_max_time_to_first_token_ms: float = 0.0
  dflash_max_decode_ms: float = 0.0
  dflash_max_total_ms: float = 0.0
  dflash_retry_cooldown_frames: int = 0
  dflash_graph_capture: bool = False
  processor_model: str | None = None
  source: str = "remoteServer"
  require_flashvla_target: bool = True


def _running_under_wsl() -> bool:
  return "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False


def _default_path(wsl_path: Path, win_path: Path) -> Path:
  return wsl_path if _running_under_wsl() else win_path


def _env_path(name: str, default: Path) -> Path:
  return Path(os.environ.get(name, str(default)))


def _env_path_any(names: tuple[str, ...], default: Path) -> Path:
  for name in names:
    value = os.environ.get(name)
    if value:
      return Path(value)
  return default


def _env_int(name: str, default: int) -> int:
  try:
    return int(os.environ.get(name, str(default)))
  except ValueError:
    return default


def _env_float(name: str, default: float) -> float:
  try:
    return float(os.environ.get(name, str(default)))
  except ValueError:
    return default


def _env_bool(name: str, default: bool) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _CachedCudaGraph:
  key: tuple[Any, ...]
  graph: Any
  buffered_inputs: dict[str, Any]
  outputs: tuple[Any, Any, dict[str, Any]]
  input_signature: tuple[tuple[Any, ...], ...]
  runtime_profile: dict[str, float | int]


@dataclass
class _StreamingVisionFrameSlot:
  stream: str
  signature: tuple[Any, ...]
  frame_id: int
  timestamp_eof: int
  prompt_index: int
  frame_index: int
  token_start: int
  token_count: int
  pre_rope_key_cache: Any = None
  value_cache: Any = None
  rope_applied_key_cache: Any = None
  pre_rope_key_cache_by_layer: dict[Any, Any] | None = None
  value_cache_by_layer: dict[Any, Any] | None = None
  rope_applied_key_cache_by_layer: dict[Any, Any] | None = None


class StreamingAlpamayoVisionCache:
  """Tracks FlashDrive-style per-camera visual-token reuse across sliding windows.

  This is the adapter-side boundary for streaming vision reuse. It deliberately
  keeps pre-RoPE key/value tensors separate from RoPE-applied keys so a model
  patch can later re-materialize keys at shifted positions instead of treating
  stale post-RoPE KV as reusable.
  """

  def __init__(self, camera_streams: tuple[str, ...], num_frames: int, max_entries_per_stream: int):
    self.camera_streams = camera_streams
    self.num_frames = num_frames
    self.max_entries_per_stream = max(1, max_entries_per_stream)
    self._lock = threading.Lock()
    self._slots: dict[str, OrderedDict[tuple[Any, ...], _StreamingVisionFrameSlot]] = {
      stream: OrderedDict() for stream in camera_streams
    }
    self._last_attention_mask = None
    self._last_token_blocks: list[dict[str, Any]] = []

  @staticmethod
  def _empty_stats(enabled: bool) -> dict[str, Any]:
    return {
      "enabled": enabled,
      "frame_hits": 0,
      "frame_misses": 0,
      "retained_frames": 0,
      "new_frames": 0,
      "shifted_frames": 0,
      "stale_entries": 0,
      "evicted_entries": 0,
      "evicted_stale_entries": 0,
      "pre_rope_kv_slots": 0,
      "pre_rope_kv_layer_slots": 0,
      "pre_rope_kv_materialized": 0,
      "rope_reapply_needed_frames": 0,
      "cache_reuse_attempted": 0,
      "cache_reuse_blocks": 0,
      "cache_reuse_tokens": 0,
      "cache_reuse_misses": 0,
      "cache_reuse_shape_mismatches": 0,
      "cache_reuse_errors": 0,
      "rope_reapplied_blocks": 0,
      "streaming_attention_fastpath_calls": 0,
      "streaming_attention_fallback_calls": 0,
      "attention_mask_built": 0,
      "attention_mask_applied": 0,
      "attention_mask_apply_mismatches": 0,
      "attention_mask_missing": 0,
      "attention_mask_shape": [],
      "attention_mask_true_ratio": 0.0,
      "attention_mask_backend": "",
      "token_blocks": [],
      "cache_depth_by_stream": {},
    }

  @staticmethod
  def _grid_rows(image_grid_thw: Any) -> list[list[int]]:
    if image_grid_thw is None:
      return []
    try:
      if hasattr(image_grid_thw, "detach"):
        rows = image_grid_thw.detach().cpu().tolist()
      else:
        rows = list(image_grid_thw)
      return [[int(value) for value in row] for row in rows]
    except Exception:
      return []

  @staticmethod
  def _visual_token_counts(image_grid_thw: Any, merge_size: int) -> list[int]:
    rows = StreamingAlpamayoVisionCache._grid_rows(image_grid_thw)
    merge_area = max(1, int(merge_size) * int(merge_size))
    token_counts = []
    for row in rows:
      if len(row) != 3:
        token_counts.append(0)
        continue
      t, h, w = row
      token_counts.append(max(0, int(t) * int(h) * int(w) // merge_area))
    return token_counts

  @staticmethod
  def _build_view_major_attention_mask(torch_mod: Any, token_blocks: list[dict[str, Any]]) -> Any:
    total_tokens = sum(int(block["token_count"]) for block in token_blocks)
    if total_tokens <= 0:
      return None

    mask = torch_mod.zeros((total_tokens, total_tokens), dtype=torch_mod.bool)
    for query_block in token_blocks:
      q_start = int(query_block["token_start"])
      q_count = int(query_block["token_count"])
      q_end = q_start + q_count
      q_stream = int(query_block["stream_index"])
      q_frame = int(query_block["frame_index"])
      for key_block in token_blocks:
        k_start = int(key_block["token_start"])
        k_count = int(key_block["token_count"])
        k_end = k_start + k_count
        k_stream = int(key_block["stream_index"])
        k_frame = int(key_block["frame_index"])
        if q_count <= 0 or k_count <= 0:
          continue

        allow_full_block = False
        allow_causal_self_block = False
        if k_stream < q_stream and k_frame <= q_frame:
          allow_full_block = True
        elif k_stream == q_stream and k_frame < q_frame:
          allow_full_block = True
        elif k_stream == q_stream and k_frame == q_frame:
          allow_causal_self_block = True

        if allow_full_block:
          mask[q_start:q_end, k_start:k_end] = True
        elif allow_causal_self_block:
          mask[q_start:q_end, k_start:k_end] = torch_mod.tril(
            torch_mod.ones((q_count, k_count), dtype=torch_mod.bool)
          )
    return mask.unsqueeze(0).unsqueeze(0)

  def prepare(
    self,
    frames_by_stream: dict[str, list[dict[str, Any]]],
    image_grid_thw: Any,
    merge_size: int,
    torch_mod: Any,
  ) -> dict[str, Any]:
    stats = self._empty_stats(True)
    token_counts = self._visual_token_counts(image_grid_thw, merge_size)
    token_blocks: list[dict[str, Any]] = []
    token_start = 0
    prompt_index = 0

    with self._lock:
      for stream_index, stream in enumerate(self.camera_streams):
        stream_cache = self._slots.setdefault(stream, OrderedDict())
        active_signatures: set[tuple[Any, ...]] = set()
        for frame_index, frame in enumerate(frames_by_stream.get(stream, [])):
          signature = _frame_cache_signature(frame)
          active_signatures.add(signature)
          token_count = token_counts[prompt_index] if prompt_index < len(token_counts) else 0
          frame_id = int(frame.get("frameId", -1))
          timestamp_eof = int(frame.get("timestampEof", 0))
          existing = stream_cache.get(signature)
          if existing is None:
            stats["frame_misses"] += 1
            stats["new_frames"] += 1
            slot = _StreamingVisionFrameSlot(
              stream=stream,
              signature=signature,
              frame_id=frame_id,
              timestamp_eof=timestamp_eof,
              prompt_index=prompt_index,
              frame_index=frame_index,
              token_start=token_start,
              token_count=token_count,
            )
            stream_cache[signature] = slot
          else:
            stats["frame_hits"] += 1
            stats["retained_frames"] += 1
            if existing.frame_index != frame_index or existing.prompt_index != prompt_index:
              stats["shifted_frames"] += 1
            if existing.token_start != token_start or existing.token_count != token_count:
              stats["rope_reapply_needed_frames"] += 1
              existing.rope_applied_key_cache = None
              existing.rope_applied_key_cache_by_layer = None
            existing.frame_id = frame_id
            existing.timestamp_eof = timestamp_eof
            existing.prompt_index = prompt_index
            existing.frame_index = frame_index
            existing.token_start = token_start
            existing.token_count = token_count
            stream_cache.move_to_end(signature)
            slot = existing

          if slot.pre_rope_key_cache is not None or slot.value_cache is not None:
            stats["pre_rope_kv_slots"] += 1
          if slot.pre_rope_key_cache is not None and slot.value_cache is not None:
            stats["pre_rope_kv_materialized"] += 1
          if slot.pre_rope_key_cache_by_layer is not None and slot.value_cache_by_layer is not None:
            stats["pre_rope_kv_layer_slots"] += min(
              len(slot.pre_rope_key_cache_by_layer),
              len(slot.value_cache_by_layer),
            )

          token_blocks.append({
            "stream": stream,
            "stream_index": stream_index,
            "frame_index": frame_index,
            "prompt_index": prompt_index,
            "frame_id": frame_id,
            "signature": signature,
            "token_start": token_start,
            "token_count": token_count,
          })
          token_start += token_count
          prompt_index += 1

        stale_count = sum(1 for signature in stream_cache if signature not in active_signatures)
        stats["stale_entries"] += stale_count
        for stale_signature in [signature for signature in stream_cache if signature not in active_signatures]:
          stream_cache.pop(stale_signature, None)
          stats["evicted_entries"] += 1
          stats["evicted_stale_entries"] += 1
        while len(stream_cache) > self.max_entries_per_stream:
          stream_cache.popitem(last=False)
          stats["evicted_entries"] += 1

      stats["cache_depth_by_stream"] = {
        stream: len(stream_cache) for stream, stream_cache in self._slots.items()
      }

    attention_mask = self._build_view_major_attention_mask(torch_mod, token_blocks)
    with self._lock:
      self._last_attention_mask = attention_mask
      self._last_token_blocks = token_blocks
    stats["token_blocks"] = token_blocks
    if attention_mask is not None:
      stats["attention_mask_built"] = 1
      stats["attention_mask_shape"] = [int(dim) for dim in attention_mask.shape]
      if not getattr(attention_mask, "is_cuda", False):
        stats["attention_mask_true_ratio"] = float(attention_mask.float().mean().item())
      else:
        stats["attention_mask_true_ratio"] = None
    return stats

  def materialize_attention_mask(
    self,
    torch_mod: Any,
    *,
    device: Any,
    dtype: Any,
    total_tokens: int,
  ) -> tuple[Any | None, dict[str, Any]]:
    stats: dict[str, Any] = {
      "attention_mask_applied": 0,
      "attention_mask_apply_mismatches": 0,
      "attention_mask_missing": 0,
    }
    with self._lock:
      mask = self._last_attention_mask
    if mask is None:
      stats["attention_mask_missing"] = 1
      return None, stats
    if int(mask.shape[-1]) != int(total_tokens) or int(mask.shape[-2]) != int(total_tokens):
      stats["attention_mask_apply_mismatches"] = 1
      stats["attention_mask_expected_tokens"] = int(total_tokens)
      stats["attention_mask_cached_shape"] = [int(dim) for dim in mask.shape]
      return None, stats
    bool_mask = mask.to(device=device, non_blocking=True)
    stats["attention_mask_applied"] = 1
    if dtype == getattr(torch_mod, "bool", None):
      return bool_mask, stats
    zero = torch_mod.zeros((), device=device, dtype=dtype)
    min_value = zero.new_full((), torch_mod.finfo(dtype).min)
    return torch_mod.where(bool_mask, zero, min_value), stats

  def capture_pre_rope_states(
    self,
    key_states: Any,
    value_states: Any,
    layer_key: Any | None = None,
    token_blocks: list[dict[str, Any]] | None = None,
  ) -> dict[str, Any]:
    stats: dict[str, Any] = {
      "capture_attempted": 1,
      "capture_blocks": 0,
      "capture_tokens": 0,
      "capture_errors": 0,
    }
    if token_blocks is None:
      with self._lock:
        token_blocks = list(self._last_token_blocks)
    else:
      token_blocks = list(token_blocks)
    if not token_blocks:
      return stats

    try:
      seq_len = int(key_states.shape[0])

      for block in token_blocks:
        token_start = int(block["token_start"])
        token_count = int(block["token_count"])
        if token_count <= 0:
          continue
        token_end = token_start + token_count
        if token_start < 0 or token_end > seq_len:
          stats["capture_errors"] += 1
          continue
        if self.update_pre_rope_kv(
          str(block["stream"]),
          tuple(block["signature"]),
          key_states[token_start:token_end],
          value_states[token_start:token_end],
          layer_key=layer_key,
        ):
          stats["capture_blocks"] += 1
          stats["capture_tokens"] += token_count
    except Exception as exc:
      stats["capture_errors"] += 1
      stats["capture_error"] = f"{type(exc).__name__}: {exc}"
    return stats

  def capture_pre_rope_qkv(
    self,
    hidden_states: Any,
    qkv_module: Any,
    attention_module: Any,
    layer_key: Any | None = None,
    token_blocks: list[dict[str, Any]] | None = None,
  ) -> dict[str, Any]:
    stats: dict[str, Any] = {
      "capture_attempted": 1,
      "capture_blocks": 0,
      "capture_tokens": 0,
      "capture_errors": 0,
    }
    try:
      qkv = qkv_module(hidden_states)
      seq_len = int(qkv.shape[0])
      num_heads = int(
        getattr(attention_module, "num_heads", 0)
        or getattr(attention_module, "num_attention_heads", 0)
        or 0
      )
      if num_heads <= 0:
        raise ValueError("Qwen3VLVisionAttention num_heads unavailable")
      head_dim = int(getattr(attention_module, "head_dim", 0) or (int(qkv.shape[-1]) // (3 * num_heads)))
      qkv = qkv.reshape(seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)
      return self.capture_pre_rope_states(qkv[1].detach(), qkv[2].detach(), layer_key=layer_key, token_blocks=token_blocks)
    except Exception as exc:
      stats["capture_errors"] += 1
      stats["capture_error"] = f"{type(exc).__name__}: {exc}"
      return stats

  def apply_cached_pre_rope_kv(
    self,
    key_states: Any,
    value_states: Any,
    layer_key: Any,
    token_blocks: list[dict[str, Any]] | None = None,
  ) -> tuple[Any, Any, dict[str, Any]]:
    stats: dict[str, Any] = {
      "cache_reuse_attempted": 1,
      "cache_reuse_blocks": 0,
      "cache_reuse_tokens": 0,
      "cache_reuse_misses": 0,
      "cache_reuse_shape_mismatches": 0,
      "cache_reuse_errors": 0,
      "rope_reapplied_blocks": 0,
      "cache_reuse_inplace": 0,
    }
    if token_blocks is None:
      with self._lock:
        token_blocks = list(self._last_token_blocks)
    else:
      token_blocks = list(token_blocks)
    with self._lock:
      slots_by_signature = {
        (slot.stream, slot.signature): slot
        for stream_cache in self._slots.values()
        for slot in stream_cache.values()
      }
    if not token_blocks:
      return key_states, value_states, stats

    patched_key_states = key_states
    patched_value_states = value_states
    try:
      seq_len = int(key_states.shape[0])
      for block in token_blocks:
        token_start = int(block["token_start"])
        token_count = int(block["token_count"])
        if token_count <= 0:
          continue
        token_end = token_start + token_count
        if token_start < 0 or token_end > seq_len:
          stats["cache_reuse_shape_mismatches"] += 1
          continue
        slot = slots_by_signature.get((str(block["stream"]), tuple(block["signature"])))
        if slot is None:
          stats["cache_reuse_misses"] += 1
          continue
        key_by_layer = slot.pre_rope_key_cache_by_layer or {}
        value_by_layer = slot.value_cache_by_layer or {}
        cached_key = key_by_layer.get(layer_key)
        cached_value = value_by_layer.get(layer_key)
        if cached_key is None or cached_value is None:
          stats["cache_reuse_misses"] += 1
          continue
        if tuple(cached_key.shape) != tuple(key_states[token_start:token_end].shape) or tuple(cached_value.shape) != tuple(value_states[token_start:token_end].shape):
          stats["cache_reuse_shape_mismatches"] += 1
          continue
        patched_key_states[token_start:token_end] = cached_key.to(
          device=key_states.device,
          dtype=key_states.dtype,
          non_blocking=True,
        )
        patched_value_states[token_start:token_end] = cached_value.to(
          device=value_states.device,
          dtype=value_states.dtype,
          non_blocking=True,
        )
        stats["cache_reuse_blocks"] += 1
        stats["cache_reuse_tokens"] += token_count
        stats["rope_reapplied_blocks"] += 1
      if stats["cache_reuse_blocks"] > 0:
        stats["cache_reuse_inplace"] = 1
        return patched_key_states, patched_value_states, stats
    except Exception as exc:
      stats["cache_reuse_errors"] += 1
      stats["cache_reuse_error"] = f"{type(exc).__name__}: {exc}"
    return key_states, value_states, stats

  def cached_pre_rope_kv(
    self,
    stream: str,
    frame_signature: tuple[Any, ...],
    layer_key: Any,
  ) -> tuple[Any, Any] | None:
    with self._lock:
      slot = self._slots.get(stream, OrderedDict()).get(frame_signature)
      if slot is None:
        return None
      key_by_layer = slot.pre_rope_key_cache_by_layer or {}
      value_by_layer = slot.value_cache_by_layer or {}
      cached_key = key_by_layer.get(layer_key)
      cached_value = value_by_layer.get(layer_key)
      if cached_key is None or cached_value is None:
        return None
      return cached_key, cached_value

  def update_pre_rope_kv(
    self,
    stream: str,
    frame_signature: tuple[Any, ...],
    pre_rope_key_cache: Any,
    value_cache: Any,
    layer_key: Any | None = None,
  ) -> bool:
    with self._lock:
      slot = self._slots.get(stream, OrderedDict()).get(frame_signature)
      if slot is None:
        return False
      pre_rope_key_cache = pre_rope_key_cache.detach().clone()
      value_cache = value_cache.detach().clone()
      slot.pre_rope_key_cache = pre_rope_key_cache
      slot.value_cache = value_cache
      slot.rope_applied_key_cache = None
      if layer_key is not None:
        if slot.pre_rope_key_cache_by_layer is None:
          slot.pre_rope_key_cache_by_layer = {}
        if slot.value_cache_by_layer is None:
          slot.value_cache_by_layer = {}
        if slot.rope_applied_key_cache_by_layer is None:
          slot.rope_applied_key_cache_by_layer = {}
        slot.pre_rope_key_cache_by_layer[layer_key] = pre_rope_key_cache
        slot.value_cache_by_layer[layer_key] = value_cache
        slot.rope_applied_key_cache_by_layer.pop(layer_key, None)
      return True

  def last_token_blocks(self) -> list[dict[str, Any]]:
    with self._lock:
      return list(self._last_token_blocks)


_STREAMING_VISION_PATCH_CONTEXT = threading.local()


def _freeze_cache_key_value(value: Any) -> Any:
  if isinstance(value, dict):
    return tuple(
      (str(key), _freeze_cache_key_value(item))
      for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    )
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_cache_key_value(item) for item in value)
  if isinstance(value, set):
    return tuple(sorted((_freeze_cache_key_value(item) for item in value), key=repr))
  try:
    hash(value)
    return value
  except TypeError:
    return repr(value)


def _tensor_tree_signature(torch_mod: Any, value: Any, prefix: tuple[Any, ...] = ()) -> tuple[tuple[Any, ...], ...]:
  signature: list[tuple[Any, ...]] = []

  def _walk(node: Any, path: tuple[Any, ...]) -> None:
    if isinstance(node, dict):
      for key in node.keys():
        _walk(node[key], path + (f"dict:{key}",))
      return
    if isinstance(node, list):
      for idx, item in enumerate(node):
        _walk(item, path + (f"list:{idx}",))
      return
    if isinstance(node, tuple):
      for idx, item in enumerate(node):
        _walk(item, path + (f"tuple:{idx}",))
      return
    if isinstance(node, torch_mod.Tensor):
      signature.append(
        (
          path,
          tuple(int(item) for item in node.shape),
          tuple(int(item) for item in node.stride()),
          str(node.device),
          str(node.dtype),
          node.requires_grad,
        )
      )
      return
    signature.append((path, type(node).__name__))

  _walk(value, prefix)
  return tuple(signature)


def _tensor_tree_content_signature(
  torch_mod: Any,
  value: Any,
  prefix: tuple[Any, ...] = (),
  *,
  max_elements: int = 65536,
) -> tuple[tuple[Any, ...], ...]:
  signature: list[tuple[Any, ...]] = []

  def _walk(node: Any, path: tuple[Any, ...]) -> None:
    if isinstance(node, dict):
      for key in sorted(node.keys()):
        _walk(node[key], path + (f"dict:{key}",))
      return
    if isinstance(node, list):
      for idx, item in enumerate(node):
        _walk(item, path + (f"list:{idx}",))
      return
    if isinstance(node, tuple):
      for idx, item in enumerate(node):
        _walk(item, path + (f"tuple:{idx}",))
      return
    if isinstance(node, torch_mod.Tensor):
      meta = (
        path,
        tuple(int(item) for item in node.shape),
        tuple(int(item) for item in node.stride()),
        str(node.dtype),
        node.requires_grad,
      )
      data_ptr = int(node.data_ptr()) if hasattr(node, "data_ptr") else 0
      numel = int(node.numel())
      if getattr(node.device, "type", str(node.device).split(":")[0]) != "cpu":
        signature.append(meta + ("cuda_content_not_synced", numel, data_ptr))
        return
      if numel > max(0, int(max_elements)):
        signature.append(meta + ("content_skipped_large", numel, data_ptr))
        return
      try:
        content = node.detach()
        content = content.contiguous()
        try:
          payload = content.numpy().tobytes()
        except Exception:
          if content.is_floating_point():
            content = content.to(dtype=torch_mod.float32)
          else:
            content = content.to(dtype=torch_mod.int64)
          payload = content.contiguous().numpy().tobytes()
        signature.append(meta + ("crc32", zlib.crc32(payload) & 0xFFFFFFFF, int(content.numel())))
      except Exception as exc:
        signature.append(meta + ("content_unavailable", type(exc).__name__, data_ptr))
      return
    signature.append((path, type(node).__name__, node))

  _walk(value, prefix)
  return tuple(signature)


def _prefix_semantic_signature(torch_mod: Any, tokenized_data: dict[str, Any]) -> tuple[Any, ...]:
  return (
    "prefix_semantic_v1",
    _tensor_tree_content_signature(
      torch_mod,
      {
        key: tokenized_data.get(key)
        for key in (
          "input_ids",
          "attention_mask",
          "image_grid_thw",
          "video_grid_thw",
        )
        if key in tokenized_data
      },
    ),
  )


def _tensor_crc32_signature(torch_mod: Any, value: Any, *, max_elements: int = 32768) -> tuple[Any, ...]:
  try:
    if torch_mod is None or not isinstance(value, torch_mod.Tensor):
      return ()
    tensor = value.detach()
    numel = int(tensor.numel())
    meta = (
      "tensor_crc32_v1",
      tuple(int(item) for item in tensor.shape),
      str(tensor.dtype),
      numel,
    )
    if numel > max(0, int(max_elements)):
      return meta + ("content_skipped_large",)
    if getattr(tensor.device, "type", str(tensor.device).split(":")[0]) != "cpu":
      tensor = tensor.to("cpu", non_blocking=False)
    tensor = tensor.contiguous()
    return meta + (zlib.crc32(tensor.numpy().tobytes()) & 0xFFFFFFFF,)
  except Exception as exc:
    return ("tensor_crc32_error", type(exc).__name__)


def _visual_token_count_from_grids(image_grid: Any, video_grid: Any, merge_size: int) -> int:
  merge_area = max(1, int(merge_size) * int(merge_size))
  visual_tokens = 0
  for grid in (image_grid, video_grid):
    if grid is None or not hasattr(grid, "prod"):
      continue
    if getattr(grid, "is_cuda", False):
      return -1
    try:
      visual_tokens += int((grid.prod(dim=1) // merge_area).sum().item())
    except Exception:
      return -1
  return int(visual_tokens)


def _clone_tensor_tree(torch_mod: Any, value: Any) -> Any:
  if isinstance(value, torch_mod.Tensor):
    return torch_mod.empty_like(value)
  if isinstance(value, (bool, int, float, str)) or value is None:
    return value
  if isinstance(value, list):
    return [_clone_tensor_tree(torch_mod, item) for item in value]
  if isinstance(value, tuple):
    return tuple(_clone_tensor_tree(torch_mod, item) for item in value)
  if isinstance(value, dict):
    return {key: _clone_tensor_tree(torch_mod, item) for key, item in value.items()}
  return None


def _copy_to_tensor_tree(torch_mod: Any, destination: Any, source: Any) -> bool:
  if isinstance(destination, torch_mod.Tensor):
    if not isinstance(source, torch_mod.Tensor):
      return False
    if tuple(destination.shape) != tuple(source.shape):
      return False
    if destination.dtype != source.dtype:
      return False
    destination.copy_(source)
    return True
  if destination is None:
    return source is None
  if isinstance(destination, (bool, int, float, str, np.generic)):
    return isinstance(source, type(destination))
  if isinstance(destination, (bytes, bytearray)):
    return False
  if isinstance(destination, list):
    if not isinstance(source, list) or len(destination) != len(source):
      return False
    return all(_copy_to_tensor_tree(torch_mod, d, s) for d, s in zip(destination, source))
  if isinstance(destination, tuple):
    if not isinstance(source, tuple) or len(destination) != len(source):
      return False
    return all(_copy_to_tensor_tree(torch_mod, d, s) for d, s in zip(destination, source))
  if isinstance(destination, dict):
    if not isinstance(source, dict) or set(destination.keys()) != set(source.keys()):
      return False
    for key in destination:
      if not _copy_to_tensor_tree(torch_mod, destination[key], source[key]):
        return False
    return True
  return False


def _copy_tensor_tree_profile(dst: dict[str, Any], src: dict[str, Any]) -> None:
  for key, value in src.items():
    if key.startswith("cuda_graph_"):
      continue
    dst[key] = value


def _has_full_pipeline_profile(profile: dict[str, Any]) -> bool:
  required_stage_keys = (
    "vlm_generate_seconds",
    "expert_step_calls",
    "diffusion_seconds",
    "action_to_traj_seconds",
    "generated_sequence_length",
  )
  return all(key in profile for key in required_stage_keys)


def _model_identity_text(path: Path) -> str:
  chunks = [str(path), path.name]
  for name in ("config.json", "README.md", "model_index.json"):
    candidate = path / name
    if not candidate.exists():
      continue
    try:
      chunks.append(candidate.read_text(encoding="utf-8", errors="ignore")[:262144])
    except Exception:
      continue
  return "\n".join(chunks).lower()


def _is_flashvla_target_path(path: Path) -> bool:
  identity = _model_identity_text(path)
  normalized_name = path.name.strip().lower()
  if normalized_name in FLASHDRIVEVLA_TARGET_MODEL_NAMES:
    return True
  return any(term in identity for term in FLASHVLA_TARGET_TERMS)


def config_from_env() -> LocalAlpamayoConfig:
  default_root = _default_path(DEFAULT_ALPAMAYO_ROOT_WSL, DEFAULT_ALPAMAYO_ROOT_WIN)
  default_target = _default_path(DEFAULT_TARGET_MODEL_WSL, DEFAULT_TARGET_MODEL_WIN)
  default_dflash_model = _default_path(DEFAULT_DFLASH_DRAFT_MODEL_WSL, DEFAULT_DFLASH_DRAFT_MODEL_WIN)
  default_dflash_package_root = _default_path(DEFAULT_DFLASH_PACKAGE_ROOT_WSL, DEFAULT_DFLASH_PACKAGE_ROOT_WIN)
  streams = tuple(part.strip() for part in os.environ.get("ALPAMAYO_CAMERA_STREAMS", "wideRoad,road").split(",") if part.strip())
  return LocalAlpamayoConfig(
    alpamayo_root=_env_path("ALPAMAYO_ROOT", default_root),
    target_model=_env_path_any(("ALPAMAYO_TARGET_MODEL", "FLASHVLA_TARGET_MODEL", "FLASHDRIVEVLA_TARGET_MODEL"), default_target),
    camera_streams=streams or ("wideRoad", "road"),
    min_pixels=_env_int("ALPAMAYO_MIN_PIXELS", 65536),
    max_pixels=_env_int("ALPAMAYO_MAX_PIXELS", 65536),
    num_frames=_env_int("ALPAMAYO_NUM_FRAMES", 4),
    diffusion_steps=_env_int("ALPAMAYO_DIFFUSION_STEPS", 6),
    adaptive_flow_enabled=_env_bool("ALPAMAYO_ADAPTIVE_FLOW_ENABLED", True),
    adaptive_flow_min_steps=_env_int("ALPAMAYO_ADAPTIVE_FLOW_MIN_STEPS", 1),
    adaptive_flow_schedule=os.environ.get("ALPAMAYO_ADAPTIVE_FLOW_SCHEDULE", "cosine_ease"),
    adaptive_flow_overlap_threshold=_env_float("ALPAMAYO_ADAPTIVE_FLOW_OVERLAP_THRESHOLD", 0.5),
    adaptive_flow_reuse_middle_velocity=_env_bool("ALPAMAYO_ADAPTIVE_FLOW_REUSE_MIDDLE_VELOCITY", True),
    adaptive_flow_reuse_initial_noise=_env_bool("ALPAMAYO_ADAPTIVE_FLOW_REUSE_INITIAL_NOISE", True),
    adaptive_flow_action_cache_reuse=_env_bool("ALPAMAYO_ADAPTIVE_FLOW_ACTION_CACHE_REUSE", True),
    adaptive_flow_cache_max_entries=_env_int("ALPAMAYO_ADAPTIVE_FLOW_CACHE_MAX_ENTRIES", 4),
    max_generation_length=_env_int("ALPAMAYO_MAX_GENERATION_LENGTH", DEFAULT_MAX_GENERATION_LENGTH),
    gpu_mem_gib=_env_int("ALPAMAYO_GPU_MEM_GIB", 15),
    cpu_mem_gib=_env_int("ALPAMAYO_CPU_MEM_GIB", 96),
    split_index=_env_int("ALPAMAYO_SPLIT_INDEX", 16),
    attn_implementation=os.environ.get("ALPAMAYO_ATTN_IMPLEMENTATION", "flash_attention_2"),
    expert_attn_implementation=os.environ.get("ALPAMAYO_EXPERT_ATTN_IMPLEMENTATION", "eager"),
    greedy=_env_bool("ALPAMAYO_GREEDY", True),
    manual_generation=_env_bool("ALPAMAYO_MANUAL_GENERATION", True),
    skip_vlm_generation=_env_bool("ALPAMAYO_SKIP_VLM_GENERATION", False),
    disable_reasoning_generation=_env_bool("ALPAMAYO_DISABLE_REASONING_GENERATION", False),
    no_reasoning_trust_shifted_prompt_cache=_env_bool("ALPAMAYO_NO_REASONING_TRUST_SHIFTED_PROMPT_CACHE", False),
    require_state_fresh_no_reasoning=_env_bool("ALPAMAYO_REQUIRE_STATE_FRESH_NO_REASONING", True),
    device_map_mode=os.environ.get("ALPAMAYO_DEVICE_MAP_MODE", "current_split"),
    model_dtype=os.environ.get("ALPAMAYO_MODEL_DTYPE", "bfloat16"),
    autocast_dtype=os.environ.get("ALPAMAYO_AUTOCAST_DTYPE", "bfloat16"),
    paro_native=_env_bool("ALPAMAYO_PARO_NATIVE", False),
    paro_compute_dtype=os.environ.get("ALPAMAYO_PARO_COMPUTE_DTYPE", "float16"),
    paro_output_dtype=os.environ.get("ALPAMAYO_PARO_OUTPUT_DTYPE", "native"),
    paro_marlin_input_dtype=os.environ.get("ALPAMAYO_PARO_MARLIN_INPUT_DTYPE", "int8"),
    paro_require_cuda_modules=_env_bool("ALPAMAYO_PARO_REQUIRE_CUDA_MODULES", False),
    paro_activation_int8=_env_bool("ALPAMAYO_PARO_ACTIVATION_INT8", False),
    paro_fast_prefill=_env_bool("ALPAMAYO_PARO_FAST_PREFILL", False),
    paro_require_fast_prefill=_env_bool("ALPAMAYO_PARO_REQUIRE_FAST_PREFILL", False),
    cuda_graphs=_env_bool("ALPAMAYO_CUDA_GRAPHS", True),
    cuda_graph_cache_size=_env_int("ALPAMAYO_CUDA_GRAPH_CACHE_SIZE", 4),
    cuda_graph_capture_vlm_decode=_env_bool("ALPAMAYO_CUDA_GRAPH_CAPTURE_VLM_DECODE", False),
    static_graph_strict_shapes=_env_bool("ALPAMAYO_STATIC_GRAPH_STRICT_SHAPES", True),
    static_graph_max_prompt_tokens=_env_int("ALPAMAYO_STATIC_GRAPH_MAX_PROMPT_TOKENS", DEFAULT_STATIC_GRAPH_MAX_PROMPT_TOKENS),
    static_graph_max_visual_tokens=_env_int("ALPAMAYO_STATIC_GRAPH_MAX_VISUAL_TOKENS", DEFAULT_STATIC_GRAPH_MAX_VISUAL_TOKENS),
    graph_visual_stage=_env_bool("ALPAMAYO_GRAPH_VISUAL_STAGE", True),
    graph_prefill_stage=_env_bool("ALPAMAYO_GRAPH_PREFILL_STAGE", True),
    graph_standard_prefill_stage=_env_bool("ALPAMAYO_GRAPH_STANDARD_PREFILL_STAGE", False),
    graph_draft_verify_prefill_stage=_env_bool("ALPAMAYO_GRAPH_DRAFT_VERIFY_PREFILL_STAGE", True),
    graph_decode_stage=_env_bool("ALPAMAYO_GRAPH_DECODE_STAGE", True),
    graph_action_stage=_env_bool("ALPAMAYO_GRAPH_ACTION_STAGE", True),
    streaming_vision_cache=_env_bool("ALPAMAYO_STREAMING_VISION_CACHE", True),
    streaming_vision_attention_mask=_env_bool("ALPAMAYO_STREAMING_VISION_ATTENTION_MASK", False),
    vlm_runtime_backend=os.environ.get("ALPAMAYO_VLM_RUNTIME_BACKEND", "torch"),
    persistent_vlm_prefix_cache=_env_bool("ALPAMAYO_PERSISTENT_VLM_PREFIX_CACHE", True),
    vlm_prefix_cache_max_entries=_env_int("ALPAMAYO_VLM_PREFIX_CACHE_MAX_ENTRIES", 4),
    streaming_vlm_prefix_reuse=_env_bool("ALPAMAYO_STREAMING_VLM_PREFIX_REUSE", True),
    streaming_vlm_trust_shifted_draft=_env_bool("ALPAMAYO_STREAMING_VLM_TRUST_SHIFTED_DRAFT", True),
    streaming_vlm_source_cache_draft_verify_unverified=_env_bool(
      "ALPAMAYO_STREAMING_VLM_SOURCE_CACHE_DRAFT_VERIFY_UNVERIFIED",
      False,
    ),
    streaming_vlm_trusted_replay_refresh_interval=_env_int("ALPAMAYO_STREAMING_VLM_TRUSTED_REPLAY_REFRESH_INTERVAL", 24),
    streaming_vlm_prefix_reuse_min_overlap=_env_float("ALPAMAYO_STREAMING_VLM_PREFIX_REUSE_MIN_OVERLAP", 0.5),
    streaming_vlm_prefix_reuse_max_chain=_env_int("ALPAMAYO_STREAMING_VLM_PREFIX_REUSE_MAX_CHAIN", 128),
    dflash_enabled=_env_bool("ALPAMAYO_DFLASH_ENABLED", True),
    dflash_draft_model=_env_path("ALPAMAYO_DFLASH_DRAFT_MODEL", default_dflash_model),
    dflash_package_root=_env_path("ALPAMAYO_DFLASH_PACKAGE_ROOT", default_dflash_package_root),
    dflash_draft_device=os.environ.get("ALPAMAYO_DFLASH_DRAFT_DEVICE", "lm_head"),
    dflash_attn_implementation=os.environ.get("ALPAMAYO_DFLASH_ATTN_IMPLEMENTATION", "flash_attention_2"),
    dflash_min_acceptance_rate=_env_float("ALPAMAYO_DFLASH_MIN_ACCEPTANCE_RATE", 0.0),
    dflash_max_time_to_first_token_ms=_env_float("ALPAMAYO_DFLASH_MAX_TIME_TO_FIRST_TOKEN_MS", 0.0),
    dflash_max_decode_ms=_env_float("ALPAMAYO_DFLASH_MAX_DECODE_MS", 0.0),
    dflash_max_total_ms=_env_float("ALPAMAYO_DFLASH_MAX_TOTAL_MS", 0.0),
    dflash_retry_cooldown_frames=_env_int("ALPAMAYO_DFLASH_RETRY_COOLDOWN_FRAMES", 0),
    dflash_graph_capture=_env_bool("ALPAMAYO_DFLASH_GRAPH_CAPTURE", False),
    torch_cuda_arch_list=os.environ.get("ALPAMAYO_TORCH_CUDA_ARCH_LIST", "8.9+PTX"),
    processor_model=os.environ.get("ALPAMAYO_PROCESSOR_MODEL"),
    source=os.environ.get("ALPAMAYO_SEMANTIC_SOURCE", "remoteServer"),
    require_flashvla_target=_env_bool("ALPAMAYO_REQUIRE_FLASHVLA_TARGET", True),
  )


def _insert_alpamayo_paths(alpamayo_root: Path) -> None:
  for path in (alpamayo_root, alpamayo_root / "src"):
    text = str(path)
    if text not in sys.path:
      sys.path.insert(0, text)


def _build_current_split_device_map(split_index: int) -> dict[str, int]:
  device_map: dict[str, int] = {
    "vlm.model.visual": 0,
    "vlm.model.language_model.embed_tokens": 0,
    "vlm.model.language_model.norm": 1,
    "vlm.model.language_model.rotary_emb": 1,
    "vlm.lm_head": 1,
    "expert.norm": 1,
    "expert.rotary_emb": 1,
    "action_in_proj": 0,
    "action_out_proj": 1,
    "action_space": 0,
    "diffusion": 0,
  }
  for layer_idx in range(36):
    device = 0 if layer_idx < split_index else 1
    device_map[f"vlm.model.language_model.layers.{layer_idx}"] = device
    device_map[f"expert.layers.{layer_idx}"] = device
  return device_map


def _build_single_device_graph_map(device: int = 0) -> dict[str, int]:
  return {"": device}


def _gpu_only_max_memory(torch_mod: Any, gpu_mem_gib: int) -> dict[int, str]:
  if not torch_mod.cuda.is_available() or torch_mod.cuda.device_count() < 1:
    raise RuntimeError("Alpamayo requires CUDA; CPU model offload is forbidden for latency benchmarks")
  return {idx: f"{gpu_mem_gib}GiB" for idx in range(torch_mod.cuda.device_count())}


def _assert_no_cpu_offload(model: Any) -> None:
  device_map = getattr(model, "hf_device_map", {}) or {}
  bad = {
    name: device
    for name, device in device_map.items()
    if str(device).lower() in ("cpu", "disk") or str(device).lower().startswith("cpu")
  }
  if bad:
    preview = dict(list(bad.items())[:12])
    raise RuntimeError(f"CPU/disk offload is forbidden for Alpamayo latency runs; offending device_map entries={preview}")


def _model_uses_single_cuda_device(model: Any) -> bool:
  devices = set()
  for parameter in model.parameters():
    devices.add(str(getattr(parameter, "device", "")))
    if len(devices) > 1:
      return False
  for buffer in model.buffers():
    devices.add(str(getattr(buffer, "device", "")))
    if len(devices) > 1:
      return False
  if len(devices) != 1:
    return False
  return str(next(iter(devices))).startswith("cuda")


def _set_attention_implementation(module: Any, attn_implementation: str) -> None:
  for submodule in module.modules():
    config = getattr(submodule, "config", None)
    if config is not None:
      setattr(config, "_attn_implementation", attn_implementation)
      if hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", attn_implementation)


def _torch_dtype(torch_mod: Any, name: str) -> Any:
  normalized = name.strip().lower()
  if normalized in ("bfloat16", "bf16"):
    return torch_mod.bfloat16
  if normalized in ("float16", "fp16", "half"):
    return torch_mod.float16
  if normalized in ("float32", "fp32"):
    return torch_mod.float32
  raise ValueError(f"unsupported Alpamayo dtype: {name}")


def _patch_tie_weights_compat() -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  import alpamayo1_5.models.base_model as base_mod

  def tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
    if hasattr(self, "vlm") and hasattr(self.vlm, "tie_weights"):
      return self.vlm.tie_weights()
    return None

  base_mod.ReasoningVLA.tie_weights = tie_weights
  alpamayo_mod.Alpamayo1_5.tie_weights = tie_weights


def _full_generation_exact_current_window(
  prefix_cache_entry: dict[str, Any],
  *,
  signature_key: str = "full_vlm_window_signature",
  context_exact_key: str = "full_vlm_prompt_cache_context_exact",
) -> tuple[bool, bool, bool]:
  current_window_signature = prefix_cache_entry.get("current_window_signature")
  stored_window_signature = prefix_cache_entry.get(signature_key)
  window_signature_match = bool(
    current_window_signature is not None
    and stored_window_signature is not None
    and stored_window_signature == current_window_signature
  )
  prompt_cache_context_exact = bool(prefix_cache_entry.get(context_exact_key))
  return (
    bool(window_signature_match and prompt_cache_context_exact),
    window_signature_match,
    prompt_cache_context_exact,
  )


def _patch_manual_greedy_generation() -> None:
  import copy
  import torch
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

  if getattr(alpamayo_mod.Alpamayo1_5, "_openpilot_manual_greedy_patch", False):
    return

  def _cache_layer_pairs(cache: Any) -> list[tuple[Any, Any]]:
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or not layers:
      return []
    pairs: list[tuple[Any, Any]] = []
    for layer in layers:
      key_states = getattr(layer, "keys", None)
      value_states = getattr(layer, "values", None)
      if key_states is None or value_states is None or not hasattr(key_states, "shape") or not hasattr(value_states, "shape"):
        return []
      pairs.append((key_states, value_states))
    return pairs

  def _dynamic_cache_from_buffered_prefix(
    model_self: Any,
    source_pairs: list[tuple[Any, Any]],
    prefix_len: int,
    *,
    pool_attr: str,
    runtime_profile: dict[str, float | int] | None,
    metric_prefix: str,
  ) -> tuple[Any, list[tuple[Any, Any]]] | None:
    if not source_pairs or int(prefix_len) <= 0:
      return None
    try:
      buffer_pool_enabled = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_BUFFER_POOL", "0")))
    except Exception:
      buffer_pool_enabled = False
    if not buffer_pool_enabled:
      if runtime_profile is not None:
        runtime_profile[f"{metric_prefix}_buffered"] = 0
      return None
    try:
      from transformers.cache_utils import DynamicCache

      pool = getattr(model_self, pool_attr, None)
      if not isinstance(pool, dict):
        pool = {}
        setattr(model_self, pool_attr, pool)

      rebuilt_pairs: list[tuple[Any, Any]] = []
      allocated_layers = 0
      reused_layers = 0
      max_capacity = 0
      for layer_idx, (source_key, source_value) in enumerate(source_pairs):
        if int(source_key.shape[-2]) < int(prefix_len) or int(source_value.shape[-2]) < int(prefix_len):
          return None
        key_capacity = max(int(prefix_len), int(source_key.shape[-2]))
        value_capacity = max(int(prefix_len), int(source_value.shape[-2]))
        max_capacity = max(max_capacity, key_capacity, value_capacity)
        key_shape = tuple(int(dim) for dim in source_key.shape[:-2]) + (key_capacity, int(source_key.shape[-1]))
        value_shape = tuple(int(dim) for dim in source_value.shape[:-2]) + (value_capacity, int(source_value.shape[-1]))
        slot = pool.get(int(layer_idx))
        key_buf = slot[0] if isinstance(slot, tuple) and len(slot) == 2 else None
        value_buf = slot[1] if isinstance(slot, tuple) and len(slot) == 2 else None
        if (
          key_buf is None
          or value_buf is None
          or tuple(int(dim) for dim in key_buf.shape) != key_shape
          or tuple(int(dim) for dim in value_buf.shape) != value_shape
          or key_buf.device != source_key.device
          or value_buf.device != source_value.device
          or key_buf.dtype != source_key.dtype
          or value_buf.dtype != source_value.dtype
        ):
          key_buf = torch.empty(key_shape, device=source_key.device, dtype=source_key.dtype)
          value_buf = torch.empty(value_shape, device=source_value.device, dtype=source_value.dtype)
          pool[int(layer_idx)] = (key_buf, value_buf)
          allocated_layers += 1
        else:
          reused_layers += 1
        target_key = key_buf[..., :prefix_len, :]
        target_value = value_buf[..., :prefix_len, :]
        target_key.copy_(source_key[..., :prefix_len, :], non_blocking=True)
        target_value.copy_(source_value[..., :prefix_len, :], non_blocking=True)
        rebuilt_pairs.append((target_key, target_value))
      if runtime_profile is not None:
        runtime_profile[f"{metric_prefix}_buffered"] = 1
        runtime_profile[f"{metric_prefix}_buffer_allocated_layers"] = int(allocated_layers)
        runtime_profile[f"{metric_prefix}_buffer_reused_layers"] = int(reused_layers)
        runtime_profile[f"{metric_prefix}_buffer_capacity_tokens"] = int(max_capacity)
      return DynamicCache(ddp_cache_data=rebuilt_pairs), rebuilt_pairs
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile[f"{metric_prefix}_buffer_error"] = f"{type(exc).__name__}: {exc}"
      return None

  def _build_shifted_visual_prefix_cache(
    model_self: Any,
    source_cache: Any,
    shifted_plan: dict[str, Any],
    *,
    prefix_len: int,
    runtime_profile: dict[str, float | int] | None,
  ) -> Any | None:
    if source_cache is None or not isinstance(shifted_plan, dict) or not shifted_plan.get("valid"):
      return None
    source_pairs = _cache_layer_pairs(source_cache)
    if not source_pairs:
      return None
    try:
      from transformers.cache_utils import DynamicCache

      buffered = _dynamic_cache_from_buffered_prefix(
        model_self,
        source_pairs,
        int(prefix_len),
        pool_attr="_openpilot_shifted_kv_reconstruct_pool",
        runtime_profile=runtime_profile,
        metric_prefix="shifted_prompt_kv_reconstruct",
      )
      if buffered is None:
        prompt_cache = None
        rebuilt_pairs: list[tuple[Any, Any]] = []
        if runtime_profile is not None:
          runtime_profile["shifted_prompt_kv_reconstruct_buffered"] = 0
        for source_key, source_value in source_pairs:
          if int(source_key.shape[-2]) < int(prefix_len) or int(source_value.shape[-2]) < int(prefix_len):
            return None
          rebuilt_pairs.append((
            source_key[..., :prefix_len, :].detach().clone().contiguous(),
            source_value[..., :prefix_len, :].detach().clone().contiguous(),
          ))
      else:
        prompt_cache, rebuilt_pairs = buffered
      ranges = [item for item in shifted_plan.get("ranges", []) if isinstance(item, dict)]
      copied_tokens = 0
      for (source_key, source_value), (target_key, target_value) in zip(source_pairs, rebuilt_pairs, strict=True):
        for item in ranges:
          src_start = int(item.get("source_language_token_start", -1))
          cur_start = int(item.get("current_language_token_start", -1))
          token_count = int(item.get("token_count", 0) or 0)
          if src_start < 0 or cur_start < 0 or token_count <= 0:
            continue
          src_end = src_start + token_count
          cur_end = cur_start + token_count
          if src_end > int(source_key.shape[-2]) or cur_end > int(prefix_len):
            continue
          target_key[..., cur_start:cur_end, :].copy_(source_key[..., src_start:src_end, :])
          target_value[..., cur_start:cur_end, :].copy_(source_value[..., src_start:src_end, :])
          copied_tokens += token_count
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_reconstruct_layers"] = len(rebuilt_pairs)
        runtime_profile["shifted_prompt_kv_reconstruct_copied_tokens"] = int(copied_tokens)
        runtime_profile["shifted_prompt_kv_reconstruct_prefix_len"] = int(prefix_len)
      if prompt_cache is None:
        prompt_cache = DynamicCache(ddp_cache_data=rebuilt_pairs)
      return prompt_cache
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_reconstruct_error"] = f"{type(exc).__name__}: {exc}"
      return None

  def _shifted_prompt_recompute_start(
    input_ids: Any,
    source_sequences: Any,
    spans: list[Any],
    shifted_plan: dict[str, Any],
    *,
    visual_prefix_end: int,
    input_seq_len: int,
  ) -> tuple[int, str]:
    first_text_mismatch = int(input_seq_len)
    try:
      if source_sequences is not None and hasattr(source_sequences, "shape") and int(source_sequences.shape[1]) > 0:
        compare_len = min(int(input_seq_len), int(source_sequences.shape[1]))
        source_ids = source_sequences[:, :compare_len].to(device=input_ids.device, dtype=input_ids.dtype)
        mismatches = (input_ids[:, :compare_len] != source_ids).any(dim=0).nonzero(as_tuple=False).flatten()
        if int(mismatches.numel()) > 0:
          first_text_mismatch = int(mismatches[0].item())
    except Exception:
      first_text_mismatch = int(visual_prefix_end)

    matched_current_ranges: set[tuple[int, int]] = set()
    for item in shifted_plan.get("ranges", []):
      if not isinstance(item, dict):
        continue
      cur_start = int(item.get("current_language_token_start", -1))
      token_count = int(item.get("token_count", 0) or 0)
      if cur_start >= 0 and token_count > 0:
        matched_current_ranges.add((cur_start, cur_start + token_count))

    first_unmatched_visual = int(input_seq_len)
    for span in spans:
      if not isinstance(span, dict):
        continue
      start = int(span.get("language_token_start", -1))
      end = int(span.get("language_token_end", -1))
      if start < 0 or end <= start:
        continue
      if (start, end) not in matched_current_ranges:
        first_unmatched_visual = min(first_unmatched_visual, start)

    recompute_start = min(int(visual_prefix_end), int(first_text_mismatch), int(first_unmatched_visual))
    reason = "visual_prefix_end"
    if recompute_start == first_text_mismatch and first_text_mismatch < visual_prefix_end:
      reason = "source_input_ids_mismatch"
    if recompute_start == first_unmatched_visual and first_unmatched_visual < visual_prefix_end:
      reason = "first_unmatched_visual_span"

    for span in spans:
      if not isinstance(span, dict):
        continue
      start = int(span.get("language_token_start", -1))
      end = int(span.get("language_token_end", -1))
      if start >= 0 and start < recompute_start < end:
        recompute_start = start
        reason = f"{reason}:aligned_to_visual_span"
        break
    return int(recompute_start), reason

  def _select_image_suffix_kwargs(
    suffix_kwargs: dict[str, Any],
    spans: list[Any],
    *,
    recompute_start: int,
    runtime_profile: dict[str, float | int] | None,
  ) -> bool:
    for key in (
      "pixel_values_videos",
      "video_grid_thw",
      "_openpilot_precomputed_video_features",
      "_openpilot_precomputed_patch_pos_embeds",
      "cache_position_ids",
      "cache_rope_deltas",
    ):
      suffix_kwargs.pop(key, None)

    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, span in enumerate(spans):
      if not isinstance(span, dict):
        continue
      start = int(span.get("language_token_start", -1))
      end = int(span.get("language_token_end", -1))
      if start < 0 or end <= start:
        continue
      if start < int(recompute_start) < end:
        return False
      if end > int(recompute_start):
        selected.append((idx, span))

    if not selected:
      for key in (
        "pixel_values",
        "image_grid_thw",
        "_openpilot_precomputed_image_features",
      ):
        suffix_kwargs.pop(key, None)
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_current_suffix_image_blocks"] = 0
      return True

    image_grid = suffix_kwargs.get("image_grid_thw")
    pixel_values = suffix_kwargs.get("pixel_values")
    if image_grid is None or pixel_values is None or not hasattr(image_grid, "shape") or not hasattr(pixel_values, "shape"):
      return False

    try:
      selected_indices = [int(idx) for idx, _span in selected]
      if not selected_indices or max(selected_indices) >= int(image_grid.shape[0]):
        return False
      grid_cpu = image_grid.detach().cpu().tolist()
      patch_counts: list[int] = []
      for row in grid_cpu:
        patch_count = 1
        for dim in row:
          patch_count *= int(dim)
        patch_counts.append(int(patch_count))
      pixel_ranges: list[tuple[int, int]] = []
      offset = 0
      for patch_count in patch_counts:
        pixel_ranges.append((offset, offset + int(patch_count)))
        offset += int(patch_count)
      if offset > int(pixel_values.shape[0]):
        return False
      index_tensor = torch.tensor(selected_indices, device=image_grid.device, dtype=torch.long)
      suffix_kwargs["image_grid_thw"] = image_grid.index_select(0, index_tensor).contiguous()
      suffix_kwargs["pixel_values"] = torch.cat(
        [pixel_values[start:end] for start, end in (pixel_ranges[idx] for idx in selected_indices)],
        dim=0,
      ).contiguous()
      suffix_kwargs.pop("_openpilot_precomputed_image_features", None)
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_current_suffix_image_blocks"] = len(selected_indices)
        runtime_profile["shifted_prompt_kv_current_suffix_image_patch_tokens"] = int(suffix_kwargs["pixel_values"].shape[0])
      return True
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_current_suffix_image_subset_error"] = f"{type(exc).__name__}: {exc}"
      return False

  def _clone_cache_prefix(
    model_self: Any,
    cache: Any,
    prefix_len: int,
    *,
    runtime_profile: dict[str, float | int] | None,
  ) -> Any | None:
    pairs = _cache_layer_pairs(cache)
    if not pairs:
      return None
    try:
      from transformers.cache_utils import DynamicCache

      buffered = _dynamic_cache_from_buffered_prefix(
        model_self,
        pairs,
        int(prefix_len),
        pool_attr="_openpilot_shifted_kv_prefix_clone_pool",
        runtime_profile=runtime_profile,
        metric_prefix="shifted_prompt_kv_clone_prefix",
      )
      if buffered is not None:
        return buffered[0]
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_clone_prefix_buffered"] = 0
      rebuilt_pairs: list[tuple[Any, Any]] = []
      for key_states, value_states in pairs:
        if int(key_states.shape[-2]) < int(prefix_len) or int(value_states.shape[-2]) < int(prefix_len):
          return None
        rebuilt_pairs.append((
          key_states[..., :prefix_len, :].detach().clone().contiguous(),
          value_states[..., :prefix_len, :].detach().clone().contiguous(),
        ))
      return DynamicCache(ddp_cache_data=rebuilt_pairs)
    except Exception:
      return None

  def _copy_cache_range(target_cache: Any, source_cache: Any, start: int, end: int) -> bool:
    target_pairs = _cache_layer_pairs(target_cache)
    source_pairs = _cache_layer_pairs(source_cache)
    if not target_pairs or not source_pairs or len(target_pairs) != len(source_pairs):
      return False
    try:
      for target, source in zip(target_pairs, source_pairs, strict=True):
        target_key, target_value = target
        source_key, source_value = source
        if int(target_key.shape[-2]) < int(end) or int(source_key.shape[-2]) < int(end):
          return False
        target_key[..., start:end, :].copy_(source_key[..., start:end, :])
        target_value[..., start:end, :].copy_(source_value[..., start:end, :])
      return True
    except Exception:
      return False

  def _stored_prompt_cache_for_shifted_path(prompt_cache: Any, runtime_profile: dict[str, float | int] | None) -> Any:
    try:
      store_deepcopy = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_STORE_PROMPT_CACHE_DEEPCOPY", "1")))
    except Exception:
      store_deepcopy = True
    if store_deepcopy:
      if runtime_profile is not None:
        runtime_profile["vlm_full_generation_prompt_cache_store_deepcopy"] = 1
      return copy.deepcopy(prompt_cache)
    if runtime_profile is not None:
      runtime_profile["vlm_full_generation_prompt_cache_store_deepcopy"] = 0
      runtime_profile["vlm_full_generation_prompt_cache_store_borrowed"] = 1
    return prompt_cache

  def _unmatched_visual_spans(spans: list[Any], shifted_plan: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    matched_current_ranges: set[tuple[int, int]] = set()
    for item in shifted_plan.get("ranges", []):
      if not isinstance(item, dict):
        continue
      cur_start = int(item.get("current_language_token_start", -1))
      token_count = int(item.get("token_count", 0) or 0)
      if cur_start >= 0 and token_count > 0:
        matched_current_ranges.add((cur_start, cur_start + token_count))
    unmatched: list[tuple[int, dict[str, Any]]] = []
    for idx, span in enumerate(spans):
      if not isinstance(span, dict):
        continue
      start = int(span.get("language_token_start", -1))
      end = int(span.get("language_token_end", -1))
      if start < 0 or end <= start:
        continue
      if (start, end) not in matched_current_ranges:
        unmatched.append((idx, span))
    unmatched.sort(key=lambda item: int(item[1].get("language_token_start", 0)))
    return unmatched

  def _select_exact_image_blocks_kwargs(
    span_kwargs: dict[str, Any],
    spans: list[Any],
    selected_indices: list[int],
    *,
    runtime_profile: dict[str, float | int] | None,
  ) -> bool:
    for key in (
      "pixel_values_videos",
      "video_grid_thw",
      "_openpilot_precomputed_image_features",
      "_openpilot_precomputed_video_features",
      "_openpilot_precomputed_patch_pos_embeds",
      "cache_position_ids",
      "cache_rope_deltas",
    ):
      span_kwargs.pop(key, None)
    image_grid = span_kwargs.get("image_grid_thw")
    pixel_values = span_kwargs.get("pixel_values")
    if image_grid is None or pixel_values is None or not hasattr(image_grid, "shape") or not hasattr(pixel_values, "shape"):
      return False
    try:
      if not selected_indices or max(selected_indices) >= int(image_grid.shape[0]):
        return False
      grid_cpu = image_grid.detach().cpu().tolist()
      pixel_ranges: list[tuple[int, int]] = []
      offset = 0
      for row in grid_cpu:
        patch_count = 1
        for dim in row:
          patch_count *= int(dim)
        pixel_ranges.append((offset, offset + int(patch_count)))
        offset += int(patch_count)
      if offset > int(pixel_values.shape[0]):
        return False
      index_tensor = torch.tensor(selected_indices, device=image_grid.device, dtype=torch.long)
      span_kwargs["image_grid_thw"] = image_grid.index_select(0, index_tensor).contiguous()
      span_kwargs["pixel_values"] = torch.cat(
        [pixel_values[start:end] for start, end in (pixel_ranges[idx] for idx in selected_indices)],
        dim=0,
      ).contiguous()
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_image_blocks"] = int(runtime_profile.get("shifted_prompt_kv_visual_fill_image_blocks", 0)) + len(selected_indices)
        runtime_profile["shifted_prompt_kv_visual_fill_image_patch_tokens"] = int(runtime_profile.get("shifted_prompt_kv_visual_fill_image_patch_tokens", 0)) + int(span_kwargs["pixel_values"].shape[0])
      return True
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_error"] = f"{type(exc).__name__}: {exc}"
      return False

  def _visual_embeds_for_spans(
    model_self: Any,
    model_kwargs: dict[str, Any],
    spans: list[Any],
    selected_indices: list[int],
    *,
    runtime_profile: dict[str, float | int] | None,
  ) -> dict[int, Any] | None:
    image_grid = model_kwargs.get("image_grid_thw")
    pixel_values = model_kwargs.get("pixel_values")
    precomputed_image_features = model_kwargs.get("_openpilot_precomputed_image_features")
    if image_grid is None or not hasattr(image_grid, "shape"):
      return None
    try:
      if not selected_indices or max(selected_indices) >= int(image_grid.shape[0]):
        return None
      if precomputed_image_features is not None:
        feature_items = None
        if isinstance(precomputed_image_features, (list, tuple)) and len(precomputed_image_features) >= 1:
          feature_items = precomputed_image_features[0]
        if isinstance(feature_items, (list, tuple)) and max(selected_indices) < len(feature_items):
          split: dict[int, Any] = {}
          total_tokens = 0
          for idx in selected_indices:
            span = spans[idx]
            token_count = int(span.get("token_count", 0) or 0) if isinstance(span, dict) else 0
            if token_count <= 0:
              return None
            embeds = feature_items[idx]
            if embeds is None or not hasattr(embeds, "shape") or int(embeds.shape[0]) != token_count:
              return None
            split[idx] = embeds
            total_tokens += token_count
          if runtime_profile is not None:
            runtime_profile["shifted_prompt_kv_visual_fill_precomputed_hit"] = 1
            runtime_profile["shifted_prompt_kv_visual_fill_precomputed_blocks"] = len(selected_indices)
            runtime_profile["shifted_prompt_kv_visual_fill_precomputed_tokens"] = int(total_tokens)
            runtime_profile["shifted_prompt_kv_visual_fill_visual_seconds"] = 0.0
          return split
      if pixel_values is None or not hasattr(pixel_values, "shape"):
        return None
      grid_cpu = image_grid.detach().cpu().tolist()
      pixel_ranges: list[tuple[int, int]] = []
      offset = 0
      for row in grid_cpu:
        patch_count = 1
        for dim in row:
          patch_count *= int(dim)
        pixel_ranges.append((offset, offset + int(patch_count)))
        offset += int(patch_count)
      if offset > int(pixel_values.shape[0]):
        return None
      index_tensor = torch.tensor(selected_indices, device=image_grid.device, dtype=torch.long)
      selected_grid = image_grid.index_select(0, index_tensor).contiguous()
      selected_pixels = torch.cat(
        [pixel_values[start:end] for start, end in (pixel_ranges[idx] for idx in selected_indices)],
        dim=0,
      ).contiguous()
      visual_start = time.perf_counter()
      visual = getattr(model_self.vlm, "visual", None)
      if visual is None:
        return None
      image_embeds = visual(selected_pixels, grid_thw=selected_grid)
      if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]
      if image_embeds is None or not hasattr(image_embeds, "shape"):
        return None
      split: dict[int, Any] = {}
      embed_offset = 0
      for idx in selected_indices:
        span = spans[idx]
        token_count = int(span.get("token_count", 0) or 0) if isinstance(span, dict) else 0
        if token_count <= 0:
          return None
        split[idx] = image_embeds[embed_offset : embed_offset + token_count]
        embed_offset += token_count
      if embed_offset != int(image_embeds.shape[0]):
        return None
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_image_blocks"] = len(selected_indices)
        runtime_profile["shifted_prompt_kv_visual_fill_image_patch_tokens"] = int(selected_pixels.shape[0])
        runtime_profile["shifted_prompt_kv_visual_fill_visual_seconds"] = time.perf_counter() - visual_start
      return split
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_error"] = f"{type(exc).__name__}: {exc}"
      return None

  def _fill_unmatched_visual_prompt_cache(
    model_self: Any,
    input_ids: Any,
    model_kwargs: dict[str, Any],
    prompt_cache: Any,
    spans: list[Any],
    unmatched_spans: list[tuple[int, dict[str, Any]]],
    *,
    cache_position_seed: Any,
    prefill_position_ids: Any,
    runtime_profile: dict[str, float | int] | None,
  ) -> bool:
    def _cache_only_forward(span_kwargs: dict[str, Any], span_inputs_embeds: Any) -> Any | None:
      try:
        backbone_only = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_VISUAL_FILL_BACKBONE_ONLY", "1")))
      except Exception:
        backbone_only = True
      if backbone_only:
        backbone = getattr(model_self.vlm, "model", None)
        if callable(backbone):
          try:
            outputs = backbone(
              input_ids=None,
              inputs_embeds=span_inputs_embeds,
              return_dict=True,
              **span_kwargs,
            )
            if getattr(outputs, "past_key_values", None) is not None:
              if runtime_profile is not None:
                runtime_profile["shifted_prompt_kv_visual_fill_backbone_only"] = 1
              return outputs
          except Exception as exc:
            if runtime_profile is not None:
              runtime_profile["shifted_prompt_kv_visual_fill_backbone_error"] = f"{type(exc).__name__}: {exc}"
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_backbone_only"] = 0
      return model_self.vlm(
        input_ids=None,
        inputs_embeds=span_inputs_embeds,
        logits_to_keep=1,
        return_dict=True,
        **span_kwargs,
      )

    if not unmatched_spans:
      return True
    start_time = time.perf_counter()
    filled_tokens = 0
    selected_indices = [int(span_idx) for span_idx, _span in unmatched_spans]
    visual_embeds_by_span = _visual_embeds_for_spans(
      model_self,
      model_kwargs,
      spans,
      selected_indices,
      runtime_profile=runtime_profile,
    )
    if visual_embeds_by_span is None:
      return False
    image_token_id = getattr(getattr(model_self.vlm, "config", None), "image_token_id", None)
    embed_tokens = getattr(model_self.vlm, "get_input_embeddings", lambda: None)()
    if image_token_id is None or embed_tokens is None:
      return False
    try:
      range_start = min(int(span.get("language_token_start", -1)) for _span_idx, span in unmatched_spans)
      range_end = max(int(span.get("language_token_end", -1)) for _span_idx, span in unmatched_spans)
      unmatched_token_total = sum(
        max(0, int(span.get("language_token_end", 0)) - int(span.get("language_token_start", 0)))
        for _span_idx, span in unmatched_spans
      )
      if range_start >= 0 and range_end > range_start and (range_end - range_start) <= max(32, 2 * int(unmatched_token_total)):
        range_span_indices: list[int] = []
        for idx, span in enumerate(spans):
          if not isinstance(span, dict):
            continue
          span_start = int(span.get("language_token_start", -1))
          span_end = int(span.get("language_token_end", -1))
          if span_start >= range_start and span_end <= range_end and span_end > span_start:
            range_span_indices.append(int(idx))
        range_visual_embeds = _visual_embeds_for_spans(
          model_self,
          model_kwargs,
          spans,
          range_span_indices,
          runtime_profile=runtime_profile,
        )
        range_prefix_cache = _clone_cache_prefix(
          model_self,
          prompt_cache,
          range_start,
          runtime_profile=runtime_profile,
        )
        if range_visual_embeds is not None and range_prefix_cache is not None:
          range_ids = input_ids[:, range_start:range_end]
          range_inputs_embeds = embed_tokens(range_ids).clone()
          replaced_tokens = 0
          for span_idx in range_span_indices:
            span = spans[span_idx]
            span_start = int(span.get("language_token_start", -1))
            span_end = int(span.get("language_token_end", -1))
            span_image_embeds = range_visual_embeds.get(int(span_idx))
            if span_image_embeds is None:
              raise ValueError("missing coalesced span image embeddings")
            local_start = span_start - range_start
            local_end = span_end - range_start
            local_ids = range_ids[:, local_start:local_end]
            image_mask = local_ids == int(image_token_id)
            if int(image_mask.sum().item()) != int(span_image_embeds.shape[0]):
              raise ValueError("coalesced span image token count mismatch")
            range_inputs_embeds[:, local_start:local_end][image_mask] = span_image_embeds.to(
              device=range_inputs_embeds.device,
              dtype=range_inputs_embeds.dtype,
            )
            replaced_tokens += int(span_image_embeds.shape[0])
          range_kwargs = dict(model_kwargs)
          for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "_openpilot_precomputed_image_features",
            "_openpilot_precomputed_video_features",
            "_openpilot_precomputed_patch_pos_embeds",
            "cache_position_ids",
            "cache_rope_deltas",
          ):
            range_kwargs.pop(key, None)
          range_kwargs["past_key_values"] = range_prefix_cache
          range_kwargs["use_cache"] = True
          range_kwargs["cache_position"] = cache_position_seed[range_start:range_end]
          if prefill_position_ids is not None:
            range_kwargs["position_ids"] = prefill_position_ids[..., range_start:range_end]
          range_outputs = _cache_only_forward(range_kwargs, range_inputs_embeds)
          if range_outputs is not None and _copy_cache_range(prompt_cache, range_outputs.past_key_values, range_start, range_end):
            if runtime_profile is not None:
              runtime_profile["shifted_prompt_kv_visual_fill_coalesced"] = 1
              runtime_profile["shifted_prompt_kv_visual_fill_coalesced_span_tokens"] = int(range_end - range_start)
              runtime_profile["shifted_prompt_kv_visual_fill_coalesced_visual_tokens"] = int(replaced_tokens)
              runtime_profile["shifted_prompt_kv_visual_fill_blocks"] = len(unmatched_spans)
              runtime_profile["shifted_prompt_kv_visual_fill_tokens"] = int(sum(
                int(span.get("language_token_end", 0)) - int(span.get("language_token_start", 0))
                for _span_idx, span in unmatched_spans
              ))
              runtime_profile["shifted_prompt_kv_visual_fill_seconds"] = time.perf_counter() - start_time
            return True
      elif runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_coalesced_skipped"] = 1
        runtime_profile["shifted_prompt_kv_visual_fill_coalesced_span_tokens"] = int(range_end - range_start)
        runtime_profile["shifted_prompt_kv_visual_fill_coalesced_needed_tokens"] = int(unmatched_token_total)
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_visual_fill_coalesced_error"] = f"{type(exc).__name__}: {exc}"
    for span_idx, span in unmatched_spans:
      span_start = int(span.get("language_token_start", -1))
      span_end = int(span.get("language_token_end", -1))
      if span_start < 0 or span_end <= span_start:
        return False
      span_prefix_cache = _clone_cache_prefix(
        model_self,
        prompt_cache,
        span_start,
        runtime_profile=runtime_profile,
      )
      if span_prefix_cache is None:
        return False
      span_ids = input_ids[:, span_start:span_end]
      span_image_embeds = visual_embeds_by_span.get(int(span_idx))
      if span_image_embeds is None:
        return False
      span_inputs_embeds = embed_tokens(span_ids).clone()
      image_mask = span_ids == int(image_token_id)
      if int(image_mask.sum().item()) != int(span_image_embeds.shape[0]):
        return False
      span_inputs_embeds[image_mask] = span_image_embeds.to(
        device=span_inputs_embeds.device,
        dtype=span_inputs_embeds.dtype,
      )
      span_kwargs = dict(model_kwargs)
      for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "_openpilot_precomputed_image_features",
        "_openpilot_precomputed_video_features",
        "_openpilot_precomputed_patch_pos_embeds",
        "cache_position_ids",
        "cache_rope_deltas",
      ):
        span_kwargs.pop(key, None)
      span_kwargs["past_key_values"] = span_prefix_cache
      span_kwargs["use_cache"] = True
      span_kwargs["cache_position"] = cache_position_seed[span_start:span_end]
      if prefill_position_ids is not None:
        span_kwargs["position_ids"] = prefill_position_ids[..., span_start:span_end]
      span_outputs = _cache_only_forward(span_kwargs, span_inputs_embeds)
      if span_outputs is None:
        return False
      if not _copy_cache_range(prompt_cache, span_outputs.past_key_values, span_start, span_end):
        return False
      filled_tokens += int(span_end - span_start)
    if runtime_profile is not None:
      runtime_profile["shifted_prompt_kv_visual_fill_blocks"] = len(unmatched_spans)
      runtime_profile["shifted_prompt_kv_visual_fill_tokens"] = int(filled_tokens)
      runtime_profile["shifted_prompt_kv_visual_fill_seconds"] = time.perf_counter() - start_time
    return True

  def _try_shifted_visual_prefix_current_suffix(
    model_self: Any,
    input_ids: Any,
    model_kwargs: dict[str, Any],
    prefix_cache_entry: dict[str, Any],
    *,
    max_generation_length: int,
    eos_token_id: int,
    cache_position_seed: Any,
    prefill_position_ids: Any,
    runtime_profile: dict[str, float | int] | None,
  ) -> tuple[Any, Any] | None:
    if int(max_generation_length) != 0 or not isinstance(prefix_cache_entry, dict):
      return None
    shifted_plan = prefix_cache_entry.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan")
    source_cache = prefix_cache_entry.get("streaming_vlm_shift_source_prompt_cache")
    spans = prefix_cache_entry.get("language_visual_token_spans")
    if source_cache is None or not isinstance(shifted_plan, dict) or not shifted_plan.get("valid") or not isinstance(spans, list):
      return None
    try:
      input_seq_len = int(input_ids.shape[1])
      visual_prefix_end = max(
        int(span.get("language_token_end", -1))
        for span in spans
        if isinstance(span, dict)
      )
    except Exception:
      return None
    if visual_prefix_end <= 0 or visual_prefix_end >= input_seq_len:
      return None
    unmatched_spans = _unmatched_visual_spans(spans, shifted_plan)
    manual_prefill_start = time.perf_counter()
    try:
      tail_prefill_threshold = int(os.environ.get("ALPAMAYO_SHIFTED_KV_TAIL_PREFILL_VISUAL_TOKEN_THRESHOLD", "192"))
    except Exception:
      tail_prefill_threshold = 192
    current_visual_tokens = int(shifted_plan.get("current_visual_language_tokens", 0) or 0)
    use_tail_prefill = bool(
      unmatched_spans
      and tail_prefill_threshold > 0
      and current_visual_tokens > 0
      and current_visual_tokens <= tail_prefill_threshold
    )
    try:
      reuse_text_suffix_draft = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_REUSE_TEXT_SUFFIX_DRAFT", "0")))
    except Exception:
      reuse_text_suffix_draft = False
    try:
      skip_new_visual_fill_draft = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_SKIP_NEW_VISUAL_FILL_DRAFT", "0")))
    except Exception:
      skip_new_visual_fill_draft = False
    if reuse_text_suffix_draft and unmatched_spans:
      replay_start = time.perf_counter()
      replay_prompt_cache = _build_shifted_visual_prefix_cache(
        model_self,
        source_cache,
        shifted_plan,
        prefix_len=input_seq_len,
        runtime_profile=runtime_profile,
      )
      replay_hit = False
      if replay_prompt_cache is not None:
        if skip_new_visual_fill_draft:
          replay_hit = True
          if runtime_profile is not None:
            runtime_profile["shifted_prompt_kv_visual_fill_skipped_draft"] = 1
            runtime_profile["shifted_prompt_kv_visual_fill_skipped_blocks"] = len(unmatched_spans)
            runtime_profile["shifted_prompt_kv_visual_fill_skipped_tokens"] = int(sum(
              max(0, int(span.get("language_token_end", 0) or 0) - int(span.get("language_token_start", 0) or 0))
              for _span_idx, span in unmatched_spans
            ))
            runtime_profile["shifted_prompt_kv_visual_fill_seconds"] = 0.0
        else:
          replay_hit = _fill_unmatched_visual_prompt_cache(
            model_self,
            input_ids,
            model_kwargs,
            replay_prompt_cache,
            spans,
            unmatched_spans,
            cache_position_seed=cache_position_seed,
            prefill_position_ids=prefill_position_ids,
            runtime_profile=runtime_profile,
          )
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_text_suffix_replay_requested"] = 1
        runtime_profile["shifted_prompt_kv_text_suffix_replay_hit"] = int(bool(replay_hit))
        runtime_profile["shifted_prompt_kv_text_suffix_replay_seconds"] = time.perf_counter() - replay_start
      if replay_hit:
        prompt_cache = replay_prompt_cache
        generated_sequences = input_ids
        prefix_cache_entry["streaming_vlm_reuse_mode"] = (
          "shifted_kv_visual_patch_text_suffix_skipfill"
          if skip_new_visual_fill_draft
          else "shifted_kv_visual_patch_text_suffix_replay"
        )
        prefix_cache_entry["streaming_vlm_reuse_unverified"] = True
        prefix_cache_entry["full_vlm_generated_sequences"] = generated_sequences.detach().clone()
        prefix_cache_entry["full_vlm_prompt_cache"] = _stored_prompt_cache_for_shifted_path(prompt_cache, runtime_profile)
        prefix_cache_entry["full_vlm_prompt_cache_owner"] = (
          "shifted_visual_patch_text_suffix_skipfill"
          if skip_new_visual_fill_draft
          else "shifted_visual_patch_text_suffix_replay"
        )
        prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
          "current_window_signature",
          prefix_cache_entry.get("window_signature"),
        )
        prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = False
        prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
        prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
        prefix_cache_entry["full_vlm_input_seq_len"] = input_seq_len
        prefix_cache_entry["full_vlm_reason"] = (
          "shifted_kv_visual_patch_text_suffix_skipfill_draft"
          if skip_new_visual_fill_draft
          else "shifted_kv_visual_patch_text_suffix_replay_draft"
        )
        prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
        if runtime_profile is not None:
          runtime_profile["shifted_prompt_kv_current_suffix_hit"] = 1
          runtime_profile["shifted_prompt_kv_current_suffix_seconds"] = 0.0
          runtime_profile["shifted_prompt_kv_current_suffix_prefix_len"] = int(input_seq_len)
          runtime_profile["shifted_prompt_kv_current_suffix_visual_prefix_end"] = int(visual_prefix_end)
          runtime_profile["shifted_prompt_kv_current_suffix_recompute_reason"] = (
            "visual_patch_text_suffix_skipfill_draft"
            if skip_new_visual_fill_draft
            else "visual_patch_text_suffix_replay_draft"
          )
          runtime_profile["shifted_prompt_kv_current_suffix_tokens"] = 0
          runtime_profile["manual_vlm_prefill_seconds"] = time.perf_counter() - manual_prefill_start
          runtime_profile["manual_vlm_decode_seconds"] = 0.0
          runtime_profile["manual_vlm_decode_forwards"] = 0
          runtime_profile["manual_vlm_generated_tokens"] = 0
          runtime_profile["manual_vlm_generated_sequences_owner"] = (
            "shifted_kv_visual_patch_text_suffix_skipfill_input_ids"
            if skip_new_visual_fill_draft
            else "shifted_kv_visual_patch_text_suffix_replay_input_ids"
          )
          runtime_profile["vlm_full_generation_cache_store"] = 1
        return generated_sequences, prompt_cache
    tail_recompute_start = visual_prefix_end
    if use_tail_prefill:
      try:
        tail_recompute_start = min(int(span.get("language_token_start", visual_prefix_end)) for _idx, span in unmatched_spans)
      except Exception:
        return None
      if tail_recompute_start <= 0 or tail_recompute_start >= visual_prefix_end:
        return None
      prompt_cache = _build_shifted_visual_prefix_cache(
        model_self,
        source_cache,
        shifted_plan,
        prefix_len=tail_recompute_start,
        runtime_profile=runtime_profile,
      )
      if prompt_cache is None:
        return None
      tail_ids = input_ids[:, tail_recompute_start:visual_prefix_end]
      tail_kwargs = dict(model_kwargs)
      if not _select_image_suffix_kwargs(
        tail_kwargs,
        spans,
        recompute_start=tail_recompute_start,
        runtime_profile=runtime_profile,
      ):
        return None
      tail_kwargs["past_key_values"] = prompt_cache
      tail_kwargs["use_cache"] = True
      tail_kwargs["cache_position"] = cache_position_seed[tail_recompute_start:visual_prefix_end]
      if prefill_position_ids is not None:
        tail_kwargs["position_ids"] = prefill_position_ids[..., tail_recompute_start:visual_prefix_end]
      tail_start = time.perf_counter()
      tail_outputs = model_self.vlm(
        input_ids=tail_ids,
        logits_to_keep=1,
        return_dict=True,
        **tail_kwargs,
      )
      prompt_cache = tail_outputs.past_key_values
      if runtime_profile is not None:
        runtime_profile["shifted_prompt_kv_tail_prefill_hit"] = 1
        runtime_profile["shifted_prompt_kv_tail_prefill_threshold"] = int(tail_prefill_threshold)
        runtime_profile["shifted_prompt_kv_tail_prefill_current_visual_tokens"] = int(current_visual_tokens)
        runtime_profile["shifted_prompt_kv_tail_prefill_start"] = int(tail_recompute_start)
        runtime_profile["shifted_prompt_kv_tail_prefill_tokens"] = int(visual_prefix_end - tail_recompute_start)
        runtime_profile["shifted_prompt_kv_tail_prefill_seconds"] = time.perf_counter() - tail_start
    else:
      prompt_cache = _build_shifted_visual_prefix_cache(
        model_self,
        source_cache,
        shifted_plan,
        prefix_len=visual_prefix_end,
        runtime_profile=runtime_profile,
      )
      if prompt_cache is None:
        return None
      if not _fill_unmatched_visual_prompt_cache(
        model_self,
        input_ids,
        model_kwargs,
        prompt_cache,
        spans,
        unmatched_spans,
        cache_position_seed=cache_position_seed,
        prefill_position_ids=prefill_position_ids,
        runtime_profile=runtime_profile,
      ):
        return None
    suffix_ids = input_ids[:, visual_prefix_end:]
    suffix_kwargs = dict(model_kwargs)
    if not _select_image_suffix_kwargs(
      suffix_kwargs,
      spans,
      recompute_start=visual_prefix_end,
      runtime_profile=runtime_profile,
    ):
      return None
    suffix_kwargs["past_key_values"] = prompt_cache
    suffix_kwargs["use_cache"] = True
    suffix_kwargs["cache_position"] = cache_position_seed[visual_prefix_end:input_seq_len]
    if prefill_position_ids is not None:
      suffix_kwargs["position_ids"] = prefill_position_ids[..., visual_prefix_end:input_seq_len]
    start = time.perf_counter()
    suffix_outputs = model_self.vlm(
      input_ids=suffix_ids,
      logits_to_keep=1,
      return_dict=True,
      **suffix_kwargs,
    )
    prompt_cache = suffix_outputs.past_key_values
    prefix_cache_entry["streaming_vlm_reuse_mode"] = "shifted_kv_current_state_suffix"
    prefix_cache_entry["streaming_vlm_reuse_unverified"] = True
    prefix_cache_entry["full_vlm_generated_sequences"] = input_ids.detach().clone()
    prefix_cache_entry["full_vlm_prompt_cache"] = _stored_prompt_cache_for_shifted_path(prompt_cache, runtime_profile)
    prefix_cache_entry["full_vlm_prompt_cache_owner"] = "shifted_visual_prefix_current_state_suffix"
    prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
      "current_window_signature",
      prefix_cache_entry.get("window_signature"),
    )
    prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = False
    prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
    prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
    prefix_cache_entry["full_vlm_input_seq_len"] = input_seq_len
    prefix_cache_entry["full_vlm_reason"] = (
      "shifted_kv_current_state_suffix_ready:contiguous_tail_prefill"
      if use_tail_prefill
      else "shifted_kv_current_state_suffix_ready:sparse_visual_fill"
    )
    prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
    if runtime_profile is not None:
      runtime_profile["shifted_prompt_kv_current_suffix_hit"] = 1
      runtime_profile["shifted_prompt_kv_current_suffix_seconds"] = time.perf_counter() - start
      runtime_profile["shifted_prompt_kv_current_suffix_prefix_len"] = int(visual_prefix_end)
      runtime_profile["shifted_prompt_kv_current_suffix_visual_prefix_end"] = int(visual_prefix_end)
      runtime_profile["shifted_prompt_kv_current_suffix_recompute_reason"] = (
        "contiguous_tail_prefill" if use_tail_prefill else "sparse_visual_fill"
      )
      runtime_profile["shifted_prompt_kv_current_suffix_tokens"] = int(input_seq_len - visual_prefix_end)
      runtime_profile["manual_vlm_prefill_seconds"] = time.perf_counter() - manual_prefill_start
      runtime_profile["manual_vlm_decode_seconds"] = 0.0
      runtime_profile["manual_vlm_decode_forwards"] = 0
      runtime_profile["manual_vlm_generated_tokens"] = 0
      runtime_profile["manual_vlm_generated_sequences_owner"] = "shifted_kv_current_state_suffix_input_ids"
      runtime_profile["vlm_full_generation_cache_store"] = 1
    return input_ids, prompt_cache

  def manual_greedy_vlm_generate(
    self: Any,
    input_ids: Any,
    tokenized_data: dict[str, Any],
    eos_token_id: int,
    max_generation_length: int,
    runtime_profile: dict[str, float | int] | None = None,
  ) -> tuple[Any, Any]:
    input_seq_len = int(input_ids.shape[1])
    max_sequence_len = input_seq_len + int(max_generation_length) + 1
    sequence_buffer = getattr(self, "_openpilot_manual_generated_sequences_buffer", None)
    if (
      sequence_buffer is None
      or int(sequence_buffer.shape[0]) < int(input_ids.shape[0])
      or int(sequence_buffer.shape[1]) < max_sequence_len
      or sequence_buffer.device != input_ids.device
      or sequence_buffer.dtype != input_ids.dtype
    ):
      sequence_buffer = torch.empty(
        (int(input_ids.shape[0]), max_sequence_len),
        device=input_ids.device,
        dtype=input_ids.dtype,
      )
      setattr(self, "_openpilot_manual_generated_sequences_buffer", sequence_buffer)
      if runtime_profile is not None:
        runtime_profile["manual_vlm_generated_sequence_buffer_reused"] = 0
    elif runtime_profile is not None:
      runtime_profile["manual_vlm_generated_sequence_buffer_reused"] = 1
    sequence_buffer[: input_ids.shape[0], : input_ids.shape[1]].copy_(input_ids)
    generated_sequence_len = input_seq_len
    generated_sequences = sequence_buffer[: input_ids.shape[0], :generated_sequence_len]
    model_kwargs = dict(tokenized_data)
    model_kwargs["use_cache"] = True
    cache_position_seed = model_kwargs.pop("cache_position_seed", None)
    if cache_position_seed is None:
      cache_position_seed = torch.arange(input_ids.shape[1], device=input_ids.device)
    model_kwargs["cache_position"] = cache_position_seed[: input_ids.shape[1]]

    prefill_position_ids = model_kwargs.pop("cache_position_ids", None)
    prefill_rope_deltas = model_kwargs.pop("cache_rope_deltas", None)
    if prefill_rope_deltas is not None and hasattr(self, "vlm") and hasattr(self.vlm, "model"):
      setattr(self.vlm.model, "rope_deltas", prefill_rope_deltas)
    logits_processor = ExpertLogitsProcessor(
      traj_token_offset=self.config.traj_token_start_idx,
      traj_vocab_size=self.config.traj_vocab_size,
    )
    eos_seen = False
    prompt_cache = None
    prefill_seconds = 0.0
    decode_seconds = 0.0
    decode_forwards = 0
    first_step = True
    prefix_cache_entry = getattr(self, "_openpilot_vlm_prefix_cache_entry", None)
    prefix_cache_warm_hit = bool(
      isinstance(prefix_cache_entry, dict)
      and int(prefix_cache_entry.get("hits", 0) or 0) > 0
    )
    full_generation_ready = bool(
      isinstance(prefix_cache_entry, dict)
      and prefix_cache_entry.get("full_vlm_generated_sequences") is not None
      and prefix_cache_entry.get("full_vlm_prompt_cache") is not None
    )
    trusted_fast_current_action_replay = bool(
      isinstance(prefix_cache_entry, dict)
      and getattr(self, "_openpilot_require_fast_no_prefill", False)
      and prefix_cache_entry.get("streaming_vlm_fast_current_action_replay_allowed")
    )
    full_generation_usable = bool(
      full_generation_ready
      and int(prefix_cache_entry.get("full_vlm_max_generation_length", -1)) == int(max_generation_length)
      and int(prefix_cache_entry.get("full_vlm_eos_token_id", -1)) == int(eos_token_id)
      and (
        int(prefix_cache_entry.get("full_vlm_input_seq_len", -1)) == int(input_ids.shape[1])
        or trusted_fast_current_action_replay
      )
    )
    if full_generation_usable and isinstance(prefix_cache_entry, dict):
      exact_window_full_hit = bool(
        prefix_cache_entry.get("current_window_full_hit")
        or prefix_cache_entry.get("exact_window_full_hit")
        or prefix_cache_entry.get("window_full_hit")
      )
      streaming_reuse_mode = str(prefix_cache_entry.get("streaming_vlm_reuse_mode", ""))
      streaming_reuse_unverified = bool(prefix_cache_entry.get("streaming_vlm_reuse_unverified")) or streaming_reuse_mode.endswith("_unverified")
      trusted_replay_requested = bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_requested"))
      trusted_no_reasoning_replay = bool(
        prefix_cache_entry.get("streaming_vlm_trusted_replay_allowed")
        and int(max_generation_length) == 0
      )
      (
        exact_current_window_generation,
        generation_window_signature_match,
        prompt_cache_context_exact,
      ) = _full_generation_exact_current_window(prefix_cache_entry)
      if runtime_profile is not None:
        runtime_profile["vlm_full_generation_cache_window_signature_match"] = (
          1 if generation_window_signature_match else 0
        )
        runtime_profile["vlm_full_generation_prompt_cache_context_exact"] = (
          1 if prompt_cache_context_exact else 0
        )
      if (
        not trusted_no_reasoning_replay
        and not trusted_fast_current_action_replay
        and (streaming_reuse_unverified or not exact_window_full_hit or not exact_current_window_generation)
      ):
        full_generation_usable = False
        if streaming_reuse_unverified:
          prefix_cache_entry["full_vlm_reason"] = "disabled_for_unverified_streaming_current_prompt_freshness"
        elif not exact_window_full_hit:
          prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_window_hit"
        elif not generation_window_signature_match:
          prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_generation_window_signature"
        else:
          prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_prompt_cache_context"
        if runtime_profile is not None:
          runtime_profile["vlm_full_generation_cache_disabled_for_streaming"] = 1 if streaming_reuse_unverified else 0
          runtime_profile["vlm_full_generation_cache_disabled_without_exact_window_hit"] = 0 if exact_window_full_hit else 1
          runtime_profile["vlm_full_generation_cache_disabled_without_exact_window_signature"] = (
            0 if generation_window_signature_match else 1
          )
          runtime_profile["vlm_full_generation_cache_disabled_without_exact_prompt_cache_context"] = (
            0 if prompt_cache_context_exact else 1
          )
          runtime_profile["vlm_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = (
            1 if trusted_replay_requested or bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_allowed")) else 0
          )
      elif trusted_no_reasoning_replay and runtime_profile is not None:
        runtime_profile["vlm_full_generation_cache_trusted_replay_no_reasoning"] = 1
        runtime_profile["vlm_full_generation_cache_disabled_for_streaming"] = 0
        runtime_profile["vlm_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = 0
      elif trusted_fast_current_action_replay and runtime_profile is not None:
        runtime_profile["vlm_full_generation_cache_trusted_replay_fast_current_action"] = 1
        runtime_profile["vlm_full_generation_cache_disabled_for_streaming"] = 0
        runtime_profile["vlm_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = 0
    if (
      isinstance(prefix_cache_entry, dict)
      and full_generation_usable
    ):
      try:
        cached_sequences = prefix_cache_entry["full_vlm_generated_sequences"].to(input_ids.device)
        cached_prompt_cache = prefix_cache_entry["full_vlm_prompt_cache"]
        pooled_copy = getattr(self, "_openpilot_pooled_prompt_cache_copy", None)
        cache_mode = "prefix_cache_borrowed"
        if callable(pooled_copy) and cached_prompt_cache is not None:
          cached_prompt_cache, cache_mode = pooled_copy(
            prefix_cache_entry,
            cached_prompt_cache,
            pool_key="full_vlm_prompt_cache_live_pool",
            pool_size=4,
          )
        prefix_cache_entry["full_vlm_hits"] = int(prefix_cache_entry.get("full_vlm_hits", 0)) + 1
        if runtime_profile is not None:
          runtime_profile["vlm_full_generation_cache_hit"] = 1
          runtime_profile["manual_vlm_prefill_seconds"] = 0.0
          runtime_profile["manual_vlm_decode_seconds"] = 0.0
          runtime_profile["manual_vlm_decode_forwards"] = 0
          runtime_profile["manual_vlm_generated_tokens"] = int(cached_sequences.shape[1] - input_ids.shape[1])
          runtime_profile["vlm_full_generation_live_cache_mode"] = cache_mode
          runtime_profile["vlm_full_generation_sequences_owner"] = "prefix_cache_borrowed_immutable"
          runtime_profile["vlm_full_generation_prompt_cache_owner"] = "full_generation_borrowed_live_cache"
        return cached_sequences, cached_prompt_cache
      except Exception as exc:
        prefix_cache_entry["full_vlm_reason"] = f"full_generation_cache_copy_failed:{type(exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["vlm_full_generation_cache_copy_error"] = f"{type(exc).__name__}: {exc}"
    if isinstance(prefix_cache_entry, dict) and int(max_generation_length) == 0:
      shifted_suffix_result = _try_shifted_visual_prefix_current_suffix(
        self,
        input_ids,
        model_kwargs,
        prefix_cache_entry,
        max_generation_length=int(max_generation_length),
        eos_token_id=int(eos_token_id),
        cache_position_seed=cache_position_seed,
        prefill_position_ids=prefill_position_ids,
        runtime_profile=runtime_profile,
      )
      if shifted_suffix_result is not None:
        return shifted_suffix_result
    if isinstance(prefix_cache_entry, dict) and prefix_cache_entry.get("streaming_vlm_draft_generated_sequences") is not None:
      draft_start = time.perf_counter()
      try:
        draft_sequences = prefix_cache_entry["streaming_vlm_draft_generated_sequences"].to(
          device=input_ids.device,
          dtype=input_ids.dtype,
        )
        draft_input_len = int(prefix_cache_entry.get("streaming_vlm_draft_input_seq_len", -1))
        draft_max_generation = int(prefix_cache_entry.get("streaming_vlm_draft_max_generation_length", -1))
        draft_eos_token = int(prefix_cache_entry.get("streaming_vlm_draft_eos_token_id", -1))
        draft_new_tokens = int(draft_sequences.shape[1]) - int(input_ids.shape[1])
        draft_new_tokens = min(draft_new_tokens, int(max_generation_length))
        if (
          int(draft_sequences.shape[0]) != int(input_ids.shape[0])
          or draft_input_len != int(input_ids.shape[1])
          or draft_max_generation != int(max_generation_length)
          or draft_eos_token != int(eos_token_id)
          or draft_new_tokens <= 0
        ):
          raise ValueError("streaming draft shape/settings mismatch")
        verify_sequence_len = int(input_ids.shape[1]) + draft_new_tokens
        sequence_buffer[: input_ids.shape[0], : input_ids.shape[1]].copy_(input_ids)
        sequence_buffer[
          : input_ids.shape[0],
          input_ids.shape[1] : verify_sequence_len,
        ].copy_(draft_sequences[:, input_ids.shape[1] : verify_sequence_len])
        verify_inputs = sequence_buffer[: input_ids.shape[0], :verify_sequence_len]
        verify_kwargs = dict(model_kwargs)
        verify_kwargs["cache_position"] = cache_position_seed[:verify_sequence_len]
        verify_attention_mask = verify_kwargs.get("attention_mask")
        if verify_attention_mask is not None and hasattr(verify_attention_mask, "shape"):
          if int(verify_attention_mask.shape[-1]) == int(input_ids.shape[1]):
            verify_kwargs["attention_mask"] = torch.cat(
              [
                verify_attention_mask,
                torch.ones(
                  (*verify_attention_mask.shape[:-1], draft_new_tokens),
                  device=verify_attention_mask.device,
                  dtype=verify_attention_mask.dtype,
                ),
              ],
              dim=-1,
            )
        verify_outputs = None
        if (
          bool(getattr(self, "_openpilot_graph_draft_verify_prefill_stage_enabled", False))
          and hasattr(self, "_openpilot_vlm_prefill_with_cuda_graph")
        ):
          verify_outputs = self._openpilot_vlm_prefill_with_cuda_graph(
            input_ids=verify_inputs,
            tokenized_data={**verify_kwargs, "logits_to_keep": draft_new_tokens + 1},
            runtime_profile=runtime_profile,
            cache_position_seed=cache_position_seed,
          )
        if verify_outputs is None:
          verify_outputs = self.vlm(
            input_ids=verify_inputs,
            logits_to_keep=draft_new_tokens + 1,
            return_dict=True,
            **verify_kwargs,
          )
        verify_logits = verify_outputs.logits
        if int(verify_logits.shape[1]) < draft_new_tokens + 1:
          raise ValueError("streaming draft verifier returned too few logits")
        logits_offset = int(verify_logits.shape[1]) - (draft_new_tokens + 1)
        accepted_tokens = 0
        for draft_idx in range(draft_new_tokens):
          scores = verify_logits[:, logits_offset + draft_idx, :]
          scores = logits_processor(verify_inputs[:, : input_ids.shape[1] + draft_idx], scores)
          predicted = scores.argmax(dim=-1, keepdim=True).to(input_ids.device)
          expected = verify_inputs[
            :,
            input_ids.shape[1] + draft_idx : input_ids.shape[1] + draft_idx + 1,
          ]
          if not bool(torch.equal(predicted, expected)):
            break
          accepted_tokens += 1
        if accepted_tokens != draft_new_tokens:
          prefix_cache_entry["streaming_vlm_draft_reason"] = (
            f"verify_rejected_at:{accepted_tokens}/{draft_new_tokens}"
          )
          if runtime_profile is not None:
            runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
            runtime_profile["streaming_vlm_draft_verify_accepted_tokens"] = int(accepted_tokens)
            runtime_profile["streaming_vlm_draft_verify_tokens"] = int(draft_new_tokens)
            runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
        else:
          prompt_cache = verify_outputs.past_key_values
          generated_sequences = verify_inputs
          generated_sequence_len = verify_sequence_len
          prefix_cache_entry["streaming_vlm_draft_reason"] = "verify_accepted"
          prefix_cache_entry["streaming_vlm_reuse_unverified"] = False
          prefix_cache_entry["streaming_vlm_reuse_mode"] = "draft_verify"
          prefix_cache_entry["full_vlm_generated_sequences"] = generated_sequences.detach().clone()
          prefix_cache_entry["full_vlm_prompt_cache"] = copy.deepcopy(prompt_cache)
          prefix_cache_entry["full_vlm_prompt_cache_owner"] = "streaming_draft_verified_current_prompt_cache"
          prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
            "current_window_signature",
            prefix_cache_entry.get("window_signature"),
          )
          prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = True
          prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
          prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
          prefix_cache_entry["full_vlm_input_seq_len"] = int(input_ids.shape[1])
          prefix_cache_entry["full_vlm_reason"] = "streaming_draft_verified_full_generation_ready"
          prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
          if runtime_profile is not None:
            runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
            runtime_profile["streaming_vlm_draft_verify_hit"] = 1
            runtime_profile["streaming_vlm_draft_verify_accepted_tokens"] = int(accepted_tokens)
            runtime_profile["streaming_vlm_draft_verify_tokens"] = int(draft_new_tokens)
            runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
            runtime_profile["manual_vlm_prefill_seconds"] = time.perf_counter() - draft_start
            runtime_profile["manual_vlm_decode_seconds"] = 0.0
            runtime_profile["manual_vlm_decode_forwards"] = 0
            runtime_profile["manual_vlm_generated_tokens"] = int(draft_new_tokens)
            runtime_profile["manual_vlm_generated_sequences_owner"] = "streaming_draft_verified_buffer"
            runtime_profile["vlm_full_generation_cache_store"] = 1
          return generated_sequences, prompt_cache
      except Exception as exc:
        prefix_cache_entry["streaming_vlm_draft_reason"] = f"verify_error:{type(exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
          runtime_profile["streaming_vlm_draft_verify_error"] = f"{type(exc).__name__}: {exc}"
          runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
    if (
      prefix_cache_warm_hit
      and not full_generation_usable
      and bool(getattr(self, "_openpilot_static_graph_strict_shapes", False))
    ):
      if runtime_profile is not None:
        runtime_profile["vlm_strict_warm_full_generation_cache_miss"] = 1
      raise RuntimeError("strict warm VLM replay required but full-generation cache is not ready")
    cached_prefill_output = None
    if isinstance(prefix_cache_entry, dict) and prefix_cache_entry.get("prefill_output") is not None:
      try:
        source_prefill_output = prefix_cache_entry["prefill_output"]
        source_prompt_cache = getattr(source_prefill_output, "past_key_values", None)
        pooled_copy = getattr(self, "_openpilot_pooled_prompt_cache_copy", None)
        prefill_live_cache_mode = "deepcopy"
        try:
          if callable(pooled_copy) and source_prompt_cache is not None:
            live_prompt_cache, prefill_live_cache_mode = pooled_copy(
              prefix_cache_entry,
              source_prompt_cache,
              pool_key="prefill_live_prompt_cache_pool",
              pool_size=4,
            )
            cached_prefill_output = copy.copy(source_prefill_output)
            cached_prefill_output.past_key_values = live_prompt_cache
          else:
            cached_prefill_output = copy.deepcopy(source_prefill_output)
        except Exception as pool_exc:
          cached_prefill_output = copy.deepcopy(source_prefill_output)
          prefill_live_cache_mode = f"deepcopy_after_pool_error:{type(pool_exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["vlm_prefix_cache_hit"] = 1
          runtime_profile["vlm_prefix_cache_prefill_reused"] = 1
          runtime_profile["vlm_prefix_cache_prefill_live_cache_mode"] = prefill_live_cache_mode
      except Exception as exc:
        cached_prefill_output = None
        if runtime_profile is not None:
          runtime_profile["vlm_prefix_cache_prefill_copy_error"] = f"{type(exc).__name__}: {exc}"
    elif runtime_profile is not None and isinstance(prefix_cache_entry, dict):
      runtime_profile["vlm_prefix_cache_prefill_reused"] = 0

    for step_idx in range(max_generation_length + 1):
      if first_step:
        generated_inputs = generated_sequences
        model_call_kwargs = dict(model_kwargs)
        if prefill_position_ids is not None:
          model_call_kwargs["position_ids"] = prefill_position_ids
        first_step = False
      else:
        generated_inputs = generated_sequences[:, -1:]
        model_call_kwargs = dict(model_kwargs)
        model_call_kwargs.pop("cache_position", None)
        for key in (
          "pixel_values",
          "pixel_values_videos",
          "image_grid_thw",
          "video_grid_thw",
        ):
          model_call_kwargs.pop(key, None)

      step_start = time.perf_counter()
      if step_idx == 0 and cached_prefill_output is not None:
        vlm_outputs = cached_prefill_output
      elif (
        step_idx == 0
        and bool(getattr(self, "_openpilot_graph_standard_prefill_stage_enabled", False))
        and hasattr(self, "_openpilot_vlm_prefill_with_cuda_graph")
      ):
        vlm_outputs = self._openpilot_vlm_prefill_with_cuda_graph(
          input_ids=generated_inputs,
          tokenized_data={**model_call_kwargs, "logits_to_keep": 1},
          runtime_profile=runtime_profile,
          cache_position_seed=cache_position_seed,
        )
        prefill_graph_requested = getattr(self, "_openpilot_graph_prefill_stage_requested", None)
        if (
          vlm_outputs is None
          and bool(getattr(self, "_openpilot_static_graph_strict_shapes", False))
          and callable(prefill_graph_requested)
          and bool(prefill_graph_requested())
        ):
          if runtime_profile is not None:
            runtime_profile["graph_prefill_stage_strict_fallback_blocked"] = 1
          raise RuntimeError("VLM prefill graph unavailable under strict static graph mode")
      else:
        vlm_outputs = None
      if step_idx > 0 and vlm_outputs is None and hasattr(self, "_openpilot_vlm_decode_step_with_cuda_graph"):
        decode_graph_requested = getattr(self, "_openpilot_graph_decode_stage_requested", None)
        remaining_generation_tokens = int(max_generation_length) - int(generated_sequence_len - input_seq_len)
        if remaining_generation_tokens > 0 and hasattr(self, "_openpilot_vlm_decode_block_with_cuda_graph"):
          block_start = time.perf_counter()
          block_result = self._openpilot_vlm_decode_block_with_cuda_graph(
            input_ids=generated_inputs,
            model_call_kwargs=model_call_kwargs,
            sequence_buffer=sequence_buffer[: input_ids.shape[0], :max_sequence_len],
            sequence_len=generated_sequence_len,
            max_new_tokens=remaining_generation_tokens,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
            runtime_profile=runtime_profile,
          )
          if block_result is not None:
            decode_seconds += time.perf_counter() - block_start
            decode_forwards += int(block_result.get("decode_forwards", 0))
            generated_sequence_len = int(block_result["sequence_len"])
            generated_sequences = sequence_buffer[: input_ids.shape[0], :generated_sequence_len]
            prompt_cache = block_result["prompt_cache"]
            eos_seen = bool(block_result.get("eos_seen", False))
            if runtime_profile is not None:
              runtime_profile["manual_vlm_decode_block_used"] = 1
            break
          if (
            bool(getattr(self, "_openpilot_static_graph_strict_shapes", False))
            and callable(decode_graph_requested)
            and bool(decode_graph_requested())
          ):
            if runtime_profile is not None:
              runtime_profile["graph_decode_block_stage_strict_fallback_blocked"] = 1
            raise RuntimeError("VLM decode block graph unavailable under strict static graph mode")
        vlm_outputs = self._openpilot_vlm_decode_step_with_cuda_graph(
          input_ids=generated_inputs,
          model_call_kwargs=model_call_kwargs,
          runtime_profile=runtime_profile,
        )
        if (
          vlm_outputs is None
          and bool(getattr(self, "_openpilot_static_graph_strict_shapes", False))
          and callable(decode_graph_requested)
          and bool(decode_graph_requested())
        ):
          if runtime_profile is not None:
            runtime_profile["graph_decode_stage_strict_fallback_blocked"] = 1
          raise RuntimeError("VLM decode graph unavailable under strict static graph mode")
      if vlm_outputs is None:
        vlm_outputs = self.vlm(
          input_ids=generated_inputs,
          logits_to_keep=1,
          return_dict=True,
          **model_call_kwargs,
        )
      step_seconds = time.perf_counter() - step_start
      if step_idx == 0:
        prefill_seconds += step_seconds
      else:
        decode_seconds += step_seconds
        decode_forwards += 1

      if step_idx == 0 and cached_prefill_output is None and isinstance(prefix_cache_entry, dict):
        try:
          prefix_cache_entry["prefill_output"] = copy.deepcopy(vlm_outputs)
          prefix_cache_entry["has_prompt_cache"] = True
          prefix_cache_entry["reason"] = "prompt_cache_ready"
          prefix_cache_entry["stores"] = int(prefix_cache_entry.get("stores", 0)) + 1
          if runtime_profile is not None:
            runtime_profile["vlm_prefix_cache_store"] = 1
        except Exception as exc:
          prefix_cache_entry["has_prompt_cache"] = False
          prefix_cache_entry["reason"] = f"prefill_output_store_failed:{type(exc).__name__}"
          if runtime_profile is not None:
            runtime_profile["vlm_prefix_cache_store_error"] = f"{type(exc).__name__}: {exc}"

      prompt_cache = vlm_outputs.past_key_values
      model_kwargs = self.vlm._update_model_kwargs_for_generation(
        vlm_outputs,
        model_kwargs,
        is_encoder_decoder=False,
      )
      for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "position_ids",
      ):
        model_kwargs.pop(key, None)

      if step_idx >= max_generation_length:
        break

      scores = vlm_outputs.logits[:, -1, :]
      scores = logits_processor(generated_sequences, scores)
      next_token = scores.argmax(dim=-1, keepdim=True).to(generated_sequences.device)
      sequence_buffer[: input_ids.shape[0], generated_sequence_len : generated_sequence_len + 1].copy_(next_token)
      generated_sequence_len += 1
      generated_sequences = sequence_buffer[: input_ids.shape[0], :generated_sequence_len]
      eos_seen = bool(torch.all(next_token == eos_token_id).detach().cpu().item())
      if eos_seen:
        break

    if runtime_profile is not None:
      runtime_profile["manual_vlm_prefill_seconds"] = prefill_seconds
      runtime_profile["manual_vlm_decode_seconds"] = decode_seconds
      runtime_profile["manual_vlm_decode_forwards"] = decode_forwards
      runtime_profile["manual_vlm_generated_tokens"] = int(generated_sequence_len - input_seq_len)
      runtime_profile["manual_vlm_generated_sequences_owner"] = "model_static_buffer_borrowed"

    if prompt_cache is None:
      raise RuntimeError("manual VLM generation did not produce a cache")
    if isinstance(prefix_cache_entry, dict):
      try:
        prefix_cache_entry["full_vlm_generated_sequences"] = generated_sequences.detach().clone()
        prefix_cache_entry["full_vlm_prompt_cache"] = copy.deepcopy(prompt_cache)
        prefix_cache_entry["full_vlm_prompt_cache_owner"] = "prefix_cache_full_generation_stored_immutable_copy"
        prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
          "current_window_signature",
          prefix_cache_entry.get("window_signature"),
        )
        prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = True
        prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
        prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
        prefix_cache_entry["full_vlm_input_seq_len"] = int(input_ids.shape[1])
        prefix_cache_entry["full_vlm_reason"] = "full_generation_ready"
        prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
        if runtime_profile is not None:
          runtime_profile["vlm_full_generation_cache_store"] = 1
      except Exception as exc:
        prefix_cache_entry["full_vlm_reason"] = f"full_generation_cache_store_failed:{type(exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["vlm_full_generation_cache_store_error"] = f"{type(exc).__name__}: {exc}"
    return generated_sequences, prompt_cache

  alpamayo_mod.Alpamayo1_5._manual_greedy_vlm_generate = manual_greedy_vlm_generate
  alpamayo_mod.Alpamayo1_5._openpilot_manual_greedy_patch = True


def _patch_qwen3vl_streaming_vision_cache() -> None:
  try:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl
  except Exception:
    return

  attention_cls = getattr(qwen3vl, "Qwen3VLVisionAttention", None)
  if attention_cls is None or getattr(attention_cls, "_openpilot_streaming_vision_cache_patch", False):
    return

  original_forward = attention_cls.forward

  def record_stats(runtime_profile: Any, stats: dict[str, Any]) -> None:
    if not isinstance(runtime_profile, dict):
      return
    for key, value in stats.items():
      profile_key = f"streaming_vision_{key}"
      if isinstance(value, (int, float, np.integer, np.floating)):
        runtime_profile[profile_key] = runtime_profile.get(profile_key, 0) + value
      else:
        runtime_profile[profile_key] = str(value)

  def layer_key_for(attention_module: Any) -> str:
    layer_key = getattr(attention_module, "_openpilot_streaming_vision_layer_key", None)
    if layer_key is None:
      layer_key = f"{attention_module.__class__.__name__}:{id(attention_module)}"
      setattr(attention_module, "_openpilot_streaming_vision_layer_key", layer_key)
    return str(layer_key)

  def qkv_with_cached_kv(
    attention_module: Any,
    hidden_states: Any,
    cache: StreamingAlpamayoVisionCache,
    layer_key: str,
    token_blocks: list[dict[str, Any]] | None,
    runtime_profile: Any,
  ) -> tuple[Any, Any, Any, dict[str, Any], list[dict[str, Any]] | None] | None:
    stats: dict[str, Any] = {
      "qkv_split_projection_attempted": 1,
      "qkv_split_projection_used": 0,
      "qkv_split_projection_cached_blocks": 0,
      "qkv_split_projection_cached_tokens": 0,
      "qkv_split_projection_miss_tokens": 0,
      "qkv_split_projection_reused_query_buffer": 0,
      "qkv_split_projection_reused_kv_buffers": 0,
      "qkv_split_projection_reused_miss_buffers": 0,
      "qkv_split_projection_errors": 0,
      "qkv_split_projection_fallbacks": 0,
    }

    def fallback(reason: str) -> None:
      stats["qkv_split_projection_fallbacks"] = 1
      stats["qkv_split_projection_fallback_reason"] = reason
      record_stats(runtime_profile, stats)
      return None

    try:
      if qwen3vl.torch.is_grad_enabled():
        return fallback("grad_enabled")
      qkv_module = attention_module.qkv
      weight = getattr(qkv_module, "weight", None)
      bias = getattr(qkv_module, "bias", None)
      if weight is None or len(weight.shape) != 2:
        return fallback("unsupported_qkv_weight")
      seq_length = int(hidden_states.shape[0])
      num_heads = int(getattr(attention_module, "num_heads", 0) or 0)
      if num_heads <= 0:
        return fallback("missing_num_heads")
      qkv_out = int(weight.shape[0])
      if qkv_out % 3 != 0:
        return fallback("unsupported_qkv_rows")
      per_proj_out = qkv_out // 3
      if per_proj_out % num_heads != 0:
        return fallback("unsupported_head_dim")
      head_dim = per_proj_out // num_heads
      hidden_dim = int(hidden_states.shape[-1])
      query_buffer = getattr(attention_module, "_openpilot_streaming_query_buffer", None)
      if (
        query_buffer is None
        or int(query_buffer.shape[0]) < seq_length
        or int(query_buffer.shape[1]) != per_proj_out
        or query_buffer.device != hidden_states.device
        or query_buffer.dtype != hidden_states.dtype
      ):
        query_buffer = qwen3vl.torch.empty(
          (seq_length, per_proj_out),
          device=hidden_states.device,
          dtype=hidden_states.dtype,
        )
        setattr(attention_module, "_openpilot_streaming_query_buffer", query_buffer)
      else:
        stats["qkv_split_projection_reused_query_buffer"] = 1
      query_flat = query_buffer[:seq_length]
      qwen3vl.torch.mm(
        hidden_states,
        weight[:per_proj_out].transpose(0, 1),
        out=query_flat,
      )
      if bias is not None:
        query_flat.add_(bias[:per_proj_out])
      query_states = query_flat.reshape(seq_length, num_heads, head_dim)
      kv_shape = (seq_length, num_heads, head_dim)
      key_buffer = getattr(attention_module, "_openpilot_streaming_key_states_buffer", None)
      value_buffer = getattr(attention_module, "_openpilot_streaming_value_states_buffer", None)
      if (
        key_buffer is None
        or value_buffer is None
        or int(key_buffer.shape[0]) < seq_length
        or tuple(key_buffer.shape[1:]) != tuple(kv_shape[1:])
        or key_buffer.device != hidden_states.device
        or key_buffer.dtype != query_states.dtype
        or int(value_buffer.shape[0]) < seq_length
        or tuple(value_buffer.shape[1:]) != tuple(kv_shape[1:])
        or value_buffer.device != hidden_states.device
        or value_buffer.dtype != query_states.dtype
      ):
        key_buffer = qwen3vl.torch.empty(
          kv_shape,
          device=hidden_states.device,
          dtype=query_states.dtype,
        )
        value_buffer = qwen3vl.torch.empty_like(key_buffer)
        setattr(attention_module, "_openpilot_streaming_key_states_buffer", key_buffer)
        setattr(attention_module, "_openpilot_streaming_value_states_buffer", value_buffer)
      else:
        stats["qkv_split_projection_reused_kv_buffers"] = 1
      key_states = key_buffer[:seq_length]
      value_states = value_buffer[:seq_length]
      covered = [False] * seq_length
      miss_ranges: list[tuple[int, int]] = []
      miss_blocks: list[dict[str, Any]] = []
      for block in list(token_blocks or []):
        token_start = int(block.get("token_start", -1))
        token_count = int(block.get("token_count", 0))
        token_end = token_start + token_count
        if token_count <= 0 or token_start < 0 or token_end > seq_length:
          continue
        for idx in range(token_start, token_end):
          covered[idx] = True
        cached = cache.cached_pre_rope_kv(
          str(block.get("stream", "")),
          tuple(block.get("signature", ())),
          layer_key,
        )
        if cached is None:
          miss_ranges.append((token_start, token_end))
          miss_blocks.append(block)
          continue
        cached_key, cached_value = cached
        if tuple(cached_key.shape) != tuple(key_states[token_start:token_end].shape) or tuple(cached_value.shape) != tuple(value_states[token_start:token_end].shape):
          miss_ranges.append((token_start, token_end))
          miss_blocks.append(block)
          continue
        key_states[token_start:token_end] = cached_key.to(
          device=key_states.device,
          dtype=key_states.dtype,
          non_blocking=True,
        )
        value_states[token_start:token_end] = cached_value.to(
          device=value_states.device,
          dtype=value_states.dtype,
          non_blocking=True,
        )
        stats["qkv_split_projection_cached_blocks"] += 1
        stats["qkv_split_projection_cached_tokens"] += token_count
      gap_start = None
      for idx, is_covered in enumerate(covered):
        if not is_covered and gap_start is None:
          gap_start = idx
        elif is_covered and gap_start is not None:
          miss_ranges.append((gap_start, idx))
          gap_start = None
      if gap_start is not None:
        miss_ranges.append((gap_start, seq_length))
      if miss_ranges:
        miss_token_count = sum(end - start for start, end in miss_ranges)
        miss_hidden_buffer = getattr(attention_module, "_openpilot_streaming_miss_hidden_buffer", None)
        miss_key_buffer = getattr(attention_module, "_openpilot_streaming_miss_key_buffer", None)
        miss_value_buffer = getattr(attention_module, "_openpilot_streaming_miss_value_buffer", None)
        if (
          miss_hidden_buffer is None
          or int(miss_hidden_buffer.shape[0]) < miss_token_count
          or int(miss_hidden_buffer.shape[1]) != hidden_dim
          or miss_hidden_buffer.device != hidden_states.device
          or miss_hidden_buffer.dtype != hidden_states.dtype
          or miss_key_buffer is None
          or int(miss_key_buffer.shape[0]) < miss_token_count
          or int(miss_key_buffer.shape[1]) != per_proj_out
          or miss_key_buffer.device != hidden_states.device
          or miss_key_buffer.dtype != query_states.dtype
          or miss_value_buffer is None
          or int(miss_value_buffer.shape[0]) < miss_token_count
          or int(miss_value_buffer.shape[1]) != per_proj_out
          or miss_value_buffer.device != hidden_states.device
          or miss_value_buffer.dtype != query_states.dtype
        ):
          miss_hidden_buffer = qwen3vl.torch.empty(
            (miss_token_count, hidden_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
          )
          miss_key_buffer = qwen3vl.torch.empty(
            (miss_token_count, per_proj_out),
            device=hidden_states.device,
            dtype=query_states.dtype,
          )
          miss_value_buffer = qwen3vl.torch.empty_like(miss_key_buffer)
          setattr(attention_module, "_openpilot_streaming_miss_hidden_buffer", miss_hidden_buffer)
          setattr(attention_module, "_openpilot_streaming_miss_key_buffer", miss_key_buffer)
          setattr(attention_module, "_openpilot_streaming_miss_value_buffer", miss_value_buffer)
        else:
          stats["qkv_split_projection_reused_miss_buffers"] = 1
        miss_hidden = miss_hidden_buffer[:miss_token_count]
        miss_key_flat = miss_key_buffer[:miss_token_count]
        miss_value_flat = miss_value_buffer[:miss_token_count]
        offset = 0
        for start, end in miss_ranges:
          length = end - start
          miss_hidden[offset : offset + length].copy_(hidden_states[start:end])
          offset += length
        qwen3vl.torch.mm(
          miss_hidden,
          weight[per_proj_out : 2 * per_proj_out].transpose(0, 1),
          out=miss_key_flat,
        )
        qwen3vl.torch.mm(
          miss_hidden,
          weight[2 * per_proj_out :].transpose(0, 1),
          out=miss_value_flat,
        )
        if bias is not None:
          miss_key_flat.add_(bias[per_proj_out : 2 * per_proj_out])
          miss_value_flat.add_(bias[2 * per_proj_out :])
        miss_key = miss_key_flat.reshape(-1, num_heads, head_dim)
        miss_value = miss_value_flat.reshape(-1, num_heads, head_dim)
        offset = 0
        for start, end in miss_ranges:
          length = end - start
          key_states[start:end] = miss_key[offset : offset + length]
          value_states[start:end] = miss_value[offset : offset + length]
          stats["qkv_split_projection_miss_tokens"] += length
          offset += length
      stats["qkv_split_projection_used"] = 1
      stats["qkv_split_projection_miss_blocks"] = len(miss_blocks)
      return query_states, key_states, value_states, stats, miss_blocks
    except Exception as exc:
      stats["qkv_split_projection_errors"] = 1
      stats["qkv_split_projection_error"] = f"{type(exc).__name__}: {exc}"
      record_stats(runtime_profile, stats)
      return None

  def streaming_forward(
    attention_module: Any,
    hidden_states: Any,
    cu_seqlens: Any,
    position_embeddings: Any,
    cache: StreamingAlpamayoVisionCache,
    runtime_profile: Any,
    kwargs: dict[str, Any],
    token_blocks: list[dict[str, Any]] | None,
  ) -> Any:
    seq_length = hidden_states.shape[0]
    layer_key = layer_key_for(attention_module)
    split_qkv = qkv_with_cached_kv(attention_module, hidden_states, cache, layer_key, token_blocks, runtime_profile)
    capture_token_blocks = token_blocks
    if split_qkv is None:
      qkv = attention_module.qkv(hidden_states)
      query_states, key_states, value_states = (
        qkv.reshape(seq_length, 3, attention_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
      )
      fresh_key_states = key_states
      fresh_value_states = value_states
      key_states, value_states, reuse_stats = cache.apply_cached_pre_rope_kv(
        key_states,
        value_states,
        layer_key,
        token_blocks=token_blocks,
      )
      record_stats(runtime_profile, reuse_stats)
    else:
      query_states, key_states, value_states, split_stats, capture_token_blocks = split_qkv
      fresh_key_states = key_states
      fresh_value_states = value_states
      record_stats(runtime_profile, split_stats)
      record_stats(runtime_profile, {"streaming_attention_fastpath_calls": 1})
    capture_stats = cache.capture_pre_rope_states(
      fresh_key_states,
      fresh_value_states,
      layer_key=layer_key,
      token_blocks=capture_token_blocks,
    )
    record_stats(runtime_profile, capture_stats)

    cos, sin = position_embeddings
    query_states, key_states = qwen3vl.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    streaming_attention_mask, mask_stats = cache.materialize_attention_mask(
      qwen3vl.torch,
      device=query_states.device,
      dtype=query_states.dtype,
      total_tokens=int(seq_length),
    )
    record_stats(runtime_profile, mask_stats)

    attention_interface = qwen3vl.eager_attention_forward
    if attention_module.config._attn_implementation != "eager":
      attention_interface = qwen3vl.ALL_ATTENTION_FUNCTIONS[attention_module.config._attn_implementation]

    if streaming_attention_mask is not None:
      masked_backend = "eager"
      masked_attention_interface = qwen3vl.eager_attention_forward
      try:
        masked_attention_interface = qwen3vl.ALL_ATTENTION_FUNCTIONS["sdpa"]
        masked_backend = "sdpa"
      except Exception:
        pass
      attn_output, _ = masked_attention_interface(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask=streaming_attention_mask,
        scaling=attention_module.scaling,
        dropout=0.0 if not attention_module.training else attention_module.attention_dropout,
        is_causal=False,
        **kwargs,
      )
      record_stats(runtime_profile, {"attention_mask_backend": masked_backend})
    elif attention_module.config._attn_implementation == "flash_attention_2":
      max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
      attn_output, _ = attention_interface(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=attention_module.scaling,
        dropout=0.0 if not attention_module.training else attention_module.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
        **kwargs,
      )
    else:
      lengths = cu_seqlens[1:] - cu_seqlens[:-1]
      splits = [
        qwen3vl.torch.split(tensor, lengths.tolist(), dim=2)
        for tensor in (query_states, key_states, value_states)
      ]
      attn_outputs = [
        attention_interface(
          attention_module,
          q,
          k,
          v,
          attention_mask=None,
          scaling=attention_module.scaling,
          dropout=0.0 if not attention_module.training else attention_module.attention_dropout,
          is_causal=False,
          **kwargs,
        )[0]
        for q, k, v in zip(*splits)
      ]
      attn_output = qwen3vl.torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = attention_module.proj(attn_output)
    record_stats(runtime_profile, {"streaming_attention_calls": 1})
    return attn_output

  def patched_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    context = getattr(_STREAMING_VISION_PATCH_CONTEXT, "context", None)
    if context is not None:
      cache = context.get("cache")
      runtime_profile = context.get("runtime_profile")
      token_blocks = context.get("token_blocks")
      hidden_states = args[0] if args else kwargs.get("hidden_states")
      cu_seqlens = args[1] if len(args) > 1 else kwargs.get("cu_seqlens")
      position_embeddings = kwargs.get("position_embeddings")
      if len(args) > 3:
        position_embeddings = args[3]
      if cache is not None and hidden_states is not None and cu_seqlens is not None and position_embeddings is not None and hasattr(self, "qkv"):
        backend_kwargs = dict(kwargs)
        for consumed_key in ("hidden_states", "cu_seqlens", "rotary_pos_emb", "position_embeddings"):
          backend_kwargs.pop(consumed_key, None)
        try:
          return streaming_forward(
            self,
            hidden_states,
            cu_seqlens,
            position_embeddings,
            cache,
            runtime_profile,
            backend_kwargs,
            token_blocks,
          )
        except Exception as exc:
          record_stats(runtime_profile, {
            "streaming_attention_fallback_calls": 1,
            "streaming_attention_fallback_error": f"{type(exc).__name__}: {exc}",
          })
      elif cache is not None and hidden_states is not None and hasattr(self, "qkv"):
        stats = cache.capture_pre_rope_qkv(
          hidden_states,
          self.qkv,
          self,
          layer_key=layer_key_for(self),
          token_blocks=token_blocks,
        )
        record_stats(runtime_profile, stats)

    return original_forward(self, *args, **kwargs)

  attention_cls.forward = patched_forward
  attention_cls._openpilot_streaming_vision_cache_patch = True
  attention_cls._openpilot_streaming_vision_cache_original_forward = original_forward


def _patch_qwen3vl_capture_safety() -> None:
  try:
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl_mod
    from transformers import masking_utils as masking_utils_mod
  except Exception:
    return

  if hasattr(masking_utils_mod, "_ignore_causal_mask_sdpa") and not getattr(
    masking_utils_mod._ignore_causal_mask_sdpa, "_openpilot_capture_patch", False
  ):
    original_ignore_causal_mask_sdpa = masking_utils_mod._ignore_causal_mask_sdpa

    def _ignore_causal_mask_sdpa(
      padding_mask,
      query_length: int,
      kv_length: int,
      kv_offset: int,
      local_attention_size: int | None = None,
    ) -> bool:
      if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
      return original_ignore_causal_mask_sdpa(
        padding_mask=padding_mask,
        query_length=query_length,
        kv_length=kv_length,
        kv_offset=kv_offset,
        local_attention_size=local_attention_size,
      )

    _ignore_causal_mask_sdpa._openpilot_capture_patch = True
    masking_utils_mod._ignore_causal_mask_sdpa = _ignore_causal_mask_sdpa

  if hasattr(masking_utils_mod, "eager_mask") and not getattr(masking_utils_mod.eager_mask, "_openpilot_eager_mask_patch", False):
    original_sdpa_mask = masking_utils_mod.sdpa_mask

    def eager_mask(
      batch_size: int,
      cache_position: Any,
      kv_length: int,
      kv_offset: int = 0,
      mask_function: Any = None,
      attention_mask: Any = None,
      dtype: Any = None,
      **kwargs,
    ):
      if dtype is None:
        dtype = torch.float32
      _ = kwargs.pop("allow_is_causal_skip", None)
      mask = original_sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        dtype=dtype,
        **kwargs,
      )
      if mask is None:
        return None
      min_dtype = torch.finfo(dtype).min
      zero = torch.zeros((), device=mask.device, dtype=dtype)
      min_value = zero.new_full((), min_dtype)
      return torch.where(mask, zero, min_value)

    eager_mask._openpilot_eager_mask_patch = True
    masking_utils_mod.eager_mask = eager_mask
    if hasattr(masking_utils_mod, "ALL_MASK_ATTENTION_FUNCTIONS"):
      masking_utils_mod.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["eager"] = eager_mask

  vision_encoder_cls = getattr(qwen3_vl_mod, "Qwen3VLVisionModel", None)
  if vision_encoder_cls is not None and not getattr(
    vision_encoder_cls, "_openpilot_fast_pos_embed_interpolate_patch", False
  ):
    original_fast_pos_embed_interpolate = vision_encoder_cls.fast_pos_embed_interpolate

    def fast_pos_embed_interpolate(self, grid_thw):
      precomputed = getattr(self, "_openpilot_capture_precomputed_fast_pos_embeds", None)
      if precomputed is not None:
        return precomputed
      return original_fast_pos_embed_interpolate(self, grid_thw)

    vision_encoder_cls.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    vision_encoder_cls._openpilot_fast_pos_embed_interpolate_patch = True

  qwen_cls = getattr(qwen3_vl_mod, "Qwen3VLForConditionalGeneration", None)
  if qwen_cls is not None and not getattr(qwen_cls, "_openpilot_prepare_inputs_patch", False):
    original_prepare = qwen_cls.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
      self,
      input_ids,
      past_key_values=None,
      attention_mask=None,
      inputs_embeds=None,
      cache_position=None,
      position_ids=None,
      use_cache=True,
      pixel_values=None,
      pixel_values_videos=None,
      image_grid_thw=None,
      video_grid_thw=None,
      **kwargs,
    ):
      model_inputs = original_prepare(
        self,
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=use_cache,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
      )
      model_inputs["position_ids"] = None
      if past_key_values is not None:
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None
      return model_inputs

    qwen_cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    qwen_cls._openpilot_prepare_inputs_patch = True

  model_cls = getattr(qwen3_vl_mod, "Qwen3VLModel", None)
  if model_cls is not None and not getattr(model_cls, "_openpilot_forward_patch", False):
    original_get_image_features = model_cls.get_image_features
    original_get_video_features = model_cls.get_video_features

    def get_image_features(self, pixel_values, image_grid_thw=None):
      cached = getattr(self, "_openpilot_capture_precomputed_image_features", None)
      if cached is not None:
        return cached
      return original_get_image_features(self, pixel_values, image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
      cached = getattr(self, "_openpilot_capture_precomputed_video_features", None)
      if cached is not None:
        return cached
      return original_get_video_features(self, pixel_values_videos, video_grid_thw)

    model_cls._openpilot_original_get_image_features = original_get_image_features
    model_cls._openpilot_original_get_video_features = original_get_video_features
    model_cls.get_image_features = get_image_features
    model_cls.get_video_features = get_video_features
    model_cls._openpilot_forward_patch = True

  vision_cls = getattr(qwen3_vl_mod, "Qwen3VLVisionPatchEmbed", None)
  if vision_cls is not None and not getattr(vision_cls, "_openpilot_forward_patch", False):
    import torch

    original_forward = vision_cls.forward

    def forward(self, hidden_states: Any) -> Any:  # pragma: no branch
      if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        with torch.backends.cudnn.flags(enabled=False):
          return original_forward(self, hidden_states)
      return original_forward(self, hidden_states)

    vision_cls.forward = forward
    vision_cls._openpilot_forward_patch = True


def _patch_alpamayo_init_for_paro(torch_mod: Any, config: LocalAlpamayoConfig) -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod

  patch_key = (
    config.paro_marlin_input_dtype,
    config.paro_compute_dtype,
    config.paro_output_dtype,
  )
  if getattr(alpamayo_mod.Alpamayo1_5, "_openpilot_paro_patch_key", None) == patch_key:
    return

  repo_root = Path(__file__).resolve().parents[2]
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
  from tools.alpamayo_speed.paro_native_marlin import apply_native_paro_linear_replacements

  original_init = getattr(
    alpamayo_mod.Alpamayo1_5,
    "_openpilot_paro_original_init",
    alpamayo_mod.Alpamayo1_5.__init__,
  )

  def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    original_init(self, *args, **kwargs)
    records = apply_native_paro_linear_replacements(
      self,
      marlin_input_dtype=config.paro_marlin_input_dtype,
      compute_dtype=_torch_dtype(torch_mod, config.paro_compute_dtype),
      output_dtype=config.paro_output_dtype,
    )
    self._openpilot_paro_replacement_records = records

  alpamayo_mod.Alpamayo1_5._openpilot_paro_original_init = original_init
  alpamayo_mod.Alpamayo1_5.__init__ = patched_init
  alpamayo_mod.Alpamayo1_5._openpilot_paro_patch_key = patch_key


def _repair_flattened_visual_patch_embed(torch_mod: Any, model: Any, model_path: Path) -> dict[str, Any]:
  key = "vlm.model.visual.patch_embed.proj.weight"
  index_path = model_path / "model.safetensors.index.json"
  try:
    from safetensors.torch import safe_open

    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    if key not in weight_map:
      return {"status": "skipped", "reason": "key_not_in_index"}
    target = model.vlm.model.visual.patch_embed.proj.weight
    with safe_open(str(model_path / weight_map[key]), framework="pt", device="cpu") as handle:
      weight = handle.get_tensor(key)
    if weight.ndim != 2 or int(weight.numel()) != int(target.numel()):
      return {"status": "skipped", "reason": "shape_not_compatible", "sourceShape": list(weight.shape), "targetShape": list(target.shape)}
    with torch_mod.no_grad():
      target.copy_(weight.reshape_as(target).to(device=target.device, dtype=target.dtype))
    return {"status": "ok", "sourceShape": list(weight.shape), "targetShape": list(target.shape), "targetDtype": str(target.dtype)}
  except Exception as exc:
    return {"status": "error", "type": type(exc).__name__, "message": str(exc)}


def _patch_flattened_conv3d_loader(torch_mod: Any):
  original = torch_mod.nn.Module.load_state_dict

  def patched(module: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(module, torch_mod.nn.Conv3d) and "weight" in state_dict:
      weight = state_dict["weight"]
      target = module.weight
      if (
        getattr(weight, "ndim", None) == 2
        and tuple(target.shape)[0] == int(weight.shape[0])
        and int(weight.numel()) == int(target.numel())
      ):
        state_dict = dict(state_dict)
        state_dict["weight"] = weight.reshape_as(target)
    return original(module, state_dict, *args, **kwargs)

  torch_mod.nn.Module.load_state_dict = patched

  def restore() -> None:
    torch_mod.nn.Module.load_state_dict = original

  return restore


def _decode_frame_rgb(frame: dict[str, Any]) -> np.ndarray:
  try:
    raw = base64.b64decode(frame["dataBase64"])
  except (KeyError, ValueError, TypeError) as exc:
    raise ValueError("frame dataBase64 invalid") from exc

  encoding = frame.get("encoding")
  width = int(frame["width"])
  height = int(frame["height"])

  import cv2

  if encoding == FRAME_ENCODING_JPEG_BGR:
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
      raise ValueError("jpeg_bgr decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

  if encoding == FRAME_ENCODING_NV12:
    stride = int(frame.get("stride", width))
    expected = stride * height * 3 // 2
    if len(raw) < expected:
      raise ValueError("nv12 frame data too short")
    nv12 = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height + height // 2, stride))
    bgr = cv2.cvtColor(nv12[:, :width], cv2.COLOR_YUV2BGR_NV12)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

  raise ValueError(f"unsupported frame encoding for Alpamayo adapter: {encoding}")


def _frame_cache_signature(frame: dict[str, Any]) -> tuple[Any, ...]:
  data_b64 = frame.get("dataBase64", "")
  if not isinstance(data_b64, str):
    data_b64 = ""
  return (
    int(frame.get("frameId", -1)),
    int(frame.get("timestampSof", 0)),
    int(frame.get("timestampEof", 0)),
    int(frame.get("width", 0)),
    int(frame.get("height", 0)),
    _freeze_cache_key_value(frame.get("encoding")),
    len(data_b64),
    zlib.crc32(data_b64.encode("utf-8")) & 0xFFFFFFFF,
  )


def _window_signature(frames_by_stream: dict[str, list[dict[str, Any]]], nav_text: str | None, camera_streams: tuple[str, ...]) -> tuple[Any, ...]:
  stream_signatures = tuple(
    (stream, tuple(_frame_cache_signature(frame) for frame in frames_by_stream[stream]))
    for stream in camera_streams
  )
  nav_value = "" if nav_text is None else nav_text
  return (stream_signatures, nav_value)


def _max_suffix_prefix_overlap(
  prior_window: tuple[tuple[Any, ...], ...],
  next_window: tuple[tuple[Any, ...], ...],
) -> int:
  max_overlap = min(len(prior_window), len(next_window))
  for overlap in range(max_overlap, 0, -1):
    if prior_window[-overlap:] == next_window[:overlap]:
      return overlap
  return 0


def _group_frames(request: dict[str, Any], streams: tuple[str, ...], num_frames: int) -> dict[str, list[dict[str, Any]]]:
  grouped: dict[str, list[dict[str, Any]]] = {stream: [] for stream in streams}
  for frame in request.get("frames", []):
    stream = frame.get("stream")
    if stream in grouped:
      grouped[stream].append(frame)
  for stream, frames in grouped.items():
    frames.sort(key=lambda item: (int(item.get("timestampEof", 0)), int(item.get("frameId", 0))))
    if len(frames) < num_frames:
      raise ValueError(f"not enough {stream} frames for Alpamayo: have={len(frames)} need={num_frames}")
    grouped[stream] = frames[-num_frames:]
  return grouped


def _request_nav_text(request: dict[str, Any]) -> str | None:
  navigation = request.get("navigation")
  if not isinstance(navigation, dict):
    return None
  text = str(navigation.get("text") or "").strip()
  return text or None


def _request_ego_history(request: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
  ego = request.get("egoHistory")
  if not isinstance(ego, dict):
    raise ValueError("egoHistory missing")
  xyz = np.asarray(ego.get("xyz"), dtype=np.float32)
  rot = np.asarray(ego.get("rot"), dtype=np.float32)
  if xyz.shape != (DEFAULT_NUM_HISTORY_STEPS, 3):
    raise ValueError(f"egoHistory.xyz shape invalid: {xyz.shape}")
  if rot.shape != (DEFAULT_NUM_HISTORY_STEPS, 3, 3):
    raise ValueError(f"egoHistory.rot shape invalid: {rot.shape}")
  if not (np.all(np.isfinite(xyz)) and np.all(np.isfinite(rot))):
    raise ValueError("egoHistory contains non-finite values")
  return xyz, rot


def _euler_from_rot_mats(rot_mats: np.ndarray) -> np.ndarray:
  eulers = np.zeros((rot_mats.shape[0], 3), dtype=np.float32)
  # Intrinsic roll/pitch/yaw extraction is not needed by control here; yaw is.
  eulers[:, 2] = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0]).astype(np.float32)
  eulers[:, 2] = np.unwrap(eulers[:, 2]).astype(np.float32)
  return eulers


def _gradient(values: np.ndarray, t: np.ndarray) -> np.ndarray:
  edge_order = 2 if len(t) >= 3 else 1
  return np.gradient(values, t, axis=0, edge_order=edge_order).astype(np.float32)


def _desired_curvature(position: np.ndarray, t: np.ndarray) -> float:
  velocity = _gradient(position, t)
  accel = _gradient(velocity, t)
  denom = np.power(np.maximum(velocity[:, 0] ** 2 + velocity[:, 1] ** 2, 1e-4), 1.5)
  curvature = (velocity[:, 0] * accel[:, 1] - velocity[:, 1] * accel[:, 0]) / denom
  finite = curvature[np.isfinite(curvature)]
  if len(finite) == 0:
    return 0.0
  return float(np.clip(finite[min(2, len(finite) - 1)], -0.25, 0.25))


def semantic_response_from_prediction(
  pred_xyz: np.ndarray,
  pred_rot: np.ndarray,
  *,
  source: str = "remoteServer",
  confidence: float = 0.85,
  consistency: float = 1.0,
  debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
  position = np.asarray(pred_xyz, dtype=np.float32)
  rotations = np.asarray(pred_rot, dtype=np.float32)
  if position.shape != (64, 3):
    raise ValueError(f"pred_xyz must be [64,3], got {position.shape}")
  if rotations.shape != (64, 3, 3):
    raise ValueError(f"pred_rot must be [64,3,3], got {rotations.shape}")

  t = np.arange(1, position.shape[0] + 1, dtype=np.float32) * DEFAULT_HISTORY_DT_S
  orientation = _euler_from_rot_mats(rotations)
  velocity = _gradient(position, t)
  acceleration = _gradient(velocity, t)
  orientation_rate = _gradient(orientation, t)

  trajectory = {
    "position": xyzt_to_dict(t, position),
    "orientation": xyzt_to_dict(t, orientation),
    "velocity": xyzt_to_dict(t, velocity),
    "orientationRate": xyzt_to_dict(t, orientation_rate),
    "acceleration": xyzt_to_dict(t, acceleration),
  }
  semantic: dict[str, Any] = {
    "source": source,
    "status": "valid",
    "confidence": float(np.clip(confidence, 0.0, 1.0)),
    "consistency": float(np.clip(consistency, 0.0, 1.0)),
    "age": 0.0,
    "desiredCurvature": _desired_curvature(position, t),
    "desiredAcceleration": float(acceleration[0, 0]) if len(acceleration) else 0.0,
    "shouldStop": False,
    "blendHint": 1.0,
    "trajectory": trajectory,
  }
  if debug is not None:
    semantic["debug"] = debug
  return {"semanticPlan": semantic}


class LocalAlpamayoAdapter:
  def __init__(self, config: LocalAlpamayoConfig | None = None):
    self.config = config or config_from_env()
    self._loaded = False
    self._torch = None
    self._helper = None
    self._model = None
    self._processor = None
    self._frame_cache: dict[str, OrderedDict[int, tuple[tuple[Any, ...], Any]]] = {}
    self._cache_lock = threading.RLock()
    self._frame_cache_max_entries = max(2, self.config.num_frames * 8)
    self._stream_window_cache: dict[str, OrderedDict[tuple[tuple[Any, ...], ...], Any]] = {}
    self._stream_window_cache_max_entries = max(4, self.config.num_frames * 4)
    self._tokenized_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    self._tokenized_cache_max_entries = max(4, self.config.num_frames * 4)
    self._rope_index_cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
    self._rope_index_cache_max_entries = max(4, self.config.num_frames * 4)
    self._vlm_prefix_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    self._vlm_prefix_cache_max_entries = max(1, self.config.vlm_prefix_cache_max_entries)
    self._adaptive_flow_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    self._adaptive_flow_cache_max_entries = max(1, self.config.adaptive_flow_cache_max_entries)
    self._streaming_vision_cache = StreamingAlpamayoVisionCache(
      self.config.camera_streams,
      self.config.num_frames,
      max_entries_per_stream=max(4, self.config.num_frames * 4),
    )
    self._cuda_graph_cache: OrderedDict[tuple[Any, ...], _CachedCudaGraph] = OrderedDict()
    self._cuda_graph_stage_caches: dict[str, OrderedDict[tuple[Any, ...], _CachedCudaGraph]] = {
      "visual": OrderedDict(),
      "prefill": OrderedDict(),
      "decode": OrderedDict(),
      "action": OrderedDict(),
    }
    self._cuda_graph_cache_size = max(1, self.config.cuda_graph_cache_size)
    self._supports_cuda_graph = bool(self.config.cuda_graphs)
    self._cache_position_seed_buffer = None
    self._cache_position_seed_buffer_len = 0
    self._cache_position_seed_buffer_device = None
    self._ego_history_xyz_buffer = None
    self._ego_history_rot_buffer = None
    self._dflash_model = None
    self._dflash_mask_embedding = None
    self._dflash_layer_ids: list[int] = []
    self._dflash_loaded = False
    self._dflash_load_error: str | None = None
    self._dflash_sticky_disabled = False
    self._dflash_disable_cooldown_remaining = 0
    self._dflash_original_manual_generate = None
    self._dflash_manual_generate_installed = False
    self._target_model_identity: dict[str, Any] = {}
    self._visual_feature_cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
    self._visual_feature_cache_max_entries = max(8, len(self.config.camera_streams) * self.config.num_frames * 8)
    self._vlm_precomputed_lock = threading.Lock()
    self._vlm_text_prefix_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    self._vlm_text_prefix_cache_max_entries = max(2, self._vlm_prefix_cache_max_entries)


  @staticmethod
  def _detach_tensor_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
      return value.detach()
    if isinstance(value, list):
      return [LocalAlpamayoAdapter._detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
      return tuple(LocalAlpamayoAdapter._detach_tensor_tree(item) for item in value)
    if isinstance(value, dict):
      return {key: LocalAlpamayoAdapter._detach_tensor_tree(item) for key, item in value.items()}
    return value

  @staticmethod
  def _detach_clone_tensor_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
      detached = value.detach()
      return detached.clone() if hasattr(detached, "clone") else detached
    if isinstance(value, list):
      return [LocalAlpamayoAdapter._detach_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
      return tuple(LocalAlpamayoAdapter._detach_clone_tensor_tree(item) for item in value)
    if isinstance(value, dict):
      return {key: LocalAlpamayoAdapter._detach_clone_tensor_tree(item) for key, item in value.items()}
    return value

  @staticmethod
  def _visual_feature_token_count(feature: Any) -> int:
    shape = getattr(feature, "shape", None)
    if shape is not None and len(shape) >= 1:
      return int(shape[0])
    return -1

  @staticmethod
  def _visual_pixel_slices_for_grid(pixel_values: Any, grid_rows: list[list[int]]) -> list[Any] | None:
    if pixel_values is None or not hasattr(pixel_values, "shape") or not grid_rows:
      return None
    raw_patch_counts = [
      max(0, int(row[0]) * int(row[1]) * int(row[2])) if len(row) == 3 else 0
      for row in grid_rows
    ]
    first_dim = int(pixel_values.shape[0])
    if first_dim == sum(raw_patch_counts):
      offset = 0
      slices = []
      for count in raw_patch_counts:
        slices.append(pixel_values[offset:offset + count])
        offset += count
      return slices
    if first_dim == len(grid_rows):
      return [pixel_values[idx:idx + 1] for idx in range(len(grid_rows))]
    return None

  @staticmethod
  def _select_deepstack_item(deepstack_features: Any, index: int, item_count: int) -> Any:
    if deepstack_features is None:
      return None
    if isinstance(deepstack_features, (list, tuple)):
      if len(deepstack_features) == item_count and all(not isinstance(item, (list, tuple)) for item in deepstack_features):
        return deepstack_features[index]
      selected = []
      for layer_features in deepstack_features:
        if isinstance(layer_features, (list, tuple)) and len(layer_features) == item_count:
          selected.append(layer_features[index])
        else:
          raise ValueError("deepstack features are not split per visual item")
      return tuple(selected)
    raise ValueError("deepstack features are not split per visual item")

  @staticmethod
  def _split_tensor_by_token_counts(tensor: Any, token_counts: list[int]) -> tuple[Any, ...] | None:
    if tensor is None or not hasattr(tensor, "shape"):
      return None
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 1:
      return None
    counts = [int(count) for count in token_counts]
    if int(shape[0]) != sum(counts):
      return None
    return tuple(tensor.split(counts, dim=0))

  @staticmethod
  def _deepstack_item_matches_token_count(item: Any, token_count: int) -> bool:
    if item is None:
      return True
    if isinstance(item, (list, tuple)):
      return all(LocalAlpamayoAdapter._visual_feature_token_count(layer_item) == int(token_count) for layer_item in item)
    return LocalAlpamayoAdapter._visual_feature_token_count(item) == int(token_count)

  @staticmethod
  def _split_deepstack_items(deepstack_features: Any, token_counts: list[int]) -> list[Any]:
    item_count = len(token_counts)
    if item_count == 0:
      return []
    if deepstack_features is None:
      return [None] * item_count

    counts = [int(count) for count in token_counts]
    direct_tensor_split = LocalAlpamayoAdapter._split_tensor_by_token_counts(deepstack_features, counts)
    if direct_tensor_split is not None:
      return list(direct_tensor_split)

    if not isinstance(deepstack_features, (list, tuple)):
      raise ValueError("deepstack features are not split per visual item")

    if len(deepstack_features) == item_count and all(
      LocalAlpamayoAdapter._deepstack_item_matches_token_count(item, counts[idx])
      for idx, item in enumerate(deepstack_features)
    ):
      return list(deepstack_features)

    per_item_layers: list[list[Any]] = [[] for _ in range(item_count)]
    for layer_features in deepstack_features:
      if isinstance(layer_features, (list, tuple)) and len(layer_features) == item_count:
        layer_parts = list(layer_features)
        if not all(
          LocalAlpamayoAdapter._visual_feature_token_count(layer_parts[idx]) == counts[idx]
          for idx in range(item_count)
        ):
          raise ValueError("deepstack layer split lengths do not match visual token counts")
      else:
        split_parts = LocalAlpamayoAdapter._split_tensor_by_token_counts(layer_features, counts)
        if split_parts is None:
          raise ValueError("deepstack layer is not split by visual token counts")
        layer_parts = list(split_parts)
      for idx, part in enumerate(layer_parts):
        per_item_layers[idx].append(part)

    return [tuple(item_layers) for item_layers in per_item_layers]

  def _assemble_deepstack_items(self, deepstack_items: list[Any]) -> Any:
    if not deepstack_items or all(item is None for item in deepstack_items):
      return None
    present_items = [item for item in deepstack_items if item is not None]
    if present_items and len(present_items) == len(deepstack_items) and all(isinstance(item, (list, tuple)) for item in present_items):
      layer_count = len(present_items[0])
      if all(len(item) == layer_count for item in present_items):
        assembled_layers = []
        for layer_idx in range(layer_count):
          layer_parts = [item[layer_idx] for item in deepstack_items]
          if self._torch is not None and all(hasattr(part, "shape") for part in layer_parts):
            assembled_layers.append(self._torch.cat(layer_parts, dim=0))
          else:
            assembled_layers.append(tuple(layer_parts))
        return assembled_layers
    if self._torch is not None and len(present_items) == len(deepstack_items) and all(hasattr(item, "shape") for item in present_items):
      return [self._torch.cat(deepstack_items, dim=0)]
    return tuple(deepstack_items)

  def _visual_feature_cache_key(self, kind: str, block: dict[str, Any], grid_row: list[int]) -> tuple[Any, ...]:
    visual = getattr(getattr(getattr(self._model, "vlm", None), "model", None), "visual", None)
    visual_config = getattr(visual, "config", None)
    return (
      kind,
      str(block.get("stream", "")),
      _freeze_cache_key_value(block.get("signature", ())),
      tuple(int(value) for value in grid_row),
      int(getattr(visual, "spatial_merge_size", 0) or getattr(visual_config, "spatial_merge_size", 0) or 0),
      int(getattr(visual, "temporal_patch_size", 0) or getattr(visual_config, "temporal_patch_size", 0) or 0),
      int(getattr(visual_config, "patch_size", 0) or 0),
      str(self.config.target_model),
    )

  def _vlm_prefix_cache_key(self, window_signature: tuple[Any, ...], tokenized_data: dict[str, Any]) -> tuple[Any, ...]:
    assert self._torch is not None
    visual = getattr(getattr(getattr(self._model, "vlm", None), "model", None), "visual", None)
    visual_config = getattr(visual, "config", None)
    tokenized_signature = _tensor_tree_signature(
      self._torch,
      {
        key: tokenized_data.get(key)
        for key in (
          "input_ids",
          "attention_mask",
          "pixel_values",
          "pixel_values_videos",
          "image_grid_thw",
          "video_grid_thw",
          "cache_position_ids",
          "cache_rope_deltas",
        )
        if key in tokenized_data
      },
    )
    token_content_signature = tokenized_data.get("_openpilot_prefix_semantic_signature")
    if token_content_signature is None:
      token_content_signature = _tensor_tree_content_signature(
        self._torch,
        {
          key: tokenized_data.get(key)
          for key in (
            "input_ids",
            "attention_mask",
            "image_grid_thw",
            "video_grid_thw",
          )
          if key in tokenized_data
        },
      )
    return (
      "vlm_prefix_v2",
      self.config.vlm_runtime_backend,
      str(self.config.target_model),
      tuple(self.config.camera_streams),
      int(self.config.num_frames),
      int(getattr(visual, "spatial_merge_size", 0) or getattr(visual_config, "spatial_merge_size", 0) or 0),
      int(getattr(visual, "temporal_patch_size", 0) or getattr(visual_config, "temporal_patch_size", 0) or 0),
      int(getattr(visual_config, "patch_size", 0) or 0),
      _freeze_cache_key_value(window_signature),
      _freeze_cache_key_value(tokenized_signature),
      _freeze_cache_key_value(token_content_signature),
      _freeze_cache_key_value(tokenized_data.get("_openpilot_fused_input_ids_signature", ())),
    )

  def _language_visual_token_spans(
    self,
    tokenized_data: dict[str, Any],
    token_blocks: list[dict[str, Any]],
  ) -> list[dict[str, Any]]:
    input_ids = tokenized_data.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "shape") or len(input_ids.shape) < 2:
      return []
    model_vlm_config = getattr(getattr(getattr(self._model, "vlm", None), "config", None), "__dict__", None)
    image_token_id = getattr(getattr(self._model, "vlm", None), "config", None)
    image_token_id = getattr(image_token_id, "image_token_id", None)
    if image_token_id is None or self._torch is None:
      return []
    try:
      first_row = input_ids[0]
      positions_tensor = (first_row == int(image_token_id)).nonzero(as_tuple=False).flatten()
      positions = [int(item) for item in positions_tensor.detach().cpu().tolist()]
    except Exception:
      return []
    expected = sum(max(0, int(block.get("token_count", 0) or 0)) for block in token_blocks)
    if expected <= 0 or len(positions) != expected:
      return []
    spans: list[dict[str, Any]] = []
    offset = 0
    for block in token_blocks:
      token_count = max(0, int(block.get("token_count", 0) or 0))
      if token_count <= 0:
        continue
      block_positions = positions[offset : offset + token_count]
      offset += token_count
      if len(block_positions) != token_count:
        return []
      contiguous = all(
        int(block_positions[idx]) + 1 == int(block_positions[idx + 1])
        for idx in range(len(block_positions) - 1)
      )
      spans.append({
        "stream": str(block.get("stream", "")),
        "stream_index": int(block.get("stream_index", -1)),
        "frame_index": int(block.get("frame_index", -1)),
        "prompt_index": int(block.get("prompt_index", -1)),
        "frame_id": int(block.get("frame_id", -1)),
        "signature": tuple(block.get("signature", ())),
        "visual_token_start": int(block.get("token_start", -1)),
        "language_token_start": int(block_positions[0]),
        "language_token_end": int(block_positions[-1]) + 1,
        "token_count": token_count,
        "contiguous": bool(contiguous),
      })
    return spans

  @staticmethod
  def _shifted_prompt_kv_reuse_plan(
    source_spans: list[dict[str, Any]],
    current_spans: list[dict[str, Any]],
  ) -> dict[str, Any]:
    by_key: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    for span in source_spans:
      if not isinstance(span, dict) or not bool(span.get("contiguous")):
        continue
      by_key[(str(span.get("stream", "")), tuple(span.get("signature", ())))] = span
    ranges: list[dict[str, Any]] = []
    for span in current_spans:
      if not isinstance(span, dict) or not bool(span.get("contiguous")):
        continue
      key = (str(span.get("stream", "")), tuple(span.get("signature", ())))
      source = by_key.get(key)
      if source is None:
        continue
      token_count = int(span.get("token_count", 0) or 0)
      if token_count <= 0 or token_count != int(source.get("token_count", -1) or -1):
        continue
      source_start = int(source.get("language_token_start", -1))
      current_start = int(span.get("language_token_start", -1))
      if source_start < 0 or current_start < 0:
        continue
      ranges.append({
        "stream": key[0],
        "signature": key[1],
        "source_frame_index": int(source.get("frame_index", -1)),
        "current_frame_index": int(span.get("frame_index", -1)),
        "source_prompt_index": int(source.get("prompt_index", -1)),
        "current_prompt_index": int(span.get("prompt_index", -1)),
        "source_language_token_start": source_start,
        "current_language_token_start": current_start,
        "token_count": token_count,
        "language_position_shift": current_start - source_start,
        "source_visual_token_start": int(source.get("visual_token_start", -1)),
        "current_visual_token_start": int(span.get("visual_token_start", -1)),
      })
    total_current = sum(max(0, int(span.get("token_count", 0) or 0)) for span in current_spans if isinstance(span, dict))
    retained_tokens = sum(int(item["token_count"]) for item in ranges)
    return {
      "range_count": len(ranges),
      "retained_language_tokens": retained_tokens,
      "current_visual_language_tokens": total_current,
      "retained_ratio": float(retained_tokens) / float(total_current) if total_current > 0 else 0.0,
      "ranges": ranges,
    }

  @staticmethod
  def _validate_shifted_prompt_kv_reuse_plan(
    plan: dict[str, Any],
    *,
    source_input_seq_len: int,
    current_input_seq_len: int,
  ) -> dict[str, Any]:
    if not isinstance(plan, dict):
      return {"valid": False, "invalid_reason": "missing_plan"}
    ranges = plan.get("ranges")
    if not isinstance(ranges, list) or not ranges:
      return {**plan, "valid": False, "invalid_reason": "empty_ranges"}

    source_intervals: list[tuple[int, int]] = []
    current_intervals: list[tuple[int, int]] = []
    for idx, item in enumerate(ranges):
      if not isinstance(item, dict):
        return {**plan, "valid": False, "invalid_reason": f"bad_range:{idx}"}
      source_start = int(item.get("source_language_token_start", -1))
      current_start = int(item.get("current_language_token_start", -1))
      token_count = int(item.get("token_count", 0) or 0)
      if source_start < 0 or current_start < 0 or token_count <= 0:
        return {**plan, "valid": False, "invalid_reason": f"bad_bounds:{idx}"}
      source_end = source_start + token_count
      current_end = current_start + token_count
      if source_end > int(source_input_seq_len):
        return {**plan, "valid": False, "invalid_reason": f"source_oob:{idx}"}
      if current_end > int(current_input_seq_len):
        return {**plan, "valid": False, "invalid_reason": f"current_oob:{idx}"}
      source_intervals.append((source_start, source_end))
      current_intervals.append((current_start, current_end))

    for label, intervals in (("source", source_intervals), ("current", current_intervals)):
      ordered = sorted(intervals)
      for idx in range(1, len(ordered)):
        if ordered[idx][0] < ordered[idx - 1][1]:
          return {**plan, "valid": False, "invalid_reason": f"{label}_overlap:{idx}"}

    retained = sum(end - start for start, end in source_intervals)
    current_total = int(plan.get("current_visual_language_tokens", 0) or 0)
    return {
      **plan,
      "valid": True,
      "invalid_reason": "",
      "validated_source_input_seq_len": int(source_input_seq_len),
      "validated_current_input_seq_len": int(current_input_seq_len),
      "validated_retained_language_tokens": int(retained),
      "validated_retained_ratio": float(retained) / float(current_total) if current_total > 0 else 0.0,
    }

  @staticmethod
  def _window_suffix_prefix_overlap(prior_window_signature: Any, next_window_signature: Any) -> tuple[int, int, float]:
    try:
      prior_streams, prior_nav = prior_window_signature
      next_streams, next_nav = next_window_signature
    except Exception:
      return 0, 0, 0.0
    if prior_nav != next_nav:
      return 0, 0, 0.0
    try:
      prior_by_stream = {str(stream): tuple(signatures) for stream, signatures in prior_streams}
      next_by_stream = {str(stream): tuple(signatures) for stream, signatures in next_streams}
    except Exception:
      return 0, 0, 0.0

    overlap = 0
    total = 0
    for stream, next_signatures in next_by_stream.items():
      prior_signatures = prior_by_stream.get(stream, ())
      total += len(next_signatures)
      overlap += _max_suffix_prefix_overlap(tuple(prior_signatures), tuple(next_signatures))
    if total <= 0:
      return 0, 0, 0.0
    return overlap, total, float(overlap) / float(total)

  @staticmethod
  def _window_set_overlap(prior_window_signature: Any, next_window_signature: Any) -> tuple[int, int, float]:
    try:
      prior_streams, prior_nav = prior_window_signature
      next_streams, next_nav = next_window_signature
    except Exception:
      return 0, 0, 0.0
    if prior_nav != next_nav:
      return 0, 0, 0.0
    try:
      prior_by_stream = {str(stream): set(tuple(signatures)) for stream, signatures in prior_streams}
      next_by_stream = {str(stream): tuple(signatures) for stream, signatures in next_streams}
    except Exception:
      return 0, 0, 0.0

    overlap = 0
    total = 0
    for stream, next_signatures in next_by_stream.items():
      prior_signatures = prior_by_stream.get(stream, set())
      total += len(next_signatures)
      overlap += sum(1 for signature in next_signatures if signature in prior_signatures)
    if total <= 0:
      return 0, 0, 0.0
    return overlap, total, float(overlap) / float(total)

  def _vlm_prefix_entry_full_generation_ready(self, entry: dict[str, Any], input_seq_len: int) -> bool:
    try:
      if self.config.disable_reasoning_generation and self._require_state_fresh_no_reasoning():
        entry["full_vlm_state_fresh_no_reasoning_forced_prefill"] = True
        return False
      if entry.get("full_vlm_generated_sequences") is None or entry.get("full_vlm_prompt_cache") is None:
        return False
      trusted_no_reasoning_replay = bool(
        self.config.disable_reasoning_generation
        and self.config.no_reasoning_trust_shifted_prompt_cache
        and not self._require_state_fresh_no_reasoning()
        and entry.get("streaming_vlm_trusted_replay_allowed")
      )
      trusted_fast_current_action_replay = bool(
        getattr(self, "_openpilot_require_fast_no_prefill", False)
        and entry.get("streaming_vlm_fast_current_action_replay_allowed")
      )
      if (
        int(entry.get("full_vlm_input_seq_len", -1)) != int(input_seq_len)
        and not trusted_fast_current_action_replay
      ):
        return False
      source_window_signature = entry.get("window_signature")
      generation_window_signature = entry.get("full_vlm_window_signature")
      if (
        source_window_signature is None
        or generation_window_signature is None
        or generation_window_signature != source_window_signature
      ) and not (trusted_no_reasoning_replay or trusted_fast_current_action_replay):
        return False
      if not bool(entry.get("full_vlm_prompt_cache_context_exact")) and not (trusted_no_reasoning_replay or trusted_fast_current_action_replay):
        return False
      if bool(entry.get("streaming_vlm_reuse_unverified")) and not (trusted_no_reasoning_replay or trusted_fast_current_action_replay):
        return False
      return True
    except Exception:
      return False

  def _vlm_prefix_entry_draft_source_ready(self, entry: dict[str, Any], input_seq_len: int) -> bool:
    try:
      if entry.get("full_vlm_generated_sequences") is None or entry.get("full_vlm_prompt_cache") is None:
        return False
      if int(entry.get("full_vlm_input_seq_len", -1)) != int(input_seq_len):
        return False
      if bool(entry.get("streaming_vlm_reuse_unverified")) and not self.config.streaming_vlm_trust_shifted_draft:
        return False
      return True
    except Exception:
      return False

  @staticmethod
  def _dflash_prefix_entry_full_generation_ready(entry: dict[str, Any], input_seq_len: int) -> bool:
    try:
      if entry.get("dflash_full_generated_sequences") is None or entry.get("dflash_full_prompt_cache") is None:
        return False
      if int(entry.get("dflash_full_input_seq_len", -1)) != int(input_seq_len):
        return False
      source_window_signature = entry.get("window_signature")
      generation_window_signature = entry.get("dflash_full_window_signature")
      if (
        source_window_signature is None
        or generation_window_signature is None
        or generation_window_signature != source_window_signature
      ):
        return False
      if not bool(entry.get("dflash_full_prompt_cache_context_exact")):
        return False
      if bool(entry.get("streaming_vlm_reuse_unverified")):
        return False
      return True
    except Exception:
      return False

  def _streaming_vlm_prefix_reuse_candidate(
    self,
    *,
    window_signature: tuple[Any, ...],
    input_seq_len: int,
    cache_stats: dict[str, Any] | None = None,
  ) -> tuple[tuple[Any, ...], dict[str, Any], int, int, float] | None:
    if not (
      self.config.streaming_vision_cache
      and self.config.streaming_vlm_prefix_reuse
      and self.config.persistent_vlm_prefix_cache
    ):
      return None
    min_overlap = max(0.0, min(1.0, float(self.config.streaming_vlm_prefix_reuse_min_overlap)))
    max_chain = max(0, int(self.config.streaming_vlm_prefix_reuse_max_chain))
    best: tuple[tuple[Any, ...], dict[str, Any], int, int, float] | None = None
    for source_key, source_entry in reversed(self._vlm_prefix_cache.items()):
      if not isinstance(source_entry, dict):
        continue
      if not self._vlm_prefix_entry_draft_source_ready(source_entry, input_seq_len):
        continue
      try:
        chain_depth = int(source_entry.get("streaming_vlm_reuse_chain_depth", 0) or 0)
      except Exception:
        chain_depth = 0
      if max_chain > 0 and chain_depth >= max_chain and not self.config.streaming_vlm_trust_shifted_draft:
        continue
      overlap, total, ratio = self._window_suffix_prefix_overlap(
        source_entry.get("window_signature"),
        window_signature,
      )
      if ratio < min_overlap and self.config.streaming_vlm_trust_shifted_draft:
        set_overlap, set_total, set_ratio = self._window_set_overlap(
          source_entry.get("window_signature"),
          window_signature,
        )
        if set_ratio > ratio:
          overlap, total, ratio = set_overlap, set_total, set_ratio
      if ratio < min_overlap:
        continue
      if best is None or ratio > best[4] or (ratio == best[4] and overlap > best[2]):
        best = (source_key, source_entry, overlap, total, ratio)
    if best is not None:
      return best

    if self.config.streaming_vlm_trust_shifted_draft and isinstance(cache_stats, dict):
      streaming_stats = cache_stats.get("streamingVisionCache", {})
      if isinstance(streaming_stats, dict):
        try:
          overlap = int(streaming_stats.get("frame_hits", 0) or 0)
          misses = int(streaming_stats.get("frame_misses", 0) or 0)
          total = overlap + misses
          ratio = float(overlap) / float(total) if total > 0 else 0.0
        except Exception:
          overlap, total, ratio = 0, 0, 0.0
        if total > 0 and ratio >= min_overlap:
          for source_key, source_entry in reversed(self._vlm_prefix_cache.items()):
            if not isinstance(source_entry, dict):
              continue
            if self._vlm_prefix_entry_draft_source_ready(source_entry, input_seq_len):
              source_entry["_streaming_vlm_candidate_overlap_source"] = "vision_cache_retained_frames"
              return source_key, source_entry, overlap, total, ratio
    return None

  def _record_vlm_prefix_cache_candidate(
    self,
    *,
    window_signature: tuple[Any, ...],
    tokenized_data: dict[str, Any],
    cache_stats: dict[str, Any],
  ) -> None:
    if not self.config.persistent_vlm_prefix_cache or self._torch is None:
      cache_stats["vlmPrefixCache"] = {"enabled": False}
      if self._model is not None:
        for attr in ("_openpilot_vlm_prefix_cache_entry", "_openpilot_vlm_prefix_cache_key"):
          if hasattr(self._model, attr):
            delattr(self._model, attr)
      return
    key = self._vlm_prefix_cache_key(window_signature, tokenized_data)
    input_ids = tokenized_data.get("input_ids")
    streaming_stats = cache_stats.get("streamingVisionCache", {}) if isinstance(cache_stats, dict) else {}
    visual_token_blocks = []
    if isinstance(streaming_stats, dict):
      visual_token_blocks = [
        dict(block) for block in streaming_stats.get("token_blocks", [])
        if isinstance(block, dict)
      ]
    language_visual_token_spans = self._language_visual_token_spans(tokenized_data, visual_token_blocks)
    entry = {
      "has_prompt_cache": False,
      "prefill_output": None,
      "reason": "awaiting_prefill_output",
      "hits": 0,
      "stores": 0,
      "dflash_has_selected_hidden": False,
      "dflash_hits": 0,
      "dflash_stores": 0,
      "input_seq_len": int(input_ids.shape[1]) if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2 else 0,
      "has_cache_position_ids": tokenized_data.get("cache_position_ids") is not None,
      "has_cache_rope_deltas": tokenized_data.get("cache_rope_deltas") is not None,
      "window_signature": window_signature,
      "visual_token_blocks": visual_token_blocks,
      "language_visual_token_spans": language_visual_token_spans,
      "prefix_semantic_signature": tokenized_data.get("_openpilot_prefix_semantic_signature"),
      "fused_input_ids_signature": tokenized_data.get("_openpilot_fused_input_ids_signature", ()),
      "cache_position_ids_signature": _tensor_tree_signature(
        self._torch,
        tokenized_data.get("cache_position_ids"),
      ) if self._torch is not None and tokenized_data.get("cache_position_ids") is not None else (),
      "cache_rope_deltas_signature": _tensor_tree_signature(
        self._torch,
        tokenized_data.get("cache_rope_deltas"),
      ) if self._torch is not None and tokenized_data.get("cache_rope_deltas") is not None else (),
    }
    force_vlm_refresh = bool(getattr(self, "_openpilot_force_vlm_refresh", False))
    with self._cache_lock:
      cached = self._vlm_prefix_cache.get(key)
      if cached is not None and not force_vlm_refresh:
        self._vlm_prefix_cache.move_to_end(key)
        cached["hits"] = int(cached.get("hits", 0)) + 1
        if self._model is not None:
          setattr(self._model, "_openpilot_vlm_prefix_cache_entry", cached)
          setattr(self._model, "_openpilot_vlm_prefix_cache_key", key)
        cache_stats["vlmPrefixCache"] = {
          "enabled": True,
          "hit": 1,
          "miss": 0,
          "depth": len(self._vlm_prefix_cache),
          "hasPromptCache": bool(cached.get("has_prompt_cache")),
          "reason": str(cached.get("reason", "")),
          "hits": int(cached.get("hits", 0)),
          "stores": int(cached.get("stores", 0)),
          "dflashHasSelectedHidden": bool(cached.get("dflash_has_selected_hidden")),
          "dflashReason": str(cached.get("dflash_reason", "")),
          "dflashHits": int(cached.get("dflash_hits", 0)),
          "dflashStores": int(cached.get("dflash_stores", 0)),
          "dflashFullGenerationReady": self._dflash_prefix_entry_full_generation_ready(
            cached,
            int(entry["input_seq_len"]),
          ),
          "dflashFullGenerationReason": str(cached.get("dflash_full_reason", "")),
          "dflashFullGenerationHits": int(cached.get("dflash_full_hits", 0)),
          "dflashFullGenerationStores": int(cached.get("dflash_full_stores", 0)),
          "fullGenerationReady": self._vlm_prefix_entry_full_generation_ready(
            cached,
            int(entry["input_seq_len"]),
          ),
          "fullGenerationReason": str(cached.get("full_vlm_reason", "")),
          "fullGenerationHits": int(cached.get("full_vlm_hits", 0)),
          "fullGenerationStores": int(cached.get("full_vlm_stores", 0)),
          "exactHit": True,
          "streamingReuseHit": bool(cached.get("streaming_vlm_reuse")),
          "streamingReuseOverlapRatio": float(cached.get("streaming_vlm_reuse_overlap_ratio", 0.0) or 0.0),
          "streamingReuseOverlapFrames": int(cached.get("streaming_vlm_reuse_overlap_frames", 0) or 0),
          "streamingReuseOverlapSource": str(cached.get("streaming_vlm_reuse_overlap_source", "")),
          "streamingReuseChainDepth": int(cached.get("streaming_vlm_reuse_chain_depth", 0) or 0),
          "streamingReuseMode": str(cached.get("streaming_vlm_reuse_mode", "")),
          "streamingReuseUnverified": bool(cached.get("streaming_vlm_reuse_unverified")),
          "trustedReplayAllowed": bool(cached.get("streaming_vlm_trusted_replay_allowed")),
          "languageVisualTokenSpans": len(cached.get("language_visual_token_spans", []) or []),
          "shiftedPromptKvReusePlan": cached.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan", {}),
        }
        return
      if force_vlm_refresh:
        reuse_candidate = None
        cache_stats["vlm_prefix_force_refresh"] = 1
      else:
        reuse_candidate = self._streaming_vlm_prefix_reuse_candidate(
          window_signature=window_signature,
          input_seq_len=int(entry["input_seq_len"]),
          cache_stats=cache_stats,
        )
      if reuse_candidate is not None:
        source_key, source_entry, overlap_frames, total_frames, overlap_ratio = reuse_candidate
        overlap_source = str(source_entry.pop("_streaming_vlm_candidate_overlap_source", "window_signature"))
        try:
          source_chain_depth = int(source_entry.get("streaming_vlm_reuse_chain_depth", 0) or 0)
        except Exception:
          source_chain_depth = 0
        refresh_interval = max(0, int(self.config.streaming_vlm_trusted_replay_refresh_interval))
        refresh_due = bool(refresh_interval > 0 and source_chain_depth + 1 >= refresh_interval)
        trusted_replay_requested = bool(
          self.config.streaming_vlm_trust_shifted_draft
          and source_entry.get("full_vlm_generated_sequences") is not None
          and source_entry.get("full_vlm_prompt_cache") is not None
        )
        source_cache_context_match = bool(
          source_entry.get("window_signature") == window_signature
          and source_entry.get("prefix_semantic_signature") == entry.get("prefix_semantic_signature")
          and source_entry.get("fused_input_ids_signature", ()) == entry.get("fused_input_ids_signature", ())
          and source_entry.get("cache_position_ids_signature") == entry.get("cache_position_ids_signature")
          and source_entry.get("cache_rope_deltas_signature") == entry.get("cache_rope_deltas_signature")
        )
        shifted_prompt_kv_reuse_plan = self._shifted_prompt_kv_reuse_plan(
          source_entry.get("language_visual_token_spans", []),
          entry.get("language_visual_token_spans", []),
        )
        shifted_prompt_kv_reuse_plan = self._validate_shifted_prompt_kv_reuse_plan(
          shifted_prompt_kv_reuse_plan,
          source_input_seq_len=int(source_entry.get("input_seq_len", source_entry.get("full_vlm_input_seq_len", 0)) or 0),
          current_input_seq_len=int(entry.get("input_seq_len", 0) or 0),
        )
        shifted_prompt_kv_retained_ratio = (
          float(shifted_prompt_kv_reuse_plan.get("validated_retained_ratio", shifted_prompt_kv_reuse_plan.get("retained_ratio", 0.0)) or 0.0)
          if isinstance(shifted_prompt_kv_reuse_plan, dict)
          else 0.0
        )
        no_reasoning_trusted_replay_allowed = bool(
          self.config.disable_reasoning_generation
          and self.config.no_reasoning_trust_shifted_prompt_cache
          and not self._require_state_fresh_no_reasoning()
          and trusted_replay_requested
          and int(source_entry.get("full_vlm_max_generation_length", -1)) == 0
          and int(source_entry.get("full_vlm_input_seq_len", -1)) == int(entry["input_seq_len"])
          and isinstance(shifted_prompt_kv_reuse_plan, dict)
          and shifted_prompt_kv_reuse_plan.get("valid")
          and shifted_prompt_kv_retained_ratio >= max(0.0, min(1.0, float(self.config.streaming_vlm_prefix_reuse_min_overlap)))
        )
        fast_current_action_replay_allowed = bool(
          getattr(self, "_openpilot_require_fast_no_prefill", False)
          and trusted_replay_requested
          and isinstance(shifted_prompt_kv_reuse_plan, dict)
          and shifted_prompt_kv_reuse_plan.get("valid")
          and shifted_prompt_kv_retained_ratio >= max(0.0, min(1.0, float(self.config.streaming_vlm_prefix_reuse_min_overlap)))
        )
        trusted_full_replay_allowed = bool(no_reasoning_trusted_replay_allowed or fast_current_action_replay_allowed)
        entry.update({
          "has_prompt_cache": False,
          "reason": (
            "streaming_shift_trusted_full_replay_no_reasoning"
            if no_reasoning_trusted_replay_allowed
            else "streaming_shift_trusted_full_replay_fast_current_action"
            if fast_current_action_replay_allowed
            else (
              "streaming_shift_trusted_draft_verify_ready"
              if self.config.streaming_vlm_trust_shifted_draft
              else "streaming_shift_draft_ready"
            )
          ),
          "full_vlm_generated_sequences": source_entry.get("full_vlm_generated_sequences") if trusted_full_replay_allowed else None,
          "full_vlm_prompt_cache": source_entry.get("full_vlm_prompt_cache") if trusted_full_replay_allowed else None,
          "full_vlm_prompt_cache_owner": (
            "trusted_shifted_source_prompt_cache_no_reasoning"
            if no_reasoning_trusted_replay_allowed
            else "trusted_shifted_source_prompt_cache_fast_current_action"
            if fast_current_action_replay_allowed
            else ""
          ),
          "full_vlm_window_signature": source_entry.get("full_vlm_window_signature") if trusted_full_replay_allowed else None,
          "full_vlm_prompt_cache_context_exact": bool(source_cache_context_match),
          "full_vlm_max_generation_length": int(source_entry.get("full_vlm_max_generation_length", -1)) if trusted_full_replay_allowed else -1,
          "full_vlm_eos_token_id": int(source_entry.get("full_vlm_eos_token_id", -1)) if trusted_full_replay_allowed else -1,
          "full_vlm_input_seq_len": int(entry["input_seq_len"]) if trusted_full_replay_allowed else -1,
          "streaming_vlm_draft_generated_sequences": source_entry.get("full_vlm_generated_sequences"),
          "streaming_vlm_draft_max_generation_length": int(source_entry.get("full_vlm_max_generation_length", -1)),
          "streaming_vlm_draft_eos_token_id": int(source_entry.get("full_vlm_eos_token_id", -1)),
          "streaming_vlm_draft_input_seq_len": int(source_entry.get("full_vlm_input_seq_len", entry["input_seq_len"])),
          "streaming_vlm_draft_source_window_signature": source_entry.get("window_signature"),
          "streaming_vlm_draft_source_prefix_semantic_signature": source_entry.get("prefix_semantic_signature"),
          "streaming_vlm_draft_source_fused_input_ids_signature": source_entry.get("fused_input_ids_signature", ()),
          "streaming_vlm_draft_source_cache_position_ids_signature": source_entry.get("cache_position_ids_signature"),
          "streaming_vlm_draft_source_cache_rope_deltas_signature": source_entry.get("cache_rope_deltas_signature"),
          "streaming_vlm_draft_source_cache_context_match": source_cache_context_match,
          "streaming_vlm_draft_shifted_prompt_kv_reuse_plan": shifted_prompt_kv_reuse_plan,
          "streaming_vlm_shift_source_prompt_cache": source_entry.get("full_vlm_prompt_cache"),
          "streaming_vlm_shift_source_generated_sequences": source_entry.get("full_vlm_generated_sequences"),
          "streaming_vlm_draft_dflash_layer_ids": tuple(source_entry.get("dflash_layer_ids", ())),
          "streaming_vlm_draft_dflash_target_cache": copy.deepcopy(source_entry.get("dflash_target_cache")),
          "streaming_vlm_draft_dflash_prefill_logits": self._detach_clone_tensor_tree(
            source_entry.get("dflash_prefill_logits")
          ),
          "streaming_vlm_draft_dflash_target_hidden": self._detach_clone_tensor_tree(
            source_entry.get("dflash_target_hidden")
          ),
          "full_vlm_reason": (
            f"trusted_shifted_prompt_cache_replay_no_reasoning:{overlap_frames}/{total_frames}"
            if no_reasoning_trusted_replay_allowed
            else f"trusted_shifted_prompt_cache_replay_fast_current_action:{overlap_frames}/{total_frames}"
            if fast_current_action_replay_allowed
            else f"streaming_shift_draft_pending_verify:{overlap_frames}/{total_frames}"
          ),
          "full_vlm_hits": 0,
          "full_vlm_stores": 1 if trusted_full_replay_allowed else 0,
          "streaming_vlm_reuse": True,
          "streaming_vlm_reuse_mode": (
            "trusted_full_replay_no_reasoning"
            if no_reasoning_trusted_replay_allowed
            else "trusted_full_replay_fast_current_action"
            if fast_current_action_replay_allowed
            else "draft_verify"
          ),
          "streaming_vlm_reuse_unverified": not source_cache_context_match,
          "streaming_vlm_trusted_replay_requested": trusted_replay_requested,
          "streaming_vlm_trusted_replay_allowed": no_reasoning_trusted_replay_allowed,
          "streaming_vlm_fast_current_action_replay_allowed": fast_current_action_replay_allowed,
          "streaming_vlm_trusted_replay_disabled_for_diffusion_freshness": trusted_replay_requested and not trusted_full_replay_allowed,
          "streaming_vlm_state_fresh_no_reasoning_required": bool(
            self.config.disable_reasoning_generation and self._require_state_fresh_no_reasoning()
          ),
          "streaming_vlm_trusted_replay_refresh_interval": int(refresh_interval),
          "streaming_vlm_refresh_due": bool(refresh_due and trusted_replay_requested),
          "streaming_vlm_reuse_source_key": repr(source_key)[:512],
          "streaming_vlm_reuse_overlap_ratio": float(overlap_ratio),
          "streaming_vlm_reuse_overlap_frames": int(overlap_frames),
          "streaming_vlm_reuse_total_frames": int(total_frames),
          "streaming_vlm_reuse_overlap_source": overlap_source,
          "streaming_vlm_reuse_chain_depth": source_chain_depth + 1,
        })
        source_entry["streaming_vlm_reuse_exports"] = int(source_entry.get("streaming_vlm_reuse_exports", 0) or 0) + 1
      self._vlm_prefix_cache[key] = entry
      self._vlm_prefix_cache.move_to_end(key)
      while len(self._vlm_prefix_cache) > self._vlm_prefix_cache_max_entries:
        self._vlm_prefix_cache.popitem(last=False)
      if self._model is not None:
        setattr(self._model, "_openpilot_vlm_prefix_cache_entry", entry)
        setattr(self._model, "_openpilot_vlm_prefix_cache_key", key)
      cache_stats["vlmPrefixCache"] = {
        "enabled": True,
        "hit": 1 if entry.get("streaming_vlm_reuse") else 0,
        "miss": 0 if entry.get("streaming_vlm_reuse") else 1,
        "depth": len(self._vlm_prefix_cache),
        "hasPromptCache": bool(entry.get("has_prompt_cache")),
        "reason": entry["reason"],
        "hits": 0,
        "stores": 0,
        "dflashHasSelectedHidden": False,
        "dflashReason": "",
        "dflashHits": 0,
        "dflashStores": 0,
        "dflashFullGenerationReady": False,
        "dflashFullGenerationReason": "",
        "dflashFullGenerationHits": 0,
        "dflashFullGenerationStores": 0,
        "fullGenerationReady": self._vlm_prefix_entry_full_generation_ready(
          entry,
          int(entry["input_seq_len"]),
        ),
        "fullGenerationReason": str(entry.get("full_vlm_reason", "")),
        "fullGenerationHits": 0,
        "fullGenerationStores": 0,
        "exactHit": False,
        "streamingReuseHit": bool(entry.get("streaming_vlm_reuse")),
        "streamingReuseOverlapRatio": float(entry.get("streaming_vlm_reuse_overlap_ratio", 0.0) or 0.0),
        "streamingReuseOverlapFrames": int(entry.get("streaming_vlm_reuse_overlap_frames", 0) or 0),
        "streamingReuseTotalFrames": int(entry.get("streaming_vlm_reuse_total_frames", 0) or 0),
        "streamingReuseOverlapSource": str(entry.get("streaming_vlm_reuse_overlap_source", "")),
        "streamingReuseChainDepth": int(entry.get("streaming_vlm_reuse_chain_depth", 0) or 0),
        "streamingReuseMode": str(entry.get("streaming_vlm_reuse_mode", "")),
        "streamingReuseUnverified": bool(entry.get("streaming_vlm_reuse_unverified")),
          "trustedReplayAllowed": bool(entry.get("streaming_vlm_trusted_replay_allowed")),
          "stateFreshNoReasoningRequired": bool(entry.get("streaming_vlm_state_fresh_no_reasoning_required")),
          "stateFreshNoReasoningForcedPrefill": bool(entry.get("full_vlm_state_fresh_no_reasoning_forced_prefill")),
          "refreshDue": bool(entry.get("streaming_vlm_refresh_due")),
          "languageVisualTokenSpans": len(entry.get("language_visual_token_spans", []) or []),
        "shiftedPromptKvReusePlan": entry.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan", {}),
      }

  def _verify_visual_feature_order(
    self,
    *,
    kind: str,
    token_blocks: list[dict[str, Any]],
    grid_rows: list[list[int]],
    visual_order: list[dict[str, Any]] | None,
    cache_stats: dict[str, Any],
  ) -> bool:
    stat_prefix = f"streaming_{kind}_feature_cache"
    if visual_order is None:
      cache_stats[f"{stat_prefix}_missing_order"] = cache_stats.get(f"{stat_prefix}_missing_order", 0) + 1
      return False
    if len(visual_order) != len(token_blocks) or len(visual_order) != len(grid_rows):
      cache_stats[f"{stat_prefix}_order_mismatch"] = cache_stats.get(f"{stat_prefix}_order_mismatch", 0) + 1
      cache_stats[f"{stat_prefix}_order_mismatch_detail"] = (
        f"order={len(visual_order)} blocks={len(token_blocks)} grids={len(grid_rows)}"
      )
      return False

    visual = getattr(getattr(getattr(self._model, "vlm", None), "model", None), "visual", None)
    visual_config = getattr(visual, "config", None)
    merge_size = int(getattr(visual, "spatial_merge_size", 0) or getattr(visual_config, "spatial_merge_size", 0) or 1)
    merge_area = max(1, merge_size * merge_size)
    for idx, (block, order_item, grid_row) in enumerate(zip(token_blocks, visual_order, grid_rows)):
      same_identity = (
        str(block.get("stream", "")) == str(order_item.get("stream", ""))
        and int(block.get("stream_index", -1)) == int(order_item.get("stream_index", -2))
        and int(block.get("frame_index", -1)) == int(order_item.get("frame_index", -2))
        and tuple(block.get("signature", ())) == tuple(order_item.get("signature", ()))
      )
      if not same_identity:
        cache_stats[f"{stat_prefix}_order_mismatch"] = cache_stats.get(f"{stat_prefix}_order_mismatch", 0) + 1
        cache_stats[f"{stat_prefix}_order_mismatch_detail"] = (
          f"idx={idx} block={block.get('stream')}:{block.get('frame_index')} order="
          f"{order_item.get('stream')}:{order_item.get('frame_index')}"
        )
        return False
      if len(grid_row) == 3:
        grid_token_count = max(0, int(grid_row[0]) * int(grid_row[1]) * int(grid_row[2]) // merge_area)
        if grid_token_count != int(block.get("token_count", 0)):
          cache_stats[f"{stat_prefix}_grid_token_mismatch"] = cache_stats.get(
            f"{stat_prefix}_grid_token_mismatch", 0
          ) + 1
          cache_stats[f"{stat_prefix}_grid_token_mismatch_detail"] = (
            f"idx={idx} grid={grid_token_count} block={int(block.get('token_count', 0))}"
          )
          return False
    cache_stats[f"{stat_prefix}_order_verified"] = cache_stats.get(f"{stat_prefix}_order_verified", 0) + 1
    return True

  def _visual_features_with_cuda_graph(
    self,
    *,
    kind: str,
    getter: Any,
    vlm_model: Any,
    pixel_values: Any,
    grid: Any,
    cache_stats: dict[str, Any],
  ) -> tuple[Any, Any] | None:
    torch_mod = self._torch
    if (
      torch_mod is None
      or not self.config.cuda_graphs
      or not self.config.graph_visual_stage
      or not getattr(pixel_values, "is_cuda", False)
      or not getattr(grid, "is_cuda", False)
      or not hasattr(torch_mod, "cuda")
      or not torch_mod.cuda.is_available()
    ):
      return None

    graph_cache = self._cuda_graph_stage_caches.get("visual")
    if graph_cache is None or not hasattr(graph_cache, "get"):
      cache_stats[f"streaming_{kind}_feature_graph_error"] = "missing_visual_stage_cache"
      return None

    cache_key = (
      "streaming_visual_features",
      kind,
      id(vlm_model),
      tuple(pixel_values.shape),
      str(pixel_values.dtype),
      str(pixel_values.device),
      tuple(grid.shape),
      str(grid.dtype),
      str(grid.device),
    )
    max_entries = max(1, int(self._cuda_graph_cache_size))
    entry = graph_cache.get(cache_key)
    cache_hit = entry is not None

    def call_getter(pixels: Any, grid_value: Any) -> tuple[Any, Any]:
      if getattr(getter, "__self__", None) is None:
        return getter(vlm_model, pixels, grid_value)
      return getter(pixels, grid_value)

    try:
      if entry is None:
        static_pixels = torch_mod.empty_like(pixel_values)
        static_grid = torch_mod.empty_like(grid)
        static_pixels.copy_(pixel_values, non_blocking=True)
        static_grid.copy_(grid, non_blocking=True)

        current_stream = torch_mod.cuda.current_stream(pixel_values.device)
        warmup_stream = torch_mod.cuda.Stream(device=pixel_values.device)
        warmup_stream.wait_stream(current_stream)
        with torch_mod.cuda.stream(warmup_stream):
          call_getter(static_pixels, static_grid)
        current_stream.wait_stream(warmup_stream)

        graph = torch_mod.cuda.CUDAGraph()
        with torch_mod.cuda.graph(graph):
          static_output = call_getter(static_pixels, static_grid)
        if not isinstance(static_output, (list, tuple)) or len(static_output) != 2:
          cache_stats[f"streaming_{kind}_feature_graph_error"] = "bad_visual_feature_return"
          return None
        entry = {
          "graph": graph,
          "pixel_values": static_pixels,
          "grid": static_grid,
          "output": static_output,
        }
        graph_cache[cache_key] = entry
        graph_cache.move_to_end(cache_key)
        while len(graph_cache) > max_entries:
          graph_cache.popitem(last=False)
      else:
        graph_cache.move_to_end(cache_key)

      entry["pixel_values"].copy_(pixel_values, non_blocking=True)
      entry["grid"].copy_(grid, non_blocking=True)
      entry["graph"].replay()
      cache_stats[f"streaming_{kind}_feature_graph_mode"] = "replay" if cache_hit else "capture"
      cache_stats[f"streaming_{kind}_feature_graph_cache_hit"] = 1 if cache_hit else 0
      cache_stats[f"streaming_{kind}_feature_graph_cache_depth"] = len(graph_cache)
      cache_stats[f"streaming_{kind}_feature_graph_ready"] = 1
      return entry["output"]
    except Exception as exc:
      try:
        graph_cache.pop(cache_key, None)
      except Exception:
        pass
      cache_stats[f"streaming_{kind}_feature_graph_error"] = f"{type(exc).__name__}: {exc}"
      cache_stats[f"streaming_{kind}_feature_graph_ready"] = 0
      return None

  def _cache_streaming_visual_features(
    self,
    tokenized_data: dict[str, Any],
    cache_stats: dict[str, Any],
    *,
    kind: str,
    token_blocks: list[dict[str, Any]] | None = None,
    visual_order: list[dict[str, Any]] | None = None,
  ) -> None:
    if not self.config.streaming_vision_cache or self._torch is None or self._model is None:
      return
    if kind == "image":
      pixel_key = "pixel_values"
      grid_key = "image_grid_thw"
      precomputed_key = "_openpilot_precomputed_image_features"
      getter_name = "get_image_features"
    elif kind == "video":
      pixel_key = "pixel_values_videos"
      grid_key = "video_grid_thw"
      precomputed_key = "_openpilot_precomputed_video_features"
      getter_name = "get_video_features"
    else:
      return
    if tokenized_data.get(precomputed_key) is not None:
      return

    pixel_values = tokenized_data.get(pixel_key)
    grid = tokenized_data.get(grid_key)
    if pixel_values is None or grid is None:
      return
    grid_rows = StreamingAlpamayoVisionCache._grid_rows(grid)
    if token_blocks is None:
      token_blocks = self._streaming_vision_cache.last_token_blocks()
    else:
      token_blocks = list(token_blocks)
    if not grid_rows or len(grid_rows) != len(token_blocks):
      cache_stats[f"streaming_{kind}_feature_cache_skipped_grid_mismatch"] = cache_stats.get(
        f"streaming_{kind}_feature_cache_skipped_grid_mismatch", 0
      ) + 1
      return
    expected_token_counts = [int(block.get("token_count", 0)) for block in token_blocks]
    if not self._verify_visual_feature_order(
      kind=kind,
      token_blocks=token_blocks,
      grid_rows=grid_rows,
      visual_order=visual_order,
      cache_stats=cache_stats,
    ):
      return
    pixel_slices = self._visual_pixel_slices_for_grid(pixel_values, grid_rows)
    if pixel_slices is None or len(pixel_slices) != len(grid_rows):
      cache_stats[f"streaming_{kind}_feature_cache_skipped_pixel_shape"] = cache_stats.get(
        f"streaming_{kind}_feature_cache_skipped_pixel_shape", 0
      ) + 1
      return

    ordered_cache_keys = [
      self._visual_feature_cache_key(kind, block, grid_row)
      for block, grid_row in zip(token_blocks, grid_rows)
    ]
    ordered_features: list[tuple[Any, Any] | None] = []
    missing_indices: list[int] = []
    hits = 0
    with self._cache_lock:
      for idx, cache_key in enumerate(ordered_cache_keys):
        cached = self._visual_feature_cache.get(cache_key)
        if cached is None:
          ordered_features.append(None)
          missing_indices.append(idx)
        else:
          cached_token_count = self._visual_feature_token_count(cached[0])
          if cached_token_count != expected_token_counts[idx]:
            self._visual_feature_cache.pop(cache_key, None)
            ordered_features.append(None)
            missing_indices.append(idx)
            cache_stats[f"streaming_{kind}_feature_cache_cached_length_mismatch"] = cache_stats.get(
              f"streaming_{kind}_feature_cache_cached_length_mismatch", 0
            ) + 1
            continue
          self._visual_feature_cache.move_to_end(cache_key)
          ordered_features.append(cached)
          hits += 1

    if missing_indices:
      torch_mod = self._torch
      vlm_model = self._model.vlm.model
      getter = getattr(type(vlm_model), f"_openpilot_original_{getter_name}", None)
      if getter is None:
        getter = getattr(vlm_model, getter_name, None)
      if getter is None:
        return
      missing_pixels = torch_mod.cat([pixel_slices[idx] for idx in missing_indices], dim=0)
      if hasattr(grid, "index_select"):
        index_tensor = torch_mod.tensor(missing_indices, device=grid.device, dtype=torch_mod.long)
        missing_grid = grid.index_select(0, index_tensor)
      else:
        tensor_kwargs: dict[str, Any] = {}
        if hasattr(pixel_values, "device"):
          tensor_kwargs["device"] = pixel_values.device
        missing_grid = torch_mod.tensor([grid_rows[idx] for idx in missing_indices], dtype=torch_mod.long, **tensor_kwargs)
      missing_token_blocks: list[dict[str, Any]] = []
      missing_token_start = 0
      for source_idx in missing_indices:
        source_block = dict(token_blocks[source_idx])
        grid_row = grid_rows[source_idx]
        if len(grid_row) == 3:
          token_count = max(0, int(grid_row[0]) * int(grid_row[1]) * int(grid_row[2]))
        else:
          token_count = int(source_block.get("token_count", 0))
        source_block["token_start"] = missing_token_start
        source_block["token_count"] = token_count
        source_block["prompt_index"] = len(missing_token_blocks)
        missing_token_blocks.append(source_block)
        missing_token_start += token_count
      visual_runtime_profile: dict[str, Any] = {}
      previous_streaming_context = getattr(_STREAMING_VISION_PATCH_CONTEXT, "context", None)
      if self.config.streaming_vision_attention_mask:
        missing_attention_mask = StreamingAlpamayoVisionCache._build_view_major_attention_mask(torch_mod, missing_token_blocks)
      else:
        missing_attention_mask = None
        cache_stats[f"streaming_{kind}_vision_attention_mask_unverified_shortcut"] = 1
      with self._streaming_vision_cache._lock:
        previous_attention_mask = self._streaming_vision_cache._last_attention_mask
        previous_token_blocks = list(self._streaming_vision_cache._last_token_blocks)
        self._streaming_vision_cache._last_attention_mask = missing_attention_mask
        self._streaming_vision_cache._last_token_blocks = missing_token_blocks
      _STREAMING_VISION_PATCH_CONTEXT.context = {
        "cache": self._streaming_vision_cache,
        "runtime_profile": visual_runtime_profile,
        "token_blocks": missing_token_blocks,
      }
      try:
        missing_features = self._visual_features_with_cuda_graph(
          kind=kind,
          getter=getter,
          vlm_model=vlm_model,
          pixel_values=missing_pixels,
          grid=missing_grid,
          cache_stats=cache_stats,
        )
        if missing_features is None:
          try:
            with torch_mod.no_grad():
              if getattr(getter, "__self__", None) is None:
                missing_features = getter(vlm_model, missing_pixels, missing_grid)
              else:
                missing_features = getter(missing_pixels, missing_grid)
          except Exception as exc:
            cache_stats[f"streaming_{kind}_feature_cache_compute_error"] = f"{type(exc).__name__}: {exc}"
            return
      finally:
        _STREAMING_VISION_PATCH_CONTEXT.context = previous_streaming_context
        with self._streaming_vision_cache._lock:
          self._streaming_vision_cache._last_attention_mask = previous_attention_mask
          self._streaming_vision_cache._last_token_blocks = previous_token_blocks
        if visual_runtime_profile:
          cache_stats[f"streaming_{kind}_vision_runtime"] = dict(visual_runtime_profile)
      if not isinstance(missing_features, (list, tuple)) or len(missing_features) != 2:
        cache_stats[f"streaming_{kind}_feature_cache_bad_return"] = 1
        return
      missing_embeds, missing_deepstack = missing_features
      if not isinstance(missing_embeds, (list, tuple)) or len(missing_embeds) != len(missing_indices):
        cache_stats[f"streaming_{kind}_feature_cache_bad_split"] = 1
        return
      try:
        selected_deepstack_items = self._split_deepstack_items(
          missing_deepstack,
          [expected_token_counts[source_idx] for source_idx in missing_indices],
        )
      except ValueError as exc:
        cache_stats[f"streaming_{kind}_feature_cache_unsplittable_deepstack"] = str(exc)
        return
      if len(selected_deepstack_items) != len(missing_indices):
        cache_stats[f"streaming_{kind}_feature_cache_bad_deepstack_split"] = 1
        return
      for local_idx, source_idx in enumerate(missing_indices):
        actual_token_count = self._visual_feature_token_count(missing_embeds[local_idx])
        if actual_token_count != expected_token_counts[source_idx]:
          cache_stats[f"streaming_{kind}_feature_cache_bad_split_length"] = cache_stats.get(
            f"streaming_{kind}_feature_cache_bad_split_length", 0
          ) + 1
          return
      with self._cache_lock:
        for local_idx, source_idx in enumerate(missing_indices):
          feature_item = (
            self._detach_clone_tensor_tree(missing_embeds[local_idx]),
            self._detach_clone_tensor_tree(selected_deepstack_items[local_idx]),
          )
          ordered_features[source_idx] = feature_item
          self._visual_feature_cache[ordered_cache_keys[source_idx]] = feature_item
          self._visual_feature_cache.move_to_end(ordered_cache_keys[source_idx])
        while len(self._visual_feature_cache) > self._visual_feature_cache_max_entries:
          self._visual_feature_cache.popitem(last=False)

    if any(item is None for item in ordered_features):
      return
    for idx, item in enumerate(ordered_features):
      if item is None:
        return
      if self._visual_feature_token_count(item[0]) != expected_token_counts[idx]:
        cache_stats[f"streaming_{kind}_feature_cache_assembly_length_mismatch"] = cache_stats.get(
          f"streaming_{kind}_feature_cache_assembly_length_mismatch", 0
        ) + 1
        return
    feature_pairs = [item for item in ordered_features if item is not None]
    tokenized_data[precomputed_key] = (
      tuple(item[0] for item in feature_pairs),
      self._assemble_deepstack_items([item[1] for item in feature_pairs]),
    )
    cache_stats[f"streaming_{kind}_feature_cache_hits"] = cache_stats.get(
      f"streaming_{kind}_feature_cache_hits", 0
    ) + hits
    cache_stats[f"streaming_{kind}_feature_cache_misses"] = cache_stats.get(
      f"streaming_{kind}_feature_cache_misses", 0
    ) + len(missing_indices)
    cache_stats[f"streaming_{kind}_feature_cache_depth"] = len(self._visual_feature_cache)

  def _module_device(self, module: Any) -> Any:
    assert self._torch is not None and self._model is not None
    for parameter in module.parameters(recurse=True):
      return parameter.device
    for buffer in module.buffers(recurse=True):
      return buffer.device
    return self._model.device

  def _dflash_device(self) -> Any:
    assert self._torch is not None and self._model is not None
    requested = self.config.dflash_draft_device.strip()
    if requested == "lm_head":
      return self._module_device(self._model.vlm.lm_head)
    if requested == "model":
      return self._model.device
    return self._torch.device(requested)

  def _disable_dflash_runtime(self, reason: str, *, sticky: bool = True) -> None:
    self._dflash_model = None
    self._dflash_mask_embedding = None
    self._dflash_layer_ids = []
    self._dflash_loaded = False
    self._dflash_load_error = reason
    if sticky:
      self._dflash_sticky_disabled = True
      self._dflash_disable_cooldown_remaining = max(0, int(self.config.dflash_retry_cooldown_frames))
    if self._dflash_manual_generate_installed and self._model is not None:
      try:
        if self._dflash_original_manual_generate is None:
          if hasattr(self._model, "_manual_greedy_vlm_generate"):
            delattr(self._model, "_manual_greedy_vlm_generate")
        else:
          self._model._manual_greedy_vlm_generate = self._dflash_original_manual_generate
      except Exception:
        pass
    self._dflash_original_manual_generate = None
    self._dflash_manual_generate_installed = False

  def _dflash_runtime_enabled(self, runtime_profile: dict[str, Any] | None = None) -> bool:
    enabled = bool(
      self.config.dflash_enabled
      and self._dflash_loaded
      and self._dflash_model is not None
      and self._dflash_mask_embedding is not None
      and self._dflash_layer_ids
      and self._dflash_manual_generate_installed
    )
    if runtime_profile is not None:
      runtime_profile["dflash_runtime_enabled"] = 1 if enabled else 0
      runtime_profile["dflash_sticky_disabled"] = 1 if self._dflash_sticky_disabled else 0
      runtime_profile["dflash_disable_cooldown_remaining"] = int(self._dflash_disable_cooldown_remaining)
      if self.config.dflash_enabled and not enabled and self._dflash_load_error:
        runtime_profile["dflash_disabled_reason"] = self._dflash_load_error
    return enabled

  def _dflash_gate_failure(self, result: Any, runtime_profile: dict[str, Any] | None = None) -> str | None:
    acceptance_lengths = [int(item) for item in (getattr(result, "acceptance_lengths", []) or [])]
    accepted_tokens = int(sum(acceptance_lengths))
    block_size = max(1, int(getattr(self._dflash_model, "block_size", 1) or 1))
    acceptance_capacity = max(1, len(acceptance_lengths) * block_size)
    acceptance_rate = float(accepted_tokens) / float(acceptance_capacity)
    time_to_first_ms = float(getattr(result, "time_to_first_token_ms", 0.0) or 0.0)
    decode_ms = float(getattr(result, "decode_ms", 0.0) or 0.0)
    total_ms = time_to_first_ms + decode_ms

    if runtime_profile is not None:
      runtime_profile["dflash_acceptance_rate"] = acceptance_rate
      runtime_profile["dflash_accepted_tokens"] = accepted_tokens
      runtime_profile["dflash_acceptance_capacity"] = acceptance_capacity
      runtime_profile["dflash_time_to_first_token_ms"] = time_to_first_ms
      runtime_profile["dflash_decode_ms"] = decode_ms
      runtime_profile["dflash_total_ms"] = total_ms

    if self.config.dflash_min_acceptance_rate > 0.0 and acceptance_rate < self.config.dflash_min_acceptance_rate:
      return f"dflash_acceptance_rate_below_threshold:{acceptance_rate:.6f}<{self.config.dflash_min_acceptance_rate:.6f}"
    if self.config.dflash_max_time_to_first_token_ms > 0.0 and time_to_first_ms > self.config.dflash_max_time_to_first_token_ms:
      return f"dflash_time_to_first_token_ms_above_threshold:{time_to_first_ms:.3f}>{self.config.dflash_max_time_to_first_token_ms:.3f}"
    if self.config.dflash_max_decode_ms > 0.0 and decode_ms > self.config.dflash_max_decode_ms:
      return f"dflash_decode_ms_above_threshold:{decode_ms:.3f}>{self.config.dflash_max_decode_ms:.3f}"
    if self.config.dflash_max_total_ms > 0.0 and total_ms > self.config.dflash_max_total_ms:
      return f"dflash_total_ms_above_threshold:{total_ms:.3f}>{self.config.dflash_max_total_ms:.3f}"
    return None

  @staticmethod
  def _vlm_prefill_graph_tensor_signature(kwargs: dict[str, Any]) -> tuple[Any, ...]:
    items: list[tuple[Any, ...]] = []
    for key, value in sorted(kwargs.items(), key=lambda item: str(item[0])):
      if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        items.append((
          str(key),
          "tensor",
          tuple(int(dim) for dim in value.shape),
          str(value.dtype),
          str(value.device),
        ))
      elif isinstance(value, (bool, int, float, str)) or value is None:
        items.append((str(key), type(value).__name__, value))
      else:
        items.append((str(key), type(value).__name__))
    return tuple(items)

  def _vlm_prefill_with_cuda_graph(
    self,
    *,
    input_ids: Any,
    tokenized_data: dict[str, Any],
    runtime_profile: dict[str, Any] | None = None,
    cache_position_seed: Any | None = None,
  ) -> Any | None:
    del cache_position_seed
    torch_mod = self._torch
    model = self._model

    def fallback(reason: str) -> None:
      if runtime_profile is not None:
        runtime_profile["vlm_prefill_graph_mode"] = "fallback"
        runtime_profile["vlm_prefill_graph_error"] = reason
        runtime_profile["vlm_prefill_graph_ready"] = 0

    if torch_mod is None or model is None:
      fallback("adapter_not_loaded")
      return None
    if not self.config.cuda_graphs:
      fallback("cuda_graphs_disabled")
      return None
    if not self._supports_cuda_graph:
      fallback("unsupported_cuda_graph_configuration")
      return None
    if not hasattr(torch_mod, "cuda") or not torch_mod.cuda.is_available() or not hasattr(torch_mod.cuda, "graph"):
      fallback("torch_cuda_graph_unavailable")
      return None
    if not getattr(input_ids, "is_cuda", False):
      fallback("non_cuda_input_ids")
      return None
    graph_cache = self._cuda_graph_stage_caches.get("prefill")
    if graph_cache is None or not hasattr(graph_cache, "get"):
      fallback("missing_prefill_cache")
      return None

    graph_kwargs = copy.copy(tokenized_data)
    if graph_kwargs.get("past_key_values") is not None:
      fallback("prefill_graph_rejects_external_past_key_values")
      return None
    graph_kwargs.setdefault("use_cache", True)
    tensor_signature = self._vlm_prefill_graph_tensor_signature(graph_kwargs)
    cache_key = (
      "vlm_prefill",
      id(model.vlm),
      tuple(int(dim) for dim in input_ids.shape),
      str(input_ids.dtype),
      str(input_ids.device),
      tensor_signature,
    )
    max_entries = max(1, int(self._cuda_graph_cache_size))
    entry = graph_cache.get(cache_key)
    cache_hit = entry is not None

    try:
      if entry is None:
        static_input_ids = torch_mod.empty_like(input_ids)
        static_kwargs: dict[str, Any] = {}
        for key, value in graph_kwargs.items():
          if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
            static_kwargs[key] = torch_mod.empty_like(value)
          else:
            static_kwargs[key] = value
        static_input_ids.copy_(input_ids, non_blocking=True)
        for key, value in graph_kwargs.items():
          if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
            static_kwargs[key].copy_(value, non_blocking=True)

        current_stream = torch_mod.cuda.current_stream(input_ids.device)
        warmup_stream = torch_mod.cuda.Stream(device=input_ids.device)
        warmup_stream.wait_stream(current_stream)
        with torch_mod.cuda.stream(warmup_stream):
          with torch_mod.no_grad():
            model.vlm(
              input_ids=static_input_ids,
              return_dict=True,
              **static_kwargs,
            )
        current_stream.wait_stream(warmup_stream)

        graph = torch_mod.cuda.CUDAGraph()
        with torch_mod.cuda.graph(graph):
          static_output = model.vlm(
            input_ids=static_input_ids,
            return_dict=True,
            **static_kwargs,
          )
        entry = {
          "graph": graph,
          "input_ids": static_input_ids,
          "tokenized_data": static_kwargs,
          "output": static_output,
        }
        graph_cache[cache_key] = entry
        graph_cache.move_to_end(cache_key)
        while len(graph_cache) > max_entries:
          graph_cache.popitem(last=False)
      else:
        graph_cache.move_to_end(cache_key)

      entry["input_ids"].copy_(input_ids, non_blocking=True)
      for key, value in graph_kwargs.items():
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
          entry["tokenized_data"][key].copy_(value, non_blocking=True)
      entry["graph"].replay()

      output = copy.copy(entry["output"])
      static_prompt_cache = getattr(entry["output"], "past_key_values", None)
      live_cache_mode = "missing_prompt_cache"
      if static_prompt_cache is not None:
        try:
          from openpilot.selfdrive.alpamayo.dflash_adapter import _pooled_live_cache_copy

          live_prompt_cache, live_cache_mode = _pooled_live_cache_copy(
            entry,
            static_prompt_cache,
            pool_key="vlm_prefill_live_prompt_cache_pool",
          )
        except Exception as exc:
          live_prompt_cache = copy.deepcopy(static_prompt_cache)
          live_cache_mode = f"deepcopy_after_pool_error:{type(exc).__name__}"
        try:
          output.past_key_values = live_prompt_cache
        except Exception:
          pass
      if runtime_profile is not None:
        runtime_profile["vlm_prefill_graph_mode"] = "replay" if cache_hit else "capture"
        runtime_profile["vlm_prefill_graph_ready"] = 1
        runtime_profile["vlm_prefill_graph_cache_hit"] = 1 if cache_hit else 0
        runtime_profile["vlm_prefill_graph_cache_depth"] = int(len(graph_cache))
        runtime_profile["vlm_prefill_graph_calls"] = int(runtime_profile.get("vlm_prefill_graph_calls", 0)) + 1
        runtime_profile["vlm_prefill_graph_live_cache_mode"] = live_cache_mode
      return output
    except Exception as exc:
      try:
        graph_cache.pop(cache_key, None)
      except Exception:
        pass
      fallback(f"{type(exc).__name__}: {exc}")
      return None

  @staticmethod
  def _static_text_prefix_len(prefix_cache_entry: dict[str, Any] | None, input_ids: Any) -> int:
    if not isinstance(prefix_cache_entry, dict) or input_ids is None or not hasattr(input_ids, "shape"):
      return 0
    spans = prefix_cache_entry.get("language_visual_token_spans")
    if not isinstance(spans, list) or not spans:
      return 0
    starts = [
      int(span.get("language_token_start", -1))
      for span in spans
      if isinstance(span, dict) and int(span.get("language_token_start", -1)) >= 0
    ]
    if not starts:
      return 0
    prefix_len = min(starts)
    input_seq_len = int(input_ids.shape[1]) if len(input_ids.shape) >= 2 else 0
    if prefix_len < 16 or prefix_len >= input_seq_len:
      return 0
    return int(prefix_len)

  def _vlm_static_text_prefix_prompt_cache(
    self,
    model_self: Any,
    input_ids: Any,
    verify_kwargs: dict[str, Any],
    prefix_len: int,
    runtime_profile: dict[str, Any] | None,
  ) -> tuple[Any | None, str]:
    torch_mod = self._torch
    if torch_mod is None or prefix_len <= 0:
      return None, "disabled"
    try:
      prefix_ids = input_ids[:, :prefix_len]
      prefix_ids_cpu = prefix_ids.detach().to("cpu", non_blocking=False).contiguous()
      prefix_crc = zlib.crc32(prefix_ids_cpu.numpy().tobytes()) & 0xFFFFFFFF
      cache_key = (
        "vlm_static_text_prefix_v1",
        str(self.config.target_model),
        int(prefix_len),
        tuple(int(dim) for dim in prefix_ids.shape),
        str(prefix_ids.dtype),
        int(prefix_crc),
      )
      with self._cache_lock:
        entry = self._vlm_text_prefix_cache.get(cache_key)
        if entry is not None:
          self._vlm_text_prefix_cache.move_to_end(cache_key)
      if entry is None:
        prefix_kwargs = copy.copy(verify_kwargs)
        for key in (
          "pixel_values",
          "pixel_values_videos",
          "image_grid_thw",
          "video_grid_thw",
          "_openpilot_precomputed_image_features",
          "_openpilot_precomputed_video_features",
          "_openpilot_precomputed_patch_pos_embeds",
        ):
          prefix_kwargs.pop(key, None)
        attention_mask = prefix_kwargs.get("attention_mask")
        if attention_mask is not None and hasattr(attention_mask, "shape") and int(attention_mask.shape[-1]) >= prefix_len:
          prefix_kwargs["attention_mask"] = attention_mask[..., :prefix_len]
        prefix_kwargs["cache_position"] = torch_mod.arange(prefix_len, device=input_ids.device)
        prefix_kwargs["use_cache"] = True
        with torch_mod.no_grad():
          prefix_outputs = model_self.vlm(
            input_ids=prefix_ids,
            logits_to_keep=1,
            return_dict=True,
            **prefix_kwargs,
          )
        prompt_cache = getattr(prefix_outputs, "past_key_values", None)
        if prompt_cache is None:
          return None, "missing_prompt_cache"
        entry = {
          "prompt_cache": copy.deepcopy(prompt_cache),
          "stores": 1,
          "hits": 0,
        }
        with self._cache_lock:
          self._vlm_text_prefix_cache[cache_key] = entry
          self._vlm_text_prefix_cache.move_to_end(cache_key)
          while len(self._vlm_text_prefix_cache) > self._vlm_text_prefix_cache_max_entries:
            self._vlm_text_prefix_cache.popitem(last=False)
        mode = "store"
      else:
        entry["hits"] = int(entry.get("hits", 0)) + 1
        mode = "hit"
      try:
        from openpilot.selfdrive.alpamayo.dflash_adapter import _pooled_live_cache_copy

        live_cache, live_mode = _pooled_live_cache_copy(
          entry,
          entry["prompt_cache"],
          pool_key="vlm_static_text_prefix_live_cache_pool",
        )
        mode = f"{mode}:{live_mode}"
      except Exception as exc:
        live_cache = copy.deepcopy(entry["prompt_cache"])
        mode = f"{mode}:deepcopy_after_pool_error:{type(exc).__name__}"
      if runtime_profile is not None:
        runtime_profile["vlm_static_text_prefix_cache_len"] = int(prefix_len)
        runtime_profile["vlm_static_text_prefix_cache_mode"] = mode
        runtime_profile["vlm_static_text_prefix_cache_depth"] = int(len(self._vlm_text_prefix_cache))
      return live_cache, mode
    except Exception as exc:
      if runtime_profile is not None:
        runtime_profile["vlm_static_text_prefix_cache_error"] = f"{type(exc).__name__}: {exc}"
      return None, f"error:{type(exc).__name__}"

  def _ensure_dflash_loaded(self) -> None:
    if not self.config.dflash_enabled or self._dflash_loaded:
      return
    if self._dflash_sticky_disabled:
      if self._dflash_disable_cooldown_remaining <= 0:
        return
      self._dflash_disable_cooldown_remaining -= 1
      if self._dflash_disable_cooldown_remaining <= 0:
        self._dflash_sticky_disabled = False
      return
    assert self._torch is not None and self._model is not None
    if not self.config.dflash_package_root.exists():
      self._disable_dflash_runtime(f"FileNotFoundError: DFlash package root not found: {self.config.dflash_package_root}")
      return
    if not self.config.dflash_draft_model.exists():
      self._disable_dflash_runtime(f"FileNotFoundError: DFlash draft model not found: {self.config.dflash_draft_model}")
      return

    package_root = str(self.config.dflash_package_root)
    if package_root not in sys.path:
      sys.path.insert(0, package_root)

    try:
      from openpilot.selfdrive.alpamayo.dflash_adapter import (
        _mask_traj_token_logits,
        dflash_generate_alpamayo,
        load_dflash_draft_model,
      )

      dtype = _torch_dtype(self._torch, self.config.model_dtype)
      draft, mask_embedding, layer_ids = load_dflash_draft_model(
        self._torch,
        self.config.dflash_draft_model,
        self.config.dflash_package_root,
        self._dflash_device(),
        dtype,
        attn_implementation=self.config.dflash_attn_implementation,
      )

      self._dflash_model = draft
      self._dflash_mask_embedding = mask_embedding
      self._dflash_layer_ids = layer_ids

      adapter = self

      def dflash_vlm_generate(
        model_self: Any,
        input_ids: Any,
        tokenized_data: dict[str, Any],
        eos_token_id: int,
        max_generation_length: int,
        runtime_profile: dict[str, float | int] | None = None,
      ) -> tuple[Any, Any]:
        if adapter.config.disable_reasoning_generation or int(max_generation_length) <= 0:
          base_generate = adapter._dflash_original_manual_generate
          if base_generate is None:
            raise RuntimeError("base manual VLM prefill path unavailable for no-reasoning Alpamayo")
          if runtime_profile is not None:
            runtime_profile["dflash_no_reasoning_bypass"] = 1
            runtime_profile["vlm_autoregressive_generation_skipped"] = 1
          return base_generate(
            input_ids=input_ids,
            tokenized_data=tokenized_data,
            eos_token_id=eos_token_id,
            max_generation_length=0,
            runtime_profile=runtime_profile,
          )
        if adapter._dflash_model is None or adapter._dflash_mask_embedding is None:
          raise RuntimeError("DFlash generation requested before draft model was loaded")
        prefix_cache_entry = getattr(model_self, "_openpilot_vlm_prefix_cache_entry", None)
        trusted_fast_current_action_replay = bool(
          isinstance(prefix_cache_entry, dict)
          and getattr(adapter, "_openpilot_require_fast_no_prefill", False)
          and prefix_cache_entry.get("streaming_vlm_fast_current_action_replay_allowed")
        )
        full_generation_usable = bool(
          isinstance(prefix_cache_entry, dict)
          and prefix_cache_entry.get("full_vlm_generated_sequences") is not None
          and prefix_cache_entry.get("full_vlm_prompt_cache") is not None
          and int(prefix_cache_entry.get("full_vlm_max_generation_length", -1)) == int(max_generation_length)
          and int(prefix_cache_entry.get("full_vlm_eos_token_id", -1)) == int(eos_token_id)
          and (
            int(prefix_cache_entry.get("full_vlm_input_seq_len", -1)) == int(input_ids.shape[1])
            or trusted_fast_current_action_replay
          )
        )
        if full_generation_usable and isinstance(prefix_cache_entry, dict):
          exact_window_full_hit = bool(
            prefix_cache_entry.get("current_window_full_hit")
            or prefix_cache_entry.get("exact_window_full_hit")
            or prefix_cache_entry.get("window_full_hit")
          )
          streaming_reuse_mode = str(prefix_cache_entry.get("streaming_vlm_reuse_mode", ""))
          streaming_reuse_unverified = bool(prefix_cache_entry.get("streaming_vlm_reuse_unverified")) or streaming_reuse_mode.endswith("_unverified")
          trusted_replay_requested = bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_requested"))
          (
            exact_current_window_generation,
            generation_window_signature_match,
            prompt_cache_context_exact,
          ) = _full_generation_exact_current_window(prefix_cache_entry)
          if runtime_profile is not None:
            runtime_profile["vlm_full_generation_cache_window_signature_match"] = (
              1 if generation_window_signature_match else 0
            )
            runtime_profile["vlm_full_generation_prompt_cache_context_exact"] = (
              1 if prompt_cache_context_exact else 0
            )
          if (
            not trusted_fast_current_action_replay
            and (streaming_reuse_unverified or not exact_window_full_hit or not exact_current_window_generation)
          ):
            full_generation_usable = False
            if streaming_reuse_unverified:
              prefix_cache_entry["full_vlm_reason"] = "disabled_for_unverified_streaming_current_prompt_freshness"
            elif not exact_window_full_hit:
              prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_window_hit"
            elif not generation_window_signature_match:
              prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_generation_window_signature"
            else:
              prefix_cache_entry["full_vlm_reason"] = "disabled_without_exact_prompt_cache_context"
            if runtime_profile is not None:
              runtime_profile["vlm_full_generation_cache_disabled_for_streaming"] = 1 if streaming_reuse_unverified else 0
              runtime_profile["vlm_full_generation_cache_disabled_without_exact_window_hit"] = 0 if exact_window_full_hit else 1
              runtime_profile["vlm_full_generation_cache_disabled_without_exact_window_signature"] = (
                0 if generation_window_signature_match else 1
              )
              runtime_profile["vlm_full_generation_cache_disabled_without_exact_prompt_cache_context"] = (
                0 if prompt_cache_context_exact else 1
              )
              runtime_profile["vlm_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = (
                1 if trusted_replay_requested or bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_allowed")) else 0
              )
          elif trusted_fast_current_action_replay and runtime_profile is not None:
            runtime_profile["vlm_full_generation_cache_trusted_replay_fast_current_action"] = 1
            runtime_profile["vlm_full_generation_cache_disabled_for_streaming"] = 0
            runtime_profile["vlm_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = 0
        if full_generation_usable and isinstance(prefix_cache_entry, dict):
          cached_sequences = prefix_cache_entry["full_vlm_generated_sequences"].to(input_ids.device)
          cached_prompt_cache = prefix_cache_entry["full_vlm_prompt_cache"]
          pooled_copy = getattr(model_self, "_openpilot_pooled_prompt_cache_copy", None)
          cache_mode = "prefix_cache_borrowed"
          if callable(pooled_copy) and cached_prompt_cache is not None:
            cached_prompt_cache, cache_mode = pooled_copy(
              prefix_cache_entry,
              cached_prompt_cache,
              pool_key="full_vlm_prompt_cache_live_pool",
              pool_size=4,
            )
          prefix_cache_entry["full_vlm_hits"] = int(prefix_cache_entry.get("full_vlm_hits", 0)) + 1
          if runtime_profile is not None:
            runtime_profile["vlm_full_generation_cache_hit"] = 1
            runtime_profile["dflash_generic_full_generation_cache_hit"] = 1
            runtime_profile["manual_vlm_prefill_seconds"] = 0.0
            runtime_profile["manual_vlm_decode_seconds"] = 0.0
            runtime_profile["manual_vlm_decode_forwards"] = 0
            runtime_profile["manual_vlm_generated_tokens"] = int(cached_sequences.shape[1] - input_ids.shape[1])
            runtime_profile["vlm_full_generation_live_cache_mode"] = cache_mode
          return cached_sequences, cached_prompt_cache
        draft_source_cache_context_match = bool(
          isinstance(prefix_cache_entry, dict)
          and prefix_cache_entry.get("streaming_vlm_draft_source_cache_context_match")
        )
        draft_shifted_kv_plan = (
          prefix_cache_entry.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan")
          if isinstance(prefix_cache_entry, dict)
          else None
        )
        draft_shifted_kv_plan_valid = bool(
          isinstance(draft_shifted_kv_plan, dict)
          and draft_shifted_kv_plan.get("valid")
        )
        draft_shifted_kv_retained_ratio = (
          float(draft_shifted_kv_plan.get("validated_retained_ratio", draft_shifted_kv_plan.get("retained_ratio", 0.0)) or 0.0)
          if isinstance(draft_shifted_kv_plan, dict)
          else 0.0
        )
        draft_shifted_source_cache_verify_allowed = bool(
          draft_shifted_kv_plan_valid
          and bool(adapter.config.streaming_vlm_trust_shifted_draft)
          and draft_shifted_kv_retained_ratio >= max(0.0, min(1.0, float(adapter.config.streaming_vlm_prefix_reuse_min_overlap)))
        )
        if (
          runtime_profile is not None
          and isinstance(prefix_cache_entry, dict)
          and prefix_cache_entry.get("streaming_vlm_draft_generated_sequences") is not None
          and not draft_source_cache_context_match
        ):
          if not bool(adapter.config.streaming_vlm_source_cache_draft_verify_unverified) and not draft_shifted_source_cache_verify_allowed:
            runtime_profile["streaming_vlm_draft_verify_source_cache_skipped"] = "disabled_unverified_opt_in"
          elif not draft_shifted_kv_plan_valid:
            invalid_reason = (
              str(draft_shifted_kv_plan.get("invalid_reason", ""))
              if isinstance(draft_shifted_kv_plan, dict)
              else "missing_plan"
            )
            runtime_profile["streaming_vlm_draft_verify_source_cache_skipped"] = (
              f"invalid_shifted_prompt_kv_plan:{invalid_reason}"
            )
        if (
          isinstance(prefix_cache_entry, dict)
          and prefix_cache_entry.get("streaming_vlm_draft_generated_sequences") is not None
        ):
          draft_start = time.perf_counter()
          try:
            torch_mod = adapter._torch
            draft_sequences = prefix_cache_entry["streaming_vlm_draft_generated_sequences"].to(
              device=input_ids.device,
              dtype=input_ids.dtype,
            )
            draft_input_len = int(prefix_cache_entry.get("streaming_vlm_draft_input_seq_len", -1))
            draft_max_generation = int(prefix_cache_entry.get("streaming_vlm_draft_max_generation_length", -1))
            draft_eos_token = int(prefix_cache_entry.get("streaming_vlm_draft_eos_token_id", -1))
            draft_new_tokens = int(draft_sequences.shape[1]) - int(input_ids.shape[1])
            draft_new_tokens = min(draft_new_tokens, int(max_generation_length))
            if (
              int(draft_sequences.shape[0]) != int(input_ids.shape[0])
              or draft_input_len != int(input_ids.shape[1])
              or draft_max_generation != int(max_generation_length)
              or draft_eos_token != int(eos_token_id)
              or draft_new_tokens <= 0
            ):
              raise ValueError("streaming draft shape/settings mismatch")

            verify_suffix = draft_sequences[:, input_ids.shape[1] : input_ids.shape[1] + draft_new_tokens]
            verify_inputs = torch_mod.cat([input_ids, verify_suffix], dim=1)
            input_seq_len = int(input_ids.shape[1])

            source_cache = prefix_cache_entry.get("streaming_vlm_draft_dflash_target_cache")
            source_logits = prefix_cache_entry.get("streaming_vlm_draft_dflash_prefill_logits")
            source_layer_ids = tuple(prefix_cache_entry.get("streaming_vlm_draft_dflash_layer_ids", ()))
            source_cache_context_match = draft_source_cache_context_match
            source_kv_plan = draft_shifted_kv_plan
            source_kv_plan_valid = draft_shifted_kv_plan_valid
            source_cache_unverified = not source_cache_context_match
            if runtime_profile is not None and isinstance(source_kv_plan, dict):
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_valid"] = 1 if source_kv_plan_valid else 0
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_range_count"] = int(source_kv_plan.get("range_count", 0) or 0)
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_retained_tokens"] = int(source_kv_plan.get("retained_language_tokens", 0) or 0)
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_retained_ratio"] = float(source_kv_plan.get("retained_ratio", 0.0) or 0.0)
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_validated_retained_ratio"] = float(
                source_kv_plan.get("validated_retained_ratio", source_kv_plan.get("retained_ratio", 0.0)) or 0.0
              )
              runtime_profile["streaming_vlm_draft_shifted_kv_plan_invalid_reason"] = str(source_kv_plan.get("invalid_reason", ""))
              runtime_profile["shifted_prompt_kv_plan_valid"] = 1 if source_kv_plan_valid else 0
              runtime_profile["shifted_prompt_kv_plan_range_count"] = int(source_kv_plan.get("range_count", 0) or 0)
              runtime_profile["shifted_prompt_kv_plan_retained_tokens"] = int(source_kv_plan.get("retained_language_tokens", 0) or 0)
              runtime_profile["shifted_prompt_kv_plan_retained_ratio"] = float(source_kv_plan.get("retained_ratio", 0.0) or 0.0)
              runtime_profile["shifted_prompt_kv_plan_invalid_reason"] = str(source_kv_plan.get("invalid_reason", ""))
            shifted_source_cache_candidate = bool(
              not source_cache_context_match
              and source_kv_plan_valid
              and (
                bool(adapter.config.streaming_vlm_source_cache_draft_verify_unverified)
                or draft_shifted_source_cache_verify_allowed
              )
            )
            source_cache_allowed = bool(source_cache_context_match or shifted_source_cache_candidate)
            if runtime_profile is not None:
              runtime_profile["streaming_vlm_draft_verify_source_cache_allowed"] = 1 if source_cache_allowed else 0
              runtime_profile["streaming_vlm_draft_verify_shifted_source_cache_allowed_by_overlap"] = (
                1 if draft_shifted_source_cache_verify_allowed and not source_cache_context_match else 0
              )
              runtime_profile["streaming_vlm_draft_verify_source_cache_unverified_allowed"] = (
                1 if shifted_source_cache_candidate else 0
              )
              runtime_profile["streaming_vlm_draft_verify_shifted_source_cache_blocked_for_prompt_cache_freshness"] = (
                0
              )
            if (
              source_cache_allowed
              and
              source_cache is not None
              and source_logits is not None
              and source_layer_ids == tuple(int(item) for item in adapter._dflash_layer_ids)
            ):
              source_start = time.perf_counter()
              try:
                pooled_copy = getattr(model_self, "_openpilot_pooled_prompt_cache_copy", None)
                cache_mode = "deepcopy"
                if callable(pooled_copy):
                  suffix_prompt_cache, cache_mode = pooled_copy(
                    prefix_cache_entry,
                    source_cache,
                    pool_key="streaming_dflash_source_prompt_cache_live_pool",
                    pool_size=4,
                  )
                else:
                  suffix_prompt_cache = copy.deepcopy(source_cache)
                source_seq_len = int(suffix_prompt_cache.get_seq_length()) if hasattr(suffix_prompt_cache, "get_seq_length") else input_seq_len
                if source_seq_len != input_seq_len:
                  raise ValueError(f"source prompt cache length mismatch:{source_seq_len}!={input_seq_len}")
                source_logits = source_logits.to(device=input_ids.device)
                suffix_outputs = model_self.vlm(
                  input_ids=verify_suffix,
                  past_key_values=suffix_prompt_cache,
                  use_cache=True,
                  cache_position=torch_mod.arange(
                    source_seq_len,
                    source_seq_len + draft_new_tokens,
                    device=input_ids.device,
                  ),
                  logits_to_keep=draft_new_tokens,
                  return_dict=True,
                )
                suffix_logits = suffix_outputs.logits
                if int(source_logits.shape[1]) < 1:
                  raise ValueError("source-cache verifier missing prompt logits")
                if int(suffix_logits.shape[1]) < draft_new_tokens:
                  raise ValueError("source-cache verifier returned too few suffix logits")
                source_accepted_tokens = 0
                for draft_idx in range(draft_new_tokens):
                  if draft_idx == 0:
                    scores = source_logits[:, -1, :]
                  else:
                    scores = suffix_logits[:, draft_idx - 1, :]
                  scores = _mask_traj_token_logits(model_self, scores)
                  predicted = scores.argmax(dim=-1, keepdim=True).to(input_ids.device)
                  expected = verify_suffix[:, draft_idx : draft_idx + 1]
                  if not bool(torch_mod.equal(predicted, expected)):
                    break
                  source_accepted_tokens += 1
                if runtime_profile is not None:
                  runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
                  runtime_profile["streaming_vlm_draft_verify_accepted_tokens"] = int(source_accepted_tokens)
                  runtime_profile["streaming_vlm_draft_verify_tokens"] = int(draft_new_tokens)
                  runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
                  runtime_profile["streaming_vlm_draft_verify_suffix_seconds"] = time.perf_counter() - source_start
                  runtime_profile["streaming_vlm_draft_verify_mode"] = (
                    "exact_source_dflash_cache_suffix"
                    if source_cache_context_match
                    else "shifted_source_dflash_cache_suffix_unverified"
                  )
                  runtime_profile["streaming_vlm_draft_verify_source_cache_context_match"] = (
                    1 if source_cache_context_match else 0
                  )
                  runtime_profile["streaming_vlm_draft_verify_source_cache_mode"] = cache_mode
                  runtime_profile["dflash_streaming_vlm_draft_verify_attempted"] = 1
                if source_accepted_tokens == draft_new_tokens:
                  prompt_cache = suffix_outputs.past_key_values
                  generated_sequences = verify_inputs
                  prefix_cache_entry["streaming_vlm_draft_reason"] = (
                    "source_cache_verify_accepted"
                    if source_cache_context_match
                    else "source_cache_verify_accepted_unverified"
                  )
                  prefix_cache_entry["streaming_vlm_reuse_unverified"] = source_cache_unverified
                  prefix_cache_entry["streaming_vlm_reuse_mode"] = (
                    "source_cache_draft_verify"
                    if source_cache_context_match
                    else "source_cache_draft_verify_unverified"
                  )
                  prefix_cache_entry["full_vlm_generated_sequences"] = generated_sequences.detach().clone()
                  prefix_cache_entry["full_vlm_prompt_cache"] = copy.deepcopy(prompt_cache)
                  prefix_cache_entry["full_vlm_prompt_cache_owner"] = (
                    "dflash_exact_source_prompt_cache_suffix"
                    if source_cache_context_match
                    else "dflash_shifted_source_prompt_cache_suffix_unverified"
                  )
                  prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
                    "current_window_signature",
                    prefix_cache_entry.get("window_signature"),
                  )
                  prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = bool(source_cache_context_match)
                  prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
                  prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
                  prefix_cache_entry["full_vlm_input_seq_len"] = int(input_ids.shape[1])
                  prefix_cache_entry["full_vlm_reason"] = (
                    "dflash_exact_source_cache_suffix_verify"
                    if source_cache_context_match
                    else "dflash_shifted_source_cache_suffix_verify_unverified"
                  )
                  prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
                  prefix_cache_entry["dflash_layer_ids"] = tuple(source_layer_ids)
                  prefix_cache_entry["dflash_target_cache"] = copy.deepcopy(source_cache)
                  prefix_cache_entry["dflash_prefill_logits"] = source_logits.detach().clone()
                  prefix_cache_entry["dflash_target_hidden"] = self._detach_clone_tensor_tree(
                    prefix_cache_entry.get("streaming_vlm_draft_dflash_target_hidden")
                  )
                  prefix_cache_entry["dflash_has_selected_hidden"] = prefix_cache_entry["dflash_target_hidden"] is not None
                  prefix_cache_entry["dflash_reason"] = "shifted_source_cache_suffix_verify_unverified"
                  prefix_cache_entry["dflash_stores"] = int(prefix_cache_entry.get("dflash_stores", 0)) + 1
                  if runtime_profile is not None:
                    runtime_profile["streaming_vlm_draft_verify_hit"] = 1
                    runtime_profile["dflash_streaming_vlm_draft_verify_hit"] = 1
                    runtime_profile["manual_vlm_prefill_seconds"] = 0.0
                    runtime_profile["manual_vlm_decode_seconds"] = time.perf_counter() - source_start
                    runtime_profile["manual_vlm_decode_forwards"] = 1
                    runtime_profile["manual_vlm_generated_tokens"] = int(draft_new_tokens)
                    runtime_profile["manual_vlm_generated_sequences_owner"] = (
                      "dflash_exact_source_cache_verified_buffer"
                      if source_cache_context_match
                      else "dflash_shifted_source_cache_verified_buffer_unverified"
                    )
                    runtime_profile["vlm_full_generation_cache_store"] = 1
                    runtime_profile["dflash_streaming_draft_full_generation_cache_store"] = 1
                  return generated_sequences, prompt_cache
                if source_accepted_tokens != draft_new_tokens:
                  prefix_cache_entry["streaming_vlm_draft_reason"] = (
                    f"source_cache_verify_rejected_at:{source_accepted_tokens}/{draft_new_tokens}"
                  )
              except Exception as exc:
                prefix_cache_entry["streaming_vlm_draft_reason"] = f"source_cache_verify_error:{type(exc).__name__}"
                if runtime_profile is not None:
                  runtime_profile["streaming_vlm_draft_verify_source_cache_error"] = f"{type(exc).__name__}: {exc}"
            elif runtime_profile is not None and source_cache is not None and source_logits is not None:
              if not source_cache_context_match and not source_kv_plan_valid:
                runtime_profile["streaming_vlm_draft_verify_source_cache_skipped"] = "invalid_shifted_prompt_kv_plan"
              elif (
                not source_cache_context_match
                and not bool(adapter.config.streaming_vlm_source_cache_draft_verify_unverified)
                and not draft_shifted_source_cache_verify_allowed
              ):
                runtime_profile["streaming_vlm_draft_verify_source_cache_skipped"] = "disabled_unverified_opt_in"
              else:
                runtime_profile["streaming_vlm_draft_verify_source_cache_skipped"] = "missing_source_cache_requirements"

            verify_sequence_len = int(verify_inputs.shape[1])
            verify_kwargs = copy.copy(tokenized_data)
            verify_kwargs.pop("input_ids", None)
            verify_kwargs.pop("logits_to_keep", None)
            verify_kwargs["use_cache"] = True
            verify_kwargs["cache_position"] = torch_mod.arange(
              verify_sequence_len,
              device=input_ids.device,
            )
            prefill_rope_deltas = verify_kwargs.pop("cache_rope_deltas", None)
            verify_kwargs.pop("cache_position_ids", None)
            if prefill_rope_deltas is not None and hasattr(model_self, "vlm") and hasattr(model_self.vlm, "model"):
              setattr(model_self.vlm.model, "rope_deltas", prefill_rope_deltas)
            verify_attention_mask = verify_kwargs.get("attention_mask")
            if verify_attention_mask is not None and hasattr(verify_attention_mask, "shape"):
              if int(verify_attention_mask.shape[-1]) == int(input_seq_len):
                verify_kwargs["attention_mask"] = torch_mod.cat(
                  [
                    verify_attention_mask,
                    torch_mod.ones(
                      (*verify_attention_mask.shape[:-1], draft_new_tokens),
                      device=verify_attention_mask.device,
                      dtype=verify_attention_mask.dtype,
                    ),
                  ],
                  dim=-1,
                )

            verify_start = time.perf_counter()
            verify_outputs = None
            verify_mode = "current_prompt_full"
            text_prefix_len = adapter._static_text_prefix_len(prefix_cache_entry, input_ids)
            if text_prefix_len > 0:
              try:
                text_prefix_cache, text_prefix_mode = adapter._vlm_static_text_prefix_prompt_cache(
                  model_self,
                  input_ids,
                  verify_kwargs,
                  text_prefix_len,
                  runtime_profile,
                )
                if text_prefix_cache is not None:
                  suffix_verify_inputs = verify_inputs[:, text_prefix_len:]
                  suffix_verify_kwargs = copy.copy(verify_kwargs)
                  suffix_verify_kwargs["past_key_values"] = text_prefix_cache
                  suffix_verify_kwargs["cache_position"] = torch_mod.arange(
                    text_prefix_len,
                    verify_sequence_len,
                    device=input_ids.device,
                  )
                  verify_mode = "current_prompt_text_prefix_cache"
                  if (
                    bool(getattr(model_self, "_openpilot_graph_draft_verify_prefill_stage_enabled", False))
                    and hasattr(model_self, "_openpilot_vlm_prefill_with_cuda_graph")
                  ):
                    verify_outputs = model_self._openpilot_vlm_prefill_with_cuda_graph(
                      input_ids=suffix_verify_inputs,
                      tokenized_data={**suffix_verify_kwargs, "logits_to_keep": draft_new_tokens + 1},
                      runtime_profile=runtime_profile,
                      cache_position_seed=suffix_verify_kwargs["cache_position"],
                    )
                    if verify_outputs is not None:
                      verify_mode = "current_prompt_text_prefix_cache_graph"
                  if verify_outputs is None:
                    verify_outputs = model_self.vlm(
                      input_ids=suffix_verify_inputs,
                      logits_to_keep=draft_new_tokens + 1,
                      return_dict=True,
                      **suffix_verify_kwargs,
                    )
                  if runtime_profile is not None:
                    runtime_profile["streaming_vlm_draft_verify_text_prefix_cache_mode"] = text_prefix_mode
              except Exception as exc:
                verify_outputs = None
                verify_mode = "current_prompt_full_after_text_prefix_error"
                if runtime_profile is not None:
                  runtime_profile["streaming_vlm_draft_verify_text_prefix_error"] = f"{type(exc).__name__}: {exc}"
            if verify_outputs is None:
              if (
                bool(getattr(model_self, "_openpilot_graph_draft_verify_prefill_stage_enabled", False))
                and hasattr(model_self, "_openpilot_vlm_prefill_with_cuda_graph")
              ):
                verify_outputs = model_self._openpilot_vlm_prefill_with_cuda_graph(
                  input_ids=verify_inputs,
                  tokenized_data={**verify_kwargs, "logits_to_keep": draft_new_tokens + 1},
                  runtime_profile=runtime_profile,
                  cache_position_seed=verify_kwargs["cache_position"],
                )
                if verify_outputs is not None:
                  verify_mode = "current_prompt_full_graph"
            if verify_outputs is None:
              verify_outputs = model_self.vlm(
                input_ids=verify_inputs,
                logits_to_keep=draft_new_tokens + 1,
                return_dict=True,
                **verify_kwargs,
              )
            if runtime_profile is not None:
              runtime_profile["streaming_vlm_draft_verify_prefill_seconds"] = time.perf_counter() - verify_start
              runtime_profile["streaming_vlm_draft_verify_mode"] = verify_mode

            verify_logits = verify_outputs.logits
            if int(verify_logits.shape[1]) < draft_new_tokens + 1:
              raise ValueError("streaming draft verifier returned too few logits")
            logits_offset = int(verify_logits.shape[1]) - (draft_new_tokens + 1)
            accepted_tokens = 0
            for draft_idx in range(draft_new_tokens):
              scores = verify_logits[:, logits_offset + draft_idx, :]
              scores = _mask_traj_token_logits(model_self, scores)
              predicted = scores.argmax(dim=-1, keepdim=True).to(input_ids.device)
              expected = verify_inputs[:, input_seq_len + draft_idx : input_seq_len + draft_idx + 1]
              if not bool(torch_mod.equal(predicted, expected)):
                break
              accepted_tokens += 1

            if runtime_profile is not None:
              runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
              runtime_profile["streaming_vlm_draft_verify_accepted_tokens"] = int(accepted_tokens)
              runtime_profile["streaming_vlm_draft_verify_tokens"] = int(draft_new_tokens)
              runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
              runtime_profile["dflash_streaming_vlm_draft_verify_attempted"] = 1
            if accepted_tokens == draft_new_tokens:
              prompt_cache = verify_outputs.past_key_values
              generated_sequences = verify_inputs
              prefix_cache_entry["streaming_vlm_draft_reason"] = "verify_accepted"
              prefix_cache_entry["streaming_vlm_reuse_unverified"] = False
              prefix_cache_entry["streaming_vlm_reuse_mode"] = "draft_verify"
              prefix_cache_entry["full_vlm_generated_sequences"] = generated_sequences.detach().clone()
              prefix_cache_entry["full_vlm_prompt_cache"] = copy.deepcopy(prompt_cache)
              prefix_cache_entry["full_vlm_prompt_cache_owner"] = "dflash_streaming_draft_verified_current_prompt_cache"
              prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
                "current_window_signature",
                prefix_cache_entry.get("window_signature"),
              )
              prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = True
              prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
              prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
              prefix_cache_entry["full_vlm_input_seq_len"] = int(input_ids.shape[1])
              prefix_cache_entry["full_vlm_reason"] = "dflash_streaming_draft_verified_full_generation_ready"
              prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
              if runtime_profile is not None:
                runtime_profile["streaming_vlm_draft_verify_hit"] = 1
                runtime_profile["dflash_streaming_vlm_draft_verify_hit"] = 1
                runtime_profile["manual_vlm_prefill_seconds"] = time.perf_counter() - draft_start
                runtime_profile["manual_vlm_decode_seconds"] = 0.0
                runtime_profile["manual_vlm_decode_forwards"] = 0
                runtime_profile["manual_vlm_generated_tokens"] = int(draft_new_tokens)
                runtime_profile["manual_vlm_generated_sequences_owner"] = "dflash_streaming_draft_verified_buffer"
                runtime_profile["vlm_full_generation_cache_store"] = 1
                runtime_profile["dflash_streaming_draft_full_generation_cache_store"] = 1
              return generated_sequences, prompt_cache

            prefix_cache_entry["streaming_vlm_draft_reason"] = (
              f"verify_rejected_at:{accepted_tokens}/{draft_new_tokens}"
            )
          except Exception as exc:
            prefix_cache_entry["streaming_vlm_draft_reason"] = f"verify_error:{type(exc).__name__}"
            if runtime_profile is not None:
              runtime_profile["streaming_vlm_draft_verify_attempted"] = 1
              runtime_profile["streaming_vlm_draft_verify_error"] = f"{type(exc).__name__}: {exc}"
              runtime_profile["streaming_vlm_draft_verify_seconds"] = time.perf_counter() - draft_start
        temperature = 0.0 if adapter.config.greedy else 0.6
        result = dflash_generate_alpamayo(
          adapter._torch,
          model_self,
          adapter._dflash_model,
          adapter._dflash_mask_embedding,
          tokenized_data,
          input_ids,
          max_generation_length,
          temperature,
          runtime_profile,
        )
        if runtime_profile is not None:
          runtime_profile["dflash_enabled"] = 1
          runtime_profile["dflash_time_to_first_token_seconds"] = float(result.time_to_first_token_ms) / 1000.0
          runtime_profile["dflash_decode_seconds"] = float(result.decode_ms) / 1000.0
          runtime_profile["dflash_draft_seconds"] = float(result.draft_ms) / 1000.0
          runtime_profile["dflash_validate_seconds"] = float(result.validate_ms) / 1000.0
          runtime_profile["dflash_acceptance_blocks"] = len(result.acceptance_lengths)
          runtime_profile["dflash_acceptance_tokens"] = int(sum(result.acceptance_lengths))
          runtime_profile["dflash_generated_new_tokens"] = int(result.generated_new_tokens)
        gate_failure = adapter._dflash_gate_failure(result, runtime_profile)
        if gate_failure is not None:
          if runtime_profile is not None:
            runtime_profile["dflash_gate_failed"] = gate_failure
          adapter._disable_dflash_runtime(gate_failure)
        if isinstance(prefix_cache_entry, dict):
          try:
            prefix_cache_entry["full_vlm_generated_sequences"] = result.generated_sequences.detach().clone()
            prefix_cache_entry["full_vlm_prompt_cache"] = copy.deepcopy(result.prompt_cache)
            prefix_cache_entry["full_vlm_prompt_cache_owner"] = "dflash_generated_prompt_cache"
            prefix_cache_entry["full_vlm_window_signature"] = prefix_cache_entry.get(
              "current_window_signature",
              prefix_cache_entry.get("window_signature"),
            )
            prefix_cache_entry["full_vlm_prompt_cache_context_exact"] = True
            prefix_cache_entry["full_vlm_max_generation_length"] = int(max_generation_length)
            prefix_cache_entry["full_vlm_eos_token_id"] = int(eos_token_id)
            prefix_cache_entry["full_vlm_input_seq_len"] = int(input_ids.shape[1])
            prefix_cache_entry["full_vlm_reason"] = "dflash_full_generation_ready"
            prefix_cache_entry["full_vlm_stores"] = int(prefix_cache_entry.get("full_vlm_stores", 0)) + 1
            if runtime_profile is not None:
              runtime_profile["vlm_full_generation_cache_store"] = 1
              runtime_profile["dflash_full_generation_cache_store"] = 1
          except Exception as exc:
            prefix_cache_entry["full_vlm_reason"] = f"dflash_full_generation_cache_store_failed:{type(exc).__name__}"
            if runtime_profile is not None:
              runtime_profile["dflash_full_generation_cache_store_error"] = f"{type(exc).__name__}: {exc}"
        return result.generated_sequences, result.prompt_cache

      if not self._dflash_manual_generate_installed:
        self._dflash_original_manual_generate = getattr(self._model, "_manual_greedy_vlm_generate", None)
      self._model._manual_greedy_vlm_generate = dflash_vlm_generate.__get__(self._model, type(self._model))
      self._dflash_manual_generate_installed = True
      self._dflash_loaded = True
      self._dflash_sticky_disabled = False
      self._dflash_disable_cooldown_remaining = 0
      self._dflash_load_error = None
    except Exception as exc:
      self._disable_dflash_runtime(f"{type(exc).__name__}: {exc}")


  def _ensure_loaded(self) -> None:
    if self._loaded:
      return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if self.config.processor_model:
      os.environ["ALPAMAYO_PROCESSOR_MODEL"] = str(self.config.processor_model)
    _insert_alpamayo_paths(self.config.alpamayo_root)

    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    if not self.config.target_model.exists():
      raise FileNotFoundError(f"Alpamayo target model not found: {self.config.target_model}")
    target_is_flashvla = _is_flashvla_target_path(self.config.target_model)
    self._target_model_identity = {
      "path": str(self.config.target_model),
      "name": self.config.target_model.name,
      "isFlashVlaLike": target_is_flashvla,
      "requireFlashVlaTarget": bool(self.config.require_flashvla_target),
    }
    if self.config.require_flashvla_target and not target_is_flashvla:
      raise RuntimeError(
        "ALPAMAYO_REQUIRE_FLASHVLA_TARGET=1 but selected target model is not FlashVLA/FlashDriveVLA-like: "
        f"{self.config.target_model}. Set ALPAMAYO_TARGET_MODEL or FLASHVLA_TARGET_MODEL to the downloaded FlashVLA target."
      )

    if self.config.paro_native:
      os.environ.setdefault("TORCH_CUDA_ARCH_LIST", self.config.torch_cuda_arch_list)
      repo_root = Path(__file__).resolve().parents[2]
      if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
      from tools.alpamayo_speed.paro_native_marlin import ensure_native_paro_runtime

      ensure_native_paro_runtime()
      _patch_tie_weights_compat()
      _patch_alpamayo_init_for_paro(torch, self.config)

    if self.config.cuda_graphs:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

    if self.config.streaming_vision_cache:
      _patch_qwen3vl_streaming_vision_cache()
      _patch_qwen3vl_capture_safety()

    if self.config.manual_generation or self.config.cuda_graphs or self.config.dflash_enabled:
      _patch_manual_greedy_generation()
      _patch_qwen3vl_capture_safety()

    if self.config.device_map_mode == "auto":
      device_map: str | dict[str, int] = "auto"
    elif self.config.device_map_mode == "current_split":
      device_map = _build_current_split_device_map(self.config.split_index)
    elif self.config.device_map_mode == "single_device":
      device_map = _build_single_device_graph_map()
    else:
      raise ValueError(f"unsupported ALPAMAYO_DEVICE_MAP_MODE={self.config.device_map_mode}")

    max_memory = _gpu_only_max_memory(torch, self.config.gpu_mem_gib)
    load_kwargs = {
      "dtype": _torch_dtype(torch, self.config.model_dtype),
      "attn_implementation": self.config.attn_implementation,
      "device_map": device_map,
      "max_memory": max_memory,
      "low_cpu_mem_usage": True,
    }
    if self.config.paro_native:
      load_kwargs["ignore_mismatched_sizes"] = True

    restore_loader = _patch_flattened_conv3d_loader(torch)
    try:
      model = Alpamayo1_5.from_pretrained(str(self.config.target_model), **load_kwargs)
    finally:
      restore_loader()
    _assert_no_cpu_offload(model)
    self._supports_cuda_graph = bool(self.config.cuda_graphs and _model_uses_single_cuda_device(model))
    self._visual_patch_repair = (
      _repair_flattened_visual_patch_embed(torch, model, self.config.target_model)
      if self.config.paro_native
      else None
    )

    preconfigured_dtypes = {torch.float32, _torch_dtype(torch, self.config.model_dtype)}
    model_parameters = tuple(model.parameters())
    if model_parameters:
      model_device = model_parameters[0].device
    else:
      model_device = torch.device("cuda", 0)
    for tokenizer_name in ("traj_tokenizer", "hist_traj_tokenizer"):
      tokenizer = getattr(model, tokenizer_name, None)
      if tokenizer is None:
        continue
      precache_bounds = getattr(tokenizer, "precache_bounds", None)
      if callable(precache_bounds):
        for dtype in preconfigured_dtypes:
          precache_bounds(dtype=dtype, device=model_device)

    if self.config.expert_attn_implementation:
      _set_attention_implementation(model.expert, self.config.expert_attn_implementation)
    self._paro_finalize = None
    if self.config.paro_native:
      repo_root = Path(__file__).resolve().parents[2]
      if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
      from tools.alpamayo_speed.paro_native_marlin import finalize_native_paro_modules_for_device_map

      self._paro_finalize = finalize_native_paro_modules_for_device_map(model)

    helper.MIN_PIXELS = self.config.min_pixels
    helper.MAX_PIXELS = self.config.max_pixels
    if self.config.processor_model:
      helper.BASE_PROCESSOR_NAME = self.config.processor_model
    self._processor = helper.get_processor(model.tokenizer)
    self._configure_processor_pixel_budget(self._processor)
    self._target_model_identity["targetModelResolved"] = str(self.config.target_model.resolve())
    self._target_model_identity["processorSource"] = str(getattr(helper, "BASE_PROCESSOR_NAME", ""))
    self._target_model_identity["dflashDraftModel"] = str(self.config.dflash_draft_model)
    model_config = getattr(model, "config", None)
    if model_config is not None:
      self._target_model_identity["loadedModelType"] = str(getattr(model_config, "model_type", ""))
      self._target_model_identity["loadedVlmNameOrPath"] = str(getattr(model_config, "vlm_name_or_path", ""))
    self._model = model.eval()
    self._model._openpilot_vlm_prefill_with_cuda_graph = self._vlm_prefill_with_cuda_graph
    self._helper = helper
    self._torch = torch
    self._loaded = True

  def _configure_processor_pixel_budget(self, processor: Any) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
      return
    min_pixels = max(1, int(self.config.min_pixels))
    max_pixels = max(min_pixels, int(self.config.max_pixels))
    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
      try:
        setattr(image_processor, name, value)
      except Exception:
        pass
    try:
      image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
      }
    except Exception:
      pass
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
      init_kwargs = getattr(tokenizer, "init_kwargs", None)
      if isinstance(init_kwargs, dict):
        init_kwargs["min_pixels"] = min_pixels
        init_kwargs["max_pixels"] = max_pixels
      for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        try:
          setattr(tokenizer, name, value)
        except Exception:
          pass

  def _processor_pixel_config_debug(self) -> dict[str, Any]:
    image_processor = getattr(self._processor, "image_processor", None)
    if image_processor is None:
      return {}
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
      debug_size = {str(key): int(value) if isinstance(value, (int, float)) else value for key, value in size.items()}
    else:
      debug_size = str(size)
    tokenizer = getattr(self._processor, "tokenizer", None)
    tokenizer_init_kwargs = getattr(tokenizer, "init_kwargs", None) if tokenizer is not None else None
    return {
      "min_pixels": getattr(image_processor, "min_pixels", None),
      "max_pixels": getattr(image_processor, "max_pixels", None),
      "size": debug_size,
      "patch_size": getattr(image_processor, "patch_size", None),
      "merge_size": getattr(image_processor, "merge_size", None),
      "tokenizer_min_pixels": tokenizer_init_kwargs.get("min_pixels") if isinstance(tokenizer_init_kwargs, dict) else None,
      "tokenizer_max_pixels": tokenizer_init_kwargs.get("max_pixels") if isinstance(tokenizer_init_kwargs, dict) else None,
    }

  def _resize_frame_tensor_for_pixel_budget(self, frame_tensor: Any, torch_mod: Any) -> Any:
    max_pixels = int(self.config.max_pixels)
    if max_pixels <= 0 or not hasattr(frame_tensor, "shape") or len(frame_tensor.shape) != 3:
      return frame_tensor
    channels, height, width = (int(dim) for dim in frame_tensor.shape)
    if channels <= 0 or height <= 0 or width <= 0:
      return frame_tensor
    current_pixels = height * width
    if current_pixels <= max_pixels:
      return frame_tensor

    scale = math.sqrt(float(max_pixels) / float(current_pixels))
    patch_multiple = 32
    new_height = max(patch_multiple, int(math.floor(float(height) * scale / patch_multiple)) * patch_multiple)
    new_width = max(patch_multiple, int(math.floor(float(width) * scale / patch_multiple)) * patch_multiple)
    if new_height * new_width > max_pixels:
      while new_height * new_width > max_pixels and (new_height > patch_multiple or new_width > patch_multiple):
        if new_width >= new_height and new_width > patch_multiple:
          new_width -= patch_multiple
        elif new_height > patch_multiple:
          new_height -= patch_multiple
        else:
          break
    if new_height == height and new_width == width:
      return frame_tensor

    resized = torch_mod.nn.functional.interpolate(
      frame_tensor.unsqueeze(0).float(),
      size=(new_height, new_width),
      mode="bilinear",
      align_corners=False,
    ).squeeze(0)
    if not getattr(frame_tensor, "is_floating_point", lambda: False)():
      resized = resized.round().clamp_(0, 255).to(dtype=frame_tensor.dtype)
    else:
      resized = resized.to(dtype=frame_tensor.dtype)
    return resized.contiguous()

  def _resolve_frame_tensor(
    self,
    stream: str,
    frame: dict[str, Any],
    torch_mod: Any,
    cache_stats: dict[str, int],
  ) -> Any:
    frame_id = int(frame.get("frameId", -1))
    signature = _frame_cache_signature(frame)
    with self._cache_lock:
      stream_cache = self._frame_cache.setdefault(stream, OrderedDict())
      cached = stream_cache.get(frame_id)
      if cached is not None and cached[0] == signature:
        stream_cache.move_to_end(frame_id)
        cache_stats["frame_hits"] += 1
        return cached[1]
      cache_stats["frame_misses"] += 1

    decoded = _decode_frame_rgb(frame)
    frame_tensor = torch_mod.from_numpy(np.ascontiguousarray(decoded)).permute(2, 0, 1)
    frame_tensor = self._resize_frame_tensor_for_pixel_budget(frame_tensor, torch_mod)

    with self._cache_lock:
      stream_cache = self._frame_cache.setdefault(stream, OrderedDict())
      stream_cache[frame_id] = (signature, frame_tensor)
      stream_cache.move_to_end(frame_id)
      while len(stream_cache) > self._frame_cache_max_entries:
        stream_cache.popitem(last=False)
    return frame_tensor

  def _build_model_inputs(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assert self._torch is not None and self._helper is not None and self._processor is not None and self._model is not None
    torch = self._torch
    helper = self._helper
    processor = self._processor
    ego_xyz, ego_rot = _request_ego_history(request)
    cache_stats = {
      "frame_hits": 0,
      "frame_misses": 0,
      "tokenized_cache_hits": 0,
      "tokenized_cache_misses": 0,
      "window_full_hit": 0,
      "stream_window_hits": 0,
      "stream_window_misses": 0,
      "stream_overlap_frames": 0,
      "stream_window_overlap_events": 0,
      "streamingVisionCache": StreamingAlpamayoVisionCache._empty_stats(self.config.streaming_vision_cache),
    }

    frames_by_stream = _group_frames(request, self.config.camera_streams, self.config.num_frames)
    image_frame_order: list[dict[str, Any]] = []
    for order_stream_index, order_stream in enumerate(self.config.camera_streams):
      camera_index = STREAM_TO_ALPAMAYO_CAMERA_INDEX.get(order_stream)
      for order_frame_index, order_frame in enumerate(frames_by_stream.get(order_stream, [])):
        image_frame_order.append({
          "kind": "image",
          "stream": order_stream,
          "stream_index": order_stream_index,
          "frame_index": order_frame_index,
          "signature": _frame_cache_signature(order_frame),
          "camera_index": camera_index,
        })
    nav_text = _request_nav_text(request)
    window_signature = _window_signature(frames_by_stream, nav_text, self.config.camera_streams)
    with self._cache_lock:
      cached_tokenized = self._tokenized_cache.get(window_signature)
      if cached_tokenized is not None:
        cache_stats["tokenized_cache_hits"] += 1
        cache_stats["window_full_hit"] = 1
      else:
        cache_stats["tokenized_cache_misses"] += 1

    total_frames = sum(len(frames) for frames in frames_by_stream.values())
    cache_stats["total_frames"] = total_frames

    tokenized = None
    if cached_tokenized is not None:
      tokenized = cached_tokenized

    if tokenized is None:
      image_frames = []
      camera_indices = []
      for stream in self.config.camera_streams:
        if stream not in STREAM_TO_ALPAMAYO_CAMERA_INDEX:
          raise ValueError(f"no Alpamayo camera index mapping for stream={stream}")
        stream_frames = frames_by_stream[stream]
        stream_window_signature = tuple(_frame_cache_signature(frame) for frame in stream_frames)
        with self._cache_lock:
          stream_cache = self._stream_window_cache.setdefault(stream, OrderedDict())
          stream_tensor = stream_cache.get(stream_window_signature)
          if stream_tensor is not None:
            stream_cache.move_to_end(stream_window_signature)
            cache_stats["stream_window_hits"] += 1
          else:
            cache_stats["stream_window_misses"] += 1
            best_overlap = 0
            best_overlap_tensor = None
            for prior_signature, prior_tensor in reversed(stream_cache.items()):
              overlap = _max_suffix_prefix_overlap(prior_signature, stream_window_signature)
              if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_tensor = prior_tensor
                if overlap == len(stream_window_signature):
                  break
            if best_overlap > 0 and best_overlap_tensor is not None:
              cache_stats["stream_window_overlap_events"] += 1
              cache_stats["stream_overlap_frames"] += best_overlap
              suffix_new = stream_frames[best_overlap:]
              stream_suffix = [
                self._resolve_frame_tensor(
                  stream=stream,
                  frame=frame,
                  torch_mod=torch,
                  cache_stats=cache_stats,
                )
                for frame in suffix_new
              ]
              if stream_suffix:
                stream_tensor = torch.cat([best_overlap_tensor[-best_overlap:], torch.stack(stream_suffix, dim=0)], dim=0)
              else:
                stream_tensor = best_overlap_tensor

        if stream_tensor is None:
          decoded = [
            self._resolve_frame_tensor(
              stream=stream,
              frame=frame,
              torch_mod=torch,
              cache_stats=cache_stats,
            )
            for frame in stream_frames
          ]
          stream_tensor = torch.stack(decoded, dim=0)
        with self._cache_lock:
          stream_cache = self._stream_window_cache.setdefault(stream, OrderedDict())
          stream_cache[stream_window_signature] = stream_tensor
          stream_cache.move_to_end(stream_window_signature)
          while len(stream_cache) > self._stream_window_cache_max_entries:
            stream_cache.popitem(last=False)
        image_frames.append(stream_tensor)
        camera_indices.append(STREAM_TO_ALPAMAYO_CAMERA_INDEX[stream])

      images = torch.stack(image_frames, dim=0)
      cam_ids = torch.tensor(camera_indices, dtype=torch.int64)
      messages = helper.create_message(
        frames=images.flatten(0, 1),
        camera_indices=cam_ids,
        num_frames_per_camera=self.config.num_frames,
        nav_text=nav_text,
      )
      tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
      )
      tokenized_merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
      tokenized["_openpilot_prefix_semantic_signature"] = _prefix_semantic_signature(self._torch, tokenized)
      tokenized["_openpilot_visual_token_count"] = _visual_token_count_from_grids(
        tokenized.get("image_grid_thw"),
        tokenized.get("video_grid_thw"),
        tokenized_merge_size,
      )
      with self._cache_lock:
        self._tokenized_cache[window_signature] = tokenized
        self._tokenized_cache.move_to_end(window_signature)
        while len(self._tokenized_cache) > self._tokenized_cache_max_entries:
          self._tokenized_cache.popitem(last=False)
    if self.config.streaming_vision_cache:
      merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
      cache_stats["streamingVisionCache"] = self._streaming_vision_cache.prepare(
        frames_by_stream=frames_by_stream,
        image_grid_thw=tokenized.get("image_grid_thw"),
        merge_size=merge_size,
        torch_mod=torch,
      )
    cache_stats["frame_overlap_ratio"] = (
      cache_stats["frame_hits"] / total_frames if total_frames > 0 else 0.0
    )
    num_streams = len(self.config.camera_streams)
    cache_stats["stream_window_overlap_ratio"] = (
      cache_stats["stream_window_hits"] / num_streams if num_streams > 0 else 0.0
    )
    cache_stats["stream_overlap_ratio"] = (
      cache_stats["stream_overlap_frames"] / total_frames if total_frames > 0 else 0.0
    )

    if cached_tokenized is not None:
      # Avoid mutating cached tokenization because fused history tokens depend on ego history.
      tokenized = {key: value for key, value in tokenized.items()}

    if "_openpilot_prefix_semantic_signature" not in tokenized or "_openpilot_visual_token_count" not in tokenized:
      producer_merge_size = int(getattr(getattr(processor, "image_processor", None), "merge_size", 1) or 1)
      tokenized["_openpilot_prefix_semantic_signature"] = _prefix_semantic_signature(self._torch, tokenized)
      tokenized["_openpilot_visual_token_count"] = _visual_token_count_from_grids(
        tokenized.get("image_grid_thw"),
        tokenized.get("video_grid_thw"),
        producer_merge_size,
      )

    model_inputs = helper.to_device(
      {
        "tokenized_data": tokenized,
        "ego_history_xyz": self._ego_history_tensor(ego_xyz, "_ego_history_xyz_buffer"),
        "ego_history_rot": self._ego_history_tensor(ego_rot, "_ego_history_rot_buffer"),
      },
      self._model.device,
    )

    tokenized_data = model_inputs["tokenized_data"]
    traj_data_vlm = {
      "ego_history_xyz": model_inputs["ego_history_xyz"],
      "ego_history_rot": model_inputs["ego_history_rot"],
    }

    # Fuse trajectory tokens before CUDA graph capture to avoid executing this path inside captured region.
    try:
      tokenized_data["input_ids"] = self._model.fuse_traj_tokens(tokenized_data["input_ids"], traj_data_vlm)
      tokenized_data["_skip_traj_fusion"] = True
      tokenized_data["_openpilot_fused_input_ids_signature"] = _tensor_crc32_signature(
        self._torch,
        tokenized_data.get("input_ids"),
      )
    except Exception:
      tokenized_data["_skip_traj_fusion"] = False

    input_seq_len = tokenized_data["input_ids"].shape[1]
    tokenized_data["cache_position_seed"] = self._cache_position_seed_for_length(
      input_seq_len + self.config.max_generation_length + 1
    )
    if self.config.cuda_graphs or self.config.manual_generation:
      try:
        rope_cache_key = (
          "rope_index_v1",
          str(self.config.target_model),
          str(self._model.device),
          int(input_seq_len),
          tokenized_data.get("_openpilot_prefix_semantic_signature"),
          _tensor_tree_signature(
            self._torch,
            {
              "attention_mask": tokenized_data.get("attention_mask"),
              "image_grid_thw": tokenized_data.get("image_grid_thw"),
              "video_grid_thw": tokenized_data.get("video_grid_thw"),
            },
          ),
        )
        with self._cache_lock:
          cached_rope = self._rope_index_cache.get(rope_cache_key)
          if cached_rope is not None:
            self._rope_index_cache.move_to_end(rope_cache_key)
        if cached_rope is None:
          with self._torch.no_grad():
            cache_position_ids, cache_rope_deltas = self._model.vlm.model.get_rope_index(
              tokenized_data["input_ids"],
              tokenized_data.get("image_grid_thw"),
              tokenized_data.get("video_grid_thw"),
              attention_mask=tokenized_data.get("attention_mask"),
            )
          with self._cache_lock:
            self._rope_index_cache[rope_cache_key] = (cache_position_ids, cache_rope_deltas)
            self._rope_index_cache.move_to_end(rope_cache_key)
            while len(self._rope_index_cache) > self._rope_index_cache_max_entries:
              self._rope_index_cache.popitem(last=False)
          cache_stats["rope_index_cache_hit"] = 0
        else:
          cache_position_ids, cache_rope_deltas = cached_rope
          cache_stats["rope_index_cache_hit"] = 1
        cache_stats["rope_index_cache_depth"] = len(self._rope_index_cache)
        tokenized_data["cache_position_ids"] = cache_position_ids
        tokenized_data["cache_rope_deltas"] = cache_rope_deltas
        if cache_rope_deltas is not None:
          setattr(self._model.vlm.model, "rope_deltas", cache_rope_deltas)
      except Exception:
        pass

    self._record_vlm_prefix_cache_candidate(
      window_signature=window_signature,
      tokenized_data=tokenized_data,
      cache_stats=cache_stats,
    )
    prefix_stats = cache_stats.get("vlmPrefixCache", {})
    prefix_entry = getattr(self._model, "_openpilot_vlm_prefix_cache_entry", None)
    exact_window_full_hit = int(cache_stats.get("window_full_hit", 0) or 0) == 1 or bool(prefix_stats.get("exactHit"))
    if isinstance(prefix_entry, dict):
      prefix_entry["current_window_full_hit"] = exact_window_full_hit
      prefix_entry["current_window_signature"] = window_signature
      prefix_entry["window_full_hit"] = exact_window_full_hit
    trusted_visual_replay_requested = bool(
      isinstance(prefix_entry, dict)
      and prefix_entry.get("streaming_vlm_trusted_replay_allowed")
    )
    trusted_visual_replay_allowed = bool(
      trusted_visual_replay_requested
      and self.config.disable_reasoning_generation
      and self.config.no_reasoning_trust_shifted_prompt_cache
      and not self._require_state_fresh_no_reasoning()
    )
    fast_current_action_visual_replay_allowed = bool(
      trusted_visual_replay_requested
      and getattr(self, "_openpilot_require_fast_no_prefill", False)
      and isinstance(prefix_entry, dict)
      and prefix_entry.get("streaming_vlm_fast_current_action_replay_allowed")
    )
    if trusted_visual_replay_requested and not (trusted_visual_replay_allowed or fast_current_action_visual_replay_allowed):
      cache_stats["streaming_visual_feature_precompute_trusted_replay_disabled_for_diffusion_freshness"] = 1
    skip_visual_for_full_generation_replay = bool(
      isinstance(prefix_stats, dict)
      and int(prefix_stats.get("hit", 0) or 0) == 1
      and bool(prefix_stats.get("fullGenerationReady"))
      and (exact_window_full_hit or trusted_visual_replay_allowed or fast_current_action_visual_replay_allowed)
    )
    source_cache_shifted_kv_plan = (
      prefix_entry.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan")
      if isinstance(prefix_entry, dict)
      else None
    )
    source_cache_shifted_kv_plan_valid = bool(
      isinstance(source_cache_shifted_kv_plan, dict)
      and source_cache_shifted_kv_plan.get("valid")
    )
    source_cache_shifted_kv_retained_ratio = (
      float(source_cache_shifted_kv_plan.get("validated_retained_ratio", source_cache_shifted_kv_plan.get("retained_ratio", 0.0)) or 0.0)
      if isinstance(source_cache_shifted_kv_plan, dict)
      else 0.0
    )
    source_cache_shifted_verify_allowed_by_overlap = bool(
      source_cache_shifted_kv_plan_valid
      and bool(self.config.streaming_vlm_trust_shifted_draft)
      and source_cache_shifted_kv_retained_ratio >= max(0.0, min(1.0, float(self.config.streaming_vlm_prefix_reuse_min_overlap)))
    )
    source_cache_draft_verify_allowed = bool(
      isinstance(prefix_entry, dict)
      and (
        bool(prefix_entry.get("streaming_vlm_draft_source_cache_context_match"))
        or (
          source_cache_shifted_kv_plan_valid
          and (
            bool(self.config.streaming_vlm_source_cache_draft_verify_unverified)
            or source_cache_shifted_verify_allowed_by_overlap
          )
        )
      )
    )
    skip_visual_for_source_cache_draft_verify = bool(
      isinstance(prefix_stats, dict)
      and int(prefix_stats.get("hit", 0) or 0) == 1
      and str(prefix_stats.get("streamingReuseMode", "")) == "draft_verify"
      and isinstance(prefix_entry, dict)
      and source_cache_draft_verify_allowed
      and prefix_entry.get("streaming_vlm_draft_dflash_target_cache") is not None
      and prefix_entry.get("streaming_vlm_draft_dflash_prefill_logits") is not None
    )
    try:
      shifted_kv_visual_fill_precompute = bool(int(os.environ.get("ALPAMAYO_SHIFTED_KV_VISUAL_FILL_PRECOMPUTE", "0")))
    except Exception:
      shifted_kv_visual_fill_precompute = False
    skip_visual_for_shifted_kv_current_suffix = bool(
      isinstance(prefix_stats, dict)
      and int(prefix_stats.get("hit", 0) or 0) == 1
      and self.config.disable_reasoning_generation
      and self._require_state_fresh_no_reasoning()
      and isinstance(prefix_entry, dict)
      and prefix_entry.get("streaming_vlm_shift_source_prompt_cache") is not None
      and source_cache_shifted_verify_allowed_by_overlap
      and not shifted_kv_visual_fill_precompute
    )
    if skip_visual_for_full_generation_replay:
      cache_stats["streaming_visual_feature_precompute_skipped_full_generation_replay"] = 1
    if skip_visual_for_source_cache_draft_verify:
      cache_stats["streaming_visual_feature_precompute_skipped_source_cache_draft_verify_unverified"] = 1
    if skip_visual_for_shifted_kv_current_suffix:
      cache_stats["streaming_visual_feature_precompute_skipped_shifted_kv_current_suffix"] = 1
    if isinstance(prefix_entry, dict):
      streaming_reuse_mode = str(prefix_entry.get("streaming_vlm_reuse_mode", ""))
      streaming_reuse_unverified = bool(prefix_entry.get("streaming_vlm_reuse_unverified")) or streaming_reuse_mode.endswith("_unverified")
      if (
        (
          skip_visual_for_full_generation_replay
          and not exact_window_full_hit
          and not trusted_visual_replay_allowed
          and not fast_current_action_visual_replay_allowed
        )
        or (streaming_reuse_unverified and not (trusted_visual_replay_allowed or fast_current_action_visual_replay_allowed))
      ):
        if skip_visual_for_full_generation_replay:
          cache_stats["streaming_visual_feature_precompute_full_generation_replay_disabled_for_freshness"] = 1
        if skip_visual_for_source_cache_draft_verify and streaming_reuse_unverified:
          cache_stats["streaming_visual_feature_precompute_source_cache_skip_disabled_for_freshness"] = 1
        skip_visual_for_full_generation_replay = False
        if streaming_reuse_unverified:
          skip_visual_for_source_cache_draft_verify = False

    skip_visual_precompute = (
      skip_visual_for_full_generation_replay
      or skip_visual_for_source_cache_draft_verify
      or skip_visual_for_shifted_kv_current_suffix
    )

    if self.config.streaming_vision_cache and not skip_visual_precompute:
      streaming_token_blocks = cache_stats.get("streamingVisionCache", {}).get("token_blocks")
      self._cache_streaming_visual_features(
        tokenized_data,
        cache_stats,
        kind="image",
        token_blocks=streaming_token_blocks,
        visual_order=image_frame_order,
      )
      self._cache_streaming_visual_features(
        tokenized_data,
        cache_stats,
        kind="video",
        token_blocks=streaming_token_blocks,
        visual_order=None,
      )

    if self.config.cuda_graphs and not skip_visual_precompute:
      if (
        "_openpilot_precomputed_image_features" not in tokenized_data
        and "pixel_values" in tokenized_data
        and tokenized_data["pixel_values"] is not None
      ):
        with self._torch.no_grad():
          precomputed = self._model.vlm.model.get_image_features(
            tokenized_data["pixel_values"],
            tokenized_data.get("image_grid_thw"),
          )
        tokenized_data["_openpilot_precomputed_image_features"] = precomputed
        if tokenized_data.get("image_grid_thw") is not None:
          with self._torch.no_grad():
            tokenized_data["_openpilot_precomputed_patch_pos_embeds"] = (
              self._model.vlm.model.visual.fast_pos_embed_interpolate(tokenized_data["image_grid_thw"])
            )

    return model_inputs, cache_stats

  def _assert_visual_inputs(self, tokenized_data: dict[str, Any], torch_mod: Any) -> dict[str, list[int]]:
    for key in ("input_ids", "pixel_values", "image_grid_thw"):
      if key not in tokenized_data:
        raise RuntimeError(f"missing required visual tokenized field: {key}")

    for key in ("input_ids", "pixel_values"):
      value = tokenized_data[key]
      if not hasattr(value, "shape"):
        raise RuntimeError(f"invalid visual tokenized field type for {key}")

    input_ids = tokenized_data["input_ids"]
    if input_ids.ndim != 2:
      raise RuntimeError("input_ids must be 2-D [B,L]")
    if int(input_ids.shape[0]) < 1 or int(input_ids.shape[1]) < 1:
      raise RuntimeError("input_ids must contain at least one token")

    pixel_values = tokenized_data["pixel_values"]
    if not hasattr(pixel_values, "numel") or int(pixel_values.numel()) == 0:
      raise RuntimeError("pixel_values must be non-empty")
    if not hasattr(pixel_values, "ndim"):
      raise RuntimeError("invalid pixel_values type")
    if int(pixel_values.ndim) == 4:
      if int(pixel_values.shape[1]) != 3:
        raise RuntimeError("pixel_values channel count must be 3")
    elif int(pixel_values.ndim) != 2:
      raise RuntimeError("pixel_values must be [N, 3, H, W] or [N, P]")
    if not torch_mod.isfinite(pixel_values).all():
      raise RuntimeError("pixel_values contains non-finite values")

    image_grid = tokenized_data["image_grid_thw"]
    if not hasattr(image_grid, "shape"):
      raise RuntimeError("invalid image_grid_thw type")
    if image_grid.ndim != 2 or int(image_grid.shape[1]) != 3:
      raise RuntimeError("image_grid_thw must be shape [N, 3]")
    if int(image_grid.numel()) == 0:
      raise RuntimeError("image_grid_thw must be non-empty")
    if torch_mod is not None:
      if torch_mod.any(image_grid <= 0):
        raise RuntimeError("image_grid_thw contains invalid geometry values")

    if int(pixel_values.ndim) == 4 and int(pixel_values.shape[0]) != int(image_grid.shape[0]):
      raise RuntimeError("pixel_values batch count must match image_grid_thw")

    if hasattr(self._model, "vlm"):
      model_vlm = self._model.vlm
      if not hasattr(model_vlm, "config"):
        raise RuntimeError("model.vlm.config missing required multimodal metadata")

      model_vlm_config = model_vlm.config
      if not hasattr(model_vlm_config, "image_token_id"):
        raise RuntimeError("model.vlm.config.image_token_id missing required multimodal metadata")

      image_grid_thw = image_grid
      processor = self._processor
      merge_size = 1
      if processor is not None and hasattr(processor, "image_processor"):
        merge_size = int(getattr(processor.image_processor, "merge_size", merge_size))
      expected_merge_area = max(1, int(merge_size) ** 2)
      cuda_count_tensors = bool(
        getattr(image_grid_thw, "is_cuda", False) or getattr(input_ids, "is_cuda", False)
      )
      expected_image_tokens = None
      expected_pixel_rows = None
      if not cuda_count_tensors:
        expected_image_tokens = int((image_grid_thw.prod(dim=1) // expected_merge_area).sum().item())
        expected_pixel_rows = int(image_grid_thw.prod(dim=1).sum().item())
      if (
        expected_pixel_rows is not None
        and int(pixel_values.ndim) == 2
        and int(pixel_values.shape[0]) != expected_pixel_rows
      ):
        raise RuntimeError("flattened pixel_values count does not match image_grid_thw")
      if expected_image_tokens is not None and expected_image_tokens <= 0:
        raise RuntimeError("no deepstack image tokens expected from image_grid_thw")
      image_token_id = int(model_vlm_config.image_token_id)
      if expected_image_tokens is not None:
        actual_image_tokens = int((input_ids == image_token_id).sum().item())
        if actual_image_tokens != expected_image_tokens:
          raise RuntimeError("deepstack image token count does not match image_grid_thw")

    return {
      "inputIdsShape": [int(item) for item in input_ids.shape],
      "pixelValuesShape": [int(item) for item in pixel_values.shape],
      "imageGridThwShape": [int(item) for item in image_grid.shape],
    }

  def _assert_reasoning_mode(self, request: dict[str, Any]) -> str:
    runtime_config = request.get("runtimeConfig")
    if not isinstance(runtime_config, dict):
      raise RuntimeError("runtimeConfig missing")
    reasoning_mode = runtime_config.get("reasoningMode")
    if reasoning_mode is None:
      raise RuntimeError("runtimeConfig.reasoningMode missing")
    if _is_truthy_skip_flag(runtime_config.get("skipVlmGeneration")):
      raise RuntimeError("runtimeConfig.skipVlmGeneration is forbidden for production Alpamayo")
    requested_reasoning_mode = str(reasoning_mode)
    if self.config.disable_reasoning_generation:
      if requested_reasoning_mode not in (REQUIRED_REASONING_MODE, "disabled", "none", "no_reasoning"):
        raise RuntimeError(
          "runtimeConfig.reasoningMode must be 'full', 'disabled', 'none', or 'no_reasoning' "
          "when ALPAMAYO_DISABLE_REASONING_GENERATION=1"
        )
      return "disabled"
    if requested_reasoning_mode != REQUIRED_REASONING_MODE:
      raise RuntimeError(
        f"runtimeConfig.reasoningMode must be '{REQUIRED_REASONING_MODE}' in production local Alpamayo"
      )
    return requested_reasoning_mode

  def _cache_position_seed_for_length(self, length: int) -> Any:
    assert self._torch is not None and self._model is not None
    requested_len = max(1, int(length))
    device = self._model.device
    buffer = self._cache_position_seed_buffer
    if (
      buffer is None
      or int(self._cache_position_seed_buffer_len) < requested_len
      or self._cache_position_seed_buffer_device != device
    ):
      self._cache_position_seed_buffer = self._torch.arange(
        requested_len,
        device=device,
        dtype=self._torch.long,
      )
      self._cache_position_seed_buffer_len = requested_len
      self._cache_position_seed_buffer_device = device
    return self._cache_position_seed_buffer[:requested_len]

  def _ego_history_tensor(self, value: np.ndarray, attr_name: str) -> Any:
    assert self._torch is not None and self._model is not None
    cpu_tensor = self._torch.from_numpy(value).unsqueeze(0).unsqueeze(0)
    device = self._model.device
    buffer = getattr(self, attr_name)
    if (
      buffer is None
      or tuple(buffer.shape) != tuple(cpu_tensor.shape)
      or buffer.dtype != cpu_tensor.dtype
      or buffer.device != device
    ):
      buffer = self._torch.empty(
        tuple(cpu_tensor.shape),
        device=device,
        dtype=cpu_tensor.dtype,
      )
      setattr(self, attr_name, buffer)
    buffer.copy_(cpu_tensor, non_blocking=True)
    return buffer

  @staticmethod
  def _model_inputs_without_tokenized_metadata(model_inputs: dict[str, Any]) -> dict[str, Any]:
    tokenized_data = model_inputs.get("tokenized_data")
    if not isinstance(tokenized_data, dict) or not any(key in tokenized_data for key in OPENPILOT_TOKENIZED_METADATA_KEYS):
      return model_inputs
    clean_tokenized = dict(tokenized_data)
    for key in OPENPILOT_TOKENIZED_METADATA_KEYS:
      clean_tokenized.pop(key, None)
    clean_inputs = dict(model_inputs)
    clean_inputs["tokenized_data"] = clean_tokenized
    return clean_inputs

  @staticmethod
  def _rollout_model_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    clean_inputs = LocalAlpamayoAdapter._model_inputs_without_tokenized_metadata(model_inputs)
    tokenized_data = clean_inputs.get("tokenized_data")
    if not isinstance(tokenized_data, dict):
      return clean_inputs
    rollout_inputs = dict(clean_inputs)
    rollout_inputs["tokenized_data"] = dict(tokenized_data)
    return rollout_inputs

  def _assert_reasoning_output(self, extra: dict[str, Any], prompt_tokens: int) -> int:
    if not isinstance(extra, dict):
      raise RuntimeError("missing extra model outputs for reasoning text generation")
    generated_sequence_length = extra.get("generated_sequence_length")
    if not isinstance(generated_sequence_length, (int, float, np.integer, np.floating)):
      raise RuntimeError("model extra output missing generated_sequence_length")
    generated_tokens = int(generated_sequence_length) - int(prompt_tokens)
    if self.config.disable_reasoning_generation:
      if int(generated_sequence_length) < prompt_tokens:
        raise RuntimeError("model generated sequence shorter than prompt in no-reasoning Alpamayo mode")
      return max(0, generated_tokens)
    if generated_tokens <= 0:
      raise RuntimeError("skipping VLM reasoning output is forbidden for production local Alpamayo")
    return generated_tokens

  def _build_infer_graph_signature(
    self,
    model_inputs: dict[str, Any],
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    diffusion_kwargs: dict[str, Any],
    do_sample: bool,
    manual_generation: bool,
    skip_vlm_generation: bool,
    return_extra: bool,
  ) -> tuple[Any, ...]:
    assert self._torch is not None
    model_inputs = self._model_inputs_without_tokenized_metadata(model_inputs)
    return (
      "sample_trajectories_from_data_with_vlm_rollout",
      _tensor_tree_signature(self._torch, model_inputs),
      float(top_p),
      float(temperature),
      int(num_traj_samples),
      int(max_generation_length),
      _freeze_cache_key_value(tuple(sorted(diffusion_kwargs.items()))),
      bool(do_sample),
      bool(manual_generation),
      bool(skip_vlm_generation),
      bool(return_extra),
      tuple(sorted(self.config.camera_streams)),
      self.config.cuda_graphs,
      self.config.autocast_dtype,
      self.config.model_dtype,
    )

  def _sample_with_model(
    self,
    model_inputs: dict[str, Any],
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    diffusion_kwargs: dict[str, Any],
    runtime_profile: dict[str, float | int],
    do_sample: bool,
    manual_generation: bool,
    skip_vlm_generation: bool,
    return_extra: bool,
  ) -> tuple[Any, Any, dict[str, Any]]:
    assert self._model is not None
    assert self._torch is not None
    model_inputs = self._model_inputs_without_tokenized_metadata(model_inputs)
    runtime_profile["torch_inference_mode"] = 0
    runtime_profile["torch_no_grad_mode"] = 1
    tokenized_data = model_inputs.get("tokenized_data", {})
    action_space = getattr(self._model, "action_space", None)
    original_action_to_traj = None
    hooked_action_to_traj = False
    if action_space is not None:
      candidate_action_to_traj = getattr(action_space, "action_to_traj", None)
      if callable(candidate_action_to_traj):
        original_action_to_traj = candidate_action_to_traj

        def timed_action_to_traj(*args: Any, **kwargs: Any) -> Any:
          action_to_traj_start = time.perf_counter()
          try:
            return original_action_to_traj(*args, **kwargs)
          finally:
            runtime_profile["action_to_traj_seconds"] = runtime_profile.get("action_to_traj_seconds", 0.0) + (
              time.perf_counter() - action_to_traj_start
            )

        try:
          action_space.action_to_traj = timed_action_to_traj
          hooked_action_to_traj = True
        except Exception:
          hooked_action_to_traj = False
    vlm_model = self._model.vlm.model
    precomputed_image_features = tokenized_data.get("_openpilot_precomputed_image_features")
    precomputed_video_features = tokenized_data.get("_openpilot_precomputed_video_features")
    precomputed_patch_pos_embeds = tokenized_data.get("_openpilot_precomputed_patch_pos_embeds")
    vlm_visual = getattr(vlm_model, "visual", None)
    has_precomputed = (
      precomputed_image_features is not None or precomputed_video_features is not None or precomputed_patch_pos_embeds is not None
    )
    precomputed_lock_acquired = False
    original_patch_pos_embed_cache = None
    original_image_cache = None
    original_video_cache = None
    if has_precomputed:
      self._vlm_precomputed_lock.acquire()
      precomputed_lock_acquired = True
      original_patch_pos_embed_cache = (
        getattr(vlm_visual, "_openpilot_capture_precomputed_fast_pos_embeds", None) if vlm_visual is not None else None
      )
      original_image_cache = getattr(vlm_model, "_openpilot_capture_precomputed_image_features", None)
      original_video_cache = getattr(vlm_model, "_openpilot_capture_precomputed_video_features", None)
      if precomputed_patch_pos_embeds is not None and vlm_visual is not None:
        vlm_visual._openpilot_capture_precomputed_fast_pos_embeds = precomputed_patch_pos_embeds
      if precomputed_image_features is not None:
        vlm_model._openpilot_capture_precomputed_image_features = precomputed_image_features
      if precomputed_video_features is not None:
        vlm_model._openpilot_capture_precomputed_video_features = precomputed_video_features
      model_inputs = dict(model_inputs)
      model_inputs["tokenized_data"] = tokenized_data

    streaming_context_active = self.config.streaming_vision_cache
    if streaming_context_active:
      _STREAMING_VISION_PATCH_CONTEXT.context = {
        "cache": self._streaming_vision_cache,
        "runtime_profile": runtime_profile,
        "token_blocks": self._streaming_vision_cache.last_token_blocks(),
      }
    dflash_runtime_active = self._dflash_runtime_enabled(runtime_profile)
    dflash_dispatch_do_sample = bool(do_sample)
    if dflash_runtime_active and manual_generation:
      # Alpamayo's manual-generation entry point rejects do_sample=True before
      # the DFlash monkey patch is invoked. Force deterministic dispatch through
      # that gate; the DFlash adapter still uses the configured sampling
      # temperature internally.
      dflash_dispatch_do_sample = False
      runtime_profile["dflash_manual_dispatch_forced_do_sample_false"] = 1
    try:
      try:
        with self._torch.no_grad():
          result = self._model.sample_trajectories_from_data_with_vlm_rollout(
            data=self._rollout_model_inputs(model_inputs),
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=return_extra,
            diffusion_kwargs=diffusion_kwargs,
            runtime_profile=runtime_profile,
            do_sample=dflash_dispatch_do_sample,
            manual_generation=manual_generation,
            skip_vlm_generation=skip_vlm_generation,
            _skip_input_deepcopy=True,
          )
      except Exception as exc:
        if not dflash_runtime_active:
          raise
        dflash_error = f"{type(exc).__name__}: {exc}"
        runtime_profile["dflash_error"] = dflash_error
        runtime_profile["dflash_fallback_to_base"] = 1
        if self.config.static_graph_strict_shapes and self.config.dflash_enabled:
          runtime_profile["dflash_strict_fallback_blocked"] = 1
          raise RuntimeError(
            "DFlash generation unavailable under strict static graph mode: " + dflash_error
          ) from exc
        self._disable_dflash_runtime(dflash_error)
        with self._torch.no_grad():
          result = self._model.sample_trajectories_from_data_with_vlm_rollout(
            data=self._rollout_model_inputs(model_inputs),
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=return_extra,
            diffusion_kwargs=diffusion_kwargs,
            runtime_profile=runtime_profile,
            do_sample=do_sample,
            manual_generation=self.config.manual_generation,
            skip_vlm_generation=skip_vlm_generation,
            _skip_input_deepcopy=True,
          )
    finally:
      if streaming_context_active and hasattr(_STREAMING_VISION_PATCH_CONTEXT, "context"):
        delattr(_STREAMING_VISION_PATCH_CONTEXT, "context")
      if hooked_action_to_traj and action_space is not None and original_action_to_traj is not None:
        try:
          action_space.action_to_traj = original_action_to_traj
        except Exception:
          pass
      if has_precomputed:
        if precomputed_patch_pos_embeds is not None and vlm_visual is not None:
          if original_patch_pos_embed_cache is None:
            if hasattr(vlm_visual, "_openpilot_capture_precomputed_fast_pos_embeds"):
              delattr(vlm_visual, "_openpilot_capture_precomputed_fast_pos_embeds")
          else:
            vlm_visual._openpilot_capture_precomputed_fast_pos_embeds = original_patch_pos_embed_cache
        if precomputed_image_features is not None:
          if original_image_cache is None:
            if hasattr(vlm_model, "_openpilot_capture_precomputed_image_features"):
              delattr(vlm_model, "_openpilot_capture_precomputed_image_features")
          else:
            vlm_model._openpilot_capture_precomputed_image_features = original_image_cache
        if precomputed_video_features is not None:
          if original_video_cache is None:
            if hasattr(vlm_model, "_openpilot_capture_precomputed_video_features"):
              delattr(vlm_model, "_openpilot_capture_precomputed_video_features")
          else:
            vlm_model._openpilot_capture_precomputed_video_features = original_video_cache
        if precomputed_lock_acquired:
          self._vlm_precomputed_lock.release()
    if return_extra:
      pred_xyz, pred_rot, extra = result
      if runtime_profile.get("generated_sequence_length") is not None and not extra.get("generated_sequence_length"):
        generated_sequence_length = runtime_profile.get("generated_sequence_length")
        if isinstance(generated_sequence_length, (int, float, np.integer, np.floating)):
          extra["generated_sequence_length"] = int(generated_sequence_length)
      return pred_xyz, pred_rot, extra

    pred_xyz, pred_rot = result
    generated_sequence_length = runtime_profile.get("generated_sequence_length", 0)
    if not isinstance(generated_sequence_length, (int, float, np.integer, np.floating)):
      extra = {}
    else:
      extra = {"generated_sequence_length": int(generated_sequence_length)}
    return pred_xyz, pred_rot, extra

  def _run_inference_with_cuda_graph(
    self,
    model_inputs: dict[str, Any],
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    diffusion_kwargs: dict[str, Any],
    runtime_profile: dict[str, float | int],
    do_sample: bool,
    manual_generation: bool,
    skip_vlm_generation: bool,
    return_extra: bool,
  ) -> tuple[Any, Any, dict[str, Any]] | None:
    assert self._torch is not None
    model_inputs = self._model_inputs_without_tokenized_metadata(model_inputs)
    if not self._supports_cuda_graph:
      runtime_profile["cuda_graph_mode"] = "disabled"
      if self.config.cuda_graphs:
        runtime_profile["cuda_graph_error"] = "unsupported_cuda_graph_configuration"
      else:
        runtime_profile["cuda_graph_error"] = "cuda_graphs_disabled"
      return None
    if not skip_vlm_generation and not self.config.cuda_graph_capture_vlm_decode:
      runtime_profile["cuda_graph_mode"] = "decode_capture_disabled"
      runtime_profile["cuda_graph_error"] = "python_vlm_decode_capture_disallowed"
      runtime_profile["cuda_graph_stage_coverage"] = "none"
      runtime_profile["cuda_graph_decode_capture_allowed"] = 0
      return None
    runtime_profile["cuda_graph_decode_capture_allowed"] = 1 if self.config.cuda_graph_capture_vlm_decode else 0
    if not self._torch.cuda.is_available() or not hasattr(self._torch.cuda, "graph"):
      runtime_profile["cuda_graph_mode"] = "disabled"
      runtime_profile["cuda_graph_error"] = "torch_cuda_graph_unavailable"
      return None
    graph_key = self._build_infer_graph_signature(
      model_inputs=model_inputs,
      top_p=top_p,
      temperature=temperature,
      num_traj_samples=num_traj_samples,
      max_generation_length=max_generation_length,
      diffusion_kwargs=diffusion_kwargs,
      do_sample=do_sample,
      manual_generation=manual_generation,
      skip_vlm_generation=skip_vlm_generation,
      return_extra=return_extra,
    )
    current_signature = _tensor_tree_signature(self._torch, model_inputs)

    cached = self._cuda_graph_cache.get(graph_key)
    if cached is not None and cached.input_signature != current_signature:
      self._cuda_graph_cache.pop(graph_key, None)
      cached = None

    if cached is not None:
      if not _copy_to_tensor_tree(self._torch, cached.buffered_inputs, model_inputs):
        self._cuda_graph_cache.pop(graph_key, None)
        runtime_profile["cuda_graph_mode"] = "copy_failed"
        return None
      try:
        cached.graph.replay()
        self._cuda_graph_cache.move_to_end(graph_key)
        runtime_profile["cuda_graph_mode"] = "replayed"
        _copy_tensor_tree_profile(runtime_profile, cached.runtime_profile)
        runtime_profile["cuda_graph_stage_coverage"] = (
          "full" if _has_full_pipeline_profile(cached.runtime_profile) else "partial"
        )
        if not _has_full_pipeline_profile(cached.runtime_profile):
          runtime_profile["cuda_graph_mode"] = "replay_incomplete"
          return None
        runtime_profile["cuda_graph_replayed"] = 1
        return cached.outputs
      except Exception as exc:
        self._cuda_graph_cache.pop(graph_key, None)
        runtime_profile["cuda_graph_mode"] = "replay_failed"
        runtime_profile["cuda_graph_error"] = type(exc).__name__
        return None

    buffered_inputs = _clone_tensor_tree(self._torch, model_inputs)
    if buffered_inputs is None or not _copy_to_tensor_tree(self._torch, buffered_inputs, model_inputs):
      runtime_profile["cuda_graph_mode"] = "buffered_input_init_failed"
      return None

    graph = self._torch.cuda.CUDAGraph()
    graph_profile: dict[str, float | int] = {}
    try:
      with self._torch.cuda.graph(graph):
        outputs = self._sample_with_model(
          model_inputs=buffered_inputs,
          top_p=top_p,
          temperature=temperature,
          num_traj_samples=num_traj_samples,
          max_generation_length=max_generation_length,
          diffusion_kwargs=diffusion_kwargs,
          runtime_profile=graph_profile,
          do_sample=do_sample,
          manual_generation=manual_generation,
          skip_vlm_generation=skip_vlm_generation,
          return_extra=return_extra,
        )
      cached = _CachedCudaGraph(
        key=graph_key,
        graph=graph,
        buffered_inputs=buffered_inputs,
        outputs=outputs,
        input_signature=current_signature,
        runtime_profile=graph_profile,
      )
      self._cuda_graph_cache[graph_key] = cached
      self._cuda_graph_cache.move_to_end(graph_key)
      while len(self._cuda_graph_cache) > self._cuda_graph_cache_size:
        self._cuda_graph_cache.popitem(last=False)
      runtime_profile["cuda_graph_mode"] = "captured"
      runtime_profile["cuda_graph_stage_coverage"] = (
        "full" if _has_full_pipeline_profile(graph_profile) else "partial"
      )
      _copy_tensor_tree_profile(runtime_profile, graph_profile)
      runtime_profile["cuda_graph_captured"] = 1
      if runtime_profile["cuda_graph_stage_coverage"] != "full":
        self._cuda_graph_cache.pop(graph_key, None)
        runtime_profile["cuda_graph_mode"] = "capture_incomplete"
        runtime_profile["cuda_graph_error"] = "missing_full_pipeline_profile"
        return None
      return outputs
    except Exception as exc:
      runtime_profile["cuda_graph_mode"] = "capture_failed"
      runtime_profile["cuda_graph_error"] = type(exc).__name__
      return None

  def _run_inference_with_runtime_backend(
    self,
    model_inputs: dict[str, Any],
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    diffusion_kwargs: dict[str, Any],
    runtime_profile: dict[str, float | int],
    do_sample: bool,
    manual_generation: bool,
    skip_vlm_generation: bool,
    return_extra: bool,
  ) -> tuple[Any, Any, dict[str, Any]] | None:
    backend = self.config.vlm_runtime_backend.strip().lower()
    runtime_profile["vlm_runtime_backend"] = backend
    if backend in ("", "torch", "pytorch"):
      runtime_profile["vlm_backend_mode"] = "torch_native"
      return None
    if backend == "vllm":
      runtime_profile["vlm_backend_mode"] = "vllm_not_supported"
      runtime_profile["vlm_backend_fallback_to_torch"] = 0
      raise RuntimeError(
        "ALPAMAYO_VLM_RUNTIME_BACKEND=vllm is not production-wired: it does not return the "
        "generated_sequences + HF prompt_cache contract required by the Alpamayo action expert"
      )
    runtime_profile["vlm_backend_mode"] = "unsupported"
    runtime_profile["vlm_backend_fallback_to_torch"] = 1
    runtime_profile["vlm_backend_unavailable_reason"] = f"unsupported backend: {backend}"
    return None

  @staticmethod
  def _warm_overlap_ratio(cache_stats: dict[str, Any]) -> float:
    ratios: list[float] = []
    try:
      ratios.append(float(cache_stats.get("stream_overlap_ratio", 0.0) or 0.0))
    except Exception:
      pass

    prefix_stats = cache_stats.get("vlmPrefixCache", {})
    if isinstance(prefix_stats, dict) and int(prefix_stats.get("hit", 0) or 0):
      ratios.append(1.0)

    streaming_stats = cache_stats.get("streamingVisionCache", {})
    if isinstance(streaming_stats, dict):
      hit_count = 0
      miss_count = 0
      for key in ("hits", "cache_hits", "retained_frames", "retainedFrames"):
        try:
          hit_count += int(streaming_stats.get(key, 0) or 0)
        except Exception:
          pass
      for key in ("misses", "cache_misses", "new_frames", "newFrames", "stale_entries", "staleEntries"):
        try:
          miss_count += int(streaming_stats.get(key, 0) or 0)
        except Exception:
          pass
      total = hit_count + miss_count
      if total > 0:
        ratios.append(float(hit_count) / float(total))

    if not ratios:
      return 0.0
    return max(0.0, min(1.0, max(ratios)))

  def _build_diffusion_kwargs(
    self,
    cache_stats: dict[str, Any],
    runtime_profile: dict[str, Any],
    force_vlm_refresh: bool | None = None,
  ) -> dict[str, Any]:
    base_steps = max(1, int(self.config.diffusion_steps))
    kwargs: dict[str, Any] = {"inference_step": base_steps}
    overlap_ratio = self._warm_overlap_ratio(cache_stats)
    force_vlm_refresh = bool(getattr(self, "_openpilot_force_vlm_refresh", False) if force_vlm_refresh is None else force_vlm_refresh)
    runtime_profile["adaptive_flow_enabled"] = 1 if self.config.adaptive_flow_enabled else 0
    runtime_profile["adaptive_flow_base_steps"] = base_steps
    runtime_profile["adaptive_flow_selected_steps"] = base_steps
    runtime_profile["adaptive_flow_schedule"] = self.config.adaptive_flow_schedule
    runtime_profile["adaptive_flow_overlap_ratio"] = overlap_ratio
    runtime_profile["adaptive_flow_force_refresh_full_diffusion"] = 1 if force_vlm_refresh else 0
    if not self.config.adaptive_flow_enabled:
      runtime_profile["adaptive_flow_mode"] = "disabled"
      return kwargs

    min_steps = max(1, min(base_steps, int(self.config.adaptive_flow_min_steps)))
    selected_steps = base_steps if force_vlm_refresh else (min_steps if overlap_ratio >= float(self.config.adaptive_flow_overlap_threshold) else base_steps)
    graphable_one_step = selected_steps <= 1 and selected_steps < base_steps
    reuse_middle_velocity = bool(self.config.adaptive_flow_reuse_middle_velocity and not force_vlm_refresh)
    reuse_initial_noise = bool(self.config.adaptive_flow_reuse_initial_noise and not force_vlm_refresh)
    action_cache_reuse = bool(self.config.adaptive_flow_action_cache_reuse and not graphable_one_step and not force_vlm_refresh)
    prefix_stats = cache_stats.get("vlmPrefixCache", {}) if isinstance(cache_stats, dict) else {}
    streaming_mode = str(prefix_stats.get("streamingReuseMode", "")) if isinstance(prefix_stats, dict) else ""
    streaming_shifted_noise_key = bool(
      isinstance(prefix_stats, dict)
      and (
        bool(prefix_stats.get("streamingReuseHit", False))
        or streaming_mode in ("trusted_full_replay_no_reasoning", "shifted_kv_current_state_suffix")
      )
    )
    if streaming_shifted_noise_key:
      reuse_middle_velocity = False
      action_cache_reuse = False
    kwargs["inference_step"] = selected_steps
    kwargs["adaptive_flow_schedule"] = self.config.adaptive_flow_schedule
    kwargs["adaptive_flow_reuse_middle_velocity"] = reuse_middle_velocity
    kwargs["adaptive_flow_reuse_initial_noise"] = reuse_initial_noise
    kwargs["adaptive_flow_action_cache_reuse"] = action_cache_reuse
    kwargs["adaptive_flow_overlap_ratio"] = overlap_ratio
    kwargs["adaptive_flow_streaming_shifted_noise_key"] = streaming_shifted_noise_key
    runtime_profile["adaptive_flow_selected_steps"] = selected_steps
    runtime_profile["adaptive_flow_min_steps"] = min_steps
    runtime_profile["adaptive_flow_overlap_threshold"] = float(self.config.adaptive_flow_overlap_threshold)
    runtime_profile["adaptive_flow_reuse_middle_velocity"] = 1 if reuse_middle_velocity else 0
    runtime_profile["adaptive_flow_reuse_initial_noise"] = 1 if reuse_initial_noise else 0
    runtime_profile["adaptive_flow_action_cache_reuse"] = 1 if action_cache_reuse else 0
    runtime_profile["adaptive_flow_streaming_shifted_noise_key"] = 1 if streaming_shifted_noise_key else 0
    runtime_profile["adaptive_flow_streaming_reuse_mode"] = streaming_mode
    runtime_profile["adaptive_flow_graphable_one_step"] = 1 if graphable_one_step else 0
    runtime_profile["adaptive_flow_mode"] = (
      "overlap_reduced_steps_graphable"
      if graphable_one_step
      else ("overlap_reduced_steps" if selected_steps != base_steps else "base_steps")
    )
    return kwargs

  def _adaptive_flow_cache_key(
    self,
    cache_stats: dict[str, Any],
    diffusion_kwargs: dict[str, Any],
  ) -> tuple[Any, ...]:
    streaming_stats = cache_stats.get("streamingVisionCache", {})
    token_blocks = streaming_stats.get("token_blocks", []) if isinstance(streaming_stats, dict) else []
    block_signature = tuple(
      (
        str(block.get("stream", "")),
        int(block.get("frame_index", -1)),
        _freeze_cache_key_value(block.get("signature", ())),
        int(block.get("token_start", -1)),
        int(block.get("token_count", -1)),
      )
      for block in token_blocks
      if isinstance(block, dict)
    )
    inference_step = int(diffusion_kwargs.get("inference_step", self.config.diffusion_steps))
    overlap_ratio = float(diffusion_kwargs.get("adaptive_flow_overlap_ratio", 0.0) or 0.0)
    if bool(diffusion_kwargs.get("adaptive_flow_streaming_shifted_noise_key", False)):
      return (
        "adaptive_flow_v1_streaming_shifted_noise",
        str(self.config.target_model),
        tuple(self.config.camera_streams),
        int(self.config.num_frames),
        str(diffusion_kwargs.get("adaptive_flow_schedule", "uniform")),
        inference_step,
      )
    if inference_step <= 1 and overlap_ratio >= float(self.config.adaptive_flow_overlap_threshold):
      return (
        "adaptive_flow_v1_warm_one_step_coarse",
        str(self.config.target_model),
        tuple(self.config.camera_streams),
        int(self.config.num_frames),
        str(diffusion_kwargs.get("adaptive_flow_schedule", "uniform")),
        inference_step,
        round(overlap_ratio, 2),
      )
    return (
      "adaptive_flow_v1",
      str(self.config.target_model),
      tuple(self.config.camera_streams),
      int(self.config.num_frames),
      str(diffusion_kwargs.get("adaptive_flow_schedule", "uniform")),
      inference_step,
      round(overlap_ratio, 6),
      block_signature,
    )

  def _record_adaptive_flow_cache_candidate(
    self,
    cache_stats: dict[str, Any],
    diffusion_kwargs: dict[str, Any],
    runtime_profile: dict[str, Any],
  ) -> None:
    if not self.config.adaptive_flow_enabled or not (
      self.config.adaptive_flow_reuse_middle_velocity
      or self.config.adaptive_flow_reuse_initial_noise
      or self.config.adaptive_flow_action_cache_reuse
    ):
      runtime_profile["adaptive_flow_cache_enabled"] = 0
      return

    cache_key = self._adaptive_flow_cache_key(cache_stats, diffusion_kwargs)
    diffusion_kwargs["adaptive_flow_cache_key"] = cache_key
    runtime_profile["adaptive_flow_cache_enabled"] = 1
    runtime_profile["adaptive_flow_cache_key_size"] = len(cache_key)
    runtime_profile["adaptive_flow_cache_key_mode"] = str(cache_key[0]) if cache_key else ""
    if self._model is not None:
      self._model._openpilot_adaptive_flow_cache = self._adaptive_flow_cache
    with self._cache_lock:
      cached = self._adaptive_flow_cache.get(cache_key)
      if cached is not None:
        self._adaptive_flow_cache.move_to_end(cache_key)
        cached["hits"] = int(cached.get("hits", 0)) + 1
        runtime_profile["adaptive_flow_cache_hit"] = 1
        runtime_profile["adaptive_flow_cache_miss"] = 0
        runtime_profile["adaptive_flow_cache_has_action_state"] = 1 if cached.get("has_action_state") else 0
        runtime_profile["adaptive_flow_cache_has_velocity_state"] = 1 if cached.get("has_velocity_state") else 0
        runtime_profile["adaptive_flow_cache_has_noise_state"] = 1 if cached.get("has_noise_state") else 0
      else:
        self._adaptive_flow_cache[cache_key] = {
          "hits": 0,
          "has_action_state": False,
          "has_velocity_state": False,
          "has_noise_state": False,
          "reason": "metadata_only_until_model_side_adaptive_flow_consumer_is_wired",
        }
        self._adaptive_flow_cache.move_to_end(cache_key)
        while len(self._adaptive_flow_cache) > self._adaptive_flow_cache_max_entries:
          self._adaptive_flow_cache.popitem(last=False)
        runtime_profile["adaptive_flow_cache_hit"] = 0
        runtime_profile["adaptive_flow_cache_miss"] = 1
        runtime_profile["adaptive_flow_cache_has_action_state"] = 0
        runtime_profile["adaptive_flow_cache_has_velocity_state"] = 0
        runtime_profile["adaptive_flow_cache_has_noise_state"] = 0
      runtime_profile["adaptive_flow_cache_depth"] = len(self._adaptive_flow_cache)

  def _collect_paro_runtime_stats(self) -> dict[str, Any]:
    stats: dict[str, Any] = {
      "enabled": bool(self.config.paro_native),
      "requireCudaModules": bool(self.config.paro_require_cuda_modules),
      "activationInt8Requested": bool(self.config.paro_activation_int8),
      "fastPrefillRequested": bool(self.config.paro_fast_prefill),
      "requireFastPrefill": bool(self.config.paro_require_fast_prefill),
      "activationInt8Available": False,
      "fastPrefillAvailable": False,
      "nativeModules": 0,
      "replacedLinearModules": 0,
      "activationInt8ConfiguredModules": 0,
      "activationInt8Modules": 0,
      "fastPrefillModules": 0,
      "marlinRuntimeReadyModules": 0,
      "cudaModules": 0,
      "cpuModules": 0,
      "otherDeviceModules": 0,
      "missingDeviceModules": 0,
      "nonCudaModuleSamples": [],
      "fastPrefillReasons": {},
      "allActivationInt8": False,
      "allFastPrefill": False,
      "allCuda": False,
    }
    if not self.config.paro_native or self._model is None or not hasattr(self._model, "named_modules"):
      return stats

    for module_name, module in self._model.named_modules():
      if not bool(getattr(module, "_openpilot_paro_native", False)):
        continue
      module_type = type(module)
      module_type_text = f"{getattr(module_type, '__module__', '')}.{getattr(module_type, '__name__', '')}".lower()
      stats["nativeModules"] += 1
      stats["replacedLinearModules"] += 1
      marlin_input_dtype = str(getattr(module, "marlin_input_dtype", "")).lower()
      if marlin_input_dtype == "int8":
        stats["activationInt8ConfiguredModules"] += 1
      if bool(getattr(module, "_openpilot_paro_activation_int8_ready", False)):
        stats["activationInt8Available"] = True
        stats["activationInt8Modules"] += 1
      if bool(getattr(module, "_openpilot_paro_marlin_runtime_ready", False)):
        stats["marlinRuntimeReadyModules"] += 1
      if bool(getattr(module, "_openpilot_paro_fast_prefill_ready", False)):
        stats["fastPrefillAvailable"] = True
        stats["fastPrefillModules"] += 1
      fast_prefill_reason = str(getattr(module, "_openpilot_paro_fast_prefill_reason", ""))
      if fast_prefill_reason:
        reasons = stats["fastPrefillReasons"]
        reasons[fast_prefill_reason] = int(reasons.get(fast_prefill_reason, 0)) + 1
      device_text = None
      try:
        for parameter in module.parameters(recurse=True):
          device_text = str(parameter.device)
          break
      except Exception:
        device_text = None
      if device_text is None:
        try:
          for buffer in module.buffers(recurse=True):
            device_text = str(buffer.device)
            break
        except Exception:
          device_text = None

      if device_text is None:
        stats["missingDeviceModules"] += 1
        device_text = "missing"
      elif device_text.startswith("cuda"):
        stats["cudaModules"] += 1
      elif device_text == "cpu":
        stats["cpuModules"] += 1
      else:
        stats["otherDeviceModules"] += 1

      if not device_text.startswith("cuda") and len(stats["nonCudaModuleSamples"]) < 16:
        stats["nonCudaModuleSamples"].append(f"{module_name}:{device_text}:{module_type_text}")

    stats["allCuda"] = bool(stats["nativeModules"] > 0 and stats["cudaModules"] == stats["nativeModules"])
    stats["allActivationInt8"] = bool(
      stats["nativeModules"] > 0 and stats["activationInt8Modules"] == stats["nativeModules"]
    )
    stats["allFastPrefill"] = bool(
      stats["nativeModules"] > 0 and stats["fastPrefillModules"] == stats["nativeModules"]
    )
    return stats

  def _check_static_graph_eligibility(
    self,
    tokenized_data: dict[str, Any],
    runtime_profile: dict[str, Any],
  ) -> bool:
    input_ids = tokenized_data.get("input_ids")
    input_tokens = int(input_ids.shape[1]) if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2 else 0
    image_grid = tokenized_data.get("image_grid_thw")
    video_grid = tokenized_data.get("video_grid_thw")
    visual = getattr(getattr(getattr(self._model, "vlm", None), "model", None), "visual", None)
    visual_config = getattr(visual, "config", None)
    merge_size = int(getattr(visual, "spatial_merge_size", 0) or getattr(visual_config, "spatial_merge_size", 0) or 1)
    merge_area = max(1, merge_size * merge_size)

    visual_tokens = int(tokenized_data.get("_openpilot_visual_token_count", -2) or 0)
    if visual_tokens == -2:
      visual_tokens = 0
      for grid in (image_grid, video_grid):
        if grid is not None and hasattr(grid, "prod"):
          try:
            if getattr(grid, "is_cuda", False):
              visual_tokens = -1
              break
            visual_tokens += int((grid.prod(dim=1) // merge_area).sum().item())
          except Exception:
            visual_tokens = -1
            break

    reasons: list[str] = []
    if not self.config.cuda_graphs:
      reasons.append("cuda_graphs_disabled")
    if self.config.static_graph_max_prompt_tokens > 0 and input_tokens > self.config.static_graph_max_prompt_tokens:
      reasons.append(
        f"prompt_tokens_exceed_cap:{input_tokens}>{self.config.static_graph_max_prompt_tokens}"
      )
    if visual_tokens < 0:
      reasons.append("visual_token_count_unavailable")
    elif self.config.static_graph_max_visual_tokens > 0 and visual_tokens > self.config.static_graph_max_visual_tokens:
      reasons.append(
        f"visual_tokens_exceed_cap:{visual_tokens}>{self.config.static_graph_max_visual_tokens}"
      )

    eligible = not reasons
    runtime_profile["static_graph_strict_shapes"] = 1 if self.config.static_graph_strict_shapes else 0
    runtime_profile["static_graph_input_tokens"] = input_tokens
    runtime_profile["static_graph_visual_tokens"] = visual_tokens
    runtime_profile["static_graph_max_prompt_tokens"] = int(self.config.static_graph_max_prompt_tokens)
    runtime_profile["static_graph_max_visual_tokens"] = int(self.config.static_graph_max_visual_tokens)
    runtime_profile["static_graph_shape_eligible"] = 1 if eligible else 0
    if reasons:
      runtime_profile["static_graph_shape_reject_reasons"] = ",".join(reasons)
    return eligible

  def _record_graph_stage_plan(self, runtime_profile: dict[str, Any]) -> None:
    requested = {
      "visual": bool(self.config.cuda_graphs and self.config.graph_visual_stage),
      "prefill": bool(self.config.cuda_graphs and self.config.graph_prefill_stage),
      "decode": bool(self.config.cuda_graphs and self.config.graph_decode_stage),
      "action": bool(self.config.cuda_graphs and self.config.graph_action_stage),
    }
    runtime_profile["graph_stage_requested"] = {stage: 1 if enabled else 0 for stage, enabled in requested.items()}
    runtime_profile["graph_stage_cache_depths"] = {
      stage: len(cache) for stage, cache in self._cuda_graph_stage_caches.items()
    }
    if self._model is not None:
      self._model._openpilot_graph_stage_requested = requested
      self._model._openpilot_graph_stage_caches = self._cuda_graph_stage_caches
      self._model._openpilot_graph_stage_cache_size = self._cuda_graph_cache_size
      self._model._openpilot_static_graph_strict_shapes = bool(self.config.static_graph_strict_shapes)
      self._model._openpilot_dflash_graph_capture_enabled = bool(self.config.dflash_graph_capture)
      self._model._openpilot_graph_standard_prefill_stage_enabled = bool(self.config.graph_standard_prefill_stage)
      self._model._openpilot_graph_draft_verify_prefill_stage_enabled = bool(self.config.graph_draft_verify_prefill_stage)
    runtime_profile["dflash_graph_capture_enabled"] = 1 if self.config.dflash_graph_capture else 0
    runtime_profile["graph_standard_prefill_stage_enabled"] = 1 if self.config.graph_standard_prefill_stage else 0
    runtime_profile["graph_draft_verify_prefill_stage_enabled"] = 1 if self.config.graph_draft_verify_prefill_stage else 0
    runtime_profile["graph_stage_mode"] = "metadata_only"
    runtime_profile["graph_stage_ready"] = 0

  def _graph_stage_any_enabled(self) -> bool:
    return bool(
      self.config.graph_visual_stage
      or self.config.graph_prefill_stage
      or self.config.graph_decode_stage
      or self.config.graph_action_stage
    )

  def _require_state_fresh_no_reasoning(self) -> bool:
    return bool(
      self.config.require_state_fresh_no_reasoning
      and not bool(getattr(self, "_openpilot_defer_state_fresh_no_reasoning", False))
    )

  def infer(self, request: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    if self.config.skip_vlm_generation:
      raise RuntimeError("skip_vlm_generation is forbidden in local production Alpamayo mode")
    requested_reasoning_mode = self._assert_reasoning_mode(request)
    reasoning_generation_disabled = bool(self.config.disable_reasoning_generation)
    effective_skip_vlm_generation = bool(self.config.skip_vlm_generation)
    effective_max_generation_length = 0 if reasoning_generation_disabled else int(self.config.max_generation_length)
    runtime_config = request.get("runtimeConfig", {})
    force_vlm_refresh = bool(
      isinstance(runtime_config, dict)
      and str(runtime_config.get("alpamayoForceVlmRefresh", runtime_config.get("forceVlmRefresh", ""))).strip().lower()
      in ("1", "true", "yes", "on")
    )
    defer_state_fresh_no_reasoning = bool(
      isinstance(runtime_config, dict)
      and str(runtime_config.get("alpamayoDeferStateFreshNoReasoning", "")).strip().lower()
      in ("1", "true", "yes", "on")
    )
    require_fast_no_prefill = bool(
      isinstance(runtime_config, dict)
      and str(runtime_config.get("alpamayoRequireFastNoPrefill", "")).strip().lower()
      in ("1", "true", "yes", "on")
    )
    setattr(self, "_openpilot_defer_state_fresh_no_reasoning", defer_state_fresh_no_reasoning)
    setattr(self, "_openpilot_require_fast_no_prefill", require_fast_no_prefill)

    self._ensure_loaded()
    self._ensure_dflash_loaded()
    assert self._torch is not None and self._model is not None
    try:
      setattr(self._model, "_openpilot_require_fast_no_prefill", require_fast_no_prefill)
    except Exception:
      pass
    torch = self._torch
    runtime_profile: dict[str, float | int] = {}
    runtime_profile["reasoning_generation_disabled"] = 1 if reasoning_generation_disabled else 0
    runtime_profile["vlm_autoregressive_generation_skipped"] = 1 if reasoning_generation_disabled else 0
    runtime_profile["effective_max_generation_length"] = int(effective_max_generation_length)
    runtime_profile["defer_state_fresh_no_reasoning"] = 1 if defer_state_fresh_no_reasoning else 0
    runtime_profile["require_fast_no_prefill"] = 1 if require_fast_no_prefill else 0
    paro_runtime_stats = self._collect_paro_runtime_stats()
    if not self.config.paro_native:
      if self.config.paro_require_cuda_modules:
        raise RuntimeError("ALPAMAYO_PARO_REQUIRE_CUDA_MODULES is set but ALPAMAYO_PARO_NATIVE is disabled")
      if self.config.paro_require_fast_prefill:
        raise RuntimeError("ALPAMAYO_PARO_REQUIRE_FAST_PREFILL is set but ALPAMAYO_PARO_NATIVE is disabled")
      if self.config.paro_activation_int8:
        raise RuntimeError("ALPAMAYO_PARO_ACTIVATION_INT8 is set but ALPAMAYO_PARO_NATIVE is disabled")
      if self.config.paro_fast_prefill:
        raise RuntimeError("ALPAMAYO_PARO_FAST_PREFILL is set but ALPAMAYO_PARO_NATIVE is disabled")
    if self.config.paro_require_cuda_modules and self.config.paro_native:
      if int(paro_runtime_stats.get("nativeModules", 0)) <= 0:
        raise RuntimeError("ALPAMAYO_PARO_REQUIRE_CUDA_MODULES is set but no native Paro/Marlin modules were found")
      if not bool(paro_runtime_stats.get("allCuda", False)):
        raise RuntimeError(
          "ALPAMAYO_PARO_REQUIRE_CUDA_MODULES is set but some native Paro/Marlin modules are not on CUDA"
        )
    if self.config.paro_require_fast_prefill and self.config.paro_native:
      if not bool(paro_runtime_stats.get("allFastPrefill", False)):
        raise RuntimeError("ALPAMAYO_PARO_REQUIRE_FAST_PREFILL is set but not all native Paro modules are fast-prefill-ready")
    if self.config.paro_activation_int8 and self.config.paro_native:
      if not bool(paro_runtime_stats.get("allActivationInt8", False)):
        raise RuntimeError("ALPAMAYO_PARO_ACTIVATION_INT8 is set but not all native Paro modules are activation-INT8-ready")
    setattr(self, "_openpilot_force_vlm_refresh", force_vlm_refresh)
    try:
      model_inputs, cache_stats = self._build_model_inputs(request)
    finally:
      setattr(self, "_openpilot_force_vlm_refresh", False)
    if require_fast_no_prefill:
      prefix_stats = cache_stats.get("vlmPrefixCache", {}) if isinstance(cache_stats, dict) else {}
      prefix_hit = bool(isinstance(prefix_stats, dict) and int(prefix_stats.get("hit", 0) or 0) == 1)
      streaming_mode = str(prefix_stats.get("streamingReuseMode", "")) if isinstance(prefix_stats, dict) else ""
      full_generation_ready = bool(isinstance(prefix_stats, dict) and prefix_stats.get("fullGenerationReady"))
      shifted_plan = prefix_stats.get("shiftedPromptKvReusePlan", {}) if isinstance(prefix_stats, dict) else {}
      draft_verify_fast_ready = bool(
        streaming_mode == "draft_verify"
        and isinstance(shifted_plan, dict)
        and shifted_plan.get("valid")
      )
      no_prefill_ready = bool(
        prefix_hit
        and (
          full_generation_ready
          or streaming_mode in ("trusted_full_replay_no_reasoning", "shifted_kv_current_state_suffix")
          or draft_verify_fast_ready
        )
      )
      runtime_profile["fast_no_prefill_prefix_hit"] = 1 if prefix_hit else 0
      runtime_profile["fast_no_prefill_full_generation_ready"] = 1 if full_generation_ready else 0
      runtime_profile["fast_no_prefill_draft_verify_ready"] = 1 if draft_verify_fast_ready else 0
      runtime_profile["fast_no_prefill_ready"] = 1 if no_prefill_ready else 0
      if not no_prefill_ready:
        reason = str(prefix_stats.get("reason", "missing_prefix_cache")) if isinstance(prefix_stats, dict) else "missing_prefix_cache"
        raise RuntimeError(f"alpamayo_fast_no_prefill_required: {reason}")
    visual_input_shape = self._assert_visual_inputs(model_inputs["tokenized_data"], torch)
    prompt_token_count = int(model_inputs["tokenized_data"]["input_ids"].shape[1])
    static_graph_eligible = self._check_static_graph_eligibility(model_inputs["tokenized_data"], runtime_profile)
    self._record_graph_stage_plan(runtime_profile)
    if self.config.cuda_graphs and self.config.static_graph_strict_shapes and not static_graph_eligible:
      raise RuntimeError(
        "CUDA graph static shape guard failed: "
        f"{runtime_profile.get('static_graph_shape_reject_reasons', 'unknown')}"
      )
    diffusion_kwargs = self._build_diffusion_kwargs(cache_stats, runtime_profile, force_vlm_refresh=force_vlm_refresh)
    self._record_adaptive_flow_cache_candidate(cache_stats, diffusion_kwargs, runtime_profile)

    autocast_dtype = _torch_dtype(torch, self.config.autocast_dtype)
    do_sample = not self.config.greedy
    generation_top_p = 1.0 if self.config.greedy else 0.98
    generation_temperature = 0.0 if self.config.greedy else 0.6
    dflash_runtime_enabled = self._dflash_runtime_enabled(runtime_profile)
    graph_stage_any_enabled = self._graph_stage_any_enabled()
    runtime_profile["graph_stage_any_enabled"] = 1 if graph_stage_any_enabled else 0
    runtime_manual_generation = (
      dflash_runtime_enabled
      or self.config.manual_generation
      or (self.config.cuda_graphs and self.config.graph_decode_stage and not do_sample)
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
      outputs = self._run_inference_with_runtime_backend(
        model_inputs=model_inputs,
        top_p=generation_top_p,
        temperature=generation_temperature,
        num_traj_samples=1,
        max_generation_length=effective_max_generation_length,
        diffusion_kwargs=diffusion_kwargs,
        runtime_profile=runtime_profile,
        do_sample=do_sample,
        manual_generation=runtime_manual_generation,
        skip_vlm_generation=effective_skip_vlm_generation,
        return_extra=False,
      )
      if (
        outputs is None
        and self.config.cuda_graphs
        and graph_stage_any_enabled
        and (not dflash_runtime_enabled or self.config.dflash_graph_capture)
      ):
        outputs = self._run_inference_with_cuda_graph(
        model_inputs=model_inputs,
        top_p=generation_top_p,
        temperature=generation_temperature,
        num_traj_samples=1,
        max_generation_length=effective_max_generation_length,
        diffusion_kwargs=diffusion_kwargs,
        runtime_profile=runtime_profile,
        do_sample=do_sample,
        manual_generation=runtime_manual_generation,
        skip_vlm_generation=effective_skip_vlm_generation,
        return_extra=False,
        )
      elif outputs is None and self.config.cuda_graphs and dflash_runtime_enabled and not self.config.dflash_graph_capture:
        runtime_profile["cuda_graph_mode"] = "dflash_graph_disabled"
        runtime_profile["cuda_graph_error"] = "dflash_runtime_non_graph_compatibility_mode"
        runtime_profile["cuda_graph_stage_coverage"] = "none"
        runtime_profile["dflash_graph_path_disabled"] = 1
      elif outputs is None and self.config.cuda_graphs:
        runtime_profile["cuda_graph_mode"] = "stage_disabled"
        runtime_profile["cuda_graph_error"] = "no_graph_stage_enabled"
        runtime_profile["cuda_graph_stage_coverage"] = "none"
      if outputs is None:
        if self.config.cuda_graphs:
          cuda_graph_mode = runtime_profile.get("cuda_graph_mode", "unknown")
          cuda_graph_error = runtime_profile.get("cuda_graph_error")
          runtime_profile["cuda_graph_full_pipeline_fallback_to_partitioned"] = 1
          runtime_profile["cuda_graph_full_pipeline_fallback_reason"] = (
            f"mode={cuda_graph_mode}, error={cuda_graph_error}"
          )
        outputs = self._sample_with_model(
          model_inputs=model_inputs,
          top_p=generation_top_p,
          temperature=generation_temperature,
          num_traj_samples=1,
          max_generation_length=effective_max_generation_length,
          diffusion_kwargs=diffusion_kwargs,
          runtime_profile=runtime_profile,
          do_sample=do_sample,
          manual_generation=runtime_manual_generation,
          skip_vlm_generation=effective_skip_vlm_generation,
          return_extra=True,
        )
    if (
      torch.cuda.is_available()
      and isinstance(runtime_profile, dict)
      and bool(runtime_profile.get("alpamayo_sync_timing", False))
    ):
      torch.cuda.synchronize()

    pred_xyz, pred_rot, extra = outputs
    reasoning_new_tokens = self._assert_reasoning_output(extra, prompt_token_count)
    generated_sequence_length = int(extra.get("generated_sequence_length", prompt_token_count))
    final_dflash_runtime_enabled = self._dflash_runtime_enabled(runtime_profile)
    xyz = pred_xyz.detach().float().cpu().numpy()[0, 0, 0]
    rot = pred_rot.detach().float().cpu().numpy()[0, 0, 0]
    debug: dict[str, Any] = {
      "adapterLatencyMs": round((time.perf_counter() - start) * 1000.0, 3),
      "runtimeProfile": runtime_profile,
      "cudaGraphsEnabled": self.config.cuda_graphs,
      "cudaGraphCacheSize": self._cuda_graph_cache_size,
      "cudaGraphCaptureVlmDecode": self.config.cuda_graph_capture_vlm_decode,
      "staticGraphStrictShapes": self.config.static_graph_strict_shapes,
      "staticGraphMaxPromptTokens": self.config.static_graph_max_prompt_tokens,
      "staticGraphMaxVisualTokens": self.config.static_graph_max_visual_tokens,
      "graphVisualStage": self.config.graph_visual_stage,
      "graphPrefillStage": self.config.graph_prefill_stage,
      "graphStandardPrefillStage": self.config.graph_standard_prefill_stage,
      "graphDraftVerifyPrefillStage": self.config.graph_draft_verify_prefill_stage,
      "graphDecodeStage": self.config.graph_decode_stage,
      "graphActionStage": self.config.graph_action_stage,
      "vlmRuntimeBackend": self.config.vlm_runtime_backend,
      "persistentVlmPrefixCacheEnabled": self.config.persistent_vlm_prefix_cache,
      "vlmPrefixCacheDepth": len(self._vlm_prefix_cache),
      "streamingVlmPrefixReuse": self.config.streaming_vlm_prefix_reuse,
      "streamingVlmTrustShiftedDraft": self.config.streaming_vlm_trust_shifted_draft,
      "streamingVlmSourceCacheDraftVerifyUnverified": self.config.streaming_vlm_source_cache_draft_verify_unverified,
      "streamingVlmTrustedReplayRefreshInterval": self.config.streaming_vlm_trusted_replay_refresh_interval,
      "targetModel": str(self.config.target_model),
      "targetModelIdentity": self._target_model_identity,
      "requireFlashVlaTarget": self.config.require_flashvla_target,
      "cameraStreams": list(self.config.camera_streams),
      "numFrames": self.config.num_frames,
      "minPixels": self.config.min_pixels,
      "maxPixels": self.config.max_pixels,
      "processorPixelConfig": self._processor_pixel_config_debug(),
      "diffusionSteps": self.config.diffusion_steps,
      "adaptiveFlowEnabled": self.config.adaptive_flow_enabled,
      "adaptiveFlowSchedule": self.config.adaptive_flow_schedule,
      "adaptiveFlowMinSteps": self.config.adaptive_flow_min_steps,
      "adaptiveFlowOverlapThreshold": self.config.adaptive_flow_overlap_threshold,
      "adaptiveFlowReuseMiddleVelocity": self.config.adaptive_flow_reuse_middle_velocity,
      "adaptiveFlowReuseInitialNoise": self.config.adaptive_flow_reuse_initial_noise,
      "adaptiveFlowActionCacheReuse": self.config.adaptive_flow_action_cache_reuse,
      "adaptiveFlowCacheMaxEntries": self.config.adaptive_flow_cache_max_entries,
      "adaptiveFlowCacheDepth": len(self._adaptive_flow_cache),
      "reasoningMode": requested_reasoning_mode,
      "streamingVisionCacheEnabled": self.config.streaming_vision_cache,
      "streamingVisionAttentionMaskEnabled": self.config.streaming_vision_attention_mask,
      "dflashEnabled": self.config.dflash_enabled,
      "dflashLoaded": self._dflash_loaded,
      "dflashRuntimeEnabled": final_dflash_runtime_enabled,
      "dflashRuntimeEnabledAtStart": dflash_runtime_enabled,
      "dflashLoadError": self._dflash_load_error,
      "dflashStickyDisabled": self._dflash_sticky_disabled,
      "dflashDisableCooldownRemaining": self._dflash_disable_cooldown_remaining,
      "dflashRetryCooldownFrames": self.config.dflash_retry_cooldown_frames,
      "dflashGraphCapture": self.config.dflash_graph_capture,
      "dflashDraftModel": str(self.config.dflash_draft_model),
      "dflashLayerIds": list(self._dflash_layer_ids),
      "dflashMinAcceptanceRate": self.config.dflash_min_acceptance_rate,
      "dflashMaxTimeToFirstTokenMs": self.config.dflash_max_time_to_first_token_ms,
      "dflashMaxDecodeMs": self.config.dflash_max_decode_ms,
      "dflashMaxTotalMs": self.config.dflash_max_total_ms,
      "paroNative": self.config.paro_native,
      "paroRequireCudaModules": self.config.paro_require_cuda_modules,
      "paroActivationInt8": self.config.paro_activation_int8,
      "paroFastPrefill": self.config.paro_fast_prefill,
      "paroRequireFastPrefill": self.config.paro_require_fast_prefill,
      "paroRuntimeStats": paro_runtime_stats,
      "modelDtype": self.config.model_dtype,
      "autocastDtype": self.config.autocast_dtype,
      "manualGeneration": runtime_manual_generation,
      "skipVlmGeneration": self.config.skip_vlm_generation,
      "reasoningGenerationDisabled": reasoning_generation_disabled,
      "vlmAutoregressiveGenerationSkipped": reasoning_generation_disabled,
      "configuredMaxGenerationLength": self.config.max_generation_length,
      "effectiveMaxGenerationLength": effective_max_generation_length,
      "forceVlmRefresh": force_vlm_refresh,
      "deferStateFreshNoReasoning": defer_state_fresh_no_reasoning,
      "reasoningGeneratedTokens": reasoning_new_tokens,
      "deepstackInputShapes": visual_input_shape,
    }
    prefix_entry = getattr(self._model, "_openpilot_vlm_prefix_cache_entry", None)
    prefix_debug = cache_stats.get("vlmPrefixCache")
    if isinstance(prefix_entry, dict) and isinstance(prefix_debug, dict):
      prefix_input_seq_len = int(prefix_entry.get("input_seq_len", prefix_entry.get("full_vlm_input_seq_len", 0)) or 0)
      prefix_debug.update({
        "fullGenerationReady": self._vlm_prefix_entry_full_generation_ready(prefix_entry, prefix_input_seq_len),
        "fullGenerationReason": str(prefix_entry.get("full_vlm_reason", "")),
        "fullGenerationHits": int(prefix_entry.get("full_vlm_hits", 0) or 0),
        "fullGenerationStores": int(prefix_entry.get("full_vlm_stores", 0) or 0),
        "streamingReuseHit": bool(prefix_entry.get("streaming_vlm_reuse")),
        "streamingReuseMode": str(prefix_entry.get("streaming_vlm_reuse_mode", "")),
        "streamingReuseUnverified": bool(prefix_entry.get("streaming_vlm_reuse_unverified")),
        "streamingReuseChainDepth": int(prefix_entry.get("streaming_vlm_reuse_chain_depth", 0) or 0),
        "trustedReplayAllowed": bool(prefix_entry.get("streaming_vlm_trusted_replay_allowed")),
        "trustedReplayRequested": bool(prefix_entry.get("streaming_vlm_trusted_replay_requested")),
        "stateFreshNoReasoningRequired": bool(prefix_entry.get("streaming_vlm_state_fresh_no_reasoning_required")),
        "stateFreshNoReasoningForcedPrefill": bool(prefix_entry.get("full_vlm_state_fresh_no_reasoning_forced_prefill")),
        "trustedReplayDisabledForDiffusionFreshness": bool(
          prefix_entry.get("streaming_vlm_trusted_replay_disabled_for_diffusion_freshness")
        ),
        "refreshDue": bool(prefix_entry.get("streaming_vlm_refresh_due")),
        "languageVisualTokenSpans": len(prefix_entry.get("language_visual_token_spans", []) or []),
        "shiftedPromptKvReusePlan": prefix_entry.get("streaming_vlm_draft_shifted_prompt_kv_reuse_plan", {}),
      })
    debug["frameCacheStats"] = cache_stats
    with self._cache_lock:
      debug["tokenizedCacheDepth"] = len(self._tokenized_cache)
      stream_window_cache_depth = 0
      for stream, stream_cache in self._stream_window_cache.items():
        stream_window_cache_depth += len(stream_cache)
      debug["streamWindowCacheDepth"] = stream_window_cache_depth
      debug["frameCacheDepth"] = sum(len(cache) for cache in self._frame_cache.values())
    if getattr(self, "_visual_patch_repair", None) is not None:
      debug["visualPatchEmbedRepair"] = self._visual_patch_repair
    if getattr(self, "_paro_finalize", None) is not None:
      debug["paroFinalizeCount"] = len(self._paro_finalize)
    try:
      cot = extra["cot"][0, 0, 0]
      cot_text = str(cot)
      debug["cotText"] = cot_text
      debug["cotPreview"] = cot_text[:300].replace("\n", " ")
    except Exception:
      pass
    response = semantic_response_from_prediction(
      xyz,
      rot,
      source=self.config.source,
      debug=debug,
    )
    cot_text = debug.get("cotText")
    if isinstance(cot_text, str) and cot_text:
      response["semanticPlan"]["reasoningText"] = cot_text
    return response


def create_adapter() -> LocalAlpamayoAdapter:
  return LocalAlpamayoAdapter()
