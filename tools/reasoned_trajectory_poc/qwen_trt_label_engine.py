#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import math
import random
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _prefer_cuda_13_2() -> None:
  cuda_13 = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
  if cuda_13.exists():
    os.environ["CUDA_PATH"] = str(cuda_13)
    os.environ["CUDA_HOME"] = str(cuda_13)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    cuda_bin = str(cuda_13 / "bin")
    cuda_libnvvp = str(cuda_13 / "libnvvp")
    cuda_cupti = str(cuda_13 / "extras" / "CUPTI" / "lib64")
    for part in (cuda_cupti, cuda_libnvvp, cuda_bin):
      if part not in path_parts:
        path_parts.insert(0, part)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    os.environ.setdefault("CL", "/Zc:preprocessor")


_prefer_cuda_13_2()

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import tensorrt as trt
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.reasoned_trajectory_poc.qwen_label_rtp_worker import (
  CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M,
  CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M,
  CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION,
  CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN,
  CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE,
  CONSTRUCTION_ACTION_IMMEDIATE_MARGIN,
  CONSTRUCTION_ACTION_IMMEDIATE_SCORE,
  CONSTRUCTION_CLEAR_RTP_CONFIDENCE,
  CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES,
  CONSTRUCTION_DIRECT_EDGE_MARGIN,
  CONSTRUCTION_DIRECT_EDGE_SCORE,
  CONSTRUCTION_DIRECT_SEMANTIC_MARGIN,
  CONSTRUCTION_DIRECT_SEMANTIC_SCORE,
  CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN,
  CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX,
  CONSTRUCTION_EDGE_BOOTSTRAP_SCORE,
  CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
  CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE,
  CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
  CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_PRESENCE_CLEAR_CONFIRM_FRAMES,
  CONSTRUCTION_PRESENCE_HOLD_FRAMES,
  CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE,
  CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN,
  CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE,
  CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
  CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M,
  CONSTRUCTION_SIDE_CLEAR_MAX_SCORE,
  CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M,
  CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE,
  CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX,
  CONSTRUCTION_STATE_MACHINE_VERSION,
  DEFAULT_DURABLE_SCORE_LABELS,
  DEFAULT_MODEL_DIR,
  DEFAULT_SCORE_LABEL_GROUPS,
  PATH_AGENT_RTP_VERSION,
  PATH_AGENT_STOP_S,
  PATH_AGENT_YIELD_SPEED_CAP,
  PATH_BLOCKING_AGENT_LABELS,
  PATH_ENTERING_AGENT_LABELS,
  RotatingScoreState,
  SCORE_PROMPT,
  SCORE_QUESTIONS,
  _image_from_payload,
  _inference_images,
  _labels_to_rtp,
  _parse_score_label_groups,
  _resolve_exclusive_labels,
  _score_label_ids,
  _validate_score_labels,
  _with_visual_fallbacks,
)
from selfdrive.controls.reasoned.lead_semantics import (
  classify_lead_track,
  lead_choice_word,
  lead_track_metrics,
)

try:
  from selfdrive.controls.reasoned.ui_scene_board import OverlayGeometry
except Exception:
  OverlayGeometry = None


DEFAULT_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "qwen_trt_export"
DEFAULT_IMAGE = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_construction_loop_proof" / "vlm" / "vlm_input_0020.png"
DEFAULT_SIGNAL_RED_IMAGE = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "traffic_light_visual_probe_clean_signalhead" / "static" / "vlm_input_0000.png"
DEFAULT_SIGNAL_GREEN_IMAGE = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "traffic_light_visual_probe_clean_signalhead_green" / "static" / "vlm_input_0000.png"
DEFAULT_SIGNAL_NONE_IMAGE = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "traffic_light_visual_probe_no_signal" / "static" / "vlm_input_0000.png"
MANIFEST_VERSION = 1
SIGNAL_CLASSES = ("none", "red_stop_light", "green_go_light")
SIGNAL_LABELS = ("red_stop_light", "green_go_light")
CHOICE_WORDS = (
  "red",
  "green",
  "go",
  "clear",
  "stop",
  "left",
  "right",
  "blocked",
  "present",
  "absent",
  "entering",
  "moving",
  "slower",
  "braking",
  "stopped",
  "merge",
  "crossing",
  "irrelevant",
  "blue",
  "purple",
  "orange",
  "cyan",
  "pink",
  "A",
  "B",
  "C",
  "none",
)
CHOICE_WORDS_PRE_ABC_SIGNAL = tuple(word for word in CHOICE_WORDS if word != "C")
CHOICE_WORDS_PRE_GO = tuple(word for word in CHOICE_WORDS_PRE_ABC_SIGNAL if word != "go")
CHOICE_WORDS_NO_GREEN = tuple(word for word in CHOICE_WORDS_PRE_GO if word != "green")
CHOICE_WORDS_V3 = tuple(word for word in CHOICE_WORDS if word not in {"A", "B"})
CHOICE_WORDS_V2 = tuple(word for word in CHOICE_WORDS if word not in {"blue", "orange", "cyan", "pink", "A", "B"})
LEGACY_CHOICE_WORDS = (
  "red",
  "clear",
  "stop",
  "left",
  "right",
  "blocked",
  "entering",
  "none",
)
CHOICE_WORDS_V1 = (
  "red",
  "clear",
  "stop",
  "left",
  "right",
  "blocked",
  "entering",
  "moving",
  "slower",
  "braking",
  "stopped",
  "merge",
  "crossing",
  "irrelevant",
  "none",
)


def _choice_words_for_output_width(width: int) -> tuple[str, ...]:
  for words in (
    CHOICE_WORDS,
    CHOICE_WORDS_PRE_ABC_SIGNAL,
    LEGACY_CHOICE_WORDS,
    CHOICE_WORDS_PRE_GO,
    CHOICE_WORDS_NO_GREEN,
    CHOICE_WORDS_V1,
    CHOICE_WORDS_V3,
    CHOICE_WORDS_V2,
  ):
    if len(words) == width:
      return words
  return ()


CONSTRUCTION_LABELS = ("construction_left", "construction_right")
CONSTRUCTION_EDGE_LABELS = ("construction_blue_edge", "construction_purple_edge")
CONSTRUCTION_ACTION_LABELS = ("construction_drive_left", "construction_drive_right")
CONSTRUCTION_SHIFT_LABELS = ("construction_shift_left", "construction_shift_right")
CONSTRUCTION_CANDIDATE_LABELS = ("construction_blocks_left_candidate", "construction_blocks_right_candidate")
CONSTRUCTION_LABEL_SET = frozenset(CONSTRUCTION_LABELS)
CONSTRUCTION_EDGE_LABEL_SET = frozenset(CONSTRUCTION_EDGE_LABELS)
CONSTRUCTION_ACTION_LABEL_SET = frozenset(CONSTRUCTION_ACTION_LABELS)
CONSTRUCTION_SHIFT_LABEL_SET = frozenset(CONSTRUCTION_SHIFT_LABELS)
CONSTRUCTION_CANDIDATE_LABEL_SET = frozenset(CONSTRUCTION_CANDIDATE_LABELS)
CONSTRUCTION_CANDIDATE_IMMEDIATE_SCORE = 1.5
CONSTRUCTION_CANDIDATE_IMMEDIATE_MARGIN = 1.5
CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN = 1.0
CONSTRUCTION_STATE_LABEL_SET = (
  frozenset(("cones", "barrier")) |
  CONSTRUCTION_LABEL_SET |
  CONSTRUCTION_EDGE_LABEL_SET |
  CONSTRUCTION_ACTION_LABEL_SET |
  CONSTRUCTION_SHIFT_LABEL_SET |
  CONSTRUCTION_CANDIDATE_LABEL_SET
)
PHYSICAL_STATE_LABEL_SET = frozenset((
  "true_moving_lead",
  "slower_lead",
  "braking_lead",
  "stopped_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
  "irrelevant_vehicle",
))
CONSTRUCTION_MIRROR_LABEL = {
  "construction_left": "construction_right",
  "construction_right": "construction_left",
  "construction_blue_edge": "construction_purple_edge",
  "construction_purple_edge": "construction_blue_edge",
  "construction_drive_left": "construction_drive_right",
  "construction_drive_right": "construction_drive_left",
  "construction_shift_left": "construction_shift_right",
  "construction_shift_right": "construction_shift_left",
  "construction_blocks_left_candidate": "construction_blocks_right_candidate",
  "construction_blocks_right_candidate": "construction_blocks_left_candidate",
}
MODEL_CONTRACT_FILES = (
  "config.json",
  "generation_config.json",
  "preprocessor_config.json",
  "tokenizer_config.json",
  "chat_template.json",
  "model.safetensors.index.json",
)


def _vehicle_state_for_labels(args, labels: Sequence[str], payload_vehicle_state: str | None) -> str:
  if not bool(getattr(args, "use_payload_vehicle_state", True)):
    return str(getattr(args, "vehicle_state", ""))
  scope = str(getattr(args, "payload_vehicle_state_scope", "auto"))
  if scope == "none":
    return str(getattr(args, "vehicle_state", ""))
  if scope == "all":
    return str(payload_vehicle_state if payload_vehicle_state is not None else getattr(args, "vehicle_state", ""))
  if PHYSICAL_STATE_LABEL_SET.intersection(labels) or CONSTRUCTION_STATE_LABEL_SET.intersection(labels):
    return str(payload_vehicle_state if payload_vehicle_state is not None else getattr(args, "vehicle_state", ""))
  return str(getattr(args, "vehicle_state", ""))


def percentile(values: Sequence[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
  return float(ordered[idx])


def _torch_attn_implementation(args_or_value) -> str:
  value = args_or_value
  if not isinstance(args_or_value, str):
    value = getattr(args_or_value, "torch_attn_implementation", "sdpa")
  impl = str(value).lower()
  if impl not in ("sdpa", "eager"):
    raise ValueError(f"unknown torch attention implementation: {impl}")
  return impl


def _attn_suffix(args) -> str:
  impl = _torch_attn_implementation(args)
  return "" if impl == "sdpa" else f"_attn_{impl}"


def _load_qwen(model_dir: Path, torch_attn_implementation: str = "sdpa"):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation=_torch_attn_implementation(torch_attn_implementation),
  )
  model.eval()
  return processor, model


def _set_qwen_attention_implementation(model, torch_attn_implementation: str) -> str:
  impl = _torch_attn_implementation(torch_attn_implementation)
  model.config._attn_implementation = impl
  model.model.language_model.config._attn_implementation = impl
  return impl


def _score_labels(raw: str) -> tuple[str, ...]:
  labels = tuple(label.strip() for label in raw.split(",") if label.strip())
  unknown = [label for label in labels if label not in SCORE_QUESTIONS]
  if unknown:
    raise ValueError(f"unknown score labels: {unknown}")
  if not labels:
    raise ValueError("at least one score label is required")
  return labels


def _labels_key(labels: Sequence[str]) -> str:
  return "__".join(label.replace("-", "_").replace("/", "_") for label in labels)


def _text_seq_suffix(args) -> str:
  text_seq_len = int(getattr(args, "text_seq_len", 0))
  return "" if text_seq_len <= 0 else f"_seq{text_seq_len}"


def _text_precision(args) -> str:
  precision = str(getattr(args, "text_precision", "nvfp4")).lower()
  if precision not in ("nvfp4", "fp8", "fp16"):
    raise ValueError(f"unknown text precision: {precision}")
  return precision


def _text_output(args) -> str:
  output = str(getattr(args, "text_output", "logits")).lower()
  if output not in ("logits", "hidden", "full_hidden", "fixed_logits"):
    raise ValueError(f"unknown text output: {output}")
  return output


def _text_output_tensor_name(output: str) -> str:
  output = output.lower()
  if output == "logits":
    return "selected_logits"
  if output == "hidden":
    return "selected_hidden"
  if output == "full_hidden":
    return "full_hidden"
  if output == "fixed_logits":
    return "selected_logits"
  raise ValueError(f"unknown text output: {output}")


def _select_text_output(requested_output: str, tensor_names: set[str]) -> tuple[str, str]:
  requested_output = requested_output.lower()
  preferences = {
    "logits": ("selected_logits", "selected_hidden", "full_hidden"),
    "hidden": ("selected_hidden", "full_hidden", "selected_logits"),
    "full_hidden": ("full_hidden", "selected_hidden", "selected_logits"),
    "fixed_logits": ("selected_logits",),
  }.get(requested_output)
  if preferences is None:
    raise ValueError(f"unknown text output: {requested_output}")
  name_to_output = {
    "selected_logits": "logits",
    "selected_hidden": "hidden",
    "full_hidden": "full_hidden",
  }
  for name in preferences:
    if name in tensor_names:
      if requested_output == "fixed_logits" and name == "selected_logits":
        return "fixed_logits", name
      return name_to_output[name], name
  raise RuntimeError(
    "text TensorRT engine is missing selected output tensor; "
    f"expected one of selected_logits, selected_hidden, or full_hidden, got {sorted(tensor_names)}"
  )


def _label_decision_mode(args) -> str:
  mode = str(getattr(args, "label_decision_mode", "binary")).lower()
  if mode not in ("binary", "choice"):
    raise ValueError(f"unknown label decision mode: {mode}")
  return mode


def _text_position_mode(args) -> str:
  mode = str(getattr(args, "text_position_mode", "auto")).lower()
  if mode not in ("auto", "qwen", "clamp127", "zero"):
    raise ValueError(f"unknown text position mode: {mode}")
  if mode == "auto":
    return "clamp127" if _label_decision_mode(args) == "choice" else "qwen"
  return mode


def _text_position_dtype(args) -> str:
  dtype = str(getattr(args, "text_position_dtype", "int64")).lower()
  if dtype not in ("int64", "int32"):
    raise ValueError(f"unknown text position dtype: {dtype}")
  return dtype


def _apply_text_position_mode(position_ids: torch.Tensor, args) -> torch.Tensor:
  mode = _text_position_mode(args)
  if mode == "qwen":
    out = position_ids.contiguous()
  elif mode == "clamp127":
    out = position_ids.clamp(max=127).contiguous()
  elif mode == "zero":
    out = torch.zeros_like(position_ids).contiguous()
  else:
    raise ValueError(f"unknown text position mode: {mode}")
  if _text_position_dtype(args) == "int32":
    return out.to(torch.int32).contiguous()
  return out.to(torch.int64).contiguous()


def _text_position_dtype_suffix(args) -> str:
  dtype = _text_position_dtype(args)
  return "" if dtype == "int64" else f"_pos_{dtype}"


def _text_engine_dir(args) -> Path:
  precision = _text_precision(args)
  return args.artifact_dir / f"{precision}_trt"


def _vision_precision(args) -> str:
  precision = str(getattr(args, "vision_precision", "fp32")).lower()
  if precision not in ("fp32", "fp16"):
    raise ValueError(f"unknown vision precision: {precision}")
  return precision


def _vision_engine_path(args) -> Path:
  precision = _vision_precision(args)
  return args.artifact_dir / f"vision_static_{precision}" / f"qwen_vision_full{int(args.image_size)}_static_{precision}.engine"


def _vision_onnx_path(args) -> Path:
  precision = _vision_precision(args)
  return args.artifact_dir / f"vision_static_{precision}" / f"qwen_vision_full{int(args.image_size)}_static_{precision}.onnx"


def _signal_head_dir(args) -> Path:
  return args.artifact_dir / "signal_head"


def _signal_head_weights_path(args) -> Path:
  return args.signal_head_weights or (_signal_head_dir(args) / f"qwen_signal_head_full{int(args.image_size)}.pt")


def _signal_head_onnx_path(args) -> Path:
  return _signal_head_dir(args) / f"qwen_signal_head_full{int(args.image_size)}.onnx"


def _signal_head_engine_path(args) -> Path:
  return args.signal_head_engine or (_signal_head_dir(args) / f"qwen_signal_head_full{int(args.image_size)}.engine")


def _signal_head_calibration_path(args) -> Path:
  return _signal_head_dir(args) / f"qwen_signal_head_full{int(args.image_size)}_calibration.json"


def _generic_text_engine_path(args) -> Path:
  precision = _text_precision(args)
  output_suffix = "" if _text_output(args) == "logits" else f"_{_text_output(args)}"
  decision_suffix = "" if _label_decision_mode(args) == "binary" else f"_{_label_decision_mode(args)}"
  prompt_suffix = _score_prompt_suffix(args)
  position_suffix = _text_position_dtype_suffix(args)
  typed_suffix = "_strong" if bool(getattr(args, "text_strongly_typed", False)) else ""
  return _text_engine_dir(args) / f"qwen_text_36layer_{precision}{_text_seq_suffix(args)}{output_suffix}{decision_suffix}{prompt_suffix}{position_suffix}{_attn_suffix(args)}{typed_suffix}_trt.engine"


def _generic_text_onnx_path(args) -> Path:
  precision = _text_precision(args)
  output_suffix = "" if _text_output(args) == "logits" else f"_{_text_output(args)}"
  decision_suffix = "" if _label_decision_mode(args) == "binary" else f"_{_label_decision_mode(args)}"
  prompt_suffix = _score_prompt_suffix(args)
  position_suffix = _text_position_dtype_suffix(args)
  typed_suffix = "_strong" if bool(getattr(args, "text_strongly_typed", False)) else ""
  return _text_engine_dir(args) / f"qwen_text_36layer_{precision}{_text_seq_suffix(args)}{output_suffix}{decision_suffix}{prompt_suffix}{position_suffix}{_attn_suffix(args)}{typed_suffix}_trt.onnx"


def _keyed_text_engine_path(args, labels: Sequence[str]) -> Path:
  precision = _text_precision(args)
  output_suffix = "" if _text_output(args) == "logits" else f"_{_text_output(args)}"
  decision_suffix = "" if _label_decision_mode(args) == "binary" else f"_{_label_decision_mode(args)}"
  prompt_suffix = _score_prompt_suffix(args)
  position_suffix = _text_position_dtype_suffix(args)
  typed_suffix = "_strong" if bool(getattr(args, "text_strongly_typed", False)) else ""
  return _text_engine_dir(args) / f"qwen_text_36layer_{precision}{_text_seq_suffix(args)}{output_suffix}{decision_suffix}{prompt_suffix}{position_suffix}_{_labels_key(labels)}{_attn_suffix(args)}{typed_suffix}_trt.engine"


def _keyed_text_onnx_path(args, labels: Sequence[str]) -> Path:
  precision = _text_precision(args)
  output_suffix = "" if _text_output(args) == "logits" else f"_{_text_output(args)}"
  decision_suffix = "" if _label_decision_mode(args) == "binary" else f"_{_label_decision_mode(args)}"
  prompt_suffix = _score_prompt_suffix(args)
  position_suffix = _text_position_dtype_suffix(args)
  typed_suffix = "_strong" if bool(getattr(args, "text_strongly_typed", False)) else ""
  return _text_engine_dir(args) / f"qwen_text_36layer_{precision}{_text_seq_suffix(args)}{output_suffix}{decision_suffix}{prompt_suffix}{position_suffix}_{_labels_key(labels)}{_attn_suffix(args)}{typed_suffix}_trt.onnx"


def _resolve_text_engine_path(args, labels: Sequence[str], *, require_keyed: bool = False) -> Path:
  if args.text_engine is not None and not require_keyed:
    return args.text_engine
  keyed = _keyed_text_engine_path(args, labels)
  if keyed.exists():
    return keyed
  if require_keyed:
    raise FileNotFoundError(f"missing label-specific TensorRT text engine for {','.join(labels)}: {keyed}")
  return _generic_text_engine_path(args)


def _score_groups(raw: str) -> tuple[tuple[str, ...], ...]:
  groups = _parse_score_label_groups(raw)
  if not groups:
    raise ValueError("at least one score label group is required")
  for group in groups:
    unknown = [label for label in group if label not in SCORE_QUESTIONS]
    if unknown:
      raise ValueError(f"unknown score label group entries: {unknown}")
  return groups


def _image_from_payload_for_labels(payload: dict, labels: Sequence[str]) -> Image.Image:
  label_set = set(labels)
  aux_key = None
  if label_set and label_set.issubset(CONSTRUCTION_CANDIDATE_LABEL_SET):
    if label_set == CONSTRUCTION_CANDIDATE_LABEL_SET:
      aux_key = "candidate_pair"
    elif "construction_blocks_left_candidate" in label_set:
      aux_key = "candidate_left"
    elif "construction_blocks_right_candidate" in label_set:
      aux_key = "candidate_right"
  if aux_key is not None:
    aux = payload.get("scene_board_aux_images_b64")
    if isinstance(aux, dict) and aux.get(aux_key):
      return Image.open(BytesIO(base64.b64decode(str(aux[aux_key])))).convert("RGB")
  return _image_from_payload(payload)


def _construction_text_precision(args) -> str:
  precision = str(getattr(args, "construction_text_precision", "same")).lower()
  if precision not in ("same", "nvfp4", "fp8", "fp16"):
    raise ValueError(f"unknown construction text precision: {precision}")
  return precision


def _args_for_score_group(args, group: Sequence[str]):
  if (
    bool(getattr(args, "construction_side_choice", False)) and
    group and
    set(group).issubset(CONSTRUCTION_LABEL_SET)
  ):
    group_args = argparse.Namespace(**vars(args))
    group_args.label_decision_mode = "choice"
    group_args.score_prompt_mode = "construction_compact"
    construction_engine = getattr(args, "construction_text_engine", None)
    if construction_engine is not None:
      group_args.text_engine = construction_engine
    construction_seq_len = int(getattr(args, "construction_text_seq_len", 0) or 0)
    if construction_seq_len > 0:
      group_args.text_seq_len = construction_seq_len
    construction_precision = _construction_text_precision(args)
    if construction_precision != "same":
      group_args.text_precision = construction_precision
    return group_args
  if (
    bool(getattr(args, "construction_edge_choice", False)) and
    group and
    set(group).issubset(CONSTRUCTION_EDGE_LABEL_SET)
  ):
    group_args = argparse.Namespace(**vars(args))
    group_args.label_decision_mode = "choice"
    group_args.score_prompt_mode = "construction_compact"
    construction_engine = getattr(args, "construction_text_engine", None)
    if construction_engine is not None:
      group_args.text_engine = construction_engine
    construction_seq_len = int(getattr(args, "construction_text_seq_len", 0) or 0)
    if construction_seq_len > 0:
      group_args.text_seq_len = construction_seq_len
    construction_precision = _construction_text_precision(args)
    if construction_precision != "same":
      group_args.text_precision = construction_precision
    return group_args
  if (
    bool(getattr(args, "construction_edge_binary", False)) and
    group and
    set(group).issubset(CONSTRUCTION_EDGE_LABEL_SET)
  ):
    group_args = argparse.Namespace(**vars(args))
    group_args.label_decision_mode = "binary"
    group_args.score_prompt_mode = "construction_compact"
    return group_args
  if (
    bool(getattr(args, "construction_candidate_choice", False)) and
    group and
    set(group).issubset(CONSTRUCTION_CANDIDATE_LABEL_SET)
  ):
    group_args = argparse.Namespace(**vars(args))
    group_args.label_decision_mode = "choice"
    group_args.score_prompt_mode = "construction_compact"
    construction_engine = getattr(args, "construction_text_engine", None)
    if construction_engine is not None:
      group_args.text_engine = construction_engine
    construction_seq_len = int(getattr(args, "construction_text_seq_len", 0) or 0)
    if construction_seq_len > 0:
      group_args.text_seq_len = construction_seq_len
    construction_precision = _construction_text_precision(args)
    if construction_precision != "same":
      group_args.text_precision = construction_precision
    return group_args
  if (
    bool(getattr(args, "construction_candidate_binary", False)) and
    group and
    set(group).issubset(CONSTRUCTION_CANDIDATE_LABEL_SET)
  ):
    group_args = argparse.Namespace(**vars(args))
    group_args.label_decision_mode = "binary"
    group_args.score_prompt_mode = "construction_compact"
    return group_args
  if bool(getattr(args, "construction_candidate_choice", False)):
    return args
  if not _is_construction_group(group):
    return args
  construction_engine = getattr(args, "construction_text_engine", None)
  construction_precision = _construction_text_precision(args)
  if construction_engine is None and construction_precision == "same":
    return args
  group_args = argparse.Namespace(**vars(args))
  if construction_engine is not None:
    group_args.text_engine = construction_engine
  construction_seq_len = int(getattr(args, "construction_text_seq_len", 0) or 0)
  if construction_seq_len > 0:
    group_args.text_seq_len = construction_seq_len
  if construction_precision != "same":
    group_args.text_precision = construction_precision
  return group_args


def _score_prompt_mode(args) -> str:
  mode = str(getattr(args, "score_prompt_mode", "full")).lower().replace("-", "_")
  if mode not in ("full", "worker_full", "construction_compact", "construction_score"):
    raise ValueError(f"unknown score prompt mode: {mode}")
  return mode


def _score_prompt_suffix(args) -> str:
  mode = _score_prompt_mode(args)
  return "" if mode == "full" else f"_{mode}"


def _score_text(label: str, vehicle_state: str, score_prompt_mode: str = "full") -> str:
  if score_prompt_mode == "worker_full":
    return (
      f"{SCORE_PROMPT}\nVehicle state: {vehicle_state}\n"
      f"Question: {SCORE_QUESTIONS[label]}"
    )
  if score_prompt_mode == "construction_score" and label in CONSTRUCTION_LABELS:
    return (
      f"Scored label: {label}\n"
      "Score one driving-scene construction-side label from the driver-view image.\n"
      "The green overlay is the ego planned path and vehicle-width corridor.\n"
      "Answer exactly yes or no.\n"
      "Answer yes only when the question condition is directly visible.\n"
      "Only consider real cones, blue cones, pylons, bollards, barrels, barricades, checker panels, barriers, or blocked-lane panels that overlap the green path, intrude into the green corridor, narrow the green corridor, or block the path ahead.\n"
      "Ignore objects merely visible beside the road, on shoulders, along walls, behind lane lines, far outside the planned path, or only visible as tiny distant horizon dots.\n"
      "Ignore traffic lights, stop signs, UI text, route arrows, lane paint, lane lines, road edges, shadows, poles, and decorative off-path construction objects.\n"
      "construction_left and construction_right mean the side of the green path in the driver-view image, not simulator lane-coordinate sign and not steering direction.\n"
      "If the relevant construction hazard intrudes from image-left / driver-left into the green path, answer yes only to construction_left.\n"
      "If the relevant construction hazard intrudes from image-right / driver-right into the green path, answer yes only to construction_right.\n"
      f"Vehicle state: {vehicle_state}\n"
      f"Question: {SCORE_QUESTIONS[label]}\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in CONSTRUCTION_LABEL_SET:
    side = "image-left / driver-left" if label == "construction_left" else "image-right / driver-right"
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. The green overlay is the actual tracked path and vehicle-width corridor.\n"
      "A thin magenta dashed line, if present, is the unmodified base openpilot path before VLM lateral bias.\n"
      "Image left is driver left. Image right is driver right.\n"
      "Count only real cones, barrels, pylons, bollards, barricades, barriers, or blocked-lane panels that touch, overlap, intrude into, narrow, or block the green corridor.\n"
      "Ignore lane lines, road edges, traffic lights, signs, poles, route arrows, UI text, shadows, walls, and distant off-path background objects.\n"
      f"Question: Is a construction hazard intruding from {side} into the green corridor?\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in {"cones", "barrier"}:
    object_text = (
      "real traffic cone, blue cone, orange cone, pylon, bollard, barrel, or cone-shaped lane marker"
      if label == "cones" else
      "real road barrier, barricade, checker panel, blocked-lane panel, or construction barrier"
    )
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. The translucent green overlay is the actual tracked path and vehicle-width corridor.\n"
      "The blue and purple lines are the corridor edges. Edge-local inset panels, if present, are extra visual evidence from the same camera frame.\n"
      "Count only real construction objects that touch, overlap, intrude into, narrow, or block the green corridor ahead.\n"
      "Ignore lane paint, road edges, shoulders, walls, traffic lights, signs, poles, UI text, colored overlay text, shadows, and tiny distant off-path objects.\n"
      f"Question: Is any {object_text} affecting the green planned corridor?\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in CONSTRUCTION_EDGE_LABEL_SET:
    edge = "blue ego-left corridor edge" if label == "construction_blue_edge" else "purple ego-right corridor edge"
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. The translucent green overlay is the actual tracked path and vehicle-width corridor.\n"
      "The blue line is the ego-left edge of the green corridor. The purple line is the ego-right edge.\n"
      "Count only real cones, barrels, pylons, bollards, barricades, barriers, or blocked-lane panels that touch, overlap, intrude into, narrow, or block the green corridor.\n"
      "Ignore lane paint, road edges, shoulders, walls, traffic lights, signs, poles, UI text, colored overlay text, shadows, and tiny distant off-path objects.\n"
      f"Question: Is the relevant construction hazard touching, overlapping, or narrowing the {edge}?\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in CONSTRUCTION_ACTION_LABEL_SET:
    direction = "left" if label == "construction_drive_left" else "right"
    opposite = "right" if label == "construction_drive_left" else "left"
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. The translucent green overlay is the actual tracked path and vehicle-width corridor.\n"
      "Image left is ego/driver left. Image right is ego/driver right.\n"
      "The blue line is the ego-left edge of the green corridor. The purple line is the ego-right edge.\n"
      "Count only real cones, barrels, pylons, bollards, barricades, barriers, or blocked-lane panels that touch, overlap, intrude into, narrow, or block the green corridor.\n"
      "Ignore lane paint, road edges, shoulders, walls, traffic lights, signs, poles, UI text, colored overlay text, shadows, and distant off-path background objects.\n"
      "Do not answer the side where the construction object is located. Answer the safe ego driving direction away from it.\n"
      f"Question: Should ego drive {direction} within the lane because construction narrows the {opposite} side of the green corridor?\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in CONSTRUCTION_CANDIDATE_LABEL_SET:
    candidate = "left-shifted" if label == "construction_blocks_left_candidate" else "right-shifted"
    color = "cyan / blue-green" if label == "construction_blocks_left_candidate" else "magenta / pink"
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. This image may be either a single candidate board or a same-frame candidate-comparison board.\n"
      f"For this label, inspect only the {candidate} candidate path and vehicle-width corridor.\n"
      f"If the board shows two colored candidates, the {candidate} candidate is the {color} corridor.\n"
      "Do not score the other colored candidate for this label.\n"
      "If the board shows one green candidate corridor, inspect that green candidate corridor.\n"
      "Ignore the old base path, colored edge lines, UI text, labels, lane paint, road edges, shoulders, walls, traffic lights, signs, poles, and shadows.\n"
      "Count only real cones, barrels, pylons, bollards, barricades, barriers, or blocked-lane panels that touch, overlap, intrude into, narrow, or block the green candidate corridor ahead.\n"
      "On a two-candidate board, count construction touching the matching colored candidate corridor even though it is not green.\n"
      "Answer yes if this candidate corridor is obstructed by construction.\n"
      "Answer no if this candidate corridor is clear of construction even though construction may be visible elsewhere.\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  if score_prompt_mode == "construction_compact" and label in CONSTRUCTION_SHIFT_LABEL_SET:
    direction = "left" if label == "construction_shift_left" else "right"
    opposite = "right" if label == "construction_shift_left" else "left"
    return (
      f"Scored label: {label}\n"
      "Driver-view scene board. The green overlay is the actual tracked path and vehicle-width corridor.\n"
      "A thin magenta dashed line, if present, is the unmodified base openpilot path before VLM lateral bias.\n"
      "Image left is driver left. Image right is driver right.\n"
      "Count only real cones, barrels, pylons, bollards, barricades, barriers, or blocked-lane panels that touch, overlap, intrude into, narrow, or block the green corridor.\n"
      "Ignore lane lines, road edges, traffic lights, signs, poles, route arrows, UI text, shadows, walls, and distant off-path background objects.\n"
      "If the green path has drifted toward construction, answer the shift direction that moves the tracked path away from the construction.\n"
      f"Question: Should the green planned path shift {direction} within the lane because construction narrows the {opposite} side of the corridor?\n"
      "Answer exactly yes or no.\n"
      "Answer:"
    )
  return (
    f"Scored label: {label}\n"
    f"{SCORE_PROMPT}\n"
    f"Vehicle state: {vehicle_state}\n"
    f"Question: {SCORE_QUESTIONS[label]}\n"
    "Answer:"
  )


def _choice_token_id_variants(processor, words: Sequence[str] = CHOICE_WORDS) -> dict[str, tuple[int, ...]]:
  variants: dict[str, tuple[int, ...]] = {}
  for word in words:
    ids: list[int] = []
    capitalized = word[:1].upper() + word[1:]
    forms = (word, capitalized, f" {word}", f" {capitalized}", f"\n{word}", f"\n{capitalized}")
    for text in forms:
      token_ids = processor.tokenizer(text, add_special_tokens=False).input_ids
      if len(token_ids) == 1 and int(token_ids[0]) not in ids:
        ids.append(int(token_ids[0]))
    if not ids:
      raise RuntimeError(f"choice word {word!r} has no single-token variant")
    variants[word] = tuple(ids)
  return variants


def _choice_token_ids(processor) -> torch.Tensor:
  # Primary ids are used by selected-logit TensorRT engines, whose output width
  # is fixed. Prefer the no-space token because the prompt ends with "Answer:".
  variants = _choice_token_id_variants(processor)
  return torch.tensor([variants[word][0] for word in CHOICE_WORDS], device="cuda", dtype=torch.long)


def _choice_flat_token_ids_and_words(processor) -> tuple[torch.Tensor, tuple[str, ...]]:
  ids: list[int] = []
  words: list[str] = []
  for word, token_ids in _choice_token_id_variants(processor).items():
    for token_id in token_ids:
      ids.append(token_id)
      words.append(word)
  return torch.tensor(ids, device="cuda", dtype=torch.long), tuple(words)


def _debug_tensor_stats(tensor: torch.Tensor) -> dict:
  data = tensor.detach()
  finite = torch.isfinite(data)
  numeric = data.float()
  return {
    "shape": list(data.shape),
    "dtype": str(data.dtype).replace("torch.", ""),
    "device": str(data.device),
    "min": float(numeric.min().detach().cpu()) if data.numel() else 0.0,
    "max": float(numeric.max().detach().cpu()) if data.numel() else 0.0,
    "mean": float(numeric.mean().detach().cpu()) if data.numel() else 0.0,
    "absmax": float(numeric.abs().max().detach().cpu()) if data.numel() else 0.0,
    "nonzero_count": int(torch.count_nonzero(data).detach().cpu()) if data.numel() else 0,
    "nan_count": int(torch.isnan(numeric).sum().detach().cpu()) if data.numel() else 0,
    "inf_count": int(torch.isinf(numeric).sum().detach().cpu()) if data.numel() else 0,
    "finite_count": int(finite.sum().detach().cpu()) if data.numel() else 0,
  }


def _choice_group_spec(labels: Sequence[str]) -> dict:
  group = tuple(labels)
  label_set = set(group)
  if label_set == {"red_stop_light", "green_go_light"}:
    return {
      "allowed": ("A", "B", "C"),
      "word_to_label": {"B": "red_stop_light", "C": "green_go_light"},
      "min_margin": 0.5,
      "prompt_kind": "traffic_signal",
      "description": (
        "A = no relevant traffic signal controls the ego path. The green planned-path overlay is not a traffic signal.\n"
        "B = a red traffic signal or red arrow controls the ego path and requires stop.\n"
        "C = a green traffic signal or green arrow controls the ego path and authorizes proceeding."
      ),
    }
  if label_set == {"stop_sign", "green_go_light"}:
    return {
      "allowed": ("stop", "go", "absent"),
      "word_to_label": {"stop": "stop_sign", "go": "green_go_light"},
      "min_margin": 0.5,
      "prompt_kind": "stop_or_signal",
      "description": (
        "stop = STOP sign faces or controls the ego lane or green planned path.\n"
        "go = visible green traffic signal head or green arrow authorizes the ego path.\n"
        "absent = neither relevant stop sign nor green traffic signal controls the ego path. The green planned-path overlay is not a traffic signal."
      ),
    }
  if label_set == {"construction_left", "construction_right"}:
    return {
      "allowed": ("left", "right"),
      "word_to_label": {"left": "construction_left", "right": "construction_right"},
      "min_margin": 0.5,
      "prompt_kind": "construction_side",
      "description": (
        "left = construction touches, overlaps, blocks, or narrows the image-left / driver-left side of the green planned corridor.\n"
        "right = construction touches, overlaps, blocks, or narrows the image-right / driver-right side of the green planned corridor.\n"
      ),
    }
  if label_set == {"construction_blue_edge", "construction_purple_edge"}:
    return {
      "allowed": ("blue", "purple"),
      "word_to_label": {"blue": "construction_blue_edge", "purple": "construction_purple_edge"},
      "min_margin": 0.5,
      "prompt_kind": "construction_edge_color",
      "description": (
        "blue = relevant construction touches, overlaps, blocks, or narrows the blue edge of the green planned corridor.\n"
        "purple = relevant construction touches, overlaps, blocks, or narrows the purple edge of the green planned corridor.\n"
      ),
    }
  if label_set == {"cones", "barrier"}:
    return {
      "allowed": ("present", "absent"),
      "word_to_label": {"present": "cones"},
      "min_margin": 4.0,
      "prompt_kind": "construction_presence",
      "description": (
        "present = construction cones, pylons, barrels, barricades, blocked-lane panels, or barriers are visible in or immediately beside the green planned corridor.\n"
        "absent = no construction object affects the green planned corridor."
      ),
    }
  if label_set == {"construction_shift_left", "construction_shift_right"}:
    return {
      "allowed": ("A", "B", "clear"),
      "word_to_label": {"A": "construction_shift_left", "B": "construction_shift_right"},
      "min_margin": 0.5,
      "competitor_min_margin": 0.5,
      "prompt_kind": "construction_shift",
      "description": (
        "A = candidate path A is the safe bounded path around construction.\n"
        "B = candidate path B is the safe bounded path around construction.\n"
        "clear = construction is absent, decorative, too distant, or clearly outside the full green planned corridor."
      ),
    }
  if label_set == {"construction_drive_left", "construction_drive_right"}:
    return {
      "allowed": ("left", "right", "clear"),
      "word_to_label": {"left": "construction_drive_left", "right": "construction_drive_right"},
      "min_margin": 0.5,
      "competitor_min_margin": 0.5,
      "prompt_kind": "construction_drive_direction",
      "description": (
        "left = the safe bounded ego path response is to drive left, toward image-left / driver-left, away from construction on the right side of the corridor.\n"
        "right = the safe bounded ego path response is to drive right, toward image-right / driver-right, away from construction on the left side of the corridor.\n"
        "clear = construction is absent, decorative, too distant, or clearly outside the full green planned corridor."
      ),
    }
  if label_set == {"construction_blocks_left_candidate", "construction_blocks_right_candidate"}:
    return {
      "allowed": ("cyan", "pink", "none"),
      "word_to_label": {"cyan": "construction_blocks_left_candidate", "pink": "construction_blocks_right_candidate"},
      "min_margin": 0.5,
      "prompt_kind": "construction_candidate_obstruction",
      "description": (
        "cyan = the cyan / blue-green left-shifted candidate corridor is obstructed by construction.\n"
        "pink = the magenta / pink right-shifted candidate corridor is obstructed by construction.\n"
        "none = neither candidate corridor is obstructed by construction."
      ),
    }
  if label_set == {"pedestrian_in_path", "pedestrian_entering_path"}:
    return {
      "allowed": ("blocked", "entering", "none"),
      "word_to_label": {"blocked": "pedestrian_in_path", "entering": "pedestrian_entering_path"},
      "min_margin": 3.0,
      "description": (
        "blocked = visible human body is partly or fully inside the green path or directly blocking the ego lane.\n"
        "entering = visible human body is next to the green path and clearly moving into it soon.\n"
        "none = no pedestrian is in or imminently entering the green path."
      ),
    }
  if label_set == {"vehicle_in_path", "vehicle_entering_path"}:
    return {
      "allowed": ("blocked", "entering", "none"),
      "word_to_label": {"blocked": "vehicle_in_path", "entering": "vehicle_entering_path"},
      "min_margin": 3.0,
      "description": (
        "blocked = vehicle, bicycle, motorcycle, or similar road user overlaps the green path or blocks the ego lane.\n"
        "entering = vehicle, bicycle, motorcycle, or similar road user is clearly moving into the ego lane soon.\n"
        "none = no vehicle is in or imminently entering the green path."
      ),
    }
  if label_set == {"true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle"}:
    return {
      "allowed": ("moving", "slower", "braking", "stopped", "merge", "crossing", "irrelevant", "none"),
      "word_to_label": {
        "moving": "true_moving_lead",
        "slower": "slower_lead",
        "braking": "braking_lead",
        "stopped": "stopped_lead",
        "merge": "cut_in_vehicle",
        "crossing": "crossing_vehicle",
        "irrelevant": "irrelevant_vehicle",
      },
      "min_margin": 0.5,
      "description": (
        "When the vehicle state says lead present yes, use distance, lateral offset, lead speed, relative speed, closing, acceleration, and lateral velocity as production lead-track evidence. "
        "If absolute lateral offset is small, choose moving, slower, braking, or stopped, not merge, crossing, irrelevant, or none. "
        "stopped requires near-zero lead speed; braking requires negative accel, brake lights, or rapid closing; slower requires nonzero lead speed plus negative relative speed or positive closing; moving means stable/opening spacing. "
        "merge/crossing require lateral offset plus lateral motion into/across the path.\n"
        "moving = lead vehicle ahead in the green path is moving near ego speed with stable spacing, so no extra slowdown is needed.\n"
        "slower = lead vehicle ahead in the green path is slower than ego or spacing is closing, so proportional slowdown is needed.\n"
        "braking = lead vehicle ahead is braking, brake lights are visible, or closing is rapid, so stronger slowdown is needed.\n"
        "stopped = stopped or nearly stopped vehicle blocks the green path, so stop, creep, or route response is needed.\n"
        "merge = adjacent-lane vehicle is cutting in, merging, or about to enter the green path ahead.\n"
        "crossing = side vehicle is crossing the green path with conflict risk.\n"
        "irrelevant = visible vehicles do not affect the green path, so no yield or slowdown is needed.\n"
        "none = no relevant vehicle or lead state is visible."
      ),
    }
  if label_set == {"animal_in_path", "animal_entering_path"}:
    return {
      "allowed": ("blocked", "entering", "none"),
      "word_to_label": {"blocked": "animal_in_path", "entering": "animal_entering_path"},
      "min_margin": 3.0,
      "description": (
        "blocked = animal overlaps the green path or blocks the ego lane.\n"
        "entering = animal is beside the green path and clearly moving into it soon.\n"
        "none = no animal is in or imminently entering the green path."
      ),
    }
  raise ValueError(f"choice mode does not support label group: {group}")


def _choice_text(labels: Sequence[str], vehicle_state: str) -> str:
  spec = _choice_group_spec(labels)
  if spec.get("prompt_kind") == "traffic_signal":
    return (
      "You are scoring the traffic signal state for one forward driving scene board.\n"
      "Use only visible evidence in the image and supplied vehicle state.\n"
      "The image is an ego-driver forward camera view. The green planned-path overlay is not a traffic light and must be ignored as a color cue.\n"
      "Look for a real traffic signal head, traffic signal icon, or traffic arrow that faces, governs, or controls the ego lane, ego road, or green planned path.\n"
      "Do not classify lane lines, road-edge dots, construction markers, overlay text, the green path, the blue/purple corridor edges, or dashboard/status text as signal lamps.\n"
      "Only the bright illuminated lamp matters. Dark unlit lenses in a signal housing do not count as their color.\n"
      "For a vertical signal head, the top lamp position is red, the middle lamp position is yellow, and the bottom lamp position is green.\n"
      "Answer A when no relevant controlling signal is visible, the signal controls cross traffic or another lane, or the visible lamp is not clearly red or green.\n"
      "Answer B when the illuminated controlling signal is red or a red arrow controls ego.\n"
      "Answer C when the illuminated controlling signal is green or a green arrow authorizes ego to proceed.\n"
      "A tiny, distant, high, or off-center signal still counts if it controls ego and its lit color is visible.\n"
      "If both red and green are visible, choose the lit lamp position that controls the ego lane and current planned path. Do not infer from vehicle speed.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: A, B, C\n"
      "Choose exactly one allowed answer word. No prose.\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "stop_or_signal":
    return (
      "You are scoring whether ego must obey a STOP sign or may proceed on a green signal in one forward driving scene board.\n"
      "Use only visible evidence in the image and supplied vehicle state.\n"
      "The image is an ego-driver forward camera view. The green planned-path overlay is not a traffic signal and must be ignored as a color cue.\n"
      "Look for a real STOP sign that faces, governs, or controls the ego lane, ego road, or green planned path.\n"
      "Look for a real traffic signal head, traffic signal icon, or traffic arrow that faces, governs, or controls the ego lane, ego road, or green planned path.\n"
      "Do not classify lane lines, road-edge dots, construction markers, overlay text, the green path, the blue/purple corridor edges, or dashboard/status text as signs or signal lamps.\n"
      "Only the bright illuminated traffic-signal lamp matters. Dark unlit lenses in a signal housing do not count as their color.\n"
      "For a vertical signal head, the top lamp position is red, the middle lamp position is yellow, and the bottom lamp position is green.\n"
      "Answer stop when a STOP sign controls the ego path.\n"
      "Answer go when the illuminated controlling signal is green or a green arrow authorizes ego to proceed.\n"
      "Answer absent when no relevant STOP sign or green controlling signal is visible, or the visible object controls cross traffic or another lane.\n"
      "A tiny, distant, high, or off-center sign/signal still counts if it controls ego and is readable.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: stop, go, absent\n"
      "Choose exactly one allowed answer word. No prose.\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_shift":
    return (
      "You are scoring the SAFE BOUNDED PATH SHIFT for construction in one forward driving scene board.\n"
      "Use only visible evidence in the image, the green planned path/corridor, and supplied vehicle state.\n"
      "The scene-board image is a forward ego-driver camera view: image left is ego vehicle left, image right is ego vehicle right.\n"
      "The green overlay is the current ego planned path and vehicle-width corridor.\n"
      "The cyan PATH A line is a candidate path shifted left from the green path.\n"
      "The pink PATH B line is a candidate path shifted right from the green path.\n"
      "Choose the candidate path ID that avoids visible construction while staying in the lane.\n"
      "The blue edge line marks the ego-left side of the green corridor.\n"
      "The purple edge line marks the ego-right side of the green corridor.\n"
      "Answer A if PATH A clears the construction hazard better than PATH B.\n"
      "Answer B if PATH B clears the construction hazard better than PATH A.\n"
      "Do not answer the side where the cone or barrier is located. Answer the safe candidate path ID away from the hazard.\n"
      "Only answer clear when no real cone, pylon, bollard, barrel, barrier, barricade, or blocked-lane panel is visible in or immediately beside the green corridor.\n"
      "Ignore lane lines, road edges, the green overlay itself, and distant off-path background objects.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: A, B, clear\n"
      "A = use PATH A.\n"
      "B = use PATH B.\n"
      "clear = no construction hazard affects the green planned corridor.\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_drive_direction":
    return (
      "You are scoring the SAFE EGO DRIVING DIRECTION for construction in one forward driver-view scene board.\n"
      "Use only visible evidence in the image, the green planned path/corridor, and supplied vehicle state.\n"
      "The scene-board image is a forward ego-driver camera view: image left is ego/driver left, image right is ego/driver right.\n"
      "The green overlay is the current ego planned path and vehicle-width corridor.\n"
      "The blue line is the ego-left edge of the green corridor. The purple line is the ego-right edge.\n"
      "Only consider real cones, barrels, pylons, bollards, barricades, blocked-lane panels, or barriers that touch, overlap, intrude into, narrow, or block the green corridor ahead.\n"
      "Ignore tiny distant cone rows, lane paint, road edges, shoulders, UI text, colored overlay text, signs, poles, shadows, and off-path background objects.\n"
      "Answer left when the safe bounded path should move toward image-left / driver-left to clear construction on the right or purple side of the corridor.\n"
      "Answer right when the safe bounded path should move toward image-right / driver-right to clear construction on the left or blue side of the corridor.\n"
      "Do not answer the side where the cone or barrier is located. Answer the safe direction ego should drive away from the hazard.\n"
      "Only answer clear when no real construction hazard affects the green planned corridor.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: left, right, clear\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_presence":
    return (
      "You are scoring whether construction objects actually affect the ego driving corridor in one forward driver-view scene board.\n"
      "The translucent green overlay is the ego planned driving corridor. The colored blue and purple lines are overlay edges, not construction objects.\n"
      "The scene board may darken pixels outside the widened ego corridor. Treat darkened off-corridor objects as irrelevant unless they are clearly touching, entering, or blocking the green corridor.\n"
      "Answer present only when at least one real three-dimensional cone, barrel, pylon, bollard, barricade, blocked-lane panel, or barrier is visibly on the road surface and touching, overlapping, intruding into, narrowing, or immediately blocking the green corridor ahead.\n"
      "Answer absent when the only orange/red marks are tiny distant roadside dots, horizon markers, shoulder markers, wall markers, lane paint, road texture, UI text, colored overlay lines, or objects clearly outside the full green corridor.\n"
      "Do not count the green path overlay, blue/purple corridor edge lines, cyan/pink candidate guide lines, white lane markings, yellow lane markings, or LEFT/RIGHT text labels as construction.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: present, absent\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_side":
    return (
      "You are scoring construction side for one forward driver-view scene board.\n"
      "Use only visible evidence in the image and the planned path overlay.\n"
      "The translucent green overlay is the ego planned driving corridor.\n"
      "The blue edge line marks the ego-left edge of that green corridor.\n"
      "The purple edge line marks the ego-right edge of that green corridor.\n"
      "Judge construction side by the nearest colored corridor edge that the object narrows or touches.\n"
      "If the cone or barrier is nearest the blue edge, answer left, whether it is just outside, on, or just inside the blue line.\n"
      "If the cone or barrier is nearest the purple edge, answer right, whether it is just outside, on, or just inside the purple line.\n"
      "Do not flip side merely because a cone is slightly right of the blue line or slightly left of the purple line; use the nearest corridor edge.\n"
      "The scene board may darken pixels outside the widened ego corridor. Darkened off-corridor objects are irrelevant unless they touch, enter, or block the green corridor.\n"
      "Colored overlay text, if present, is not evidence. Judge the cone/barrier contact point against the blue or purple corridor edge line.\n"
      "The corridor narrows toward the horizon, so judge side by the colored corridor edge the hazard touches, not by absolute image x position.\n"
      "If a cone row or barrier is on, touching, or just inside the blue ego-left edge line, answer left.\n"
      "If a cone row or barrier is on, touching, or just inside the purple ego-right edge line, answer right.\n"
      "Only consider real cones, barrels, pylons, bollards, barricades, blocked-lane panels, or barriers that touch, overlap, intrude into, narrow, or block the green corridor ahead.\n"
      "Ignore tiny distant cone rows, lane paint, road edges, shoulders, UI text, colored overlay text, signs, poles, shadows, and off-path background objects.\n"
      "Answer left only when the relevant construction hazard is touching or narrowing the blue / ego-left side of the green corridor.\n"
      "Answer right only when the relevant construction hazard is touching or narrowing the purple / ego-right side of the green corridor.\n"
      "Do not answer the direction to steer. Answer the side of the green corridor where the relevant construction hazard is located.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: left, right\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_edge_color":
    return (
      "You are scoring which colored planned-corridor edge is affected by construction in one forward driver-view scene board.\n"
      "Use only visible evidence in the image and the planned path overlay.\n"
      "The translucent green overlay is the ego planned driving corridor.\n"
      "The blue line is the ego-left edge of the green corridor.\n"
      "The purple line is the ego-right edge of the green corridor.\n"
      "Answer blue when the relevant cone, barrel, pylon, bollard, barricade, blocked-lane panel, or barrier touches, overlaps, narrows, or blocks the blue corridor edge.\n"
      "Answer purple when the relevant cone, barrel, pylon, bollard, barricade, blocked-lane panel, or barrier touches, overlaps, narrows, or blocks the purple corridor edge.\n"
      "Do not answer a steering direction. Do not answer image-left or image-right. Answer only the colored edge that the hazard affects.\n"
      "If a cone row is between both colored edges, choose the colored edge that is closest to the row where it narrows the usable green corridor.\n"
      "Ignore tiny distant cone rows, lane paint, road edges, shoulders, UI text, colored overlay text, signs, poles, shadows, and off-path background objects.\n"
      "Only consider real construction objects that touch, overlap, intrude into, narrow, or block the green corridor ahead.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: blue, purple\n"
      "Answer:"
    )
  if spec.get("prompt_kind") == "construction_candidate_obstruction":
    return (
      "You are scoring which SAME-FRAME candidate trajectory corridor is obstructed by construction in one forward driver-view scene board.\n"
      "Use only visible evidence in the image and the two colored candidate overlays.\n"
      "The cyan / blue-green corridor is the bounded candidate path shifted left from the current base path.\n"
      "The magenta / pink corridor is the bounded candidate path shifted right from the current base path.\n"
      "The dashed white line, if present, is only the unmodified base path reference.\n"
      "Answer cyan if real cones, barrels, pylons, bollards, barricades, blocked-lane panels, or barriers touch, overlap, intrude into, narrow, or block the cyan left-shifted candidate corridor.\n"
      "Answer pink if real cones, barrels, pylons, bollards, barricades, blocked-lane panels, or barriers touch, overlap, intrude into, narrow, or block the magenta right-shifted candidate corridor.\n"
      "If both candidates are obstructed, choose the one with the larger obstruction into its corridor. If the difference is unclear, answer none.\n"
      "Ignore lane paint, road edges, shoulders, UI text, colored overlay labels, signs, poles, shadows, and distant off-path background objects.\n"
      "Do not answer the safe path. Answer the candidate corridor that is blocked by construction.\n"
      "Choose exactly one answer word from the allowed list. No prose.\n"
      f"Vehicle state: {vehicle_state}\n"
      "Allowed answer words: cyan, pink, none\n"
      "Answer:"
    )
  allowed = ", ".join(spec["allowed"])
  return (
    "You are scoring one real-time driving scene-board label group.\n"
    "Use only visible evidence, the green planned path/corridor, and supplied vehicle state.\n"
    "Ignore objects that are merely visible off to the side and do not overlap, narrow, block, control, or imminently enter the green planned path.\n"
    "Choose exactly one answer word from the allowed list. No prose.\n"
    f"Vehicle state: {vehicle_state}\n"
    f"Allowed answer words: {allowed}\n"
    f"{spec['description']}\n"
    "Answer:"
  )


def _validate_fixed_text_length(processor, prompts: Sequence[str], text_seq_len: int) -> None:
  if text_seq_len <= 0:
    return
  lengths = [len(processor.tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
  max_len = max(lengths, default=0)
  if max_len > text_seq_len:
    raise RuntimeError(
      f"text_seq_len={text_seq_len} would truncate score prompt tokens; "
      f"max prompt length is {max_len}. Increase --text-seq-len."
    )


def _parse_score_threshold_map(raw: str) -> dict[str, float]:
  thresholds: dict[str, float] = {}
  if not raw.strip():
    return thresholds
  for item in raw.split(","):
    if not item.strip():
      continue
    if ":" not in item:
      raise ValueError(f"invalid score threshold item: {item}")
    label, value_raw = item.split(":", 1)
    label = label.strip()
    if label not in SCORE_QUESTIONS:
      raise ValueError(f"unknown score threshold label: {label}")
    thresholds[label] = float(value_raw)
  return thresholds


def _sha256_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _manifest_path(args) -> Path:
  manifest = getattr(args, "manifest", None)
  return manifest if manifest is not None else args.artifact_dir / "qwen_trt_runtime_manifest.json"


def _model_revision(model_dir: Path) -> dict:
  files = {}
  for name in MODEL_CONTRACT_FILES:
    path = model_dir / name
    if path.exists():
      files[name] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
      }
    else:
      files[name] = {
        "missing": True,
      }
  weights = []
  for path in sorted(model_dir.glob("*.safetensors")):
    weights.append({
      "name": path.name,
      "bytes": path.stat().st_size,
      "mtime": path.stat().st_mtime,
    })
  return {
    "model_dir_name": model_dir.name,
    "files": files,
    "weights": weights,
  }


def _scene_board_contract() -> dict:
  if OverlayGeometry is None:
    return {"overlay_geometry_importable": False}
  geometry = OverlayGeometry()
  return {
    "overlay_geometry_importable": True,
    "planned_corridor_half_width_m": float(geometry.planned_corridor_half_width_m),
    "focus_corridor_extra_width_m": float(geometry.focus_corridor_extra_width_m),
    "dim_outside_corridor": bool(geometry.dim_outside_corridor),
    "outside_corridor_dim_alpha": int(geometry.outside_corridor_dim_alpha),
    "lane_width_m": float(geometry.lane_width_m),
    "camera_height_m": float(geometry.camera_height_m),
    "horizon_ratio": float(geometry.horizon_ratio),
    "focal_ratio": float(geometry.focal_ratio),
    "max_draw_distance_m": float(geometry.max_draw_distance_m),
    "draw_candidate_labels": bool(geometry.draw_candidate_labels),
    "draw_corridor_side_guides": bool(geometry.draw_corridor_side_guides),
    "draw_corridor_side_labels": bool(geometry.draw_corridor_side_labels),
    "draw_candidate_obstruction_boards": bool(geometry.draw_candidate_obstruction_boards),
    "candidate_obstruction_offset_m": float(geometry.candidate_obstruction_offset_m),
  }


def _runtime_contract(args, groups: Sequence[Sequence[str]]) -> dict:
  normalized_groups = tuple(tuple(group) for group in groups)
  labels = tuple(dict.fromkeys(label for group in normalized_groups for label in group))
  questions = {label: SCORE_QUESTIONS[label] for label in labels}
  prompt_mode = _score_prompt_mode(args)
  rendered_score_prompts = {
    label: _score_text(label, str(args.vehicle_state), prompt_mode)
    for label in labels
  }
  payload = {
    "manifest_version": MANIFEST_VERSION,
    "model": _model_revision(args.model_dir),
    "prompt": {
      "score_prompt_sha256": _sha256_text(SCORE_PROMPT),
      "score_questions_sha256": _sha256_text(json.dumps(questions, sort_keys=True, separators=(",", ":"))),
      "rendered_score_prompts_sha256": _sha256_text(json.dumps(rendered_score_prompts, sort_keys=True, separators=(",", ":"))),
      "labels": labels,
    },
    "scene_board": _scene_board_contract(),
    "runtime": {
      "runtime_mode": args.runtime_mode,
      "image_mode": args.image_mode,
      "image_size": int(args.image_size),
      "vision_precision": _vision_precision(args),
      "vision_feature_clip_abs": float(getattr(args, "vision_feature_clip_abs", 0.0)),
      "text_seq_len": int(args.text_seq_len),
      "text_precision": _text_precision(args),
      "text_output": _text_output(args),
      "fixed_output_index": int(getattr(args, "fixed_output_index", -1)),
      "label_decision_mode": _label_decision_mode(args),
      "score_prompt_mode": prompt_mode,
      "text_position_mode": _text_position_mode(args),
      "text_position_dtype": _text_position_dtype(args),
      "torch_attn_implementation": _torch_attn_implementation(args),
      "text_strongly_typed": bool(getattr(args, "text_strongly_typed", False)),
      "score_label_groups": normalized_groups,
      "score_rotate_groups": bool(args.score_rotate_groups),
      "score_rotate_shared_engine": bool(args.score_rotate_shared_engine),
      "vehicle_state": args.vehicle_state,
      "use_payload_vehicle_state": bool(getattr(args, "use_payload_vehicle_state", True)),
      "payload_vehicle_state_scope": str(getattr(args, "payload_vehicle_state_scope", "auto")),
      "score_threshold": float(args.score_threshold),
      "score_thresholds": dict(sorted(getattr(args, "score_thresholds_map", {}).items())),
      "score_cache_ttl_frames": int(args.score_cache_ttl_frames),
      "score_durable_labels": tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip()),
      "score_negative_clear_threshold": float(args.score_negative_clear_threshold),
      "construction_mirror_consistency": bool(getattr(args, "construction_mirror_consistency", False)),
      "construction_mirror_fusion": bool(getattr(args, "construction_mirror_fusion", False)),
      "construction_side_choice": bool(getattr(args, "construction_side_choice", False)),
      "construction_edge_choice": bool(getattr(args, "construction_edge_choice", False)),
      "construction_edge_binary": bool(getattr(args, "construction_edge_binary", False)),
      "construction_candidate_binary": bool(getattr(args, "construction_candidate_binary", False)),
      "construction_candidate_choice": bool(getattr(args, "construction_candidate_choice", False)),
      "construction_candidate_relative_choice": bool(getattr(args, "construction_candidate_relative_choice", False)),
      "construction_candidate_relative_margin": float(getattr(args, "construction_candidate_relative_margin", 1.5)),
      "construction_candidate_relative_neutral_margin": float(getattr(args, "construction_candidate_relative_neutral_margin", CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN)),
      "construction_candidate_score_resolve": bool(getattr(args, "construction_candidate_score_resolve", False)),
      "construction_candidate_diff_margin": float(getattr(args, "construction_candidate_diff_margin", 0.4)),
      "construction_presence_hold_frames": CONSTRUCTION_PRESENCE_HOLD_FRAMES,
      "construction_state_machine_version": CONSTRUCTION_STATE_MACHINE_VERSION,
      "construction_presence_clear_confirm_frames": CONSTRUCTION_PRESENCE_CLEAR_CONFIRM_FRAMES,
      "construction_reactivate_min_presence_score": CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE,
      "construction_edge_bootstrap_score": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE,
      "construction_edge_bootstrap_margin": CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN,
      "construction_edge_bootstrap_opposite_max": CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX,
      "construction_edge_neutral_bootstrap_score": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE,
      "construction_edge_neutral_bootstrap_margin": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
      "construction_edge_toward_path_override_score": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE,
      "construction_edge_toward_path_override_margin": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
      "construction_direct_consensus_max_age_frames": CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES,
      "construction_direct_semantic_score": CONSTRUCTION_DIRECT_SEMANTIC_SCORE,
      "construction_direct_semantic_margin": CONSTRUCTION_DIRECT_SEMANTIC_MARGIN,
      "construction_direct_edge_score": CONSTRUCTION_DIRECT_EDGE_SCORE,
      "construction_direct_edge_margin": CONSTRUCTION_DIRECT_EDGE_MARGIN,
      "construction_semantic_neutral_bootstrap_score": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE,
      "construction_semantic_neutral_bootstrap_margin": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN,
      "construction_semantic_toward_path_override_score": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE,
      "construction_semantic_toward_path_override_margin": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
      "construction_candidate_immediate_score": CONSTRUCTION_CANDIDATE_IMMEDIATE_SCORE,
      "construction_candidate_immediate_margin": CONSTRUCTION_CANDIDATE_IMMEDIATE_MARGIN,
      "construction_action_bootstrap_max_tracked_offset_m": CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M,
      "construction_action_continue_min_tracked_offset_m": CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M,
      "construction_action_immediate_score": CONSTRUCTION_ACTION_IMMEDIATE_SCORE,
      "construction_action_immediate_margin": CONSTRUCTION_ACTION_IMMEDIATE_MARGIN,
      "construction_action_contradictory_override_score": CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE,
      "construction_action_contradictory_override_margin": CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN,
      "construction_committed_conflict_requires_action": CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION,
      "construction_side_clear_max_score": CONSTRUCTION_SIDE_CLEAR_MAX_SCORE,
      "construction_side_clear_min_tracked_offset_m": CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M,
      "construction_side_committed_away_min_tracked_offset_m": CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M,
      "construction_side_toward_path_override_score": CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE,
      "construction_side_early_reversal_score": CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE,
      "construction_side_early_reversal_margin": CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN,
      "construction_side_early_reversal_opposite_max": CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX,
      "construction_side_early_reversal_max_tracked_offset_m": CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M,
      "construction_side_early_reversal_max_age_frames": CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES,
      "construction_clear_rtp_confidence": CONSTRUCTION_CLEAR_RTP_CONFIDENCE,
      "path_agent_rtp_version": PATH_AGENT_RTP_VERSION,
      "path_blocking_agent_labels": tuple(sorted(PATH_BLOCKING_AGENT_LABELS)),
      "path_entering_agent_labels": tuple(sorted(PATH_ENTERING_AGENT_LABELS)),
      "path_agent_stop_s": PATH_AGENT_STOP_S,
      "path_agent_yield_speed_cap": PATH_AGENT_YIELD_SPEED_CAP,
      "construction_text_precision": _construction_text_precision(args),
      "construction_text_engine": "" if getattr(args, "construction_text_engine", None) is None else str(args.construction_text_engine),
      "construction_text_seq_len": int(getattr(args, "construction_text_seq_len", 0) or 0),
      "enable_signal_head": bool(args.enable_signal_head or args.runtime_mode == "visual-head"),
      "signal_min_probability": float(args.signal_min_probability),
      "signal_min_margin": float(args.signal_min_margin),
    },
  }
  return {
    "contract_sha256": _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
    "contract": payload,
  }


def _read_manifest(path: Path) -> tuple[dict | None, str | None]:
  if not path.exists():
    return None, f"missing manifest: {path}"
  try:
    return json.loads(path.read_text(encoding="utf-8")), None
  except Exception as exc:
    return None, f"failed to read manifest {path}: {exc!r}"


def _validate_manifest(args, groups: Sequence[Sequence[str]]) -> dict:
  path = _manifest_path(args)
  expected = _runtime_contract(args, groups)
  actual, error = _read_manifest(path)
  issues = []
  if error is not None:
    issues.append(error)
    actual_sha = ""
  else:
    actual_sha = str(actual.get("contract_sha256", ""))
    if actual_sha != expected["contract_sha256"]:
      issues.append(
        f"manifest contract sha mismatch: actual {actual_sha} expected {expected['contract_sha256']}"
      )
  return {
    "path": str(path),
    "exists": path.exists(),
    "ok": not issues,
    "issues": issues,
    "actual_contract_sha256": actual_sha,
    "expected_contract_sha256": expected["contract_sha256"],
  }


def _write_manifest(args, groups: Sequence[Sequence[str]], result: dict) -> dict:
  path = _manifest_path(args)
  contract = _runtime_contract(args, groups)
  manifest = {
    "kind": "qwen_trt_runtime_manifest",
    "created_unix": time.time(),
    **contract,
    "artifact_dir": str(args.artifact_dir),
    "vision_engine": result.get("vision_engine", {}),
    "text_engine": result.get("text_engine", {}),
    "signal_head": result.get("signal_head", {}),
    "cuda": result.get("cuda", {}),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return {
    "path": str(path),
    "written": True,
    "contract_sha256": contract["contract_sha256"],
  }


def _groups_for_runtime(args) -> tuple[tuple[str, ...], ...]:
  if getattr(args, "runtime_mode", "score") == "visual-head":
    return (SIGNAL_LABELS,)
  if getattr(args, "score_rotate_groups", False) or getattr(args, "cmd", "") in ("benchmark-groups", "check-artifacts"):
    return _score_groups(args.score_label_groups)
  return (_score_labels(args.score_labels),)


def _enforce_manifest(args, groups: Sequence[Sequence[str]]) -> None:
  manifest = _validate_manifest(args, groups)
  if not manifest["ok"]:
    raise RuntimeError("; ".join(manifest["issues"]))


def _summarize_timing_rows(timing_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
  if not timing_rows:
    return {}
  summary = {}
  for key in timing_rows[0]:
    vals = [row[key] for row in timing_rows]
    summary[key] = {
      "median": statistics.median(vals),
      "p90": percentile(vals, 90),
      "p99": percentile(vals, 99),
      "p999": percentile(vals, 99.9),
      "max": max(vals),
      "min": min(vals),
    }
  return summary


def _build_inputs(
  processor,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
  score_prompt_mode: str = "full",
):
  images, _ = _inference_images(image, image_mode, image_size)
  prompts: list[str] = []
  batch_images: list[Image.Image] = []
  for label in labels:
    content = [{"type": "image", "image": view} for view in images]
    content.append({"type": "text", "text": _score_text(label, vehicle_state, score_prompt_mode)})
    prompts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    batch_images.extend(images)
  _validate_fixed_text_length(processor, prompts, text_seq_len)
  processor_kwargs = {
    "text": prompts,
    "images": batch_images,
    "padding": True,
    "return_tensors": "pt",
  }
  if text_seq_len > 0:
    processor_kwargs["padding"] = "max_length"
    processor_kwargs["max_length"] = text_seq_len
    processor_kwargs["truncation"] = True
  return processor(**processor_kwargs).to("cuda")


def _build_vision_inputs(processor, image: Image.Image, image_mode: str, image_size: int):
  images, _ = _inference_images(image, image_mode, image_size)
  prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": "x"}]}],
    tokenize=False,
    add_generation_prompt=True,
  )
  return processor(text=[prompt], images=images[:1], padding=True, return_tensors="pt").to("cuda")


def _build_choice_inputs(
  processor,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
  position_args=None,
):
  images, _ = _inference_images(image, image_mode, image_size)
  content = [{"type": "image", "image": images[0]}, {"type": "text", "text": _choice_text(labels, vehicle_state)}]
  prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
  _validate_fixed_text_length(processor, [prompt], text_seq_len)
  processor_kwargs = {
    "text": [prompt],
    "images": images[:1],
    "padding": True,
    "return_tensors": "pt",
  }
  if text_seq_len > 0:
    processor_kwargs["padding"] = "max_length"
    processor_kwargs["max_length"] = text_seq_len
    processor_kwargs["truncation"] = True
  return processor(**processor_kwargs).to("cuda")


def _last_token_indices_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
  if attention_mask.ndim != 2:
    raise ValueError(f"attention_mask must be rank 2, got {tuple(attention_mask.shape)}")
  flipped = attention_mask.to(torch.long).flip(dims=(1,))
  return (attention_mask.shape[1] - 1 - flipped.argmax(dim=1)).to(torch.long)


def _last_token_mask_from_indices(indices: torch.Tensor, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
  mask = torch.zeros(shape, device=indices.device, dtype=dtype)
  mask.scatter_(1, indices.view(-1, 1, 1), 1.0)
  return mask.contiguous()


def _prepare_text_tensors(
  processor,
  model,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
  position_args=None,
  score_prompt_mode: str = "full",
):
  inputs = _build_inputs(processor, image, labels, image_mode, image_size, vehicle_state, text_seq_len, score_prompt_mode)
  with torch.no_grad():
    qwen = model.model
    inputs_embeds = qwen.get_input_embeddings()(inputs.input_ids)
    image_features = torch.cat(qwen.get_image_features(inputs.pixel_values, inputs.image_grid_thw), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen.get_placeholder_mask(inputs.input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features).contiguous().detach().clone()
    position_ids, _ = qwen.get_rope_index(
      inputs.input_ids,
      inputs.image_grid_thw,
      None,
      second_per_grid_ts=None,
      attention_mask=inputs.attention_mask,
    )
    if position_args is not None:
      position_ids = _apply_text_position_mode(position_ids, position_args)
    position_ids = position_ids.contiguous().detach().clone()
    last_token_indices = _last_token_indices_from_attention_mask(inputs.attention_mask)
    last_token_mask = _last_token_mask_from_indices(
      last_token_indices,
      (inputs_embeds.shape[0], inputs_embeds.shape[1], 1),
      inputs_embeds.dtype,
    ).detach().clone()
    yes_ids, no_ids = _score_label_ids(processor)
    selected_ids = torch.tensor(list(yes_ids) + list(no_ids), device="cuda", dtype=torch.long)
  return inputs, inputs_embeds, position_ids, last_token_mask, selected_ids


def _prepare_choice_text_tensors(
  processor,
  model,
  image: Image.Image,
  labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state: str,
  text_seq_len: int = 0,
  position_args=None,
):
  inputs = _build_choice_inputs(processor, image, labels, image_mode, image_size, vehicle_state, text_seq_len)
  with torch.no_grad():
    qwen = model.model
    inputs_embeds = qwen.get_input_embeddings()(inputs.input_ids)
    image_features = torch.cat(qwen.get_image_features(inputs.pixel_values, inputs.image_grid_thw), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = qwen.get_placeholder_mask(inputs.input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features).contiguous().detach().clone()
    position_ids, _ = qwen.get_rope_index(
      inputs.input_ids,
      inputs.image_grid_thw,
      None,
      second_per_grid_ts=None,
      attention_mask=inputs.attention_mask,
    )
    if position_args is not None:
      position_ids = _apply_text_position_mode(position_ids, position_args)
    position_ids = position_ids.contiguous().detach().clone()
    last_token_indices = _last_token_indices_from_attention_mask(inputs.attention_mask)
    last_token_mask = _last_token_mask_from_indices(
      last_token_indices,
      (inputs_embeds.shape[0], inputs_embeds.shape[1], 1),
      inputs_embeds.dtype,
    ).detach().clone()
    selected_ids = _choice_token_ids(processor)
  return inputs, inputs_embeds, position_ids, last_token_mask, selected_ids


class TextScore(nn.Module):
  def __init__(self, language_model, lm_head, selected_ids: torch.Tensor):
    super().__init__()
    self.language_model = language_model
    self.lm_head = lm_head
    self.register_buffer("selected_ids", selected_ids.detach().clone())

  def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, last_token_mask: torch.Tensor) -> torch.Tensor:
    out = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=None,
      past_key_values=None,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
      cache_position=None,
    )
    last = (out[0] * last_token_mask).sum(dim=1, keepdim=True)
    selected_weight = self.lm_head.weight.index_select(0, self.selected_ids)
    return torch.matmul(last, selected_weight.t())


class TextFixedScore(nn.Module):
  def __init__(self, language_model, lm_head, selected_ids: torch.Tensor, output_index: int):
    super().__init__()
    self.language_model = language_model
    self.lm_head = lm_head
    self.output_index = int(output_index)
    self.register_buffer("selected_ids", selected_ids.detach().clone())

  def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, last_token_mask: torch.Tensor) -> torch.Tensor:
    del last_token_mask
    out = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=None,
      past_key_values=None,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
      cache_position=None,
    )
    last = out[0][:, self.output_index:self.output_index + 1, :]
    selected_weight = self.lm_head.weight.index_select(0, self.selected_ids)
    return torch.matmul(last, selected_weight.t())


class TextHidden(nn.Module):
  def __init__(self, language_model):
    super().__init__()
    self.language_model = language_model

  def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, last_token_mask: torch.Tensor) -> torch.Tensor:
    out = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=None,
      past_key_values=None,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
      cache_position=None,
    )
    return (out[0] * last_token_mask).sum(dim=1, keepdim=True)


class TextFullHidden(nn.Module):
  def __init__(self, language_model):
    super().__init__()
    self.language_model = language_model

  def forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, last_token_mask: torch.Tensor) -> torch.Tensor:
    del last_token_mask
    out = self.language_model(
      input_ids=None,
      position_ids=position_ids,
      attention_mask=None,
      past_key_values=None,
      inputs_embeds=inputs_embeds,
      use_cache=False,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
      cache_position=None,
    )
    return out[0]


def _select_last_hidden_from_full(full_hidden: torch.Tensor, last_token_indices: torch.Tensor) -> torch.Tensor:
  gather_index = last_token_indices.view(-1, 1, 1).expand(-1, 1, full_hidden.shape[-1])
  return full_hidden.gather(1, gather_index)


def _select_last_hidden_from_mask(full_hidden: torch.Tensor, last_token_mask: torch.Tensor) -> torch.Tensor:
  last_token_indices = last_token_mask.squeeze(-1).argmax(dim=1).to(torch.long)
  return _select_last_hidden_from_full(full_hidden, last_token_indices)


def _scores_from_selected_hidden(
  selected_hidden: torch.Tensor,
  model,
  processor,
  labels: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, float], dict]:
  selected_ids, token_words = _choice_flat_token_ids_and_words(processor)
  selected_weight = model.lm_head.weight.index_select(0, selected_ids.to(model.lm_head.weight.device))
  logits = torch.matmul(selected_hidden.to(selected_weight.device, selected_weight.dtype), selected_weight.t())
  return _choice_scores_from_selected_logits(logits, labels, token_words)


def _verify_text_onnx_fidelity(
  onnx_path: Path,
  ref: torch.Tensor,
  inputs_embeds: torch.Tensor,
  position_ids: torch.Tensor,
  last_token_mask: torch.Tensor,
  *,
  output_name: str,
  labels: Sequence[str],
  processor,
  model,
  text_output: str,
  label_decision_mode: str,
  max_mean_error: float,
  max_abs_error: float,
  require_choice_match: bool,
) -> dict:
  try:
    import onnxruntime as ort
  except ImportError as exc:
    raise RuntimeError("--verify-text-onnx-fidelity requires onnxruntime") from exc

  session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  feed_candidates = {
    "inputs_embeds": inputs_embeds.detach().cpu().numpy(),
    "position_ids": position_ids.detach().cpu().numpy(),
    "last_token_mask": last_token_mask.detach().cpu().numpy(),
  }
  actual_np = session.run(
    [output_name],
    {inp.name: feed_candidates[inp.name] for inp in session.get_inputs()},
  )[0]
  ref_np = ref.detach().cpu().numpy()
  diff = np.abs(ref_np.astype(np.float32) - actual_np.astype(np.float32))
  summary = {
    "enabled": True,
    "onnx": str(onnx_path),
    "output_name": output_name,
    "providers": session.get_providers(),
    "abs_mean": float(diff.mean()),
    "abs_max": float(diff.max()),
    "max_mean_error": float(max_mean_error),
    "max_abs_error": float(max_abs_error),
    "require_choice_match": bool(require_choice_match),
  }

  issues: list[str] = []
  if summary["abs_mean"] > max_mean_error:
    issues.append(f"abs_mean {summary['abs_mean']:.6g} > {max_mean_error:.6g}")
  if summary["abs_max"] > max_abs_error:
    issues.append(f"abs_max {summary['abs_max']:.6g} > {max_abs_error:.6g}")

  if require_choice_match and label_decision_mode == "choice":
    actual_tensor = torch.from_numpy(actual_np).to(ref.device, ref.dtype)
    if text_output in ("hidden", "full_hidden"):
      ref_hidden = ref if text_output == "hidden" else _select_last_hidden_from_mask(ref, last_token_mask)
      actual_hidden = actual_tensor if text_output == "hidden" else _select_last_hidden_from_mask(actual_tensor, last_token_mask)
      ref_selected, ref_scores, ref_choice = _scores_from_selected_hidden(ref_hidden, model, processor, labels)
      actual_selected, actual_scores, actual_choice = _scores_from_selected_hidden(actual_hidden, model, processor, labels)
    else:
      ref_selected, ref_scores, ref_choice = _choice_scores_from_selected_logits(ref, labels)
      actual_selected, actual_scores, actual_choice = _choice_scores_from_selected_logits(actual_tensor, labels)
    summary["choice"] = {
      "reference_labels": list(ref_selected),
      "onnx_labels": list(actual_selected),
      "reference_scores": ref_scores,
      "onnx_scores": actual_scores,
      "reference_choice": ref_choice,
      "onnx_choice": actual_choice,
    }
    if tuple(ref_selected) != tuple(actual_selected):
      issues.append(f"choice labels differ: reference {ref_selected} onnx {actual_selected}")

  summary["ok"] = not issues
  summary["issues"] = issues
  if issues:
    raise RuntimeError(f"ONNX text fidelity check failed for {onnx_path}: {'; '.join(issues)}")
  return summary


class VisionStatic(nn.Module):
  def __init__(
    self,
    visual,
    rotary_pos_emb: torch.Tensor,
    window_index: torch.Tensor,
    cu_window: torch.Tensor,
    cu_seqlens: torch.Tensor,
    reverse: torch.Tensor,
  ):
    super().__init__()
    self.visual = visual
    self.register_buffer("rotary_const", rotary_pos_emb.detach().clone())
    self.register_buffer("window_index", window_index.detach().clone())
    self.register_buffer("cu_window", cu_window.detach().clone())
    self.register_buffer("cu_seqlens", cu_seqlens.detach().clone())
    self.register_buffer("reverse", reverse.detach().clone())

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    hidden_states = self.visual.patch_embed(pixel_values)
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
    hidden_states = hidden_states[self.window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = self.rotary_const.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[self.window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    for layer_num, block in enumerate(self.visual.blocks):
      cu_now = self.cu_seqlens if layer_num in self.visual.fullatt_block_indexes else self.cu_window
      hidden_states = block(hidden_states, cu_seqlens=cu_now, position_embeddings=position_embeddings)

    hidden_states = self.visual.merger(hidden_states)
    return hidden_states[self.reverse, :]


class SignalHead(nn.Module):
  def __init__(self, feature_dim: int = 2048, hidden_dim: int = 128, class_count: int = len(SIGNAL_CLASSES)):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(feature_dim * 3),
      nn.Linear(feature_dim * 3, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, class_count),
    )

  def forward(self, image_features: torch.Tensor) -> torch.Tensor:
    if image_features.ndim == 2:
      x = image_features.unsqueeze(0)
    else:
      x = image_features
    x = x.float()
    pooled = torch.cat((
      x.mean(dim=1),
      x.amax(dim=1),
      x.amin(dim=1),
    ), dim=1)
    return self.net(pooled)


def _build_trt_engine(
  onnx_path: Path,
  engine_path: Path,
  *,
  fp16: bool = True,
  fp8: bool = False,
  fp4: bool = False,
  strongly_typed: bool = False,
  workspace_gb: int = 6,
) -> dict:
  logger = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(logger)
  network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  if strongly_typed:
    network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
  network = builder.create_network(network_flags)
  parser = trt.OnnxParser(network, logger)
  parse_start = time.perf_counter()
  ok = parser.parse_from_file(str(onnx_path))
  parse_ms = (time.perf_counter() - parse_start) * 1000.0
  errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
  if not ok:
    raise RuntimeError(f"TensorRT failed to parse {onnx_path}: {errors}")

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
  if fp16 and not strongly_typed:
    config.set_flag(trt.BuilderFlag.FP16)
  if fp8 and not strongly_typed:
    config.set_flag(trt.BuilderFlag.FP8)
  if fp4 and not strongly_typed:
    config.set_flag(trt.BuilderFlag.FP4)

  build_start = time.perf_counter()
  serialized = builder.build_serialized_network(network, config)
  build_ms = (time.perf_counter() - build_start) * 1000.0
  if serialized is None:
    raise RuntimeError(f"TensorRT failed to build {onnx_path}")

  engine_path.parent.mkdir(parents=True, exist_ok=True)
  serialized_bytes = bytes(serialized)
  engine_path.write_bytes(serialized_bytes)
  return {
    "onnx": str(onnx_path),
    "engine": str(engine_path),
    "parse_ms": parse_ms,
    "build_ms": build_ms,
    "engine_bytes": len(serialized_bytes),
    "strongly_typed": strongly_typed,
    "parse_errors": errors,
  }


def _parse_path_list(raw: str, default: Sequence[Path] = ()) -> list[Path]:
  paths: list[Path] = []
  for item in raw.split(","):
    item = item.strip()
    if not item:
      continue
    matches = [Path(path) for path in sorted(glob.glob(item))] if any(ch in item for ch in "*?[]") else []
    if matches:
      paths.extend(path for path in matches if path.is_file())
    else:
      path = Path(item)
      if path.is_file():
        paths.append(path)
  if not paths:
    paths.extend(path for path in default if path.is_file())
  return paths


def _augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
  out = image.copy().convert("RGB")
  if rng.random() < 0.8:
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.72, 1.35))
  if rng.random() < 0.8:
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.75, 1.35))
  if rng.random() < 0.5:
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.7, 1.25))
  if rng.random() < 0.35:
    out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.7)))
  if rng.random() < 0.45:
    w, h = out.size
    crop = rng.uniform(0.92, 1.0)
    cw, ch = int(w * crop), int(h * crop)
    left = rng.randint(0, max(0, w - cw))
    top = rng.randint(0, max(0, h - ch))
    out = out.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.BICUBIC)
  if rng.random() < 0.3:
    out = ImageOps.autocontrast(out, cutoff=rng.uniform(0.0, 1.0))
  return out


def _signal_image_sets(args) -> dict[str, list[Path]]:
  return {
    "red_stop_light": _parse_path_list(args.signal_red_images, (DEFAULT_SIGNAL_RED_IMAGE,)),
    "green_go_light": _parse_path_list(args.signal_green_images, (DEFAULT_SIGNAL_GREEN_IMAGE,)),
    "none": _parse_path_list(args.signal_none_images, (DEFAULT_SIGNAL_NONE_IMAGE, DEFAULT_IMAGE)),
  }


def _signal_feature_rows(args, processor, vision_runner: "TrtVisionRunner") -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
  rng = random.Random(args.signal_train_seed)
  features: list[torch.Tensor] = []
  targets: list[int] = []
  rows: list[dict] = []
  image_sets = _signal_image_sets(args)
  for label, paths in image_sets.items():
    if not paths:
      raise RuntimeError(f"no signal training images found for {label}")
    class_idx = SIGNAL_CLASSES.index(label)
    for path in paths:
      source = Image.open(path).convert("RGB")
      reps = max(1, args.signal_augmentations)
      for aug_idx in range(reps):
        image = source if aug_idx == 0 else _augment_image(source, rng)
        inputs = _build_vision_inputs(processor, image, args.image_mode, args.image_size)
        with torch.no_grad():
          feature, _ = vision_runner.run(inputs, 1)
        features.append(feature.detach().clone())
        targets.append(class_idx)
        rows.append({"label": label, "path": str(path), "augmentation": aug_idx})
  return torch.stack(features, dim=0), torch.tensor(targets, device="cuda", dtype=torch.long), rows


def build_signal_head(args) -> dict:
  out_dir = _signal_head_dir(args)
  out_dir.mkdir(parents=True, exist_ok=True)
  processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
  runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
  vision_runner = TrtVisionRunner(args, runtime)
  features, targets, rows = _signal_feature_rows(args, processor, vision_runner)
  head = SignalHead().cuda().train()
  opt = torch.optim.AdamW(head.parameters(), lr=args.signal_train_lr, weight_decay=1e-3)
  train_start = time.perf_counter()
  for _ in range(args.signal_train_epochs):
    opt.zero_grad(set_to_none=True)
    logits = head(features)
    loss = F.cross_entropy(logits, targets, label_smoothing=0.03)
    loss.backward()
    opt.step()
  torch.cuda.synchronize()
  train_ms = (time.perf_counter() - train_start) * 1000.0

  head.eval()
  with torch.no_grad():
    logits = head(features)
    probs = logits.softmax(dim=1)
    pred = logits.argmax(dim=1)
    correct = pred.eq(targets)
    margins = logits.topk(2, dim=1).values
    margin = margins[:, 0] - margins[:, 1]
  accuracy = float(correct.float().mean().detach().cpu())
  min_margin = float(margin.min().detach().cpu())
  per_class = {}
  for idx, name in enumerate(SIGNAL_CLASSES):
    mask = targets == idx
    per_class[name] = {
      "count": int(mask.sum().detach().cpu()),
      "accuracy": float(correct[mask].float().mean().detach().cpu()) if bool(mask.any()) else 0.0,
      "min_probability": float(probs[mask, idx].min().detach().cpu()) if bool(mask.any()) else 0.0,
      "min_margin": float(margin[mask].min().detach().cpu()) if bool(mask.any()) else 0.0,
    }

  weights_path = _signal_head_weights_path(args)
  torch.save({
    "state_dict": head.state_dict(),
    "classes": SIGNAL_CLASSES,
    "image_size": int(args.image_size),
    "vision_engine": str(vision_runner.vision_engine_path),
    "rows": rows,
  }, weights_path)

  onnx_path = _signal_head_onnx_path(args)
  engine_path = _signal_head_engine_path(args)
  sample = torch.zeros((24, 2048), device="cuda", dtype=torch.float32)
  export_start = time.perf_counter()
  with torch.no_grad():
    torch.onnx.export(
      head.float(),
      sample,
      str(onnx_path),
      input_names=["image_features"],
      output_names=["signal_logits"],
      opset_version=17,
      dynamo=False,
      do_constant_folding=True,
    )
  export_ms = (time.perf_counter() - export_start) * 1000.0
  build = _build_trt_engine(onnx_path, engine_path, fp16=True, fp4=False, workspace_gb=args.workspace_gb)

  calibration = {
    "classes": SIGNAL_CLASSES,
    "accuracy": accuracy,
    "min_margin": min_margin,
    "per_class": per_class,
    "rows": rows,
    "weights": str(weights_path),
    "onnx": str(onnx_path),
    "engine": str(engine_path),
  }
  _signal_head_calibration_path(args).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
  return {
    "kind": "signal_head",
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "vision_engine": str(vision_runner.vision_engine_path),
    "weights": str(weights_path),
    "train_samples": len(rows),
    "train_ms": train_ms,
    "export_ms": export_ms,
    "accuracy": accuracy,
    "min_margin": min_margin,
    "per_class": per_class,
    **build,
  }


def build_text_engine(args) -> dict:
  labels = _score_labels(args.score_labels)
  precision = _text_precision(args)
  out_dir = _text_engine_dir(args)
  raw_dir = args.artifact_dir / f"{precision}_raw"
  shutil.rmtree(raw_dir, ignore_errors=True)
  out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir.mkdir(parents=True, exist_ok=True)

  image = Image.open(args.image).convert("RGB")
  processor, model = _load_qwen(args.model_dir, _torch_attn_implementation(args))
  text_attn_implementation = _set_qwen_attention_implementation(model, _torch_attn_implementation(args))
  _, inputs_embeds, position_ids, last_token_mask, selected_ids = _prepare_text_tensors(
    processor,
    model,
    image,
    labels,
    args.image_mode,
    args.image_size,
    args.vehicle_state,
    args.text_seq_len,
    args,
    _score_prompt_mode(args),
  ) if _label_decision_mode(args) == "binary" else _prepare_choice_text_tensors(
    processor,
    model,
    image,
    labels,
    args.image_mode,
    args.image_size,
    args.vehicle_state,
    args.text_seq_len,
    args,
  )

  text_output = _text_output(args)
  output_name = _text_output_tensor_name(text_output)
  if text_output == "hidden":
    wrapper = TextHidden(model.model.language_model).cuda().half().eval()
  elif text_output == "full_hidden":
    wrapper = TextFullHidden(model.model.language_model).cuda().half().eval()
  elif text_output == "fixed_logits":
    output_index = int(getattr(args, "fixed_output_index", -1))
    if output_index < 0:
      if _label_decision_mode(args) != "choice":
        raise ValueError("fixed_logits currently requires choice mode when --fixed-output-index is not set")
      output_index = int(last_token_mask.squeeze(-1).argmax(dim=1)[0].detach().cpu())
    wrapper = TextFixedScore(model.model.language_model, model.lm_head, selected_ids, output_index).cuda().half().eval()
  else:
    wrapper = TextScore(model.model.language_model, model.lm_head, selected_ids).cuda().half().eval()
  with torch.no_grad():
    ref = wrapper(inputs_embeds, position_ids, last_token_mask)
    torch.cuda.synchronize()

  if getattr(args, "label_keyed_text_engine", False):
    final_onnx = _keyed_text_onnx_path(args, labels)
    engine_path = _keyed_text_engine_path(args, labels)
  else:
    final_onnx = _generic_text_onnx_path(args)
    engine_path = _generic_text_engine_path(args)

  quant_ms = 0.0
  postprocess_ms = 0.0
  save_ms = 0.0
  export_start = time.perf_counter()
  if precision == "fp16":
    with torch.no_grad():
      torch.onnx.export(
        wrapper,
        (inputs_embeds, position_ids, last_token_mask),
        str(final_onnx),
        input_names=["inputs_embeds", "position_ids", "last_token_mask"],
        output_names=[output_name],
        opset_version=21,
        dynamo=False,
        do_constant_folding=True,
        external_data=True,
      )
    raw_export_ms = (time.perf_counter() - export_start) * 1000.0
  else:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
    import onnx
    if precision == "fp8":
      from modelopt.onnx.export.fp8_exporter import FP8QuantExporter as QuantExporter
      quant_cfg = mtq.FP8_DEFAULT_CFG
    elif precision == "nvfp4":
      from modelopt.onnx.export.nvfp4_exporter import NVFP4QuantExporter as QuantExporter
      quant_cfg = mtq.NVFP4_DEFAULT_CFG
    else:
      raise RuntimeError(f"unsupported quantized text precision: {precision}")

    quant_start = time.perf_counter()
    qwrapper = mtq.quantize(wrapper, quant_cfg, forward_loop=lambda mdl: mdl(inputs_embeds, position_ids, last_token_mask))
    torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - quant_start) * 1000.0
    qwrapper.eval()

    raw_onnx = raw_dir / f"qwen_text_36layer_{precision}_raw.onnx"
    with torch.no_grad(), configure_linear_module_onnx_quantizers(qwrapper):
      torch.onnx.export(
        qwrapper,
        (inputs_embeds, position_ids, last_token_mask),
        str(raw_onnx),
        input_names=["inputs_embeds", "position_ids", "last_token_mask"],
        output_names=[output_name],
        opset_version=21,
        dynamo=False,
        do_constant_folding=True,
        external_data=True,
      )
    raw_export_ms = (time.perf_counter() - export_start) * 1000.0

    post_start = time.perf_counter()
    raw_model = onnx.load(str(raw_onnx))
    processed = QuantExporter.process_model(raw_model)
    postprocess_ms = (time.perf_counter() - post_start) * 1000.0

    save_start = time.perf_counter()
    external_data = final_onnx.with_suffix(final_onnx.suffix + ".data")
    external_data.unlink(missing_ok=True)
    onnx.save_model(
      processed,
      str(final_onnx),
      save_as_external_data=True,
      all_tensors_to_one_file=True,
      location=external_data.name,
      size_threshold=1024,
      convert_attribute=False,
    )
    save_ms = (time.perf_counter() - save_start) * 1000.0
  shutil.rmtree(raw_dir, ignore_errors=True)

  text_onnx_fidelity = {"enabled": False}
  if bool(getattr(args, "verify_text_onnx_fidelity", False)):
    text_onnx_fidelity = _verify_text_onnx_fidelity(
      final_onnx,
      ref,
      inputs_embeds,
      position_ids,
      last_token_mask,
      output_name=output_name,
      labels=labels,
      processor=processor,
      model=model,
      text_output=text_output,
      label_decision_mode=_label_decision_mode(args),
      max_mean_error=float(getattr(args, "text_onnx_max_mean_error", 0.05)),
      max_abs_error=float(getattr(args, "text_onnx_max_error", 2.0)),
      require_choice_match=bool(getattr(args, "text_onnx_require_choice_match", True)),
    )

  build = _build_trt_engine(
    final_onnx,
    engine_path,
    fp16=True,
    fp8=(precision == "fp8"),
    fp4=(precision == "nvfp4"),
    strongly_typed=bool(getattr(args, "text_strongly_typed", False)),
    workspace_gb=args.workspace_gb,
  )
  return {
    "kind": f"text_{precision}",
    "labels": labels,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "text_seq_len": int(args.text_seq_len),
    "text_precision": precision,
    "text_output": text_output,
    "fixed_output_index": int(getattr(args, "fixed_output_index", -1)) if text_output != "fixed_logits" else int(wrapper.output_index),
    "label_decision_mode": _label_decision_mode(args),
    "text_position_mode": _text_position_mode(args),
    "text_position_dtype": _text_position_dtype(args),
    "torch_attn_implementation": text_attn_implementation,
    "ref_mean": float(ref.mean()),
    "text_onnx_fidelity": text_onnx_fidelity,
    "quant_ms": quant_ms,
    "raw_export_ms": raw_export_ms,
    "postprocess_ms": postprocess_ms,
    "save_ms": save_ms,
    **build,
  }


def build_vision_engine(args) -> dict:
  precision = _vision_precision(args)
  out_dir = args.artifact_dir / f"vision_static_{precision}"
  out_dir.mkdir(parents=True, exist_ok=True)
  image = Image.open(args.image).convert("RGB")
  processor, model = _load_qwen(args.model_dir, _torch_attn_implementation(args))
  visual = model.model.visual
  visual.config._attn_implementation = "eager"

  images, _ = _inference_images(image, args.image_mode, args.image_size)
  prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": images[0]}, {"type": "text", "text": "x"}]}],
    tokenize=False,
    add_generation_prompt=True,
  )
  inputs = processor(text=[prompt], images=images[:1], padding=True, return_tensors="pt").to("cuda")
  pixel_values = inputs.pixel_values.contiguous()
  grid = inputs.image_grid_thw[:1].contiguous()

  with torch.no_grad():
    rotary = visual.rot_pos_emb(grid)
    window_index, cu_window_list = visual.get_window_index(grid)
    cu_window = torch.tensor(cu_window_list, device=pixel_values.device, dtype=torch.int32)
    cu_window = torch.unique_consecutive(cu_window)
    cu_seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    reverse = torch.argsort(window_index)

  wrapper = VisionStatic(visual, rotary, window_index, cu_window, cu_seqlens, reverse).cuda().half().eval()
  with torch.no_grad():
    ref_hf = visual(pixel_values, grid)
    ref = wrapper(pixel_values)
    torch.cuda.synchronize()
    mse = float((ref.float() - ref_hf.float()).square().mean())

  onnx_path = _vision_onnx_path(args)
  engine_path = _vision_engine_path(args)
  export_start = time.perf_counter()
  with torch.no_grad():
    torch.onnx.export(
      wrapper,
      (pixel_values,),
      str(onnx_path),
      input_names=["pixel_values"],
      output_names=["image_features"],
      opset_version=20,
      dynamo=False,
      do_constant_folding=True,
      external_data=True,
    )
  export_ms = (time.perf_counter() - export_start) * 1000.0

  build = _build_trt_engine(onnx_path, engine_path, fp16=(precision == "fp16"), fp4=False, workspace_gb=args.workspace_gb)
  return {
    "kind": f"vision_static_{precision}",
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "vision_precision": precision,
    "pixel_shape": tuple(pixel_values.shape),
    "image_feature_shape": tuple(ref.shape),
    "static_wrapper_mse_vs_hf": mse,
    "export_ms": export_ms,
    **build,
  }


def _load_engine(runtime, engine_path: Path):
  if not engine_path.exists():
    raise FileNotFoundError(engine_path)
  engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
  if engine is None:
    raise RuntimeError(f"failed to deserialize {engine_path}")
  return engine, engine.create_execution_context()


def _trt_shape(engine, tensor_name: str) -> tuple[int, ...]:
  return tuple(int(dim) for dim in engine.get_tensor_shape(tensor_name))


def _trt_tensor_names(engine) -> set[str]:
  return {engine.get_tensor_name(idx) for idx in range(engine.num_io_tensors)}


def _trt_torch_dtype(engine, tensor_name: str) -> torch.dtype:
  dtype = engine.get_tensor_dtype(tensor_name)
  if dtype == trt.DataType.FLOAT:
    return torch.float32
  if dtype == trt.DataType.HALF:
    return torch.float16
  if dtype == trt.DataType.INT32:
    return torch.int32
  if dtype == trt.DataType.INT64:
    return torch.int64
  if hasattr(trt.DataType, "BF16") and dtype == trt.DataType.BF16:
    return torch.bfloat16
  raise RuntimeError(f"unsupported TensorRT dtype for {tensor_name}: {dtype}")


def _file_info(path: Path) -> dict:
  exists = path.exists()
  return {
    "path": str(path),
    "exists": exists,
    "bytes": path.stat().st_size if exists else 0,
    "mtime": path.stat().st_mtime if exists else 0.0,
  }


def _engine_info(runtime: trt.Runtime, path: Path) -> tuple[dict, list[str]]:
  issues: list[str] = []
  info = _file_info(path)
  if not path.exists():
    issues.append(f"missing engine: {path}")
    info["deserialized"] = False
    info["tensors"] = {}
    return info, issues
  engine = runtime.deserialize_cuda_engine(path.read_bytes())
  if engine is None:
    issues.append(f"failed to deserialize engine: {path}")
    info["deserialized"] = False
    info["tensors"] = {}
    return info, issues
  tensors = {}
  for idx in range(engine.num_io_tensors):
    name = engine.get_tensor_name(idx)
    tensors[name] = {
      "shape": _trt_shape(engine, name),
      "dtype": str(engine.get_tensor_dtype(name)),
      "mode": str(engine.get_tensor_mode(name)),
    }
  info["deserialized"] = True
  info["tensors"] = tensors
  return info, issues


def _check_shape(info: dict, tensor_name: str, expected: tuple[int, ...], issues: list[str], label: str) -> None:
  tensors = info.get("tensors", {})
  actual = tuple(tensors.get(tensor_name, {}).get("shape", ()))
  if actual != expected:
    issues.append(f"{label} {tensor_name} shape {actual} != expected {expected}")


def _nvcc_info() -> dict:
  cuda_path = Path(os.environ.get("CUDA_PATH", ""))
  nvcc = cuda_path / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
  if not nvcc.exists():
    found = shutil.which("nvcc")
    nvcc = Path(found) if found else nvcc
  info = {
    "path": str(nvcc),
    "exists": nvcc.exists(),
    "version": "",
    "has_compute_120": False,
    "has_sm_120": False,
  }
  if not nvcc.exists():
    return info
  try:
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=15, check=False)
    arch = subprocess.run([str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, timeout=15, check=False)
    code = subprocess.run([str(nvcc), "--list-gpu-code"], capture_output=True, text=True, timeout=15, check=False)
  except Exception as exc:
    info["version"] = repr(exc)
    return info
  info["version"] = version.stdout.strip()
  info["has_compute_120"] = "compute_120" in arch.stdout
  info["has_sm_120"] = "sm_120" in code.stdout
  return info


class TrtVisionRunner:
  def __init__(self, args, runtime: trt.Runtime):
    self.args = args
    self.vision_engine_path = args.vision_engine or _vision_engine_path(args)
    self.vision_engine, self.vision_ctx = _load_engine(runtime, self.vision_engine_path)
    self.vision_stream = torch.cuda.Stream()
    self.vision_in_shape = _trt_shape(self.vision_engine, "pixel_values")
    self.vision_out_shape = _trt_shape(self.vision_engine, "image_features")
    self.vision_out = torch.empty(self.vision_out_shape, device="cuda", dtype=torch.float16)
    self.vision_ctx.set_tensor_address("image_features", self.vision_out.data_ptr())

  def run(self, inputs, label_count: int) -> tuple[torch.Tensor, float]:
    rows_per_image = inputs.pixel_values.shape[0] // label_count
    pixel_one = inputs.pixel_values[:rows_per_image].contiguous()
    if tuple(pixel_one.shape) != self.vision_in_shape:
      raise RuntimeError(f"vision input shape {tuple(pixel_one.shape)} does not match engine {self.vision_in_shape}")
    self.vision_ctx.set_tensor_address("pixel_values", pixel_one.data_ptr())
    start = time.perf_counter()
    with torch.cuda.stream(self.vision_stream):
      if not self.vision_ctx.execute_async_v3(self.vision_stream.cuda_stream):
        raise RuntimeError("vision TensorRT execute_async_v3 failed")
    self.vision_stream.synchronize()
    return self.vision_out, (time.perf_counter() - start) * 1000.0


class TrtSignalHeadRunner:
  def __init__(self, args, runtime: trt.Runtime):
    self.args = args
    self.engine_path = _signal_head_engine_path(args)
    self.engine, self.ctx = _load_engine(runtime, self.engine_path)
    self.stream = torch.cuda.Stream()
    self.in_shape = _trt_shape(self.engine, "image_features")
    self.out_shape = _trt_shape(self.engine, "signal_logits")
    self.in_dtype = _trt_torch_dtype(self.engine, "image_features")
    self.out_dtype = _trt_torch_dtype(self.engine, "signal_logits")
    self.out = torch.empty(self.out_shape, device="cuda", dtype=self.out_dtype)
    self.ctx.set_tensor_address("signal_logits", self.out.data_ptr())
    if self.out_shape[-1] != len(SIGNAL_CLASSES):
      raise RuntimeError(f"signal head output width {self.out_shape[-1]} != {len(SIGNAL_CLASSES)}")

  def run(self, image_features: torch.Tensor) -> tuple[torch.Tensor, float]:
    features = image_features.to(dtype=self.in_dtype).contiguous()
    if tuple(features.shape) != self.in_shape:
      raise RuntimeError(f"signal head input shape {tuple(features.shape)} does not match engine {self.in_shape}")
    self.ctx.set_tensor_address("image_features", features.data_ptr())
    start = time.perf_counter()
    with torch.cuda.stream(self.stream):
      if not self.ctx.execute_async_v3(self.stream.cuda_stream):
        raise RuntimeError("signal head TensorRT execute_async_v3 failed")
    self.stream.synchronize()
    return self.out, (time.perf_counter() - start) * 1000.0


def _signal_scores_from_logits(logits: torch.Tensor) -> tuple[list[str], dict[str, float], dict]:
  values = logits.reshape(-1).float()
  probs = values.softmax(dim=0)
  top = torch.topk(values, 2)
  pred_idx = int(top.indices[0].detach().cpu())
  margin = float((top.values[0] - top.values[1]).detach().cpu())
  probability = float(probs[pred_idx].detach().cpu())
  label_scores = {
    "red_stop_light": float((values[SIGNAL_CLASSES.index("red_stop_light")] - values[SIGNAL_CLASSES.index("none")]).detach().cpu()),
    "green_go_light": float((values[SIGNAL_CLASSES.index("green_go_light")] - values[SIGNAL_CLASSES.index("none")]).detach().cpu()),
  }
  if pred_idx == SIGNAL_CLASSES.index("red_stop_light"):
    labels = ["red_stop_light"]
  elif pred_idx == SIGNAL_CLASSES.index("green_go_light"):
    labels = ["green_go_light"]
  else:
    labels = ["none"]
  return labels, label_scores, {
    "class": SIGNAL_CLASSES[pred_idx],
    "probability": probability,
    "margin": margin,
    "logits": {name: float(values[idx].detach().cpu()) for idx, name in enumerate(SIGNAL_CLASSES)},
    "probabilities": {name: float(probs[idx].detach().cpu()) for idx, name in enumerate(SIGNAL_CLASSES)},
  }


class QwenTrtVisualHeadScorer:
  def __init__(self, args, *, processor=None, runtime: trt.Runtime | None = None):
    self.args = args
    self.processor = processor or AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if runtime is None:
      runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    self.runtime = runtime
    self.vision_runner = TrtVisionRunner(args, runtime)
    self.signal_runner = TrtSignalHeadRunner(args, runtime)

  def warmup(self, count: int) -> None:
    warm = Image.open(self.args.image).convert("RGB") if self.args.image.exists() else Image.new("RGB", (384, 216), (20, 20, 20))
    for _ in range(max(0, count)):
      self.score(warm, self.args.vehicle_state)

  def score(self, image: Image.Image, vehicle_state: str = "") -> dict:
    del vehicle_state
    parts: dict[str, float] = {}
    wall_start = time.perf_counter()
    start = time.perf_counter()
    inputs = _build_vision_inputs(self.processor, image, self.args.image_mode, self.args.image_size)
    torch.cuda.synchronize()
    parts["processor_ms"] = (time.perf_counter() - start) * 1000.0
    image_features, parts["trt_vision_ms"] = self.vision_runner.run(inputs, 1)
    logits, parts["signal_head_ms"] = self.signal_runner.run(image_features)
    labels, scores, signal = _signal_scores_from_logits(logits)
    if signal["probability"] < self.args.signal_min_probability or signal["margin"] < self.args.signal_min_margin:
      labels = ["none"]
    rtp_text = _labels_to_rtp(tuple(labels))
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    parts["total_ms"] = total_ms
    return {
      "text": rtp_text,
      "rtp_text": rtp_text,
      "labels_text": ",".join(labels),
      "labels": labels,
      "label_mode": "visual_head",
      "image_mode": self.args.image_mode,
      "label_scores": scores,
      "signal": signal,
      "generated_token_count": 0,
      "prefill_ms": parts["processor_ms"] + parts["trt_vision_ms"],
      "decode_ms": parts["signal_head_ms"],
      "total_ms": total_ms,
      "timings_ms": parts,
      "backend": f"qwen2.5-vl-trt-visual-head-{self.args.image_mode}{self.args.image_size}",
    }


def _scores_from_selected_logits(selected_logits: torch.Tensor, yes_count: int, labels: Sequence[str]) -> dict[str, float]:
  logits = selected_logits[:, 0, :]
  yes = logits[:, :yes_count].max(dim=1).values
  no = logits[:, yes_count:].max(dim=1).values
  raw_scores = (yes - no).detach().cpu().tolist()
  return {label: float(score) for label, score in zip(labels, raw_scores, strict=True)}


def _choice_scores_from_selected_logits(
  selected_logits: torch.Tensor,
  labels: Sequence[str],
  token_words: Sequence[str] = CHOICE_WORDS,
  min_margin_by_label: dict[str, float] | None = None,
  default_min_margin: float | None = None,
) -> tuple[tuple[str, ...], dict[str, float], dict]:
  values = selected_logits.reshape(-1).float()
  if len(values) != len(token_words):
    raise RuntimeError(f"choice logits width {len(values)} does not match token word count {len(token_words)}")
  word_scores: dict[str, float] = {}
  for idx, word in enumerate(token_words):
    score = float(values[idx].detach().cpu())
    word_scores[word] = max(score, word_scores.get(word, float("-inf")))
  return _choice_scores_from_word_scores(word_scores, labels, min_margin_by_label, default_min_margin)


def _choice_scores_from_word_scores(
  word_scores: dict[str, float],
  labels: Sequence[str],
  min_margin_by_label: dict[str, float] | None = None,
  default_min_margin: float | None = None,
) -> tuple[tuple[str, ...], dict[str, float], dict]:
  spec = _choice_group_spec(labels)
  for word in spec["allowed"]:
    if word not in word_scores:
      raise RuntimeError(f"choice score output is missing required word {word!r}")
  if spec.get("prompt_kind") == "traffic_signal":
    required_words = ("A", "B", "C")
    for word in required_words:
      if word not in word_scores:
        raise RuntimeError(f"traffic-signal choice score output is missing required word {word!r}")
    # Use neutral answer letters for signal state to avoid priors from words such as
    # red, green, stop, none, and absent. The neutral/no-signal option is A because
    # Qwen has a strong first-option prior; a signal answer must beat neutral and the
    # opposite signal choice.
    red_score = min(word_scores["B"] - word_scores["A"], word_scores["B"] - word_scores["C"])
    green_score = min(word_scores["C"] - word_scores["A"], word_scores["C"] - word_scores["B"])
    label_scores = {
      "red_stop_light": red_score,
      "green_go_light": green_score,
    }
    selected_label = None
    best_label = max(label_scores, key=label_scores.get)
    spec_min_margin = float(spec.get("min_margin", 0.0))
    min_margin = spec_min_margin
    if min_margin_by_label and best_label in min_margin_by_label:
      min_margin = float(min_margin_by_label[best_label])
    elif default_min_margin is not None:
      min_margin = float(default_min_margin)
    rejected_by_margin = label_scores[best_label] < min_margin
    if not rejected_by_margin:
      selected_label = best_label
    answer = "A"
    if selected_label == "red_stop_light":
      answer = "B"
    elif selected_label == "green_go_light":
      answer = "C"
    selected = (selected_label,) if selected_label else ("none",)
    return selected, label_scores, {
      "answer": answer,
      "allowed": spec["allowed"],
      "neutral": "A",
      "min_margin": min_margin,
      "spec_min_margin": spec_min_margin,
      "competitor_margin": label_scores[best_label],
      "competitor_min_margin": 0.0,
      "rejected_by_margin": rejected_by_margin,
      "word_scores": {word: word_scores[word] for word in spec["allowed"]},
      "all_word_scores": {word: word_scores.get(word, float("-inf")) for word in CHOICE_WORDS},
      "traffic_signal_calibration": {
        "B_minus_A": word_scores["B"] - word_scores["A"],
        "B_minus_C": word_scores["B"] - word_scores["C"],
        "C_minus_A": word_scores["C"] - word_scores["A"],
        "C_minus_B": word_scores["C"] - word_scores["B"],
      },
    }
  neutral_candidates = [word for word in spec["allowed"] if word not in spec["word_to_label"]]
  neutral_word = "none" if "none" in neutral_candidates else (neutral_candidates[-1] if neutral_candidates else None)
  if neutral_word is None:
    label_scores = {}
    for word, label in spec["word_to_label"].items():
      other_scores = [word_scores[other] for other in spec["allowed"] if other != word]
      label_scores[label] = word_scores[word] - max(other_scores, default=float("-inf"))
  else:
    none_score = word_scores[neutral_word]
    label_scores = {
      label: word_scores[word] - none_score
      for word, label in spec["word_to_label"].items()
    }
  best_word = max(spec["allowed"], key=lambda word: word_scores[word])
  selected_label = spec["word_to_label"].get(best_word)
  spec_min_margin = float(spec.get("min_margin", 0.0))
  min_margin = spec_min_margin
  if selected_label is not None:
    if min_margin_by_label and selected_label in min_margin_by_label:
      min_margin = float(min_margin_by_label[selected_label])
    elif default_min_margin is not None:
      min_margin = float(default_min_margin)
  competitor_margin = None
  competitor_min_margin = float(spec.get("competitor_min_margin", 0.0))
  if selected_label and competitor_min_margin > 0.0:
    selected_word = best_word
    competitor_scores = [
      word_scores[word]
      for word, label in spec["word_to_label"].items()
      if label != selected_label and word != selected_word
    ]
    if competitor_scores:
      competitor_margin = word_scores[selected_word] - max(competitor_scores)
  rejected_by_margin = False
  if selected_label and label_scores[selected_label] < min_margin:
    rejected_by_margin = True
    selected_label = None
  if selected_label and competitor_margin is not None and competitor_margin < competitor_min_margin:
    rejected_by_margin = True
    selected_label = None
  selected = (selected_label,) if selected_label else ("none",)
  return selected, label_scores, {
    "answer": best_word,
    "allowed": spec["allowed"],
    "neutral": neutral_word,
    "min_margin": min_margin,
    "spec_min_margin": spec_min_margin,
    "competitor_margin": competitor_margin,
    "competitor_min_margin": competitor_min_margin,
    "rejected_by_margin": rejected_by_margin,
    "word_scores": {word: word_scores[word] for word in spec["allowed"]},
    "all_word_scores": {word: word_scores.get(word, float("-inf")) for word in CHOICE_WORDS},
  }


def _is_construction_group(labels: Sequence[str]) -> bool:
  group = frozenset(labels)
  return group == CONSTRUCTION_LABEL_SET or group == CONSTRUCTION_EDGE_LABEL_SET or group == CONSTRUCTION_ACTION_LABEL_SET or group == CONSTRUCTION_SHIFT_LABEL_SET or group == CONSTRUCTION_CANDIDATE_LABEL_SET


def _construction_labels_only(labels: Sequence[str]) -> tuple[str, ...]:
  return tuple(label for label in labels if label in CONSTRUCTION_LABEL_SET or label in CONSTRUCTION_EDGE_LABEL_SET or label in CONSTRUCTION_ACTION_LABEL_SET or label in CONSTRUCTION_SHIFT_LABEL_SET or label in CONSTRUCTION_CANDIDATE_LABEL_SET)


def _mirror_construction_labels_to_original(labels: Sequence[str]) -> tuple[str, ...]:
  mapped = tuple(CONSTRUCTION_MIRROR_LABEL[label] for label in labels if label in CONSTRUCTION_MIRROR_LABEL)
  return mapped if mapped else ("none",)


def _apply_construction_mirror_consistency(
  original: dict,
  mirrored: dict,
  *,
  negative_clear_threshold: float,
) -> dict:
  original_construction = _construction_labels_only(tuple(original.get("labels", ())))
  mirrored_construction = _construction_labels_only(tuple(mirrored.get("labels", ())))
  mirrored_as_original = _mirror_construction_labels_to_original(mirrored_construction)
  mirrored_as_original_construction = _construction_labels_only(mirrored_as_original)
  consistent = original_construction == mirrored_as_original_construction

  choice = dict(original.get("choice") or {})
  choice["construction_mirror_consistency"] = {
    "enabled": True,
    "accepted": consistent,
    "original_labels": list(original_construction) if original_construction else ["none"],
    "mirrored_labels": list(mirrored_construction) if mirrored_construction else ["none"],
    "mirrored_labels_mapped_to_original": list(mirrored_as_original_construction) if mirrored_as_original_construction else ["none"],
    "mirrored_choice": mirrored.get("choice"),
  }

  adjusted = dict(original)
  adjusted["choice"] = choice
  if consistent:
    return adjusted

  labels = tuple(
    label for label in tuple(original.get("labels", ()))
    if label not in CONSTRUCTION_LABEL_SET and label not in CONSTRUCTION_EDGE_LABEL_SET and label not in CONSTRUCTION_ACTION_LABEL_SET and label != "none"
  )
  if not labels:
    labels = ("none",)

  scores = dict(original.get("label_scores") or {})
  forced_negative = -max(negative_clear_threshold + 0.25, 0.25)
  for label in (*CONSTRUCTION_LABELS, *CONSTRUCTION_EDGE_LABELS, *CONSTRUCTION_ACTION_LABELS, *CONSTRUCTION_SHIFT_LABELS):
    scores[label] = min(float(scores.get(label, forced_negative)), forced_negative)

  rtp_text = _labels_to_rtp(labels)
  adjusted.update({
    "labels": list(labels),
    "labels_text": ",".join(labels),
    "label_scores": scores,
    "rtp_text": rtp_text,
    "text": rtp_text,
  })
  return adjusted


def _apply_construction_mirror_fusion(original: dict, mirrored: dict, labels: Sequence[str] | None = None) -> dict:
  labels = tuple(labels or original.get("label_group") or original.get("score_labels") or original.get("labels") or ())
  construction_labels = _construction_labels_only(labels)
  if not construction_labels or set(construction_labels) not in (CONSTRUCTION_LABEL_SET, CONSTRUCTION_EDGE_LABEL_SET, CONSTRUCTION_ACTION_LABEL_SET, CONSTRUCTION_SHIFT_LABEL_SET):
    construction_labels = _construction_labels_only(tuple(original.get("labels", ())))
  if not construction_labels:
    return original

  spec = _choice_group_spec(construction_labels)
  label_to_word = {label: word for word, label in spec["word_to_label"].items()}
  original_choice = dict(original.get("choice") or {})
  mirrored_choice = dict(mirrored.get("choice") or {})
  original_words = dict(original_choice.get("word_scores") or {})
  mirrored_words = dict(mirrored_choice.get("word_scores") or {})
  if not original_words or not mirrored_words:
    return original

  mapped_mirror_word_scores: dict[str, float] = {}
  for word in spec["allowed"]:
    label = spec["word_to_label"].get(word)
    if label is None:
      mapped_mirror_word_scores[word] = float(mirrored_words.get(word, float("-inf")))
      continue
    mirrored_label = CONSTRUCTION_MIRROR_LABEL.get(label)
    mirrored_word = label_to_word.get(mirrored_label)
    if mirrored_word is None:
      mapped_mirror_word_scores[word] = float("-inf")
      continue
    mapped_mirror_word_scores[word] = float(mirrored_words.get(mirrored_word, float("-inf")))

  original_choice_labels, original_scores, original_scored_choice = _choice_scores_from_word_scores(original_words, construction_labels)
  mapped_choice_labels, mapped_scores, mapped_choice = _choice_scores_from_word_scores(mapped_mirror_word_scores, construction_labels)
  original_selected = tuple(label for label in original_choice_labels if label != "none")
  mapped_selected = tuple(label for label in mapped_choice_labels if label != "none")

  fused_word_scores: dict[str, float] | None = None
  labels_out: tuple[str, ...] | None = None
  if original_selected and mapped_selected and original_selected == mapped_selected:
    fused_word_scores = {
      word: float(original_words.get(word, float("-inf"))) + float(mapped_mirror_word_scores.get(word, float("-inf")))
      for word in spec["allowed"]
    }
    choice_labels, scores, choice = _choice_scores_from_word_scores(fused_word_scores, construction_labels)
    fusion_policy = "agreement_fused"
  elif original_selected:
    selected_label = original_selected[0]
    selected_margin = float(original_scores.get(selected_label, float("-inf")))
    if mapped_selected and mapped_selected != original_selected and selected_margin < 1.0:
      labels_out = ("none",)
      scores = {
        label: min(float(original_scores.get(label, -2.25)), -2.25)
        for label in construction_labels
      }
      choice = dict(original_scored_choice)
      fusion_policy = "disagreement_cleared_weak_original"
    else:
      choice_labels, scores, choice = original_choice_labels, original_scores, original_scored_choice
      labels_out = original_selected
      fusion_policy = "kept_original"
  else:
    choice_labels, scores, choice = original_choice_labels, original_scores, original_scored_choice
    fusion_policy = "kept_original_none"
    if bool(original_scored_choice.get("rejected_by_margin", False)):
      labels_out = ("none",)
      scores = {
        label: min(float(original_scores.get(label, -2.25)), -2.25)
        for label in construction_labels
      }
      fusion_policy = "cleared_margin_rejected_original"

  if labels_out is None:
    selected = tuple(label for label in choice_labels if label != "none")
    labels_out = selected if selected else ("none",)
  rtp_text = _with_rtp_confidence(
    _labels_to_rtp(labels_out),
    _score_calibrated_construction_confidence(labels_out, scores),
  )

  choice = dict(choice)
  choice["construction_mirror_fusion"] = {
    "enabled": True,
    "policy": fusion_policy,
    "original_choice": original_choice,
    "mirrored_choice": mirrored_choice,
    "original_scored_choice": original_scored_choice,
    "mapped_mirror_choice": mapped_choice,
    "mapped_mirror_word_scores": {word: mapped_mirror_word_scores[word] for word in spec["allowed"]},
    "fused_word_scores": None if fused_word_scores is None else {word: fused_word_scores[word] for word in spec["allowed"]},
    "selected_labels": list(labels_out),
  }

  adjusted = dict(original)
  adjusted.update({
    "labels": list(labels_out),
    "labels_text": ",".join(labels_out),
    "label_scores": scores,
    "choice": choice,
    "rtp_text": rtp_text,
    "text": rtp_text,
  })
  return adjusted


def _score_calibrated_construction_confidence(
  labels: Sequence[str],
  scores: dict[str, float],
  base_confidence: float = 0.72,
  *,
  strong_edge_immediate: bool = False,
) -> float | None:
  label_set = set(labels)
  if "construction_drive_left" in label_set and "construction_drive_right" not in label_set:
    selected, other = "construction_drive_left", "construction_drive_right"
  elif "construction_drive_right" in label_set and "construction_drive_left" not in label_set:
    selected, other = "construction_drive_right", "construction_drive_left"
  elif "construction_shift_left" in label_set and "construction_shift_right" not in label_set:
    selected, other = "construction_shift_left", "construction_shift_right"
  elif "construction_shift_right" in label_set and "construction_shift_left" not in label_set:
    selected, other = "construction_shift_right", "construction_shift_left"
  elif "construction_left" in label_set and "construction_right" not in label_set:
    selected, other = "construction_left", "construction_right"
  elif "construction_right" in label_set and "construction_left" not in label_set:
    selected, other = "construction_right", "construction_left"
  elif "construction_blue_edge" in label_set and "construction_purple_edge" not in label_set:
    selected, other = "construction_blue_edge", "construction_purple_edge"
  elif "construction_purple_edge" in label_set and "construction_blue_edge" not in label_set:
    selected, other = "construction_purple_edge", "construction_blue_edge"
  elif "construction_blocks_left_candidate" in label_set and "construction_blocks_right_candidate" not in label_set:
    selected, other = "construction_blocks_left_candidate", "construction_blocks_right_candidate"
  elif "construction_blocks_right_candidate" in label_set and "construction_blocks_left_candidate" not in label_set:
    selected, other = "construction_blocks_right_candidate", "construction_blocks_left_candidate"
  else:
    return None
  selected_score = float(scores.get(selected, float("-inf")))
  other_score = float(scores.get(other, float("-inf")))
  if not math.isfinite(selected_score) or not math.isfinite(other_score):
    return base_confidence
  margin = selected_score - other_score
  if selected in CONSTRUCTION_ACTION_LABEL_SET:
    if selected_score >= CONSTRUCTION_ACTION_IMMEDIATE_SCORE and margin >= CONSTRUCTION_ACTION_IMMEDIATE_MARGIN:
      return 0.96
    if selected_score >= 1.0 and margin >= 1.0:
      return 0.84
    if selected_score >= 1.25 and margin >= 0.75:
      return 0.80
    return base_confidence
  if selected in CONSTRUCTION_LABEL_SET or selected in CONSTRUCTION_EDGE_LABEL_SET or selected in CONSTRUCTION_CANDIDATE_LABEL_SET:
    same_side_edge = None
    if selected in CONSTRUCTION_LABEL_SET:
      selected_side = "left" if selected == "construction_left" else "right"
      same_side_edge = "construction_blue_edge" if selected_side == "left" else "construction_purple_edge"
    if same_side_edge in label_set:
      opposite_edge = "construction_purple_edge" if same_side_edge == "construction_blue_edge" else "construction_blue_edge"
      edge_score = float(scores.get(same_side_edge, float("-inf")))
      opposite_edge_score = float(scores.get(opposite_edge, float("-inf")))
      edge_margin = edge_score - opposite_edge_score
      if (
        math.isfinite(edge_score) and
        math.isfinite(edge_margin) and
        edge_score >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE and
        edge_margin >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN
      ):
        return 0.96
    if (
      strong_edge_immediate and
      selected in CONSTRUCTION_EDGE_LABEL_SET and
      selected_score >= CONSTRUCTION_EDGE_BOOTSTRAP_SCORE and
      other_score <= CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX and
      margin >= CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN
    ):
      return 0.96
    if (
      strong_edge_immediate and
      selected in CONSTRUCTION_EDGE_LABEL_SET and
      selected_score >= CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE and
      margin >= CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN
    ):
      return 0.96
    if (
      strong_edge_immediate and
      selected in CONSTRUCTION_EDGE_LABEL_SET and
      selected_score >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE and
      margin >= CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN
    ):
      return 0.96
    if (
      selected in CONSTRUCTION_CANDIDATE_LABEL_SET and
      selected_score >= CONSTRUCTION_CANDIDATE_IMMEDIATE_SCORE and
      other_score <= CONSTRUCTION_SIDE_IMMEDIATE_LOCK_OPPOSITE_MAX and
      margin >= CONSTRUCTION_CANDIDATE_IMMEDIATE_MARGIN
    ):
      return 0.96
    if (
      selected in CONSTRUCTION_LABEL_SET and
      selected_score >= CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE and
      margin >= CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN
    ):
      return 0.96
    if selected_score >= 0.6 and margin >= 1.0:
      return 0.84
    if selected_score >= 0.3 and margin >= 0.6:
      return 0.80
    return base_confidence
  if selected_score >= 1.0 and margin >= 1.0:
    return 0.84
  if selected_score >= 1.25 and margin >= 0.75:
    return 0.80
  return base_confidence


def _resolve_candidate_obstruction_scores(labels: Sequence[str], scores: dict[str, float], margin: float) -> tuple[str, ...]:
  label_set = set(labels)
  left_label = "construction_blocks_left_candidate"
  right_label = "construction_blocks_right_candidate"
  if left_label not in label_set or right_label not in label_set:
    return tuple(labels)
  left_score = float(scores.get(left_label, float("nan")))
  right_score = float(scores.get(right_label, float("nan")))
  if not math.isfinite(left_score) or not math.isfinite(right_score) or abs(left_score - right_score) < max(0.0, float(margin)):
    label_set.discard(left_label)
    label_set.discard(right_label)
  elif left_score > right_score:
    label_set.discard(right_label)
  else:
    label_set.discard(left_label)
  ordered = tuple(label for label in SCORE_QUESTIONS if label in label_set)
  return ordered if ordered else ("none",)


def _apply_candidate_relative_choice(
  choice_labels: Sequence[str],
  scores: dict[str, float],
  choice: dict | None,
  labels: Sequence[str],
  *,
  enabled: bool,
  min_margin: float,
  neutral_margin: float = CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN,
) -> tuple[tuple[str, ...], dict[str, float], dict | None]:
  if not enabled or set(labels) != CONSTRUCTION_CANDIDATE_LABEL_SET or choice is None:
    return tuple(choice_labels), scores, choice
  selected = tuple(label for label in choice_labels if label != "none")
  if selected:
    return tuple(choice_labels), scores, choice
  word_scores = dict(choice.get("word_scores") or {})
  cyan_score = float(word_scores.get("cyan", float("nan")))
  pink_score = float(word_scores.get("pink", float("nan")))
  neutral_score = float(word_scores.get("none", float("nan")))
  if not math.isfinite(cyan_score) or not math.isfinite(pink_score):
    return tuple(choice_labels), scores, choice
  best_candidate_score = max(cyan_score, pink_score)
  if math.isfinite(neutral_score) and neutral_score - best_candidate_score > max(0.0, float(neutral_margin)):
    adjusted_choice = dict(choice)
    adjusted_choice["candidate_relative_choice"] = {
      "enabled": False,
      "reason": "neutral_dominant",
      "best_candidate_minus_neutral": best_candidate_score - neutral_score,
      "neutral_margin": max(0.0, float(neutral_margin)),
      "original_answer": choice.get("answer"),
      "original_word_scores": word_scores,
    }
    return tuple(choice_labels), scores, adjusted_choice
  margin = cyan_score - pink_score
  if abs(margin) < max(0.0, float(min_margin)):
    return tuple(choice_labels), scores, choice

  selected_label = "construction_blocks_left_candidate" if margin > 0.0 else "construction_blocks_right_candidate"
  other_label = "construction_blocks_right_candidate" if selected_label == "construction_blocks_left_candidate" else "construction_blocks_left_candidate"
  adjusted_scores = dict(scores)
  adjusted_scores[selected_label] = abs(margin)
  adjusted_scores[other_label] = 0.0
  adjusted_choice = dict(choice)
  adjusted_choice["answer"] = "cyan" if selected_label == "construction_blocks_left_candidate" else "pink"
  adjusted_choice["candidate_relative_choice"] = {
    "enabled": True,
    "selected_label": selected_label,
    "cyan_minus_pink": margin,
    "min_margin": max(0.0, float(min_margin)),
    "best_candidate_minus_neutral": best_candidate_score - neutral_score if math.isfinite(neutral_score) else None,
    "neutral_margin": max(0.0, float(neutral_margin)),
    "original_answer": choice.get("answer"),
    "original_word_scores": word_scores,
  }
  return (selected_label,), adjusted_scores, adjusted_choice


def _with_rtp_confidence(text: str, confidence: float | None) -> str:
  if confidence is None:
    return text
  return re.sub(r"(?m)^confidence=[0-9.]+$", f"confidence={confidence:.2f}", text)


def _state_metric(vehicle_state: str, key: str) -> float | None:
  legacy_keys = (key,)
  if key == "desired_speed_mps":
    legacy_keys = ("desired_speed_mps", "desired_speed")
  legacy = None
  for legacy_key in legacy_keys:
    legacy = re.search(rf"\b{re.escape(legacy_key)}=(-?\d+(?:\.\d+)?|none)", vehicle_state)
    if legacy is not None:
      break
  if legacy is not None:
    raw = legacy.group(1)
    if raw == "none":
      return None
    try:
      value = float(raw)
    except ValueError:
      return None
    return value if math.isfinite(value) else None

  plain_patterns = {
    "lead_present": (
      r"\blead\s+present\s+(yes|no|true|false|1|0)\b",
      r"\blead\s+(yes|no|true|false|1|0)\b",
    ),
    "lead_distance_m": (r"\bdistance\s+(-?\d+(?:\.\d+)?|none)\s*m\b",),
    "desired_speed_mps": (r"\bdesired\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*m/s\b", r"\bdesired\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*mps\b"),
    "lead_lateral_m": (r"\blateral\s+offset\s+(-?\d+(?:\.\d+)?|none)\s*m\b",),
    "lead_speed_mps": (r"\blead\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*m/s\b", r"\blead\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*mps\b"),
    "lead_rel_speed_mps": (r"\brelative\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*m/s\b", r"\brelative\s+speed\s+(-?\d+(?:\.\d+)?|none)\s*mps\b"),
    "lead_closing_mps": (r"\bclosing\s+(-?\d+(?:\.\d+)?|none)\s*m/s\b", r"\bclosing\s+(-?\d+(?:\.\d+)?|none)\s*mps\b"),
    "lead_accel_mps2": (r"\bacceleration\s+(-?\d+(?:\.\d+)?|none)\s*m/s2\b", r"\baccel(?:eration)?\s+(-?\d+(?:\.\d+)?|none)\s*mps2\b"),
    "lead_lateral_velocity_mps": (r"\blateral\s+velocity\s+(-?\d+(?:\.\d+)?|none)\s*m/s\b", r"\blateral\s+velocity\s+(-?\d+(?:\.\d+)?|none)\s*mps\b"),
  }
  for pattern in plain_patterns.get(key, ()):
    match = re.search(pattern, vehicle_state, flags=re.IGNORECASE)
    if match is None:
      continue
    raw = match.group(1).lower()
    if raw == "none":
      return None
    if key == "lead_present":
      return 1.0 if raw in ("yes", "true", "1") else 0.0
    try:
      value = float(raw)
    except ValueError:
      return None
    return value if math.isfinite(value) else None
  return None


def _lead_state_word(vehicle_state: str) -> tuple[str | None, dict[str, float]]:
  record = {
    "lead_present": _state_metric(vehicle_state, "lead_present"),
    "desired_speed_mps": _state_metric(vehicle_state, "desired_speed_mps"),
    "lead_distance_m": _state_metric(vehicle_state, "lead_distance_m"),
    "lead_lateral_m": _state_metric(vehicle_state, "lead_lateral_m"),
    "lead_speed_mps": _state_metric(vehicle_state, "lead_speed_mps"),
    "lead_rel_speed_mps": _state_metric(vehicle_state, "lead_rel_speed_mps"),
    "lead_closing_mps": _state_metric(vehicle_state, "lead_closing_mps"),
    "lead_accel_mps2": _state_metric(vehicle_state, "lead_accel_mps2"),
    "lead_lateral_velocity_mps": _state_metric(vehicle_state, "lead_lateral_velocity_mps"),
  }
  metrics = lead_track_metrics(record)
  requirement = classify_lead_track(record)
  lead_class = str(requirement["expected_class"])
  if lead_class == "none" or str(requirement["reason"]) in {"no_lead_track", "incomplete_track"}:
    return None, metrics
  return lead_choice_word(lead_class), metrics


def _apply_lead_state_consistency(
  selected: tuple[str, ...],
  scores: dict[str, float],
  choice: dict | None,
  labels: Sequence[str],
  vehicle_state: str,
  enabled: bool = True,
) -> tuple[tuple[str, ...], dict[str, float], dict | None]:
  lead_group = {"true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle"}
  if not enabled or set(labels) != lead_group:
    return selected, scores, choice
  physical_word, metrics = _lead_state_word(vehicle_state)
  if physical_word is None:
    return selected, scores, choice
  spec = _choice_group_spec(labels)
  physical_label = spec["word_to_label"].get(physical_word)
  if physical_label is None:
    return selected, scores, choice
  adjusted_scores = dict(scores)
  adjusted_scores[physical_label] = max(float(adjusted_scores.get(physical_label, 0.0)), float(spec.get("min_margin", 0.0)) + 0.5)
  adjusted_choice = dict(choice or {})
  previous_answer = adjusted_choice.get("answer")
  adjusted_choice.update({
    "answer": physical_word,
    "physical_state_override": previous_answer != physical_word,
    "answer_before_physical_filter": previous_answer,
    "physical_state_metrics": metrics,
    "physical_state_rule": "production lead-track feasibility",
  })
  return (physical_label,), adjusted_scores, adjusted_choice


class TrtTextRunner:
  def __init__(self, args, labels: Sequence[str], runtime: trt.Runtime, yes_count: int, no_count: int, *, require_keyed: bool = False):
    self.args = args
    self.labels = tuple(labels)
    self.text_engine_path = _resolve_text_engine_path(args, self.labels, require_keyed=require_keyed)
    self.text_engine, self.text_ctx = _load_engine(runtime, self.text_engine_path)
    self.text_stream = torch.cuda.Stream()
    self.text_tensor_names = _trt_tensor_names(self.text_engine)
    self.text_output, self.text_output_name = _select_text_output(_text_output(args), self.text_tensor_names)
    self.text_embed_shape = _trt_shape(self.text_engine, "inputs_embeds")
    self.text_position_shape = _trt_shape(self.text_engine, "position_ids")
    self.has_last_token_mask = "last_token_mask" in self.text_tensor_names
    self.text_last_mask_shape = _trt_shape(self.text_engine, "last_token_mask") if self.has_last_token_mask else None
    self.text_out_shape = _trt_shape(self.text_engine, self.text_output_name)
    self.text_out = torch.empty(self.text_out_shape, device="cuda", dtype=torch.float16)
    self.text_ctx.set_tensor_address(self.text_output_name, self.text_out.data_ptr())
    self.choice_mode = _label_decision_mode(args) == "choice"
    self.fixed_output_index = int(getattr(args, "fixed_output_index", -1))
    self.choice_words = CHOICE_WORDS
    expected_batch = 1 if self.choice_mode else len(self.labels)
    if self.text_out_shape[0] != expected_batch:
      raise RuntimeError(f"text engine batch {self.text_out_shape[0]} does not match expected {expected_batch}")
    if self.text_output in ("logits", "fixed_logits"):
      expected_width = len(CHOICE_WORDS) if self.choice_mode else yes_count + no_count
      if self.choice_mode:
        choice_words = _choice_words_for_output_width(self.text_out_shape[-1])
        if choice_words:
          self.choice_words = choice_words
      if self.text_out_shape[-1] != (len(self.choice_words) if self.choice_mode else expected_width):
        raise RuntimeError(
          f"text engine selected logit width {self.text_out_shape[-1]} does not match expected {len(self.choice_words) if self.choice_mode else expected_width}"
        )
    if self.text_output in ("hidden", "full_hidden") and self.text_out_shape[-1] != 2048:
      raise RuntimeError(f"text engine {self.text_output_name} width {self.text_out_shape[-1]} != 2048")
    if self.text_output == "full_hidden" and len(self.text_out_shape) != 3:
      raise RuntimeError(f"text engine full_hidden shape {self.text_out_shape} must be rank 3")

  def run(
    self,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    last_token_mask: torch.Tensor,
    yes_count: int,
    labels: Sequence[str] | None = None,
  ) -> tuple[dict[str, float], float]:
    output_labels = tuple(labels) if labels is not None else self.labels
    expected_batch = 1 if self.choice_mode else len(output_labels)
    if self.text_out_shape[0] != expected_batch:
      raise RuntimeError(f"text runner output batch {self.text_out_shape[0]} does not match expected {expected_batch}")
    if tuple(inputs_embeds.shape) != self.text_embed_shape:
      raise RuntimeError(f"text inputs_embeds shape {tuple(inputs_embeds.shape)} does not match engine {self.text_embed_shape}")
    if tuple(position_ids.shape) != self.text_position_shape:
      raise RuntimeError(f"text position_ids shape {tuple(position_ids.shape)} does not match engine {self.text_position_shape}")
    if self.has_last_token_mask and tuple(last_token_mask.shape) != self.text_last_mask_shape:
      raise RuntimeError(f"text last_token_mask shape {tuple(last_token_mask.shape)} does not match engine {self.text_last_mask_shape}")
    self.text_ctx.set_tensor_address(self.text_output_name, self.text_out.data_ptr())
    self.text_ctx.set_tensor_address("inputs_embeds", inputs_embeds.data_ptr())
    self.text_ctx.set_tensor_address("position_ids", position_ids.data_ptr())
    if self.has_last_token_mask:
      self.text_ctx.set_tensor_address("last_token_mask", last_token_mask.data_ptr())
    start = time.perf_counter()
    with torch.cuda.stream(self.text_stream):
      if not self.text_ctx.execute_async_v3(self.text_stream.cuda_stream):
        raise RuntimeError("text TensorRT execute_async_v3 failed")
    self.text_stream.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if self.text_output in ("hidden", "full_hidden"):
      return {}, elapsed_ms
    if _label_decision_mode(self.args) == "choice":
      logits = self.text_out.reshape(-1).detach().cpu().tolist()
      return {word: float(score) for word, score in zip(self.choice_words, logits, strict=True)}, elapsed_ms
    return _scores_from_selected_logits(self.text_out, yes_count, output_labels), elapsed_ms


class QwenTrtLabelScorer:
  def __init__(
    self,
    args,
    labels: Sequence[str] | None = None,
    *,
    processor=None,
    model=None,
    runtime: trt.Runtime | None = None,
    vision_runner: TrtVisionRunner | None = None,
    text_runner: TrtTextRunner | None = None,
    require_keyed_text_engine: bool = False,
  ):
    self.args = args
    self.labels = tuple(labels) if labels is not None else _score_labels(args.score_labels)
    self.processor = processor
    self.model = model
    if self.processor is None or self.model is None:
      self.processor, self.model = _load_qwen(args.model_dir, _torch_attn_implementation(args))
    self.choice_mode = _label_decision_mode(args) == "choice"
    if self.choice_mode:
      _choice_group_spec(self.labels)
    self.yes_ids, self.no_ids = _score_label_ids(self.processor)
    self._runtime_owner = runtime is None
    if runtime is None:
      logger = trt.Logger(trt.Logger.WARNING)
      runtime = trt.Runtime(logger)
    self.runtime = runtime
    self.vision_runner = vision_runner or TrtVisionRunner(args, runtime)
    self.text_runner = text_runner or TrtTextRunner(
        args,
        self.labels,
        runtime,
        len(self.yes_ids),
        len(self.no_ids),
        require_keyed=require_keyed_text_engine,
      )
    self.text_engine_path = self.text_runner.text_engine_path
    self.vision_engine_path = self.vision_runner.vision_engine_path
    self.signal_runner = TrtSignalHeadRunner(args, runtime) if getattr(args, "enable_signal_head", False) else None
    self.choice_logit_words = CHOICE_WORDS
    if self.choice_mode and self.text_runner.text_output == "hidden":
      selected_ids, self.choice_logit_words = _choice_flat_token_ids_and_words(self.processor)
    else:
      selected_ids = _choice_token_ids(self.processor) if self.choice_mode else torch.tensor(list(self.yes_ids) + list(self.no_ids), device="cuda", dtype=torch.long)
    self.selected_lm_weight = self.model.lm_head.weight.index_select(0, selected_ids).contiguous()

  def warmup(self, count: int) -> None:
    warm = Image.open(self.args.image).convert("RGB") if self.args.image.exists() else Image.new("RGB", (384, 216), (20, 20, 20))
    for _ in range(max(0, count)):
      self.score(warm, self.args.vehicle_state)

  def score(self, image: Image.Image, vehicle_state: str) -> dict:
    response = self._score_once(image, vehicle_state)
    if (
      (
        bool(getattr(self.args, "construction_mirror_consistency", False)) or
        bool(getattr(self.args, "construction_mirror_fusion", False))
      ) and
      _is_construction_group(self.labels) and
      set(self.labels) != CONSTRUCTION_CANDIDATE_LABEL_SET
    ):
      mirror_start = time.perf_counter()
      mirror_response = self._score_once(ImageOps.mirror(image), vehicle_state)
      mirror_elapsed_ms = (time.perf_counter() - mirror_start) * 1000.0
      if bool(getattr(self.args, "construction_mirror_fusion", False)):
        response = _apply_construction_mirror_fusion(response, mirror_response, self.labels)
      else:
        response = _apply_construction_mirror_consistency(
          response,
          mirror_response,
          negative_clear_threshold=float(getattr(self.args, "score_negative_clear_threshold", 2.0)),
        )
      parts = dict(response.get("timings_ms") or {})
      parts["construction_mirror_total_ms"] = mirror_elapsed_ms
      parts["total_ms"] = float(parts.get("total_ms", response.get("total_ms", 0.0))) + mirror_elapsed_ms
      response["timings_ms"] = parts
      response["total_ms"] = parts["total_ms"]
      response["prefill_ms"] = float(response.get("prefill_ms", 0.0)) + float(mirror_response.get("prefill_ms", 0.0))
      response["decode_ms"] = float(response.get("decode_ms", 0.0)) + float(mirror_response.get("decode_ms", 0.0))
      mirror_mode = "fusion" if bool(getattr(self.args, "construction_mirror_fusion", False)) else "consistency"
      response["backend"] = f"{response.get('backend', 'qwen2.5-vl-trt')}-construction-mirror-{mirror_mode}"
    return response

  def _score_once(self, image: Image.Image, vehicle_state: str) -> dict:
    parts: dict[str, float] = {}
    debug_stats: dict | None = {} if bool(getattr(self.args, "debug_tensor_stats", False)) else None
    wall_start = time.perf_counter()

    start = time.perf_counter()
    if self.choice_mode:
      inputs = _build_choice_inputs(
        self.processor,
        image,
        self.labels,
        self.args.image_mode,
        self.args.image_size,
        vehicle_state,
        self.args.text_seq_len,
      )
      label_batch = 1
    else:
      inputs = _build_inputs(
        self.processor,
        image,
        self.labels,
        self.args.image_mode,
        self.args.image_size,
        vehicle_state,
        self.args.text_seq_len,
        _score_prompt_mode(self.args),
      )
      label_batch = len(self.labels)
    torch.cuda.synchronize()
    parts["processor_ms"] = (time.perf_counter() - start) * 1000.0

    vision_out, parts["trt_vision_ms"] = self.vision_runner.run(inputs, label_batch)
    if debug_stats is not None:
      debug_stats["input_ids"] = {
        "shape": list(inputs.input_ids.shape),
        "attention_sum": int(inputs.attention_mask.detach().sum().cpu()),
        "first_nonpad_index": int(inputs.attention_mask.detach().cpu().to(torch.long).argmax(dim=1)[0]),
      }
      if hasattr(inputs, "image_grid_thw"):
        debug_stats["image_grid_thw"] = inputs.image_grid_thw.detach().cpu().tolist()
      if hasattr(inputs, "pixel_values"):
        debug_stats["pixel_values"] = _debug_tensor_stats(inputs.pixel_values)
      debug_stats["vision_out"] = _debug_tensor_stats(vision_out)
    clip_abs = float(getattr(self.args, "vision_feature_clip_abs", 0.0) or 0.0)
    if clip_abs > 0.0:
      vision_out = vision_out.clamp(min=-clip_abs, max=clip_abs).contiguous()
      if debug_stats is not None:
        debug_stats["vision_feature_clip_abs"] = clip_abs
        debug_stats["vision_out_after_clip"] = _debug_tensor_stats(vision_out)

    with torch.no_grad():
      qwen = self.model.model
      start = time.perf_counter()
      inputs_embeds = qwen.get_input_embeddings()(inputs.input_ids)
      torch.cuda.synchronize()
      parts["embed_ms"] = (time.perf_counter() - start) * 1000.0
      if debug_stats is not None:
        debug_stats["inputs_embeds_before_scatter"] = _debug_tensor_stats(inputs_embeds)

      start = time.perf_counter()
      image_features = vision_out.contiguous() if self.choice_mode else vision_out.repeat((len(self.labels), 1)).contiguous()
      image_mask, _ = qwen.get_placeholder_mask(inputs.input_ids, inputs_embeds=inputs_embeds, image_features=image_features)
      inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features).contiguous()
      torch.cuda.synchronize()
      parts["scatter_ms"] = (time.perf_counter() - start) * 1000.0
      if debug_stats is not None:
        debug_stats["image_mask_true_count"] = int(image_mask.detach().sum().cpu())
        debug_stats["inputs_embeds_after_scatter"] = _debug_tensor_stats(inputs_embeds)

      start = time.perf_counter()
      position_ids, _ = qwen.get_rope_index(
        inputs.input_ids,
        inputs.image_grid_thw,
        None,
        second_per_grid_ts=None,
        attention_mask=inputs.attention_mask,
      )
      position_ids = _apply_text_position_mode(position_ids, self.args)
      last_token_indices = _last_token_indices_from_attention_mask(inputs.attention_mask).contiguous()
      last_token_mask = _last_token_mask_from_indices(
        last_token_indices,
        (inputs_embeds.shape[0], inputs_embeds.shape[1], 1),
        inputs_embeds.dtype,
      )
      torch.cuda.synchronize()
      parts["rope_ms"] = (time.perf_counter() - start) * 1000.0
      if debug_stats is not None:
        debug_stats["position_ids"] = _debug_tensor_stats(position_ids)
        debug_stats["last_token_indices"] = last_token_indices.detach().cpu().tolist()
        debug_stats["last_token_mask"] = _debug_tensor_stats(last_token_mask)

    if self.text_runner.text_output == "fixed_logits" and self.text_runner.fixed_output_index >= 0:
      expected_index = torch.full_like(last_token_indices, self.text_runner.fixed_output_index)
      if not torch.equal(last_token_indices, expected_index):
        raise RuntimeError(
          "fixed_logits text engine prompt index mismatch: "
          f"runtime {last_token_indices.detach().cpu().tolist()} != engine {self.text_runner.fixed_output_index}"
        )

    raw_scores, parts["trt_text_ms"] = self.text_runner.run(inputs_embeds, position_ids, last_token_mask, len(self.yes_ids), self.labels)
    if debug_stats is not None:
      debug_stats["text_output_name"] = self.text_runner.text_output_name
      debug_stats["text_output_mode"] = self.text_runner.text_output
      debug_stats["text_out"] = _debug_tensor_stats(self.text_runner.text_out)
    choice: dict | None = None
    selected: list[str] = []
    if self.text_runner.text_output in ("hidden", "full_hidden"):
      start = time.perf_counter()
      selected_hidden = (
        _select_last_hidden_from_full(self.text_runner.text_out, last_token_indices)
        if self.text_runner.text_output == "full_hidden" else
        self.text_runner.text_out
      )
      selected_logits = torch.matmul(selected_hidden, self.selected_lm_weight.t())
      torch.cuda.synchronize()
      parts["lm_head_ms"] = (time.perf_counter() - start) * 1000.0
      if self.choice_mode:
        choice_labels, scores, choice = _choice_scores_from_selected_logits(
          selected_logits,
          self.labels,
          self.choice_logit_words,
          getattr(self.args, "score_thresholds_map", {}),
          self.args.score_threshold if self.args.score_threshold > 0.0 else None,
        )
        choice_labels, scores, choice = _apply_candidate_relative_choice(
          choice_labels,
          scores,
          choice,
          self.labels,
          enabled=bool(getattr(self.args, "construction_candidate_relative_choice", False)),
          min_margin=float(getattr(self.args, "construction_candidate_relative_margin", 1.5)),
          neutral_margin=float(getattr(self.args, "construction_candidate_relative_neutral_margin", CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN)),
        )
        choice_labels, scores, choice = _apply_lead_state_consistency(
          choice_labels,
          scores,
          choice,
          self.labels,
          vehicle_state,
          enabled=self.args.lead_state_consistency,
        )
        selected = [label for label in choice_labels if label != "none"]
      else:
        scores = _scores_from_selected_logits(selected_logits, len(self.yes_ids), self.labels)
    elif self.choice_mode:
      choice_labels, scores, choice = _choice_scores_from_word_scores(
        raw_scores,
        self.labels,
        getattr(self.args, "score_thresholds_map", {}),
        self.args.score_threshold if self.args.score_threshold > 0.0 else None,
      )
      choice_labels, scores, choice = _apply_candidate_relative_choice(
        choice_labels,
        scores,
        choice,
        self.labels,
        enabled=bool(getattr(self.args, "construction_candidate_relative_choice", False)),
        min_margin=float(getattr(self.args, "construction_candidate_relative_margin", 1.5)),
        neutral_margin=float(getattr(self.args, "construction_candidate_relative_neutral_margin", CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN)),
      )
      choice_labels, scores, choice = _apply_lead_state_consistency(
        choice_labels,
        scores,
        choice,
        self.labels,
        vehicle_state,
        enabled=self.args.lead_state_consistency,
      )
      selected = [label for label in choice_labels if label != "none"]
    else:
      scores = raw_scores
    signal: dict | None = None
    signal_labels: list[str] = ["none"]
    if self.signal_runner is not None:
      signal_logits, parts["signal_head_ms"] = self.signal_runner.run(vision_out)
      signal_labels, signal_scores, signal = _signal_scores_from_logits(signal_logits)
      signal["accepted"] = signal["probability"] >= self.args.signal_min_probability and signal["margin"] >= self.args.signal_min_margin
      if not signal["accepted"]:
        signal_labels = ["none"]
      scores = {**scores, **signal_scores}
    if not self.choice_mode:
      selected = [
        label for label in self.labels
        if scores[label] >= self.args.score_thresholds_map.get(label, self.args.score_threshold)
      ]
    if self.signal_runner is not None:
      selected_set = set(label for label in selected if label not in SIGNAL_LABELS)
      selected_set.update(label for label in signal_labels if label != "none")
    else:
      selected_set = set(selected)
    label_order = {label: idx for idx, label in enumerate(tuple(self.labels) + SIGNAL_LABELS)}
    selected = sorted(_resolve_exclusive_labels(selected_set, scores), key=lambda label: label_order.get(label, len(label_order)))
    labels = tuple(selected) if selected else ("none",)
    labels = _with_visual_fallbacks(
      image,
      labels,
      enable_signal=self.args.enable_visual_fallbacks or self.args.enable_visual_signal_fallback,
      enable_construction=self.args.enable_visual_fallbacks or self.args.enable_visual_construction_fallback,
      enable_stop=self.args.enable_visual_fallbacks or self.args.enable_visual_stop_fallback,
    )
    if bool(getattr(self.args, "construction_candidate_score_resolve", False)):
      labels = _resolve_candidate_obstruction_scores(
        labels,
        scores,
        float(getattr(self.args, "construction_candidate_diff_margin", 0.4)),
      )
    rtp_text = _with_rtp_confidence(
      _labels_to_rtp(labels),
      _score_calibrated_construction_confidence(
        labels,
        scores,
        strong_edge_immediate=bool(getattr(self.args, "construction_edge_binary", False)),
      ),
    )
    total_ms = (time.perf_counter() - wall_start) * 1000.0
    parts["total_ms"] = total_ms
    return {
      "text": rtp_text,
      "rtp_text": rtp_text,
      "labels_text": ",".join(labels),
      "labels": list(labels),
      "score_labels": list(self.labels),
      "label_mode": "score+signal_head" if self.signal_runner is not None else "score",
      "image_mode": self.args.image_mode,
      "label_scores": scores,
      "choice": choice,
      "signal": signal,
      "generated_token_count": 0,
      "prefill_ms": parts["processor_ms"] + parts["trt_vision_ms"] + parts["embed_ms"] + parts["scatter_ms"] + parts["rope_ms"],
      "decode_ms": parts["trt_text_ms"] + parts.get("signal_head_ms", 0.0),
      "total_ms": total_ms,
      "timings_ms": parts,
      "backend": f"qwen2.5-vl-3b-trt-{_text_precision(self.args)}-{self.args.image_mode}{self.args.image_size}-score",
      **({"debug_tensor_stats": debug_stats} if debug_stats is not None else {}),
    }


class QwenTrtRotatingLabelScorer:
  def __init__(self, args, groups: Sequence[Sequence[str]]):
    self.args = args
    self.groups = tuple(tuple(group) for group in groups)
    if getattr(args, "require_manifest", False):
      _enforce_manifest(args, self.groups)
    self.processor, self.model = _load_qwen(args.model_dir, _torch_attn_implementation(args))
    logger = trt.Logger(trt.Logger.WARNING)
    self.runtime = trt.Runtime(logger)
    self.vision_runner = TrtVisionRunner(args, self.runtime)
    has_mixed_binary = bool(getattr(args, "construction_edge_binary", False)) or bool(getattr(args, "construction_candidate_binary", False))
    mixed_hidden_shared = has_mixed_binary and str(getattr(args, "text_output", "")) == "hidden"
    shared_text_runner = None
    binary_shape_shared = has_mixed_binary and _label_decision_mode(args) == "binary"
    if args.score_rotate_shared_engine and (not has_mixed_binary or mixed_hidden_shared or binary_shape_shared):
      shared_text_runner = TrtTextRunner(
        args,
        self.groups[0],
        self.runtime,
        *[len(ids) for ids in _score_label_ids(self.processor)],
        require_keyed=False,
      )
    scorer_cache: dict[tuple[str, ...], QwenTrtLabelScorer] = {}
    self.scorers = []
    for group in self.groups:
      group_args = _args_for_score_group(args, group)
      cache_key = (
        *group,
        f"text_engine={getattr(group_args, 'text_engine', None)}",
        f"text_precision={_text_precision(group_args)}",
      )
      if cache_key not in scorer_cache:
        shared_shape_compatible = (
          shared_text_runner is not None and
          _label_decision_mode(group_args) == _label_decision_mode(args) and
          int(getattr(group_args, "text_seq_len", 0)) == int(getattr(args, "text_seq_len", 0)) and
          _text_output(group_args) == _text_output(args) and
          getattr(group_args, "text_engine", None) == getattr(args, "text_engine", None)
        )
        use_shared_text_runner = shared_text_runner if shared_shape_compatible or mixed_hidden_shared else None
        require_keyed = (
          not args.score_rotate_shared_engine and
          getattr(group_args, "text_engine", None) is None
        )
        # construction_edge_binary/construction_candidate_binary may change the
        # prompt text while keeping the same fixed binary TensorRT engine shape.
        # Share the runner in that case to avoid duplicate 36-layer engines.
        scorer_cache[cache_key] = QwenTrtLabelScorer(
          group_args,
          group,
          processor=self.processor,
          model=self.model,
          runtime=self.runtime,
          vision_runner=self.vision_runner,
          text_runner=use_shared_text_runner,
          require_keyed_text_engine=require_keyed,
        )
      self.scorers.append(scorer_cache[cache_key])
    durable_labels = tuple(label.strip() for label in args.score_durable_labels.split(",") if label.strip())
    self.rotating_state = RotatingScoreState(
      self.groups,
      args.score_cache_ttl_frames,
      durable_labels or DEFAULT_DURABLE_SCORE_LABELS,
      args.score_negative_clear_threshold,
    )

  def reset_runtime_state(self) -> None:
    durable_labels = tuple(label.strip() for label in self.args.score_durable_labels.split(",") if label.strip())
    self.rotating_state = RotatingScoreState(
      self.groups,
      self.args.score_cache_ttl_frames,
      durable_labels or DEFAULT_DURABLE_SCORE_LABELS,
      self.args.score_negative_clear_threshold,
    )

  def warmup(self, count: int) -> None:
    warm = Image.open(self.args.image).convert("RGB") if self.args.image.exists() else Image.new("RGB", (384, 216), (20, 20, 20))
    for idx, scorer in enumerate(self.scorers):
      for _ in range(max(0, count)):
        scorer.score(warm, self.args.vehicle_state)

  def score(self, image: Image.Image, vehicle_state: str, frame_id: int) -> dict:
    group_idx, request_labels = self.rotating_state.next_group()
    return self._score_group(image, vehicle_state, frame_id, group_idx, request_labels)

  def score_payload(self, payload: dict, vehicle_state: str, frame_id: int) -> dict:
    group_idx, request_labels = self.rotating_state.next_group()
    image = _image_from_payload_for_labels(payload, request_labels)
    vehicle_state = _vehicle_state_for_labels(self.args, request_labels, vehicle_state)
    return self._score_group(image, vehicle_state, frame_id, group_idx, request_labels)

  def _score_group(self, image: Image.Image, vehicle_state: str, frame_id: int, group_idx: int, request_labels: Sequence[str]) -> dict:
    scorer = self.scorers[group_idx]
    response = scorer.score(image, vehicle_state)
    cached_labels = self.rotating_state.update(
      request_labels,
      response["labels"],
      response["label_scores"],
      frame_id,
      vehicle_state,
    )
    cached_labels = _with_visual_fallbacks(
      image,
      cached_labels,
      enable_signal=self.args.enable_visual_fallbacks or self.args.enable_visual_signal_fallback,
      enable_construction=self.args.enable_visual_fallbacks or self.args.enable_visual_construction_fallback,
      enable_stop=self.args.enable_visual_fallbacks or self.args.enable_visual_stop_fallback,
    )
    cached_scores = self.rotating_state.active_scores(frame_id)
    if bool(getattr(self.args, "construction_candidate_score_resolve", False)):
      cached_labels = _resolve_candidate_obstruction_scores(
        cached_labels,
        cached_scores,
        float(getattr(self.args, "construction_candidate_diff_margin", 0.4)),
      )
    rtp_confidence = _score_calibrated_construction_confidence(
      cached_labels,
      cached_scores,
      strong_edge_immediate=bool(getattr(self.args, "construction_edge_binary", False)),
    )
    if self.rotating_state.construction_clear_active(frame_id):
      rtp_confidence = max(rtp_confidence, CONSTRUCTION_CLEAR_RTP_CONFIDENCE)
    rtp_text = _with_rtp_confidence(_labels_to_rtp(cached_labels), rtp_confidence)
    response["labels_scored_this_request"] = list(request_labels)
    response["score_group_index"] = group_idx
    response["labels_current_group"] = response["labels"]
    response["labels"] = list(cached_labels)
    response["label_scores_cached"] = cached_scores
    response["label_state_debug"] = self.rotating_state.debug_state(frame_id)
    response["rtp_text"] = rtp_text
    response["text"] = rtp_text
    response["backend"] = f"{response['backend']}-rotating"
    return response


def build_text_group_engines(args) -> dict:
  groups = _score_groups(args.score_label_groups)
  unique_groups = tuple(dict.fromkeys(groups))
  results = []
  for group in unique_groups:
    group_args = argparse.Namespace(**vars(args))
    group_args.score_labels = ",".join(group)
    group_args.label_keyed_text_engine = True
    engine_path = _keyed_text_engine_path(group_args, group)
    onnx_path = _keyed_text_onnx_path(group_args, group)
    if engine_path.exists():
      results.append({
        "kind": f"text_{_text_precision(args)}",
        "labels": group,
        "image_mode": args.image_mode,
        "image_size": args.image_size,
        "text_seq_len": int(args.text_seq_len),
        "text_precision": _text_precision(args),
        "engine": str(engine_path),
        "onnx": str(onnx_path),
        "skipped_existing": True,
      })
    else:
      results.append(build_text_engine(group_args))
  return {
    "kind": f"text_{_text_precision(args)}_groups",
    "groups": groups,
    "unique_groups": unique_groups,
    "results": results,
  }


def benchmark(args) -> dict:
  if args.runtime_mode == "visual-head":
    return benchmark_visual_head(args)
  if args.require_manifest:
    _enforce_manifest(args, (_score_labels(args.score_labels),))
  image = Image.open(args.image).convert("RGB")
  scorer = QwenTrtLabelScorer(args)

  for _ in range(args.warmup):
    scorer.score(image, args.vehicle_state)

  rows: list[dict[str, float]] = []
  response: dict | None = None
  for _ in range(args.iters):
    response = scorer.score(image, args.vehicle_state)
    rows.append(response["timings_ms"])

  return {
    "kind": "benchmark",
    "labels": scorer.labels,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "iters": args.iters,
    "text_engine": str(scorer.text_engine_path),
    "vision_engine": str(scorer.vision_engine_path),
    "stage_summary": _summarize_timing_rows(rows),
    "scores": {} if response is None else response["label_scores"],
    "last_response": response or {},
  }


def _signal_eval_images(args) -> list[tuple[str, Path]]:
  image_sets = _signal_image_sets(args)
  rows: list[tuple[str, Path]] = []
  for label, paths in image_sets.items():
    rows.extend((label, path) for path in paths)
  if not rows:
    raise RuntimeError("no signal evaluation images found")
  return rows


def benchmark_visual_head(args) -> dict:
  if args.require_manifest:
    _enforce_manifest(args, _groups_for_runtime(args))
  scorer = QwenTrtVisualHeadScorer(args)
  warm = Image.open(args.image).convert("RGB") if args.image.exists() else Image.open(_signal_eval_images(args)[0][1]).convert("RGB")
  for _ in range(args.warmup):
    scorer.score(warm, args.vehicle_state)

  eval_rows = _signal_eval_images(args)
  rows: list[dict[str, float]] = []
  responses: list[dict] = []
  for idx in range(args.iters):
    expected, path = eval_rows[idx % len(eval_rows)]
    response = scorer.score(Image.open(path).convert("RGB"), args.vehicle_state)
    response["expected_signal"] = expected
    response["image"] = str(path)
    responses.append(response)
    rows.append(response["timings_ms"])

  return {
    "kind": "benchmark_visual_head",
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "iters": args.iters,
    "vision_engine": str(scorer.vision_runner.vision_engine_path),
    "signal_head_engine": str(scorer.signal_runner.engine_path),
    "stage_summary": _summarize_timing_rows(rows),
    "last_response": responses[-1] if responses else {},
    "responses": responses,
  }


def signal_acceptance(args) -> dict:
  scorer = QwenTrtVisualHeadScorer(args)
  eval_rows = _signal_eval_images(args)
  for _ in range(args.warmup):
    scorer.score(Image.open(eval_rows[0][1]).convert("RGB"), args.vehicle_state)

  rows: list[dict[str, float]] = []
  cases = []
  for expected, path in eval_rows:
    image = Image.open(path).convert("RGB")
    per_image = []
    for aug_idx in range(max(1, args.signal_acceptance_augmentations)):
      sample = image if aug_idx == 0 else _augment_image(image, random.Random(args.signal_train_seed + aug_idx + len(cases) * 1009))
      response = scorer.score(sample, args.vehicle_state)
      rows.append(response["timings_ms"])
      predicted = response["signal"]["class"] if response["labels"] != ["none"] else "none"
      ok = predicted == expected
      per_image.append({
        "augmentation": aug_idx,
        "predicted": predicted,
        "expected": expected,
        "ok": ok,
        "signal": response["signal"],
        "timings_ms": response["timings_ms"],
      })
    cases.append({"image": str(path), "expected": expected, "checks": per_image})

  total = _summarize_timing_rows(rows).get("total_ms", {})
  all_checks = [check for case in cases for check in case["checks"]]
  bad = [check for check in all_checks if not check["ok"]]
  p99 = float(total.get("p99", 0.0))
  max_latency = float(total.get("max", 0.0))
  issues = []
  if bad:
    issues.append(f"{len(bad)} signal acceptance checks misclassified")
  if p99 > args.deadline_ms:
    issues.append(f"p99 total latency {p99:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  if max_latency > args.deadline_ms:
    issues.append(f"max total latency {max_latency:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  return {
    "kind": "signal_acceptance",
    "ok": not issues,
    "deadline_ms": args.deadline_ms,
    "issues": issues,
    "stage_summary": _summarize_timing_rows(rows),
    "cases": cases,
  }


def benchmark_groups(args) -> dict:
  image = Image.open(args.image).convert("RGB")
  groups = _score_groups(args.score_label_groups)
  scorer = QwenTrtRotatingLabelScorer(args, groups)
  scorer.warmup(args.warmup)

  rows: list[dict[str, float]] = []
  group_rows: dict[int, list[dict[str, float]]] = {idx: [] for idx in range(len(groups))}
  responses: list[dict] = []
  for idx in range(args.iters):
    response = scorer.score(image, args.vehicle_state, idx)
    responses.append(response)
    timings = response["timings_ms"]
    rows.append(timings)
    group_rows[int(response["score_group_index"])].append(timings)

  return {
    "kind": "benchmark_groups",
    "groups": groups,
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "iters": args.iters,
    "vision_engine": str(scorer.vision_runner.vision_engine_path),
    "text_engines": [str(group_scorer.text_engine_path) for group_scorer in scorer.scorers],
    "stage_summary": _summarize_timing_rows(rows),
    "group_stage_summary": {str(idx): _summarize_timing_rows(group_rows[idx]) for idx in group_rows},
    "last_response": responses[-1] if responses else {},
  }


def check_artifacts(args) -> dict:
  issues: list[str] = []
  groups = _groups_for_runtime(args)
  first_group = groups[0]
  expected_label_count = len(first_group)
  if args.score_rotate_shared_engine:
    for group in groups:
      if len(group) != expected_label_count:
        issues.append(f"shared text engine requires equal group sizes: {groups}")

  model_dir_info = _file_info(args.model_dir)
  if not args.model_dir.exists():
    issues.append(f"missing model dir: {args.model_dir}")

  logger = trt.Logger(trt.Logger.WARNING)
  runtime = trt.Runtime(logger)
  vision_path = args.vision_engine or _vision_engine_path(args)
  vision_info, vision_issues = _engine_info(runtime, vision_path)
  issues.extend(vision_issues)
  signal_enabled = args.runtime_mode == "visual-head" or args.enable_signal_head
  text_info = {"skipped": args.runtime_mode == "visual-head"}
  signal_info = {"skipped": not signal_enabled}
  if signal_enabled:
    signal_info, signal_issues = _engine_info(runtime, _signal_head_engine_path(args))
    issues.extend(signal_issues)
  if args.runtime_mode != "visual-head":
    text_path = _resolve_text_engine_path(args, first_group, require_keyed=not args.score_rotate_shared_engine and args.text_engine is None)
    text_info, text_issues = _engine_info(runtime, text_path)
    issues.extend(text_issues)

  if vision_info.get("deserialized"):
    if args.image_mode == "full" and args.image_size == 168:
      _check_shape(vision_info, "pixel_values", (96, 1176), issues, "vision")
      _check_shape(vision_info, "image_features", (24, 2048), issues, "vision")
  if args.runtime_mode != "visual-head" and text_info.get("deserialized"):
    text_batch = 1 if _label_decision_mode(args) == "choice" else expected_label_count
    text_output = _text_output(args)
    if args.text_seq_len > 0:
      _check_shape(text_info, "inputs_embeds", (text_batch, args.text_seq_len, 2048), issues, "text")
      _check_shape(text_info, "position_ids", (3, text_batch, args.text_seq_len), issues, "text")
      if text_output != "full_hidden" or "last_token_mask" in text_info.get("tensors", {}):
        _check_shape(text_info, "last_token_mask", (text_batch, args.text_seq_len, 1), issues, "text")
    output_name = _text_output_tensor_name(text_output)
    selected_shape = tuple(text_info.get("tensors", {}).get(output_name, {}).get("shape", ()))
    if selected_shape and selected_shape[0] != text_batch:
      issues.append(f"text {output_name} batch {selected_shape[0]} != expected text batch {text_batch}")
    if text_output in ("logits", "fixed_logits") and selected_shape:
      if _label_decision_mode(args) == "choice":
        choice_words = _choice_words_for_output_width(selected_shape[-1])
        if not choice_words:
          issues.append(f"text selected_logits width {selected_shape[-1]} is not a known choice vocabulary width")
        else:
          for group in groups:
            if _choice_group_spec(group).get("prompt_kind") == "traffic_signal":
              missing_signal_words = tuple(word for word in ("A", "B", "C") if word not in choice_words)
              if missing_signal_words:
                issues.append(
                  "traffic-signal choice engine lacks required "
                  f"{','.join(missing_signal_words)} token(s); rebuild the text engine"
                )
      else:
        expected_width = sum(len(ids) for ids in _score_label_ids(AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)))
        if selected_shape[-1] != expected_width:
          issues.append(f"text selected_logits width {selected_shape[-1]} != expected {expected_width}")
    if text_output in ("hidden", "full_hidden") and selected_shape and selected_shape[-1] != 2048:
      issues.append(f"text {output_name} width {selected_shape[-1]} != 2048")
    if text_output == "full_hidden" and selected_shape and tuple(selected_shape[1:]) != (args.text_seq_len, 2048):
      issues.append(f"text full_hidden shape {selected_shape} != ({text_batch}, {args.text_seq_len}, 2048)")
  if signal_enabled and signal_info.get("deserialized"):
    _check_shape(signal_info, "image_features", (24, 2048), issues, "signal")
    selected_shape = tuple(signal_info.get("tensors", {}).get("signal_logits", {}).get("shape", ()))
    if selected_shape and selected_shape[-1] != len(SIGNAL_CLASSES):
      issues.append(f"signal logits width {selected_shape[-1]} != {len(SIGNAL_CLASSES)}")

  nvcc = _nvcc_info()
  if not nvcc["exists"]:
    issues.append("nvcc not found")
  if not nvcc["has_compute_120"]:
    issues.append("nvcc does not report compute_120")
  if not nvcc["has_sm_120"]:
    issues.append("nvcc does not report sm_120")

  cuda = {
    "cuda_path": os.environ.get("CUDA_PATH", ""),
    "cuda_home": os.environ.get("CUDA_HOME", ""),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "torch_device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (),
    "tensorrt_version": trt.__version__,
    "tensorrt_fp4": hasattr(trt.DataType, "FP4") and hasattr(trt.BuilderFlag, "FP4"),
    "tensorrt_fp8": hasattr(trt.DataType, "FP8") and hasattr(trt.BuilderFlag, "FP8"),
    "nvcc": nvcc,
  }
  # (12, 0) = RTX 5060 Ti (Blackwell), the original validated NVFP4 dev box.
  # (8, 9) = Ada Lovelace (e.g. L40S), validated here on the fp8 text-precision path.
  _validated_device_capabilities = ((12, 0), (8, 9))
  if cuda["torch_device_capability"] and tuple(cuda["torch_device_capability"]) not in _validated_device_capabilities:
    issues.append(f"unexpected CUDA device capability: {cuda['torch_device_capability']}")
  if args.runtime_mode != "visual-head" and _text_precision(args) == "nvfp4" and not cuda["tensorrt_fp4"]:
    issues.append("TensorRT FP4 support is unavailable")
  if args.runtime_mode != "visual-head" and _text_precision(args) == "fp8" and not cuda["tensorrt_fp8"]:
    issues.append("TensorRT FP8 support is unavailable")

  contract = _runtime_contract(args, groups)
  manifest = _validate_manifest(args, groups)
  if args.require_manifest:
    issues.extend(manifest["issues"])

  result = {
    "kind": "check_artifacts",
    "ok": not issues,
    "issues": issues,
    "model_dir": model_dir_info,
    "artifact_dir": str(args.artifact_dir),
    "image_mode": args.image_mode,
    "image_size": args.image_size,
    "text_seq_len": args.text_seq_len,
    "score_label_groups": groups,
    "score_rotate_shared_engine": args.score_rotate_shared_engine,
    "expected_label_count": expected_label_count,
    "vision_engine": vision_info,
    "text_engine": text_info,
    "signal_head": signal_info,
    "cuda": cuda,
    "runtime_contract": {
      "contract_sha256": contract["contract_sha256"],
      "manifest_version": MANIFEST_VERSION,
    },
    "manifest": manifest,
  }
  if args.write_manifest and not issues:
    if signal_enabled:
      acceptance = signal_acceptance(args)
      result["signal_acceptance"] = acceptance
      if not acceptance["ok"]:
        result["ok"] = False
        result["issues"] = acceptance["issues"]
        return result
    result["manifest"] = _write_manifest(args, groups, result)
  return result


def gate(args) -> dict:
  if args.iters <= 0:
    raise ValueError("gate requires --iters > 0")
  gate_args = argparse.Namespace(**vars(args))
  gate_args.require_manifest = True
  artifact_check = check_artifacts(gate_args)
  if not artifact_check["ok"]:
    return {
      "kind": "gate",
      "ok": False,
      "deadline_ms": args.deadline_ms,
      "artifact_check": artifact_check,
      "benchmark": None,
      "issues": ["artifact or manifest validation failed"],
    }

  signal_gate = None
  benchmark_result = signal_acceptance(gate_args) if args.runtime_mode == "visual-head" else benchmark_groups(gate_args)
  total = benchmark_result["stage_summary"].get("total_ms", {})
  p99 = float(total.get("p99", 0.0))
  max_latency = float(total.get("max", 0.0))
  issues = []
  if p99 > args.deadline_ms:
    issues.append(f"p99 total latency {p99:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  if max_latency > args.deadline_ms:
    issues.append(f"max total latency {max_latency:.3f} ms exceeds deadline {args.deadline_ms:.3f} ms")
  if args.runtime_mode != "visual-head" and args.enable_signal_head:
    signal_gate = signal_acceptance(gate_args)
    if not signal_gate["ok"]:
      issues.append("signal acceptance failed")
      issues.extend(signal_gate["issues"])
  return {
    "kind": "gate",
    "ok": not issues,
    "deadline_ms": args.deadline_ms,
    "p99_total_ms": p99,
    "max_total_ms": max_latency,
    "issues": issues,
    "artifact_check": {
      "ok": artifact_check["ok"],
      "issues": artifact_check["issues"],
      "manifest": artifact_check["manifest"],
      "runtime_contract": artifact_check["runtime_contract"],
    },
    "benchmark": benchmark_result,
    "signal_acceptance": signal_gate,
  }


def serve(args) -> None:
  if args.require_manifest:
    _enforce_manifest(args, _groups_for_runtime(args))
  if args.runtime_mode == "visual-head":
    if args.score_rotate_groups:
      raise RuntimeError("--score-rotate-groups is not valid with --runtime-mode visual-head")
    scorer = QwenTrtVisualHeadScorer(args)
  elif args.score_rotate_groups:
    scorer = QwenTrtRotatingLabelScorer(args, _score_groups(args.score_label_groups))
  else:
    scorer = QwenTrtLabelScorer(args)
  scorer.warmup(args.warmup)
  if args.ready_jsonl:
    print(json.dumps({"ready": True}, separators=(",", ":")), flush=True)
  for line in sys.stdin:
    try:
      payload = json.loads(line)
      if payload.get("control") == "reset_runtime_state":
        reset_runtime_state = getattr(scorer, "reset_runtime_state", None)
        if callable(reset_runtime_state):
          reset_runtime_state()
        response = {"ok": True, "control": "reset_runtime_state"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        continue
      payload_vehicle_state = str(payload.get("scene_board_state_text", args.vehicle_state))
      frame_id = int(payload.get("frame_id", 0))
      if args.runtime_mode == "visual-head":
        image = _image_from_payload(payload)
        vehicle_state = _vehicle_state_for_labels(args, (), payload_vehicle_state)
        response = scorer.score(image, vehicle_state)
      elif args.score_rotate_groups:
        response = scorer.score_payload(payload, payload_vehicle_state, frame_id)
      else:
        image = _image_from_payload(payload)
        vehicle_state = _vehicle_state_for_labels(args, getattr(scorer, "labels", ()), payload_vehicle_state)
        response = scorer.score(image, vehicle_state)
      response["frame_id"] = frame_id
      response["source_frame_id"] = response["frame_id"]
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="Build and benchmark fixed-shape Qwen2.5-VL TensorRT label scoring engines.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--artifact-dir", type=Path, default=Path(os.environ.get("QWEN_TRT_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)))
  parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
  parser.add_argument("--image-mode", choices=("full",), default="full")
  parser.add_argument("--image-size", type=int, default=168)
  parser.add_argument("--runtime-mode", choices=("score", "visual-head"), default="score", help="score uses the Qwen text tower label scorer; visual-head uses Qwen visual features plus an exported signal classifier head.")
  parser.add_argument("--vision-precision", choices=("fp32", "fp16"), default="fp32", help="Vision engine build precision. fp32 is the verified-correct default for Qwen's visual tower; fp16 is faster but currently numerically wrong.")
  parser.add_argument("--text-seq-len", type=int, default=0, help="Fixed processor max_length for shared-shape text engines. 0 keeps dynamic padding.")
  parser.add_argument("--text-precision", choices=("nvfp4", "fp8", "fp16"), default="nvfp4", help="Text engine precision. nvfp4 is the fast Blackwell path; fp8 is a higher-fidelity Blackwell candidate; fp16 is a correctness control path.")
  parser.add_argument("--text-output", choices=("logits", "hidden", "full_hidden", "fixed_logits"), default="logits", help="Text engine output. hidden runs lm_head outside TensorRT to avoid selected-logit export drift; full_hidden returns the sequence and gathers the answer token outside TensorRT; fixed_logits bakes a fixed answer-token index into the engine.")
  parser.add_argument("--debug-tensor-stats", action=argparse.BooleanOptionalAction, default=False, help="Include tensor min/max/nonzero diagnostics in benchmark responses for TRT scoring root-cause work.")
  parser.add_argument("--fixed-output-index", type=int, default=-1, help="Answer-token index baked into --text-output fixed_logits. -1 computes it from the build prompt; runtime rejects mismatched prompt lengths when this is nonnegative.")
  parser.add_argument("--text-strongly-typed", action=argparse.BooleanOptionalAction, default=False, help="Build text TensorRT engines with NetworkDefinitionCreationFlag.STRONGLY_TYPED. This is a FP8/QDQ correctness probe and keeps separate engine filenames.")
  parser.add_argument("--torch-attn-implementation", choices=("sdpa", "eager"), default="sdpa", help="Torch attention implementation used while exporting Qwen. eager creates separate text engine filenames for ONNX/TensorRT correctness probes.")
  parser.add_argument("--verify-text-onnx-fidelity", action=argparse.BooleanOptionalAction, default=False, help="After text ONNX export, run ONNX Runtime against the same tensors and fail the build if it diverges from the PyTorch wrapper.")
  parser.add_argument("--text-onnx-max-mean-error", type=float, default=0.05, help="Maximum allowed mean absolute error for --verify-text-onnx-fidelity.")
  parser.add_argument("--text-onnx-max-error", type=float, default=2.0, help="Maximum allowed absolute error for --verify-text-onnx-fidelity.")
  parser.add_argument("--text-onnx-require-choice-match", action=argparse.BooleanOptionalAction, default=True, help="For choice-mode exports, require ONNX and PyTorch to select the same labels during --verify-text-onnx-fidelity.")
  parser.add_argument("--label-decision-mode", choices=("binary", "choice"), default="binary", help="binary scores yes/no label questions; choice scores one answer word for a whole label group.")
  parser.add_argument("--score-prompt-mode", choices=("full", "worker-full", "construction-compact", "construction-score"), default="full", help="full uses the TensorRT full per-label score prompt; worker-full matches qwen_label_rtp_worker's PyTorch score prompt exactly; construction-compact uses a short construction prompt; construction-score keeps the critical full-prompt construction side rules under fixed text lengths.")
  parser.add_argument("--text-position-mode", choices=("auto", "qwen", "clamp127", "zero"), default="auto", help="Position ids fed to the TensorRT text tower. auto keeps Qwen positions for binary mode and clamp127 for choice mode.")
  parser.add_argument("--text-position-dtype", choices=("int64", "int32"), default="int64", help="Integer dtype for TensorRT text position_ids. int32 writes distinct engine filenames and is a correctness probe for TensorRT position binding issues.")
  parser.add_argument("--score-labels", default="construction_left,construction_right")
  parser.add_argument("--score-label-groups", default=";".join(",".join(group) for group in DEFAULT_SCORE_LABEL_GROUPS))
  parser.add_argument("--score-rotate-groups", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--score-rotate-shared-engine", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--score-cache-ttl-frames", type=int, default=3)
  parser.add_argument("--score-durable-labels", default=",".join(DEFAULT_DURABLE_SCORE_LABELS))
  parser.add_argument("--score-negative-clear-threshold", type=float, default=2.0)
  parser.add_argument(
    "--construction-mirror-consistency",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="For the construction left/right choice group, also score a horizontally mirrored scene board and accept lateral construction labels only when the mirrored answer maps back to the original answer.",
  )
  parser.add_argument(
    "--construction-mirror-fusion",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="For construction choice groups, also score a horizontally mirrored scene board and fuse original+mirrored answer-word scores after mapping mirror left/right back to the original frame.",
  )
  parser.add_argument("--enable-visual-fallbacks", action="store_true", help="Demo-only pixel fallback for synthetic scene boards. Do not use for production road runs.")
  parser.add_argument("--construction-side-choice", action=argparse.BooleanOptionalAction, default=False, help="In rotating score mode, score construction_left / construction_right groups with the construction choice prompt and optional construction text engine.")
  parser.add_argument("--construction-edge-choice", action=argparse.BooleanOptionalAction, default=False, help="In rotating score mode, score construction_blue_edge / construction_purple_edge groups with the construction choice prompt and optional construction text engine so mirror fusion can use edge-color word scores.")
  parser.add_argument("--construction-edge-binary", action=argparse.BooleanOptionalAction, default=False, help="In rotating score mode, score construction_blue_edge / construction_purple_edge groups with compact binary yes/no prompts while leaving other groups in the selected decision mode.")
  parser.add_argument("--construction-candidate-binary", action=argparse.BooleanOptionalAction, default=False, help="In rotating score mode, score construction_blocks_left_candidate / construction_blocks_right_candidate groups against auxiliary candidate boards with compact binary prompts.")
  parser.add_argument("--construction-candidate-choice", action=argparse.BooleanOptionalAction, default=False, help="In rotating score mode, score construction_blocks_left_candidate / construction_blocks_right_candidate groups with the construction choice prompt, usually using --construction-text-engine.")
  parser.add_argument("--construction-candidate-relative-choice", action=argparse.BooleanOptionalAction, default=False, help="For construction candidate choice groups, allow the cyan-vs-pink relative logit margin to select the more obstructed candidate only when neutral none is not dominant.")
  parser.add_argument("--construction-candidate-relative-margin", type=float, default=1.5)
  parser.add_argument("--construction-candidate-relative-neutral-margin", type=float, default=CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN, help="When --construction-candidate-relative-choice is enabled, keep neutral none if its logit is more than this many points above the best cyan/pink candidate word.")
  parser.add_argument("--construction-candidate-score-resolve", action=argparse.BooleanOptionalAction, default=False, help="When both candidate obstruction labels are positive, keep only the higher-obstruction candidate if the Qwen score gap exceeds --construction-candidate-diff-margin; otherwise clear candidate side.")
  parser.add_argument("--construction-candidate-diff-margin", type=float, default=0.4)
  parser.add_argument("--enable-visual-signal-fallback", action="store_true", help="Demo-only red/green pixel fallback for synthetic traffic-light overlays.")
  parser.add_argument("--enable-visual-construction-fallback", action="store_true", help="Demo-only cone/barrier color fallback for synthetic MetaDrive boards.")
  parser.add_argument("--enable-visual-stop-fallback", action="store_true", help="Demo-only stop-sign pixel fallback for synthetic boards.")
  parser.add_argument("--enable-signal-head", action=argparse.BooleanOptionalAction, default=False, help="Run the calibrated traffic-signal visual head in score mode and merge its red/green labels into the text-label result.")
  parser.add_argument("--lead-state-consistency", action=argparse.BooleanOptionalAction, default=True, help="Use production lead-track physics to veto impossible Qwen lead choices such as none/merge for a centered lead.")
  parser.add_argument("--vehicle-state", default="speed=5.0 mps")
  parser.add_argument("--workspace-gb", type=int, default=6)
  parser.add_argument("--text-engine", type=Path, default=None)
  parser.add_argument("--vision-feature-clip-abs", type=float, default=0.0, help="Optional symmetric absolute clamp on visual embeddings before the text tower. 0 disables clipping.")
  parser.add_argument("--construction-text-engine", type=Path, default=None, help="Optional text TensorRT engine used only for construction choice groups in rotating score mode.")
  parser.add_argument("--construction-text-precision", choices=("same", "nvfp4", "fp8", "fp16"), default="same", help="Metadata/engine-resolution precision override used only for construction choice groups.")
  parser.add_argument("--construction-text-seq-len", type=int, default=0, help="Optional fixed text sequence length used only with --construction-text-engine construction group overrides.")
  parser.add_argument("--vision-engine", type=Path, default=None)
  parser.add_argument("--signal-head-engine", type=Path, default=None)
  parser.add_argument("--signal-head-weights", type=Path, default=None)
  parser.add_argument("--signal-red-images", default=str(DEFAULT_SIGNAL_RED_IMAGE))
  parser.add_argument("--signal-green-images", default=str(DEFAULT_SIGNAL_GREEN_IMAGE))
  parser.add_argument("--signal-none-images", default=f"{DEFAULT_SIGNAL_NONE_IMAGE},{DEFAULT_IMAGE}")
  parser.add_argument("--signal-augmentations", type=int, default=48)
  parser.add_argument("--signal-acceptance-augmentations", type=int, default=8)
  parser.add_argument("--signal-train-seed", type=int, default=7)
  parser.add_argument("--signal-train-epochs", type=int, default=300)
  parser.add_argument("--signal-train-lr", type=float, default=3e-3)
  parser.add_argument("--signal-min-probability", type=float, default=0.70)
  parser.add_argument("--signal-min-margin", type=float, default=0.75)
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--score-thresholds", default="")
  parser.add_argument(
    "--use-payload-vehicle-state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Permit scene_board_state_text from each JSONL request. The default scope uses it only for label groups that need physical track state.",
  )
  parser.add_argument(
    "--payload-vehicle-state-scope",
    choices=("auto", "all", "none"),
    default="auto",
    help="auto uses live payload state only for physical lead/vehicle label groups; all uses it for every group; none always uses --vehicle-state.",
  )
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--ready-jsonl", action="store_true", help="Emit a one-line JSON ready marker on stdout after loading and warmup.")
  parser.add_argument("--iters", type=int, default=60)
  parser.add_argument("--deadline-ms", type=float, default=50.0, help="Latency deadline for the gate subcommand.")
  parser.add_argument("--out", type=Path, default=None, help="Optional JSON path for the command result.")
  parser.add_argument("--manifest", type=Path, default=None, help="Runtime manifest path. Defaults to artifact-dir/qwen_trt_runtime_manifest.json.")
  parser.add_argument("--write-manifest", action="store_true", help="Write the runtime contract manifest from check-artifacts when validation succeeds.")
  parser.add_argument("--require-manifest", action="store_true", help="Reject runtime commands if the manifest contract does not match current args/code/model metadata.")
  parser.set_defaults(label_keyed_text_engine=False)
  sub = parser.add_subparsers(dest="cmd", required=True)
  sub.add_parser("build-text")
  sub.add_parser("build-text-groups")
  sub.add_parser("build-signal-head")
  sub.add_parser("build-vision")
  sub.add_parser("benchmark")
  sub.add_parser("benchmark-visual-head")
  sub.add_parser("benchmark-groups")
  sub.add_parser("signal-acceptance")
  sub.add_parser("check-artifacts")
  sub.add_parser("gate")
  sub.add_parser("serve")
  sub.add_parser("all")
  args = parser.parse_args()
  args.score_thresholds_map = _parse_score_threshold_map(args.score_thresholds)

  args.artifact_dir.mkdir(parents=True, exist_ok=True)
  if args.cmd == "build-text":
    result = build_text_engine(args)
  elif args.cmd == "build-text-groups":
    result = build_text_group_engines(args)
  elif args.cmd == "build-signal-head":
    result = build_signal_head(args)
  elif args.cmd == "build-vision":
    result = build_vision_engine(args)
  elif args.cmd == "benchmark":
    result = benchmark(args)
  elif args.cmd == "benchmark-visual-head":
    result = benchmark_visual_head(args)
  elif args.cmd == "benchmark-groups":
    result = benchmark_groups(args)
  elif args.cmd == "signal-acceptance":
    result = signal_acceptance(args)
  elif args.cmd == "check-artifacts":
    result = check_artifacts(args)
  elif args.cmd == "gate":
    result = gate(args)
  elif args.cmd == "serve":
    serve(args)
    return
  elif args.cmd == "all":
    result = {
      "text": build_text_engine(args),
      "vision": build_vision_engine(args),
      "benchmark": benchmark(args),
    }
  else:
    raise AssertionError(args.cmd)

  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))
  if args.cmd in ("check-artifacts", "gate", "signal-acceptance") and not result.get("ok", False):
    sys.exit(2)


if __name__ == "__main__":
  main()
