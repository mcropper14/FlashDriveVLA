#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000
DEFAULT_PARO_TARGET_MODEL = Path("/mnt/j/temp_alpamayo/Alpamayo-1.5-10B-finetuned-PARO")
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def _insert_alpamayo_paths(alpamayo_root: Path) -> None:
  sys.path.insert(0, str(alpamayo_root))
  sys.path.insert(0, str(alpamayo_root / "src"))
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _normalize_path(raw: str | Path) -> Path:
  value = str(raw)
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return Path(os.path.expanduser(value))
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return Path(f"/mnt/{drive}/{rest}")


def _torch_dtype(torch_mod: Any, name: str) -> Any:
  normalized = name.strip().lower()
  if normalized in ("bfloat16", "bf16"):
    return torch_mod.bfloat16
  if normalized in ("float16", "fp16", "half"):
    return torch_mod.float16
  if normalized in ("float32", "fp32"):
    return torch_mod.float32
  raise ValueError(f"unsupported Alpamayo dtype: {name}")


def _sync(torch_mod: Any) -> None:
  if torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  pos = (len(ordered) - 1) * q
  lo = int(pos)
  hi = min(lo + 1, len(ordered) - 1)
  frac = pos - lo
  return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values: list[float]) -> dict[str, float | int]:
  return {
    "count": len(values),
    "mean_ms": round(mean(values), 3) if values else 0.0,
    "min_ms": round(min(values), 3) if values else 0.0,
    "p50_ms": round(_percentile(values, 0.50), 3),
    "p95_ms": round(_percentile(values, 0.95), 3),
    "p99_ms": round(_percentile(values, 0.99), 3),
    "max_ms": round(max(values), 3) if values else 0.0,
  }


def _gpu_stats(torch_mod: Any) -> dict[int, dict[str, float]]:
  stats: dict[int, dict[str, float]] = {}
  if not torch_mod.cuda.is_available():
    return stats
  for idx in range(torch_mod.cuda.device_count()):
    free_bytes, total_bytes = torch_mod.cuda.mem_get_info(idx)
    stats[idx] = {
      "used_gib": round((total_bytes - free_bytes) / (1024 ** 3), 3),
      "free_gib": round(free_bytes / (1024 ** 3), 3),
      "total_gib": round(total_bytes / (1024 ** 3), 3),
    }
  return stats


def _progress(event: str, **fields: Any) -> None:
  payload = {"event": event, "time_unix": round(time.time(), 3)}
  payload.update(fields)
  print(json.dumps(payload, sort_keys=True), flush=True)


def _module_device(module: Any) -> Any:
  try:
    return next(module.parameters()).device
  except StopIteration:
    raise RuntimeError(f"module has no parameters: {module}")


def _load_dflash_model(model_dir: Path, *, torch_mod: Any, dtype: Any, attn_implementation: str) -> tuple[Any, str]:
  from dflash.model import DFlashDraftModel

  try:
    model = DFlashDraftModel.from_pretrained(
      str(model_dir),
      attn_implementation=attn_implementation,
      dtype=dtype,
    )
    return model, "from_pretrained"
  except AttributeError as exc:
    if "dflash_config" not in str(exc):
      raise

  from safetensors.torch import load_file
  from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

  raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
  config = Qwen3Config(**raw_config)
  config.dflash_config = raw_config
  config._attn_implementation = attn_implementation
  for key, value in raw_config.items():
    if not hasattr(config, key):
      setattr(config, key, value)
  model = DFlashDraftModel(config)
  state = load_file(str(model_dir / "model.safetensors"), device="cpu")
  missing, unexpected = model.load_state_dict(state, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"DFlash state dict mismatch: missing={missing}, unexpected={unexpected}")
  return model.to(dtype=dtype), "manual_config_shim"


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


def _set_attention_implementation(module: Any, attn_implementation: str) -> None:
  for submodule in module.modules():
    config = getattr(submodule, "config", None)
    if config is not None:
      setattr(config, "_attn_implementation", attn_implementation)
      if hasattr(config, "attn_implementation"):
        setattr(config, "attn_implementation", attn_implementation)


def _patch_tie_weights_compat() -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  import alpamayo1_5.models.base_model as base_mod

  def tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
    if hasattr(self, "vlm") and hasattr(self.vlm, "tie_weights"):
      return self.vlm.tie_weights()
    return None

  base_mod.ReasoningVLA.tie_weights = tie_weights
  alpamayo_mod.Alpamayo1_5.tie_weights = tie_weights


def _patch_manual_greedy_generation() -> None:
  import torch
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

  if getattr(alpamayo_mod.Alpamayo1_5, "_openpilot_manual_greedy_patch", False):
    return

  def manual_greedy_vlm_generate(
    self: Any,
    input_ids: Any,
    tokenized_data: dict[str, Any],
    eos_token_id: int,
    max_generation_length: int,
    runtime_profile: dict[str, float | int] | None = None,
  ) -> tuple[Any, Any]:
    generated_sequences = input_ids
    model_kwargs = dict(tokenized_data)
    model_kwargs["use_cache"] = True
    if model_kwargs.get("position_ids") is None:
      model_kwargs["position_ids"] = self.vlm._prepare_position_ids_for_generation(input_ids, model_kwargs)
    logits_processor = ExpertLogitsProcessor(
      traj_token_offset=self.config.traj_token_start_idx,
      traj_vocab_size=self.config.traj_vocab_size,
    )
    eos_seen = False
    prompt_cache = None
    prefill_seconds = 0.0
    decode_seconds = 0.0
    decode_forwards = 0

    for step_idx in range(max_generation_length + 1):
      model_inputs = self.vlm.prepare_inputs_for_generation(
        generated_sequences,
        next_sequence_length=input_ids.shape[1] if step_idx == 0 else 1,
        is_first_iteration=step_idx == 0,
        **model_kwargs,
      )
      model_inputs["logits_to_keep"] = 1
      step_start = time.perf_counter()
      vlm_outputs = self.vlm(**model_inputs, return_dict=True)
      step_seconds = time.perf_counter() - step_start
      if step_idx == 0:
        prefill_seconds += step_seconds
      else:
        decode_seconds += step_seconds
        decode_forwards += 1

      prompt_cache = vlm_outputs.past_key_values
      model_kwargs = self.vlm._update_model_kwargs_for_generation(
        vlm_outputs,
        model_kwargs,
        is_encoder_decoder=False,
      )
      for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        model_kwargs.pop(key, None)

      if step_idx >= max_generation_length:
        break

      scores = vlm_outputs.logits[:, -1, :]
      scores = logits_processor(generated_sequences, scores)
      next_token = scores.argmax(dim=-1, keepdim=True).to(generated_sequences.device)
      generated_sequences = torch.cat([generated_sequences, next_token], dim=-1)
      if eos_seen:
        break
      eos_seen = bool(torch.all(next_token == eos_token_id).detach().cpu().item())

    if runtime_profile is not None:
      runtime_profile["manual_vlm_prefill_seconds"] = prefill_seconds
      runtime_profile["manual_vlm_decode_seconds"] = decode_seconds
      runtime_profile["manual_vlm_decode_forwards"] = decode_forwards
      runtime_profile["manual_vlm_generated_tokens"] = int(generated_sequences.shape[1] - input_ids.shape[1])

    if prompt_cache is None:
      raise RuntimeError("manual VLM generation did not produce a cache")
    return generated_sequences, prompt_cache

  alpamayo_mod.Alpamayo1_5._manual_greedy_vlm_generate = manual_greedy_vlm_generate
  alpamayo_mod.Alpamayo1_5._openpilot_manual_greedy_patch = True


def _patch_alpamayo_init_for_paro(
  *,
  marlin_input_dtype: str,
  compute_dtype: Any,
  output_dtype: str,
) -> None:
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_mod
  from paro_native_marlin import apply_native_paro_linear_replacements

  patch_key = (marlin_input_dtype, str(compute_dtype), output_dtype)
  if getattr(alpamayo_mod.Alpamayo1_5, "_openpilot_paro_patch_key", None) == patch_key:
    return

  original_init = getattr(
    alpamayo_mod.Alpamayo1_5,
    "_openpilot_paro_original_init",
    alpamayo_mod.Alpamayo1_5.__init__,
  )

  def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    original_init(self, *args, **kwargs)
    records = apply_native_paro_linear_replacements(
      self,
      marlin_input_dtype=marlin_input_dtype,
      compute_dtype=compute_dtype,
      output_dtype=output_dtype,
    )
    self._openpilot_paro_replacement_records = records

  alpamayo_mod.Alpamayo1_5._openpilot_paro_original_init = original_init
  alpamayo_mod.Alpamayo1_5.__init__ = patched_init
  alpamayo_mod.Alpamayo1_5._openpilot_paro_patch_key = patch_key


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


def _repair_flattened_visual_patch_embed(
  torch_mod: Any,
  model: Any,
  model_dir: Path,
) -> dict[str, Any]:
  key = "vlm.model.visual.patch_embed.proj.weight"
  index_path = model_dir / "model.safetensors.index.json"
  try:
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
  except Exception as exc:
    return {"status": "error", "type": type(exc).__name__, "message": str(exc)}
  if key not in weight_map:
    return {"status": "skipped", "reason": "key_not_in_index", "key": key}

  from safetensors.torch import safe_open

  target = model.vlm.model.visual.patch_embed.proj.weight
  try:
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as handle:
      weight = handle.get_tensor(key)
  except Exception as exc:
    return {"status": "error", "type": type(exc).__name__, "message": str(exc)}

  if weight.ndim != 2 or int(weight.numel()) != int(target.numel()):
    return {
      "status": "skipped",
      "reason": "shape_not_compatible",
      "key": key,
      "source_shape": list(weight.shape),
      "target_shape": list(target.shape),
    }

  try:
    with torch_mod.no_grad():
      target.copy_(weight.reshape_as(target).to(device=target.device, dtype=target.dtype))
  except Exception as exc:
    return {"status": "error", "type": type(exc).__name__, "message": str(exc)}

  return {
    "status": "ok",
    "key": key,
    "source_shape": list(weight.shape),
    "target_shape": list(target.shape),
  }


def _synthetic_image_frames(torch_mod: Any, width: int, height: int, num_frames: int) -> Any:
  import numpy as np

  streams = []
  for camera_idx in range(2):
    frames = []
    for frame_idx in range(num_frames):
      y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
      x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
      r = np.broadcast_to((x + frame_idx * 7 + camera_idx * 17).astype(np.uint8), (height, width))
      g = np.broadcast_to((y + frame_idx * 5 + camera_idx * 11).astype(np.uint8), (height, width))
      b = np.full((height, width), 80 + camera_idx * 40, dtype=np.uint8)
      rgb = np.stack([r, g, b], axis=2)
      frames.append(torch_mod.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
    streams.append(torch_mod.stack(frames, dim=0))
  return torch_mod.stack(streams, dim=0)


def _synthetic_ego_history(torch_mod: Any) -> tuple[Any, Any]:
  import numpy as np

  t = np.arange(16, dtype=np.float32) * 0.1
  xyz = np.column_stack([t * 2.0, np.zeros_like(t), np.zeros_like(t)]).astype(np.float32)
  rot = np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0)
  return (
    torch_mod.from_numpy(xyz).unsqueeze(0).unsqueeze(0),
    torch_mod.from_numpy(rot).unsqueeze(0).unsqueeze(0),
  )


def _load_contract_prompt(args: argparse.Namespace, torch_mod: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
  if args.contract_request_json is None:
    raise ValueError("--contract-request-json is required when --prompt-source=contract")

  import numpy as np
  from openpilot.selfdrive.alpamayo.local_adapter import (
    STREAM_TO_ALPAMAYO_CAMERA_INDEX,
    _decode_frame_rgb,
    _group_frames,
    _request_ego_history,
    _request_nav_text,
  )

  path = _normalize_path(args.contract_request_json)
  wrapper = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(wrapper, dict):
    raise ValueError(f"contract request JSON root must be an object: {path}")
  request = wrapper.get("request_payload") or wrapper.get("request") or wrapper
  if not isinstance(request, dict):
    raise ValueError(f"contract request JSON does not contain an object request: {path}")

  camera_bundle = request.get("cameraBundle") if isinstance(request.get("cameraBundle"), dict) else {}
  streams = tuple(camera_bundle.get("streamOrder") or ("wideRoad", "road"))
  frames_per_camera = int(camera_bundle.get("framesPerCamera") or args.num_frames)
  frames_by_stream = _group_frames(request, streams, frames_per_camera)

  image_frames = []
  camera_indices = []
  prompt_frames = []
  for stream in streams:
    if stream not in STREAM_TO_ALPAMAYO_CAMERA_INDEX:
      raise ValueError(f"no Alpamayo camera index mapping for stream={stream}")
    decoded = [_decode_frame_rgb(frame) for frame in frames_by_stream[stream]]
    stream_tensor = torch_mod.stack([
      torch_mod.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1) for rgb in decoded
    ], dim=0)
    image_frames.append(stream_tensor)
    camera_indices.append(STREAM_TO_ALPAMAYO_CAMERA_INDEX[stream])
    prompt_frames.extend({
      "stream": stream,
      "frameId": int(frame.get("frameId", -1)),
      "timestampEof": int(frame.get("timestampEof", 0)),
      "encoding": str(frame.get("encoding", "")),
      "width": int(frame.get("width", 0)),
      "height": int(frame.get("height", 0)),
    } for frame in frames_by_stream[stream])

  ego_xyz, ego_rot = _request_ego_history(request)
  metadata = wrapper.get("metadata") if isinstance(wrapper.get("metadata"), dict) else None
  contract_meta = {
    "contract_request_json": str(path),
    "contract_metadata": metadata,
    "contract_request_id": request.get("requestId"),
    "contract_camera_bundle": camera_bundle,
    "contract_frames": prompt_frames,
    "nav_text": _request_nav_text(request),
    "frames_per_camera": frames_per_camera,
    "streams": list(streams),
  }
  return (
    torch_mod.stack(image_frames, dim=0),
    torch_mod.tensor(camera_indices, dtype=torch_mod.int64),
    torch_mod.from_numpy(ego_xyz).unsqueeze(0).unsqueeze(0),
    torch_mod.from_numpy(ego_rot).unsqueeze(0).unsqueeze(0),
    contract_meta,
  )


def _build_target_model(args: argparse.Namespace, torch_mod: Any) -> Any:
  from transformers import BitsAndBytesConfig
  from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

  if args.device_map_mode == "auto":
    device_map: str | dict[str, int] = "auto"
  elif args.device_map_mode == "current_split":
    device_map = _build_current_split_device_map(args.split_index)
  else:
    raise ValueError(f"unsupported device_map_mode={args.device_map_mode}")

  target_model = _normalize_path(args.target_model)
  load_kwargs: dict[str, Any] = {
    "dtype": torch_mod.bfloat16 if not args.paro_native else _torch_dtype(torch_mod, args.model_dtype),
    "attn_implementation": args.attn_implementation,
    "device_map": device_map,
    "max_memory": _gpu_only_max_memory(torch_mod, args.gpu_mem_gib),
    "low_cpu_mem_usage": True,
  }
  if args.paro_native:
    if args.quantization != "none":
      raise ValueError("quantization is unsupported when --paro-native is set")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)
    import paroquant.inference.backends.vllm.plugin  # noqa: F401
    _patch_tie_weights_compat()
    _patch_alpamayo_init_for_paro(
      marlin_input_dtype=args.paro_marlin_input_dtype,
      compute_dtype=_torch_dtype(torch_mod, args.paro_compute_dtype),
      output_dtype=args.paro_output_dtype,
    )
    load_kwargs["ignore_mismatched_sizes"] = True

  if args.quantization != "none":
    compute_dtype = getattr(torch_mod, args.bnb_compute_dtype)
    if args.quantization == "bnb_4bit_nf4":
      load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
      )
    elif args.quantization == "bnb_4bit_fp4":
      load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=compute_dtype,
      )
    else:
      raise ValueError(f"unsupported quantization={args.quantization}")

  restore_loader = _patch_flattened_conv3d_loader(torch_mod)
  try:
    model = Alpamayo1_5.from_pretrained(str(target_model), **load_kwargs)
  finally:
    restore_loader()
  _assert_no_cpu_offload(model)
  if args.paro_native:
    model._openpilot_paro_visual_patch_embed_repair = _repair_flattened_visual_patch_embed(
      torch_mod,
      model,
      target_model,
    )
    from paro_native_marlin import finalize_native_paro_modules_for_device_map

    model._openpilot_paro_finalize = finalize_native_paro_modules_for_device_map(model)
  if args.expert_attn_implementation is not None:
    _set_attention_implementation(model.expert, args.expert_attn_implementation)
  return model


def _prepare_prompt(args: argparse.Namespace, model: Any, torch_mod: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
  from alpamayo1_5 import helper

  nav_text = None
  source_meta: dict[str, Any] = {}
  num_frames_per_camera = args.num_frames
  if args.prompt_source == "synthetic":
    image_frames = _synthetic_image_frames(
      torch_mod,
      width=args.synthetic_width,
      height=args.synthetic_height,
      num_frames=args.num_frames,
    )
    camera_indices = torch_mod.tensor([1, 6], dtype=torch_mod.int64)
    ego_history_xyz, ego_history_rot = _synthetic_ego_history(torch_mod)
  elif args.prompt_source == "contract":
    image_frames, camera_indices, ego_history_xyz, ego_history_rot, source_meta = _load_contract_prompt(args, torch_mod)
    nav_text = source_meta.get("nav_text")
    num_frames_per_camera = int(source_meta["frames_per_camera"])
  else:
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from timed_inference import select_camera_features

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    camera_features = select_camera_features(avdi, args.camera_mode)
    data = load_physical_aiavdataset(
      args.clip_id,
      t0_us=args.t0_us,
      avdi=avdi,
      camera_features=camera_features,
      num_frames=args.num_frames,
    )
    image_frames = data["image_frames"]
    camera_indices = data["camera_indices"]
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]

  messages = helper.create_message(
    frames=image_frames.flatten(0, 1),
    camera_indices=camera_indices,
    num_frames_per_camera=num_frames_per_camera,
    nav_text=nav_text,
  )
  helper.MIN_PIXELS = args.min_pixels
  helper.MAX_PIXELS = args.max_pixels
  if args.processor_model:
    helper.BASE_PROCESSOR_NAME = str(_normalize_path(args.processor_model))
  processor = helper.get_processor(model.tokenizer)
  inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
  )
  tokenized_data = helper.to_device(inputs, model.device)
  traj_data_vlm = {
    "ego_history_xyz": helper.to_device(ego_history_xyz, model.device),
    "ego_history_rot": helper.to_device(ego_history_rot, model.device),
  }
  input_ids = model.fuse_traj_tokens(tokenized_data.pop("input_ids"), traj_data_vlm)
  prompt_meta = {
    "prompt_source": args.prompt_source,
    "num_frames_per_camera": int(num_frames_per_camera),
    "image_frames_shape": [int(x) for x in image_frames.shape],
    "camera_indices": camera_indices.detach().cpu().tolist(),
    "input_ids_shape": [int(x) for x in input_ids.shape],
  }
  prompt_meta.update(source_meta)
  if "image_grid_thw" in tokenized_data:
    image_grid = tokenized_data["image_grid_thw"].detach().cpu()
    merge_len = int(processor.image_processor.merge_size) ** 2
    prompt_meta["image_grid_thw"] = image_grid.tolist()
    prompt_meta["image_tokens"] = int((image_grid.prod(dim=1) // merge_len).sum().item())
  return prompt_meta, tokenized_data, input_ids


def _standard_generate_once(
  model: Any,
  tokenized_data: dict[str, Any],
  input_ids: Any,
  *,
  max_generation_length: int,
  greedy: bool,
  mode: str,
  hf_cache_implementation: str | None,
  hf_compile: bool,
  hf_compile_fullgraph: bool,
  hf_compile_dynamic: str,
  hf_compile_backend: str,
  hf_compile_mode: str,
) -> Any:
  from transformers import LogitsProcessorList, StoppingCriteriaList
  from transformers.generation.configuration_utils import CompileConfig
  from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
  from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token

  eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
  if mode == "manual":
    if not greedy:
      raise ValueError("--standard-mode=manual requires --greedy")
    runtime_profile: dict[str, float | int] = {}
    sequences, _prompt_cache = model._manual_greedy_vlm_generate(
      input_ids=input_ids,
      tokenized_data=copy.copy(tokenized_data),
      eos_token_id=eos_token_id,
      max_generation_length=max_generation_length,
      runtime_profile=runtime_profile,
    )
    return SimpleNamespace(sequences=sequences, runtime_profile=runtime_profile)
  if mode != "hf":
    raise ValueError(f"unsupported standard generation mode: {mode}")

  generation_config = copy.deepcopy(model.vlm.generation_config)
  generation_config.top_p = 0.98
  generation_config.temperature = 0.6
  generation_config.do_sample = not greedy
  generation_config.num_return_sequences = 1
  generation_config.max_new_tokens = max_generation_length
  generation_config.output_logits = False
  generation_config.return_dict_in_generate = True
  generation_config.pad_token_id = model.tokenizer.pad_token_id
  generation_config.cache_implementation = hf_cache_implementation
  generation_config.disable_compile = not hf_compile
  generation_config.compile_config = None
  if hf_cache_implementation == "static":
    _patch_static_cache_mixed_kv_dtype()
  if hf_compile:
    generation_config.compile_config = CompileConfig(
      fullgraph=hf_compile_fullgraph,
      dynamic=_compile_dynamic(hf_compile_dynamic),
      backend=hf_compile_backend,
      mode=hf_compile_mode,
    )
  logits_processor = LogitsProcessorList([
    ExpertLogitsProcessor(
      traj_token_offset=model.config.traj_token_start_idx,
      traj_vocab_size=model.config.traj_vocab_size,
    )
  ])
  return model.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
    logits_processor=logits_processor,
    **copy.copy(tokenized_data),
  )


def _sample(torch_mod: Any, logits: Any, temperature: float) -> Any:
  if temperature < 1e-5:
    return torch_mod.argmax(logits, dim=-1)
  bsz, seq_len, vocab_size = logits.shape
  logits = logits.reshape(-1, vocab_size) / temperature
  probs = torch_mod.softmax(logits, dim=-1)
  return torch_mod.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _mask_traj_token_logits(model: Any, logits: Any) -> Any:
  masked = logits.clone()
  offset = int(model.config.traj_token_start_idx)
  size = int(model.config.traj_vocab_size)
  masked[..., offset:offset + size] = float("-inf")
  return masked


def _qwen3vl_forward(
  torch_mod: Any,
  target: Any,
  ids: Any,
  *,
  tokenized_data: dict[str, Any] | None,
  past_key_values: Any,
  output_hidden_states: bool,
  logits_to_keep: int = 0,
) -> Any:
  past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
  cache_position = torch_mod.arange(past_len, past_len + ids.shape[1], device=ids.device)
  kwargs: dict[str, Any] = {
    "input_ids": ids,
    "past_key_values": past_key_values,
    "use_cache": True,
    "cache_position": cache_position,
    "output_hidden_states": output_hidden_states,
    "logits_to_keep": logits_to_keep,
    "return_dict": True,
  }
  if past_len == 0 and tokenized_data is not None:
    kwargs.update(copy.copy(tokenized_data))
  return target(**kwargs)


def _hidden_from_layer_output(output: Any) -> Any:
  if isinstance(output, (tuple, list)):
    return output[0]
  if hasattr(output, "last_hidden_state"):
    return output.last_hidden_state
  return output


def _qwen3vl_forward_with_selected_hidden(
  torch_mod: Any,
  target: Any,
  ids: Any,
  *,
  tokenized_data: dict[str, Any] | None,
  past_key_values: Any,
  layer_ids: list[int],
  logits_to_keep: int = 0,
) -> tuple[Any, Any]:
  import torch

  captured: dict[int, Any] = {}
  hooks = []
  layers = target.model.language_model.layers
  for layer_id in layer_ids:
    if layer_id >= len(layers):
      raise RuntimeError(f"DFlash target layer {layer_id} out of range for {len(layers)} language layers")

    def hook(_module: Any, _inputs: Any, output: Any, *, layer_id: int = layer_id) -> None:
      captured[layer_id] = _hidden_from_layer_output(output)

    hooks.append(layers[layer_id].register_forward_hook(hook))

  try:
    output = _qwen3vl_forward(
      torch_mod,
      target,
      ids,
      tokenized_data=tokenized_data,
      past_key_values=past_key_values,
      output_hidden_states=False,
      logits_to_keep=logits_to_keep,
    )
  finally:
    for handle in hooks:
      handle.remove()

  missing = [layer_id for layer_id in layer_ids if layer_id not in captured]
  if missing:
    raise RuntimeError(f"did not capture DFlash target layers: {missing}")
  target_device = captured[layer_ids[-1]].device
  selected = [captured[layer_id].to(target_device) for layer_id in layer_ids]
  return output, torch.cat(selected, dim=-1)


def _load_mask_embedding(torch_mod: Any, draft_model_dir: Path, device: Any, dtype: Any) -> Any:
  path = draft_model_dir / "mask_embedding.pt"
  if not path.exists():
    raise FileNotFoundError(f"DFlash mask embedding is required but missing: {path}")
  try:
    value = torch_mod.load(str(path), map_location="cpu", weights_only=True)
  except TypeError:
    value = torch_mod.load(str(path), map_location="cpu")
  if isinstance(value, dict):
    if "mask_embedding" in value:
      value = value["mask_embedding"]
    elif len(value) == 1:
      value = next(iter(value.values()))
    else:
      raise ValueError(f"cannot identify mask embedding in {path}: keys={list(value.keys())}")
  if getattr(value, "ndim", None) != 1:
    raise ValueError(f"mask embedding must be rank-1, got shape={getattr(value, 'shape', None)}")
  return value.to(device=device, dtype=dtype)


def _embed_block_tokens(
  torch_mod: Any,
  embed_tokens: Any,
  block_output_ids: Any,
  *,
  mask_token_id: int,
  mask_embedding: Any,
  output_device: Any,
) -> Any:
  embed_device = _module_device(embed_tokens)
  mask = block_output_ids == mask_token_id
  safe_ids = block_output_ids.masked_fill(mask, 0)
  embeds = embed_tokens(safe_ids.to(embed_device)).to(output_device)
  if bool(mask.any().detach().cpu().item()):
    replacement = mask_embedding.to(device=output_device, dtype=embeds.dtype).view(1, 1, -1)
    embeds = torch_mod.where(mask.to(output_device).unsqueeze(-1), replacement, embeds)
  return embeds


def _dflash_generate_once(
  torch_mod: Any,
  model: Any,
  draft: Any,
  tokenized_data: dict[str, Any],
  input_ids: Any,
  *,
  mask_embedding: Any,
  max_generation_length: int,
  temperature: float,
) -> SimpleNamespace:
  from transformers import DynamicCache
  from alpamayo1_5.models.token_utils import to_special_token

  target = model.vlm
  eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
  block_size = int(draft.block_size)
  draft_device = _module_device(draft)
  lm_head_device = _module_device(target.lm_head)
  embed_tokens = target.model.language_model.embed_tokens

  num_input_tokens = int(input_ids.shape[1])
  max_length = num_input_tokens + max_generation_length
  output_ids = torch_mod.full(
    (1, max_length + block_size),
    int(draft.mask_token_id),
    dtype=torch_mod.long,
    device=input_ids.device,
  )
  output_ids[:, :num_input_tokens] = input_ids
  target_cache = DynamicCache()
  draft_cache = DynamicCache()
  layer_ids = list(draft.target_layer_ids)

  prefill_start = time.perf_counter()
  _progress("dflash_prefill_start", input_tokens=num_input_tokens, layer_ids=[int(x) for x in layer_ids])
  output, target_hidden = _qwen3vl_forward_with_selected_hidden(
    torch_mod,
    target,
    input_ids,
    tokenized_data=tokenized_data,
    past_key_values=target_cache,
    layer_ids=layer_ids,
    logits_to_keep=1,
  )
  _sync(torch_mod)
  first = _sample(torch_mod, _mask_traj_token_logits(model, output.logits), temperature)[:, -1:]
  output_ids[:, num_input_tokens:num_input_tokens + 1] = first.to(output_ids.device)
  target_hidden = target_hidden.to(draft_device)
  time_to_first_token_ms = (time.perf_counter() - prefill_start) * 1000.0
  _progress("dflash_prefill_done", time_to_first_token_ms=round(time_to_first_token_ms, 3), target_hidden_shape=[int(x) for x in target_hidden.shape])

  start = num_input_tokens
  acceptance_lengths: list[int] = []
  draft_ms = 0.0
  validate_ms = 0.0
  decode_start = time.perf_counter()

  while start < max_length:
    block_output_ids = output_ids[:, start:start + block_size].clone()
    if block_output_ids.shape[1] < block_size:
      break
    if block_size > 1:
      draft_start = time.perf_counter()
      _progress("dflash_draft_block_start", start_token=int(start), block_size=block_size)
      noise_embedding = _embed_block_tokens(
        torch_mod,
        embed_tokens,
        block_output_ids,
        mask_token_id=int(draft.mask_token_id),
        mask_embedding=mask_embedding,
        output_device=draft_device,
      )
      position_ids = torch_mod.arange(
        draft_cache.get_seq_length(),
        start + block_size,
        device=draft_device,
        dtype=torch_mod.long,
      ).unsqueeze(0)
      draft_hidden = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=draft_cache,
        use_cache=True,
        is_causal=False,
      )[:, 1 - block_size:, :]
      draft_logits = _mask_traj_token_logits(model, target.lm_head(draft_hidden.to(lm_head_device)))
      _sync(torch_mod)
      draft_ms += (time.perf_counter() - draft_start) * 1000.0
      draft_cache.crop(start)
      block_output_ids[:, 1:] = _sample(torch_mod, draft_logits, temperature).to(block_output_ids.device)
      _progress("dflash_draft_block_done", start_token=int(start), cumulative_draft_ms=round(draft_ms, 3))

    validate_start = time.perf_counter()
    _progress("dflash_validate_block_start", start_token=int(start), block_tokens=int(block_output_ids.shape[1]))
    output, target_hidden = _qwen3vl_forward_with_selected_hidden(
      torch_mod,
      target,
      block_output_ids,
      tokenized_data=None,
      past_key_values=target_cache,
      layer_ids=layer_ids,
    )
    _sync(torch_mod)
    validate_ms += (time.perf_counter() - validate_start) * 1000.0
    _progress("dflash_validate_block_done", start_token=int(start), cumulative_validate_ms=round(validate_ms, 3))

    posterior = _sample(torch_mod, _mask_traj_token_logits(model, output.logits), temperature).to(block_output_ids.device)
    accepted = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    accepted_len = int(accepted) + 1
    output_ids[:, start:start + accepted_len] = block_output_ids[:, :accepted_len]
    if start + accepted_len < output_ids.shape[1]:
      output_ids[:, start + accepted_len] = posterior[:, int(accepted)]
    start += accepted_len
    target_cache.crop(start)
    acceptance_lengths.append(accepted_len)

    if block_size > 1:
      target_hidden = target_hidden[:, :accepted_len, :].to(draft_device)

    if eos_token_id in output_ids[:, num_input_tokens:start]:
      break

  output_ids = output_ids[:, :min(start + 1, max_length)]
  new_tokens = int(output_ids.shape[1] - num_input_tokens)
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  return SimpleNamespace(
    sequences=output_ids,
    new_tokens=new_tokens,
    time_to_first_token_ms=time_to_first_token_ms,
    decode_ms=decode_ms,
    draft_ms=draft_ms,
    validate_ms=validate_ms,
    acceptance_lengths=acceptance_lengths,
  )


def _decode_new(model: Any, sequences: Any, input_len: int) -> str:
  ids = sequences[0, input_len:].detach().cpu().tolist()
  return model.tokenizer.decode(ids, skip_special_tokens=False)


def _compile_dynamic(value: str) -> bool | None:
  normalized = value.strip().lower()
  if normalized == "none":
    return None
  if normalized == "true":
    return True
  if normalized == "false":
    return False
  raise ValueError(f"unsupported compile dynamic value: {value}")


def _compile_config_report(config: Any) -> dict[str, Any] | None:
  if config is None:
    return None
  return {
    "fullgraph": bool(config.fullgraph),
    "dynamic": config.dynamic,
    "backend": config.backend,
    "mode": config.mode,
    "options": config.options,
  }


def _patch_static_cache_mixed_kv_dtype() -> bool:
  import torch
  from transformers import cache_utils

  static_layer = cache_utils.StaticLayer
  if getattr(static_layer, "_openpilot_mixed_kv_dtype_patch", False):
    return False

  def lazy_initialization(self: Any, key_states: Any, value_states: Any) -> None:
    self.dtype, self.device = key_states.dtype, key_states.device
    self.value_dtype = value_states.dtype
    self.max_batch_size, self.num_heads = key_states.shape[:2]
    self.v_head_dim = value_states.shape[-1]
    self.k_head_dim = key_states.shape[-1]
    self.keys = torch.zeros(
      (self.max_batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
      dtype=self.dtype,
      device=self.device,
    )
    self.values = torch.zeros(
      (self.max_batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
      dtype=self.value_dtype,
      device=self.device,
    )
    self.cumulative_length = self.cumulative_length.to(self.device)
    if not cache_utils.is_torchdynamo_compiling():
      torch._dynamo.mark_static_address(self.keys)
      torch._dynamo.mark_static_address(self.values)
      torch._dynamo.mark_static_address(self.cumulative_length)
    self.is_initialized = True

  static_layer.lazy_initialization = lazy_initialization
  static_layer._openpilot_mixed_kv_dtype_patch = True
  return True


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark local Alpamayo target VLM generation with and without local DFlash draft.")
  parser.add_argument("--alpamayo-root", type=Path, default=Path("/mnt/g/alpamayo1.5"))
  parser.add_argument("--target-model", type=Path, default=None)
  parser.add_argument("--model-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="bfloat16")
  parser.add_argument("--autocast-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="bfloat16")
  parser.add_argument("--paro-native", action="store_true")
  parser.add_argument("--paro-compute-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="float16")
  parser.add_argument("--paro-output-dtype", choices=("input", "compute", "native"), default="native")
  parser.add_argument("--paro-marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--draft-model", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
  parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
  parser.add_argument("--prompt-source", choices=("dataset", "synthetic", "contract"), default="dataset")
  parser.add_argument("--contract-request-json", type=Path, help="Captured strict Alpamayo contract request JSON for --prompt-source=contract.")
  parser.add_argument("--synthetic-width", type=int, default=512)
  parser.add_argument("--synthetic-height", type=int, default=384)
  parser.add_argument("--camera-mode", choices=("full", "front2", "front1"), default="front2")
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--min-pixels", type=int, default=65536)
  parser.add_argument("--max-pixels", type=int, default=65536)
  parser.add_argument("--processor-model", default=os.environ.get("ALPAMAYO_PROCESSOR_MODEL"))
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--device-map-mode", choices=("auto", "current_split"), default="current_split")
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="flash_attention_2")
  parser.add_argument("--expert-attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
  parser.add_argument("--draft-attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
  parser.add_argument("--draft-device", default="lm_head", help="Device for DFlash draft weights. Use 'lm_head' or a torch device such as cuda:0.")
  parser.add_argument("--quantization", choices=("none", "bnb_4bit_nf4", "bnb_4bit_fp4"), default="none")
  parser.add_argument("--bnb-compute-dtype", choices=("float16", "bfloat16"), default="bfloat16")
  parser.add_argument("--repeat", type=int, default=2)
  parser.add_argument("--max-generation-length", type=int, default=32)
  parser.add_argument("--greedy", action="store_true")
  parser.add_argument("--standard-mode", choices=("hf", "manual"), default="hf")
  parser.add_argument("--hf-cache-implementation", choices=("dynamic", "dynamic_full", "static", "quantized"), default=None)
  parser.add_argument("--hf-compile", action="store_true", help="Enable Transformers decode compilation through CompileConfig.")
  parser.add_argument("--hf-compile-fullgraph", action="store_true")
  parser.add_argument("--hf-compile-dynamic", choices=("none", "true", "false"), default="none")
  parser.add_argument("--hf-compile-backend", default="inductor")
  parser.add_argument("--hf-compile-mode", default="reduce-overhead")
  parser.add_argument("--float32-matmul-precision", choices=("highest", "high", "medium"), default=None)
  parser.add_argument("--skip-standard", action="store_true")
  parser.add_argument("--skip-dflash", action="store_true")
  args = parser.parse_args()

  if args.target_model is None:
    if args.paro_native:
      args.target_model = DEFAULT_PARO_TARGET_MODEL
    else:
      parser.error("--target-model is required unless --paro-native is set")
  if args.standard_mode == "manual" and not args.greedy:
    parser.error("--standard-mode=manual requires --greedy")
  if args.standard_mode != "hf" and (args.hf_cache_implementation is not None or args.hf_compile):
    parser.error("HF cache/compile flags require --standard-mode=hf")

  _insert_alpamayo_paths(args.alpamayo_root)
  if args.standard_mode == "manual":
    _patch_manual_greedy_generation()

  import torch
  if args.float32_matmul_precision is not None:
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

  report: dict[str, Any] = {
    "status": "started",
    "created_at_unix": time.time(),
    "cwd": os.getcwd(),
    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "torch": torch.__version__,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_initial": _gpu_stats(torch),
    "standard_generation_controls": {
      "cache_implementation": args.hf_cache_implementation,
      "compile_enabled": bool(args.hf_compile),
      "compile_fullgraph": bool(args.hf_compile_fullgraph),
      "compile_dynamic": _compile_dynamic(args.hf_compile_dynamic),
      "compile_backend": args.hf_compile_backend,
      "compile_mode": args.hf_compile_mode,
      "disable_compile_effective_for_hf": not args.hf_compile,
      "static_cache_mixed_kv_dtype_patch": args.hf_cache_implementation == "static",
    },
  }
  _progress("start", gpu_initial=report["gpu_initial"])

  load_start = time.perf_counter()
  _progress("target_load_start", target_model=str(args.target_model))
  model = _build_target_model(args, torch).eval()
  _sync(torch)
  report["target_load_ms"] = round((time.perf_counter() - load_start) * 1000.0, 3)
  report["hf_device_map"] = {str(k): int(v) for k, v in getattr(model, "hf_device_map", {}).items()}
  if args.paro_native:
    report["paro_target_model"] = str(args.target_model)
    report["paro_replacement_records"] = len(getattr(model, "_openpilot_paro_replacement_records", []))
    report["paro_visual_patch_embed_repair"] = getattr(model, "_openpilot_paro_visual_patch_embed_repair", None)
    report["paro_finalize"] = getattr(model, "_openpilot_paro_finalize", None)
  report["gpu_post_target_load"] = _gpu_stats(torch)
  _progress(
    "target_load_done",
    target_load_ms=report["target_load_ms"],
    hf_device_map=report["hf_device_map"],
    gpu=report["gpu_post_target_load"],
  )

  draft = None
  if not args.skip_dflash:
    draft_start = time.perf_counter()
    _progress("draft_load_start", draft_model=str(args.draft_model))
    draft, draft_load_mode = _load_dflash_model(
      args.draft_model,
      torch_mod=torch,
      dtype=torch.bfloat16,
      attn_implementation=args.draft_attn_implementation,
    )
    draft_device = _module_device(model.vlm.lm_head) if args.draft_device == "lm_head" else torch.device(args.draft_device)
    draft = draft.to(draft_device).eval()
    _sync(torch)
    report["draft_load_ms"] = round((time.perf_counter() - draft_start) * 1000.0, 3)
    report["draft_load_mode"] = draft_load_mode
    report["draft_device"] = str(draft_device)
    mask_embedding = _load_mask_embedding(torch, args.draft_model, draft_device, torch.bfloat16)
    report["mask_embedding_shape"] = [int(x) for x in mask_embedding.shape]
    report["mask_embedding_device"] = str(mask_embedding.device)
    report["gpu_post_draft_load"] = _gpu_stats(torch)
    _progress(
      "draft_load_done",
      draft_load_ms=report["draft_load_ms"],
      draft_load_mode=draft_load_mode,
      draft_device=str(draft_device),
      mask_embedding_shape=report["mask_embedding_shape"],
      gpu=report["gpu_post_draft_load"],
    )
  else:
    mask_embedding = None

  prep_start = time.perf_counter()
  _progress("prompt_prepare_start")
  prompt_meta, tokenized_data, input_ids = _prepare_prompt(args, model, torch)
  _sync(torch)
  report["prompt_prepare_ms"] = round((time.perf_counter() - prep_start) * 1000.0, 3)
  report["prompt"] = prompt_meta
  report["gpu_pre_benchmark"] = _gpu_stats(torch)
  _progress("prompt_prepare_done", prompt_prepare_ms=report["prompt_prepare_ms"], prompt=prompt_meta, gpu=report["gpu_pre_benchmark"])

  standard_times: list[float] = []
  dflash_times: list[float] = []
  standard_reports: list[dict[str, Any]] = []
  dflash_reports: list[dict[str, Any]] = []

  with torch.inference_mode(), torch.autocast("cuda", dtype=_torch_dtype(torch, args.autocast_dtype)):
    for repeat_idx in range(args.repeat):
      torch.cuda.manual_seed_all(42 + repeat_idx)
      if not args.skip_standard:
        _sync(torch)
        _progress("standard_generate_start", repeat=repeat_idx)
        start = time.perf_counter()
        out = _standard_generate_once(
          model,
          tokenized_data,
          input_ids,
          max_generation_length=args.max_generation_length,
          greedy=args.greedy,
          mode=args.standard_mode,
          hf_cache_implementation=args.hf_cache_implementation,
          hf_compile=args.hf_compile,
          hf_compile_fullgraph=args.hf_compile_fullgraph,
          hf_compile_dynamic=args.hf_compile_dynamic,
          hf_compile_backend=args.hf_compile_backend,
          hf_compile_mode=args.hf_compile_mode,
        )
        _sync(torch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        standard_times.append(elapsed_ms)
        _progress(
          "standard_generate_done",
          repeat=repeat_idx,
          elapsed_ms=round(elapsed_ms, 3),
          new_tokens=int(out.sequences.shape[1] - input_ids.shape[1]),
        )
        standard_reports.append({
          "repeat": repeat_idx,
          "mode": args.standard_mode,
          "elapsed_ms": round(elapsed_ms, 3),
          "sequence_len": int(out.sequences.shape[1]),
          "new_tokens": int(out.sequences.shape[1] - input_ids.shape[1]),
          "runtime_profile": getattr(out, "runtime_profile", None),
          "new_text_preview": _decode_new(model, out.sequences, input_ids.shape[1])[:500],
        })

      if draft is not None:
        _sync(torch)
        _progress("dflash_generate_start", repeat=repeat_idx)
        start = time.perf_counter()
        out = _dflash_generate_once(
          torch,
          model,
          draft,
          tokenized_data,
          input_ids,
          mask_embedding=mask_embedding,
          max_generation_length=args.max_generation_length,
          temperature=0.0 if args.greedy else 0.6,
        )
        _sync(torch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        dflash_times.append(elapsed_ms)
        _progress(
          "dflash_generate_done",
          repeat=repeat_idx,
          elapsed_ms=round(elapsed_ms, 3),
          new_tokens=out.new_tokens,
          mean_acceptance=round(mean(out.acceptance_lengths), 3) if out.acceptance_lengths else 0.0,
        )
        dflash_reports.append({
          "repeat": repeat_idx,
          "elapsed_ms": round(elapsed_ms, 3),
          "new_tokens": out.new_tokens,
          "time_to_first_token_ms": round(out.time_to_first_token_ms, 3),
          "decode_ms": round(out.decode_ms, 3),
          "draft_ms": round(out.draft_ms, 3),
          "validate_ms": round(out.validate_ms, 3),
          "acceptance_lengths": [int(x) for x in out.acceptance_lengths],
          "mean_acceptance": round(mean(out.acceptance_lengths), 3) if out.acceptance_lengths else 0.0,
          "new_text_preview": _decode_new(model, out.sequences, input_ids.shape[1])[:500],
        })

  report["status"] = "ok"
  report["standard"] = {
    "summary": _summary(standard_times),
    "runs": standard_reports,
  }
  report["dflash"] = {
    "summary": _summary(dflash_times),
    "runs": dflash_reports,
  }
  report["gpu_post_benchmark"] = _gpu_stats(torch)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  _progress("done", output=str(args.output), standard=report["standard"]["summary"], dflash=report["dflash"]["summary"])
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
