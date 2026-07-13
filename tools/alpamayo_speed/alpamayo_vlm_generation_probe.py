from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000
DEFAULT_TARGET_MODEL = Path("/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-10B-finetuned")
_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _insert_alpamayo_paths(alpamayo_root: Path) -> None:
  sys.path.insert(0, str(alpamayo_root))
  sys.path.insert(0, str(alpamayo_root / "src"))
  repo_root = Path(__file__).resolve().parents[2]
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _normalize_path(raw: str | Path) -> Path:
  value = str(raw)
  match = _WIN_DRIVE_RE.match(value)
  if match is None or os.name != "posix":
    return Path(os.path.expanduser(value))
  drive = match.group(1).lower()
  rest = match.group(2).replace("\\", "/")
  return Path(f"/mnt/{drive}/{rest}")


def _sync(torch_mod: Any) -> None:
  if torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


@contextmanager
def _cuda_timer(torch_mod: Any):
  _sync(torch_mod)
  start = time.perf_counter()
  try:
    yield lambda: time.perf_counter() - start
  finally:
    _sync(torch_mod)


def _gpu_stats(torch_mod: Any) -> dict[int, dict[str, float]]:
  stats: dict[int, dict[str, float]] = {}
  for idx in range(torch_mod.cuda.device_count()):
    free_bytes, total_bytes = torch_mod.cuda.mem_get_info(idx)
    stats[idx] = {
      "used_gib": round((total_bytes - free_bytes) / (1024 ** 3), 3),
      "free_gib": round(free_bytes / (1024 ** 3), 3),
      "total_gib": round(total_bytes / (1024 ** 3), 3),
    }
  return stats


def _summarize_ms(values: list[float]) -> dict[str, float | int]:
  if not values:
    return {"count": 0}
  ordered = sorted(values)

  def pct(q: float) -> float:
    if len(ordered) == 1:
      return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

  return {
    "count": len(values),
    "sum_ms": round(sum(values) * 1000.0, 3),
    "mean_ms": round((sum(values) / len(values)) * 1000.0, 3),
    "min_ms": round(min(values) * 1000.0, 3),
    "p50_ms": round(pct(0.50) * 1000.0, 3),
    "p95_ms": round(pct(0.95) * 1000.0, 3),
    "p99_ms": round(pct(0.99) * 1000.0, 3),
    "max_ms": round(max(values) * 1000.0, 3),
  }


def _first_tensor_device(module: Any) -> str:
  try:
    return str(next(module.parameters()).device)
  except StopIteration:
    return "<no-params>"


def _cache_seq_len(past_key_values: Any) -> int | None:
  if past_key_values is None:
    return None
  if hasattr(past_key_values, "get_seq_length"):
    try:
      return int(past_key_values.get_seq_length())
    except Exception:
      return None
  return None


def _shape_of(value: Any) -> list[int] | None:
  shape = getattr(value, "shape", None)
  if shape is None:
    return None
  return [int(item) for item in shape]


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


def _build_vlm_gpu0_expert_gpu1_device_map() -> dict[str, int]:
  device_map: dict[str, int] = {
    "vlm.model.visual": 0,
    "vlm.model.language_model.embed_tokens": 0,
    "vlm.model.language_model.norm": 0,
    "vlm.model.language_model.rotary_emb": 0,
    "vlm.lm_head": 0,
    "expert.norm": 1,
    "expert.rotary_emb": 1,
    "action_in_proj": 1,
    "action_out_proj": 1,
    "action_space": 1,
    "diffusion": 1,
  }
  for layer_idx in range(36):
    device_map[f"vlm.model.language_model.layers.{layer_idx}"] = 0
    device_map[f"expert.layers.{layer_idx}"] = 1
  return device_map


def _build_vlm_gpu1_expert_gpu0_device_map() -> dict[str, int]:
  device_map: dict[str, int] = {
    "vlm.model.visual": 1,
    "vlm.model.language_model.embed_tokens": 1,
    "vlm.model.language_model.norm": 1,
    "vlm.model.language_model.rotary_emb": 1,
    "vlm.lm_head": 1,
    "expert.norm": 0,
    "expert.rotary_emb": 0,
    "action_in_proj": 0,
    "action_out_proj": 0,
    "action_space": 0,
    "diffusion": 0,
  }
  for layer_idx in range(36):
    device_map[f"vlm.model.language_model.layers.{layer_idx}"] = 1
    device_map[f"expert.layers.{layer_idx}"] = 0
  return device_map


def _build_model(args: argparse.Namespace, torch_mod: Any) -> Any:
  from transformers import BitsAndBytesConfig
  from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

  if args.device_map_mode == "auto":
    device_map: str | dict[str, int] = "auto"
  elif args.device_map_mode == "current_split":
    device_map = _build_current_split_device_map(args.split_index)
  elif args.device_map_mode == "vlm_gpu0_expert_gpu1":
    device_map = _build_vlm_gpu0_expert_gpu1_device_map()
  elif args.device_map_mode == "vlm_gpu1_expert_gpu0":
    device_map = _build_vlm_gpu1_expert_gpu0_device_map()
  else:
    raise ValueError(f"unsupported device map mode: {args.device_map_mode}")

  load_kwargs: dict[str, Any] = {
    "dtype": torch_mod.bfloat16,
    "attn_implementation": args.attn_implementation,
    "device_map": device_map,
    "max_memory": {idx: f"{args.gpu_mem_gib}GiB" for idx in range(torch_mod.cuda.device_count())},
    "low_cpu_mem_usage": True,
  }
  if args.quantization != "none":
    compute_dtype = getattr(torch_mod, args.bnb_compute_dtype)
    if args.quantization == "bnb_8bit":
      load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["action_in_proj", "action_out_proj"],
      )
    elif args.quantization in ("bnb_4bit_nf4", "bnb_4bit_fp4"):
      load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4" if args.quantization == "bnb_4bit_nf4" else "fp4",
        bnb_4bit_compute_dtype=compute_dtype,
      )
    else:
      raise ValueError(f"unsupported quantization: {args.quantization}")

  from benchmark_dflash_alpamayo_generation import _patch_flattened_conv3d_loader

  restore_loader = _patch_flattened_conv3d_loader(torch_mod)
  try:
    model = Alpamayo1_5.from_pretrained(str(_normalize_path(args.target_model)), **load_kwargs)
  finally:
    restore_loader()
  _assert_no_cpu_offload(model)
  return model


def _assert_no_cpu_offload(model: Any) -> None:
  device_map = getattr(model, "hf_device_map", {}) or {}
  bad = {
    name: device
    for name, device in device_map.items()
    if str(device).lower() in ("cpu", "disk") or str(device).lower().startswith("cpu")
  }
  if bad:
    preview = dict(list(bad.items())[:12])
    raise RuntimeError(f"CPU/disk offload is forbidden for Alpamayo latency probes; offending entries={preview}")


def _wrap_forward(
  owner: Any,
  attr_name: str,
  torch_mod: Any,
  rows: list[dict[str, Any]],
  name: str,
) -> Callable[[], None]:
  original = getattr(owner, attr_name)

  def wrapped(*call_args: Any, **kwargs: Any) -> Any:
    row = {
      "name": name,
      "call_index": len(rows),
      "module_device": _first_tensor_device(owner),
      "input_ids_shape": _shape_of(kwargs.get("input_ids")),
      "inputs_embeds_shape": _shape_of(kwargs.get("inputs_embeds")),
      "attention_mask_shape": _shape_of(kwargs.get("attention_mask")),
      "position_ids_shape": _shape_of(kwargs.get("position_ids")),
      "cache_position_shape": _shape_of(kwargs.get("cache_position")),
      "past_seq_len_before": _cache_seq_len(kwargs.get("past_key_values")),
      "logits_to_keep": kwargs.get("logits_to_keep"),
    }
    _sync(torch_mod)
    start = time.perf_counter()
    out = original(*call_args, **kwargs)
    _sync(torch_mod)
    row["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    row["past_seq_len_after"] = _cache_seq_len(getattr(out, "past_key_values", None))
    rows.append(row)
    return out

  setattr(owner, attr_name, wrapped)

  def restore() -> None:
    setattr(owner, attr_name, original)

  return restore


def _disable_deepstack(language_model: Any) -> Callable[[], None]:
  original = language_model._deepstack_process

  def disabled(hidden_states: Any, visual_pos_masks: Any, visual_embeds: Any) -> Any:
    return hidden_states

  language_model._deepstack_process = disabled

  def restore() -> None:
    language_model._deepstack_process = original

  return restore


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
  values: dict[str, Any] = {}
  for key, value in vars(args).items():
    values[key] = str(value) if isinstance(value, Path) else value
  return values


def _export_visual_prefix(
  export_dir: Path,
  model: Any,
  input_ids: Any,
  tokenized_data: dict[str, Any],
  torch_mod: Any,
) -> dict[str, Any]:
  export_dir.mkdir(parents=True, exist_ok=True)

  pixel_values = tokenized_data.get("pixel_values")
  image_grid_thw = tokenized_data.get("image_grid_thw")
  if pixel_values is None or image_grid_thw is None:
    raise ValueError("visual-prefix export currently expects image pixel_values and image_grid_thw")

  with torch_mod.inference_mode(), torch_mod.autocast("cuda", dtype=torch_mod.bfloat16):
    image_embeds_split, deepstack_image_embeds = model.vlm.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch_mod.cat(image_embeds_split, dim=0).to(dtype=torch_mod.float16).contiguous()

  image_token_id = int(model.vlm.config.image_token_id)
  image_mask = input_ids[0] == image_token_id
  image_token_count = int(image_mask.sum().detach().cpu().item())
  if image_token_count != int(image_embeds.shape[0]):
    raise ValueError(f"image token count {image_token_count} != image embeds {tuple(image_embeds.shape)}")

  vocab_size = int(model.vlm.config.text_config.vocab_size)
  trt_input_ids = input_ids.detach().clone().to(dtype=torch_mod.int32)
  fake_ids = torch_mod.arange(
    vocab_size,
    vocab_size + image_token_count,
    device=trt_input_ids.device,
    dtype=trt_input_ids.dtype,
  )
  trt_input_ids[0, image_mask] = fake_ids

  prompt_table = image_embeds.unsqueeze(0).detach().cpu().numpy()
  trt_ids_np = trt_input_ids[0].detach().cpu().numpy().astype(np.int32, copy=False)
  original_ids_np = input_ids[0].detach().cpu().numpy().astype(np.int32, copy=False)

  prompt_table_path = export_dir / "prompt_table_image_embeds_fp16.npy"
  trt_input_ids_path = export_dir / "trt_input_ids.npy"
  original_input_ids_path = export_dir / "original_input_ids.npy"
  mrope_position_ids_path = export_dir / "mrope_position_ids.npy"
  mrope_position_deltas_path = export_dir / "mrope_position_deltas.npy"
  np.save(prompt_table_path, prompt_table)
  np.save(trt_input_ids_path, trt_ids_np)
  np.save(original_input_ids_path, original_ids_np)

  attention_mask = tokenized_data.get("attention_mask")
  with torch_mod.inference_mode():
    position_ids, rope_deltas = model.vlm.model.get_rope_index(
      input_ids,
      image_grid_thw,
      None,
      attention_mask=attention_mask,
    )
  position_ids_np = position_ids.detach().cpu().numpy().astype(np.int32, copy=False)
  rope_deltas_np = rope_deltas.detach().cpu().numpy().astype(np.int32, copy=False)
  np.save(mrope_position_ids_path, position_ids_np)
  np.save(mrope_position_deltas_path, rope_deltas_np)

  meta = {
    "prompt_table_path": str(prompt_table_path),
    "trt_input_ids_path": str(trt_input_ids_path),
    "original_input_ids_path": str(original_input_ids_path),
    "mrope_position_ids_path": str(mrope_position_ids_path),
    "mrope_position_deltas_path": str(mrope_position_deltas_path),
    "prompt_table_shape": list(prompt_table.shape),
    "trt_input_ids_shape": list(trt_ids_np.shape),
    "original_input_ids_shape": list(original_ids_np.shape),
    "mrope_position_ids_shape": list(position_ids_np.shape),
    "mrope_position_deltas_shape": list(rope_deltas_np.shape),
    "mrope_position_deltas": rope_deltas_np.tolist(),
    "mrope_position_ids_min": int(position_ids_np.min()),
    "mrope_position_ids_max": int(position_ids_np.max()),
    "image_token_id": image_token_id,
    "image_token_count": image_token_count,
    "fake_prompt_id_start": vocab_size,
    "fake_prompt_id_end_exclusive": vocab_size + image_token_count,
    "image_grid_thw": image_grid_thw.detach().cpu().tolist(),
    "deepstack_visual_embeds_shapes": [list(item.shape) for item in deepstack_image_embeds],
    "semantic_caveat": (
      "This exports base image token embeddings for TensorRT prompt tuning. "
      "Qwen3-VL also injects deepstack_visual_embeds into early text layers; "
      "the current classic TensorRT text engine does not consume those tensors."
    ),
  }
  (export_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
  return meta


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--alpamayo-root", type=Path, default=Path(r"G:\alpamayo1.5"))
  parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
  parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
  parser.add_argument("--prompt-source", choices=["dataset", "contract"], default="dataset")
  parser.add_argument("--contract-request-json", type=Path)
  parser.add_argument("--processor-model", default=os.environ.get("ALPAMAYO_PROCESSOR_MODEL"))
  parser.add_argument("--camera-mode", choices=["full", "front2", "front1"], default="front2")
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--min-pixels", type=int, default=65536)
  parser.add_argument("--max-pixels", type=int, default=65536)
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--device-map-mode", choices=["auto", "current_split", "vlm_gpu0_expert_gpu1", "vlm_gpu1_expert_gpu0"], default="current_split")
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="flash_attention_2")
  parser.add_argument("--quantization", choices=["none", "bnb_8bit", "bnb_4bit_nf4", "bnb_4bit_fp4"], default="none")
  parser.add_argument("--bnb-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
  parser.add_argument("--repeat", type=int, default=2)
  parser.add_argument("--max-generation-length", type=int, default=256)
  parser.add_argument("--greedy", action="store_true")
  parser.add_argument("--manual-generate", action="store_true", help="Use Alpamayo's explicit greedy loop with logits_to_keep=1 instead of HF generate().")
  parser.add_argument("--export-visual-prefix-dir", type=Path)
  parser.add_argument("--export-only", action="store_true")
  parser.add_argument("--disable-deepstack", action="store_true", help="Diagnostic: suppress Qwen3-VL DeepStack injection while keeping base visual token embeddings.")
  args = parser.parse_args()

  _insert_alpamayo_paths(args.alpamayo_root)

  import physical_ai_av
  import torch
  from transformers import LogitsProcessorList, StoppingCriteriaList
  from alpamayo1_5 import helper
  from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
  from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
  from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token
  from timed_inference import select_camera_features

  report: dict[str, Any] = {
    "created_at_unix": time.time(),
    "cwd": os.getcwd(),
    "args": _jsonable_args(args),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_initial": _gpu_stats(torch),
  }

  if args.prompt_source == "contract":
    from benchmark_dflash_alpamayo_generation import _load_contract_prompt

    image_frames, camera_indices, ego_history_xyz, ego_history_rot, source_meta = _load_contract_prompt(args, torch)
    num_frames_per_camera = int(source_meta["frames_per_camera"])
    nav_text = source_meta.get("nav_text")
    report["prompt_source_meta"] = source_meta
  else:
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
    num_frames_per_camera = args.num_frames
    nav_text = None

  report["image_frames_shape"] = tuple(image_frames.shape)
  report["camera_indices"] = camera_indices.tolist()

  messages = helper.create_message(
    frames=image_frames.flatten(0, 1),
    camera_indices=camera_indices,
    num_frames_per_camera=num_frames_per_camera,
    nav_text=nav_text,
  )

  with _cuda_timer(torch) as elapsed:
    model = _build_model(args, torch)
  report["model_load_ms"] = round(elapsed() * 1000.0, 3)
  report["hf_device_map"] = {str(k): int(v) for k, v in getattr(model, "hf_device_map", {}).items()}
  report["gpu_post_load"] = _gpu_stats(torch)

  helper.MIN_PIXELS = args.min_pixels
  helper.MAX_PIXELS = args.max_pixels
  if args.processor_model:
    helper.BASE_PROCESSOR_NAME = str(_normalize_path(args.processor_model))
  processor = helper.get_processor(model.tokenizer)
  with _cuda_timer(torch) as elapsed:
    inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=False,
      continue_final_message=True,
      return_dict=True,
      return_tensors="pt",
    )
  report["processor_ms"] = round(elapsed() * 1000.0, 3)
  report["input_ids_shape"] = tuple(inputs["input_ids"].shape)
  if "image_grid_thw" in inputs:
    image_grid = inputs["image_grid_thw"].detach().cpu()
    merge_len = int(processor.image_processor.merge_size) ** 2
    report["image_grid_thw"] = image_grid.tolist()
    report["image_tokens"] = int((image_grid.prod(dim=1) // merge_len).sum().item())

  tokenized_data = helper.to_device(inputs, model.device)
  traj_data_vlm = {
    "ego_history_xyz": helper.to_device(ego_history_xyz, model.device),
    "ego_history_rot": helper.to_device(ego_history_rot, model.device),
  }
  input_ids = model.fuse_traj_tokens(tokenized_data.pop("input_ids"), traj_data_vlm)
  if args.export_visual_prefix_dir is not None:
    with _cuda_timer(torch) as elapsed:
      report["visual_prefix_export"] = _export_visual_prefix(
        args.export_visual_prefix_dir,
        model,
        input_ids,
        tokenized_data,
        torch,
      )
    report["visual_prefix_export_ms"] = round(elapsed() * 1000.0, 3)
    if args.export_only:
      report["status"] = "ok"
      report["gpu_post_export"] = _gpu_stats(torch)
      _write_json(args.output, report)
      print(json.dumps(report, indent=2, sort_keys=True), flush=True)
      return

  eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
  generation_config = model.vlm.generation_config
  generation_config.top_p = 0.98
  generation_config.temperature = 0.6
  generation_config.do_sample = not args.greedy
  generation_config.num_return_sequences = 1
  generation_config.max_new_tokens = args.max_generation_length
  generation_config.output_logits = False
  generation_config.return_dict_in_generate = True
  generation_config.pad_token_id = model.tokenizer.pad_token_id
  logits_processor = LogitsProcessorList([
    ExpertLogitsProcessor(
      traj_token_offset=model.config.traj_token_start_idx,
      traj_vocab_size=model.config.traj_vocab_size,
    )
  ])

  rows: list[dict[str, Any]] = []
  infer_reports: list[dict[str, Any]] = []
  deepstack_restore = _disable_deepstack(model.vlm.model.language_model) if args.disable_deepstack else None
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    try:
      for repeat_idx in range(args.repeat):
        torch.cuda.manual_seed_all(42 + repeat_idx)
        forward_rows: list[dict[str, Any]] = []
        restores = [
          _wrap_forward(model.vlm.model.language_model, "forward", torch, forward_rows, "language_model"),
          _wrap_forward(model.vlm.model.visual, "forward", torch, forward_rows, "visual"),
        ]
        try:
          _sync(torch)
          start = time.perf_counter()
          if args.manual_generate:
            if not args.greedy:
              raise ValueError("--manual-generate requires --greedy")
            sequences, _prompt_cache = model._manual_greedy_vlm_generate(
              input_ids=input_ids,
              tokenized_data=tokenized_data,
              eos_token_id=eos_token_id,
              max_generation_length=args.max_generation_length,
            )
          else:
            outputs = model.vlm.generate(
              input_ids=input_ids,
              generation_config=generation_config,
              stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)]),
              logits_processor=logits_processor,
              **tokenized_data,
            )
            sequences = outputs.sequences
          _sync(torch)
          elapsed_s = time.perf_counter() - start
        finally:
          for restore in reversed(restores):
            restore()

        raw_sequences = sequences.detach().clone()
        sequences = replace_padding_after_eos(
          token_ids=raw_sequences,
          eos_token_id=eos_token_id,
          pad_token_id=model.tokenizer.pad_token_id,
        ) if not args.manual_generate else sequences
        rows.extend(forward_rows)
        language_ms = [row["elapsed_ms"] / 1000.0 for row in forward_rows if row["name"] == "language_model"]
        visual_ms = [row["elapsed_ms"] / 1000.0 for row in forward_rows if row["name"] == "visual"]
        raw_new_token_ids = raw_sequences[0, input_ids.shape[1]:].detach().cpu().tolist()
        new_token_ids = sequences[0, input_ids.shape[1]:].detach().cpu().tolist()
        infer_reports.append({
          "repeat_index": repeat_idx,
          "elapsed_ms": round(elapsed_s * 1000.0, 3),
          "generated_sequence_length": int(sequences.shape[1]),
          "new_tokens": int(sequences.shape[1] - input_ids.shape[1]),
          "raw_generated_new_token_ids": [int(token_id) for token_id in raw_new_token_ids],
          "raw_generated_new_text": model.tokenizer.decode(raw_new_token_ids, skip_special_tokens=False),
          "generated_new_token_ids": [int(token_id) for token_id in new_token_ids],
          "generated_new_text": model.tokenizer.decode(new_token_ids, skip_special_tokens=False),
          "generated_full_text": model.tokenizer.decode(sequences[0].detach().cpu().tolist(), skip_special_tokens=False),
          "language_summary": _summarize_ms(language_ms),
          "visual_summary": _summarize_ms(visual_ms),
        })
    finally:
      if deepstack_restore is not None:
        deepstack_restore()

  language_ms_all = [row["elapsed_ms"] / 1000.0 for row in rows if row["name"] == "language_model"]
  visual_ms_all = [row["elapsed_ms"] / 1000.0 for row in rows if row["name"] == "visual"]
  report.update({
    "status": "ok",
    "inferences": infer_reports,
    "forward_rows": rows,
    "language_summary_all": _summarize_ms(language_ms_all),
    "visual_summary_all": _summarize_ms(visual_ms_all),
    "gpu_post_infer": _gpu_stats(torch),
  })
  _write_json(args.output, report)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
