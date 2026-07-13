#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000


def _insert_local_tool_path() -> None:
  sys.path.insert(0, str(Path(__file__).resolve().parent))


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _extract_generated_tokens(report: dict[str, Any], source_kind: str) -> list[int]:
  if source_kind == "pytorch_vlm_probe":
    inferences = report.get("inferences") or []
    if not inferences:
      raise ValueError("PyTorch VLM probe report has no inferences")
    tokens = inferences[0].get("generated_new_token_ids") or []
  elif source_kind == "trtllm_mrope":
    outputs = report.get("outputs") or []
    if not outputs:
      raise ValueError("TensorRT-LLM report has no outputs")
    measure_outputs = [row for row in outputs if row.get("phase") == "measure"]
    tokens = (measure_outputs[0] if measure_outputs else outputs[0]).get("generated_token_ids") or []
  elif source_kind == "parity":
    trt = report.get("tensorrt_llm") or {}
    tokens = trt.get("generated_token_ids") or []
  else:
    raise ValueError(f"unsupported source kind: {source_kind}")
  return [int(token) for token in tokens]


def _truncate_tokens(tokens: list[int], stop_token_id: int | None, include_after_stop: int) -> list[int]:
  if stop_token_id is None:
    return tokens
  try:
    stop_index = tokens.index(stop_token_id)
  except ValueError:
    return tokens
  end = min(len(tokens), stop_index + 1 + max(include_after_stop, 0))
  return tokens[:end]


def _sync(torch_mod: Any) -> None:
  if torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


def _timed_ms(torch_mod: Any, fn: Any) -> tuple[Any, float]:
  _sync(torch_mod)
  start = time.perf_counter()
  result = fn()
  _sync(torch_mod)
  return result, (time.perf_counter() - start) * 1000.0


def _prepare_prompt_with_data(args: argparse.Namespace, model: Any, torch_mod: Any) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
  from alpamayo1_5 import helper

  nav_text = None
  source_meta: dict[str, Any] = {}
  num_frames_per_camera = args.num_frames
  if args.prompt_source == "contract":
    from benchmark_dflash_alpamayo_generation import _load_contract_prompt

    image_frames, camera_indices, ego_history_xyz, ego_history_rot, source_meta = _load_contract_prompt(args, torch_mod)
    num_frames_per_camera = int(source_meta["frames_per_camera"])
    nav_text = source_meta.get("nav_text")
    raw_data = {
      "image_frames": image_frames,
      "camera_indices": camera_indices,
      "ego_history_xyz": ego_history_xyz,
      "ego_history_rot": ego_history_rot,
    }
  else:
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from timed_inference import select_camera_features

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    camera_features = select_camera_features(avdi, args.camera_mode)
    raw_data = load_physical_aiavdataset(
      args.clip_id,
      t0_us=args.t0_us,
      avdi=avdi,
      camera_features=camera_features,
      num_frames=args.num_frames,
    )
    image_frames = raw_data["image_frames"]
    camera_indices = raw_data["camera_indices"]
    ego_history_xyz = raw_data["ego_history_xyz"]
    ego_history_rot = raw_data["ego_history_rot"]

  messages = helper.create_message(
    frames=image_frames.flatten(0, 1),
    camera_indices=camera_indices,
    num_frames_per_camera=num_frames_per_camera,
    nav_text=nav_text,
  )
  helper.MIN_PIXELS = args.min_pixels
  helper.MAX_PIXELS = args.max_pixels
  if args.processor_model:
    from benchmark_dflash_alpamayo_generation import _normalize_path

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
  return prompt_meta, tokenized_data, input_ids, traj_data_vlm, raw_data


def _rebuild_vlm_cache(torch_mod: Any, model: Any, tokenized_data: dict[str, Any], full_ids: Any) -> Any:
  forward_kwargs = copy.copy(tokenized_data)
  forward_kwargs["attention_mask"] = torch_mod.ones_like(full_ids, dtype=torch_mod.long, device=full_ids.device)
  return model.vlm(
    input_ids=full_ids,
    use_cache=True,
    return_dict=True,
    logits_to_keep=1,
    **forward_kwargs,
  )


def _run_action_from_cache(
  torch_mod: Any,
  model: Any,
  input_ids: Any,
  tokenized_data: dict[str, Any],
  traj_data_vlm: dict[str, Any],
  generated_sequences: Any,
  prompt_cache: Any,
  rope_deltas: Any,
  *,
  diffusion_steps: int,
  num_traj_samples: int,
  num_traj_sets: int,
) -> tuple[Any, Any, dict[str, float | int]]:
  import einops

  runtime: dict[str, float | int] = {
    "expert_step_calls": 0,
    "expert_step_seconds": 0.0,
  }

  ego_history_xyz = traj_data_vlm["ego_history_xyz"]
  ego_history_rot = traj_data_vlm["ego_history_rot"]
  B, n_traj_group, _, _ = ego_history_xyz.shape
  assert n_traj_group == 1
  n_samples_total = num_traj_samples * num_traj_sets
  device = input_ids.device

  eos_token_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
  b_star = generated_sequences.shape[0]
  n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
  prefill_seq_len = int(prompt_cache.get_seq_length())
  offset = model._find_eos_offset(
    sequences=generated_sequences,
    eos_token_id=eos_token_id,
    device=device,
  )
  prefix_mask = tokenized_data.get("attention_mask")
  if prefix_mask is not None:
    prefix_mask = torch_mod.repeat_interleave(prefix_mask, n_samples_total, dim=0)
  position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
    offset=offset,
    rope_deltas=rope_deltas,
    kv_cache_seq_len=prefill_seq_len,
    n_diffusion_tokens=n_diffusion_tokens,
    b_star=b_star,
    device=device,
    prefix_mask=prefix_mask,
  )

  forward_kwargs: dict[str, Any] = {}
  if model.config.expert_non_causal_attention:
    forward_kwargs["is_causal"] = False

  def step_fn(x: Any, t: Any) -> Any:
    step_start = time.perf_counter()
    b_step = x.shape[0]
    future_token_embeds = model.action_in_proj(x, t)
    if future_token_embeds.dim() == 2:
      future_token_embeds = future_token_embeds.view(b_step, n_diffusion_tokens, -1)
    expert_out_base = model.expert(
      inputs_embeds=future_token_embeds,
      position_ids=position_ids,
      past_key_values=prompt_cache,
      attention_mask=attention_mask,
      use_cache=True,
      **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
    pred = model.action_out_proj(last_hidden).view(
      -1,
      *model.action_space.get_action_space_dims(),
    )
    _sync(torch_mod)
    runtime["expert_step_calls"] = int(runtime["expert_step_calls"]) + 1
    runtime["expert_step_seconds"] = float(runtime["expert_step_seconds"]) + (time.perf_counter() - step_start)
    return pred.to(x.device)

  diffusion_start = time.perf_counter()
  sampled_action = model.diffusion.sample(
    batch_size=B * n_samples_total,
    step_fn=step_fn,
    device=device,
    return_all_steps=False,
    inference_step=diffusion_steps,
  )
  _sync(torch_mod)
  runtime["diffusion_seconds"] = time.perf_counter() - diffusion_start

  hist_xyz_rep = einops.repeat(
    ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
  )
  hist_rot_rep = einops.repeat(
    ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
  )
  pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz_rep, hist_rot_rep)
  pred_xyz = einops.rearrange(
    pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
  )
  pred_rot = einops.rearrange(
    pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
  )
  runtime["prefill_seq_len"] = prefill_seq_len
  runtime["generated_sequence_length"] = int(generated_sequences.shape[1])
  runtime["offset_min"] = int(offset.min().detach().cpu().item())
  runtime["offset_max"] = int(offset.max().detach().cpu().item())
  return pred_xyz, pred_rot, runtime


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark accelerated-token cache rebuild plus Alpamayo action expert.")
  parser.add_argument("--alpamayo-root", type=Path, default=Path("/mnt/g/alpamayo1.5"))
  parser.add_argument("--target-model", type=Path, required=True)
  parser.add_argument("--model-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="bfloat16")
  parser.add_argument("--autocast-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="bfloat16")
  parser.add_argument("--paro-native", action="store_true")
  parser.add_argument("--paro-compute-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="float16")
  parser.add_argument("--paro-output-dtype", choices=("input", "compute", "native"), default="native")
  parser.add_argument("--paro-marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--token-report", type=Path, required=True)
  parser.add_argument("--token-source-kind", choices=("pytorch_vlm_probe", "trtllm_mrope", "parity"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
  parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
  parser.add_argument("--prompt-source", choices=("dataset", "contract"), default="dataset")
  parser.add_argument("--contract-request-json", type=Path)
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
  parser.add_argument("--quantization", choices=("none", "bnb_4bit_nf4", "bnb_4bit_fp4"), default="none")
  parser.add_argument("--bnb-compute-dtype", choices=("float16", "bfloat16"), default="bfloat16")
  parser.add_argument("--float32-matmul-precision", choices=("highest", "high", "medium"), default=None)
  parser.add_argument("--repeat", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--diffusion-steps", type=int, default=6)
  parser.add_argument("--num-traj-samples", type=int, default=1)
  parser.add_argument("--num-traj-sets", type=int, default=1)
  parser.add_argument("--stop-token-id", type=int, default=155681)
  parser.add_argument("--include-after-stop", type=int, default=0)
  args = parser.parse_args()

  _insert_local_tool_path()
  from benchmark_dflash_alpamayo_generation import (
    _build_target_model,
    _gpu_stats,
    _insert_alpamayo_paths,
    _torch_dtype,
  )

  _insert_alpamayo_paths(args.alpamayo_root)

  import torch
  from alpamayo1_5.models.token_utils import extract_text_tokens

  if args.float32_matmul_precision is not None:
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

  report: dict[str, Any] = {
    "status": "started",
    "created_at_unix": time.time(),
    "cwd": os.getcwd(),
    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "gpu_initial": _gpu_stats(torch),
    "torch": torch.__version__,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
  }

  token_report = _load_json(args.token_report)
  raw_tokens = _extract_generated_tokens(token_report, args.token_source_kind)
  generated_tokens = _truncate_tokens(raw_tokens, args.stop_token_id, args.include_after_stop)
  report["token_source"] = {
    "raw_count": len(raw_tokens),
    "used_count": len(generated_tokens),
    "stop_token_id": args.stop_token_id,
    "include_after_stop": args.include_after_stop,
  }

  load_start = time.perf_counter()
  model = _build_target_model(args, torch).eval()
  _sync(torch)
  report["target_load_ms"] = round((time.perf_counter() - load_start) * 1000.0, 3)
  report["hf_device_map"] = {str(k): (int(v) if isinstance(v, int) else str(v)) for k, v in getattr(model, "hf_device_map", {}).items()}
  report["gpu_post_target_load"] = _gpu_stats(torch)

  prep_start = time.perf_counter()
  prompt_meta, tokenized_data, input_ids, traj_data_vlm, _raw_data = _prepare_prompt_with_data(args, model, torch)
  _sync(torch)
  report["prompt_prepare_ms"] = round((time.perf_counter() - prep_start) * 1000.0, 3)
  report["prompt"] = prompt_meta

  token_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=input_ids.device)
  full_ids = torch.cat([input_ids, token_tensor], dim=1)
  report["full_sequence"] = {
    "input_tokens": int(input_ids.shape[1]),
    "generated_tokens": len(generated_tokens),
    "total_tokens": int(full_ids.shape[1]),
    "generated_text": model.tokenizer.decode(generated_tokens, skip_special_tokens=False),
  }

  cache_times: list[float] = []
  action_times: list[float] = []
  total_times: list[float] = []
  runs: list[dict[str, Any]] = []

  with torch.inference_mode(), torch.autocast("cuda", dtype=_torch_dtype(torch, args.autocast_dtype)):
    for repeat_idx in range(args.warmup + args.repeat):
      phase = "warmup" if repeat_idx < args.warmup else "measure"
      torch.cuda.manual_seed_all(42 + repeat_idx)

      overall_start = time.perf_counter()
      vlm_outputs, cache_ms = _timed_ms(
        torch,
        lambda: _rebuild_vlm_cache(torch, model, tokenized_data, full_ids),
      )
      prompt_cache = vlm_outputs.past_key_values
      rope_deltas = model.vlm.model.rope_deltas
      _sync(torch)

      action_start = time.perf_counter()
      pred_xyz, pred_rot, runtime = _run_action_from_cache(
        torch,
        model,
        input_ids,
        tokenized_data,
        traj_data_vlm,
        full_ids,
        prompt_cache,
        rope_deltas,
        diffusion_steps=args.diffusion_steps,
        num_traj_samples=args.num_traj_samples,
        num_traj_sets=args.num_traj_sets,
      )
      _sync(torch)
      action_ms = (time.perf_counter() - action_start) * 1000.0
      total_ms = (time.perf_counter() - overall_start) * 1000.0

      cot = extract_text_tokens(model.tokenizer, full_ids)["cot"][0]
      run = {
        "phase": phase,
        "repeat_index": repeat_idx,
        "cache_rebuild_ms": round(cache_ms, 3),
        "action_ms": round(action_ms, 3),
        "total_ms": round(total_ms, 3),
        "runtime": {k: round(v, 6) if isinstance(v, float) else v for k, v in runtime.items()},
        "cot_preview": cot[:500],
        "pred_xyz_shape": [int(x) for x in pred_xyz.shape],
        "pred_rot_shape": [int(x) for x in pred_rot.shape],
      }
      runs.append(run)

      if phase == "measure":
        cache_times.append(cache_ms)
        action_times.append(action_ms)
        total_times.append(total_ms)

      del vlm_outputs, prompt_cache, pred_xyz, pred_rot
      torch.cuda.empty_cache()

  report.update({
    "status": "ok",
    "runs": runs,
    "summary": {
      "cache_rebuild": _summary(cache_times),
      "action": _summary(action_times),
      "total": _summary(total_times),
    },
    "gpu_post_benchmark": _gpu_stats(torch),
  })
  _write_json(args.output, report)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
