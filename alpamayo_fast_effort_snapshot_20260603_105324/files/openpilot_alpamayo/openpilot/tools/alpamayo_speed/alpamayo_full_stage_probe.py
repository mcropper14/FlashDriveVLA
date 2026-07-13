from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000


def _insert_alpamayo_paths(alpamayo_root: Path) -> None:
  sys.path.insert(0, str(alpamayo_root))
  sys.path.insert(0, str(alpamayo_root / "src"))


def _sync(torch_mod: Any) -> None:
  if torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


@contextmanager
def _cuda_timer(torch_mod: Any, records: dict[str, list[float]], name: str):
  _sync(torch_mod)
  start = time.perf_counter()
  try:
    yield
  finally:
    _sync(torch_mod)
    records.setdefault(name, []).append(time.perf_counter() - start)


def _summarize_records(records: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
  summary: dict[str, dict[str, float | int]] = {}
  for key, values in sorted(records.items()):
    if not values:
      continue
    summary[key] = {
      "count": len(values),
      "sum_ms": round(sum(values) * 1000.0, 3),
      "mean_ms": round((sum(values) / len(values)) * 1000.0, 3),
      "min_ms": round(min(values) * 1000.0, 3),
      "max_ms": round(max(values) * 1000.0, 3),
    }
  return summary


def _wrap_bound(
  owner: Any,
  attr_name: str,
  torch_mod: Any,
  records: dict[str, list[float]],
  timer_name: str,
) -> Callable[[], None]:
  original = getattr(owner, attr_name)

  def wrapped(*args: Any, **kwargs: Any) -> Any:
    with _cuda_timer(torch_mod, records, timer_name):
      return original(*args, **kwargs)

  setattr(owner, attr_name, wrapped)

  def restore() -> None:
    setattr(owner, attr_name, original)

  return restore


def _maybe_wrap_bound(
  owner: Any,
  attr_name: str,
  torch_mod: Any,
  records: dict[str, list[float]],
  timer_name: str,
) -> Callable[[], None] | None:
  if owner is None or not hasattr(owner, attr_name):
    return None
  return _wrap_bound(owner, attr_name, torch_mod, records, timer_name)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_int_list(value: str | None) -> list[int] | None:
  if value is None:
    return None
  items = []
  for raw_part in value.split(","):
    part = raw_part.strip()
    if not part:
      continue
    parsed = int(part)
    if parsed <= 0:
      raise ValueError(f"diffusion step count must be positive: {parsed}")
    items.append(parsed)
  if not items:
    raise ValueError("--diffusion-steps-list did not contain any positive integers")
  return items


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--alpamayo-root", type=Path, default=Path(r"G:\alpamayo1.5"))
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
  parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
  parser.add_argument("--num-traj-samples", type=int, default=1)
  parser.add_argument("--max-generation-length", type=int, default=256)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--camera-mode", choices=["full", "front2", "front1"], default="front2")
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--min-pixels", type=int, default=65536)
  parser.add_argument("--max-pixels", type=int, default=65536)
  parser.add_argument("--device-map-mode", choices=["auto", "manual_split"], default="manual_split")
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--repeat-infer", type=int, default=1)
  parser.add_argument("--diffusion-steps", type=int, default=6)
  parser.add_argument("--diffusion-steps-list", help="Comma-separated diffusion step counts to sweep after one model load, for example 6,4,2,1.")
  parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="flash_attention_2")
  parser.add_argument("--expert-attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
  parser.add_argument("--quantization", choices=["none", "bnb_8bit", "bnb_4bit_nf4", "bnb_4bit_fp4"], default="none")
  parser.add_argument("--bnb-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
  parser.add_argument("--greedy", action="store_true")
  parser.add_argument("--manual-generate", action="store_true")
  parser.add_argument("--deep-vlm-timers", action="store_true", help="Synchronize and time each VLM forward subcall. This adds diagnostic overhead.")
  args = parser.parse_args()
  diffusion_steps_list = _parse_int_list(args.diffusion_steps_list) or [args.diffusion_steps]

  _insert_alpamayo_paths(args.alpamayo_root)

  import numpy as np
  import physical_ai_av
  import torch

  from alpamayo1_5 import helper
  import alpamayo1_5.models.alpamayo1_5 as alpamayo_model_module
  from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
  from timed_inference import build_model, gpu_stats, select_camera_features, summarize_min_ade

  overall_start = time.perf_counter()
  report: dict[str, Any] = {
    "status": "started",
    "created_at_unix": time.time(),
    "cwd": os.getcwd(),
    "alpamayo_root": str(args.alpamayo_root),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "args": vars(args) | {"output": str(args.output), "alpamayo_root": str(args.alpamayo_root)},
    "diffusion_steps_list": diffusion_steps_list,
    "gpu_initial": gpu_stats(),
  }

  avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
  camera_features = select_camera_features(avdi, args.camera_mode)

  with _cuda_timer(torch, {}, "noop"):
    pass

  stage_records: dict[str, list[float]] = {}

  with _cuda_timer(torch, stage_records, "dataset_load"):
    data = load_physical_aiavdataset(
      args.clip_id,
      t0_us=args.t0_us,
      avdi=avdi,
      camera_features=camera_features,
      num_frames=args.num_frames,
    )
  report["image_frames_shape"] = tuple(data["image_frames"].shape)
  report["camera_indices"] = data["camera_indices"].tolist()

  with _cuda_timer(torch, stage_records, "message_build"):
    messages = helper.create_message(
      frames=data["image_frames"].flatten(0, 1),
      camera_indices=data["camera_indices"],
    )

  with _cuda_timer(torch, stage_records, "model_load"):
    model = build_model(
      gpu_mem_gib=args.gpu_mem_gib,
      cpu_mem_gib=args.cpu_mem_gib,
      device_map_mode=args.device_map_mode,
      split_index=args.split_index,
      attn_implementation=args.attn_implementation,
      expert_attn_implementation=args.expert_attn_implementation,
      quantization=args.quantization,
      bnb_compute_dtype=args.bnb_compute_dtype,
    )
  report["model_device"] = str(model.device)
  report["hf_device_map"] = {str(k): int(v) for k, v in getattr(model, "hf_device_map", {}).items()}
  report["gpu_post_load"] = gpu_stats()

  with _cuda_timer(torch, stage_records, "processor"):
    helper.MIN_PIXELS = args.min_pixels
    helper.MAX_PIXELS = args.max_pixels
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=False,
      continue_final_message=True,
      return_dict=True,
      return_tensors="pt",
    )
  report["input_ids_shape"] = tuple(inputs["input_ids"].shape)

  with _cuda_timer(torch, stage_records, "move_inputs"):
    model_inputs = {
      "tokenized_data": inputs,
      "ego_history_xyz": data["ego_history_xyz"],
      "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, model.device)
  report["gpu_pre_infer"] = gpu_stats()

  inference_reports: list[dict[str, Any]] = []
  pred_xyz = pred_rot = extra = None
  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for diffusion_steps in diffusion_steps_list:
      for repeat_idx in range(args.repeat_infer):
        infer_idx = len(inference_reports)
        torch.cuda.manual_seed_all(args.seed + infer_idx)
        records: dict[str, list[float]] = {}
        runtime_profile: dict[str, float | int] = {}

        restores = [
          _wrap_bound(model.vlm, "generate", torch, records, "vlm_generate_sync"),
          _wrap_bound(model.diffusion, "sample", torch, records, "diffusion_sample_sync"),
          _wrap_bound(model.action_space, "action_to_traj", torch, records, "action_to_traj_sync"),
          _wrap_bound(model.expert, "forward", torch, records, "expert_forward_sync"),
          _wrap_bound(model.action_in_proj, "forward", torch, records, "action_in_proj_sync"),
          _wrap_bound(model.action_out_proj, "forward", torch, records, "action_out_proj_sync"),
        ]
        if args.deep_vlm_timers:
          maybe_restores = [
            _maybe_wrap_bound(model.vlm, "forward", torch, records, "vlm_forward_sync"),
            _maybe_wrap_bound(getattr(model.vlm, "model", None), "forward", torch, records, "vlm_model_forward_sync"),
            _maybe_wrap_bound(getattr(getattr(model.vlm, "model", None), "visual", None), "forward", torch, records, "vlm_visual_forward_sync"),
            _maybe_wrap_bound(getattr(getattr(model.vlm, "model", None), "language_model", None), "forward", torch, records, "vlm_language_forward_sync"),
          ]
          restores.extend(restore for restore in maybe_restores if restore is not None)
        original_extract = alpamayo_model_module.extract_text_tokens

        def wrapped_extract(*extract_args: Any, **extract_kwargs: Any) -> Any:
          with _cuda_timer(torch, records, "extract_text_tokens_sync"):
            return original_extract(*extract_args, **extract_kwargs)

        alpamayo_model_module.extract_text_tokens = wrapped_extract
        try:
          with _cuda_timer(torch, records, "full_sample_sync"):
            outputs = model.sample_trajectories_from_data_with_vlm_rollout(
              data=model_inputs,
              top_p=0.98,
              temperature=0.6,
              num_traj_samples=args.num_traj_samples,
              max_generation_length=args.max_generation_length,
              return_extra=True,
              diffusion_kwargs={"inference_step": diffusion_steps},
              runtime_profile=runtime_profile,
              do_sample=not args.greedy,
              manual_generation=args.manual_generate,
            )
        finally:
          alpamayo_model_module.extract_text_tokens = original_extract
          for restore in reversed(restores):
            restore()

        pred_xyz, pred_rot, extra = outputs
        inference_report: dict[str, Any] = {
          "infer_index": infer_idx,
          "repeat_index": repeat_idx,
          "diffusion_steps": diffusion_steps,
          "synced_stage_summary": _summarize_records(records),
          "runtime_profile_unsynced": runtime_profile,
          "min_ade_m": round(float(summarize_min_ade(data, pred_xyz)), 6),
        }
        if extra is not None:
          cot = extra["cot"][0, 0, 0]
          inference_report["cot_chars"] = len(cot)
          inference_report["cot_preview"] = cot[:600].replace("\n", " ")
        inference_reports.append(inference_report)
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  if pred_xyz is None or pred_rot is None:
    raise RuntimeError("inference did not return a trajectory")

  min_ade = summarize_min_ade(data, pred_xyz)
  report.update(
    {
      "status": "ok",
      "stage_summary": _summarize_records(stage_records),
      "inferences": inference_reports,
      "min_ade_m": round(float(min_ade), 6),
      "pred_xyz_shape": tuple(pred_xyz.shape),
      "pred_rot_shape": tuple(pred_rot.shape),
      "gpu_post_infer": gpu_stats(),
      "overall_seconds": round(time.perf_counter() - overall_start, 3),
    }
  )
  # Force JSON-friendly scalar conversion for nested numpy values if a caller adds them later.
  report = json.loads(json.dumps(report, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
  _write_json(args.output, report)
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
