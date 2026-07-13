#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


DEFAULT_TARGET_MODEL = Path("/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-10B-finetuned")


def _add_alpamayo_paths(alpamayo_root: Path) -> None:
  sys.path.insert(0, str(alpamayo_root))
  sys.path.insert(0, str(alpamayo_root / "src"))
  script_dir = Path(__file__).resolve().parent
  repo_root = script_dir.parents[1]
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
  if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


def _json_default(value: Any) -> Any:
  try:
    import torch

    if isinstance(value, torch.dtype):
      return str(value)
    if isinstance(value, torch.device):
      return str(value)
    if isinstance(value, torch.Tensor):
      return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
      }
  except Exception:
    pass
  return str(value)


def _write_report(path: Path, report: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")


def _gpu_stats(torch_module: Any) -> dict[int, dict[str, float]]:
  stats: dict[int, dict[str, float]] = {}
  for idx in range(torch_module.cuda.device_count()):
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(idx)
    stats[idx] = {
      "used_gib": round((total_bytes - free_bytes) / (1024**3), 3),
      "free_gib": round(free_bytes / (1024**3), 3),
      "total_gib": round(total_bytes / (1024**3), 3),
    }
  return stats


def _summarize_model(model: Any) -> dict[str, Any]:
  vlm = model.vlm
  language_model = getattr(getattr(vlm, "model", None), "language_model", None)
  visual = getattr(getattr(vlm, "model", None), "visual", None)
  expert = model.expert
  config = getattr(vlm, "config", None)
  text_config = getattr(config, "text_config", None)
  return {
    "alpamayo_class": type(model).__name__,
    "vlm_class": type(vlm).__name__,
    "visual_class": type(visual).__name__ if visual is not None else None,
    "language_model_class": type(language_model).__name__ if language_model is not None else None,
    "expert_class": type(expert).__name__,
    "vlm_model_type": getattr(config, "model_type", None),
    "vlm_architectures": getattr(config, "architectures", None),
    "text_model_type": getattr(text_config, "model_type", None),
    "num_text_layers": getattr(text_config, "num_hidden_layers", None),
    "hidden_size": getattr(text_config, "hidden_size", None),
    "intermediate_size": getattr(text_config, "intermediate_size", None),
    "num_attention_heads": getattr(text_config, "num_attention_heads", None),
    "num_key_value_heads": getattr(text_config, "num_key_value_heads", None),
    "vocab_size": getattr(text_config, "vocab_size", None),
    "original_vocab_size": getattr(model, "original_vocab_size", None),
    "hf_device_map": getattr(model, "hf_device_map", None),
  }


def _build_calibration_inputs(args: argparse.Namespace, helper: Any, model: Any) -> dict[str, Any]:
  nav_text = None
  num_frames_per_camera = args.num_frames
  source_meta: dict[str, Any] = {"calibration_source": args.calibration_source}
  if args.calibration_source == "contract":
    import torch
    from benchmark_dflash_alpamayo_generation import _load_contract_prompt

    image_frames, camera_indices, ego_history_xyz, ego_history_rot, source_meta = _load_contract_prompt(args, torch)
    num_frames_per_camera = int(source_meta["frames_per_camera"])
    nav_text = source_meta.get("nav_text")
    data = {
      "image_frames": image_frames,
      "camera_indices": camera_indices,
      "ego_history_xyz": ego_history_xyz,
      "ego_history_rot": ego_history_rot,
    }
  else:
    import physical_ai_av
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    if args.camera_mode == "front2":
      camera_features = [
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
      ]
    elif args.camera_mode == "front1":
      camera_features = [avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV]
    elif args.camera_mode == "full":
      camera_features = None
    else:
      raise ValueError(f"Unsupported camera_mode={args.camera_mode}")

    data = load_physical_aiavdataset(
      args.clip_id,
      t0_us=args.t0_us,
      avdi=avdi,
      camera_features=camera_features,
      num_frames=args.num_frames,
    )
  messages = helper.create_message(
    frames=data["image_frames"].flatten(0, 1),
    camera_indices=data["camera_indices"],
    num_frames_per_camera=num_frames_per_camera,
    nav_text=nav_text,
  )
  helper.MIN_PIXELS = args.min_pixels
  helper.MAX_PIXELS = args.max_pixels
  if args.processor_model:
    from benchmark_dflash_alpamayo_generation import _normalize_path

    helper.BASE_PROCESSOR_NAME = str(_normalize_path(args.processor_model))
  processor = helper.get_processor(model.tokenizer)
  tokenized = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
  )
  tokenized["input_ids"] = model.fuse_traj_tokens(
    tokenized["input_ids"],
    {
      "ego_history_xyz": data["ego_history_xyz"],
      "ego_history_rot": data["ego_history_rot"],
    },
  )
  return {
    "raw_data": data,
    "messages": messages,
    "tokenized": tokenized,
    "input_ids_shape": tuple(tokenized["input_ids"].shape),
    "image_frames_shape": tuple(data["image_frames"].shape),
    "camera_indices": data["camera_indices"].tolist(),
    "source_meta": source_meta,
    "num_frames_per_camera": num_frames_per_camera,
  }


def _move_tokenized_to_device(tokenized: dict[str, Any], device: Any) -> dict[str, Any]:
  moved: dict[str, Any] = {}
  for key, value in tokenized.items():
    if hasattr(value, "to"):
      moved[key] = value.to(device)
    else:
      moved[key] = value
  return moved


def _resolve_quant_config(args: argparse.Namespace, mtq: Any) -> dict[str, Any]:
  config = copy.deepcopy(getattr(mtq, args.quant_config))
  if args.quant_scope == "text":
    config.setdefault("quant_cfg", []).extend([
      {"quantizer_name": "*visual*", "enable": False},
      {"quantizer_name": "*vision*", "enable": False},
      {"quantizer_name": "*mm_encoder*", "enable": False},
      {"parent_class": "nn.Conv1d", "quantizer_name": "*", "enable": False},
      {"parent_class": "nn.Conv2d", "quantizer_name": "*", "enable": False},
      {"parent_class": "nn.Conv3d", "quantizer_name": "*", "enable": False},
    ])
  return config


def _quantize_vlm(args: argparse.Namespace, model: Any, calibration: dict[str, Any],
                  report: dict[str, Any]) -> Any:
  import torch
  import modelopt.torch.quantization as mtq

  config = _resolve_quant_config(args, mtq)
  report["quant_config_name"] = args.quant_config
  report["quant_scope"] = args.quant_scope
  report["quant_algorithm"] = copy.deepcopy(config).get("algorithm")

  quant_start = time.perf_counter()
  if args.quant_scope == "text":
    language_model = model.vlm.model.language_model
    first_device = next(language_model.parameters()).device
    input_ids = calibration["tokenized"]["input_ids"].to(first_device)

    def forward_loop(lm: Any) -> None:
      with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        lm(input_ids=input_ids, use_cache=True, return_dict=True)

    model.vlm.model.language_model = mtq.quantize(language_model, config, forward_loop)
    quantized_target = model.vlm.model.language_model
  else:
    tokenized_for_vlm = _move_tokenized_to_device(copy.deepcopy(calibration["tokenized"]), model.device)
    input_ids = tokenized_for_vlm.pop("input_ids")

    def forward_loop(vlm: Any) -> None:
      with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        vlm(input_ids=input_ids, use_cache=True, return_dict=True, **tokenized_for_vlm)

    model.vlm = mtq.quantize(model.vlm, config, forward_loop)
    quantized_target = model.vlm
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  report["quantize_seconds"] = round(time.perf_counter() - quant_start, 3)
  report["gpu_after_quantize"] = _gpu_stats(torch)
  report["quantized_target_class"] = type(quantized_target).__name__
  return quantized_target


def _export_vlm(args: argparse.Namespace, model: Any, report: dict[str, Any], helper: Any) -> None:
  import torch
  from modelopt.torch.export import export_hf_checkpoint

  export_dir = Path(args.export_dir)
  export_dir.mkdir(parents=True, exist_ok=True)
  export_start = time.perf_counter()
  export_hf_checkpoint(
    model.vlm,
    dtype=torch.bfloat16,
    export_dir=export_dir,
    max_shard_size=args.max_shard_size,
  )
  model.tokenizer.save_pretrained(export_dir)
  try:
    processor = helper.get_processor(model.tokenizer)
    processor.save_pretrained(export_dir)
    report["processor_saved"] = True
  except Exception as e:
    report["processor_saved"] = False
    report["processor_save_error"] = f"{type(e).__name__}: {e}"
  report["export_seconds"] = round(time.perf_counter() - export_start, 3)
  report["export_dir"] = str(export_dir)
  report["export_files"] = sorted(p.name for p in export_dir.iterdir())


def main() -> int:
  parser = argparse.ArgumentParser(description="Inspect or export Alpamayo VLM for NVFP4 runtime work.")
  parser.add_argument("--alpamayo-root", type=Path, default=Path("/mnt/g/alpamayo1.5"))
  parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
  parser.add_argument("--model-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="bfloat16")
  parser.add_argument("--paro-native", action="store_true")
  parser.add_argument("--paro-compute-dtype", choices=("fp16", "bf16", "fp32", "float16", "float32", "bfloat16"), default="float16")
  parser.add_argument("--paro-output-dtype", choices=("input", "compute", "native"), default="native")
  parser.add_argument("--paro-marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--artifact-dir", type=Path, default=Path("/mnt/e/ture_opamayo/openpilot_alpamayo/openpilot/artifacts/alpamayo_speed"))
  parser.add_argument("--report-name", default="alpamayo_nvfp4_export_report.json")
  parser.add_argument("--export-dir", type=Path, default=Path("/mnt/g/alpamayo1.5/nvfp4_exports/alpamayo_vlm_nvfp4"))
  parser.add_argument("--clip-id", default="030c760c-ae38-49aa-9ad8-f5650a545d26")
  parser.add_argument("--t0-us", type=int, default=5_100_000)
  parser.add_argument("--calibration-source", choices=("dataset", "contract"), default="dataset")
  parser.add_argument("--contract-request-json", type=Path)
  parser.add_argument("--camera-mode", choices=["front1", "front2", "full"], default="front2")
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--min-pixels", type=int, default=65536)
  parser.add_argument("--max-pixels", type=int, default=65536)
  parser.add_argument("--processor-model", default=os.environ.get("ALPAMAYO_PROCESSOR_MODEL"))
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--device-map-mode", choices=("auto", "current_split"), default="current_split")
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--attn-implementation", default="flash_attention_2")
  parser.add_argument("--expert-attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
  parser.add_argument("--quantization", choices=("none", "bnb_4bit_nf4", "bnb_4bit_fp4"), default="none")
  parser.add_argument("--bnb-compute-dtype", choices=("float16", "bfloat16"), default="bfloat16")
  parser.add_argument("--quant-config", default="NVFP4_DEFAULT_CFG")
  parser.add_argument("--quant-scope", choices=["all", "text"], default="all")
  parser.add_argument("--max-shard-size", default="5GB")
  parser.add_argument("--inspect-only", action="store_true")
  parser.add_argument("--skip-quantize", action="store_true")
  parser.add_argument("--export", action="store_true")
  args = parser.parse_args()

  _add_alpamayo_paths(args.alpamayo_root)

  import torch
  from alpamayo1_5 import helper
  from benchmark_dflash_alpamayo_generation import _build_target_model

  report: dict[str, Any] = {
    "created_at_unix": time.time(),
    "alpamayo_root": str(args.alpamayo_root),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_initial": _gpu_stats(torch),
    "args": vars(args),
  }
  report_path = args.artifact_dir / args.report_name

  try:
    load_start = time.perf_counter()
    model = _build_target_model(args, torch)
    report["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
    report["gpu_after_load"] = _gpu_stats(torch)
    report["model"] = _summarize_model(model)

    calib_start = time.perf_counter()
    calibration = _build_calibration_inputs(args, helper, model)
    report["calibration_build_seconds"] = round(time.perf_counter() - calib_start, 3)
    report["calibration"] = {
      "input_ids_shape": calibration["input_ids_shape"],
      "image_frames_shape": calibration["image_frames_shape"],
      "camera_indices": calibration["camera_indices"],
      "source_meta": calibration["source_meta"],
      "num_frames_per_camera": calibration["num_frames_per_camera"],
    }

    if not args.inspect_only:
      if not args.skip_quantize:
        _quantize_vlm(args, model, calibration, report)
      if args.export:
        _export_vlm(args, model, report, helper)
    report["status"] = "ok"
  except Exception as e:
    report["status"] = "error"
    report["error_type"] = type(e).__name__
    report["error"] = str(e)
    _write_report(report_path, report)
    raise

  _write_report(report_path, report)
  print(json.dumps(report, indent=2, sort_keys=True, default=_json_default), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
