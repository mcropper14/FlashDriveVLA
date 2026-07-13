#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_dflash_alpamayo_generation import (
  _build_target_model,
  _gpu_stats,
  _prepare_prompt,
  _progress,
  _insert_alpamayo_paths,
  _standard_generate_once,
  _summary,
  _sync,
  _torch_dtype,
)


def _compile_vlm_forward(args: argparse.Namespace, model: Any, torch_mod: Any) -> dict[str, Any]:
  if not args.compile_vlm_forward:
    return {
      "attempted": False,
      "status": "skipped",
    }

  try:
    compiled_forward = torch_mod.compile(model.vlm.forward, mode="max-autotune", fullgraph=False)
    model.vlm.forward = compiled_forward
    return {
      "attempted": True,
      "status": "ok",
      "mode": "max-autotune",
      "fullgraph": False,
    }
  except Exception as exc:
    return {
      "attempted": True,
      "status": "error",
      "type": type(exc).__name__,
      "message": str(exc),
    }


def _decode_token_ids(torch_mod: Any, tokenizer: Any, token_ids: list[int], stop: int, *, max_after_stop: int = 2) -> dict[str, Any]:
  text = tokenizer.decode(token_ids, skip_special_tokens=False) if token_ids else ""
  details: dict[str, Any] = {
    "token_count": len(token_ids),
    "token_ids": token_ids,
    "text": text,
    "stop_token_id": stop,
  }
  if stop in token_ids:
    details["stop_token_index"] = token_ids.index(stop)
    details["token_prefix_after_stop"] = token_ids[details["stop_token_index"] + 1 : details["stop_token_index"] + 1 + max_after_stop]
  return details


def _manual_greedy_vlm_generate(
  model: Any,
  tokenized_data: dict[str, Any],
  input_ids: Any,
  *,
  torch_mod: Any,
  max_generation_length: int,
  traj_future_start_id: int,
) -> dict[str, Any]:
  from transformers import LogitsProcessorList
  from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor

  logits_processor = LogitsProcessorList([
    ExpertLogitsProcessor(
      traj_token_offset=model.config.traj_token_start_idx,
      traj_vocab_size=model.config.traj_vocab_size,
    )
  ])
  generated_sequences = input_ids
  model_kwargs = dict(tokenized_data)
  model_kwargs["use_cache"] = True
  generated_ids: list[int] = []
  decode_forward_ms: list[float] = []
  prefill_ms = 0.0
  decode_ms = 0.0
  decode_forwards = 0
  eos_seen = False
  stop_reason = "max_length"

  for step_idx in range(max_generation_length + 1):
    if step_idx == 0:
      model_inputs = model.vlm.prepare_inputs_for_generation(
        generated_sequences,
        is_first_iteration=True,
        **model_kwargs,
      )
    else:
      model_inputs = model.vlm.prepare_inputs_for_generation(
        generated_sequences,
        next_sequence_length=1,
        is_first_iteration=False,
        **model_kwargs,
      )
    model_inputs["logits_to_keep"] = 1
    step_start = time.perf_counter()
    outputs = model.vlm(**model_inputs, return_dict=True)
    _sync(torch_mod)
    step_ms = (time.perf_counter() - step_start) * 1000.0
    if step_idx == 0:
      prefill_ms += step_ms
    else:
      decode_ms += step_ms
      decode_forward_ms.append(step_ms)
      decode_forwards += 1

    model_kwargs = model.vlm._update_model_kwargs_for_generation(
      outputs,
      model_kwargs,
      is_encoder_decoder=False,
    )
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
      model_kwargs.pop(key, None)

    if eos_seen:
      stop_reason = "traj_future_start_plus_one"
      break
    if step_idx >= max_generation_length:
      stop_reason = "max_length"
      break

    next_scores = logits_processor(generated_sequences, outputs.logits[:, -1, :])
    next_token_ids = next_scores.argmax(dim=-1, keepdim=True)
    next_token_ids = next_token_ids.to(generated_sequences.device)
    next_id = int(next_token_ids[0, 0].detach().cpu().item())
    generated_ids.append(next_id)
    if next_id == traj_future_start_id:
      eos_seen = True
      stop_reason = "traj_future_start_reached"
    generated_sequences = torch_mod.cat([generated_sequences, next_token_ids], dim=-1)

  return {
    "generated_ids": generated_ids,
    "prefill_ms": round(prefill_ms, 3),
    "decode_ms": round(decode_ms, 3),
    "decode_forward_ms": [round(v, 3) for v in decode_forward_ms],
    "decode_forwards": decode_forwards,
    "generated_tokens": int(len(generated_ids)),
    "stop_reason": stop_reason,
    "input_len": int(input_ids.shape[1]),
    "stop_after_traj_start": bool(eos_seen),
  }


def _prepare_and_load_model(args: argparse.Namespace, torch_mod: Any) -> Any:
  args.paro_native = True
  args.device_map_mode = "current_split"
  if args.quantization != "none":
    raise ValueError("--quantization must be none for PARO native mode")

  return _build_target_model(args, torch_mod)


def _prepare_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Profile PARO native VLM-only benchmark with HF generate and manual greedy loop."
  )
  parser.add_argument("--alpamayo-root", type=Path, default=Path("/mnt/g/alpamayo1.5"))
  parser.add_argument(
    "--target-model",
    type=Path,
    default=Path("/mnt/j/temp_alpamayo/Alpamayo-1.5-10B-finetuned-PARO"),
  )
  parser.add_argument("--processor-model", default=os.environ.get("ALPAMAYO_PROCESSOR_MODEL"))
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repeat", type=int, default=3)
  parser.add_argument("--max-generation-length", type=int, default=32)
  parser.add_argument("--num-frames", type=int, default=2)
  parser.add_argument("--synthetic-width", type=int, default=512)
  parser.add_argument("--synthetic-height", type=int, default=384)
  parser.add_argument("--min-pixels", type=int, default=65536)
  parser.add_argument("--max-pixels", type=int, default=65536)
  parser.add_argument("--gpu-mem-gib", type=int, default=15)
  parser.add_argument("--cpu-mem-gib", type=int, default=96)
  parser.add_argument("--split-index", type=int, default=16)
  parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
  parser.add_argument("--expert-attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
  parser.add_argument("--model-dtype", choices=("fp16", "float16", "bf16", "bfloat16", "fp32", "float32"), default="float16")
  parser.add_argument("--autocast-dtype", choices=("fp16", "float16", "bf16", "bfloat16", "fp32", "float32"), default="bfloat16")
  parser.add_argument("--paro-compute-dtype", choices=("fp16", "float16", "bf16", "bfloat16", "fp32", "float32"), default="float16")
  parser.add_argument("--paro-output-dtype", choices=("input", "compute", "native"), default="native")
  parser.add_argument("--paro-marlin-input-dtype", choices=("int8", "fp8", "none"), default="int8")
  parser.add_argument("--torch-cuda-arch-list", default="8.9+PTX")
  parser.add_argument("--quantization", choices=("none",), default="none")
  parser.add_argument("--bnb-compute-dtype", choices=("float16", "bfloat16"), default="float16")
  parser.add_argument("--compile-vlm-forward", action="store_true", help="Attempt torch.compile(model.vlm.forward, mode='max-autotune').")
  parser.add_argument("--skip-manual", action="store_true", help="Only benchmark HF generate().")
  parser.add_argument("--greedy", dest="greedy", action="store_true", help="HF generate greedy mode.")
  parser.add_argument("--sample", dest="greedy", action="store_false", help="HF generate sample mode.")
  parser.set_defaults(greedy=True)
  return parser


def main() -> None:
  parser = _prepare_parser()
  args = parser.parse_args()
  args.prompt_source = "synthetic"
  if args.output is None:
    parser.error("--output is required")

  _insert_alpamayo_paths(args.alpamayo_root)

  import torch
  from alpamayo1_5.models.token_utils import to_special_token

  report: dict[str, Any] = {
    "status": "started",
    "created_at_unix": time.time(),
    "cwd": os.getcwd(),
    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "gpu_initial": _gpu_stats(torch),
  }
  _progress("start", args=report["args"], gpu_initial=report["gpu_initial"])

  try:
    _progress("target_load_start", target_model=str(args.target_model))
    load_start = time.perf_counter()
    model = _prepare_and_load_model(args, torch).eval()
    _sync(torch)
    report["target_load_ms"] = round((time.perf_counter() - load_start) * 1000.0, 3)
    report["hf_device_map"] = {str(k): int(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    report["target_model"] = str(args.target_model)
    report["gpu_post_load"] = _gpu_stats(torch)
    report["paro_replacement_records"] = len(getattr(model, "_openpilot_paro_replacement_records", []))
    report["paro_finalize"] = getattr(model, "_openpilot_paro_finalize", None)
    _progress(
      "target_load_done",
      target_load_ms=report["target_load_ms"],
      hf_device_map=report["hf_device_map"],
      gpu=report["gpu_post_load"],
    )

    compile_result = _compile_vlm_forward(args, model, torch)
    report["compile"] = compile_result
    _progress("compile_vlm_forward_done", compile=compile_result)

    _progress("prompt_prepare_start")
    prompt_start = time.perf_counter()
    prompt_meta, tokenized_data, input_ids = _prepare_prompt(args, model, torch)
    _sync(torch)
    report["prompt"] = prompt_meta
    report["prompt_prepare_ms"] = round((time.perf_counter() - prompt_start) * 1000.0, 3)
    report["gpu_pre_benchmark"] = _gpu_stats(torch)
    _progress("prompt_prepare_done", prompt=prompt_meta, gpu=report["gpu_pre_benchmark"])

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    if eos_token_id is None:
      raise RuntimeError("traj_future_start token id is missing")

    hf_generate_runs: list[float] = []
    manual_decode_forwards: list[int] = []
    hf_run_detail: list[dict[str, Any]] = []
    manual_run_detail: list[dict[str, Any]] = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=_torch_dtype(torch, args.autocast_dtype)):
      for repeat_idx in range(args.repeat):
        torch.cuda.manual_seed_all(42 + repeat_idx)
        _progress("hf_generate_start", repeat=repeat_idx)
        start = time.perf_counter()
        outputs = _standard_generate_once(
          model,
          tokenized_data,
          input_ids,
          max_generation_length=args.max_generation_length,
          greedy=args.greedy,
        )
        _sync(torch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        hf_generate_runs.append(round(elapsed_ms, 3))
        new_tokens = int(outputs.sequences.shape[1] - input_ids.shape[1])
        _progress(
          "hf_generate_done",
          repeat=repeat_idx,
          elapsed_ms=round(elapsed_ms, 3),
          new_tokens=new_tokens,
        )
        hf_run_detail.append({
          "repeat": repeat_idx,
          "elapsed_ms": round(elapsed_ms, 3),
          "new_tokens": new_tokens,
          "sequence_len": int(outputs.sequences.shape[1]),
          "generated": _decode_token_ids(
            torch,
            model.tokenizer,
            [int(x) for x in outputs.sequences[0, input_ids.shape[1]:].detach().cpu().tolist()],
            eos_token_id,
          ),
        })

        if not args.skip_manual:
          _progress("manual_greedy_start", repeat=repeat_idx)
          manual_start = time.perf_counter()
          manual = _manual_greedy_vlm_generate(
            model=model,
            tokenized_data=tokenized_data,
            input_ids=input_ids,
            torch_mod=torch,
            max_generation_length=args.max_generation_length,
            traj_future_start_id=eos_token_id,
          )
          _sync(torch)
          elapsed_manual = (time.perf_counter() - manual_start) * 1000.0
          manual_decode_forwards.append(int(manual["decode_forwards"]))
          _progress(
            "manual_greedy_done",
            repeat=repeat_idx,
            prefill_ms=manual["prefill_ms"],
            decode_ms=manual["decode_ms"],
            decode_forwards=int(manual["decode_forwards"]),
          )
          manual_run_detail.append({
            "repeat": repeat_idx,
            "prefill_ms": manual["prefill_ms"],
            "decode_ms": manual["decode_ms"],
            "decode_forwards": int(manual["decode_forwards"]),
            "generated_tokens": int(manual["generated_tokens"]),
            "stop_reason": manual["stop_reason"],
            "decode_forward_ms": manual["decode_forward_ms"],
            "generated": _decode_token_ids(
              torch,
              model.tokenizer,
              manual["generated_ids"],
              eos_token_id,
            ),
            "elapsed_ms": round(elapsed_manual, 3),
          })

    report["timings"] = {
      "hf_generate_ms": hf_generate_runs,
      "manual_prefill_ms": [x["prefill_ms"] for x in manual_run_detail],
      "manual_decode_ms": [x["decode_ms"] for x in manual_run_detail],
      "manual_decode_forwards": manual_decode_forwards,
    }
    report["summaries"] = {
      "hf_generate": _summary(hf_generate_runs),
      "manual_prefill": _summary([x["prefill_ms"] for x in manual_run_detail]),
      "manual_decode": _summary([x["decode_ms"] for x in manual_run_detail]),
      "manual_decode_forwards": {
        "count": len(manual_decode_forwards),
        "mean": round(mean(manual_decode_forwards), 3) if manual_decode_forwards else 0.0,
        "min": min(manual_decode_forwards) if manual_decode_forwards else 0,
        "max": max(manual_decode_forwards) if manual_decode_forwards else 0,
      },
    }
    report["runs"] = {
      "hf_generate": hf_run_detail,
      "manual_greedy": manual_run_detail,
    }
    report["status"] = "ok"
    report["gpu_post_benchmark"] = _gpu_stats(torch)
    _progress("done", status=report["status"], output=str(args.output))

  except Exception as exc:
    report["status"] = "error"
    report["error"] = {
      "type": type(exc).__name__,
      "message": str(exc),
    }
    _progress("error", error=report["error"])

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
