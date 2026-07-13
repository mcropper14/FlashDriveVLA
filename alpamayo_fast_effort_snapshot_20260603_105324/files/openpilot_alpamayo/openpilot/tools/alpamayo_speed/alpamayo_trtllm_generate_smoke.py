#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path


def _make_test_image():
  from PIL import Image, ImageDraw

  image = Image.new("RGB", (384, 256), (35, 38, 42))
  draw = ImageDraw.Draw(image)
  draw.rectangle((130, 40, 250, 220), fill=(65, 68, 70))
  draw.line((192, 240, 192, 40), fill=(240, 240, 240), width=3)
  draw.rectangle((250, 150, 360, 220), fill=(220, 120, 20))
  draw.text((18, 18), "front camera scene", fill=(255, 255, 255))
  draw.text((260, 125), "cones", fill=(255, 255, 255))
  return image


def _build_input(model_dir: str, mode: str, prompt: str, image_path: str | None, processor=None):
  if mode == "text":
    return prompt

  from PIL import Image
  image = Image.open(image_path).convert("RGB") if image_path else _make_test_image()
  if processor is None:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
  messages = [{
    "role": "user",
    "content": [
      {"type": "image", "image": image},
      {"type": "text", "text": prompt},
    ],
  }]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  return {
    "prompt": text,
    "multi_modal_data": {
      "image": [image],
    },
  }


def _output_text(result) -> str:
  outputs = getattr(result, "outputs", None)
  if not outputs:
    return repr(result)
  first = outputs[0]
  return str(getattr(first, "text", first))


def main() -> int:
  parser = argparse.ArgumentParser(description="Benchmark TensorRT-LLM generation from an Alpamayo VLM export.")
  parser.add_argument("--model-dir", required=True)
  parser.add_argument("--workspace", required=True)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--mode", choices=("text", "image"), default="image")
  parser.add_argument("--image")
  parser.add_argument("--prompt", default="Describe the driving-relevant scene in one short sentence.")
  parser.add_argument("--tensor-parallel-size", type=int, default=1)
  parser.add_argument("--max-batch-size", type=int, default=1)
  parser.add_argument("--max-input-len", type=int, default=1024)
  parser.add_argument("--max-seq-len", type=int, default=1056)
  parser.add_argument("--max-num-tokens", type=int, default=1152)
  parser.add_argument("--max-output-tokens", type=int, default=8)
  parser.add_argument("--backend", choices=("tensorrt", "pytorch"), default="pytorch")
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--disable-overlap-scheduler", action="store_true")
  parser.add_argument("--disable-cuda-graph", action="store_true")
  parser.add_argument("--disable-kv-cache-reuse", action="store_true")
  args = parser.parse_args()

  report = {
    "status": "started",
    "created_at_unix": time.time(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "env": {
      "TLLM_DISABLE_FLASHINFER_NORM": os.environ.get("TLLM_DISABLE_FLASHINFER_NORM"),
      "TLLM_DISABLE_MM_PROFILE": os.environ.get("TLLM_DISABLE_MM_PROFILE"),
      "TLLM_DISABLE_MPI": os.environ.get("TLLM_DISABLE_MPI"),
      "CUDA_HOME": os.environ.get("CUDA_HOME"),
    },
    "model_dir": args.model_dir,
    "workspace": args.workspace,
    "backend": args.backend,
    "mode": args.mode,
    "prompt": args.prompt,
    "image": args.image,
    "latency_ms": [],
    "warmup_latency_ms": [],
    "outputs": [],
  }

  llm = None
  try:
    import torch
    if args.backend == "tensorrt":
      from tensorrt_llm._tensorrt_engine import LLM
    else:
      from tensorrt_llm import LLM
    from tensorrt_llm import SamplingParams

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    kwargs = {
      "model": args.model_dir,
      "trust_remote_code": True,
      "tensor_parallel_size": args.tensor_parallel_size,
      "dtype": "auto",
      "max_batch_size": args.max_batch_size,
      "max_input_len": args.max_input_len,
      "max_seq_len": args.max_seq_len,
      "max_num_tokens": args.max_num_tokens,
    }
    ignored_for_backend: list[str] = []
    if args.disable_overlap_scheduler and args.backend == "pytorch":
      kwargs["disable_overlap_scheduler"] = True
    elif args.disable_overlap_scheduler:
      ignored_for_backend.append("disable_overlap_scheduler")
    if args.disable_cuda_graph and args.backend == "pytorch":
      kwargs["cuda_graph_config"] = None
    elif args.disable_cuda_graph:
      ignored_for_backend.append("disable_cuda_graph")
    if args.disable_kv_cache_reuse and args.backend == "pytorch":
      from tensorrt_llm.llmapi.llm_args import KvCacheConfig
      kwargs["kv_cache_config"] = KvCacheConfig(enable_block_reuse=False, enable_partial_reuse=False)
    elif args.disable_kv_cache_reuse:
      ignored_for_backend.append("disable_kv_cache_reuse")
    if args.backend == "tensorrt":
      Path(args.workspace).mkdir(parents=True, exist_ok=True)
      kwargs["workspace"] = args.workspace
    if ignored_for_backend:
      report["ignored_for_backend"] = ignored_for_backend

    t_load = time.perf_counter()
    llm = LLM(**kwargs)
    report["load_seconds"] = round(time.perf_counter() - t_load, 3)

    sampling = SamplingParams(max_tokens=args.max_output_tokens, temperature=0.0)

    processor = None
    if args.mode == "image":
      from transformers import AutoProcessor
      processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)

    for _ in range(args.warmup):
      request_input = _build_input(args.model_dir, args.mode, args.prompt, args.image, processor)
      t0 = time.perf_counter()
      result = llm.generate(request_input, sampling_params=sampling, use_tqdm=False, cache_salt=f"warmup-{time.time_ns()}")
      report["warmup_latency_ms"].append(round((time.perf_counter() - t0) * 1000.0, 3))
      report["outputs"].append(_output_text(result))

    for _ in range(args.iters):
      request_input = _build_input(args.model_dir, args.mode, args.prompt, args.image, processor)
      t0 = time.perf_counter()
      result = llm.generate(request_input, sampling_params=sampling, use_tqdm=False, cache_salt=f"iter-{time.time_ns()}")
      report["latency_ms"].append(round((time.perf_counter() - t0) * 1000.0, 3))
      report["outputs"].append(_output_text(result))

    if report["latency_ms"]:
      report["mean_latency_ms"] = round(statistics.mean(report["latency_ms"]), 3)
      report["min_latency_ms"] = round(min(report["latency_ms"]), 3)
      report["max_latency_ms"] = round(max(report["latency_ms"]), 3)
    report["status"] = "ok"
  except Exception as e:
    report["status"] = "error"
    report["error_type"] = type(e).__name__
    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
  finally:
    if llm is not None:
      try:
        llm.shutdown()
      except Exception as e:
        report["shutdown_error"] = repr(e)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))

  return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
