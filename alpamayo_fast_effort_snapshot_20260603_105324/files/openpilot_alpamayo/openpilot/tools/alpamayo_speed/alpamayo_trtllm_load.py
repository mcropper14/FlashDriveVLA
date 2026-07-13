#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description="Smoke-load a TensorRT-LLM LLM from an Alpamayo VLM export.")
  parser.add_argument("--model-dir", required=True)
  parser.add_argument("--workspace", required=True)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--tensor-parallel-size", type=int, default=1)
  parser.add_argument("--max-batch-size", type=int, default=1)
  parser.add_argument("--max-input-len", type=int, default=1024)
  parser.add_argument("--max-seq-len", type=int, default=1056)
  parser.add_argument("--max-num-tokens", type=int, default=1152)
  parser.add_argument("--max-prompt-embedding-table-size", type=int, default=4096)
  parser.add_argument("--backend", choices=("tensorrt", "pytorch"), default="tensorrt")
  parser.add_argument("--disable-overlap-scheduler", action="store_true")
  parser.add_argument("--disable-cuda-graph", action="store_true")
  parser.add_argument("--disable-kv-cache-reuse", action="store_true")
  args = parser.parse_args()

  report = {
    "status": "started",
    "created_at_unix": time.time(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "model_dir": args.model_dir,
    "workspace": args.workspace,
  }
  if args.backend == "tensorrt":
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    if args.backend == "tensorrt":
      from tensorrt_llm._tensorrt_engine import LLM
    else:
      from tensorrt_llm import LLM
    report["import_seconds"] = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
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
      kwargs["workspace"] = args.workspace
    if ignored_for_backend:
      report["ignored_for_backend"] = ignored_for_backend
    llm = LLM(**kwargs)
    report["load_seconds"] = round(time.perf_counter() - t1, 3)
    report["status"] = "ok"
    report["llm_class"] = type(llm).__name__
    report["backend"] = args.backend
    engine_dir = getattr(llm, "_engine_dir", None)
    if engine_dir is not None:
      report["engine_dir"] = str(engine_dir)
  except Exception as e:
    report["status"] = "error"
    report["error_type"] = type(e).__name__
    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
  finally:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))

  return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
