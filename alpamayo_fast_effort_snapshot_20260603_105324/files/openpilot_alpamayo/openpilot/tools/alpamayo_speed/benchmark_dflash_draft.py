#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any


def _sync(torch_mod: Any) -> None:
  if torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


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


def _load_dflash_model(model_dir: Path, *, dtype: Any, attn_implementation: str) -> tuple[Any, str, dict[str, Any]]:
  from dflash.model import DFlashDraftModel

  try:
    return (
      DFlashDraftModel.from_pretrained(
        str(model_dir),
        attn_implementation=attn_implementation,
        dtype=dtype,
      ),
      "from_pretrained",
      {},
    )
  except AttributeError as exc:
    if "dflash_config" not in str(exc):
      raise

  import torch
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
  return model.to(dtype=dtype), "manual_config_shim", {"manual_config_keys": sorted(raw_config.keys())}


def _run_shape(
  torch_mod: Any,
  model: Any,
  *,
  device: str,
  dtype: Any,
  batch_size: int,
  context_len: int,
  block_size: int,
  warmup: int,
  iters: int,
  use_cache: bool,
) -> dict[str, Any]:
  hidden = int(model.config.hidden_size)
  target_layers = list(getattr(model, "target_layer_ids", []))
  target_width = hidden * len(target_layers)
  target_hidden = torch_mod.randn(batch_size, context_len, target_width, device=device, dtype=dtype)
  noise_embedding = torch_mod.randn(batch_size, block_size, hidden, device=device, dtype=dtype)
  position_ids = torch_mod.arange(context_len + block_size, device=device, dtype=torch_mod.long).unsqueeze(0).expand(batch_size, -1)

  latencies_ms: list[float] = []
  with torch_mod.inference_mode(), torch_mod.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
    for idx in range(warmup + iters):
      if use_cache:
        from transformers import DynamicCache

        past_key_values = DynamicCache()
      else:
        past_key_values = None
      _sync(torch_mod)
      start = time.perf_counter()
      out = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        is_causal=False,
      )
      _sync(torch_mod)
      if idx >= warmup:
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
      if tuple(out.shape) != (batch_size, block_size, hidden):
        raise RuntimeError(f"unexpected output shape {tuple(out.shape)}")

  return {
    "batch_size": batch_size,
    "context_len": context_len,
    "block_size": block_size,
    "target_layer_count": len(target_layers),
    "target_hidden_shape": [batch_size, context_len, target_width],
    "noise_embedding_shape": [batch_size, block_size, hidden],
    "use_cache": use_cache,
    "latency_ms": [round(v, 3) for v in latencies_ms],
    "summary": _summary(latencies_ms),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark local FlashDrive/DFlash draft weights without loading the target VLA.")
  parser.add_argument("--draft-model", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
  parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="flash_attention_2")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--context-lens", type=int, nargs="+", default=None)
  parser.add_argument("--block-size", type=int)
  parser.add_argument("--use-cache", action="store_true")
  args = parser.parse_args()

  import torch
  dtype = getattr(torch, args.dtype)
  report: dict[str, Any] = {
    "created_at_unix": time.time(),
    "draft_model": str(args.draft_model),
    "device": args.device,
    "dtype": args.dtype,
    "attn_implementation": args.attn_implementation,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_initial": _gpu_stats(torch),
  }

  _sync(torch)
  start = time.perf_counter()
  model, load_mode, load_extra = _load_dflash_model(
    args.draft_model,
    dtype=dtype,
    attn_implementation=args.attn_implementation,
  )
  model = model.to(args.device).eval()
  _sync(torch)
  report["load_mode"] = load_mode
  report.update(load_extra)
  report["load_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
  report["gpu_post_load"] = _gpu_stats(torch)
  report["config"] = {
    "hidden_size": int(model.config.hidden_size),
    "num_hidden_layers": int(model.config.num_hidden_layers),
    "num_attention_heads": int(model.config.num_attention_heads),
    "num_key_value_heads": int(model.config.num_key_value_heads),
    "intermediate_size": int(model.config.intermediate_size),
    "block_size": int(model.block_size),
    "context_len": int(getattr(model.config, "context_len", 0) or getattr(model.config, "dflash_config", {}).get("context_len", 0) or 0),
    "target_layer_ids": [int(x) for x in model.target_layer_ids],
    "mask_token_id": int(model.mask_token_id),
  }

  context_lens = args.context_lens
  if context_lens is None:
    config_context = int(report["config"]["context_len"]) or 8
    context_lens = [config_context, 841]
  block_size = args.block_size or int(model.block_size)
  report["runs"] = [
    _run_shape(
      torch,
      model,
      device=args.device,
      dtype=dtype,
      batch_size=args.batch_size,
      context_len=context_len,
      block_size=block_size,
      warmup=args.warmup,
      iters=args.iters,
      use_cache=args.use_cache,
    )
    for context_len in context_lens
  ]
  report["gpu_post_benchmark"] = _gpu_stats(torch)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
