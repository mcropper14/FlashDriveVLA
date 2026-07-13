#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _extract_pytorch(report: dict[str, Any]) -> dict[str, Any]:
  inferences = report.get("inferences") or []
  if not inferences:
    raise ValueError("PyTorch report has no inferences")
  inference = inferences[0]
  return {
    "source": "pytorch",
    "generated_token_ids": inference.get("generated_new_token_ids") or [],
    "generated_text": inference.get("generated_new_text", ""),
    "elapsed_ms": inference.get("elapsed_ms"),
    "language_summary": inference.get("language_summary") or report.get("language_summary_all"),
    "visual_summary": inference.get("visual_summary") or report.get("visual_summary_all"),
    "mrope_position_deltas": (
      (report.get("visual_prefix_export") or {}).get("mrope_position_deltas")
    ),
  }


def _extract_trt(report: dict[str, Any]) -> dict[str, Any]:
  outputs = report.get("outputs") or []
  if not outputs:
    raise ValueError("TensorRT report has no outputs")
  output = outputs[0]
  return {
    "source": "tensorrt_llm",
    "generated_token_ids": output.get("generated_token_ids") or [],
    "generated_text": output.get("generated_text", ""),
    "mean_latency_ms": report.get("mean_latency_ms"),
    "min_latency_ms": report.get("min_latency_ms"),
    "max_latency_ms": report.get("max_latency_ms"),
    "latency_ms": report.get("latency_ms"),
    "input_tokens": report.get("input_tokens"),
    "mrope_position_deltas": report.get("mrope_position_deltas_loaded"),
  }


def _common_prefix_len(left: list[int], right: list[int]) -> int:
  count = 0
  for lhs, rhs in zip(left, right):
    if lhs != rhs:
      break
    count += 1
  return count


def _truncate_after(tokens: list[int], stop_token_id: int | None) -> list[int]:
  if stop_token_id is None:
    return tokens
  try:
    stop_index = tokens.index(stop_token_id)
  except ValueError:
    return tokens
  return tokens[: stop_index + 1]


def compare(pytorch_report: Path, trt_report: Path, stop_token_id: int | None) -> dict[str, Any]:
  pytorch = _extract_pytorch(_load_json(pytorch_report))
  trt = _extract_trt(_load_json(trt_report))
  pytorch_tokens = [int(token) for token in pytorch["generated_token_ids"]]
  trt_tokens = [int(token) for token in trt["generated_token_ids"]]
  compare_pytorch_tokens = _truncate_after(pytorch_tokens, stop_token_id)
  compare_trt_tokens = _truncate_after(trt_tokens, stop_token_id)
  common = _common_prefix_len(compare_pytorch_tokens, compare_trt_tokens)
  matched = compare_pytorch_tokens == compare_trt_tokens
  mismatch = None
  if not matched:
    mismatch = {
      "index": common,
      "pytorch_token": compare_pytorch_tokens[common] if common < len(compare_pytorch_tokens) else None,
      "trt_token": compare_trt_tokens[common] if common < len(compare_trt_tokens) else None,
    }

  pytorch_language_sum = None
  if isinstance(pytorch.get("language_summary"), dict):
    pytorch_language_sum = pytorch["language_summary"].get("sum_ms")

  speedup_vs_pytorch_language = None
  if pytorch_language_sum and trt.get("mean_latency_ms"):
    speedup_vs_pytorch_language = round(float(pytorch_language_sum) / float(trt["mean_latency_ms"]), 4)

  return {
    "created_at_unix": time.time(),
    "pytorch_report": str(pytorch_report),
    "trt_report": str(trt_report),
    "matched": matched,
    "stop_token_id": stop_token_id,
    "common_prefix_tokens": common,
    "pytorch_generated_tokens": len(pytorch_tokens),
    "trt_generated_tokens": len(trt_tokens),
    "pytorch_compared_tokens": len(compare_pytorch_tokens),
    "trt_compared_tokens": len(compare_trt_tokens),
    "raw_tokens_match": pytorch_tokens == trt_tokens,
    "first_mismatch": mismatch,
    "speedup_vs_pytorch_language_sum": speedup_vs_pytorch_language,
    "pytorch": pytorch,
    "tensorrt_llm": trt,
    "conclusion": (
      (
        "TensorRT output matches PyTorch through the configured stop token. "
        "Tokens after the stop token are outside the compared VLM reasoning contract."
      )
      if matched and stop_token_id is not None
      else (
        "TensorRT output matches PyTorch for the full generated token sequence."
        if matched
        else "TensorRT is faster, but token parity fails. This decoder path is not an acceptable Alpamayo replacement until the mismatch is resolved."
      )
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pytorch-report", type=Path, required=True)
  parser.add_argument("--trt-report", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--stop-token-id", type=int)
  args = parser.parse_args()

  payload = compare(args.pytorch_report, args.trt_report, args.stop_token_id)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
