#!/usr/bin/env python3
import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean


RUNTIME_RE = re.compile(r"runtime_profile_(\d+)\s+(\{.*\})")
INFER_RE = re.compile(r"infer_seconds_(\d+)\s+([0-9.]+)")


def percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
  return ordered[idx]


def summarize_values(values: list[float]) -> dict[str, float]:
  return {
    "count": len(values),
    "mean": round(mean(values), 6) if values else 0.0,
    "p50": round(percentile(values, 50), 6),
    "p95": round(percentile(values, 95), 6),
    "p99": round(percentile(values, 99), 6),
    "min": round(min(values), 6) if values else 0.0,
    "max": round(max(values), 6) if values else 0.0,
  }


def parse_log(path: Path) -> dict:
  infer_seconds: dict[int, float] = {}
  profiles: dict[int, dict] = {}
  metadata: dict[str, str] = {}

  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    clean = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line).strip()
    infer_match = INFER_RE.search(clean)
    if infer_match:
      infer_seconds[int(infer_match.group(1))] = float(infer_match.group(2))
      continue
    runtime_match = RUNTIME_RE.search(clean)
    if runtime_match:
      profiles[int(runtime_match.group(1))] = ast.literal_eval(runtime_match.group(2))
      continue
    if " " in clean:
      key, value = clean.split(" ", 1)
      if key in {
        "camera_mode",
        "num_frames",
        "min_pixels",
        "max_pixels",
        "device_map_mode",
        "split_index",
        "diffusion_steps",
        "attn_implementation",
        "expert_attn_implementation",
        "reasoning_mode",
        "quantization",
        "do_sample",
        "manual_generate",
      }:
        metadata[key] = value.strip()

  hot_indices = sorted(idx for idx in infer_seconds if idx > 0)
  cold_indices = sorted(idx for idx in infer_seconds if idx == 0)
  hot_profiles = [profiles[idx] for idx in hot_indices if idx in profiles]

  stage_keys = sorted({key for profile in hot_profiles for key in profile if key.endswith("_seconds")})
  stage_summary = {
    key: summarize_values([float(profile[key]) for profile in hot_profiles if key in profile])
    for key in stage_keys
  }

  return {
    "path": str(path),
    "metadata": metadata,
    "cold_infer_seconds": [infer_seconds[idx] for idx in cold_indices],
    "hot_infer_seconds": [infer_seconds[idx] for idx in hot_indices],
    "hot_infer_summary": summarize_values([infer_seconds[idx] for idx in hot_indices]),
    "hot_stage_summary": stage_summary,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("logs", nargs="+", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  payload = {"logs": [parse_log(path) for path in args.logs]}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
