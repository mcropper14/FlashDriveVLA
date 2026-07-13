#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FRAME_SUFFIXES = [
  "frame_id=120000 ego_speed_mps=12.4 curvature=0.002 blinker=none lead_s=42.0 lead_v=11.8 signal=none hazards=nominal",
  "frame_id=120001 ego_speed_mps=12.5 curvature=0.002 blinker=none lead_s=39.2 lead_v=10.9 signal=none hazards=cones_right_s30_55",
  "frame_id=120002 ego_speed_mps=12.4 curvature=0.003 blinker=none lead_s=35.5 lead_v=10.2 signal=red_ahead_s62 hazards=cones_right_s28_54",
  "frame_id=120003 ego_speed_mps=12.3 curvature=0.003 blinker=none lead_s=31.8 lead_v=9.7 signal=green_ahead_s58 hazards=ped_right_edge_moving_left",
  "frame_id=120004 ego_speed_mps=12.2 curvature=0.004 blinker=none lead_s=28.4 lead_v=9.1 signal=green_ahead_s53 hazards=vehicle_cut_in_left",
  "frame_id=120005 ego_speed_mps=12.1 curvature=0.004 blinker=right lead_s=25.1 lead_v=8.8 signal=none hazards=lane_split_right_nav",
]


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="\n") as f:
    for record in records:
      f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
  parser = argparse.ArgumentParser(description="Create prompt-sequence JSONL files for TensorRT-LLM KV reuse probes.")
  parser.add_argument("--base-prompt", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  args = parser.parse_args()

  base = args.base_prompt.read_text(encoding="utf-8").strip()
  tail_records = []
  head_records = []
  for i, suffix in enumerate(FRAME_SUFFIXES):
    tail_records.append({
      "name": f"tail_frame_{i:02d}",
      "prompt": f"{base}\n\nDynamic frame/state block:\n{suffix}\nOutput one compact Alpamayo action-token summary.",
    })
    head_records.append({
      "name": f"head_frame_{i:02d}",
      "prompt": f"Dynamic frame/state block:\n{suffix}\n\n{base}\nOutput one compact Alpamayo action-token summary.",
    })

  _write_jsonl(args.out_dir / "mrope_sequence_static_prefix_dynamic_tail_2026-05-26.jsonl", tail_records)
  _write_jsonl(args.out_dir / "mrope_sequence_dynamic_head_static_suffix_2026-05-26.jsonl", head_records)
  print(json.dumps({
    "base_prompt": str(args.base_prompt),
    "out_dir": str(args.out_dir),
    "tail_records": len(tail_records),
    "head_records": len(head_records),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
