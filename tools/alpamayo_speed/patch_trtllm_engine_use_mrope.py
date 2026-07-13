#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description="Set build_config.use_mrope=true in a TensorRT-LLM engine config.")
  parser.add_argument("--engine-dir", required=True, type=Path)
  parser.add_argument("--require-qwen3-vl", action="store_true")
  args = parser.parse_args()

  config_path = args.engine_dir / "config.json"
  with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

  pretrained = config.get("pretrained_config", {})
  if args.require_qwen3_vl and not pretrained.get("qwen3_vl_text_from_mm", False):
    raise RuntimeError(f"refusing to patch non-Qwen3-VL engine config: {config_path}")

  build_config = config.setdefault("build_config", {})
  old = build_config.get("use_mrope")
  build_config["use_mrope"] = True

  backup_path = config_path.with_suffix(config_path.suffix + ".use_mrope_false.bak")
  if not backup_path.exists():
    backup_path.write_text(json.dumps({**config, "build_config": {**build_config, "use_mrope": old}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({
    "config": str(config_path),
    "old_use_mrope": old,
    "new_use_mrope": True,
    "qwen3_vl_text_from_mm": pretrained.get("qwen3_vl_text_from_mm"),
    "position_embedding_type": pretrained.get("position_embedding_type"),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
