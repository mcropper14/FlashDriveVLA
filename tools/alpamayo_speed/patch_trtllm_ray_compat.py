#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


OLD = """from ray.util.placement_group import (PlacementGroupSchedulingStrategy,
                                      get_current_placement_group,
                                      placement_group)
"""

NEW = """try:
    from ray.util.placement_group import PlacementGroupSchedulingStrategy
except ImportError:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import (get_current_placement_group,
                                      placement_group)
"""


def main() -> int:
  parser = argparse.ArgumentParser(description="Patch TensorRT-LLM ray_executor.py for newer Ray placement strategy import location.")
  parser.add_argument("--ray-executor-path", required=True, type=Path)
  args = parser.parse_args()

  path = args.ray_executor_path
  text = path.read_text()
  if NEW in text:
    print("already_patched")
    return 0
  if OLD not in text:
    raise RuntimeError(f"expected Ray import block not found in {path}")
  backup = path.with_suffix(path.suffix + ".alpamayo_ray_patch.bak")
  if not backup.exists():
    backup.write_text(text)
  path.write_text(text.replace(OLD, NEW))
  print("patched")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
