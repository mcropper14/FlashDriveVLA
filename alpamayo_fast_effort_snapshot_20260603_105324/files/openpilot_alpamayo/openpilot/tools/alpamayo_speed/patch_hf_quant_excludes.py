#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def patch_quant_config(path: Path, excludes: list[str]) -> None:
  data = json.loads(path.read_text())
  quant = data.setdefault("quantization", {})
  current = quant.setdefault("exclude_modules", [])
  changed = False
  for item in excludes:
    if item not in current:
      current.append(item)
      changed = True
  if not changed:
    print(f"already patched: {path}")
    return

  backup = path.with_suffix(path.suffix + ".alpamayo_excludes_patch.bak")
  if not backup.exists():
    backup.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Add excluded modules to a HF quant config JSON.")
  parser.add_argument("--quant-config", required=True, type=Path)
  parser.add_argument("--exclude", action="append", default=[])
  args = parser.parse_args()
  patch_quant_config(args.quant_config, args.exclude)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
