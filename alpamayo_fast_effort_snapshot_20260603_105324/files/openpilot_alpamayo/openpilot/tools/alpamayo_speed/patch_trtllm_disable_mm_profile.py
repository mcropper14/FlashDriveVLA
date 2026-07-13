#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# alpamayo local patch: optional multimodal profiling disable"
PATCH = '''\
        {marker}
        if os.environ.get("TLLM_DISABLE_MM_PROFILE", "0") == "1":
            return requests

'''.format(marker=MARKER)


def patch_file(path: Path) -> None:
  text = path.read_text()
  if MARKER in text:
    print(f"already patched: {path}")
    return

  needle = "        requests = []\n"
  start = text.find("    def _create_dummy_mm_context_request(")
  if start < 0:
    raise SystemExit(f"could not find _create_dummy_mm_context_request in {path}")
  insert = text.find(needle, start)
  if insert < 0:
    raise SystemExit(f"could not find insertion point in {path}")
  insert += len(needle)

  backup = path.with_suffix(path.suffix + ".alpamayo_mm_profile_patch.bak")
  if not backup.exists():
    backup.write_text(text)

  path.write_text(text[:insert] + PATCH + text[insert:])
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--util-path", required=True, type=Path)
  args = parser.parse_args()
  patch_file(args.util_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
