#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# alpamayo local patch: missing weight context"
PATCH = '''\
            {marker}
            if not (getattr(sub_module, "is_qkv", False) and not tllm_key.endswith("weight")) and (v is None or (isinstance(v, list) and any(item is None for item in v))):
                raise KeyError(f"missing external tensor for {{tllm_key}} -> {{external_key}}")
'''.format(marker=MARKER)


def patch_file(path: Path) -> None:
  text = path.read_text()
  if MARKER in text:
    old = '''\
            {marker}
            if v is None or (isinstance(v, list) and any(item is None for item in v)):
                raise KeyError(f"missing external tensor for {{tllm_key}} -> {{external_key}}")
'''.format(marker=MARKER)
    if old in text:
      backup = path.with_suffix(path.suffix + ".alpamayo_weight_context_patch.bak")
      if not backup.exists():
        backup.write_text(text)
      path.write_text(text.replace(old, PATCH, 1))
      print(f"updated patch: {path}")
    else:
      print(f"already patched: {path}")
    return

  needle = """        else:
            postprocess_kwargs = {"config": self.model.config}
"""
  if needle not in text:
    raise SystemExit(f"could not find postprocess block in {path}")

  backup = path.with_suffix(path.suffix + ".alpamayo_weight_context_patch.bak")
  if not backup.exists():
    backup.write_text(text)

  path.write_text(text.replace(needle, "        else:\n" + PATCH + "            postprocess_kwargs = {\"config\": self.model.config}\n", 1))
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--loader-path", required=True, type=Path)
  args = parser.parse_args()
  patch_file(args.loader_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
