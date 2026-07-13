#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# alpamayo local patch: optional FlashInfer RMSNorm disable"
PATCH = '''\

{marker}
import os as _alpamayo_os
_ALPAMAYO_DISABLE_FLASHINFER_NORM = _alpamayo_os.environ.get("TLLM_DISABLE_FLASHINFER_NORM", "0") == "1"
if _ALPAMAYO_DISABLE_FLASHINFER_NORM:
    IS_FLASHINFER_AVAILABLE = False
'''.format(marker=MARKER)


def patch_file(path: Path) -> None:
  text = path.read_text()
  if MARKER in text:
    print(f"already patched: {path}")
    return

  needle = "from ..utils import Fp4QuantizedTensor\n"
  if needle not in text:
    raise SystemExit(f"could not find insertion point in {path}")

  backup = path.with_suffix(path.suffix + ".alpamayo_flashinfer_norm_patch.bak")
  if not backup.exists():
    backup.write_text(text)

  path.write_text(text.replace(needle, needle + PATCH, 1))
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--rms-norm-path", required=True, type=Path)
  args = parser.parse_args()
  patch_file(args.rms_norm_path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
