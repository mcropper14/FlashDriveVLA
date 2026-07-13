#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# alpamayo_mrope_empty_multimodal_patch"


def main() -> int:
  parser = argparse.ArgumentParser(description="Patch TensorRT-LLM ModelRunnerCpp to route MRoPE requests through multimodal metadata.")
  parser.add_argument(
    "--site-packages",
    default="/mnt/g/alpamayo1.5/trtllm_venv/lib/python3.12/site-packages",
    type=Path,
  )
  parser.add_argument("--restore", action="store_true")
  args = parser.parse_args()

  path = args.site_packages / "tensorrt_llm" / "runtime" / "model_runner_cpp.py"
  backup = path.with_suffix(path.suffix + ".alpamayo_mrope_multimodal_patch.bak")
  if args.restore:
    if not backup.exists():
      raise RuntimeError(f"restore backup not found: {backup}")
    path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"restored: {path}")
    return 0

  text = path.read_text(encoding="utf-8")
  if MARKER in text:
    print(f"already patched: {path}")
    return 0

  old = """                cross_attention_mask=cross_attention_masks[i].contiguous() if
                (cross_attention_masks is not None
                 and cross_attention_masks[i] is not None) else None,
                max_tokens=max_new_tokens,
"""
  new = f"""                cross_attention_mask=cross_attention_masks[i].contiguous() if
                (cross_attention_masks is not None
                 and cross_attention_masks[i] is not None) else None,
                multimodal_input=(trtllm.MultimodalInput([[]], [], []) if mrope_config is not None else None),  {MARKER}
                max_tokens=max_new_tokens,
"""
  if old not in text:
    raise RuntimeError(f"target block not found in {path}")
  if not backup.exists():
    backup.write_text(text, encoding="utf-8")
  path.write_text(text.replace(old, new, 1), encoding="utf-8")
  print(f"patched: {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
