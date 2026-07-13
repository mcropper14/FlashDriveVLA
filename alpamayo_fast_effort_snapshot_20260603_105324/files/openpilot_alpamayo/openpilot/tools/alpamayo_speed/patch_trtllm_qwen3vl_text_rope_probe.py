#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER_CONFIG = "# alpamayo local patch: Qwen3VL text-only normal RoPE probe"
MARKER_BUILDER = "# alpamayo local patch: Qwen3VL no mrope input for text-only probe"


def _backup_once(path: Path, suffix: str) -> None:
  backup = path.with_suffix(path.suffix + suffix)
  if not backup.exists():
    backup.write_text(path.read_text())


def patch_qwen_config(path: Path) -> None:
  text = path.read_text()
  if MARKER_CONFIG in text:
    print(f"already patched: {path}")
    return
  old = """        if qwen_type == 'qwen2_vl' or qwen3_vl_text_from_mm:
            pe_type = 'mrope'
            rotary_embedding_percentage = getattr(hf_config, 'rotary_pct', 1.0)
            rotary_embedding_dim = getattr(
                hf_config, 'rotary_dim',
                int(hf_config.hidden_size / hf_config.num_attention_heads *
                    rotary_embedding_percentage))
            rotary_scaling = rotary_scaling or {}
            rotary_scaling['type'] = 'mrope'
        else:
            pe_type = 'rope_gpt_neox'
            rotary_embedding_dim = None
"""
  new = f"""        if qwen_type == 'qwen2_vl':
            pe_type = 'mrope'
            rotary_embedding_percentage = getattr(hf_config, 'rotary_pct', 1.0)
            rotary_embedding_dim = getattr(
                hf_config, 'rotary_dim',
                int(hf_config.hidden_size / hf_config.num_attention_heads *
                    rotary_embedding_percentage))
            rotary_scaling = rotary_scaling or {{}}
            rotary_scaling['type'] = 'mrope'
        else:
            pe_type = 'rope_gpt_neox'  {MARKER_CONFIG}
            rotary_embedding_dim = None
            rotary_scaling = None
"""
  if old not in text:
    raise RuntimeError(f"target Qwen MRoPE block not found in {path}")
  _backup_once(path, ".alpamayo_qwen3vl_text_rope_probe.bak")
  path.write_text(text.replace(old, new, 1))
  print(f"patched: {path}")


def patch_builder(path: Path) -> None:
  text = path.read_text()
  if MARKER_BUILDER in text:
    print(f"already patched: {path}")
    return
  old = """        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel" or getattr(model.config, "qwen3_vl_text_from_mm", False):  # alpamayo local patch: classic engine Qwen3VL mrope inputs
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
  new = f"""        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel":  {MARKER_BUILDER}
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
  if old not in text:
    raise RuntimeError(f"target builder MRoPE block not found in {path}")
  _backup_once(path, ".alpamayo_qwen3vl_text_rope_probe.bak")
  path.write_text(text.replace(old, new, 1))
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Patch isolated TensorRT-LLM venv for a Qwen3-VL text-only normal-RoPE timing probe.")
  parser.add_argument("--trtllm-root", required=True, type=Path)
  parser.add_argument("--restore", action="store_true")
  args = parser.parse_args()

  if args.restore:
    for path in (
      args.trtllm_root / "models" / "qwen" / "config.py",
      args.trtllm_root / "builder.py",
    ):
      backup = path.with_suffix(path.suffix + ".alpamayo_qwen3vl_text_rope_probe.bak")
      if not backup.exists():
        raise RuntimeError(f"restore backup not found: {backup}")
      path.write_text(backup.read_text())
      print(f"restored: {path}")
    return 0

  patch_qwen_config(args.trtllm_root / "models" / "qwen" / "config.py")
  patch_builder(args.trtllm_root / "builder.py")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
