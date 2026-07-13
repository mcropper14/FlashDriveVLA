#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER_INIT = "# alpamayo local patch: classic engine Qwen3VL map"
MARKER_CONFIG = "# alpamayo local patch: classic engine Qwen3VL text config"
MARKER_MODEL = "# alpamayo local patch: classic engine Qwen3VL weight prefix"
MARKER_BUILDER = "# alpamayo local patch: classic engine Qwen3VL mrope inputs"
MARKER_BUILDER_USE_MROPE = "# alpamayo local patch: classic engine Qwen3VL build_config.use_mrope"
MARKER_COMMAND_BUILD = "# alpamayo local patch: classic engine Qwen3VL command use_mrope"


def _backup_once(path: Path, suffix: str) -> None:
  backup = path.with_suffix(path.suffix + suffix)
  if not backup.exists():
    backup.write_text(path.read_text())


def patch_models_init(path: Path) -> None:
  text = path.read_text()
  if MARKER_INIT in text:
    print(f"already patched: {path}")
    return

  needle = "    'Qwen3ForCausalLM': QWenForCausalLM,\n"
  if needle not in text:
    raise SystemExit(f"could not find Qwen3ForCausalLM mapping in {path}")
  replacement = needle + f"    'Qwen3VLForConditionalGeneration': QWenForCausalLM,  {MARKER_INIT}\n"
  _backup_once(path, ".alpamayo_qwen3vl_engine_patch.bak")
  path.write_text(text.replace(needle, replacement, 1))
  print(f"patched: {path}")


def patch_qwen_config(path: Path) -> None:
  text = path.read_text()
  if MARKER_CONFIG in text:
    print(f"already patched: {path}")
    return

  needle = """        if qwen_type == 'qwen2_audio':
            hf_config = hf_config.text_config
            hf_config.architectures = ['Qwen2ForCausalLM']

        valid_types = ('qwen', 'qwen2', 'qwen2_moe', 'qwen2_llava_onevision',
"""
  replacement = """        if qwen_type == 'qwen2_audio':
            hf_config = hf_config.text_config
            hf_config.architectures = ['Qwen2ForCausalLM']

        qwen3_vl_text_from_mm = qwen_type in ('qwen3_vl', 'qwen3_vl_text')  # alpamayo local patch: classic engine Qwen3VL text config
        if qwen3_vl_text_from_mm:
            hf_config = hf_config.text_config if hasattr(hf_config, 'text_config') else hf_config
            hf_config.architectures = ['Qwen3ForCausalLM']
            qwen_type = 'qwen3'

        valid_types = ('qwen', 'qwen2', 'qwen2_moe', 'qwen2_llava_onevision',
"""
  if needle not in text:
    raise SystemExit(f"could not find qwen_type insertion point in {path}")
  text = text.replace(needle, replacement, 1)

  needle = """        if qwen_type == 'qwen2_vl':
            pe_type = 'mrope'
            rotary_embedding_percentage = getattr(hf_config, 'rotary_pct', 1.0)
            rotary_embedding_dim = getattr(
                hf_config, 'rotary_dim',
                int(hf_config.hidden_size / hf_config.num_attention_heads *
                    rotary_embedding_percentage))
            rotary_scaling['type'] = 'mrope'
        else:
            pe_type = 'rope_gpt_neox'
            rotary_embedding_dim = None
"""
  replacement = """        if qwen_type == 'qwen2_vl' or qwen3_vl_text_from_mm:
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
  if needle not in text:
    raise SystemExit(f"could not find qwen mrope insertion point in {path}")
  text = text.replace(needle, replacement, 1)

  needle = """            qwen_type=qwen_type,
            moe_intermediate_size=moe_intermediate_size,
"""
  replacement = """            qwen_type=qwen_type,
            qwen3_vl_text_from_mm=qwen3_vl_text_from_mm,
            moe_intermediate_size=moe_intermediate_size,
"""
  if needle not in text:
    raise SystemExit(f"could not find qwen config return insertion point in {path}")
  text = text.replace(needle, replacement, 1)

  _backup_once(path, ".alpamayo_qwen3vl_engine_patch.bak")
  path.write_text(text)
  print(f"patched: {path}")


def patch_qwen_model(path: Path) -> None:
  text = path.read_text()
  if MARKER_MODEL in text:
    print(f"already patched: {path}")
    return

  needle = """            elif config.qwen_type == "qwen3":
                custom_dict = {
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
"""
  replacement = """            elif config.qwen_type == "qwen3":
                custom_dict = {
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
                if getattr(config, "qwen3_vl_text_from_mm", False):  # alpamayo local patch: classic engine Qwen3VL weight prefix
                    custom_dict["transformer"] = "model.language_model"
"""
  if needle not in text:
    raise SystemExit(f"could not find qwen3 custom_dict block in {path}")

  _backup_once(path, ".alpamayo_qwen3vl_engine_patch.bak")
  path.write_text(text.replace(needle, replacement, 1))
  print(f"patched: {path}")


def patch_builder(path: Path) -> None:
  text = path.read_text()
  changed = False
  if MARKER_BUILDER in text and MARKER_BUILDER_USE_MROPE in text:
    print(f"already patched: {path}")
    return

  if MARKER_BUILDER not in text:
    needle = """        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel":
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
    replacement = """        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel" or getattr(model.config, "qwen3_vl_text_from_mm", False):  # alpamayo local patch: classic engine Qwen3VL mrope inputs
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
    if needle not in text:
      raise SystemExit(f"could not find builder mrope block in {path}")
    text = text.replace(needle, replacement, 1)
    changed = True

  if MARKER_BUILDER_USE_MROPE not in text:
    needle = """        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel" or getattr(model.config, "qwen3_vl_text_from_mm", False):  # alpamayo local patch: classic engine Qwen3VL mrope inputs
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
    replacement = """        if model.config.architecture == "Qwen2VLForConditionalGeneration" or model.config.architecture == "Qwen2VLModel" or getattr(model.config, "qwen3_vl_text_from_mm", False):  # alpamayo local patch: classic engine Qwen3VL mrope inputs
            build_config.use_mrope = True  # alpamayo local patch: classic engine Qwen3VL build_config.use_mrope
            prepare_input_args[
                'mrope_rotary_cos_sin_size'] = model.config.max_position_embeddings * model.config.rotary_embedding_dim
"""
    if needle not in text:
      raise SystemExit(f"could not find patched builder mrope block in {path}")
    text = text.replace(needle, replacement, 1)
    changed = True

  if changed:
    _backup_once(path, ".alpamayo_qwen3vl_engine_patch.bak")
    path.write_text(text)
  print(f"patched: {path}")


def patch_command_build(path: Path) -> None:
  text = path.read_text()
  if MARKER_COMMAND_BUILD in text:
    print(f"already patched: {path}")
    return

  needle = """            use_mrope=getattr(model_config, "qwen_type", None) == "qwen2_vl",
"""
  replacement = """            use_mrope=(getattr(model_config, "qwen_type", None) == "qwen2_vl" or getattr(model_config, "qwen3_vl_text_from_mm", False)),  # alpamayo local patch: classic engine Qwen3VL command use_mrope
"""
  if needle not in text:
    raise SystemExit(f"could not find command build use_mrope block in {path}")
  _backup_once(path, ".alpamayo_qwen3vl_engine_patch.bak")
  path.write_text(text.replace(needle, replacement, 1))
  print(f"patched: {path}")


def main() -> int:
  parser = argparse.ArgumentParser(description="Patch isolated TensorRT-LLM venv for classic-engine Qwen3-VL text decoder probing.")
  parser.add_argument("--trtllm-root", required=True, type=Path)
  args = parser.parse_args()

  patch_models_init(args.trtllm_root / "models" / "__init__.py")
  patch_qwen_config(args.trtllm_root / "models" / "qwen" / "config.py")
  patch_qwen_model(args.trtllm_root / "models" / "qwen" / "model.py")
  patch_builder(args.trtllm_root / "builder.py")
  patch_command_build(args.trtllm_root / "commands" / "build.py")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
