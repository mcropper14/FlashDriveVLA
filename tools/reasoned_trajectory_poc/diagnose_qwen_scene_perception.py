#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "vlm" / "qwen2_5_vl_3b_instruct"
DEFAULT_IMAGE_DIR = REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_novel_scene_probe_prompt2"

PROMPTS = {
  "freeform": "Describe visible road hazards in this driving scene. Mention cones, barriers, pedestrians, red lights, or stop signs if visible. Be concise.",
  "forced_labels": "Which labels are visibly present? Choose any from: cones, barrier, pedestrian, red_stop_light, stop_sign, none. Output only labels and brief evidence.",
  "rtp_no_example": """Return exactly this RTPv1 program format and no prose:
RTPv1
scene=<nominal|construction_merge|blocked_lane|pedestrian_crosswalk|stop_sign|unknown>
evidence=[token]
meta=<BASE|BIAS_LEFT_AND_SLOW|BIAS_RIGHT_AND_SLOW|YIELD|STOP|SLOW>
branch=<base|C0|C1|C2|C3|C4>
lat_bias_m=<float from -0.8 to 0.8>
speed_cap_mps=<float, percentage like 25%, or none>
stop_s=<float or none>
avoid=[token]
weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]
confidence=<float from 0.0 to 1.0>

If cones or barriers are visible near the green corridor, do not output BASE.
If a pedestrian is visible near the green corridor, do not output BASE.
If a red stop light or STOP sign is visible ahead, do not output BASE.
Compile the program for this scene.""",
}

CROPS = {
  "construction": (0, 95, 512, 310),
  "pedestrian": (170, 95, 342, 282),
  "stop_sign": (170, 55, 345, 250),
}


def load_model(model_dir: Path):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
  )
  model.eval()
  return processor, model


def generate(processor, model, image: Image.Image, prompt_text: str, max_new_tokens: int) -> dict:
  messages = [{
    "role": "user",
    "content": [
      {"type": "image", "image": image},
      {"type": "text", "text": prompt_text},
    ],
  }]
  start = time.perf_counter()
  prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
  torch.cuda.synchronize()
  prefill_ms = (time.perf_counter() - start) * 1000.0

  decode_start = time.perf_counter()
  with torch.inference_mode():
    output_ids = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      temperature=None,
      top_p=None,
      use_cache=True,
    )
  torch.cuda.synchronize()
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
  text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
  return {
    "text": text,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "generated_token_count": int(output_ids.shape[-1] - inputs.input_ids.shape[-1]),
  }


def prepare_images(image_dir: Path, out_dir: Path, resize: int) -> dict[str, dict[str, Path]]:
  out_dir.mkdir(parents=True, exist_ok=True)
  prepared: dict[str, dict[str, Path]] = {}
  for scene in ("construction", "pedestrian", "stop_sign"):
    src = image_dir / f"{scene}_vlm_input.png"
    image = Image.open(src).convert("RGB")
    full = image.copy()
    full.thumbnail((resize, resize), Image.Resampling.BILINEAR)
    full_path = out_dir / f"{scene}_full_{full.width}x{full.height}.png"
    full.save(full_path)

    crop = image.crop(CROPS[scene])
    crop = crop.resize((resize, resize), Image.Resampling.BICUBIC)
    crop_path = out_dir / f"{scene}_crop_{resize}.png"
    crop.save(crop_path)
    prepared[scene] = {"full": full_path, "crop": crop_path}
  return prepared


def main() -> None:
  parser = argparse.ArgumentParser(description="Diagnose why Qwen emits BASE for novel scene boards.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
  parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc" / "qwen_scene_diagnosis")
  parser.add_argument("--resize", type=int, default=384)
  parser.add_argument("--max-new-tokens", type=int, default=96)
  args = parser.parse_args()

  prepared = prepare_images(args.image_dir, args.out, args.resize)
  processor, model = load_model(args.model_dir)
  rows = []
  for scene, variants in prepared.items():
    for variant, image_path in variants.items():
      image = Image.open(image_path).convert("RGB")
      for prompt_name, prompt_text in PROMPTS.items():
        result = generate(processor, model, image, prompt_text, args.max_new_tokens)
        rows.append({
          "scene": scene,
          "variant": variant,
          "image_path": str(image_path),
          "prompt": prompt_name,
          **result,
        })
        print(json.dumps(rows[-1], separators=(",", ":")), flush=True)

  summary = {"rows": rows}
  (args.out / "diagnosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(f"artifacts={args.out}")


if __name__ == "__main__":
  main()
