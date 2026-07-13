#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "vlm" / "smolvlm2_500m_video_instruct"

SYSTEM_PROMPT = """Compile a real-time driving RTPv1 program from the image.
Output exactly these lines and no other text:
RTPv1
scene=<lower_snake_case>
evidence=[<token>,<token>]
meta=<BASE|BIAS_LEFT|BIAS_RIGHT|BIAS_LEFT_AND_SLOW|BIAS_RIGHT_AND_SLOW|SLOW|YIELD|STOP|TAKE_LEFT_BRANCH|TAKE_RIGHT_BRANCH|REJECT_BASE|OCCLUSION_CAUTION|EMERGENCY_CAUTION>
branch=<base|C0|C1|C2|C3|C4>
lat_bias_m=<float from -0.8 to 0.8>
speed_cap_mps=<float, percentage like 25%, or none>
stop_s=<float or none>
avoid=[<token>]
weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]
confidence=<float from 0.0 to 1.0>
Prefer BASE unless visible evidence requires a bounded constraint."""


def _load(model_dir: Path):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
  )
  model.eval()
  return processor, model


def _image_from_payload(payload: dict) -> Image.Image:
  data = base64.b64decode(payload["scene_board_image_b64"])
  image = Image.open(BytesIO(data)).convert("RGB")
  image.thumbnail((int(os.getenv("RTP_VLM_IMAGE_SIZE", "384")), int(os.getenv("RTP_VLM_IMAGE_SIZE", "384"))), Image.Resampling.BILINEAR)
  return image


def _normalize_rtp(text: str) -> str:
  text = text.strip()
  text = re.sub(r"^```(?:text)?", "", text).strip()
  text = re.sub(r"```$", "", text).strip()
  idx = text.find("RTPv1")
  if idx >= 0:
    text = text[idx:]
  return text.strip()


def _extract_new_text(processor, inputs, output_ids) -> str:
  trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
  return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def generate(processor, model, payload: dict, max_new_tokens: int) -> dict:
  image = _image_from_payload(payload)
  messages = [{
    "role": "user",
    "content": [
      {"type": "image"},
      {"type": "text", "text": f"{SYSTEM_PROMPT}\nVehicle state: {payload.get('scene_board_state_text', '')}"},
    ],
  }]

  prefill_start = time.perf_counter()
  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
  inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
  torch.cuda.synchronize()
  prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

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
  text = _normalize_rtp(_extract_new_text(processor, inputs, output_ids))
  return {
    "text": text,
    "generated_token_count": int(output_ids.shape[-1] - inputs.input_ids.shape[-1]),
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "backend": "smolvlm2-500m-transformers",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent SmolVLM2 RTP worker. Reads JSONL on stdin, writes JSONL on stdout.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--max-new-tokens", type=int, default=48)
  args = parser.parse_args()

  processor, model = _load(args.model_dir)
  warm = Image.new("RGB", (384, 384), (20, 20, 20))
  buf = BytesIO()
  warm.save(buf, format="PNG")
  generate(processor, model, {"scene_board_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "scene_board_state_text": "warmup"}, 8)

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      response = generate(processor, model, payload, args.max_new_tokens)
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
  main()
