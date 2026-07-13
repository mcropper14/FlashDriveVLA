#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time

from PIL import Image
import torch


NANOVLM_DIR = Path(__file__).resolve().parents[1] / "nanoVLM"
sys.path.insert(0, str(NANOVLM_DIR))

from data.processors import get_image_processor, get_image_string, get_tokenizer  # noqa: E402
from models.vision_language_model import VisionLanguageModel  # noqa: E402


DEFAULT_HF_MODEL = "lusxvr/nanoVLM-230M-8k"
SYSTEM_PROMPT = (
  "Output only RTPv1 trajectory DSL. No prose. "
  "Prefer BASE unless visible evidence requires bounded constraints."
)


def _image_from_payload(payload: dict) -> Image.Image:
  data = base64.b64decode(payload["scene_board_image_b64"])
  image = Image.open(BytesIO(data)).convert("RGB")
  size = int(os.getenv("RTP_VLM_IMAGE_SIZE", "512"))
  image.thumbnail((size, size), Image.Resampling.BILINEAR)
  return image


def _normalize_rtp(text: str) -> str:
  text = text.strip()
  text = re.sub(r"^```(?:text|python)?", "", text).strip()
  text = re.sub(r"```$", "", text).strip()
  idx = text.find("RTPv1")
  if idx >= 0:
    text = text[idx:]
  return text.strip()


class NanoVlmRuntime:
  def __init__(self, hf_model: str, device: str, single_crop: bool, prompt_mode: str):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    self.device = torch.device(device)
    with contextlib.redirect_stdout(sys.stderr):
      self.model = VisionLanguageModel.from_pretrained(hf_model).eval().to(self.device)
      self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer, self.model.cfg.vlm_extra_tokens, self.model.cfg.lm_chat_template)
      self.image_processor = get_image_processor(
        self.model.cfg.max_img_size,
        self.model.cfg.vit_img_size,
        False if single_crop else getattr(self.model.cfg, "resize_to_max_side_len", False),
      )
    self.prompt_mode = prompt_mode

  def _make_inputs(self, image: Image.Image, state_text: str):
    processed_image, split_ratio = self.image_processor(image)
    if not hasattr(self.tokenizer, "global_image_token") and split_ratio[0] * split_ratio[1] == len(processed_image) - 1:
      processed_image = processed_image[1:]
    image_string = get_image_string(self.tokenizer, [split_ratio], self.model.cfg.mp_image_token_length)
    text = "RTPv1" if self.prompt_mode == "tiny" else SYSTEM_PROMPT + "\n" + state_text
    messages = [{
      "role": "user",
      "content": image_string + text,
    }]
    encoded = self.tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor(encoded, device=self.device)
    if input_ids.ndim == 1:
      input_ids = input_ids.unsqueeze(0)
    return input_ids, processed_image.to(self.device)

  def generate(self, payload: dict, max_new_tokens: int) -> dict:
    image = _image_from_payload(payload)
    start = time.perf_counter()
    input_ids, image_tensor = self._make_inputs(image, payload.get("scene_board_state_text", ""))
    if self.device.type == "cuda":
      torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000.0

    decode_start = time.perf_counter()
    generated = self.model.generate(input_ids, image_tensor, max_new_tokens=max_new_tokens, greedy=True)
    if self.device.type == "cuda":
      torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return {
      "text": _normalize_rtp(text),
      "generated_token_count": int(generated.shape[-1]) if hasattr(generated, "shape") else max_new_tokens,
      "prefill_ms": prefill_ms,
      "decode_ms": decode_ms,
      "backend": "nanovlm-230m-pytorch",
    }


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent nanoVLM RTP worker. Reads JSONL on stdin, writes JSONL on stdout.")
  parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--max-new-tokens", type=int, default=8)
  parser.add_argument("--prompt-mode", choices=("rtp", "tiny"), default=os.getenv("RTP_NANOVLM_PROMPT_MODE", "rtp"))
  parser.add_argument("--single-crop", action="store_true", default=True)
  args = parser.parse_args()

  runtime = NanoVlmRuntime(args.hf_model, args.device, args.single_crop, args.prompt_mode)
  warm = Image.new("RGB", (128, 128), (20, 20, 20))
  buf = BytesIO()
  warm.save(buf, format="PNG")
  runtime.generate({"scene_board_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "scene_board_state_text": "warmup"}, 1)

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      response = runtime.generate(payload, args.max_new_tokens)
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
  main()
