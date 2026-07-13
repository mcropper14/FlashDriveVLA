#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

ort.set_default_logger_severity(4)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "vlm" / "smolvlm2_256m_video_instruct"
DEFAULT_ONNX_DIR = DEFAULT_MODEL_DIR / "onnx"

SYSTEM_PROMPT = """Output only RTPv1.
RTPv1
scene=nominal
evidence=[base_path_visible]
meta=BASE
branch=base
lat_bias_m=0.0
speed_cap_mps=none
stop_s=none
avoid=[]
weights=[obs1.0,lane1.0,comfort1.0,base1.0,vlm1.0]
confidence=0.90
Change fields only if visible evidence requires a bounded path constraint."""


def _providers(provider: str) -> list:
  if provider == "tensorrt":
    if os.name == "nt":
      try:
        import tensorrt_libs
        os.add_dll_directory(str(Path(tensorrt_libs.__file__).parent))
      except Exception:
        pass
    cache = str((Path(__file__).resolve().parents[2] / "artifacts" / "reasoned_trajectory_poc" / "trt_cache").resolve())
    Path(cache).mkdir(parents=True, exist_ok=True)
    return [
      ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_fp16_enable": True,
      }),
      "CUDAExecutionProvider",
      "CPUExecutionProvider",
    ]
  if provider == "cuda":
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
  return ["CPUExecutionProvider"]


def _session(path: Path, providers: list) -> ort.InferenceSession:
  if not path.exists():
    raise FileNotFoundError(path)
  opts = ort.SessionOptions()
  opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  opts.enable_mem_pattern = False
  opts.log_severity_level = 4
  return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def _load(model_dir: Path, onnx_dir: Path, variant: str, vision_provider: str, decoder_provider: str):
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  vision_providers = _providers(vision_provider)
  decoder_providers = _providers(decoder_provider)
  vision_variant = "fp16" if vision_provider == "tensorrt" and variant == "q4f16" else variant
  vision = _session(onnx_dir / f"vision_encoder_{vision_variant}.onnx", vision_providers)
  embed = _session(onnx_dir / f"embed_tokens_{variant}.onnx", decoder_providers)
  decoder = _session(onnx_dir / f"decoder_model_merged_{variant}.onnx", decoder_providers)
  return processor, vision, embed, decoder


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


def _make_inputs(processor, image: Image.Image, state_text: str):
  messages = [{
    "role": "user",
    "content": [
      {"type": "image"},
      {"type": "text", "text": f"{SYSTEM_PROMPT}\nVehicle state: {state_text}"},
    ],
  }]
  prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
  )
  return processor(text=prompt, images=[image], return_tensors="np")


def _merge_image_features(input_ids: np.ndarray, inputs_embeds: np.ndarray, image_features: np.ndarray, image_token_id: int) -> np.ndarray:
  merged = inputs_embeds.copy()
  image_mask = input_ids == image_token_id
  if not image_mask.any():
    return merged
  flat_features = image_features.reshape(-1, image_features.shape[-1]).astype(np.float32, copy=False)
  positions = np.argwhere(image_mask)
  n = min(len(positions), len(flat_features))
  for idx in range(n):
    b, s = positions[idx]
    merged[b, s, :] = flat_features[idx]
  return merged


def _empty_past(num_layers: int = 30) -> dict[str, np.ndarray]:
  return {
    f"past_key_values.{layer}.{kind}": np.zeros((1, 3, 0, 64), dtype=np.float16)
    for layer in range(num_layers)
    for kind in ("key", "value")
  }


def _decode(
  processor,
  embed: ort.InferenceSession,
  decoder: ort.InferenceSession,
  input_ids: np.ndarray,
  inputs_embeds: np.ndarray,
  attention_mask: np.ndarray,
  max_new_tokens: int,
) -> tuple[str, int]:
  past = _empty_past()
  seq_len = inputs_embeds.shape[1]
  position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
  decoder_inputs = {
    "inputs_embeds": inputs_embeds.astype(np.float32, copy=False),
    "attention_mask": attention_mask.astype(np.int64, copy=False),
    "position_ids": position_ids,
    **past,
  }
  outputs = decoder.run(None, decoder_inputs)
  logits = outputs[0]
  past_names = [out.name for out in decoder.get_outputs()[1:]]
  past = dict(zip(past_names, outputs[1:]))
  next_token = np.array([[int(np.argmax(logits[:, -1, :], axis=-1)[0])]], dtype=np.int64)
  generated: list[int] = []
  eos_ids = processor.tokenizer.eos_token_id
  eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

  for _ in range(max_new_tokens):
    token_id = int(next_token[0, 0])
    if token_id in eos_set:
      break
    generated.append(token_id)
    token_embeds = embed.run(None, {"input_ids": next_token})[0].astype(np.float32, copy=False)
    attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
    position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)
    decoder_inputs = {
      "inputs_embeds": token_embeds,
      "attention_mask": attention_mask,
      "position_ids": position_ids,
      **{name.replace("present.", "past_key_values."): value for name, value in past.items()},
    }
    outputs = decoder.run(None, decoder_inputs)
    logits = outputs[0]
    past = dict(zip(past_names, outputs[1:]))
    next_token = np.array([[int(np.argmax(logits[:, -1, :], axis=-1)[0])]], dtype=np.int64)

  return processor.tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def generate(processor, vision, embed, decoder, payload: dict, max_new_tokens: int, backend_name: str) -> dict:
  image = _image_from_payload(payload)
  prefill_start = time.perf_counter()
  proc_inputs = _make_inputs(processor, image, payload.get("scene_board_state_text", ""))
  input_ids = np.asarray(proc_inputs["input_ids"], dtype=np.int64)
  attention_mask = np.asarray(proc_inputs["attention_mask"], dtype=np.int64)
  pixel_values = np.asarray(proc_inputs["pixel_values"], dtype=np.float32)
  pixel_attention_mask = np.asarray(proc_inputs["pixel_attention_mask"], dtype=bool)
  image_features = vision.run(None, {"pixel_values": pixel_values, "pixel_attention_mask": pixel_attention_mask})[0]
  inputs_embeds = embed.run(None, {"input_ids": input_ids})[0]
  inputs_embeds = _merge_image_features(input_ids, inputs_embeds, image_features, int(processor.image_token_id))
  prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

  decode_start = time.perf_counter()
  text, generated_count = _decode(processor, embed, decoder, input_ids, inputs_embeds, attention_mask, max_new_tokens)
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  return {
    "text": _normalize_rtp(text),
    "generated_token_count": generated_count,
    "prefill_ms": prefill_ms,
    "decode_ms": decode_ms,
    "backend": backend_name,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Persistent SmolVLM2 ONNX Runtime RTP worker. Reads JSONL on stdin, writes JSONL on stdout.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
  parser.add_argument("--provider", choices=("cuda", "tensorrt", "cpu"), default=os.getenv("RTP_ORT_PROVIDER", "cuda"))
  parser.add_argument("--vision-provider", choices=("cuda", "tensorrt", "cpu"), default=None)
  parser.add_argument("--decoder-provider", choices=("cuda", "tensorrt", "cpu"), default=None)
  parser.add_argument("--variant", default=os.getenv("RTP_ORT_VARIANT", "q4f16"))
  parser.add_argument("--max-new-tokens", type=int, default=48)
  args = parser.parse_args()

  vision_provider = args.vision_provider or args.provider
  decoder_provider = args.decoder_provider or args.provider
  processor, vision, embed, decoder = _load(args.model_dir, args.onnx_dir, args.variant, vision_provider, decoder_provider)
  image_size = int(os.getenv("RTP_VLM_IMAGE_SIZE", "384"))
  backend_name = f"smolvlm2-256m-onnxruntime-v{vision_provider}-d{decoder_provider}-{args.variant}"
  warm = Image.new("RGB", (image_size, image_size), (20, 20, 20))
  buf = BytesIO()
  warm.save(buf, format="PNG")
  generate(processor, vision, embed, decoder, {"scene_board_image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "scene_board_state_text": "warmup"}, 2, backend_name)

  for line in sys.stdin:
    try:
      payload = json.loads(line)
      response = generate(processor, vision, embed, decoder, payload, args.max_new_tokens, backend_name)
    except Exception as exc:
      response = {"error": repr(exc)}
    print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
  main()
