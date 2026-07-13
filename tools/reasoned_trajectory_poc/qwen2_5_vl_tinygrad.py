#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "vlm" / "qwen2_5_vl_3b_instruct"


def _ensure_tinygrad_import() -> None:
  tinygrad_repo = Path(os.getenv("QWEN_TINYGRAD_REPO", str(REPO_ROOT / "tinygrad_repo")))
  if tinygrad_repo.exists():
    import sys
    sys.path.insert(0, str(tinygrad_repo))


_ensure_tinygrad_import()

from tinygrad import Tensor, Device, TinyJit, dtypes, nn  # noqa: E402
from tinygrad.helpers import DEV, JIT_BATCH_SIZE  # noqa: E402
from tinygrad.nn.state import load_state_dict, safe_load, safe_load_metadata  # noqa: E402


_QWEN_LINEAR_LOAD_IN_OUT = False


class QwenLinear:
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    self.weight = Tensor.empty(in_features, out_features) if _QWEN_LINEAR_LOAD_IN_OUT else Tensor.empty(out_features, in_features)
    self.bias = Tensor.empty(out_features) if bias else None
    self._weight_in_out = _QWEN_LINEAR_LOAD_IN_OUT

  def prepare_inference_layout(self, realize: bool = True) -> None:
    if self._weight_in_out:
      return
    self.weight = self.weight.transpose().contiguous()
    if realize:
      self.weight = self.weight.realize()
    self._weight_in_out = True
    if self.bias is not None and realize:
      self.bias = self.bias.realize()

  def __call__(self, x: Tensor) -> Tensor:
    if self._weight_in_out:
      out = x @ self.weight
      return out if self.bias is None else out + self.bias
    return x.linear(self.weight.transpose(), self.bias)


def _iter_qwen_linears(obj, seen: set[int] | None = None) -> Iterable[QwenLinear]:
  if seen is None:
    seen = set()
  oid = id(obj)
  if oid in seen:
    return
  seen.add(oid)
  if isinstance(obj, QwenLinear):
    yield obj
    return
  if isinstance(obj, Tensor) or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
    return
  if isinstance(obj, dict):
    for value in obj.values():
      yield from _iter_qwen_linears(value, seen)
    return
  if isinstance(obj, (list, tuple)):
    for value in obj:
      yield from _iter_qwen_linears(value, seen)
    return
  if hasattr(obj, "__dict__"):
    for value in vars(obj).values():
      yield from _iter_qwen_linears(value, seen)


def _prepare_linear_inference_layout(obj, realize: bool = True) -> int:
  count = 0
  for linear in _iter_qwen_linears(obj):
    linear.prepare_inference_layout(realize=realize)
    count += 1
  return count


class QwenVisionPatchEmbed:
  def __init__(self, embed_dim: int, in_channels: int, temporal_patch_size: int, patch_size: int):
    self.in_channels = in_channels
    self.temporal_patch_size = temporal_patch_size
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.proj = {
      "weight": Tensor.empty(embed_dim, in_channels, temporal_patch_size, patch_size, patch_size),
    }

  def __call__(self, x: Tensor) -> Tensor:
    weight = self.proj["weight"].reshape(self.embed_dim, -1)
    return x.cast(weight.dtype).linear(weight.transpose()).reshape(-1, self.embed_dim)


def _rotate_half(x: Tensor) -> Tensor:
  x1 = x[..., :x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2:]
  return (-x2).cat(x1, dim=-1)


def _apply_vision_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
  cos = cos.unsqueeze(-2).float()
  sin = sin.unsqueeze(-2).float()
  qf, kf = q.float(), k.float()
  return (qf * cos + _rotate_half(qf) * sin).cast(q.dtype), (kf * cos + _rotate_half(kf) * sin).cast(k.dtype)


def _segment_attention(q: Tensor, k: Tensor, v: Tensor, segments: list[tuple[int, int]], scaling: float) -> Tensor:
  outs: list[Tensor] = []
  for start, end in segments:
    if end <= start:
      continue
    qh = q[start:end].transpose(0, 1).unsqueeze(0)
    kh = k[start:end].transpose(0, 1).unsqueeze(0)
    vh = v[start:end].transpose(0, 1).unsqueeze(0)
    attn = (qh @ kh.transpose(-1, -2) * scaling).softmax(-1)
    outs.append((attn @ vh).squeeze(0).transpose(0, 1))
  if not outs:
    raise ValueError("vision attention has no non-empty segments")
  return outs[0].cat(*outs[1:], dim=0) if len(outs) > 1 else outs[0]


class QwenVisionAttention:
  def __init__(self, hidden_size: int, num_heads: int):
    self.dim = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.qkv = QwenLinear(hidden_size, hidden_size * 3, bias=True)
    self.proj = QwenLinear(hidden_size, hidden_size, bias=True)
    self.scaling = self.head_dim ** -0.5

  def __call__(self, x: Tensor, segments: list[tuple[int, int]], cos: Tensor, sin: Tensor) -> Tensor:
    seq_len = x.shape[0]
    qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q, k = _apply_vision_rope(q, k, cos, sin)
    return self.proj(_segment_attention(q, k, v, segments, self.scaling).reshape(seq_len, self.dim))


class QwenVisionMLP:
  def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
    self.gate_proj = QwenLinear(hidden_size, intermediate_size, bias=bias)
    self.up_proj = QwenLinear(hidden_size, intermediate_size, bias=bias)
    self.down_proj = QwenLinear(intermediate_size, hidden_size, bias=bias)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class QwenVisionBlock:
  def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
    self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
    self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
    self.attn = QwenVisionAttention(hidden_size, num_heads)
    self.mlp = QwenVisionMLP(hidden_size, intermediate_size, bias=True)

  def __call__(self, x: Tensor, segments: list[tuple[int, int]], cos: Tensor, sin: Tensor) -> Tensor:
    x = x + self.attn(self.norm1(x), segments, cos, sin)
    return x + self.mlp(self.norm2(x))


class QwenVisionPatchMerger:
  def __init__(self, out_hidden_size: int, context_dim: int, spatial_merge_size: int):
    hidden_size = context_dim * (spatial_merge_size ** 2)
    self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
    self.mlp = [
      QwenLinear(hidden_size, hidden_size, bias=True),
      QwenLinear(hidden_size, out_hidden_size, bias=True),
    ]

  def __call__(self, x: Tensor) -> Tensor:
    x = self.ln_q(x).reshape(-1, self.mlp[0].weight.shape[1])
    return self.mlp[1](self.mlp[0](x).gelu(approximate="none"))


class Qwen2_5_VLVisionTinygrad:
  def __init__(self, config: dict, max_blocks: int | None = None):
    self.config = config
    self.hidden_size = int(config["hidden_size"])
    self.num_heads = int(config["num_heads"])
    self.head_dim = self.hidden_size // self.num_heads
    self.spatial_merge_size = int(config["spatial_merge_size"])
    self.spatial_merge_unit = self.spatial_merge_size ** 2
    self.patch_size = int(config["patch_size"])
    self.window_size = int(config["window_size"])
    self.fullatt_block_indexes = tuple(int(x) for x in config["fullatt_block_indexes"])
    depth = int(config["depth"]) if max_blocks is None else min(int(config["depth"]), max(0, int(max_blocks)))

    self.patch_embed = QwenVisionPatchEmbed(
      embed_dim=self.hidden_size,
      in_channels=int(config["in_chans"]),
      temporal_patch_size=int(config["temporal_patch_size"]),
      patch_size=self.patch_size,
    )
    self.blocks = [QwenVisionBlock(self.hidden_size, int(config["intermediate_size"]), self.num_heads) for _ in range(depth)]
    self.merger = QwenVisionPatchMerger(int(config["out_hidden_size"]), self.hidden_size, self.spatial_merge_size)

  def __call__(self, pixel_values: Tensor, grid_thw: list[tuple[int, int, int]]) -> Tensor:
    return self.forward_cached(pixel_values, self.build_grid_cache(grid_thw))

  def build_grid_cache(self, grid_thw: list[tuple[int, int, int]]) -> "QwenVisionGridCache":
    cos, sin = _vision_rotary_cos_sin(grid_thw, self.head_dim, self.spatial_merge_size)
    cos, sin = cos.realize(), sin.realize()
    window_index, window_segments = _window_index_and_segments(
      grid_thw,
      self.spatial_merge_size,
      self.patch_size,
      self.window_size,
    )
    reverse = _argsort_indices(window_index)
    return QwenVisionGridCache(
      grid_thw=tuple(grid_thw),
      cos=cos,
      sin=sin,
      window_index=None if _is_identity_index(window_index) else Tensor(window_index, dtype=dtypes.int32).realize(),
      reverse_index=None if _is_identity_index(reverse) else Tensor(reverse, dtype=dtypes.int32).realize(),
      window_segments=window_segments,
      full_segments=_full_segments(grid_thw),
    )

  def forward_cached(self, pixel_values: Tensor, cache: "QwenVisionGridCache") -> Tensor:
    x = self.patch_embed(pixel_values)

    seq_len = x.shape[0]
    x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    if cache.window_index is not None:
      x = x[cache.window_index]
    x = x.reshape(seq_len, -1)
    cos = cache.cos.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    sin = cache.sin.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    if cache.window_index is not None:
      cos = cos[cache.window_index]
      sin = sin[cache.window_index]
    cos = cos.reshape(seq_len, -1)
    sin = sin.reshape(seq_len, -1)

    for layer_num, block in enumerate(self.blocks):
      segments = cache.full_segments if layer_num in self.fullatt_block_indexes else cache.window_segments
      x = block(x, segments, cos, sin)

    x = self.merger(x)
    return x if cache.reverse_index is None else x[cache.reverse_index]


def _cat_tensors(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
  if not tensors:
    raise ValueError("cannot concatenate an empty tensor list")
  return tensors[0].cat(*tensors[1:], dim=dim) if len(tensors) > 1 else tensors[0]


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
  if n_rep == 1:
    return x
  bsz, heads, seqlen, head_dim = x.shape
  return x.transpose(1, 2).repeat((1, 1, 1, n_rep)).reshape(bsz, seqlen, heads * n_rep, head_dim).transpose(1, 2)


def _apply_multimodal_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, mrope_section: Sequence[int]) -> tuple[Tensor, Tensor]:
  sections = [int(v) for v in mrope_section] * 2
  cos_parts: list[Tensor] = []
  sin_parts: list[Tensor] = []
  start = 0
  for idx, width in enumerate(sections):
    end = start + width
    cos_parts.append(cos[idx % 3][..., start:end])
    sin_parts.append(sin[idx % 3][..., start:end])
    start = end
  cos_mix = _cat_tensors(cos_parts, dim=-1).unsqueeze(1)
  sin_mix = _cat_tensors(sin_parts, dim=-1).unsqueeze(1)
  qf, kf = q.float(), k.float()
  return (qf * cos_mix + _rotate_half(qf) * sin_mix).cast(q.dtype), (kf * cos_mix + _rotate_half(kf) * sin_mix).cast(k.dtype)


def _text_rope_cos_sin(position_ids: Tensor, head_dim: int, theta: float, mrope_section: Sequence[int]) -> tuple[Tensor, Tensor]:
  if sum(int(v) for v in mrope_section) * 2 != head_dim:
    raise ValueError(f"mrope sections {mrope_section} do not sum to head_dim {head_dim}")
  inv_freq = Tensor([1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)], dtype=dtypes.float32)
  freqs = position_ids.float().unsqueeze(-1) * inv_freq.reshape(1, 1, 1, -1)
  emb = freqs.cat(freqs, dim=-1)
  return emb.cos(), emb.sin()


def _causal_padding_mask(attention_mask: Tensor, dtype) -> Tensor:
  bsz, seq_len = attention_mask.shape
  causal = Tensor.full((1, 1, seq_len, seq_len), -float("inf"), dtype=dtype).triu(1)
  pad = (attention_mask == 0).reshape(bsz, 1, 1, seq_len).where(-float("inf"), 0.0).cast(dtype)
  return causal + pad


_CAUSAL_MASK_CACHE: dict[tuple[int, str], Tensor] = {}


def _cached_causal_mask(seq_len: int, dtype) -> Tensor:
  key = (seq_len, str(dtype))
  mask = _CAUSAL_MASK_CACHE.get(key)
  if mask is None:
    idx = Tensor.arange(seq_len).realize()
    mask = (idx.reshape(seq_len, 1) < idx.reshape(1, seq_len)).where(-float("inf"), 0.0)
    mask = mask.reshape(1, seq_len, seq_len).cast(dtype).realize()
    _CAUSAL_MASK_CACHE[key] = mask
  return mask


def _explicit_gqa_attention(q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None) -> Tensor:
  bsz, heads, seq_len, head_dim = q.shape
  _, kv_heads, _, _ = k.shape
  if heads % kv_heads != 0:
    raise ValueError(f"num heads {heads} is not divisible by kv heads {kv_heads}")

  kv_groups = heads // kv_heads
  qf = q.reshape(bsz * heads, seq_len, head_dim).contiguous()
  kg = k.reshape(bsz, kv_heads, 1, seq_len, head_dim).expand(bsz, kv_heads, kv_groups, seq_len, head_dim)
  vg = v.reshape(bsz, kv_heads, 1, seq_len, head_dim).expand(bsz, kv_heads, kv_groups, seq_len, head_dim)
  kg = kg.reshape(bsz * heads, seq_len, head_dim).contiguous()
  vg = vg.reshape(bsz * heads, seq_len, head_dim).contiguous()

  scores = (qf @ kg.transpose(1, 2)) * (head_dim ** -0.5)
  if attention_mask is None:
    probs = (scores + _cached_causal_mask(seq_len, scores.dtype)).softmax(-1)
    out = probs @ vg
  else:
    causal = _cached_causal_mask(seq_len, scores.dtype).reshape(1, 1, seq_len, seq_len)
    pad = (attention_mask == 0).reshape(bsz, 1, 1, seq_len).where(-float("inf"), 0.0).cast(scores.dtype)
    probs = (scores.reshape(bsz, heads, seq_len, seq_len) + causal + pad).softmax(-1)
    out = probs.reshape(bsz * heads, seq_len, seq_len) @ vg
  return out.reshape(bsz, heads, seq_len, head_dim).transpose(1, 2)


class QwenTextAttention:
  def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, rope_scaling: dict):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.num_kv_groups = num_heads // num_kv_heads
    self.head_dim = hidden_size // num_heads
    self.rope_scaling = rope_scaling
    self.q_proj = QwenLinear(hidden_size, num_heads * self.head_dim, bias=True)
    self.k_proj = QwenLinear(hidden_size, num_kv_heads * self.head_dim, bias=True)
    self.v_proj = QwenLinear(hidden_size, num_kv_heads * self.head_dim, bias=True)
    self.o_proj = QwenLinear(num_heads * self.head_dim, hidden_size, bias=False)

  def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, attention_mask: Tensor | None) -> Tensor:
    bsz, seq_len, _ = x.shape
    q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
    q, k = _apply_multimodal_rope(q, k, cos, sin, self.rope_scaling["mrope_section"])
    if os.getenv("QWEN_TEXT_ATTN", "explicit") == "explicit":
      attn = _explicit_gqa_attention(q, k, v, attention_mask)
    elif os.getenv("QWEN_USE_FLASH_ATTN") == "1" and attention_mask is None:
      from extra.thunder.tiny.fa import flash_attention
      attn = flash_attention(q, k, v, is_causal=True).transpose(1, 2)
    elif attention_mask is not None:
      mask = _causal_padding_mask(attention_mask, x.dtype)
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).transpose(1, 2)
    else:
      attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).transpose(1, 2)
    return self.o_proj(attn.reshape(bsz, seq_len, self.num_heads * self.head_dim))


class QwenTextMLP:
  def __init__(self, hidden_size: int, intermediate_size: int):
    self.gate_proj = QwenLinear(hidden_size, intermediate_size, bias=False)
    self.up_proj = QwenLinear(hidden_size, intermediate_size, bias=False)
    self.down_proj = QwenLinear(intermediate_size, hidden_size, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x.contiguous_backward()))


class QwenTextBlock:
  def __init__(self, config: dict):
    hidden_size = int(config["hidden_size"])
    self.input_layernorm = nn.RMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
    self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
    self.self_attn = QwenTextAttention(
      hidden_size,
      int(config["num_attention_heads"]),
      int(config["num_key_value_heads"]),
      dict(config["rope_scaling"]),
    )
    self.mlp = QwenTextMLP(hidden_size, int(config["intermediate_size"]))

  def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, attention_mask: Tensor | None) -> Tensor:
    h = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
    return (h + self.mlp(self.post_attention_layernorm(h))).contiguous().contiguous_backward()


class Qwen2_5_VLTextTinygrad:
  def __init__(self, config: dict, max_layers: int | None = None, with_embeddings: bool = True):
    self.config = config
    self.hidden_size = int(config["hidden_size"])
    self.num_heads = int(config["num_attention_heads"])
    self.head_dim = self.hidden_size // self.num_heads
    self.rope_theta = float(config.get("rope_theta", 1000000.0))
    self.with_embeddings = with_embeddings
    depth = int(config["num_hidden_layers"]) if max_layers is None else min(int(config["num_hidden_layers"]), max(0, int(max_layers)))
    self.embed_tokens = nn.Embedding(int(config["vocab_size"]), self.hidden_size) if with_embeddings else None
    self.layers = [QwenTextBlock(config) for _ in range(depth)]
    self.norm = nn.RMSNorm(self.hidden_size, eps=float(config["rms_norm_eps"]))

  def embed(self, input_ids: Tensor) -> Tensor:
    if self.embed_tokens is None:
      raise RuntimeError("text embeddings were not loaded; pass packed input_embeds instead")
    return self.embed_tokens(input_ids)

  def forward_embeds(self, inputs_embeds: Tensor, position_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    cos, sin = _text_rope_cos_sin(
      position_ids,
      self.head_dim,
      self.rope_theta,
      self.config["rope_scaling"]["mrope_section"],
    )
    h = inputs_embeds
    for layer in self.layers:
      h = layer(h, cos.cast(h.dtype), sin.cast(h.dtype), attention_mask)
    return self.norm(h)

  def candidate_logits(self, hidden: Tensor, candidate_ids: Sequence[int]) -> Tensor:
    if self.embed_tokens is None:
      raise RuntimeError("text embeddings were not loaded; pass packed candidate_weight instead")
    ids = Tensor([int(v) for v in candidate_ids], dtype=dtypes.int32)
    weights = self.embed_tokens.weight[ids]
    return hidden[:, -1, :] @ weights.transpose()

  def candidate_logits_from_weight(self, hidden: Tensor, candidate_weight: Tensor) -> Tensor:
    return hidden[:, -1, :] @ candidate_weight.transpose()


class Qwen2_5_VLLabelScorerTinygrad:
  def __init__(
    self,
    config: dict,
    max_vision_blocks: int | None = None,
    max_text_layers: int | None = None,
    with_text_embeddings: bool = True,
  ):
    self.config = config
    self.image_token_id = int(config["image_token_id"])
    self.visual = Qwen2_5_VLVisionTinygrad(config["vision_config"], max_blocks=max_vision_blocks)
    self.text = Qwen2_5_VLTextTinygrad(config, max_layers=max_text_layers, with_embeddings=with_text_embeddings)

  def build_vision_cache(self, grid_thw: list[tuple[int, int, int]]) -> QwenVisionGridCache:
    return self.visual.build_grid_cache(grid_thw)

  def input_embeds_with_images(
    self,
    input_ids: Tensor,
    pixel_values: Tensor,
    image_index: Tensor,
    vision_cache: QwenVisionGridCache,
    text_embeds: Tensor | None = None,
  ) -> Tensor:
    token_embeds = self.text.embed(input_ids) if text_embeds is None else text_embeds
    image_features = self.visual.forward_cached(pixel_values, vision_cache)
    flat_index = image_index.reshape(-1)
    gathered = image_features[flat_index].reshape(*input_ids.shape, self.text.hidden_size)
    image_mask = (input_ids == self.image_token_id).unsqueeze(-1)
    return image_mask.where(gathered, token_embeds)

  def score_candidates(
    self,
    input_ids: Tensor,
    pixel_values: Tensor,
    image_index: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    vision_cache: QwenVisionGridCache,
    candidate_ids: Sequence[int],
  ) -> Tensor:
    embeds = self.input_embeds_with_images(input_ids, pixel_values, image_index, vision_cache)
    hidden = self.text.forward_embeds(embeds, position_ids, attention_mask)
    return self.text.candidate_logits(hidden, candidate_ids)

  def score_packed_candidates(
    self,
    input_ids: Tensor,
    text_embeds: Tensor,
    pixel_values: Tensor,
    image_index: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    vision_cache: QwenVisionGridCache,
    candidate_weight: Tensor,
  ) -> Tensor:
    embeds = self.input_embeds_with_images(input_ids, pixel_values, image_index, vision_cache, text_embeds=text_embeds)
    hidden = self.text.forward_embeds(embeds, position_ids, attention_mask)
    return self.text.candidate_logits_from_weight(hidden, candidate_weight)


@dataclass(frozen=True)
class QwenVisionGridCache:
  grid_thw: tuple[tuple[int, int, int], ...]
  cos: Tensor
  sin: Tensor
  window_index: Tensor | None
  reverse_index: Tensor | None
  window_segments: list[tuple[int, int]]
  full_segments: list[tuple[int, int]]


def _argsort_indices(values: list[int]) -> list[int]:
  return sorted(range(len(values)), key=values.__getitem__)


def _is_identity_index(values: list[int]) -> bool:
  return all(idx == value for idx, value in enumerate(values))


def _full_segments(grid_thw: Iterable[tuple[int, int, int]]) -> list[tuple[int, int]]:
  segments: list[tuple[int, int]] = []
  offset = 0
  for t, h, w in grid_thw:
    n = t * h * w
    segments.append((offset, offset + n))
    offset += n
  return segments


def _vision_rotary_cos_sin(grid_thw: list[tuple[int, int, int]], head_dim: int, spatial_merge_size: int) -> tuple[Tensor, Tensor]:
  rope_dim = head_dim // 2
  inv_freq = [1.0 / (10000.0 ** (i / rope_dim)) for i in range(0, rope_dim, 2)]
  max_grid = max(max(h, w) for _, h, w in grid_thw)
  table = [[pos * f for f in inv_freq] for pos in range(max_grid)]

  rows: list[list[float]] = []
  for t, h, w in grid_thw:
    for _ in range(t):
      for hb in range(h // spatial_merge_size):
        for wb in range(w // spatial_merge_size):
          for hi in range(spatial_merge_size):
            for wi in range(spatial_merge_size):
              hpos = hb * spatial_merge_size + hi
              wpos = wb * spatial_merge_size + wi
              rotary = table[hpos] + table[wpos]
              rows.append(rotary + rotary)
  cos = Tensor(rows, dtype=dtypes.float32).cos()
  sin = Tensor(rows, dtype=dtypes.float32).sin()
  return cos, sin


def _window_index_and_segments(
  grid_thw: list[tuple[int, int, int]],
  spatial_merge_size: int,
  patch_size: int,
  window_size: int,
) -> tuple[list[int], list[tuple[int, int]]]:
  vit_merger_window_size = window_size // spatial_merge_size // patch_size
  spatial_merge_unit = spatial_merge_size ** 2
  window_index: list[int] = []
  segment_lengths: list[int] = []
  window_index_id = 0

  for grid_t, grid_h, grid_w in grid_thw:
    llm_h = grid_h // spatial_merge_size
    llm_w = grid_w // spatial_merge_size
    pad_h = vit_merger_window_size - llm_h % vit_merger_window_size
    pad_w = vit_merger_window_size - llm_w % vit_merger_window_size
    num_windows_h = (llm_h + pad_h) // vit_merger_window_size
    num_windows_w = (llm_w + pad_w) // vit_merger_window_size
    for tt in range(grid_t):
      for wh in range(num_windows_h):
        for ww in range(num_windows_w):
          valid: list[int] = []
          for ih in range(vit_merger_window_size):
            for iw in range(vit_merger_window_size):
              gh = wh * vit_merger_window_size + ih
              gw = ww * vit_merger_window_size + iw
              if gh < llm_h and gw < llm_w:
                valid.append(window_index_id + tt * llm_h * llm_w + gh * llm_w + gw)
          if valid:
            window_index.extend(valid)
            segment_lengths.append(len(valid) * spatial_merge_unit)
    window_index_id += grid_t * llm_h * llm_w

  segments: list[tuple[int, int]] = []
  offset = 0
  for length in segment_lengths:
    segments.append((offset, offset + length))
    offset += length
  return window_index, segments


def _load_config(model_dir: Path) -> dict:
  with open(model_dir / "config.json", "r", encoding="utf-8") as f:
    return json.load(f)


def _sync_device() -> None:
  try:
    Device[Device.DEFAULT].synchronize()
  except Exception:
    pass


def _maybe_pretranspose_linear_weight(key: str, value: Tensor, enabled: bool) -> Tensor:
  if enabled and key.endswith(".weight") and len(value.shape) == 2:
    return value.transpose().contiguous()
  return value


def _load_visual_state(
  model_dir: Path,
  max_blocks: int | None = None,
  visual_safetensors: Path | None = None,
  pretranspose_linear: bool = False,
) -> dict[str, Tensor]:
  if visual_safetensors is not None:
    state: dict[str, Tensor] = {}
    for key, value in safe_load(visual_safetensors).items():
      if _keep_visual_key(key, max_blocks):
        state[_tinygrad_visual_key(key)] = _maybe_pretranspose_linear_weight(key, value, pretranspose_linear)
    return state

  index_path = model_dir / "model.safetensors.index.json"
  if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]
    files = sorted({model_dir / name for key, name in weight_map.items() if _keep_visual_key(key, max_blocks)})
  else:
    files = sorted(model_dir.glob("*.safetensors"))

  state: dict[str, Tensor] = {}
  for fn in files:
    for key, value in safe_load(fn).items():
      if _keep_visual_key(key, max_blocks):
        state[_tinygrad_visual_key(key)] = _maybe_pretranspose_linear_weight(key, value, pretranspose_linear)
  return state


def _tinygrad_visual_key(key: str) -> str:
  key = key.removeprefix("visual.")
  # HF stores merger.mlp as Linear, GELU, Linear. The tinygrad port stores only
  # stateful layers, so the second Linear is index 1 instead of 2.
  return key.replace("merger.mlp.2.", "merger.mlp.1.")


def _keep_visual_key(key: str, max_blocks: int | None = None) -> bool:
  if not key.startswith("visual."):
    return False
  if max_blocks is None:
    return True
  if not key.startswith("visual.blocks."):
    return True
  try:
    block_idx = int(key.split(".", 3)[2])
  except (IndexError, ValueError):
    return True
  return block_idx < max_blocks


def _keep_text_key(key: str, max_layers: int | None = None, skip_embeddings: bool = False) -> bool:
  if key.startswith("visual."):
    return False
  if skip_embeddings and key in ("model.embed_tokens.weight", "lm_head.weight"):
    return False
  if max_layers is None:
    return True
  if not key.startswith("model.layers."):
    return True
  try:
    layer_idx = int(key.split(".", 3)[2])
  except (IndexError, ValueError):
    return True
  return layer_idx < max_layers


def _tinygrad_text_key(key: str) -> str:
  return key.removeprefix("model.")


def _load_label_state(
  model_dir: Path,
  max_vision_blocks: int | None = None,
  max_text_layers: int | None = None,
  visual_safetensors: Path | None = None,
  text_safetensors: Path | None = None,
  skip_text_embeddings: bool = False,
  pretranspose_linear: bool = False,
) -> dict[str, Tensor]:
  state: dict[str, Tensor] = {}
  for key, value in _load_visual_state(model_dir, max_vision_blocks, visual_safetensors, pretranspose_linear=pretranspose_linear).items():
    state[f"visual.{key}"] = value

  if text_safetensors is not None:
    text_files = [text_safetensors]
    text_items = None
  else:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
      with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
      text_files = sorted({model_dir / name for key, name in weight_map.items() if _keep_text_key(key, max_text_layers, skip_text_embeddings)})
      text_items = weight_map
    else:
      text_files = sorted(model_dir.glob("*.safetensors"))
      text_items = None

  for fn in text_files:
    for key, value in safe_load(fn).items():
      if not _keep_text_key(key, max_text_layers, skip_text_embeddings):
        continue
      if text_items is not None and key not in text_items:
        continue
      state[f"text.{_tinygrad_text_key(key)}"] = _maybe_pretranspose_linear_weight(key, value, pretranspose_linear)
  if "text.embed_tokens.weight" in state and "text.output.weight" not in state:
    # Qwen2.5-VL-3B ties the output projection to token embeddings and does not store lm_head.weight.
    pass
  return state


def _metadata_keys(fn: Path) -> list[str]:
  _, _, metadata = safe_load_metadata(fn)
  return [key for key in metadata if key != "__metadata__"]


def export_visual_subset(model_dir: Path, out: Path, max_blocks: int | None, weight_dtype: str) -> None:
  from safetensors import safe_open
  import torch
  from safetensors.torch import save_file

  index_path = model_dir / "model.safetensors.index.json"
  if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]
    files = sorted({model_dir / name for key, name in weight_map.items() if _keep_visual_key(key, max_blocks)})
  else:
    files = sorted(model_dir.glob("*.safetensors"))

  tensors = {}
  for fn in files:
    with safe_open(fn, framework="pt", device="cpu") as f:
      for key in f.keys():
        if _keep_visual_key(key, max_blocks):
          tensor = f.get_tensor(key)
          if weight_dtype == "float16" and tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
          tensors[key] = tensor
  out.parent.mkdir(parents=True, exist_ok=True)
  save_file(tensors, str(out), metadata={"format": "pt", "source": "qwen2.5-vl-visual-subset", "weight_dtype": weight_dtype})


def export_text_subset(model_dir: Path, out: Path, max_layers: int | None, weight_dtype: str, skip_embeddings: bool = False) -> None:
  from safetensors import safe_open
  import torch
  from safetensors.torch import save_file

  index_path = model_dir / "model.safetensors.index.json"
  if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]
    files = sorted({model_dir / name for key, name in weight_map.items() if _keep_text_key(key, max_layers, skip_embeddings)})
  else:
    files = sorted(model_dir.glob("*.safetensors"))

  tensors = {}
  for fn in files:
    with safe_open(fn, framework="pt", device="cpu") as f:
      for key in f.keys():
        if _keep_text_key(key, max_layers, skip_embeddings):
          tensor = f.get_tensor(key)
          if weight_dtype == "float16" and tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
          tensors[key] = tensor
  out.parent.mkdir(parents=True, exist_ok=True)
  save_file(tensors, str(out), metadata={"format": "pt", "source": "qwen2.5-vl-text-subset", "weight_dtype": weight_dtype})


def _tokenizer_single_token_ids(tokenizer, variants: Sequence[str]) -> tuple[int, ...]:
  ids: set[int] = set()
  for variant in variants:
    encoded = tokenizer(variant, add_special_tokens=False).input_ids
    if len(encoded) == 1:
      ids.add(int(encoded[0]))
  return tuple(sorted(ids))


def _score_candidate_ids(processor) -> tuple[tuple[int, ...], tuple[int, ...]]:
  yes_ids = _tokenizer_single_token_ids(processor.tokenizer, ("yes", "Yes", " yes", " Yes"))
  no_ids = _tokenizer_single_token_ids(processor.tokenizer, ("no", "No", " no", " No"))
  if not yes_ids or not no_ids:
    raise RuntimeError(f"failed to find yes/no token ids: yes={yes_ids} no={no_ids}")
  return yes_ids, no_ids


def _load_score_prompt_bits():
  import sys

  script_dir = Path(__file__).resolve().parent
  sys.path.insert(0, str(script_dir))
  from qwen_label_rtp_worker import SCORE_PROMPT, SCORE_QUESTIONS, _inference_images

  return SCORE_PROMPT, SCORE_QUESTIONS, _inference_images


def _python_rope_index(
  input_ids_batch: list[list[int]],
  attention_mask_batch: list[list[int]],
  image_grid_thw: list[list[int]],
  config: dict,
) -> list[list[list[int]]]:
  spatial_merge_size = int(config["vision_config"]["spatial_merge_size"])
  image_token_id = int(config["image_token_id"])
  video_token_id = int(config["video_token_id"])
  vision_start_token_id = int(config["vision_start_token_id"])
  tokens_per_second = int(config["vision_config"].get("tokens_per_second", 2))
  batch = len(input_ids_batch)
  seq_len = len(input_ids_batch[0])
  position_ids = [[[1 for _ in range(seq_len)] for _ in range(batch)] for _ in range(3)]
  image_index = 0

  for batch_idx, full_input_ids in enumerate(input_ids_batch):
    active_positions = [idx for idx, value in enumerate(attention_mask_batch[batch_idx]) if value == 1]
    input_ids = [full_input_ids[idx] for idx in active_positions]
    vision_start_indices = [idx for idx, tok in enumerate(input_ids) if tok == vision_start_token_id]
    vision_tokens = [input_ids[idx + 1] for idx in vision_start_indices if idx + 1 < len(input_ids)]
    image_nums = sum(1 for tok in vision_tokens if tok == image_token_id)
    video_nums = sum(1 for tok in vision_tokens if tok == video_token_id)
    st = 0
    remain_images = image_nums
    remain_videos = video_nums
    chunks: list[list[list[int]]] = []

    for _ in range(image_nums + video_nums):
      ed_image = input_ids.index(image_token_id, st) if image_token_id in input_ids[st:] and remain_images > 0 else len(input_ids) + 1
      ed_video = input_ids.index(video_token_id, st) if video_token_id in input_ids[st:] and remain_videos > 0 else len(input_ids) + 1
      if ed_image < ed_video:
        t, h, w = image_grid_thw[image_index]
        image_index += 1
        remain_images -= 1
        ed = ed_image
        second_per_grid_t = 0
      else:
        raise NotImplementedError("packed tinygrad label scoring does not support video inputs")

      llm_grid_t = int(t)
      llm_grid_h = int(h) // spatial_merge_size
      llm_grid_w = int(w) // spatial_merge_size
      text_len = ed - st
      st_idx = max(max(row) for chunk in chunks for row in chunk) + 1 if chunks else 0
      chunks.append([[pos + st_idx for pos in range(text_len)] for _ in range(3)])

      t_index: list[int] = []
      h_index: list[int] = []
      w_index: list[int] = []
      for tt in range(llm_grid_t):
        for hh in range(llm_grid_h):
          for ww in range(llm_grid_w):
            t_index.append(int(tt * second_per_grid_t * tokens_per_second))
            h_index.append(hh)
            w_index.append(ww)
      chunks.append([
        [value + text_len + st_idx for value in t_index],
        [value + text_len + st_idx for value in h_index],
        [value + text_len + st_idx for value in w_index],
      ])
      st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_ids):
      st_idx = max(max(row) for chunk in chunks for row in chunk) + 1 if chunks else 0
      text_len = len(input_ids) - st
      chunks.append([[pos + st_idx for pos in range(text_len)] for _ in range(3)])

    merged = [sum((chunk[axis] for chunk in chunks), []) for axis in range(3)]
    for compact_pos, original_pos in enumerate(active_positions):
      for axis in range(3):
        position_ids[axis][batch_idx][original_pos] = int(merged[axis][compact_pos])

  return position_ids


def export_label_pack(
  model_dir: Path,
  image_path: Path,
  out: Path,
  score_labels: Sequence[str],
  image_mode: str,
  image_size: int,
  vehicle_state_text: str,
  pack_embeddings: bool = False,
) -> None:
  from PIL import Image
  from safetensors import safe_open
  import torch
  from transformers import AutoProcessor

  SCORE_PROMPT, SCORE_QUESTIONS, inference_images = _load_score_prompt_bits()
  processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
  source = Image.open(image_path).convert("RGB")
  images, _ = inference_images(source, image_mode, image_size)

  prompts: list[str] = []
  batch_images = []
  for label in score_labels:
    if label not in SCORE_QUESTIONS:
      raise ValueError(f"unknown score label: {label}")
    content = [{"type": "image", "image": image} for image in images]
    content.append({
      "type": "text",
      "text": f"{SCORE_PROMPT}\nVehicle state: {vehicle_state_text}\nQuestion: {SCORE_QUESTIONS[label]}",
    })
    prompts.append(processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    batch_images.extend(images)

  prefill_start = time.perf_counter()
  inputs = processor(text=prompts, images=batch_images, padding=True, return_tensors="pt")
  prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
  yes_ids, no_ids = _score_candidate_ids(processor)
  config = _load_config(model_dir)

  input_ids_tensor = inputs.input_ids.cpu().to(dtype=torch.int64)
  input_ids = input_ids_tensor.tolist()
  attention_mask = inputs.attention_mask.cpu().to(dtype=torch.int64).tolist()
  image_grid_thw = inputs.image_grid_thw.cpu().to(dtype=torch.int64).tolist()
  pixel_values = inputs.pixel_values.cpu().to(dtype=torch.float32).tolist()
  position_ids = _python_rope_index(input_ids, attention_mask, image_grid_thw, config)

  flat_image_index: list[int] = []
  next_image_token = 0
  image_token_id = int(config["image_token_id"])
  for row in input_ids:
    for tok in row:
      if int(tok) == image_token_id:
        flat_image_index.append(next_image_token)
        next_image_token += 1
      else:
        flat_image_index.append(0)
  image_index = [flat_image_index[idx:idx + len(input_ids[0])] for idx in range(0, len(flat_image_index), len(input_ids[0]))]

  payload = {
    "source_image": str(image_path),
    "image_mode": image_mode,
    "image_size": image_size,
    "score_labels": list(score_labels),
    "vehicle_state_text": vehicle_state_text,
    "processor_prefill_ms": prefill_ms,
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "pixel_values": pixel_values,
    "image_grid_thw": image_grid_thw,
    "image_index": image_index,
    "yes_ids": list(yes_ids),
    "no_ids": list(no_ids),
  }

  if pack_embeddings:
    index_path = model_dir / "model.safetensors.index.json"
    embed_file = None
    if index_path.exists():
      with open(index_path, "r", encoding="utf-8") as f:
        embed_file = model_dir / json.load(f)["weight_map"]["model.embed_tokens.weight"]
    else:
      for fn in sorted(model_dir.glob("*.safetensors")):
        with safe_open(fn, framework="pt", device="cpu") as f:
          if "model.embed_tokens.weight" in f.keys():
            embed_file = fn
            break
    if embed_file is None:
      raise RuntimeError("model.embed_tokens.weight not found")
    with safe_open(embed_file, framework="pt", device="cpu") as f:
      embed = f.get_tensor("model.embed_tokens.weight").to(torch.float16)
    candidate_ids = torch.tensor(list(yes_ids) + list(no_ids), dtype=torch.int64)
    payload["input_embeds"] = embed[input_ids_tensor].tolist()
    payload["candidate_weight"] = embed[candidate_ids].tolist()

  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))


def run_visual_smoke(
  model_dir: Path,
  grid_thw: tuple[int, int, int],
  max_blocks: int | None,
  realize: bool,
  visual_safetensors: Path | None = None,
  repeat_forwards: int = 1,
  use_jit: bool = False,
  jit_prune: bool = False,
  input_mode: str = "ramp",
  jit_batch_size: int | None = None,
) -> dict:
  if jit_batch_size is not None:
    JIT_BATCH_SIZE.value = jit_batch_size
  config = _load_config(model_dir)["vision_config"]
  model = Qwen2_5_VLVisionTinygrad(config, max_blocks=max_blocks)
  load_start = time.perf_counter()
  loaded = load_state_dict(
    model,
    _load_visual_state(model_dir, max_blocks=max_blocks, visual_safetensors=visual_safetensors),
    strict=True,
    consume=True,
    realize=realize,
  )
  load_ms = (time.perf_counter() - load_start) * 1000.0

  t, h, w = grid_thw
  patch_dim = int(config["in_chans"]) * int(config["temporal_patch_size"]) * int(config["patch_size"]) * int(config["patch_size"])
  input_elems = t * h * w * patch_dim
  if input_mode == "zeros":
    pixel_values = Tensor.zeros(t * h * w, patch_dim, dtype=dtypes.float32)
  else:
    pixel_values = (Tensor.arange(input_elems, dtype=dtypes.float32).reshape(t * h * w, patch_dim) / max(1, input_elems)).contiguous()
  pixel_values = pixel_values.clone().realize()
  cache = model.build_grid_cache([grid_thw])
  forward = TinyJit(lambda px: model.forward_cached(px, cache), prune=jit_prune) if use_jit else lambda px: model.forward_cached(px, cache)
  forward_ms: list[float] = []
  out = None
  for _ in range(max(1, repeat_forwards)):
    _sync_device()
    fwd_start = time.perf_counter()
    out = forward(pixel_values).realize()
    _sync_device()
    forward_ms.append((time.perf_counter() - fwd_start) * 1000.0)
  assert out is not None
  return {
    "backend_device": Device.DEFAULT,
    "grid_thw": list(grid_thw),
    "max_blocks": max_blocks,
    "loaded_tensors": len(loaded),
    "loaded_bytes": int(sum(t.nbytes() for t in loaded)),
    "output_shape": list(out.shape),
    "load_ms": load_ms,
    "use_jit": use_jit,
    "jit_batch_size": JIT_BATCH_SIZE.value,
    "jit_prune": jit_prune,
    "input_mode": input_mode,
    "forward_ms": forward_ms[-1],
    "forward_ms_all": forward_ms,
  }


def _make_pixel_values(config: dict, grid_thw: tuple[int, int, int], input_mode: str) -> Tensor:
  t, h, w = grid_thw
  patch_dim = int(config["in_chans"]) * int(config["temporal_patch_size"]) * int(config["patch_size"]) * int(config["patch_size"])
  input_elems = t * h * w * patch_dim
  if input_mode == "zeros":
    pixel_values = Tensor.zeros(t * h * w, patch_dim, dtype=dtypes.float32)
  else:
    pixel_values = (Tensor.arange(input_elems, dtype=dtypes.float32).reshape(t * h * w, patch_dim) / max(1, input_elems)).contiguous()
  return pixel_values.clone().realize()


def run_visual_grid_benchmark(
  model_dir: Path,
  grid_thws: list[tuple[int, int, int]],
  max_blocks: int | None,
  realize: bool,
  visual_safetensors: Path | None = None,
  repeat_forwards: int = 1,
  use_jit: bool = False,
  jit_prune: bool = False,
  input_mode: str = "ramp",
  stream_jsonl: bool = False,
  jit_batch_size: int | None = None,
) -> dict:
  if jit_batch_size is not None:
    JIT_BATCH_SIZE.value = jit_batch_size
  config = _load_config(model_dir)["vision_config"]
  model = Qwen2_5_VLVisionTinygrad(config, max_blocks=max_blocks)
  load_start = time.perf_counter()
  loaded = load_state_dict(
    model,
    _load_visual_state(model_dir, max_blocks=max_blocks, visual_safetensors=visual_safetensors),
    strict=True,
    consume=True,
    realize=realize,
  )
  load_ms = (time.perf_counter() - load_start) * 1000.0

  results = []
  for grid_thw in grid_thws:
    pixel_values = _make_pixel_values(config, grid_thw, input_mode)
    cache = model.build_grid_cache([grid_thw])
    forward = TinyJit(lambda px, cache=cache: model.forward_cached(px, cache), prune=jit_prune) if use_jit else lambda px, cache=cache: model.forward_cached(px, cache)
    forward_ms: list[float] = []
    out = None
    for _ in range(max(1, repeat_forwards)):
      _sync_device()
      fwd_start = time.perf_counter()
      out = forward(pixel_values).realize()
      _sync_device()
      forward_ms.append((time.perf_counter() - fwd_start) * 1000.0)
    assert out is not None
    results.append({
      "grid_thw": list(grid_thw),
      "output_shape": list(out.shape),
      "forward_ms": forward_ms[-1],
      "forward_ms_all": forward_ms,
    })
    if stream_jsonl:
      print(json.dumps({
        "event": "grid_result",
        "backend_device": Device.DEFAULT,
        "grid_thw": list(grid_thw),
        "max_blocks": max_blocks,
        "loaded_tensors": len(loaded),
        "loaded_bytes": int(sum(t.nbytes() for t in loaded)),
        "load_ms": load_ms,
        "use_jit": use_jit,
        "jit_batch_size": JIT_BATCH_SIZE.value,
        "jit_prune": jit_prune,
        "input_mode": input_mode,
        "output_shape": list(out.shape),
        "forward_ms": forward_ms[-1],
        "forward_ms_all": forward_ms,
      }, separators=(",", ":")), flush=True)

  return {
    "backend_device": Device.DEFAULT,
    "max_blocks": max_blocks,
    "loaded_tensors": len(loaded),
    "loaded_bytes": int(sum(t.nbytes() for t in loaded)),
    "load_ms": load_ms,
    "use_jit": use_jit,
    "jit_batch_size": JIT_BATCH_SIZE.value,
    "jit_prune": jit_prune,
    "input_mode": input_mode,
    "results": results,
  }


def run_label_pack(
  model_dir: Path,
  label_pack: Path,
  max_vision_blocks: int | None,
  max_text_layers: int | None,
  realize: bool,
  visual_safetensors: Path | None = None,
  text_safetensors: Path | None = None,
  repeat_forwards: int = 1,
  use_jit: bool = False,
  jit_prune: bool = False,
  jit_batch_size: int | None = None,
  pretranspose_linear: bool = True,
) -> dict:
  if jit_batch_size is not None:
    JIT_BATCH_SIZE.value = jit_batch_size
  with open(label_pack, "r", encoding="utf-8") as f:
    pack = json.load(f)

  config = _load_config(model_dir)
  packed_embeddings = "input_embeds" in pack and "candidate_weight" in pack
  global _QWEN_LINEAR_LOAD_IN_OUT
  old_linear_load_in_out = _QWEN_LINEAR_LOAD_IN_OUT
  _QWEN_LINEAR_LOAD_IN_OUT = pretranspose_linear
  try:
    scorer = Qwen2_5_VLLabelScorerTinygrad(
      config,
      max_vision_blocks=max_vision_blocks,
      max_text_layers=max_text_layers,
      with_text_embeddings=not packed_embeddings,
    )
  finally:
    _QWEN_LINEAR_LOAD_IN_OUT = old_linear_load_in_out
  load_start = time.perf_counter()
  loaded = load_state_dict(
    scorer,
    _load_label_state(
      model_dir,
      max_vision_blocks=max_vision_blocks,
      max_text_layers=max_text_layers,
      visual_safetensors=visual_safetensors,
      text_safetensors=text_safetensors,
      skip_text_embeddings=packed_embeddings,
      pretranspose_linear=pretranspose_linear,
    ),
    strict=True,
    consume=True,
    realize=realize,
  )
  loaded_count = len(loaded)
  loaded_bytes = int(sum(t.nbytes() for t in loaded))
  pretranspose_ms = 0.0
  pretranspose_count = sum(1 for _ in _iter_qwen_linears(scorer)) if pretranspose_linear else 0
  if pretranspose_linear and not all(linear._weight_in_out for linear in _iter_qwen_linears(scorer)):
    pretranspose_start = time.perf_counter()
    pretranspose_count = _prepare_linear_inference_layout(scorer, realize=realize)
    pretranspose_ms = (time.perf_counter() - pretranspose_start) * 1000.0
  if pretranspose_linear:
    loaded.clear()
    gc.collect()
  load_ms = (time.perf_counter() - load_start) * 1000.0

  grid_thw = [tuple(int(v) for v in row) for row in pack["image_grid_thw"]]
  vision_cache = scorer.build_vision_cache(grid_thw)
  input_ids = Tensor(pack["input_ids"], dtype=dtypes.int32).realize()
  attention_mask_values = pack["attention_mask"]
  attention_mask_all_ones = all(int(value) == 1 for row in attention_mask_values for value in row)
  attention_mask = (
    None
    if attention_mask_all_ones and os.getenv("QWEN_USE_FLASH_ATTN") == "1"
    else Tensor(attention_mask_values, dtype=dtypes.int32).realize()
  )
  position_ids = Tensor(pack["position_ids"], dtype=dtypes.int32).realize()
  pixel_values = Tensor(pack["pixel_values"], dtype=dtypes.float32).realize()
  image_index = Tensor(pack["image_index"], dtype=dtypes.int32).realize()
  text_embeds = Tensor(pack["input_embeds"], dtype=dtypes.float16).realize() if packed_embeddings else None
  candidate_weight = Tensor(pack["candidate_weight"], dtype=dtypes.float16).realize() if packed_embeddings else None
  candidate_ids = [int(v) for v in pack["yes_ids"]] + [int(v) for v in pack["no_ids"]]
  yes_count = len(pack["yes_ids"])
  score_labels = [str(label) for label in pack["score_labels"]]

  def forward_unmasked(ids: Tensor, pixels: Tensor, img_idx: Tensor, pos: Tensor) -> Tensor:
    if text_embeds is not None and candidate_weight is not None:
      return scorer.score_packed_candidates(ids, text_embeds, pixels, img_idx, pos, None, vision_cache, candidate_weight)
    return scorer.score_candidates(ids, pixels, img_idx, pos, None, vision_cache, candidate_ids)

  def forward_masked(ids: Tensor, pixels: Tensor, img_idx: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
    if text_embeds is not None and candidate_weight is not None:
      return scorer.score_packed_candidates(ids, text_embeds, pixels, img_idx, pos, mask, vision_cache, candidate_weight)
    return scorer.score_candidates(ids, pixels, img_idx, pos, mask, vision_cache, candidate_ids)

  forward_fn = TinyJit(forward_unmasked, prune=jit_prune) if use_jit and attention_mask is None else (
    TinyJit(forward_masked, prune=jit_prune) if use_jit else (forward_unmasked if attention_mask is None else forward_masked)
  )
  forward_ms: list[float] = []
  logits = None
  for _ in range(max(1, repeat_forwards)):
    _sync_device()
    fwd_start = time.perf_counter()
    if attention_mask is None:
      logits = forward_fn(input_ids, pixel_values, image_index, position_ids).realize()
    else:
      logits = forward_fn(input_ids, pixel_values, image_index, position_ids, attention_mask).realize()
    _sync_device()
    forward_ms.append((time.perf_counter() - fwd_start) * 1000.0)
  assert logits is not None
  logits_list = logits.tolist()

  label_scores: dict[str, float] = {}
  selected: list[str] = []
  for idx, label in enumerate(score_labels):
    row = logits_list[idx]
    yes_score = max(float(v) for v in row[:yes_count])
    no_score = max(float(v) for v in row[yes_count:])
    score = yes_score - no_score
    label_scores[label] = score
    if score >= 0.0:
      selected.append(label)

  return {
    "backend_device": Device.DEFAULT,
    "source_image": pack.get("source_image"),
    "image_mode": pack.get("image_mode"),
    "image_size": pack.get("image_size"),
    "score_labels": score_labels,
    "labels": selected if selected else ["none"],
    "label_scores": label_scores,
    "pack_processor_prefill_ms": pack.get("processor_prefill_ms"),
    "packed_embeddings": packed_embeddings,
    "attention_mask_all_ones": attention_mask_all_ones,
    "input_shape": [len(pack["input_ids"]), len(pack["input_ids"][0])],
    "pixel_values_shape": [len(pack["pixel_values"]), len(pack["pixel_values"][0]) if pack["pixel_values"] else 0],
    "image_grid_thw": pack["image_grid_thw"],
    "max_vision_blocks": max_vision_blocks,
    "max_text_layers": max_text_layers,
    "loaded_tensors": loaded_count,
    "loaded_bytes": loaded_bytes,
    "load_ms": load_ms,
    "pretranspose_linear": pretranspose_linear,
    "pretranspose_linear_count": pretranspose_count,
    "pretranspose_linear_ms": pretranspose_ms,
    "use_jit": use_jit,
    "jit_batch_size": JIT_BATCH_SIZE.value,
    "jit_prune": jit_prune,
    "forward_ms": forward_ms[-1],
    "forward_ms_all": forward_ms,
  }


def _parse_grid_thw(value: str) -> tuple[int, int, int]:
  grid_thw = tuple(int(part) for part in value.split(","))
  if len(grid_thw) != 3:
    raise ValueError("grid must have exactly three comma-separated integers")
  return grid_thw


def main() -> None:
  parser = argparse.ArgumentParser(description="Tinygrad Qwen2.5-VL visual tower and image-label scoring tools.")
  parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
  parser.add_argument("--device", default=None, help="Set Device.DEFAULT, for example CPU, AMD, or USB+AMD.")
  parser.add_argument("--grid-thw", default="1,12,12", help="Visual grid as t,h,w. 1,12,12 is the measured C3X/eGPU hot default.")
  parser.add_argument("--grid-thw-list", default=None, help="Semicolon-separated visual grids, for example 1,16,16;1,20,20.")
  parser.add_argument("--stream-jsonl", action="store_true", help="For --grid-thw-list, print one compact JSON line after each grid finishes.")
  parser.add_argument("--max-vision-blocks", type=int, default=1, help="Number of visual transformer blocks to load/run. Use -1 for all blocks.")
  parser.add_argument("--max-text-layers", type=int, default=-1, help="Number of text decoder layers to load/run. Use -1 for all layers.")
  parser.add_argument("--repeat-forwards", type=int, default=1, help="Run this many forwards after one model load to measure steady state.")
  parser.add_argument("--use-jit", action="store_true", help="Wrap the cached visual forward in TinyJit.")
  parser.add_argument("--jit-batch-size", type=int, default=1024, help="Tinygrad JIT graph batch size. Larger values reduce USB AMD launch overhead.")
  parser.add_argument("--jit-prune", action="store_true", help="Allow TinyJit to precompute graph parts independent of tensor inputs.")
  parser.add_argument("--input-mode", choices=("ramp", "zeros"), default="ramp", help="Synthetic pixel input for smoke testing.")
  parser.add_argument("--export-weight-dtype", choices=("float16", "native"), default="native", help="Optional dtype conversion when exporting a visual subset.")
  parser.add_argument("--skip-text-embeddings", action="store_true", help="When exporting text weights, omit embed/lm-head weights for packed fixed prompts.")
  parser.add_argument("--no-realize-load", action="store_true", help="Do not realize weights during load_state_dict.")
  parser.add_argument("--visual-safetensors", type=Path, default=None, help="Load visual weights from this visual-only safetensors file.")
  parser.add_argument("--text-safetensors", type=Path, default=None, help="Load text weights from this text-only safetensors file.")
  parser.add_argument("--export-visual-subset", type=Path, default=None, help="Write a visual-only safetensors subset for C3X smoke tests.")
  parser.add_argument("--export-text-subset", type=Path, default=None, help="Write a text-only safetensors subset for C3X label scoring.")
  parser.add_argument("--export-label-pack", type=Path, default=None, help="Write a preprocessed image-label request pack for C3X tinygrad execution.")
  parser.add_argument("--label-pack", type=Path, default=None, help="Run a preprocessed image-label request pack through visual and text towers.")
  parser.add_argument("--image", type=Path, default=None, help="Source image for --export-label-pack.")
  parser.add_argument("--image-mode", choices=("full", "composite", "multi"), default="full", help="Image packing mode used for --export-label-pack.")
  parser.add_argument("--image-size", type=int, default=168, help="Rendered image side length for --export-label-pack. 168 maps to Qwen grid 1,12,12.")
  parser.add_argument("--score-labels", default="construction_left,construction_right", help="Comma-separated labels to score in --export-label-pack.")
  parser.add_argument("--vehicle-state", default="speed=5.0mps curvature=0.0 blinkers=off", help="Vehicle state text embedded in score prompts.")
  parser.add_argument("--pack-embeddings", action="store_true", help="Store fixed prompt token embeddings and yes/no candidate weights inside --export-label-pack.")
  parser.add_argument("--no-pretranspose-linear", action="store_true", help="Keep HF (out,in) linear weight layout instead of pretransposing once for inference.")
  parser.add_argument("--list-visual-keys", action="store_true", help="Print visual key counts in the HF safetensors and exit.")
  parser.add_argument("--hard-exit", action="store_true", help="Exit with os._exit(0) after printing JSON to bypass USB runtime teardown hangs.")
  args = parser.parse_args()

  if args.device:
    os.environ["DEV"] = args.device
    DEV.value = args.device

  max_blocks = None if args.max_vision_blocks < 0 else args.max_vision_blocks
  max_text_layers = None if args.max_text_layers < 0 else args.max_text_layers

  if args.list_visual_keys:
    total = 0
    for fn in sorted(args.model_dir.glob("*.safetensors")):
      keys = [key for key in _metadata_keys(fn) if _keep_visual_key(key, max_blocks)]
      total += len(keys)
      print(f"{fn.name}: {len(keys)} visual keys")
    print(f"total visual keys: {total}")
    return

  if args.export_visual_subset is not None:
    export_visual_subset(args.model_dir, args.export_visual_subset, max_blocks, args.export_weight_dtype)
    print(json.dumps({"exported": str(args.export_visual_subset), "max_blocks": max_blocks, "weight_dtype": args.export_weight_dtype}, separators=(",", ":")))
    return

  if args.export_text_subset is not None:
    export_text_subset(args.model_dir, args.export_text_subset, max_text_layers, args.export_weight_dtype, skip_embeddings=args.skip_text_embeddings)
    print(json.dumps({"exported": str(args.export_text_subset), "max_text_layers": max_text_layers, "weight_dtype": args.export_weight_dtype, "skip_text_embeddings": args.skip_text_embeddings}, separators=(",", ":")))
    return

  if args.export_label_pack is not None:
    if args.image is None:
      parser.error("--export-label-pack requires --image")
    labels = tuple(label.strip() for label in args.score_labels.split(",") if label.strip())
    export_label_pack(args.model_dir, args.image, args.export_label_pack, labels, args.image_mode, args.image_size, args.vehicle_state, pack_embeddings=args.pack_embeddings)
    print(json.dumps({"exported": str(args.export_label_pack), "image": str(args.image), "score_labels": list(labels), "image_mode": args.image_mode, "image_size": args.image_size, "pack_embeddings": args.pack_embeddings}, separators=(",", ":")))
    return

  if args.label_pack is not None:
    result = run_label_pack(
      args.model_dir,
      args.label_pack,
      max_blocks,
      max_text_layers,
      realize=not args.no_realize_load,
      visual_safetensors=args.visual_safetensors,
      text_safetensors=args.text_safetensors,
      repeat_forwards=args.repeat_forwards,
      use_jit=args.use_jit,
      jit_prune=args.jit_prune,
      jit_batch_size=args.jit_batch_size,
      pretranspose_linear=not args.no_pretranspose_linear,
    )
  elif args.grid_thw_list:
    result = run_visual_grid_benchmark(
      args.model_dir,
      [_parse_grid_thw(part.strip()) for part in args.grid_thw_list.split(";") if part.strip()],
      max_blocks,
      realize=not args.no_realize_load,
      visual_safetensors=args.visual_safetensors,
      repeat_forwards=args.repeat_forwards,
      use_jit=args.use_jit,
      jit_prune=args.jit_prune,
      input_mode=args.input_mode,
      stream_jsonl=args.stream_jsonl,
      jit_batch_size=args.jit_batch_size,
    )
  else:
    result = run_visual_smoke(
      args.model_dir,
      _parse_grid_thw(args.grid_thw),
      max_blocks,
      realize=not args.no_realize_load,
      visual_safetensors=args.visual_safetensors,
      repeat_forwards=args.repeat_forwards,
      use_jit=args.use_jit,
      jit_prune=args.jit_prune,
      input_mode=args.input_mode,
      jit_batch_size=args.jit_batch_size,
    )
  print(json.dumps(result, indent=2), flush=True)
  if args.hard_exit:
    os._exit(0)


if __name__ == "__main__":
  main()
