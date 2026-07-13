#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch


__all__ = [
  "load_dflash_draft_model",
  "dflash_generate_alpamayo",
]


def _module_device(module: Any) -> Any:
  try:
    return next(module.parameters()).device
  except StopIteration as exc:
    raise RuntimeError(f"module has no parameters: {module}") from exc


def _resolve_path(raw: os.PathLike[str] | str | Path) -> Path:
  return Path(os.fspath(raw))


def _load_mask_embedding(
  torch_mod: Any,
  draft_model_dir: Path,
  *,
  device: Any,
  dtype: Any,
) -> Any:
  path = draft_model_dir / "mask_embedding.pt"
  if not path.exists():
    raise FileNotFoundError(f"DFlash mask embedding is required but missing: {path}")

  try:
    value = torch_mod.load(str(path), map_location="cpu", weights_only=True)
  except TypeError:
    value = torch_mod.load(str(path), map_location="cpu")

  if isinstance(value, dict):
    if "mask_embedding" in value:
      value = value["mask_embedding"]
    elif len(value) == 1:
      value = next(iter(value.values()))
    else:
      raise ValueError(
        f"cannot identify mask embedding in {path}: keys={list(value.keys())}"
      )

  if getattr(value, "ndim", None) != 1:
    raise ValueError(f"mask embedding must be rank-1, got shape={getattr(value, 'shape', None)}")
  return value.to(device=device, dtype=dtype)


def _load_dflash_model(
  torch_mod: Any,
  model_dir: Path,
  *,
  dtype: Any,
  attn_implementation: str,
) -> Any:
  from dflash.model import DFlashDraftModel

  try:
    model = DFlashDraftModel.from_pretrained(
      str(model_dir),
      attn_implementation=attn_implementation,
      dtype=dtype,
    )
    return model
  except AttributeError as exc:
    if "dflash_config" not in str(exc):
      raise

  from safetensors.torch import load_file
  from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

  raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
  config = Qwen3Config(**raw_config)
  config.dflash_config = raw_config
  config._attn_implementation = attn_implementation
  for key, value in raw_config.items():
    if not hasattr(config, key):
      setattr(config, key, value)

  model = DFlashDraftModel(config)
  state = load_file(str(model_dir / "model.safetensors"), device="cpu")
  missing, unexpected = model.load_state_dict(state, strict=False)
  if missing or unexpected:
    raise RuntimeError(f"DFlash state dict mismatch: missing={missing}, unexpected={unexpected}")
  return model.to(dtype=dtype)


def load_dflash_draft_model(
  torch_mod: Any,
  draft_model_path: os.PathLike[str] | str | Path,
  package_root: os.PathLike[str] | str | Path,
  device: Any,
  dtype: Any,
  attn_implementation: str,
) -> tuple[Any, Any, list[int]]:
  model_dir = _resolve_path(draft_model_path)
  root = _resolve_path(package_root)

  if not model_dir.exists():
    raise FileNotFoundError(f"DFlash draft model not found: {model_dir}")
  if not root.exists():
    raise FileNotFoundError(f"DFlash package root not found: {root}")

  inserted_root = False
  root_str = str(root)
  if root_str not in sys.path:
    sys.path.insert(0, root_str)
    inserted_root = True

  try:
    draft = _load_dflash_model(
      torch_mod=torch_mod,
      model_dir=model_dir,
      dtype=dtype,
      attn_implementation=attn_implementation,
    )
  finally:
    if inserted_root:
      try:
        sys.path.remove(root_str)
      except ValueError:
        pass

  draft = draft.to(device=device, dtype=dtype).eval()
  mask_embedding = _load_mask_embedding(
    torch_mod,
    model_dir,
    device=device,
    dtype=dtype,
  )
  layer_ids = [int(item) for item in getattr(draft, "target_layer_ids", [])]
  return draft, mask_embedding, layer_ids


def _sample(torch_mod: Any, logits: Any, temperature: float) -> Any:
  if temperature < 1e-5:
    return torch_mod.argmax(logits, dim=-1)
  bsz, seq_len, vocab_size = logits.shape
  probs = torch_mod.softmax(logits.view(-1, vocab_size) / temperature, dim=-1)
  return torch_mod.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _mask_traj_token_logits(model: Any, logits: Any) -> Any:
  masked = logits.clone()
  offset = int(model.config.traj_token_start_idx)
  size = int(model.config.traj_vocab_size)
  masked[..., offset : offset + size] = float("-inf")
  return masked


def _hidden_from_layer_output(output: Any) -> Any:
  if isinstance(output, (tuple, list)):
    return output[0]
  if hasattr(output, "last_hidden_state"):
    return output.last_hidden_state
  return output


def _selected_hidden_from_output(output: Any, layer_ids: Sequence[int], num_layers: int) -> Any | None:
  hidden_states = getattr(output, "hidden_states", None)
  if hidden_states is None and isinstance(output, dict):
    hidden_states = output.get("hidden_states")
  if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
    return None

  selected = []
  has_embedding_state = len(hidden_states) == num_layers + 1
  for layer_id in layer_ids:
    output_idx = int(layer_id) + 1 if has_embedding_state else int(layer_id)
    if output_idx < 0 or output_idx >= len(hidden_states):
      return None
    hidden = hidden_states[output_idx]
    if hidden is None:
      return None
    selected.append(hidden)
  target_device = selected[-1].device
  return torch.cat([hidden.to(target_device) for hidden in selected], dim=-1)


def _qwen3vl_forward(
  torch_mod: Any,
  target: Any,
  ids: Any,
  *,
  tokenized_data: dict[str, Any] | None,
  past_key_values: Any,
  output_hidden_states: bool,
  logits_to_keep: int = 0,
) -> Any:
  past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
  cache_position = torch_mod.arange(past_len, past_len + ids.shape[1], device=ids.device)
  kwargs: dict[str, Any] = {
    "input_ids": ids,
    "past_key_values": past_key_values,
    "use_cache": True,
    "cache_position": cache_position,
    "output_hidden_states": output_hidden_states,
    "logits_to_keep": logits_to_keep,
    "return_dict": True,
  }
  if past_len == 0 and tokenized_data is not None:
    kwargs.update(copy.copy(tokenized_data))
  return target(**kwargs)


def _qwen3vl_forward_with_selected_hidden(
  torch_mod: Any,
  target: Any,
  ids: Any,
  *,
  tokenized_data: dict[str, Any] | None,
  past_key_values: Any,
  layer_ids: Sequence[int],
  logits_to_keep: int = 0,
  runtime_profile: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
  layers = target.model.language_model.layers

  for layer_id in layer_ids:
    if layer_id >= len(layers):
      raise RuntimeError(
        f"DFlash target layer {layer_id} out of range for {len(layers)} language layers"
      )

  direct_error = ""
  try:
    output = _qwen3vl_forward(
      torch_mod,
      target,
      ids,
      tokenized_data=tokenized_data,
      past_key_values=past_key_values,
      output_hidden_states=True,
      logits_to_keep=logits_to_keep,
    )
    selected = _selected_hidden_from_output(output, layer_ids, len(layers))
    if selected is not None:
      try:
        output.hidden_states = None
      except Exception:
        pass
      if runtime_profile is not None:
        runtime_profile["dflash_selected_hidden_capture_mode"] = "hidden_states"
        runtime_profile["dflash_selected_hidden_hook_fallback"] = 0
      return output, selected
    direct_error = "missing_hidden_states"
  except Exception as exc:
    direct_error = f"{type(exc).__name__}: {exc}"

  captured: dict[int, Any] = {}
  hooks = []
  for layer_id in layer_ids:
    def hook(_module: Any, _inputs: Any, output: Any, *, layer_id: int = layer_id) -> None:
      captured[layer_id] = _hidden_from_layer_output(output)

    hooks.append(layers[layer_id].register_forward_hook(hook))

  try:
    output = _qwen3vl_forward(
      torch_mod,
      target,
      ids,
      tokenized_data=tokenized_data,
      past_key_values=past_key_values,
      output_hidden_states=False,
      logits_to_keep=logits_to_keep,
    )
  finally:
    for handle in hooks:
      handle.remove()

  missing = [layer_id for layer_id in layer_ids if layer_id not in captured]
  if missing:
    raise RuntimeError(f"did not capture DFlash target layers: {missing}")

  target_device = captured[layer_ids[-1]].device
  selected = [captured[layer_id].to(target_device) for layer_id in layer_ids]
  if runtime_profile is not None:
    runtime_profile["dflash_selected_hidden_capture_mode"] = "hooks"
    runtime_profile["dflash_selected_hidden_hook_fallback"] = 1
    runtime_profile["dflash_selected_hidden_direct_error"] = direct_error
  return output, torch.cat(selected, dim=-1)


def _embed_block_tokens(
  torch_mod: Any,
  embed_tokens: Any,
  block_output_ids: Any,
  *,
  mask_token_id: int,
  mask_embedding: Any,
  output_device: Any,
) -> Any:
  embed_device = _module_device(embed_tokens)
  mask = block_output_ids == mask_token_id
  safe_ids = block_output_ids.masked_fill(mask, 0)
  embeds = embed_tokens(safe_ids.to(embed_device)).to(output_device)
  replacement = mask_embedding.to(device=output_device, dtype=embeds.dtype).view(1, 1, -1)
  return torch_mod.where(mask.to(output_device).unsqueeze(-1), replacement, embeds)


def _sync(torch_mod: Any) -> None:
  if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
    torch_mod.cuda.synchronize()


def _position_seed_buffer(
  owner: Any,
  attr_name: str,
  torch_mod: Any,
  *,
  device: Any,
  length: int,
) -> Any:
  requested_len = max(1, int(length))
  buffer = getattr(owner, attr_name, None)
  len_attr = f"{attr_name}_len"
  device_attr = f"{attr_name}_device"
  current_len = int(getattr(owner, len_attr, 0) or 0)
  current_device = getattr(owner, device_attr, None)
  if buffer is None or current_len < requested_len or current_device != device:
    buffer = torch_mod.arange(requested_len, device=device, dtype=torch_mod.long)
    setattr(owner, attr_name, buffer)
    setattr(owner, len_attr, requested_len)
    setattr(owner, device_attr, device)
  return buffer[:requested_len]


def _cpu_long_buffer(owner: Any, attr_name: str, torch_mod: Any, shape: tuple[int, ...], dtype: Any) -> Any:
  buffer = getattr(owner, attr_name, None)
  if buffer is None or tuple(buffer.shape) != tuple(shape) or buffer.dtype != dtype:
    buffer = torch_mod.empty(shape, dtype=dtype, device="cpu")
    setattr(owner, attr_name, buffer)
  return buffer


def _cache_tensor_refs(cache: Any) -> list[Any]:
  refs: list[Any] = []
  for attr in ("key_cache", "value_cache"):
    values = getattr(cache, attr, None)
    if isinstance(values, (list, tuple)):
      refs.extend(item for item in values if hasattr(item, "shape"))
  return refs


def _cache_signature(cache: Any) -> tuple[Any, ...]:
  return tuple(
    (tuple(item.shape), str(item.dtype), str(item.device))
    for item in _cache_tensor_refs(cache)
  )


def _copy_cache_tensors(dst: Any, src: Any) -> str | None:
  if dst is src:
    return None
  dst_refs = _cache_tensor_refs(dst)
  src_refs = _cache_tensor_refs(src)
  if not dst_refs or len(dst_refs) != len(src_refs):
    return "draft_cache_tensor_layout_mismatch"
  for dst_tensor, src_tensor in zip(dst_refs, src_refs):
    if (
      tuple(dst_tensor.shape) != tuple(src_tensor.shape)
      or dst_tensor.dtype != src_tensor.dtype
      or dst_tensor.device != src_tensor.device
    ):
      return "draft_cache_tensor_shape_dtype_device_mismatch"
    if dst_tensor is src_tensor:
      continue
    try:
      same_storage = (
        hasattr(dst_tensor, "data_ptr")
        and hasattr(src_tensor, "data_ptr")
        and int(dst_tensor.data_ptr()) == int(src_tensor.data_ptr())
      )
    except Exception:
      same_storage = False
    if same_storage:
      continue
    dst_tensor.copy_(src_tensor, non_blocking=True)
  for attr in ("_seen_tokens", "seen_tokens"):
    if hasattr(src, attr) and hasattr(dst, attr):
      try:
        setattr(dst, attr, getattr(src, attr))
      except Exception:
        pass
  return None


def _pooled_live_cache_copy(
  owner: dict[str, Any],
  source_cache: Any,
  *,
  pool_key: str,
  pool_size: int = 4,
) -> tuple[Any, str]:
  pool = owner.get(pool_key)
  if not isinstance(pool, list):
    pool = []
    owner[pool_key] = pool
    owner[f"{pool_key}_next"] = 0
  next_idx = int(owner.get(f"{pool_key}_next", 0) or 0)
  if len(pool) < max(1, int(pool_size)):
    live_cache = copy.deepcopy(source_cache)
    pool.append(live_cache)
    owner[f"{pool_key}_next"] = len(pool) % max(1, int(pool_size))
    return live_cache, "pool_alloc"

  pool_idx = next_idx % len(pool)
  live_cache = pool[pool_idx]
  copy_error = _copy_cache_tensors(live_cache, source_cache)
  if copy_error is not None:
    live_cache = copy.deepcopy(source_cache)
    pool[pool_idx] = live_cache
    mode = f"pool_replace:{copy_error}"
  else:
    mode = "pool_reuse"
  owner[f"{pool_key}_next"] = (pool_idx + 1) % len(pool)
  return live_cache, mode


def _dflash_decode_graph_requested(model: Any) -> bool:
  requested = getattr(model, "_openpilot_graph_stage_requested", {})
  return bool(isinstance(requested, dict) and requested.get("decode"))


def _dflash_prefill_graph_requested(model: Any) -> bool:
  requested = getattr(model, "_openpilot_graph_stage_requested", {})
  return bool(isinstance(requested, dict) and requested.get("prefill"))


def _dflash_strict_graph_required(model: Any, stage: str) -> bool:
  requested = getattr(model, "_openpilot_graph_stage_requested", {})
  return bool(
    getattr(model, "_openpilot_static_graph_strict_shapes", False)
    and isinstance(requested, dict)
    and requested.get(stage)
  )


def _tensor_kwargs_signature(kwargs: dict[str, Any]) -> tuple[Any, ...]:
  return tuple(
    sorted(
      (
        key,
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
      )
      for key, value in kwargs.items()
      if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device")
    )
  )


def _dflash_selected_hidden_prefill_with_cuda_graph(
  torch_mod: Any,
  model: Any,
  target: Any,
  *,
  input_ids: Any,
  tokenized_data: dict[str, Any],
  target_cache: Any,
  layer_ids: Sequence[int],
  runtime_profile: dict[str, Any] | None,
) -> tuple[Any, Any, Any] | None:
  def fallback(reason: str) -> None:
    if runtime_profile is not None:
      runtime_profile["dflash_prefill_graph_mode"] = "fallback"
      runtime_profile["dflash_prefill_graph_error"] = reason
      runtime_profile["dflash_prefill_graph_ready"] = 0

  if not _dflash_prefill_graph_requested(model):
    return None
  if not hasattr(torch_mod, "cuda") or not torch_mod.cuda.is_available():
    fallback("cuda_unavailable")
    return None
  if not getattr(input_ids, "is_cuda", False):
    fallback("non_cuda_prefill_input_ids")
    return None

  layers = target.model.language_model.layers
  for layer_id in layer_ids:
    if layer_id >= len(layers):
      fallback(f"target_layer_out_of_range:{layer_id}>={len(layers)}")
      return None

  stage_caches = getattr(model, "_openpilot_graph_stage_caches", None)
  if not isinstance(stage_caches, dict):
    fallback("missing_stage_cache")
    return None
  graph_cache = stage_caches.get("prefill")
  if graph_cache is None or not hasattr(graph_cache, "get"):
    fallback("missing_prefill_cache")
    return None

  graph_kwargs = copy.copy(tokenized_data)
  graph_kwargs["output_hidden_states"] = True
  tensor_signature = _tensor_kwargs_signature(graph_kwargs)
  cache_key = (
    "dflash_selected_hidden_prefill",
    id(target),
    tuple(input_ids.shape),
    str(input_ids.dtype),
    str(input_ids.device),
    tuple(int(item) for item in layer_ids),
    tensor_signature,
  )
  max_entries = max(1, int(getattr(model, "_openpilot_graph_stage_cache_size", 4)))
  entry = graph_cache.get(cache_key)
  cache_hit = entry is not None

  try:
    if entry is None:
      static_input_ids = torch_mod.empty_like(input_ids)
      static_tokenized_data: dict[str, Any] = {}
      for key, value in tokenized_data.items():
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
          static_tokenized_data[key] = torch_mod.empty_like(value)
        else:
          static_tokenized_data[key] = value
      static_target_cache = copy.deepcopy(target_cache)
      static_input_ids.copy_(input_ids, non_blocking=True)
      for key, value in tokenized_data.items():
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
          static_tokenized_data[key].copy_(value, non_blocking=True)

      def run_static() -> tuple[Any, Any]:
        static_output = _qwen3vl_forward(
          torch_mod,
          target,
          static_input_ids,
          tokenized_data=static_tokenized_data,
          past_key_values=static_target_cache,
          output_hidden_states=True,
          logits_to_keep=1,
        )
        static_selected = _selected_hidden_from_output(static_output, layer_ids, len(layers))
        if static_selected is None:
          raise RuntimeError("missing_hidden_states")
        try:
          static_output.hidden_states = None
        except Exception:
          pass
        return static_output, static_selected

      current_stream = torch_mod.cuda.current_stream(input_ids.device)
      warmup_stream = torch_mod.cuda.Stream(device=input_ids.device)
      warmup_stream.wait_stream(current_stream)
      with torch_mod.cuda.stream(warmup_stream):
        warmup_cache = copy.deepcopy(target_cache)
        warmup_output = _qwen3vl_forward(
          torch_mod,
          target,
          static_input_ids,
          tokenized_data=static_tokenized_data,
          past_key_values=warmup_cache,
          output_hidden_states=True,
          logits_to_keep=1,
        )
        warmup_selected = _selected_hidden_from_output(warmup_output, layer_ids, len(layers))
        if warmup_selected is None:
          raise RuntimeError("missing_hidden_states")
      current_stream.wait_stream(warmup_stream)

      graph = torch_mod.cuda.CUDAGraph()
      with torch_mod.cuda.graph(graph):
        static_output, static_selected = run_static()
      entry = {
        "graph": graph,
        "input_ids": static_input_ids,
        "tokenized_data": static_tokenized_data,
        "target_cache": static_target_cache,
        "output": static_output,
        "selected_hidden": static_selected,
      }
      graph_cache[cache_key] = entry
      graph_cache.move_to_end(cache_key)
      while len(graph_cache) > max_entries:
        graph_cache.popitem(last=False)
    else:
      graph_cache.move_to_end(cache_key)

    entry["input_ids"].copy_(input_ids, non_blocking=True)
    for key, value in tokenized_data.items():
      if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        entry["tokenized_data"][key].copy_(value, non_blocking=True)
    entry["graph"].replay()

    live_target_cache, live_cache_mode = _pooled_live_cache_copy(
      entry,
      entry["target_cache"],
      pool_key="live_target_cache_pool",
    )
    output = SimpleNamespace(
      logits=entry["output"].logits,
      past_key_values=live_target_cache,
    )
    selected_hidden = entry["selected_hidden"]
    if runtime_profile is not None:
      runtime_profile["dflash_prefill_graph_mode"] = "replay" if cache_hit else "capture"
      runtime_profile["dflash_prefill_graph_ready"] = 1
      runtime_profile["dflash_prefill_graph_cache_hit"] = 1 if cache_hit else 0
      runtime_profile["dflash_prefill_graph_cache_depth"] = int(len(graph_cache))
      runtime_profile["dflash_prefill_graph_calls"] = int(runtime_profile.get("dflash_prefill_graph_calls", 0)) + 1
      runtime_profile["dflash_prefill_graph_live_cache_mode"] = live_cache_mode
      runtime_profile["dflash_prefill_graph_selected_hidden_owner"] = "graph_static_borrowed"
    return output, selected_hidden, live_target_cache
  except Exception as exc:
    try:
      graph_cache.pop(cache_key, None)
    except Exception:
      pass
    fallback(f"{type(exc).__name__}: {exc}")
    return None


def _dflash_draft_block_with_cuda_graph(
  torch_mod: Any,
  model: Any,
  draft_model: Any,
  *,
  target_hidden: Any,
  noise_embedding: Any,
  position_ids: Any,
  draft_cache: Any,
  runtime_profile: dict[str, Any] | None,
) -> Any | None:
  def fallback(reason: str) -> None:
    if runtime_profile is not None:
      runtime_profile["dflash_draft_graph_mode"] = "fallback"
      runtime_profile["dflash_draft_graph_error"] = reason
      runtime_profile["dflash_draft_graph_ready"] = 0

  if not _dflash_decode_graph_requested(model):
    return None
  if not hasattr(torch_mod, "cuda") or not torch_mod.cuda.is_available():
    fallback("cuda_unavailable")
    return None
  if not (
    getattr(target_hidden, "is_cuda", False)
    and getattr(noise_embedding, "is_cuda", False)
    and getattr(position_ids, "is_cuda", False)
  ):
    fallback("non_cuda_dflash_draft_inputs")
    return None

  cache_signature = _cache_signature(draft_cache)
  if not cache_signature:
    fallback("empty_draft_cache_signature")
    return None

  stage_caches = getattr(model, "_openpilot_graph_stage_caches", None)
  if not isinstance(stage_caches, dict):
    fallback("missing_stage_cache")
    return None
  graph_cache = stage_caches.get("decode")
  if graph_cache is None or not hasattr(graph_cache, "get"):
    fallback("missing_decode_cache")
    return None

  cache_key = (
    "dflash_draft_block",
    id(draft_model),
    tuple(target_hidden.shape),
    str(target_hidden.dtype),
    str(target_hidden.device),
    tuple(noise_embedding.shape),
    str(noise_embedding.dtype),
    str(noise_embedding.device),
    tuple(position_ids.shape),
    str(position_ids.dtype),
    str(position_ids.device),
    cache_signature,
  )
  max_entries = max(1, int(getattr(model, "_openpilot_graph_stage_cache_size", 4)))
  entry = graph_cache.get(cache_key)
  cache_hit = entry is not None

  try:
    if entry is None:
      static_target_hidden = torch_mod.empty_like(target_hidden)
      static_noise_embedding = torch_mod.empty_like(noise_embedding)
      static_position_ids = torch_mod.empty_like(position_ids)
      static_draft_cache = copy.deepcopy(draft_cache)
      static_target_hidden.copy_(target_hidden, non_blocking=True)
      static_noise_embedding.copy_(noise_embedding, non_blocking=True)
      static_position_ids.copy_(position_ids, non_blocking=True)
      copy_error = _copy_cache_tensors(static_draft_cache, draft_cache)
      if copy_error is not None:
        fallback(copy_error)
        return None

      current_stream = torch_mod.cuda.current_stream(target_hidden.device)
      warmup_stream = torch_mod.cuda.Stream(device=target_hidden.device)
      warmup_stream.wait_stream(current_stream)
      with torch_mod.cuda.stream(warmup_stream):
        warmup_cache = copy.deepcopy(draft_cache)
        draft_model(
          target_hidden=static_target_hidden,
          noise_embedding=static_noise_embedding,
          position_ids=static_position_ids,
          past_key_values=warmup_cache,
          use_cache=True,
          is_causal=False,
        )
      current_stream.wait_stream(warmup_stream)

      graph = torch_mod.cuda.CUDAGraph()
      with torch_mod.cuda.graph(graph):
        static_output = draft_model(
          target_hidden=static_target_hidden,
          noise_embedding=static_noise_embedding,
          position_ids=static_position_ids,
          past_key_values=static_draft_cache,
          use_cache=True,
          is_causal=False,
        )
      entry = {
        "graph": graph,
        "target_hidden": static_target_hidden,
        "noise_embedding": static_noise_embedding,
        "position_ids": static_position_ids,
        "draft_cache": static_draft_cache,
        "output": static_output,
      }
      graph_cache[cache_key] = entry
      graph_cache.move_to_end(cache_key)
      while len(graph_cache) > max_entries:
        graph_cache.popitem(last=False)
    else:
      graph_cache.move_to_end(cache_key)

    entry["target_hidden"].copy_(target_hidden, non_blocking=True)
    entry["noise_embedding"].copy_(noise_embedding, non_blocking=True)
    entry["position_ids"].copy_(position_ids, non_blocking=True)
    copy_error = _copy_cache_tensors(entry["draft_cache"], draft_cache)
    if copy_error is not None:
      graph_cache.pop(cache_key, None)
      fallback(copy_error)
      return None
    entry["graph"].replay()
    if runtime_profile is not None:
      runtime_profile["dflash_draft_graph_mode"] = "replay" if cache_hit else "capture"
      runtime_profile["dflash_draft_graph_ready"] = 1
      runtime_profile["dflash_draft_graph_cache_hit"] = 1 if cache_hit else 0
      runtime_profile["dflash_draft_graph_cache_depth"] = int(len(graph_cache))
      runtime_profile["dflash_draft_graph_calls"] = int(runtime_profile.get("dflash_draft_graph_calls", 0)) + 1
    return entry["output"]
  except Exception as exc:
    try:
      graph_cache.pop(cache_key, None)
    except Exception:
      pass
    fallback(f"{type(exc).__name__}: {exc}")
    return None


def _dflash_target_validation_with_cuda_graph(
  torch_mod: Any,
  model: Any,
  target: Any,
  *,
  ids: Any,
  target_cache: Any,
  layer_ids: Sequence[int],
  runtime_profile: dict[str, Any] | None,
) -> tuple[Any, Any] | None:
  def fallback(reason: str) -> None:
    if runtime_profile is not None:
      runtime_profile["dflash_validate_graph_mode"] = "fallback"
      runtime_profile["dflash_validate_graph_error"] = reason
      runtime_profile["dflash_validate_graph_ready"] = 0

  if not _dflash_decode_graph_requested(model):
    return None
  if not hasattr(torch_mod, "cuda") or not torch_mod.cuda.is_available():
    fallback("cuda_unavailable")
    return None
  if not getattr(ids, "is_cuda", False):
    fallback("non_cuda_validation_ids")
    return None

  cache_signature = _cache_signature(target_cache)
  if not cache_signature:
    fallback("empty_target_cache_signature")
    return None
  target_cache_seq_len = int(target_cache.get_seq_length()) if hasattr(target_cache, "get_seq_length") else -1

  layers = target.model.language_model.layers
  for layer_id in layer_ids:
    if layer_id >= len(layers):
      fallback(f"target_layer_out_of_range:{layer_id}>={len(layers)}")
      return None

  stage_caches = getattr(model, "_openpilot_graph_stage_caches", None)
  if not isinstance(stage_caches, dict):
    fallback("missing_stage_cache")
    return None
  graph_cache = stage_caches.get("decode")
  if graph_cache is None or not hasattr(graph_cache, "get"):
    fallback("missing_decode_cache")
    return None

  cache_key = (
    "dflash_target_validation",
    id(target),
    tuple(ids.shape),
    str(ids.dtype),
    str(ids.device),
    tuple(int(item) for item in layer_ids),
    target_cache_seq_len,
    cache_signature,
  )
  max_entries = max(1, int(getattr(model, "_openpilot_graph_stage_cache_size", 4)))
  entry = graph_cache.get(cache_key)
  cache_hit = entry is not None

  try:
    if entry is None:
      static_ids = torch_mod.empty_like(ids)
      static_target_cache = copy.deepcopy(target_cache)
      static_ids.copy_(ids, non_blocking=True)
      copy_error = _copy_cache_tensors(static_target_cache, target_cache)
      if copy_error is not None:
        fallback(copy_error)
        return None

      def run_static() -> tuple[Any, Any]:
        static_output = _qwen3vl_forward(
          torch_mod,
          target,
          static_ids,
          tokenized_data=None,
          past_key_values=static_target_cache,
          output_hidden_states=True,
        )
        static_selected = _selected_hidden_from_output(static_output, layer_ids, len(layers))
        if static_selected is None:
          raise RuntimeError("missing_hidden_states")
        try:
          static_output.hidden_states = None
        except Exception:
          pass
        return static_output, static_selected

      current_stream = torch_mod.cuda.current_stream(ids.device)
      warmup_stream = torch_mod.cuda.Stream(device=ids.device)
      warmup_stream.wait_stream(current_stream)
      with torch_mod.cuda.stream(warmup_stream):
        warmup_cache = copy.deepcopy(target_cache)
        warmup_output = _qwen3vl_forward(
          torch_mod,
          target,
          static_ids,
          tokenized_data=None,
          past_key_values=warmup_cache,
          output_hidden_states=True,
        )
        warmup_selected = _selected_hidden_from_output(warmup_output, layer_ids, len(layers))
        if warmup_selected is None:
          raise RuntimeError("missing_hidden_states")
      current_stream.wait_stream(warmup_stream)

      graph = torch_mod.cuda.CUDAGraph()
      with torch_mod.cuda.graph(graph):
        static_output, static_selected = run_static()
      entry = {
        "graph": graph,
        "ids": static_ids,
        "target_cache": static_target_cache,
        "output": static_output,
        "selected_hidden": static_selected,
      }
      graph_cache[cache_key] = entry
      graph_cache.move_to_end(cache_key)
      while len(graph_cache) > max_entries:
        graph_cache.popitem(last=False)
    else:
      graph_cache.move_to_end(cache_key)

    entry["ids"].copy_(ids, non_blocking=True)
    copy_error = _copy_cache_tensors(entry["target_cache"], target_cache)
    if copy_error is not None:
      graph_cache.pop(cache_key, None)
      fallback(copy_error)
      return None
    entry["graph"].replay()

    output = SimpleNamespace(
      logits=entry["output"].logits,
      past_key_values=entry["target_cache"],
      _openpilot_graph_static_target_cache=entry["target_cache"],
      _openpilot_graph_validation_entry=entry,
    )
    selected_hidden = entry["selected_hidden"]
    if runtime_profile is not None:
      runtime_profile["dflash_validate_graph_mode"] = "replay" if cache_hit else "capture"
      runtime_profile["dflash_validate_graph_ready"] = 1
      runtime_profile["dflash_validate_graph_cache_hit"] = 1 if cache_hit else 0
      runtime_profile["dflash_validate_graph_cache_depth"] = int(len(graph_cache))
      runtime_profile["dflash_validate_graph_calls"] = int(runtime_profile.get("dflash_validate_graph_calls", 0)) + 1
      runtime_profile["dflash_validate_graph_selected_hidden_owner"] = "graph_static_borrowed"
      runtime_profile["dflash_validate_graph_target_cache_owner"] = "graph_static_borrowed"
    return output, selected_hidden
  except Exception as exc:
    try:
      graph_cache.pop(cache_key, None)
    except Exception:
      pass
    fallback(f"{type(exc).__name__}: {exc}")
    return None


def _contains_token(torch_mod: Any, ids: Any, token_id: int) -> bool:
  if ids.numel() <= 0:
    return False
  return bool((ids == int(token_id)).any().detach().cpu().item())


def _crop_at_first_token(torch_mod: Any, ids: Any, start: int, token_id: int) -> Any:
  tail = ids[0, int(start) :]
  if tail.numel() <= 0:
    return ids
  matches = (tail == int(token_id)).nonzero(as_tuple=True)[0]
  if int(matches.numel()) <= 0:
    return ids
  stop_index = int(matches[0].detach().cpu().item())
  return ids[:, : int(start) + stop_index + 1]


def dflash_generate_alpamayo(
  torch_mod: Any,
  model: Any,
  draft_model: Any,
  mask_embedding: Any,
  tokenized_data: dict[str, Any],
  input_ids: Any,
  max_generation_length: int,
  temperature: float,
  runtime_profile: dict[str, Any] | None = None,
) -> SimpleNamespace:
  from transformers import DynamicCache
  from alpamayo1_5.models.token_utils import to_special_token

  target = model.vlm
  eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
  block_size = int(draft_model.block_size)
  draft_device = _module_device(draft_model)
  lm_head_device = _module_device(target.lm_head)
  embed_tokens = target.model.language_model.embed_tokens

  num_input_tokens = int(input_ids.shape[1])
  max_length = num_input_tokens + max_generation_length
  output_buffer_shape = (1, max_length + block_size)
  output_ids_buffer = getattr(draft_model, "_openpilot_dflash_output_ids_buffer", None)
  if (
    output_ids_buffer is None
    or int(output_ids_buffer.shape[0]) != 1
    or int(output_ids_buffer.shape[1]) < int(output_buffer_shape[1])
    or output_ids_buffer.device != input_ids.device
    or output_ids_buffer.dtype != torch_mod.long
  ):
    output_ids_buffer = torch_mod.empty(
      output_buffer_shape,
      dtype=torch_mod.long,
      device=input_ids.device,
    )
    setattr(draft_model, "_openpilot_dflash_output_ids_buffer", output_ids_buffer)
  output_ids = output_ids_buffer[:, : output_buffer_shape[1]]
  output_ids.fill_(int(draft_model.mask_token_id))
  output_ids[:, :num_input_tokens] = input_ids
  block_output_ids_buffer = getattr(draft_model, "_openpilot_dflash_block_output_ids_buffer", None)
  if (
    block_output_ids_buffer is None
    or tuple(block_output_ids_buffer.shape) != (1, block_size)
    or block_output_ids_buffer.device != input_ids.device
    or block_output_ids_buffer.dtype != torch_mod.long
  ):
    block_output_ids_buffer = torch_mod.full(
      (1, block_size),
      int(draft_model.mask_token_id),
      dtype=torch_mod.long,
      device=input_ids.device,
    )
    setattr(draft_model, "_openpilot_dflash_block_output_ids_buffer", block_output_ids_buffer)
  draft_position_seed = _position_seed_buffer(
    draft_model,
    "_openpilot_dflash_draft_position_seed",
    torch_mod,
    device=draft_device,
    length=max_length + block_size,
  )

  target_cache = DynamicCache()
  draft_cache = DynamicCache()
  layer_ids = list(draft_model.target_layer_ids)
  if not layer_ids:
    raise RuntimeError("DFlash draft model target_layer_ids is empty; cannot capture target hidden layers")

  prefill_start = time.perf_counter()
  prefix_cache_entry = getattr(model, "_openpilot_vlm_prefix_cache_entry", None)
  cached_layer_ids = tuple(int(item) for item in layer_ids)
  prefix_cache_warm_hit = bool(
    isinstance(prefix_cache_entry, dict)
    and int(prefix_cache_entry.get("hits", 0) or 0) > 0
  )
  full_generation_ready = bool(
    isinstance(prefix_cache_entry, dict)
    and prefix_cache_entry.get("dflash_full_generated_sequences") is not None
    and prefix_cache_entry.get("dflash_full_prompt_cache") is not None
  )
  full_generation_usable = bool(
    temperature == 0.0
    and full_generation_ready
    and int(prefix_cache_entry.get("dflash_full_max_generation_length", -1)) == int(max_generation_length)
    and int(prefix_cache_entry.get("dflash_full_eos_token_id", -1)) == int(eos_token_id)
    and int(prefix_cache_entry.get("dflash_full_input_seq_len", -1)) == int(input_ids.shape[1])
    and tuple(prefix_cache_entry.get("dflash_layer_ids", ())) == cached_layer_ids
  )
  if full_generation_usable and isinstance(prefix_cache_entry, dict):
    exact_window_full_hit = bool(
      prefix_cache_entry.get("current_window_full_hit")
      or prefix_cache_entry.get("exact_window_full_hit")
      or prefix_cache_entry.get("window_full_hit")
    )
    streaming_reuse_mode = str(prefix_cache_entry.get("streaming_vlm_reuse_mode", ""))
    streaming_reuse_unverified = bool(prefix_cache_entry.get("streaming_vlm_reuse_unverified")) or streaming_reuse_mode.endswith("_unverified")
    trusted_replay_requested = bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_requested"))
    current_window_signature = prefix_cache_entry.get("current_window_signature")
    stored_window_signature = prefix_cache_entry.get("dflash_full_window_signature")
    generation_window_signature_match = bool(
      current_window_signature is not None
      and stored_window_signature is not None
      and stored_window_signature == current_window_signature
    )
    prompt_cache_context_exact = bool(prefix_cache_entry.get("dflash_full_prompt_cache_context_exact"))
    if runtime_profile is not None:
      runtime_profile["dflash_full_generation_cache_window_signature_match"] = (
        1 if generation_window_signature_match else 0
      )
      runtime_profile["dflash_full_generation_prompt_cache_context_exact"] = (
        1 if prompt_cache_context_exact else 0
      )
    if (
      streaming_reuse_unverified
      or not exact_window_full_hit
      or not generation_window_signature_match
      or not prompt_cache_context_exact
    ):
      full_generation_usable = False
      if streaming_reuse_unverified:
        prefix_cache_entry["dflash_full_reason"] = "disabled_for_unverified_streaming_current_prompt_freshness"
      elif not exact_window_full_hit:
        prefix_cache_entry["dflash_full_reason"] = "disabled_without_exact_window_hit"
      elif not generation_window_signature_match:
        prefix_cache_entry["dflash_full_reason"] = "disabled_without_exact_generation_window_signature"
      else:
        prefix_cache_entry["dflash_full_reason"] = "disabled_without_exact_prompt_cache_context"
      if runtime_profile is not None:
        runtime_profile["dflash_full_generation_cache_disabled_for_streaming"] = 1 if streaming_reuse_unverified else 0
        runtime_profile["dflash_full_generation_cache_disabled_without_exact_window_hit"] = 0 if exact_window_full_hit else 1
        runtime_profile["dflash_full_generation_cache_disabled_without_exact_window_signature"] = (
          0 if generation_window_signature_match else 1
        )
        runtime_profile["dflash_full_generation_cache_disabled_without_exact_prompt_cache_context"] = (
          0 if prompt_cache_context_exact else 1
        )
        runtime_profile["dflash_full_generation_cache_trusted_replay_disabled_for_diffusion_freshness"] = (
          1 if trusted_replay_requested or bool(prefix_cache_entry.get("streaming_vlm_trusted_replay_allowed")) else 0
        )
  if (
    temperature == 0.0
    and isinstance(prefix_cache_entry, dict)
    and full_generation_usable
  ):
    try:
      cached_sequences = prefix_cache_entry["dflash_full_generated_sequences"]
      if cached_sequences.device != input_ids.device:
        cached_sequences = cached_sequences.to(input_ids.device)
      cached_prompt_cache = prefix_cache_entry.get("dflash_full_prompt_cache")
      if cached_prompt_cache is not None:
        cached_prompt_cache, cache_mode = _pooled_live_cache_copy(
          prefix_cache_entry,
          cached_prompt_cache,
          pool_key="dflash_full_prompt_cache_live_pool",
        )
      else:
        cache_mode = "missing_prompt_cache"
      cached_acceptance_lengths = list(prefix_cache_entry.get("dflash_full_acceptance_lengths", []))
      generated_new_tokens = int(cached_sequences.shape[1] - num_input_tokens)
      prefix_cache_entry["dflash_full_hits"] = int(prefix_cache_entry.get("dflash_full_hits", 0)) + 1
      if runtime_profile is not None:
        runtime_profile["dflash_full_generation_cache_hit"] = 1
        runtime_profile["dflash_prefix_cache_hit"] = 1
        runtime_profile["dflash_full_generation_live_cache_mode"] = cache_mode
        runtime_profile["dflash_full_generation_prompt_cache_owner"] = "full_generation_borrowed_live_cache"
      return SimpleNamespace(
        generated_sequences=cached_sequences,
        prompt_cache=cached_prompt_cache,
        acceptance_lengths=cached_acceptance_lengths,
        generated_new_tokens=generated_new_tokens,
        time_to_first_token_ms=0.0,
        decode_ms=0.0,
        draft_ms=0.0,
        validate_ms=0.0,
        new_tokens=generated_new_tokens,
      )
    except Exception as exc:
      prefix_cache_entry["dflash_full_reason"] = f"full_generation_cache_copy_failed:{type(exc).__name__}"
      if runtime_profile is not None:
        runtime_profile["dflash_full_generation_cache_copy_error"] = f"{type(exc).__name__}: {exc}"
  if (
    prefix_cache_warm_hit
    and not full_generation_usable
    and bool(getattr(model, "_openpilot_static_graph_strict_shapes", False))
  ):
    if runtime_profile is not None:
      runtime_profile["dflash_strict_warm_full_generation_cache_miss"] = 1
    raise RuntimeError("strict warm DFlash replay required but full-generation cache is not ready")
  cached_prefill_ready = False
  sync_timing = bool(isinstance(runtime_profile, dict) and runtime_profile.get("dflash_sync_timing", False))
  if isinstance(prefix_cache_entry, dict) and tuple(prefix_cache_entry.get("dflash_layer_ids", ())) == cached_layer_ids:
    cached_logits = prefix_cache_entry.get("dflash_prefill_logits")
    cached_hidden = prefix_cache_entry.get("dflash_target_hidden")
    cached_cache = prefix_cache_entry.get("dflash_target_cache")
    if cached_logits is not None and cached_hidden is not None and cached_cache is not None:
      try:
        target_cache, cache_mode = _pooled_live_cache_copy(
          prefix_cache_entry,
          cached_cache,
          pool_key="dflash_prefix_target_cache_live_pool",
        )
        output = SimpleNamespace(
          logits=cached_logits,
          past_key_values=target_cache,
        )
        target_hidden = cached_hidden
        cached_prefill_ready = True
        prefix_cache_entry["dflash_hits"] = int(prefix_cache_entry.get("dflash_hits", 0)) + 1
        if runtime_profile is not None:
          runtime_profile["dflash_prefix_cache_hit"] = 1
          runtime_profile["dflash_prefix_cache_selected_hidden_reused"] = 1
          runtime_profile["dflash_prefix_cache_live_cache_mode"] = cache_mode
      except Exception as exc:
        target_cache = DynamicCache()
        cached_prefill_ready = False
        prefix_cache_entry["dflash_reason"] = f"selected_hidden_cache_copy_failed:{type(exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["dflash_prefix_cache_copy_error"] = f"{type(exc).__name__}: {exc}"
  if not cached_prefill_ready:
    prefill_result = _dflash_selected_hidden_prefill_with_cuda_graph(
      torch_mod,
      model,
      target,
      input_ids=input_ids,
      tokenized_data=tokenized_data,
      target_cache=target_cache,
      layer_ids=layer_ids,
      runtime_profile=runtime_profile,
    )
    if prefill_result is None:
      if _dflash_strict_graph_required(model, "prefill"):
        if runtime_profile is not None:
          runtime_profile["dflash_prefill_strict_graph_failure"] = 1
        raise RuntimeError("DFlash selected-hidden prefill graph unavailable under strict static graph mode")
      output, target_hidden = _qwen3vl_forward_with_selected_hidden(
        torch_mod,
        target,
        input_ids,
        tokenized_data=tokenized_data,
        past_key_values=target_cache,
        layer_ids=layer_ids,
        logits_to_keep=1,
        runtime_profile=runtime_profile,
      )
    else:
      output, target_hidden, target_cache = prefill_result
    if sync_timing:
      _sync(torch_mod)
    if isinstance(prefix_cache_entry, dict):
      try:
        prefix_cache_entry["dflash_layer_ids"] = cached_layer_ids
        prefix_cache_entry["dflash_target_cache"] = copy.deepcopy(target_cache)
        prefix_cache_entry["dflash_prefill_logits"] = output.logits.detach().clone()
        prefix_cache_entry["dflash_target_hidden"] = target_hidden.detach().clone()
        prefix_cache_entry["dflash_has_selected_hidden"] = True
        prefix_cache_entry["dflash_reason"] = "selected_hidden_prefill_ready"
        prefix_cache_entry["dflash_stores"] = int(prefix_cache_entry.get("dflash_stores", 0)) + 1
        if runtime_profile is not None:
          runtime_profile["dflash_prefix_cache_store"] = 1
      except Exception as exc:
        prefix_cache_entry["dflash_has_selected_hidden"] = False
        prefix_cache_entry["dflash_reason"] = f"selected_hidden_cache_store_failed:{type(exc).__name__}"
        if runtime_profile is not None:
          runtime_profile["dflash_prefix_cache_store_error"] = f"{type(exc).__name__}: {exc}"
  elif runtime_profile is not None:
    runtime_profile["dflash_prefix_cache_store"] = 0
  first = _sample(torch_mod, _mask_traj_token_logits(model, output.logits), temperature)[:, -1:]
  output_ids[:, num_input_tokens : num_input_tokens + 1] = first.to(output_ids.device)
  target_hidden = target_hidden.to(draft_device)
  time_to_first_token_ms = (time.perf_counter() - prefill_start) * 1000.0

  start = num_input_tokens
  acceptance_lengths: list[int] = []
  draft_ms = 0.0
  validate_ms = 0.0
  decode_start = time.perf_counter()
  saw_eos = False
  acceptance_block_cpu = _cpu_long_buffer(
    draft_model,
    "_openpilot_dflash_acceptance_block_cpu",
    torch_mod,
    (1, block_size),
    input_ids.dtype,
  )
  acceptance_post_cpu = _cpu_long_buffer(
    draft_model,
    "_openpilot_dflash_acceptance_post_cpu",
    torch_mod,
    (1, block_size),
    input_ids.dtype,
  )

  def _acceptance_decision(block_ids: Any, sampled_posterior: Any) -> tuple[int, bool]:
    acceptance_block_cpu[0].copy_(block_ids[0], non_blocking=True)
    acceptance_post_cpu[0].copy_(sampled_posterior[0], non_blocking=True)
    block_tokens = acceptance_block_cpu[0]
    posterior_tokens = acceptance_post_cpu[0]
    accepted_len_local = 1
    compare_count = min(int(block_tokens.shape[0]) - 1, int(posterior_tokens.shape[0]) - 1)
    if compare_count > 0:
      matches = block_tokens[1 : compare_count + 1] == posterior_tokens[:compare_count]
      mismatches = (~matches).nonzero(as_tuple=True)[0]
      if int(mismatches.numel()) > 0:
        accepted_len_local = int(mismatches[0].item()) + 1
      else:
        accepted_len_local = compare_count + 1
    eos_matches = (block_tokens[:accepted_len_local] == int(eos_token_id)).nonzero(as_tuple=True)[0]
    if int(eos_matches.numel()) > 0:
      accepted_len_local = int(eos_matches[0].item()) + 1
      return accepted_len_local, True
    return accepted_len_local, False

  while start < max_length:
    block_output_ids = block_output_ids_buffer
    block_output_ids.fill_(int(draft_model.mask_token_id))
    block_output_ids[:, :1] = output_ids[:, start : start + 1]

    if block_size > 1:
      draft_start = time.perf_counter()
      noise_embedding = _embed_block_tokens(
        torch_mod,
        embed_tokens,
        block_output_ids,
        mask_token_id=int(draft_model.mask_token_id),
        mask_embedding=mask_embedding,
        output_device=draft_device,
      )
      position_ids = draft_position_seed[draft_cache.get_seq_length() : start + block_size].unsqueeze(0)
      draft_hidden = _dflash_draft_block_with_cuda_graph(
        torch_mod,
        model,
        draft_model,
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        draft_cache=draft_cache,
        runtime_profile=runtime_profile,
      )
      if draft_hidden is None:
        if _dflash_strict_graph_required(model, "decode"):
          if runtime_profile is not None:
            runtime_profile["dflash_draft_strict_graph_failure"] = 1
          raise RuntimeError("DFlash draft graph unavailable under strict static graph mode")
        draft_hidden = draft_model(
          target_hidden=target_hidden,
          noise_embedding=noise_embedding,
          position_ids=position_ids,
          past_key_values=draft_cache,
          use_cache=True,
          is_causal=False,
        )
      draft_hidden = draft_hidden[:, 1 - block_size :, :]
      draft_logits = _mask_traj_token_logits(model, target.lm_head(draft_hidden.to(lm_head_device)))
      if sync_timing:
        _sync(torch_mod)
      draft_ms += (time.perf_counter() - draft_start) * 1000.0
      draft_cache.crop(start)
      block_output_ids[:, 1:] = _sample(torch_mod, draft_logits, temperature).to(block_output_ids.device)

    validate_start = time.perf_counter()
    validation_result = _dflash_target_validation_with_cuda_graph(
      torch_mod,
      model,
      target,
      ids=block_output_ids,
      target_cache=target_cache,
      layer_ids=layer_ids,
      runtime_profile=runtime_profile,
    )
    if validation_result is None:
      if _dflash_strict_graph_required(model, "decode"):
        if runtime_profile is not None:
          runtime_profile["dflash_validate_strict_graph_failure"] = 1
        raise RuntimeError("DFlash target validation graph unavailable under strict static graph mode")
      output, target_hidden = _qwen3vl_forward_with_selected_hidden(
        torch_mod,
        target,
        block_output_ids,
        tokenized_data=None,
        past_key_values=target_cache,
        layer_ids=layer_ids,
        runtime_profile=runtime_profile,
      )
    else:
      output, target_hidden = validation_result
    target_cache = output.past_key_values
    if sync_timing:
      _sync(torch_mod)
    validate_ms += (time.perf_counter() - validate_start) * 1000.0

    posterior = _sample(
      torch_mod,
      _mask_traj_token_logits(model, output.logits),
      temperature,
    ).to(block_output_ids.device)
    accepted_len, saw_eos = _acceptance_decision(block_output_ids, posterior)
    output_ids[:, start : start + accepted_len] = block_output_ids[:, :accepted_len]
    if not saw_eos and start + accepted_len < output_ids.shape[1]:
      output_ids[:, start + accepted_len] = posterior[:, accepted_len - 1]
    start += accepted_len
    if accepted_len < block_size:
      graph_static_cache = getattr(output, "_openpilot_graph_static_target_cache", None)
      graph_entry = getattr(output, "_openpilot_graph_validation_entry", None)
      if graph_static_cache is not None and isinstance(graph_entry, dict):
        target_cache, live_cache_mode = _pooled_live_cache_copy(
          graph_entry,
          graph_static_cache,
          pool_key="live_validation_target_cache_pool",
        )
        if runtime_profile is not None:
          runtime_profile["dflash_validate_graph_partial_accept_live_cache_mode"] = live_cache_mode
      target_cache.crop(start)
    acceptance_lengths.append(accepted_len)

    if block_size > 1:
      target_hidden = target_hidden[:, :accepted_len, :].to(draft_device)

    if saw_eos:
      break

  generated_len = min(start if saw_eos else start + 1, max_length)
  generated_sequences = _crop_at_first_token(torch_mod, output_ids[:, :generated_len], num_input_tokens, eos_token_id).to(
    input_ids.device
  )
  generated_new_tokens = int(generated_sequences.shape[1] - num_input_tokens)
  if runtime_profile is not None:
    runtime_profile["dflash_generated_sequences_owner"] = "draft_output_slab_borrowed"
  decode_ms = (time.perf_counter() - decode_start) * 1000.0
  if temperature == 0.0 and isinstance(prefix_cache_entry, dict):
    try:
      prefix_cache_entry["dflash_full_generated_sequences"] = generated_sequences.detach().clone()
      prefix_cache_entry["dflash_full_prompt_cache"] = copy.deepcopy(target_cache)
      prefix_cache_entry["dflash_full_prompt_cache_owner"] = "prefix_cache_full_generation_stored_immutable_copy"
      prefix_cache_entry["dflash_full_window_signature"] = prefix_cache_entry.get(
        "current_window_signature",
        prefix_cache_entry.get("window_signature"),
      )
      prefix_cache_entry["dflash_full_prompt_cache_context_exact"] = True
      prefix_cache_entry["dflash_full_acceptance_lengths"] = list(acceptance_lengths)
      prefix_cache_entry["dflash_full_max_generation_length"] = int(max_generation_length)
      prefix_cache_entry["dflash_full_eos_token_id"] = int(eos_token_id)
      prefix_cache_entry["dflash_full_input_seq_len"] = int(input_ids.shape[1])
      prefix_cache_entry["dflash_full_reason"] = "full_generation_ready"
      prefix_cache_entry["dflash_full_stores"] = int(prefix_cache_entry.get("dflash_full_stores", 0)) + 1
      if runtime_profile is not None:
        runtime_profile["dflash_full_generation_cache_store"] = 1
    except Exception as exc:
      prefix_cache_entry["dflash_full_reason"] = f"full_generation_cache_store_failed:{type(exc).__name__}"
      if runtime_profile is not None:
        runtime_profile["dflash_full_generation_cache_store_error"] = f"{type(exc).__name__}: {exc}"

  return SimpleNamespace(
    generated_sequences=generated_sequences,
    prompt_cache=target_cache,
    acceptance_lengths=acceptance_lengths,
    generated_new_tokens=generated_new_tokens,
    time_to_first_token_ms=time_to_first_token_ms,
    decode_ms=decode_ms,
    draft_ms=draft_ms,
    validate_ms=validate_ms,
    new_tokens=generated_new_tokens,
  )
