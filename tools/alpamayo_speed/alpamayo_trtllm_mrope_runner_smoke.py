#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import time
import traceback
from pathlib import Path


def _read_text_config(model_dir: Path) -> dict:
  with (model_dir / "config.json").open("r", encoding="utf-8") as f:
    config = json.load(f)
  return config.get("text_config", config)


def _text_hidden_and_vocab(model_dir: Path) -> tuple[int, int]:
  config = _read_text_config(model_dir)
  hidden_size = int(config["hidden_size"])
  vocab_size = int(config["vocab_size"])
  return hidden_size, vocab_size


def _tokenizer_ids(model_dir: Path, prompt: str, end_token: str | None, end_token_id: int | None):
  import torch
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)
  ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0].to(torch.int32)
  pad_id = tokenizer.pad_token_id
  if pad_id is None:
    pad_id = tokenizer.eos_token_id
  if pad_id is None:
    pad_id = 0
  end_id = tokenizer.eos_token_id
  if end_id is None:
    end_id = pad_id
  if end_token is not None:
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if end_id is None or end_id < 0:
      raise ValueError(f"could not resolve --end-token {end_token!r}")
  if end_token_id is not None:
    end_id = end_token_id
  return tokenizer, ids, int(end_id), int(pad_id)


def _load_prompt_records(args) -> list[dict[str, str]]:
  if args.prompt_sequence_jsonl is None:
    return [{"name": "prompt", "prompt": args.prompt}]

  records = []
  with args.prompt_sequence_jsonl.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      if isinstance(item, str):
        records.append({"name": f"line_{line_no}", "prompt": item})
      else:
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
          raise ValueError(f"{args.prompt_sequence_jsonl}:{line_no} missing nonempty prompt")
        name = item.get("name") or f"line_{line_no}"
        records.append({"name": str(name), "prompt": prompt})
  if not records:
    raise ValueError(f"no prompts found in {args.prompt_sequence_jsonl}")
  return records


def _common_prefix_len(sequences) -> int:
  if not sequences:
    return 0
  min_len = min(len(seq) for seq in sequences)
  common = 0
  for i in range(min_len):
    first = sequences[0][i]
    if any(seq[i] != first for seq in sequences[1:]):
      break
    common += 1
  return common


def _make_text_mrope_params(model_dir: Path, batch_size: int, device: str):
  import torch
  from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType
  from tensorrt_llm.layers import MropeParams

  text_config = _read_text_config(model_dir)
  max_pos = int(text_config["max_position_embeddings"])
  hidden_size = int(text_config["hidden_size"])
  num_heads = int(text_config["num_attention_heads"])
  head_dim = int(text_config.get("head_dim") or (hidden_size // num_heads))
  theta = float(text_config.get("rope_theta", 10000.0))

  _, rotary = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
    num_pos=max_pos,
    dim=head_dim,
    theta=theta,
    scale_type=RotaryScalingType.mrope,
  )
  rotary_tensor = torch.from_numpy(rotary).to(device=device, dtype=torch.float32)
  if batch_size != 1:
    rotary_tensor = rotary_tensor.expand(batch_size, -1).contiguous()
  deltas = torch.zeros((batch_size, 1), device=device, dtype=torch.int32)
  return MropeParams(mrope_rotary_cos_sin=rotary_tensor, mrope_position_deltas=deltas)


def _copy_engine_from_llm(llm, engine_dir: Path) -> str:
  tmp_engine_dir = getattr(llm, "_engine_dir", None)
  if tmp_engine_dir is None:
    raise RuntimeError("TensorRT LLM object did not expose _engine_dir")
  tmp_engine_dir = Path(str(tmp_engine_dir))
  if not tmp_engine_dir.exists():
    raise RuntimeError(f"TensorRT temporary engine dir does not exist: {tmp_engine_dir}")
  if engine_dir.exists():
    shutil.rmtree(engine_dir)
  engine_dir.parent.mkdir(parents=True, exist_ok=True)
  shutil.copytree(tmp_engine_dir, engine_dir)
  return str(tmp_engine_dir)


def _shutdown_llm(llm, report: dict) -> None:
  try:
    llm.shutdown()
  except Exception as e:
    report["llm_shutdown_error"] = repr(e)
  try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  except Exception as e:
    report["cuda_cleanup_error"] = repr(e)


def _build_engine(args, report: dict) -> None:
  from tensorrt_llm._tensorrt_engine import LLM

  Path(args.workspace).mkdir(parents=True, exist_ok=True)
  kwargs = {
    "model": str(args.model_dir),
    "trust_remote_code": True,
    "tensor_parallel_size": args.tensor_parallel_size,
    "dtype": "auto",
    "max_batch_size": args.max_batch_size,
    "max_input_len": args.max_input_len,
    "max_seq_len": args.max_seq_len,
    "max_num_tokens": args.max_num_tokens,
    "workspace": args.workspace,
  }
  if args.enable_prompt_adapter:
    kwargs["enable_prompt_adapter"] = True
    kwargs["max_prompt_adapter_token"] = args.max_prompt_adapter_token
    report["enable_prompt_adapter"] = True
    report["max_prompt_adapter_token"] = args.max_prompt_adapter_token

  llm = None
  t0 = time.perf_counter()
  try:
    llm = LLM(**kwargs)
    report["engine_build_load_seconds"] = round(time.perf_counter() - t0, 3)
    report["temporary_engine_dir"] = _copy_engine_from_llm(llm, args.engine_dir)
    report["persistent_engine_dir"] = str(args.engine_dir)
  finally:
    if llm is not None:
      _shutdown_llm(llm, report)


def _decode_output(tokenizer, output) -> tuple[str, dict]:
  import torch

  meta: dict = {"raw_type": type(output).__name__}
  if isinstance(output, dict):
    meta["keys"] = sorted(str(k) for k in output.keys())
    ids = output.get("output_ids")
    seq_lens = output.get("sequence_lengths")
    if seq_lens is not None:
      try:
        meta["sequence_lengths"] = torch.as_tensor(seq_lens).detach().cpu().tolist()
      except Exception:
        meta["sequence_lengths_repr"] = repr(seq_lens)
  else:
    ids = output
  if ids is None:
    return repr(output), meta

  tensor = torch.as_tensor(ids).detach().cpu()
  meta["output_ids_shape"] = list(tensor.shape)
  if tensor.ndim == 3:
    seq = tensor[0, 0].tolist()
  elif tensor.ndim == 2:
    seq = tensor[0].tolist()
  else:
    seq = tensor.reshape(-1).tolist()
  seq_len = None
  sequence_lengths = meta.get("sequence_lengths")
  if sequence_lengths is not None:
    try:
      seq_len = int(sequence_lengths[0][0] if isinstance(sequence_lengths[0], list) else sequence_lengths[0])
    except Exception:
      seq_len = None
  if seq_len is not None and seq_len > 0:
    seq = seq[:seq_len]
  meta["token_ids"] = [int(token_id) for token_id in seq]
  return tokenizer.decode(seq, skip_special_tokens=False), meta


def main() -> int:
  parser = argparse.ArgumentParser(description="Run TensorRT-LLM ModelRunnerCpp with explicit Qwen3-VL MRoPE params.")
  parser.add_argument("--model-dir", required=True, type=Path)
  parser.add_argument("--workspace", required=True)
  parser.add_argument("--engine-dir", required=True, type=Path)
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--prompt", default="Name the most relevant driving object in one word.")
  parser.add_argument("--prompt-file", type=Path)
  parser.add_argument("--prompt-sequence-jsonl", type=Path)
  parser.add_argument("--input-ids-npy", type=Path)
  parser.add_argument("--prompt-table-npy", type=Path)
  parser.add_argument("--tensor-parallel-size", type=int, default=1)
  parser.add_argument("--max-batch-size", type=int, default=1)
  parser.add_argument("--max-input-len", type=int, default=1024)
  parser.add_argument("--max-seq-len", type=int, default=1056)
  parser.add_argument("--max-num-tokens", type=int, default=1152)
  parser.add_argument("--max-output-tokens", type=int, default=8)
  parser.add_argument("--end-token", help="Override TensorRT generation end token, for example '<|traj_future_start|>'.")
  parser.add_argument("--end-token-id", type=int, help="Override TensorRT generation end token by numeric id.")
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--force-rebuild", action="store_true")
  parser.add_argument("--skip-build", action="store_true")
  parser.add_argument("--cuda-graph-mode", action="store_true")
  parser.add_argument("--kv-cache-enable-block-reuse", action="store_true")
  parser.add_argument("--mrope-device", choices=("cpu", "cuda"), default="cpu")
  parser.add_argument("--mrope-position-deltas-npy", type=Path)
  parser.add_argument("--no-mrope", action="store_true")
  parser.add_argument("--enable-prompt-adapter", action="store_true")
  parser.add_argument("--max-prompt-adapter-token", type=int, default=0)
  parser.add_argument("--dummy-prompt-table-tokens", type=int, default=0)
  parser.add_argument("--dummy-prompt-table-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
  parser.add_argument("--prompt-table-device", choices=("cpu", "cuda"), default="cpu")
  parser.add_argument("--auto-input-token-extra-ids", action="store_true")
  args = parser.parse_args()
  if args.enable_prompt_adapter and args.max_prompt_adapter_token <= 0:
    parser.error("--enable-prompt-adapter requires --max-prompt-adapter-token > 0")
  if args.dummy_prompt_table_tokens and not args.enable_prompt_adapter:
    parser.error("--dummy-prompt-table-tokens requires --enable-prompt-adapter")
  if args.dummy_prompt_table_tokens > args.max_prompt_adapter_token:
    parser.error("--dummy-prompt-table-tokens cannot exceed --max-prompt-adapter-token")
  if args.input_ids_npy is not None and args.prompt_sequence_jsonl is not None:
    parser.error("--input-ids-npy cannot be combined with --prompt-sequence-jsonl")
  if args.prompt_table_npy is not None and args.dummy_prompt_table_tokens:
    parser.error("--prompt-table-npy cannot be combined with --dummy-prompt-table-tokens")
  if args.prompt_table_npy is not None and not args.enable_prompt_adapter:
    parser.error("--prompt-table-npy requires --enable-prompt-adapter")
  if args.prompt_file is not None:
    args.prompt = args.prompt_file.read_text(encoding="utf-8")
  prompt_records = _load_prompt_records(args)

  report = {
    "status": "started",
    "created_at_unix": time.time(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "env": {
      "TLLM_DISABLE_FLASHINFER_NORM": os.environ.get("TLLM_DISABLE_FLASHINFER_NORM"),
      "TLLM_DISABLE_MM_PROFILE": os.environ.get("TLLM_DISABLE_MM_PROFILE"),
      "TLLM_DISABLE_MPI": os.environ.get("TLLM_DISABLE_MPI"),
      "CUDA_HOME": os.environ.get("CUDA_HOME"),
    },
    "model_dir": str(args.model_dir),
    "workspace": args.workspace,
    "engine_dir": str(args.engine_dir),
    "prompt": args.prompt,
    "prompt_sequence_jsonl": str(args.prompt_sequence_jsonl) if args.prompt_sequence_jsonl is not None else None,
    "input_ids_npy": str(args.input_ids_npy) if args.input_ids_npy is not None else None,
    "prompt_table_npy": str(args.prompt_table_npy) if args.prompt_table_npy is not None else None,
    "mrope_position_deltas_npy": str(args.mrope_position_deltas_npy) if args.mrope_position_deltas_npy is not None else None,
    "prompt_count": len(prompt_records),
    "enable_prompt_adapter": args.enable_prompt_adapter,
    "max_prompt_adapter_token": args.max_prompt_adapter_token,
    "dummy_prompt_table_tokens": args.dummy_prompt_table_tokens,
    "prompt_table_device": args.prompt_table_device,
    "auto_input_token_extra_ids": args.auto_input_token_extra_ids,
    "latency_ms": [],
    "warmup_latency_ms": [],
    "outputs": [],
  }

  try:
    import torch
    from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    if not args.skip_build and (args.force_rebuild or not args.engine_dir.exists()):
      _build_engine(args, report)
    elif not args.engine_dir.exists():
      raise RuntimeError(f"engine dir does not exist and --skip-build was set: {args.engine_dir}")

    tokenizer, first_input_ids, end_id, pad_id = _tokenizer_ids(args.model_dir, prompt_records[0]["prompt"], args.end_token, args.end_token_id)
    encoded_prompts = []
    if args.input_ids_npy is not None:
      import numpy as np

      input_ids_np = np.load(args.input_ids_npy)
      if input_ids_np.ndim != 1:
        raise ValueError(f"--input-ids-npy must contain a 1D array, got shape {input_ids_np.shape}")
      encoded_prompts = [torch.from_numpy(input_ids_np.astype(np.int32, copy=False)).to(torch.int32)]
      prompt_records = [{"name": args.input_ids_npy.stem, "prompt": "<input_ids_npy>"}]
      first_input_ids = encoded_prompts[0]
    else:
      for record in prompt_records:
        ids = tokenizer.encode(record["prompt"], add_special_tokens=True, return_tensors="pt")[0].to(torch.int32)
        encoded_prompts.append(ids)
    report["input_tokens"] = int(first_input_ids.numel())
    report["prompt_input_tokens"] = [int(ids.numel()) for ids in encoded_prompts]
    report["common_prefix_tokens"] = _common_prefix_len([ids.detach().cpu().tolist() for ids in encoded_prompts])
    report["prompt_records"] = [
      {"name": record["name"], "input_tokens": int(ids.numel())}
      for record, ids in zip(prompt_records, encoded_prompts, strict=True)
    ]
    report["end_id"] = end_id
    report["end_token"] = args.end_token
    report["end_token_id_arg"] = args.end_token_id
    report["pad_id"] = pad_id

    prompt_table = None
    prompt_tasks = None
    if args.prompt_table_npy is not None:
      import numpy as np

      prompt_table_np = np.load(args.prompt_table_npy)
      if prompt_table_np.ndim == 2:
        prompt_table_np = prompt_table_np[None, :, :]
      if prompt_table_np.ndim != 3:
        raise ValueError(f"--prompt-table-npy must contain a 2D or 3D array, got shape {prompt_table_np.shape}")
      prompt_table = torch.from_numpy(prompt_table_np)
      if args.prompt_table_device == "cuda":
        prompt_table = prompt_table.cuda()
      prompt_tasks = "0"
      report["prompt_table_shape"] = list(prompt_table_np.shape)
      report["prompt_table_dtype"] = str(prompt_table_np.dtype)
    elif args.dummy_prompt_table_tokens:
      hidden_size, vocab_size = _text_hidden_and_vocab(args.model_dir)
      dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
      }[args.dummy_prompt_table_dtype]
      prompt_table = torch.zeros((1, args.dummy_prompt_table_tokens, hidden_size), dtype=dtype)
      prompt_tasks = "0"
      fake_prompt_ids = (
        torch.arange(
          vocab_size,
          vocab_size + args.dummy_prompt_table_tokens,
          dtype=torch.int32,
        )
      )
      encoded_prompts = [
        torch.cat((fake_prompt_ids, ids.to(torch.int32)), dim=0)
        for ids in encoded_prompts
      ]
      report["dummy_prompt_table_hidden_size"] = hidden_size
      report["dummy_prompt_table_vocab_size"] = vocab_size
      report["dummy_prompt_table_dtype"] = args.dummy_prompt_table_dtype
      report["prompt_input_tokens_with_dummy_prefix"] = [int(ids.numel()) for ids in encoded_prompts]

    input_token_extra_ids = None
    if args.auto_input_token_extra_ids:
      _, vocab_size = _text_hidden_and_vocab(args.model_dir)
      input_token_extra_ids = []
      nonzero_counts = []
      for ids in encoded_prompts:
        extra = torch.zeros_like(ids, dtype=torch.int32)
        fake_mask = ids >= vocab_size
        extra[fake_mask] = (ids[fake_mask] - vocab_size + 1).to(torch.int32)
        input_token_extra_ids.append(extra.detach().cpu().tolist())
        nonzero_counts.append(int(fake_mask.sum().detach().cpu().item()))
      report["input_token_extra_ids_nonzero_counts"] = nonzero_counts

    mrope_params = None
    if not args.no_mrope:
      t_mrope = time.perf_counter()
      mrope_params = _make_text_mrope_params(args.model_dir, batch_size=1, device=args.mrope_device)
      if args.mrope_position_deltas_npy is not None:
        import numpy as np

        deltas_np = np.load(args.mrope_position_deltas_npy)
        if deltas_np.shape != (1, 1):
          raise ValueError(f"--mrope-position-deltas-npy must have shape (1, 1), got {deltas_np.shape}")
        mrope_params.mrope_position_deltas = torch.from_numpy(
          deltas_np.astype(np.int32, copy=False)
        ).to(device=args.mrope_device, dtype=torch.int32)
        report["mrope_position_deltas_loaded"] = deltas_np.tolist()
      report["mrope_create_seconds"] = round(time.perf_counter() - t_mrope, 3)
      report["mrope_rotary_shape"] = list(mrope_params.mrope_rotary_cos_sin.shape)
      report["mrope_rotary_device"] = str(mrope_params.mrope_rotary_cos_sin.device)
      report["mrope_position_deltas_shape"] = list(mrope_params.mrope_position_deltas.shape)
      report["mrope_position_deltas_device"] = str(mrope_params.mrope_position_deltas.device)

    t_runner = time.perf_counter()
    runner = ModelRunnerCpp.from_dir(
      str(args.engine_dir),
      max_batch_size=args.max_batch_size,
      max_input_len=args.max_input_len,
      max_output_len=args.max_output_tokens,
      kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
      cuda_graph_mode=args.cuda_graph_mode,
      device_ids=[0],
    )
    report["runner_load_seconds"] = round(time.perf_counter() - t_runner, 3)

    for i in range(args.warmup + args.iters):
      phase = "warmup" if i < args.warmup else "measure"
      for prompt_index, (prompt_record, input_ids) in enumerate(zip(prompt_records, encoded_prompts, strict=True)):
        batch = [input_ids.to(device="cuda", dtype=torch.int32)]
        t0 = time.perf_counter()
        output = runner.generate(
          batch,
          max_new_tokens=args.max_output_tokens,
          end_id=end_id,
          pad_id=pad_id,
          mrope_params=mrope_params,
          prompt_table=prompt_table,
          prompt_tasks=prompt_tasks,
          input_token_extra_ids=input_token_extra_ids,
          return_dict=True,
          output_sequence_lengths=True,
          num_beams=1,
          top_k=1,
          top_p=0.0,
          temperature=1.0,
        )
        latency = round((time.perf_counter() - t0) * 1000.0, 3)
        text, meta = _decode_output(tokenizer, output)
        input_token_count = int(input_ids.numel())
        record = {
          "latency_ms": latency,
          "pass_index": i,
          "phase": phase,
          "prompt_index": prompt_index,
          "prompt_name": prompt_record["name"],
          "text": text,
          "generated_token_ids": meta.get("token_ids", [])[input_token_count:],
          "generated_text": tokenizer.decode(
            meta.get("token_ids", [])[input_token_count:],
            skip_special_tokens=False,
          ) if meta.get("token_ids") else "",
          "meta": meta,
        }
        report["outputs"].append(record)
        if i < args.warmup:
          report["warmup_latency_ms"].append(latency)
        else:
          report["latency_ms"].append(latency)

    if report["latency_ms"]:
      report["mean_latency_ms"] = round(statistics.mean(report["latency_ms"]), 3)
      report["min_latency_ms"] = round(min(report["latency_ms"]), 3)
      report["max_latency_ms"] = round(max(report["latency_ms"]), 3)
      per_prompt = {}
      for output in report["outputs"]:
        if output.get("phase") != "measure":
          continue
        per_prompt.setdefault(output["prompt_name"], []).append(output["latency_ms"])
      report["per_prompt_latency_ms"] = {
        name: {
          "mean": round(statistics.mean(values), 3),
          "min": round(min(values), 3),
          "max": round(max(values), 3),
          "count": len(values),
        }
        for name, values in per_prompt.items()
      }
    report["status"] = "ok"
  except Exception as e:
    report["status"] = "error"
    report["error_type"] = type(e).__name__
    report["error"] = str(e)
    report["traceback"] = traceback.format_exc()
  finally:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))

  return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
