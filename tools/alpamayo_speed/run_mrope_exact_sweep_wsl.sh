#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/ture_opamayo/openpilot_alpamayo/openpilot

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
unset TLLM_DISABLE_MPI
export TLLM_DISABLE_FLASHINFER_NORM="${TLLM_DISABLE_FLASHINFER_NORM:-1}"
export TLLM_DISABLE_MM_PROFILE="${TLLM_DISABLE_MM_PROFILE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="/mnt/g/alpamayo1.5/trtllm_venv/bin:${CUDA_HOME}/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/mnt/g/alpamayo1.5/trtllm_venv/lib/python3.12/site-packages/nvidia/cu13/lib:/mnt/g/alpamayo1.5/trtllm_venv/lib/python3.12/site-packages/nvidia/cublas/lib:/mnt/g/alpamayo1.5/trtllm_venv/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-/mnt/g/alpamayo1.5/nvfp4_exports/alpamayo_vlm_nvfp4_lmtext_2026-05-26}"
ENGINE="${ENGINE:-/mnt/g/alpamayo1.5/trtllm_engines/alpamayo_vlm_nvfp4_lmtext_qwen3vl_classic_2026-05-26}"
WORKSPACE="${WORKSPACE:-/mnt/g/alpamayo1.5/trtllm_workspace}"
PROMPT="${PROMPT:-artifacts/alpamayo_speed/verbose_prompt_probe_2026-05-26.txt}"
PROMPT_SEQUENCE_JSONL="${PROMPT_SEQUENCE_JSONL:-}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-4}"
DATE_TAG="${DATE_TAG:-2026-05-26}"
MROPE_DEVICE="${MROPE_DEVICE:-cpu}"
REPORT_SUFFIX="${REPORT_SUFFIX:-usemrope_true}"
read -r -a RUNNER_EXTRA_ARGS <<< "${RUNNER_EXTRA_ARGS:-}"
if [[ "${KV_CACHE_ENABLE_BLOCK_REUSE:-0}" == "1" ]]; then
  RUNNER_EXTRA_ARGS+=(--kv-cache-enable-block-reuse)
fi
if [[ "${CUDA_GRAPH_MODE:-0}" == "1" ]]; then
  RUNNER_EXTRA_ARGS+=(--cuda-graph-mode)
fi
PROMPT_ARGS=(--prompt-file "${PROMPT}")
if [[ -n "${PROMPT_SEQUENCE_JSONL}" ]]; then
  PROMPT_ARGS=(--prompt-sequence-jsonl "${PROMPT_SEQUENCE_JSONL}")
fi

for output_tokens in "$@"; do
  report="artifacts/alpamayo_speed/trtllm_mrope_runner_nvfp4_lmtext_verbose_out${output_tokens}_${REPORT_SUFFIX}_${DATE_TAG}.json"
  echo "=== exact mrope verbose output_tokens=${output_tokens} ==="
  python3 tools/alpamayo_speed/alpamayo_trtllm_mrope_runner_smoke.py \
    --model-dir "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --engine-dir "${ENGINE}" \
    --report "${report}" \
    "${PROMPT_ARGS[@]}" \
    --skip-build \
    --max-output-tokens "${output_tokens}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --mrope-device "${MROPE_DEVICE}" \
    "${RUNNER_EXTRA_ARGS[@]}"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
done
