#!/usr/bin/env bash
# Ubuntu translation of the WSL/Windows pc_endpoint launch command in README.md.
# Differences from the original Windows/WSL command:
#   - single L40S GPU on this box, so ALPAMAYO_DEVICE_MAP_MODE=single_device
#     instead of the original 2-GPU current_split (split_index=16).
#   - uses the base gated nvidia/Alpamayo-1.5-10B model (resolved from the
#     local HF cache) instead of the custom Alpamayo-1.5-10B-finetuned
#     checkpoint, which only ever existed on the original Windows machine.
#   - ALPAMAYO_DFLASH_ENABLED=0: no DFlash draft model on this box, so
#     speculative decoding is off. Endpoint responses will be slower than the
#     36-42ms warm numbers in README.md, which depended on DFlash + the
#     fine-tuned checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALPAMAYO_ROOT="$REPO_ROOT/alpamayo1.5"

TARGET_MODEL_SNAPSHOT=$(find "$HOME/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [[ -z "$TARGET_MODEL_SNAPSHOT" ]]; then
  echo "Could not find a downloaded nvidia/Alpamayo-1.5-10B snapshot under \$HOME/.cache/huggingface/hub." >&2
  echo "Run: hf download nvidia/Alpamayo-1.5-10B" >&2
  exit 1
fi

source "$ALPAMAYO_ROOT/a1_5_venv/bin/activate"

export PYTHONPATH="$REPO_ROOT:$ALPAMAYO_ROOT:$ALPAMAYO_ROOT/src"
export ALPAMAYO_ROOT
export ALPAMAYO_TARGET_MODEL="$TARGET_MODEL_SNAPSHOT"
export ALPAMAYO_DFLASH_ENABLED=0
export ALPAMAYO_REQUIRE_FLASHVLA_TARGET=0
export ALPAMAYO_DEVICE_MAP_MODE=single_device
export ALPAMAYO_GPU_MEM_GIB=40
export ALPAMAYO_CPU_MEM_GIB=96
export ALPAMAYO_MIN_PIXELS=65536
export ALPAMAYO_MAX_PIXELS=65536
export ALPAMAYO_CUDA_GRAPHS=0
export ALPAMAYO_STATIC_GRAPH_STRICT_SHAPES=0
export ALPAMAYO_ATTN_IMPLEMENTATION=flash_attention_2
export ALPAMAYO_EXPERT_ATTN_IMPLEMENTATION=eager
export ALPAMAYO_PC_TRACE_PATH="$REPO_ROOT/artifacts/alpamayo_speed/pc_endpoint_ubuntu.trace.jsonl"
mkdir -p "$(dirname "$ALPAMAYO_PC_TRACE_PATH")"

echo "target model: $ALPAMAYO_TARGET_MODEL"
echo "device map mode: $ALPAMAYO_DEVICE_MAP_MODE"

cd "$REPO_ROOT"
exec python -u -m openpilot.selfdrive.alpamayo.pc_endpoint --host 0.0.0.0 --port 8765
