#!/usr/bin/env bash
# Ubuntu/L40S (Ada Lovelace, compute_89) translation of the TensorRT build
# sequence in README_REASONED_TRAJECTORY_POC.md.
#
# The original sequence targeted an RTX 5060 Ti (Blackwell, compute_120) and
# used an NVFP4 text engine. Blackwell-only FP4 tensor cores don't exist on
# Ada, so this uses --text-precision fp8 instead (Ada's 4th-gen tensor cores
# support FP8 natively). Vision engine stays fp32, same as the original.
#
# Requires: qwen_trt_venv/ (see README notes), CUDA 12.8 toolkit on PATH,
# models/vlm/qwen2_5_vl_3b_instruct/ downloaded, and a representative
# 168x168-ish source image for tracing (defaults to a frame saved by an
# earlier MetaDrive run; pass one explicitly with --image if none exists yet).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source qwen_trt_venv/bin/activate
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH="$REPO_ROOT"

ARTIFACT_DIR="${QWEN_TRT_ARTIFACT_DIR:-$REPO_ROOT/qwen_trt_export}"
IMAGE="${1:-$REPO_ROOT/artifacts/reasoned_trajectory_poc/smoke_test_ubuntu7/vlm/vlm_input_0020.png}"
if [[ ! -f "$IMAGE" ]]; then
  echo "No trace image at $IMAGE -- pass one explicitly, e.g. a frame from a MetaDrive run's vlm/ dir." >&2
  exit 1
fi

COMMON_ARGS=(
  --artifact-dir "$ARTIFACT_DIR"
  --label-decision-mode choice
  --text-output hidden
  --text-seq-len 576
  --image "$IMAGE"
  --image-size 168
  --text-precision fp8
  --warmup 8
  --deadline-ms 50
)

echo "=== build-vision (fp32) ==="
python3 tools/reasoned_trajectory_poc/qwen_trt_label_engine.py \
  --artifact-dir "$ARTIFACT_DIR" --image "$IMAGE" --image-size 168 \
  --vision-precision fp32 --workspace-gb 8 build-vision

echo "=== build-text (fp8) ==="
python3 tools/reasoned_trajectory_poc/qwen_trt_label_engine.py \
  "${COMMON_ARGS[@]}" --score-labels red_stop_light,green_go_light \
  --workspace-gb 8 build-text

VISION_ENGINE="$ARTIFACT_DIR/vision_static_fp32/qwen_vision_full168_static_fp32.engine"
TEXT_ENGINE="$ARTIFACT_DIR/fp8_trt/qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine"

echo "=== check-artifacts (writes manifest) ==="
python3 tools/reasoned_trajectory_poc/qwen_trt_label_engine.py \
  "${COMMON_ARGS[@]}" --vision-engine "$VISION_ENGINE" --text-engine "$TEXT_ENGINE" \
  --score-labels red_stop_light,green_go_light --write-manifest check-artifacts

echo "=== benchmark-groups (actual p50/p95/p99/max latency) ==="
python3 tools/reasoned_trajectory_poc/qwen_trt_label_engine.py \
  "${COMMON_ARGS[@]}" --vision-engine "$VISION_ENGINE" --text-engine "$TEXT_ENGINE" \
  --score-label-groups "red_stop_light,green_go_light" --iters 30 benchmark-groups
