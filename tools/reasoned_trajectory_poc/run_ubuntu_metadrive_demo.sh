#!/usr/bin/env bash
# Ubuntu translation of the MetaDrive PowerShell demo/render commands in README.md.
# Requires run_ubuntu_alpamayo_endpoint.sh already running in another shell
# (or in the background) and healthy at http://127.0.0.1:8765/alpamayo.
#
# Usage: run_ubuntu_metadrive_demo.sh <out-name> [extra run_metadrive_overlay_demo.py args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUT_NAME="${1:?usage: run_ubuntu_metadrive_demo.sh <out-name> [extra args...]}"
shift || true
OUT="artifacts/reasoned_trajectory_poc/${OUT_NAME}"

python tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py \
  --engine alpamayofast \
  --novel-scene random_mixed \
  --frames 820 \
  --speed-mps 2.5 \
  --tick-sec 0.05 \
  --deadline-ms 100 \
  --save-every 1 \
  --map 3 \
  --seed 7 \
  --random-scene-seed 42 \
  --camera-width 512 \
  --camera-height 512 \
  --board-width 640 \
  --board-height 400 \
  --alpamayo-endpoint-url http://127.0.0.1:8765/alpamayo \
  --alpamayo-endpoint-timeout-s 300 \
  --alpamayo-num-frames 4 \
  --alpamayo-query-every 2 \
  --alpamayo-catchup-stride-steps 1 \
  --alpamayo-control-mode planner_bridge \
  --alpamayo-lateral-preview-m 12 \
  --alpamayo-max-lateral-offset-m 0.8 \
  --alpamayo-steer-sign -1 \
  --alpamayo-longitudinal-preview-s 1.0 \
  --alpamayo-max-accel-mps2 1.5 \
  --alpamayo-max-decel-mps2 3.0 \
  --alpamayo-reasoning-overlay \
  --alpamayo-reasoning-overlay-chars 260 \
  --alpamayo-reasoning-overlay-line-chars 74 \
  --out "$OUT" \
  "$@"

python tools/reasoned_trajectory_poc/render_demo_videos.py \
  --run-dir "$OUT" \
  --prefix "$OUT_NAME" \
  --fps 20

echo "Video: $OUT/videos/side_by_side_${OUT_NAME}.mp4"
