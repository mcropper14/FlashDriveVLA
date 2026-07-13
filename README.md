# DRAFT: speed-first Alpamayo warm-frame runtime

## Modifed from: 
Modified from: ```https://github.com/jagoff2/OPAMAYO/tree/alpamayo-speedfirst-metadrive-20260601-openpilot-controller ``` and Claude Code

Working on Ubuntu with a NVIDIA L40 (44GBVRAM)

This is a draft operator note for the speed-first Alpamayo / FlashDriveVLA runtime, the local PC endpoint, and the MetaDrive side-by-side controller proof path. It records the current runnable path, the launch commands used for the current proof artifact, measured warm-frame timing, and which older artifacts are superseded.

Status: draft, speed-first, sim-only. The relevant metric for this effort is steady-state warm shifted rows after the cache path is resident. Cold first-frame latency is intentionally ignored.

## Current answer upfront

Current valid proof artifact:

```text
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601
```

Current valid video:

```text
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601/videos/side_by_side_openpilot_controller_norm_randommixed_820_65kpix.mp4
```

Current valid control source:

```text
alpamayo_openpilot_controller
```

Current pushed OPAMAYO branch:

```text
https://github.com/jagoff2/OPAMAYO/tree/alpamayo-speedfirst-metadrive-20260601-openpilot-controller
```

Current commit on that branch:

```text
f74e7dc66 Update Alpamayo openpilot-controller fast path
```

The older `alpamayo-speedfirst-metadrive-20260531-101751` branch exists, but the latest pushed branch is `alpamayo-speedfirst-metadrive-20260601-openpilot-controller` because the old remote branch rejected a non-fast-forward push.

## Current measured 820-frame proof result

Run:

```text
metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601
```

Summary:

```text
Stock frames: 820
Alpamayo frames: 820
Warmup stock frames inside Alpamayo episode: 32
Alpamayo active control frames: 788/788
Alpamayo control source: alpamayo_openpilot_controller
Endpoint valid/calls/errors/deadline: 197/197/0/0
Endpoint p95: 36.48899996187538 ms
Endpoint p99: 40.02240003319457 ms
Endpoint max: 42.12589998496696 ms
Alpamayo terminated/truncated: false/false
Failure audit: 0 crash/collision/impact/offroad/barrier/human/pedestrian records
```

Route-progress audit:

```text
Stock route start/final/delta: 5.005555629730225 / 205.52471024383075 / 200.51915461410053 m
Alpamayo route start/final/delta: 5.005555629730225 / 205.63082634645005 / 200.62527071671983 m
Alpamayo minus stock route delta: +0.10611610261929627 m
```

Required openpilot-controller debug keys were present on all 788 Alpamayo control frames:

```text
alpamayo_openpilot_controller
alpamayo_control_mode_openpilot_controller
alpamayo_route_progress_speed_scale
alpamayo_actuator_openpilot_desired_curvature
alpamayo_actuator_openpilot_requested_curvature
```

Interpretation:

```text
This is the current controller proof artifact. Alpamayo completed the same random_mixed course distance as stock within 0.107 m, with no logged ped impact, offroad, barrier impact, crash, or collision records, while the Alpamayo-controlled portion used the openpilot-controller path.
```

## Exact endpoint launch command for the 820 proof run

Run from PowerShell. The endpoint runs inside WSL and serves `http://127.0.0.1:8765/alpamayo`.

```powershell
wsl.exe -e bash -lc 'source /mnt/g/alpamayo1.5/a1_5_venv/bin/activate && cd /mnt/e/ture_opamayo/openpilot_alpamayo && export PYTHONPATH=/mnt/e/ture_opamayo/openpilot_alpamayo:/mnt/g/alpamayo1.5:/mnt/g/alpamayo1.5/src && export ALPAMAYO_ROOT=/mnt/g/alpamayo1.5 && export ALPAMAYO_TARGET_MODEL=/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-10B-finetuned && export ALPAMAYO_DFLASH_ENABLED=1 && export ALPAMAYO_DFLASH_DRAFT_MODEL=/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-DFlash && export ALPAMAYO_DFLASH_PACKAGE_ROOT=/mnt/e/ture_opamayo/openpilot_alpamayo/dflash && export ALPAMAYO_STREAMING_VLM_TRUST_SHIFTED_DRAFT=1 && export ALPAMAYO_STREAMING_VLM_TRUSTED_REPLAY_REFRESH_INTERVAL=24 && export ALPAMAYO_STREAMING_VLM_SOURCE_CACHE_DRAFT_VERIFY_UNVERIFIED=0 && export ALPAMAYO_MIN_PIXELS=65536 && export ALPAMAYO_MAX_PIXELS=65536 && export ALPAMAYO_CUDA_GRAPHS=0 && export ALPAMAYO_STATIC_GRAPH_STRICT_SHAPES=0 && export ALPAMAYO_GRAPH_VISUAL_STAGE=0 && export ALPAMAYO_GRAPH_PREFILL_STAGE=0 && export ALPAMAYO_GRAPH_STANDARD_PREFILL_STAGE=0 && export ALPAMAYO_GRAPH_DRAFT_VERIFY_PREFILL_STAGE=0 && export ALPAMAYO_GRAPH_DECODE_STAGE=0 && export ALPAMAYO_GRAPH_ACTION_STAGE=0 && export ALPAMAYO_PC_TRACE_PATH=/mnt/e/ture_opamayo/openpilot/artifacts/alpamayo_speed/pc_endpoint_openpilot_controller_norm_820_65kpix.trace.jsonl && python -m openpilot.selfdrive.alpamayo.pc_endpoint --host 0.0.0.0 --port 8765'
```

Stop endpoint:

```powershell
wsl.exe -e bash -lc "pkill -f 'openpilot.selfdrive.alpamayo.pc_endpoin[t]' || true"
```

## Exact MetaDrive demo and render command for the 820 proof run

Run from PowerShell after the endpoint is resident:

```powershell
cd E:\ture_opamayo\openpilot
$out = "artifacts\reasoned_trajectory_poc\metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601"
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --engine alpamayofast --novel-scene random_mixed --frames 820 --speed-mps 2.5 --tick-sec 0.05 --deadline-ms 100 --save-every 1 --map 3 --seed 7 --random-scene-seed 42 --camera-width 256 --camera-height 256 --alpamayo-endpoint-url http://127.0.0.1:8765/alpamayo --alpamayo-endpoint-timeout-s 300 --alpamayo-num-frames 4 --alpamayo-query-every 2 --alpamayo-catchup-stride-steps 1 --alpamayo-control-mode planner_bridge --alpamayo-lateral-preview-m 12 --alpamayo-max-lateral-offset-m 0.8 --alpamayo-steer-sign -1 --alpamayo-longitudinal-preview-s 1.0 --alpamayo-max-accel-mps2 1.5 --alpamayo-max-decel-mps2 3.0 --alpamayo-reasoning-overlay --alpamayo-reasoning-overlay-chars 220 --out $out
py -3.11 tools\reasoned_trajectory_poc\render_demo_videos.py --run-dir $out --prefix openpilot_controller_norm_randommixed_820_65kpix --fps 20
```

Artifact paths:

```text
Run directory:
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601

Video:
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601/videos/side_by_side_openpilot_controller_norm_randommixed_820_65kpix.mp4

Comparison JSON:
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601/comparison_alpamayofast.json

Per-frame records:
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601/vlm/episode_alpamayofast_records.json

Reasoning log:
openpilot/artifacts/reasoned_trajectory_poc/metadrive_openpilot_controller_norm_randommixed_820_65kpix_20260601/vlm/alpamayo_response_reasoning.jsonl
```

## Ubuntu / Linux Native Runbook

The commands above are the original Windows/WSL bring-up path. This fork also has a validated **native Ubuntu** path (tested on a single NVIDIA L40S, no WSL/Windows layer at all). It uses two wrapper scripts instead of the raw PowerShell/WSL commands:

```bash
# Terminal 1 -- model endpoint
tools/reasoned_trajectory_poc/run_ubuntu_alpamayo_endpoint.sh

# Terminal 2 -- MetaDrive episode + rendered video
tools/reasoned_trajectory_poc/run_ubuntu_metadrive_demo.sh <run-name> [extra run_metadrive_overlay_demo.py args...]
```

Output lands under `artifacts/reasoned_trajectory_poc/<run-name>/videos/side_by_side_<run-name>.mp4`.

### One-time environment setup

```bash
# System deps + uv sync (installs MetaDrive, builds nothing GPU-specific yet)
bash tools/ubuntu_setup.sh

# Submodules this checkout doesn't carry (opendbc, panda, msgq, rednose, teleoprtc, tinygrad)
git submodule update --init --recursive --depth 1

# Native cereal/capnp extensions
scons -j$(nproc)

# CUDA 12.8 toolkit (nvcc), matching the pinned nvidia-*-cu12 wheel versions
sudo apt-get install -y cuda-toolkit-12-8
export PATH="/usr/local/cuda-12.8/bin:$PATH"   # persist in ~/.bashrc

# alpamayo1.5/ vendored source + venv (torch/flash-attn/transformers)
# -- see the "Rebuild The Venv From Scratch" section above; identical on Ubuntu.

# Base model (gated, needs your own HF account with the license accepted)
source alpamayo1.5/a1_5_venv/bin/activate
hf auth login
hf download nvidia/Alpamayo-1.5-10B
```

### Scene library

Beyond the 820-frame `random_mixed` proof run, `run_ubuntu_metadrive_demo.sh <name> --novel-scene <scene> --frames 300` records a small library of distinct behaviors, all clean (0 endpoint errors):

| Scene | `--novel-scene` | What it shows |
|---|---|---|
| Construction | `construction` | Lateral avoidance -- most visually obvious lane-offset difference |
| Pedestrian | `pedestrian` | Yield behavior (shorter route, completes early) |
| Traffic light | `traffic_light` | Stop/go response to signal state |
| Braking lead | `braking_lead` | Longitudinal following/braking response |
| Mixed | `random_mixed` | Combined lateral + longitudinal, closest to the original proof scenario |

Output for each: `artifacts/reasoned_trajectory_poc/<name>/videos/side_by_side_<name>.mp4`. The debug/reasoning text overlay and the STOCK/VLM corner tag scale with `--board-width`/`--board-height` -- bump those (default 640x400 in the wrapper script) for a bigger, more legible video.

## Current control path

Current valid MetaDrive Alpamayo control path:

```text
--alpamayo-control-mode planner_bridge
```

Important naming caveat:

```text
The CLI mode is still named planner_bridge, but the current valid proof path no longer uses the old route-follower actuator bridge as the Alpamayo controller. In the current proof, Alpamayo semanticPlan is converted into an openpilot-shaped plan and driven through MetaDriveOpenPilotController. Per-frame records identify this as control_source=alpamayo_openpilot_controller.
```

Current behavior:

```text
Alpamayo semanticPlan -> openpilot-shaped plan -> desired curvature/speed -> openpilot curvature limiting -> MetaDrive actuator adapter
```

Production modeld semantic fusion state:

```text
Valid Alpamayo semanticPlan receives full authority in semantic_fusion.py instead of being near-horizon blended away.
```

Superseded control paths:

```text
Direct trajectory mode is not a valid MetaDrive driving interface. It previously fed Alpamayo trajectory velocity/acceleration too directly into low-level MetaDrive gas/steer and caused speed runaway or out-of-road termination.

The old route-follower planner_bridge artifacts are no longer the current proof path. They are retained only as historical timing/controller debugging references.
```

## Current speedup mechanism

The 820-frame proof uses the resident speed-first endpoint path:

```text
ALPAMAYO_DFLASH_ENABLED=1
ALPAMAYO_STREAMING_VLM_TRUST_SHIFTED_DRAFT=1
ALPAMAYO_STREAMING_VLM_TRUSTED_REPLAY_REFRESH_INTERVAL=24
ALPAMAYO_STREAMING_VLM_SOURCE_CACHE_DRAFT_VERIFY_UNVERIFIED=0
ALPAMAYO_MIN_PIXELS=65536
ALPAMAYO_MAX_PIXELS=65536
```

What this means:

```text
The endpoint stays resident.
The target model is Alpamayo-1.5-10B-finetuned.
The draft model is Alpamayo-1.5-DFlash.
The runtime uses trusted shifted-frame cache reuse on the warm path.
The warm path avoids full first-principles multimodal prompt rebuild on every endpoint response.
The controller applies the latest age-adjusted plan every sim frame rather than requiring a fully fresh endpoint response every sim frame.
```

What the 36 to 42 ms number means:

```text
It is the warm endpoint response time observed in the current 820-frame proof run after the endpoint/cache path was resident.
It is not cold model load time.
It is not proof that a full fresh 65k-pixel current-prompt VLM prefill, target verification, reasoning decode, and diffusion planner all run from scratch in 40 ms.
```

## Pixel/token knobs

The current 820-frame proof was run at:

```text
ALPAMAYO_MIN_PIXELS=65536
ALPAMAYO_MAX_PIXELS=65536
```

Other measured speed-first probes showed that lowering pixels is a real knob, but not sufficient by itself for a correctness-preserving under-100 ms fresh state path:

```text
8k road-only visual-fill precompute path: p50 about 115.5 ms, p95 about 153.4 ms.
8k road-only skip-new-visual-fill draft path: p50 about 16.7 ms, p95 about 23.6 ms, max about 76.0 ms.
```

Caveat:

```text
The under-100 ms skip-fill path is an explicit unverified speed-first draft path because it skips the newest unmatched visual-span target-VLM fill. The correctness-preserving fresh visual-fill path still misses <=100 ms.
```

## Current prompt path

Adapter prompt construction follows the released Alpamayo helper path:

```text
helper.create_message(..., nav_text=nav_text)
processor.apply_chat_template(..., add_generation_prompt=False, continue_final_message=True)
```

System prompt:

```text
You are a driving assistant that generates safe and accurate actions.
```

Current MetaDrive navigation text:

```text
Follow the lane and continue safely.
```

Serialized user prompt shape:

```text
<|traj_history_start|><|traj_history|> repeated 48 times <|traj_history_end|><|route_start|>Follow the lane and continue safely.<|route_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory.
```

Camera mapping used by the harness:

```text
wideRoad -> camera id 1 -> Front camera
road -> camera id 6 -> Front telephoto camera
```

## Workspace paths

Windows workspace:

```powershell
E:\ture_opamayo
```

openpilot git checkout:

```powershell
E:\ture_opamayo\openpilot
```

Alpamayo endpoint workspace:

```powershell
E:\ture_opamayo\openpilot_alpamayo
```

WSL endpoint workspace:

```bash
/mnt/e/ture_opamayo/openpilot_alpamayo
```

Target model:

```bash
/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-10B-finetuned
```

DFlash draft model:

```bash
/mnt/e/ture_opamayo/openpilot_alpamayo/Alpamayo-1.5-DFlash
```

Alpamayo package root:

```bash
/mnt/g/alpamayo1.5
```

DFlash package root:

```bash
/mnt/e/ture_opamayo/openpilot_alpamayo/dflash
```

Python venv:

```bash
/mnt/g/alpamayo1.5/a1_5_venv
```

## OPAMAYO branch and push state

Current pushed branch:

```text
https://github.com/jagoff2/OPAMAYO/tree/alpamayo-speedfirst-metadrive-20260601-openpilot-controller
```

Commit:

```text
f74e7dc66 Update Alpamayo openpilot-controller fast path
```

Remote:

```text
git@github.com:jagoff2/OPAMAYO.git
```

PR creation URL:

```text
https://github.com/jagoff2/OPAMAYO/pull/new/alpamayo-speedfirst-metadrive-20260601-openpilot-controller
```

Older branch:

```text
https://github.com/jagoff2/OPAMAYO/tree/alpamayo-speedfirst-metadrive-20260531-101751
```

Push note:

```text
The older branch rejected a non-fast-forward push because the remote branch had commits absent from the local checkout. The latest state was therefore pushed to the new 20260601 openpilot-controller branch instead of force-pushing or merging remote history.
```

## Superseded artifacts and commands

Superseded 300-frame route-follower/planner_bridge proof:

```text
openpilot/artifacts/reasoned_trajectory_poc/metadrive_old_controller_alpamayofast_plannerbridge_fullreason_300_20260531_095823
```

Reason superseded:

```text
It used the older route-follower actuator interpretation. It is not the current controller proof path.
```

Invalid endpoint contract video as side-by-side controller proof:

```text
openpilot_alpamayo/openpilot/artifacts/alpamayo_speed/metadrive_side_by_side_speedfirst_validplan_300f_65kpix.mp4
openpilot_alpamayo/openpilot/artifacts/alpamayo_speed/metadrive_side_by_side_speedfirst_validplan_300f_65kpix_benchmark.json
```

Reason invalid:

```text
That harness was an endpoint contract/timing harness. The produced MP4 showed stock.endpoint_calls=0, alpamayo.endpoint_calls=0, and Alpamayo stayed on stock_route_follower.
```

Superseded direct trajectory failures:

```text
openpilot/artifacts/reasoned_trajectory_poc/metadrive_old_controller_alpamayofast_300_20260531_093508
openpilot/artifacts/reasoned_trajectory_poc/metadrive_old_controller_alpamayofast_signflip_300_20260531_094112
```

Reason invalid:

```text
Direct trajectory mode is a failure reproducer, not the current valid driving path.
```

## Remaining known gaps

Current proof is strong enough as a speed-first MetaDrive controller artifact, but not production-complete.

Known gaps:

```text
The fastest warm timing depends on trusted shifted cache reuse.
The production-correct fresh current-prompt visual KV path is still not complete.
The under-100 ms skip-fill draft path is explicitly unverified for newest visual K/V.
CUDA graph/static backend partitioning is not complete.
The VLM decode/prefill path is still not a fully static TensorRT-LLM/vLLM-style backend.
W4A8/ParoQuant activation-INT8 fast prefill is not production-complete.
```

Practical current status:

```text
For MetaDrive proof video: use the 820-frame openpilot-controller command above.
For warm timing claim: cite the 820-frame proof p95 36.49 ms, p99 40.02 ms, max 42.13 ms at 65k pixels.
For production readiness: do not claim production-safe fresh current-frame cache correctness yet.
```
