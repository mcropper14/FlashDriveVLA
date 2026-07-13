# Reasoned Trajectory Program POC for openpilot

This repository is a local-PC proof of concept for adding a VLM-conditioned
trajectory compiler upstream of openpilot controls. It is not a production
driving stack, not validated for road use, and not intended to bypass panda,
opendbc, car safety hooks, driver monitoring, or the existing openpilot vehicle
interface.

The POC implements the "pseudo-Alpamayo" architecture: instead of trying to run
a full trained action decoder, the system asks a small VLM to identify bounded
scene constraints from a driver-UI-style scene board. Those constraints are
compiled into a deterministic trajectory program, then into path/curvature and
speed constraints that remain inside the existing planner/control boundary.

The current working demo runs on this PC in MetaDrive with a TensorRT
Qwen2.5-VL-3B label-scoring backend on the local NVIDIA RTX 5060 Ti. The fast
path is synchronous and same-frame: one VLM scoring pass per MetaDrive frame,
with strict manifest checks and a 50 ms planning deadline. The original target
direction remains an eGPU path, but the checked-in working demo currently uses
the local NVIDIA PC GPU backend.

## Current Status

Working:

- Openpilot master fork with a new `reasoned_plannerd` process.
- New `reasonedTrajectoryPlan` cereal event for audit/debug telemetry.
- Strict `RTPv1` parser with bounded fields and grammar validation.
- Deterministic PathSynth compiler for lateral bias, avoid zones, speed
  modifiers, stop/yield constraints, and candidate selection.
- TensorRT Qwen2.5-VL-3B fast label scorers:
  - fixed full-frame `168 px` scene-board input,
  - verified FP32-build TensorRT vision tower, with FP16 output features,
  - current red/green signal scoring through Qwen choice-mode text labels,
    not a trained signal head,
  - legacy NVFP4 TensorRT text/label scorer retained for construction,
    pedestrian, and vehicle labels,
  - strict runtime manifest for prompt, labels, thresholds, image mode, image
    size, model config, and engine shapes.
- Synchronous MetaDrive loop that publishes same-frame VLM plans at 20 Hz with
  zero stale-frame reuse in the fast path.
- Older PyTorch/full384 async Qwen worker retained for diagnostics and visual
  quality experiments.
- MetaDrive closed-loop demos comparing stock path following versus the VLM
  reasoned trajectory path.
- Video generation for stock, VLM, padded stock, and side-by-side comparisons.
- Unit tests for RTP parsing, PathSynth, relative speed caps, durable speed
  replacement, construction side detection, simulator sign conversion, and
  contradictory durable lateral plan replacement.

Known limitations:

- The realtime path is no longer open-ended Qwen image-to-text generation. It
  uses Qwen as a constrained visual label scorer, then deterministically
  compiles labels into RTP fields.
- Red/green traffic-light handling is back on the Qwen label path via
  choice-mode answer-token scoring. The earlier trained visual-head experiment
  is retained only as a diagnostic artifact and is not the production default.
- The fast runtime is shape-bound by design. Changing prompt text, label groups,
  thresholds, image size, model files, or selected engine shapes requires a
  matching manifest and may require rebuilding TensorRT engines.
- The fast input is `full168`, which is much faster than full384 but loses
  visual detail. This is the largest accuracy sacrifice in the current speed
  path.
- The latest lateral-only mixed run proves realtime lateral control influence
  and clears the sign-fixed construction seed better than stock. It does not
  validate pedestrian handling because VLM speed control is intentionally
  disabled in that mode.
- The current fast Qwen backend is CUDA/TensorRT/NVIDIA. It is not yet a
  tinygrad AMD eGPU backend.
- The red/green Qwen choice-mode path is verified under the 50 ms frame budget. The
  broader production goal still requires bringing the remaining construction,
  pedestrian, vehicle, and other labels onto an equally robust under-50 ms path.
- Model weights, TensorRT engines, and run artifacts are intentionally ignored
  by git. They must be downloaded, built, or copied locally.
- The MetaDrive demo proves control influence and closed-loop behavior in sim.
  It does not prove real-world safety.

Production translation constraints:

- The simulator is only a closed-loop harness. A sim video is accepted only when
  the same Qwen label path, RTP compiler, stale-frame policy, and bounds would
  also run on camera-derived scene boards in the car.
- No synthetic pixel fallback is allowed in the production path. The
  `--enable-visual-*fallback` switches are demo-only isolators for renderer and
  controller bugs.
- No trained shortcut head is the default production signal path. Label
  decisions must remain auditable as Qwen prompt/input/choice-score records
  unless a later replacement is explicitly validated as a new perception model.
- Every published plan must carry `frame_id`, `rtp_source_frame_id`,
  `rtp_age_frames`, labels, label scores, choice metadata, RTP text, and
  compiler output so a road log can prove what perception judgement changed the
  path.
- Steering authority comes only from construction-side labels and deterministic
  avoid-zone compilation. Pedestrians, vehicles, animals, stop signs, and
  traffic lights may slow or stop in lane, but must not create a lateral swerve.
- The current repo is not road-ready until the non-sim evaluations pass on real
  road video/log replay for construction, pedestrians, lead vehicles, cut-ins,
  stop signs, stoplights, occlusion, and mixed scenes.

## Architecture

At a high level:

```text
camera frame + vehicle state
        |
        v
driver-UI-style scene board, fixed full168 for fast path
        |
        v
TensorRT Qwen VLM label scorer
        |
        v
label-to-RTP compiler
        |
        v
strict RTPv1 program
        |
        v
PathSynth deterministic compiler
        |
        v
bounded lateral/speed plan
        |
        v
lateralManeuverPlan / controlsd path
```

The VLM does not command steering, torque, throttle, brake, CAN, or panda. In
the fast path it only scores visual labels. The label compiler emits bounded
semantic constraints, and PathSynth turns those constraints into path candidates,
speed caps, and avoid-zone costs.

The important boundary is:

- VLM output is text and can be invalid.
- Parser validates the text.
- Compiler clamps all geometry and speed changes.
- Controls consume only the compiled plan.

## Repository Changes

Core runtime:

- `selfdrive/controls/reasoned/rtp.py`
  - Strict RTP parser.
  - Bounded `scene`, `evidence`, `meta`, `branch`, `lat_bias_m`,
    `speed_cap_mps`, `stop_s`, `avoid`, `weights`, and `confidence`.
  - `speed_cap_mps` now accepts absolute values, `none`, percentage strings
    such as `25%`, and scale strings such as `0.25x`.

- `selfdrive/controls/reasoned/pathsynth.py`
  - Base plan abstraction.
  - Candidate path generation.
  - Deterministic trajectory compiler.
  - Relative speed scaling against desired/base speed.
  - Curvature clipping and bounded lateral offset handling.

- `selfdrive/controls/reasoned/planner.py`
  - Planner orchestration.
  - Scene-board rendering.
  - VLM backend call.
  - Same-frame synchronous mode and bounded async mode.
  - RTP validation and PathSynth timing.

- `selfdrive/controls/reasoned/vlm.py`
  - Static RTP backend for non-VLM tests.
  - External subprocess backend for GPU VLM workers.
  - Async worker mode with latest-frame behavior and source-frame age metadata.
  - Persistent JSONL backend with ready-marker support for prewarmed TensorRT
    workers.
  - JPEG scene-board transport for lower IPC overhead.

- `selfdrive/controls/reasoned/ui_scene_board.py`
  - Full-frame, driver-UI-style scene board used by the VLM.
  - Includes camera frame, path overlay, HUD state, and visual affordances.
  - The green planned corridor now uses a `0.90 m` half-width so Qwen evaluates
    the approximate ego vehicle envelope/risk corridor rather than a narrow
    center ribbon.
  - The magenta base-path reference is opt-in. The default VLM board shows the
    actual tracked green path without an extra synthetic line that can be
    mistaken for construction or lane geometry.

- `selfdrive/controls/reasoned_plannerd.py`
  - PC-only process that consumes `modelV2` and `carState`, runs the reasoned
    planner, publishes `lateralManeuverPlan`, and publishes audit telemetry.

Openpilot integration:

- `cereal/custom.capnp`
  - Defines `ReasonedTrajectoryPlan`.

- `cereal/log.capnp`
  - Wires the custom event at the reserved custom slot.

- `cereal/services.py`
  - Adds `reasonedTrajectoryPlan` service.

- `system/manager/process_config.py`
  - Adds PC-only `reasoned_plannerd`, gated by `ReasonedPlanner` param or
    `ENABLE_REASONED_PLANNER=1`.

- `selfdrive/controls/controlsd.py`
  - Uses `lateralManeuverPlan` mono time when a valid lateral maneuver plan is
    present, otherwise falls back to the model timing path.

POC tools:

- `tools/reasoned_trajectory_poc/qwen_trt_label_engine.py`
  - Current fast Qwen backend.
  - Exports/builds fixed-shape ONNX/TensorRT engines.
  - Runs the persistent JSONL label-scoring worker used by the synchronous demo.
  - Supports `check-artifacts`, strict runtime manifests, and a 50 ms `gate`.
  - Uses rotating two-label groups with a shared fixed-shape text engine.
  - Emits strict RTP from scored labels.

- `tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py`
  - Older PyTorch/CUDA VLM worker retained for diagnostics.
  - Uses full384 scene boards.
  - Uses batched yes/no label scoring instead of free-form RTP generation.
  - Rotates label groups to reduce latency.
  - Keeps durable labels through short occlusion/splash, with negative-score
    clearing.
  - Emits deterministic strict RTP from labels.

- `tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py`
  - Closed-loop MetaDrive demo runner.
  - Runs stock and VLM episodes.
  - Spawns random mixed construction and pedestrian scenes.
  - Converts MetaDrive camera frames from BGR to RGB before scene-board
    rendering. Without this, construction cones render blue and Qwen treats them
    like road/runway markers rather than realistic orange/white traffic-control
    devices.
  - Uses `DefaultVehicle` with visual heading equal to route heading for
    route-vehicle scenes. The saved lead-track state remains physical
    distance/lateral/speed data; simulator `expected_lead_class` is only an
    evaluation label.
  - Applies durable lateral and speed plans.
  - Tracks collisions, latency, RTP age, path deltas, speed deltas, and saved
    input frames.

- `tools/reasoned_trajectory_poc/render_demo_videos.py`
  - Builds stock, padded-stock, VLM, and side-by-side MP4s from saved PNG
    frames.

- `tools/reasoned_trajectory_poc/run_local_demo.py`
  - Hardware-free parser/compiler demo using static RTP.

- `tools/reasoned_trajectory_poc/benchmark_vlm_backend.py`
  - Backend timing harness.

- `tools/reasoned_trajectory_poc/evaluate_qwen_trt_video.py`
  - Runs the TensorRT scorer against a real input video and reports whether the
    system would have attempted lateral modification.

- `tools/reasoned_trajectory_poc/probe_qwen_novel_scenes.py`
  - One-frame Qwen scene probing in MetaDrive.

- `tools/reasoned_trajectory_poc/diagnose_qwen_scene_perception.py`
  - Diagnostic prompt/label experiments.

- `tools/reasoned_trajectory_poc/*smolvlm*`, `tools/reasoned_trajectory_poc/nanovlm_worker.py`
  - Earlier backend experiments retained for reference.

Tests:

- `selfdrive/controls/tests/test_reasoned_trajectory.py`
  - Main focused unit test suite for this POC.

## RTPv1 Program Format

The VLM path ultimately compiles to this fixed text shape:

```text
RTPv1
scene=construction_right
evidence=[cones_barrier_right_edge]
meta=BIAS_LEFT_AND_SLOW
branch=base
lat_bias_m=1.25
speed_cap_mps=25%
stop_s=none
avoid=[right_edge_s8_48_margin1.25]
weights=[obs2.5,lane1.4,comfort1.0,base0.7,vlm1.0]
confidence=0.72
```

Important fields:

- `scene`: compact scene class.
- `evidence`: visible evidence tokens.
- `meta`: maneuver prior.
- `branch`: candidate branch hint.
- `lat_bias_m`: openpilot/PathSynth lateral bias. Positive means left in the
  openpilot compiler convention.
- `speed_cap_mps`: legacy name, now supports:
  - `none`
  - absolute m/s, for compatibility
  - percent, for example `25%`
  - scale, for example `0.25x`
- `stop_s`: stop/yield distance, or `none`.
- `avoid`: bounded avoid-zone tokens such as `right_edge_s8_48_margin1.25`.
- `weights`: bounded optimizer weights.
- `confidence`: `0.0` to `1.0`.

The current Qwen label compiler emits percentages for normal slowdowns:

- Construction: `speed_cap_mps=25%`
- Mixed agent/construction yield: `speed_cap_mps=15%`
- Stop: `speed_cap_mps=0.0`

The parser stores percentage/scale values as `speed_scale`, and PathSynth
resolves them against the current desired speed. A `25%` cap means:

```text
desired 8 m/s  -> 2.0 m/s cap
desired 10 m/s -> 2.5 m/s cap
desired 12 m/s -> 3.0 m/s cap
desired 20 m/s -> 5.0 m/s cap
```

## Construction Side and Simulator Sign Convention

A previous bug made the sim appear to steer into cones. The root cause was two
separate issues:

1. Generic construction labels collapsed into a right-edge avoid program.
2. MetaDrive lane lateral sign in the harness is opposite the openpilot/PathSynth
   sign convention.

The current path fixes this by:

- Scoring `construction_left` and `construction_right` relative to the green
  planned path.
- Emitting side-specific RTP:
  - Right-side construction -> `BIAS_LEFT`, positive openpilot
    `lat_bias_m`, `right_edge...`
  - Left-side construction -> `BIAS_RIGHT`, negative openpilot
    `lat_bias_m`, `left_edge...`
- Converting openpilot lateral sign at the MetaDrive boundary.
- Converting the active MetaDrive controller offset back to openpilot sign before
  rendering the green scene-board path, so Qwen sees the same path the controller
  is actually tracking.
- Clearing active durable lateral plans that pull the opposite direction when a
  new signed plan arrives with sufficient confidence.
- Clearing stale corridor-object speed caps when the current VLM program no
  longer contains a pedestrian/vehicle/animal/path-conflict judgement.
- Ignoring new red-light stop plans once the car is already past the configured
  traffic-light stop line, and using `traffic_light_full_stop_m` as the actual
  stop-and-wait zone.

This is not a "cones always mean left" shortcut. The VLM has to identify the
construction side relative to the path, and the compiler/sign conversion then
does the deterministic movement.

## VLM Backend

The current fast backend is:

```text
tools/reasoned_trajectory_poc/qwen_trt_label_engine.py
```

Runtime model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
local model dir: models/vlm/qwen2_5_vl_3b_instruct
CUDA toolkit: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
TensorRT: 10.16.1.11
GPU used for current fast path: RTX 5060 Ti 16 GB, compute capability 12.0
```

Current external TensorRT artifacts:

```text
F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.engine
F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.onnx
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.onnx
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
```

Current behavior contract hash:

```text
cbf5827c7761cda0c61ad42eb9c83422ef881f7a82878ef1bea17ddb82ac8d54
```

Current red/green Qwen choice-mode runtime defaults:

```text
--runtime-mode score
--image-mode full
--image-size 168
--vision-engine F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.engine
--text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine
--label-decision-mode choice
--text-output hidden
--text-seq-len 576
--text-position-mode auto
--score-label-groups "red_stop_light,green_go_light"
--require-manifest
```

Legacy rotating text-label runtime defaults:

```text
--image-mode full
--image-size 168
--text-seq-len 220
--score-rotate-groups
--score-rotate-shared-engine
--score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path"
--score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5"
--require-manifest
```

The active prompt contract tells Qwen to consider only hazards that overlap,
intrude into, narrow, block, or are imminently entering the green planned
corridor. Cones, barriers, pedestrians, vehicles, or other objects that are
merely visible off to the side should not trigger steering or yielding.

Production translation rule:

```text
Default runtime: no synthetic pixel fallbacks.
Demo-only flags: --enable-visual-fallbacks,
                 --enable-visual-signal-fallback,
                 --enable-visual-construction-fallback,
                 --enable-visual-stop-fallback
```

Those flags inspect rendered scene-board pixels such as clean traffic-light
overlays or synthetic cone colors. They are useful for isolating sim-renderer
and controller bugs, but they are not a road-ready perception mechanism. A
production road run must leave those flags off and rely on Qwen label scoring,
validated scene-board construction, and deterministic compiler bounds. After
the fallback gate was added, a no-fallback probe still detected a red light on
the saved sim frame, but it did not call construction on that same frame. That
means the current construction video is a control-harness proof, not yet a
production perception proof.

After prompt or scene-board changes, the strict runtime manifest must be
rewritten only after validation. The current manifest includes the prompt,
labels, thresholds, rotating behavior, image contract, and scene-board geometry,
including the widened `0.60 m` planned-corridor half-width.

The older PyTorch/CUDA backend remains available for diagnostics:

```text
tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py
```

That older worker uses full384 scene boards and can be useful for visual-quality
experiments, but it is too slow for synchronous 20 Hz use.

Model files are intentionally ignored by git. Download them locally before
building or running the Qwen backend:

```powershell
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models\vlm\qwen2_5_vl_3b_instruct
```

AMD/tinygrad eGPU support is a future backend target, not the active working
path.

## VRAM And Runtime Requirement

Legacy PyTorch/CUDA full384 score mode measured on this machine:

```text
model load incremental VRAM:        ~7,364 MiB
after one full384 score inference:  ~7,666 MiB
PyTorch peak reserved:              ~7,522 MiB
model files on disk:                ~7.0 GB
```

Interpretation:

- 8 GB dedicated GPU: technically close, not reliable for this exact fp16 Qwen
  path.
- 8 GB desktop GPU with other applications sharing VRAM: not realistic.
- 12 GB GPU: realistic minimum for the current Qwen path.
- 16 GB GPU: comfortable, and matches the current working local setup.

For an 8 GB AMD eGPU target, the likely path is a smaller model, quantized model,
or a memory-frugal tinygrad/AMD implementation. The current CUDA worker does not
prove the 8 GB AMD path.

Current TensorRT fast path requirements are different:

- NVIDIA GPU with TensorRT engine compatibility.
- CUDA toolkit with `compute_120`/`sm_120` support for this RTX 5060 Ti setup.
- TensorRT with FP4 support for the NVFP4 Qwen text engine. The current
  red/green choice-mode runtime uses the text engine hidden output, then scores
  the small answer-token set outside TensorRT.
- External disk space for engines and ONNX files. Current artifacts live under
  `F:\qwen_trt_export` and are not committed.
- Runtime manifest must match the active prompt/label/image/threshold contract.

## Local Demo Commands

Run focused unit tests:

```powershell
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
```

Run syntax checks for the main POC files:

```powershell
py -3.11 -m py_compile `
  selfdrive\controls\reasoned\rtp.py `
  selfdrive\controls\reasoned\pathsynth.py `
  selfdrive\controls\reasoned\planner.py `
  selfdrive\controls\reasoned\vlm.py `
  selfdrive\controls\reasoned\scene_board.py `
  selfdrive\controls\reasoned_plannerd.py `
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  tools\reasoned_trajectory_poc\evaluate_qwen_trt_video.py `
  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py `
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py `
  tools\reasoned_trajectory_poc\render_demo_videos.py
```

Run a static local parser/compiler demo without the VLM:

```powershell
py -3.11 tools\reasoned_trajectory_poc\run_local_demo.py `
  --frames 3 `
  --scenario construction `
  --speed-mps 12 `
  --out artifacts\reasoned_trajectory_poc\local_static_demo
```

Check that the TensorRT fast-path artifacts and behavior manifest match the
current code contract:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"

py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine `
  --text-seq-len 220 `
  --score-rotate-shared-engine `
  --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" `
  --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" `
  --deadline-ms 50 `
  --iters 30 `
  --warmup 3 `
  --out artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_behavior_manifest.json `
  gate
```

Build and validate the current red/green Qwen choice-mode label scorer:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --label-decision-mode choice `
  --text-output hidden `
  --text-seq-len 576 `
  --score-labels red_stop_light,green_go_light `
  --image artifacts\reasoned_trajectory_poc\traffic_light_visual_probe_clean_signalhead\static\vlm_input_0000.png `
  --image-size 168 `
  --workspace-gb 8 `
  build-text

py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --vision-engine F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.engine `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine `
  --label-decision-mode choice `
  --text-output hidden `
  --text-seq-len 576 `
  --score-labels red_stop_light,green_go_light `
  --image-size 168 `
  --warmup 8 `
  --deadline-ms 50 `
  --write-manifest `
  check-artifacts

py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --vision-engine F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.engine `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine `
  --label-decision-mode choice `
  --text-output hidden `
  --text-seq-len 576 `
  --score-labels red_stop_light,green_go_light `
  --image-size 168 `
  --warmup 8 `
  --deadline-ms 50 `
  gate
```

Run the current synchronous mixed MetaDrive VLM demo at 2.5 m/s with VLM speed
control disabled, so the comparison isolates lateral behavior:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"

$out = 'artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_2p5mps_lateral_only'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:RTP_VLM_WAIT_READY = '1'
$env:RTP_VLM_IMAGE_FORMAT = 'jpeg'
$env:RTP_VLM_JPEG_QUALITY = '85'
$env:RTP_VLM_STDERR_PATH = "$out\vlm_stderr.log"
$env:RTP_VLM_SERVER_COMMAND = 'py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --require-manifest --ready-jsonl serve'

py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py `
  --engine vlm `
  --novel-scene random_mixed `
  --frames 300 `
  --speed-mps 2.5 `
  --disable-vlm-speed-control `
  --tick-sec 0.05 `
  --deadline-ms 50 `
  --save-every 1 `
  --map 3 `
  --seed 7 `
  --random-scene-seed 42 `
  --out $out
```

Render videos after a run:

```powershell
py -3.11 tools\reasoned_trajectory_poc\render_demo_videos.py `
  --run-dir artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_2p5mps_lateral_only `
  --prefix fast_mixed_300_sync_2p5mps_lateral_only `
  --fps 20
```

Expected videos:

```text
artifacts/.../videos/side_by_side_<prefix>.mp4
artifacts/.../videos/stock_<prefix>.mp4
artifacts/.../videos/stock_<prefix>_padded.mp4
artifacts/.../videos/vlm_<prefix>.mp4
```

## Ubuntu / Linux Native Runbook

The commands above are the original Windows/PowerShell path (RTX 5060 Ti, Blackwell, CUDA v13.2). This fork also has a validated **native Ubuntu** path, tested on a single NVIDIA L40S (Ada Lovelace, compute capability 8.9).

**The critical hardware difference: NVFP4 does not run here.** The original text engine used NVFP4 (4-bit float), which requires Blackwell-generation tensor cores. Ada Lovelace doesn't have them. The script already anticipates this -- `--text-precision` accepts `nvfp4`, `fp8`, or `fp16` -- so the Ubuntu path uses `fp8` instead, which Ada's 4th-generation tensor cores do accelerate natively. Measured full pipeline (vision + text) latency on the L40S: **p50 28.9ms, p99 29.8ms, max 29.8ms** against the 50ms deadline, over 30 iterations -- in the same range as the original NVFP4/Blackwell numbers (p99 34.5ms in the "Latest Measured Demo" section below).

### Build the engines

```bash
tools/reasoned_trajectory_poc/build_ubuntu_qwen_trt_engines.sh [path/to/trace-image.png]
```

This runs, in order: `build-vision` (fp32, unchanged from the original), `build-text` (fp8 instead of nvfp4), `check-artifacts --write-manifest`, `benchmark-groups` (the actual latency numbers). It needs a representative ~168px source image for tracing; defaults to a frame saved by an earlier MetaDrive run if one exists (`--image` otherwise).

### One-time environment setup

```bash
# Dedicated venv -- do not reuse the alpamayo1.5 venv or root .venv here.
# Pin explicitly: uv's default resolver drifts to incompatible torch/
# torchvision/transformers versions if you let it resolve tensorrt +
# nvidia-modelopt + torch together unconstrained.
uv venv qwen_trt_venv --python 3.12
source qwen_trt_venv/bin/activate
export PATH="/usr/local/cuda-12.8/bin:$PATH"
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "tensorrt==10.16.1.11" "transformers==4.57.1" onnx "nvidia-modelopt[onnx]" \
  accelerate pillow opencv-python-headless huggingface_hub \
  -c <(printf 'torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n')

# Public model, no gating
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/vlm/qwen2_5_vl_3b_instruct

# FP8 quantization + ONNX export of the 36-layer text model needs more RAM
# than a 30GB box has -- OOM-killed without this. Add swap once:
sudo fallocate -l 40G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

### Key differences from the Windows/RTX 5060 Ti setup, and why

- **`--text-precision fp8` instead of `nvfp4`.** NVFP4 is Blackwell-only; Ada Lovelace (L40S, RTX 40-series) has no FP4 tensor cores. `fp8` is natively accelerated on Ada and hits the same latency budget.
- **A hardcoded GPU-capability check rejected the L40S outright.** `qwen_trt_label_engine.py`'s `check-artifacts` had `if cuda_capability != (12, 0): issues.append("unexpected CUDA device capability")` -- true regardless of whether the engines actually built/ran correctly. Extended to also accept `(8, 9)`.
- **40GB swap required.** FP8 fake-quantization during ONNX tracing of the full 3B/36-layer model roughly doubles memory pressure; a 30GB-RAM box OOM-kills the export without swap headroom.
- **Explicit dependency pinning required.** Letting `uv` resolve `tensorrt` + `nvidia-modelopt` + `torch` together unconstrained silently drifted `torch`/`torchvision` to mismatched versions (`operator torchvision::nms does not exist`) and jumped `transformers` to an incompatible major version. Pin all three explicitly (see setup above).
- **A pre-existing (not Ubuntu-specific) bug in the manifest-freshness check.** `gate` internally forces `require_manifest=True` and resolves label groups narrowly (from `--score-labels`), while standalone `check-artifacts --write-manifest` always resolves the full `--score-label-groups` default list (`_groups_for_runtime`'s special-case list is `("benchmark-groups", "check-artifacts")`). The two will never agree for a single-group deployment like the red/green choice-mode engine, on any OS. Worked around by reading latency directly off `benchmark-groups --score-label-groups "red_stop_light,green_go_light"`, which is what `gate` calls internally after its (failing) manifest check.

## Latest Measured Demo

Latest mixed TensorRT VLM demo with construction, pedestrian labels, and a
cycling red/green traffic light:

```text
artifacts/reasoned_trajectory_poc/metadrive_mixed_wait_stop_go_fullstop10_20260524_0328
```

Summary:

```text
stock frames:                  420
stock mean speed:              1.0982 m/s

VLM frames:                    420
VLM valid publishes:           420
VLM deadline misses:           0
VLM async max RTP age:         4 frames
VLM p99 planner overhead:      3.583 ms
VLM max planner overhead:      4.817 ms
VLM mean speed:                0.9366 m/s
VLM min construction clearance:1.949 m route-space
VLM min pedestrian clearance:  2.546 m route-space
VLM active lateral offset:     0.0..1.25 MetaDrive m
VLM scene-board lateral offset:-1.25..0.0 openpilot m
VLM dominant avoid source:     left_edge_s8_48_margin1.25
red phase near-stop frames:    59 frames at <= 0.12 m/s
green phase acceleration:      0.05 m/s at frame 260 -> 1.51 m/s at frame 280
```

Side verification:

```text
spawned construction side:     left-side clusters, negative MetaDrive laterals
RTP source:                    left_edge_s8_48_margin1.25
controller target:             positive MetaDrive lateral, away from left cones
scene-board path:              negative openpilot lateral, same physical path
red light behavior:            slows to near-zero and waits through red
green light behavior:          visual green clears the red stop plan and target
                                speed returns to 2.5 m/s on construction-only
                                frames
```

Videos:

```text
artifacts/.../videos/side_by_side_mixed_wait_stop_go_fullstop10.mp4
artifacts/.../videos/stock_mixed_wait_stop_go_fullstop10.mp4
artifacts/.../videos/vlm_mixed_wait_stop_go_fullstop10.mp4
```

The matching manifest-gated 50 ms benchmark is:

```text
artifacts/reasoned_trajectory_poc/qwen_trt_50ms_gate_widecorridor_signfix_manifest.json
p99_total_ms 34.458
max_total_ms 34.458
contract_sha256 cf6c028ed0580f03db61300884c0b777bad6c750741998d64fdf98a0d5319f29
```

Latest red/green Qwen choice-mode signal gate:

```text
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
runtime_mode score
label_decision_mode choice
text_output hidden
text_seq_len 576
vision_engine F:\qwen_trt_export\vision_static_fp32\qwen_vision_full168_static_fp32.engine
text_engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine
contract_sha256 cbf5827c7761cda0c61ad42eb9c83422ef881f7a82878ef1bea17ddb82ac8d54
```

Latest hybrid full-label attempt:

```text
artifacts/reasoned_trajectory_poc/hybrid_seq576_benchmark_groups.json
runtime: 576-token NVFP4 text scorer plus visual signal head
signal_head_ms p99: 0.433
trt_text_ms p99: 42.899
total_ms p99: 65.910
status: over the 50 ms frame budget
```

This means the red/green signal path is fixed under budget, but the complete
all-label production path is not finished. The remaining bottleneck is the
36-layer text tower at the current verbose prompt length. The next production
path must keep the auditable label-scoring contract while making construction,
pedestrian, vehicle, and related groups fast and reliable under the same
manifest-gated runtime.

## Git Hygiene

Ignored local-only paths:

```text
artifacts/
models/
tools/nanoVLM/
tools/nanoVLM_v01/
__pycache__/
```

Do not push:

- Downloaded Hugging Face model weights.
- Generated PNG frame dumps.
- Generated MP4 videos.
- Benchmark artifacts.
- Nested third-party git clones used during experiments.

Push source and docs only:

- `GOAL.MD`
- `README_REASONED_TRAJECTORY_POC.md`
- `.gitignore`
- `cereal/custom.capnp`
- `cereal/log.capnp`
- `cereal/services.py`
- `selfdrive/controls/controlsd.py`
- `system/manager/process_config.py`
- `selfdrive/controls/reasoned/`
- `selfdrive/controls/reasoned_plannerd.py`
- `selfdrive/controls/tests/test_reasoned_trajectory.py`
- `tools/reasoned_trajectory_poc/`

## Suggested Push Workflow

Once the remote repository URL is known:

```powershell
git status --short
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
py -3.11 -m py_compile selfdrive\controls\reasoned\*.py selfdrive\controls\reasoned_plannerd.py tools\reasoned_trajectory_poc\*.py

git add .gitignore GOAL.MD README_REASONED_TRAJECTORY_POC.md
git add cereal\custom.capnp cereal\log.capnp cereal\services.py
git add selfdrive\controls\controlsd.py system\manager\process_config.py
git add selfdrive\controls\reasoned selfdrive\controls\reasoned_plannerd.py selfdrive\controls\tests\test_reasoned_trajectory.py
git add tools\reasoned_trajectory_poc

git commit -m "Add reasoned trajectory VLM POC"
git remote add origin <repo-url>
git push -u origin HEAD
```

If a remote already exists, replace the `git remote add origin` line with:

```powershell
git remote set-url origin <repo-url>
```

Before pushing, verify that `git status --short` does not list `artifacts/`,
`models/`, `tools/nanoVLM/`, or `tools/nanoVLM_v01/`.
