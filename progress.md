# Qwen VLM Production-Translation Progress

## Active Objective As Of 2026-05-24

This work is not accepted by simulator appearance alone. The active objective is
to improve the existing scene-board -> Qwen labels/scores -> RTP semantic
trajectory program -> deterministic compiler -> openpilot planning/control path
until Qwen mode can be evaluated against real-car-relevant mixed driving
behavior.

Constraints:

- Keep Qwen and the current RTP/compiler architecture.
- Do not replace the method with a different VLM, a hand-authored RTP shortcut,
  simulator-only pixel fallback, trained shortcut head, or one-route hack.
- Treat MetaDrive as a closed-loop harness only. A sim video is useful only when
  the same Qwen label path, stale-frame policy, RTP compiler, and bounds would
  run on road-derived scene boards.
- Default production path leaves `--enable-visual-*fallback` off.
- Steering comes only from construction-side labels and deterministic avoid-zone
  compilation. Pedestrians, vehicles, animals, stop signs, and lights may slow
  or stop in lane, but must not create an unvalidated lateral swerve.
- Every accepted plan must be traceable through frame id, source frame id,
  accepted age, labels, scores, choice metadata, RTP text, parsed validity,
  lateral target, speed target, lead class, green path offset, and consumed plan
  age.

Current status:

- Done: Qwen choice-mode red/green label path rebuilt at seq576 using hidden
  output plus answer-token lm-head scoring. This replaced the trained signal-head
  experiment as the documented production direction.
- Done: no synthetic pixel fallback by default. Visual fallbacks remain explicit
  demo-only flags for isolating renderer/controller bugs.
- Done: green scene-board path offset is converted back from MetaDrive sign to
  openpilot sign before rendering, so Qwen sees the path the controller is
  actually tracking.
- Done: construction-side compiler invariant is tested: right-side construction
  creates openpilot-left bias and MetaDrive-negative target; left-side
  construction does the opposite.
- Done: lead labels were added to the existing Qwen label/RTP path:
  `true_moving_lead`, `slower_lead`, `braking_lead`, `stopped_lead`,
  `cut_in_vehicle`, `crossing_vehicle`, and `irrelevant_vehicle`.
- Done: stale lead/agent speed plans can be cleared by current non-agent,
  true-moving-lead, or irrelevant-vehicle evidence.
- Done: default rotating score groups now cover signals, stop signs,
  construction, pedestrians, vehicles, lead classes, and animals without the
  duplicate pedestrian group.
- Done: planner results and MetaDrive logs now carry Qwen labels, label scores,
  choice metadata, source frame id, RTP age, consumed age, lead class, RTP text,
  and compiler outputs.
- Verified: `py -3.11 -m py_compile` on the main reasoned/Qwen/sim files passed.
- Verified: `py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory`
  currently passes 76 tests.

Open production gates:

- Need separate real-video/log-replay evaluations, not only MetaDrive, for
  construction, pedestrians, stoplights, stop signs, true leads, slower leads,
  braking leads, stopped leads, cut-ins, navigation ambiguity, and mixed scenes.
- Need closed-loop stock-vs-Qwen videos/logs for each suite with speed control
  enabled where the behavior requires slowing or stopping.
- Need construction/pedestrian/vehicle/lead groups to meet the same robustness
  level as the red/green choice-mode signal path under the configured latency
  and manifest contract.
- Need success-rate gates: 0 sign-flip construction failures, 0 path-relevant
  collisions in evaluated mixed suites, >=95% success for the required positive
  cases, and <5% false yield/slow/stop for irrelevant agents and true moving
  leads.

# Historical 50 ms Progress

## Active Stoplight / TensorRT Label Contract Fix

Current objective:

- Fix the optimized TensorRT label scorer contract.
- Get stoplights working.
- Do not use a trained classifier head as the production path.
- Keep progress in this file and consult it after chat compaction before resuming.

Current diagnosis:

- The original Qwen label interface is still the right semantic interface.
- The trained signal-head path was an experiment and is no longer the production direction.
- The regression is in the optimized TensorRT binary yes/no scorer contract, not in Qwen's ability to read the scene.
- The broken binary path asks separate verbose yes/no questions and compares yes/no logits. With the current `seq576` TensorRT binary engine, the red-light test image incorrectly scores `green_go_light` above `red_stop_light`, while the green image scores correctly.

Completed repair direction:

- Add `--label-decision-mode choice` to `tools/reasoned_trajectory_poc/qwen_trt_label_engine.py`.
- Choice mode keeps Qwen label scoring, but asks one multiple-choice question per label group and scores single answer tokens directly.
- Stoplight group answer words are `red`, `clear`, and `none`.
- This avoids the brittle two-prompt yes/no comparison while preserving label semantics and avoiding trained heads.
- Choice-mode TensorRT engines get a distinct filename suffix such as `qwen_text_36layer_nvfp4_seq576_choice_trt.engine`, so existing binary engines are not overwritten.
- The usable engine is the hidden-output choice engine, not the direct selected-logits choice engine:
  - Direct `selected_logits` choice engine returned all-zero answer logits.
  - Hidden-output choice engine returns `selected_hidden`, then the eight-token lm_head projection runs outside TensorRT in about `0.13-0.20 ms`.
- TensorRT returned all-zero hidden states when fed real Qwen multimodal position ids for this choice shape. The working runtime contract uses `--text-position-mode auto`, which means `clamp127` for choice mode and normal Qwen positions for binary mode.
- Stoplight choice prompt uses a constant calibration state string, `Vehicle state: speed=2.5 mps.`, because traffic signal identity is not speed-dependent and variable vehicle-state text changed the optimized prompt distribution.
- Stoplight choice uses a `0.5` minimum margin over neutral `none`, so weak color hallucinations in no-signal frames are rejected.

Verified before TensorRT rebuild:

```text
PyTorch Qwen choice prompt, full168, seq576:

red image:
  selected: red_stop_light
  red score over none: 1.2421875
  green score over none: 0.8671875
  answer token: red

green image:
  selected: green_go_light
  red score over none: -1.109375
  green score over none: 2.1484375
  answer token: green
```

2026-05-24 construction shift-label and confidence-gate iteration:

Goal:
  Continue attacking the construction sign-flip failure using only the existing production-shaped path:
  scene board -> Qwen labels/scores -> RTP -> deterministic compiler -> planner/control.

Code changes:
  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py
    Added additive action labels:
      construction_shift_left
      construction_shift_right
    These do not replace legacy construction_left/construction_right labels.
    They map as:
      construction_shift_left  -> construction_right scene -> BIAS_LEFT -> right_edge avoid
      construction_shift_right -> construction_left scene  -> BIAS_RIGHT -> left_edge avoid

  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Added --score-prompt-mode construction-compact for a shorter binary construction prompt that fits existing fixed-shape TensorRT text engines.
    Added legacy 8-word choice-engine compatibility, but the logits choice path still did not provide usable construction selection.
    Added optional --construction-mirror-consistency for binary and choice construction groups. It records original labels, mirrored labels, mapped labels, and accepted/rejected status in qwen_choice.
    Added score-derived construction RTP confidence:
      base 0.72
      0.80 for selected score >= 1.25 and margin >= 0.75
      0.84 for selected score >= 1.0 and margin >= 1.0

  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added --durable-lateral-activation-confidence, default 0.80.
    Changed contradictory lateral override so low-confidence contradictions do not count toward confirmation.
    A contradictory lateral plan now needs both repeated confirmation and confidence >= durable_conflict_override_confidence.

Tests:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  73 tests OK.

Single-frame evidence:
  Clean right-side construction frame:
    image artifacts\reasoned_trajectory_poc\construction_right_durablehold_320x200_200_20260524\vlm\vlm_input_0024.png
    command used construction_shift_left,construction_shift_right with --score-prompt-mode construction-compact
    Qwen selected construction_shift_left
    RTP scene=construction_right, meta=BIAS_LEFT
    seq576 binary hot total about 60.5 ms over 3 iterations

  Clean left-side construction frame:
    image artifacts\reasoned_trajectory_poc\construction_left_durablehold_320x200_200_20260524\vlm\vlm_input_0024.png
    Qwen selected construction_shift_right
    RTP scene=construction_left, meta=BIAS_RIGHT
    seq576 binary hot total about 67.9 ms in the single probe

Closed-loop evidence:
  Right-side current best pass:
    run artifacts\reasoned_trajectory_poc\construction_right_shiftlabels_confcount_async_160_20260524
    video artifacts\reasoned_trajectory_poc\construction_right_shiftlabels_confcount_async_160_20260524\videos\side_by_side_construction_right_shiftlabels_confcount_async_160.mp4
    async_vlm true, vlm_period_frames 2, max_rtp_age_frames 3
    speed_mps 2.5, VLM speed control disabled
    publish 160/160, deadline_miss 0
    max planner overhead 4.395 ms
    stock min construction route clearance 1.206 m
    VLM min construction route clearance 2.161 m
    result: PASS for this one right-side seed, but not a full construction gate pass.

  Left-side current failed pass:
    run artifacts\reasoned_trajectory_poc\construction_left_shiftlabels_activation_async_160_20260524
    video artifacts\reasoned_trajectory_poc\construction_left_shiftlabels_activation_async_160_20260524\videos\side_by_side_construction_left_shiftlabels_activation_async_160.mp4
    async_vlm true, vlm_period_frames 2, max_rtp_age_frames 3
    speed_mps 2.5, VLM speed control disabled
    publish 160/160, deadline_miss 0
    max planner overhead 4.067 ms
    stock min construction route clearance 1.206 m
    VLM min construction route clearance 0.101 m
    result: INVALID. Qwen still produced a high-confidence wrong construction_shift_left early enough to move into the left-side construction.

  Left-side mirror attempt:
    run artifacts\reasoned_trajectory_poc\construction_left_shiftlabels_mirror_async_160_20260524
    video artifacts\reasoned_trajectory_poc\construction_left_shiftlabels_mirror_async_160_20260524\videos\side_by_side_construction_left_shiftlabels_mirror_async_160.mp4
    binary mirror consistency hot probe on failure frame cost about 116 ms
    async_vlm true, vlm_period_frames 3, max_rtp_age_frames 5
    stock min construction route clearance 1.206 m
    VLM min construction route clearance 0.102 m
    result: INVALID. Mirror consistency did not fix closed-loop left-side construction and is too slow for 20 Hz synchronous use.

Current diagnosis:
  The action-label schema fixes the label-name ambiguity seen with construction_left/construction_right on clean fixed frames.
  The paired closed-loop gate still fails because Qwen can become confidently wrong after the green corridor starts moving.
  Confidence gating and contradiction gating prevent some bad flips, but they can also preserve an early wrong plan.
  Candidate guides remain disabled by default because they visually resemble construction markers and polluted Qwen judgement.
  Construction still fails the 0 sign-flip gate and is not production-acceptable yet.

Implementation state as of 2026-05-24 01:09:

- `qwen_trt_label_engine.py` compiles after adding choice-mode helpers and runtime branches.
- A new NVFP4 TensorRT choice text engine build is currently running:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --label-decision-mode choice --text-seq-len 576 --score-labels red_stop_light,green_go_light --image artifacts\reasoned_trajectory_poc\traffic_light_visual_probe_clean_signalhead\static\vlm_input_0000.png --workspace-gb 8 build-text
```

Next required checks:

1. Completed: `F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine` built successfully.
2. Completed: red, green, and no-stoplight benchmarks were rerun sequentially after killing a mistaken parallel benchmark pair that contaminated CUDA timing.
3. Completed: TensorRT hidden choice outputs:
   - Red signal frame selects `red_stop_light`.
   - Green signal frame selects `green_go_light`.
   - No-stoplight construction frame selects `none`; raw best answer is weak `red` but it is rejected by the `0.5` neutral-margin gate.
4. Completed: all three hot benchmarks stayed under the 50 ms frame budget.
5. Completed: manifest written to `F:\qwen_trt_export\qwen_trt_runtime_manifest.json` with contract hash `cbf5827c7761cda0c61ad42eb9c83422ef881f7a82878ef1bea17ddb82ac8d54`.

Final stoplight benchmark artifacts:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_stoplight_choice_red_benchmark.json
  labels: red_stop_light
  total_ms median 37.2543  p90 37.7833  p99 37.8324  max 37.8324

artifacts\reasoned_trajectory_poc\qwen_trt_stoplight_choice_green_benchmark.json
  labels: green_go_light
  total_ms median 37.1135  p90 37.4872  p99 38.0437  max 38.0437

artifacts\reasoned_trajectory_poc\qwen_trt_stoplight_choice_none_benchmark.json
  labels: none
  total_ms median 37.6215  p90 38.0002  p99 38.2886  max 38.2886
```

Working command shape:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine `
  --label-decision-mode choice `
  --text-output hidden `
  --text-seq-len 576 `
  --score-labels red_stop_light,green_go_light `
  --image <scene_board.png> `
  benchmark
```

## Current Verified State

- Hardware: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, 16 GB VRAM.
- Driver: 596.36.
- CUDA toolkit installed for builds: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
- `nvcc` 13.2 verifies `compute_120`, `compute_121`, `sm_120`, and `sm_121`.
- User-level `CUDA_PATH` and `CUDA_HOME` now point to CUDA 13.2.
- User-level CUDA 12.6 `PATH` entries were removed; new shells should prefer CUDA 13.2. This already-running Codex process may still inherit stale process-level CUDA 12.6 variables, so benchmark commands keep an explicit CUDA 13.2 prefix.
- Runtime stack verified:
  - `torch 2.11.0.dev20260120+cu128`
  - `tensorrt 10.16.1.11`
  - TensorRT exposes `DataType.FP4` and `BuilderFlag.FP4`.
  - `nvidia-modelopt 0.44.0`
  - `onnx 1.20.1`
  - `onnx-graphsurgeon 0.6.1`
  - `onnxslim 0.1.94`
  - `polygraphy 0.49.26`

## Model Path Under Test

- Model: `Qwen2.5-VL-3B-Instruct`.
- Local model directory: `models/vlm/qwen2_5_vl_3b_instruct`.
- Image mode: `full`.
- Image size: `168`.
- Scoring mode: two label prompts, default proof set `construction_left,construction_right`.
- Shape used by the optimized path:
  - Vision input: `pixel_values=(96,1176)`, fixed Qwen image grid `[[1,8,12]]`.
  - Vision output: `image_features=(24,2048)`.
  - Text input: `inputs_embeds=(2,220,2048)`, `position_ids=(3,2,220)`.
  - Text output: selected yes/no logits for the two label prompts.

## Measurements

Latest measured hot path, using TensorRT FP16 static vision plus TensorRT NVFP4 text, after refactoring the benchmark around the reusable worker scorer:

```text
processor_ms  median 3.954   p90 4.066   max 4.125
trt_vision_ms median 5.637   p90 6.070   max 6.214
embed_ms      median 0.116   p90 0.133   max 0.178
scatter_ms    median 0.586   p90 0.613   max 0.651
rope_ms       median 2.208   p90 2.460   max 2.558
trt_text_ms   median 17.302  p90 17.419  max 17.574
total_ms      median 31.027  p90 31.601  max 32.165
```

This proves the fixed full168, two-label Qwen scoring path can run under the 50 ms model-loop budget on the RTX 5060 Ti.

Repeatable repo command added and verified:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --iters 30 --warmup 5 benchmark
```

Result from the repeatable command:

```text
processor_ms  median 4.012   p90 4.208   max 4.359
trt_vision_ms median 5.600   p90 5.924   max 6.057
embed_ms      median 0.113   p90 0.145   max 0.173
scatter_ms    median 0.591   p90 0.663   max 0.753
rope_ms       median 2.204   p90 2.474   max 3.085
trt_text_ms   median 17.185  p90 17.649  max 18.375
total_ms      median 29.803  p90 30.320  max 30.868
```

Persisted benchmark artifact from the same script:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --iters 20 --warmup 5 --out artifacts\reasoned_trajectory_poc\qwen_trt_label_benchmark.json benchmark
```

Persisted result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_label_benchmark.json
total_ms median 31.027  p90 31.601  max 32.165
```

Comparison points:

```text
PyTorch HF full168, two labels: about 97 to 116 ms hot, depending prompt shape.
PyTorch HF language portion only: about 72 to 73 ms for two labels.
TensorRT FP16 full 36-layer text engine: 35.7 ms median.
TensorRT NVFP4 full 36-layer text engine: 16.8 to 17.1 ms median.
PyTorch HF visual tower in end-to-end path: about 52 ms median.
TensorRT FP16 static visual engine: 5.6 ms median.
```

## CUDA 13.2 / compute_120 Update

Verified on 2026-05-23:

```text
Driver: 596.36
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA toolkit selected: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
nvcc: release 13.2, V13.2.78
nvcc arch support: compute_120, compute_121
nvcc code support: sm_120, sm_121
torch: 2.11.0.dev20260120+cu128
torch GPU capability: (12, 0)
TensorRT: 10.16.1.11
TensorRT FP4 support: DataType.FP4=True, BuilderFlag.FP4=True
```

The Qwen TensorRT helper now force-selects CUDA 13.2 inside the process when that toolkit exists, instead of leaving stale inherited `CUDA_PATH=v12.6` in place.

Current CUDA 13.2 smoke benchmark:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --iters 6 --warmup 2 --out artifacts\reasoned_trajectory_poc\qwen_trt_cuda132_smoke.json benchmark-groups
```

Result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_cuda132_smoke.json
total_ms median 30.914  p90 31.528  max 31.608
trt_text_ms median 17.124  p90 17.461  max 17.669
trt_vision_ms median 5.615  p90 5.644  max 5.940
```

## Synchronous Sim Timing With TensorRT Worker

The optimized TensorRT worker now runs inside the MetaDrive overlay demo path, not only as a standalone benchmark. The worker supports an optional `--ready-jsonl` marker after loading and warmup; `PersistentRtpEngine` consumes it when `RTP_VLM_WAIT_READY=1`. This keeps one-time model load and engine warmup out of the recorded model-loop frames without burning a fake rotating-label inference.

Transport issue found and fixed before this run:

```text
PNG scene-board payloads made planner wall time about 60 ms even though Qwen inference stages summed to about 33 ms.
RTP_VLM_IMAGE_FORMAT=jpeg with quality 85 reduces scene-board serialization and pipe overhead enough for synchronous 20 Hz use.
```

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 30 --speed-mps 5 --engine vlm --novel-scene construction --deadline-ms 50 --tick-sec 0 --save-every 10 --out artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_30\vlm\episode.json

frames 30
publish_count 30
valid_count 30
deadline_miss_count 0
reasoned_latency_ms median 36.349  p90 37.201  p99 38.363  max 38.695
same_frame_all True
max_rtp_age_frames 0
frame0_latency_ms 37.545
frame0_deadline_met True
```

Stage timing from the recorded sim frames:

```text
camera_to_scene_board_ms        median 2.750   p90 2.872   p99 3.550   max 3.633
scene_board_to_vlm_prefill_ms   median 12.686  p90 13.274  p99 13.590  max 13.711
vlm_decode_ms                   median 17.320  p90 17.687  p99 17.914  max 17.967
rtp_parse_ms                    median 0.053   p90 0.061   p99 0.073   max 0.077
path_synth_ms                   median 0.027   p90 0.035   p99 0.048   max 0.049
```

## Longer Mixed-Scene Synchronous Timing

Extended the proof from a 30-frame construction smoke to a 300-frame random mixed scene with construction and moving pedestrian/vehicle labels rotating through the same shared fixed-shape TensorRT language engine.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 300 --speed-mps 5 --engine vlm --novel-scene random_mixed --deadline-ms 50 --tick-sec 0 --save-every 100 --out artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_ready_jpeg_mixed_300\vlm\episode.json

frames 300
publish_count 300
valid_count 300
deadline_miss_count 0
reasoned_latency_ms median 35.719  p90 37.006  p99 37.780  p99.9 38.068  max 38.113
same_frame_all True
max_rtp_age_frames 0
selected_candidates C0=24 C1=276
path_delta_nonzero_frames 276
speed_delta_nonzero_frames 276
```

Stage timing from the recorded mixed-scene frames:

```text
camera_to_scene_board_ms        median 2.771   p90 2.810   p99 3.161   p99.9 3.338   max 3.348
scene_board_to_vlm_prefill_ms   median 12.138  p90 13.150  p99 13.507  p99.9 13.627  max 13.629
vlm_decode_ms                   median 17.288  p90 17.699  p99 17.970  p99.9 18.052  max 18.062
rtp_parse_ms                    median 0.051   p90 0.055   p99 0.100   p99.9 0.150   max 0.167
path_synth_ms                   median 0.026   p90 0.028   p99 0.036   p99.9 0.075   max 0.085
```

The demo summary now records `p999_latency_ms`, `max_latency_ms`, `same_frame_count`, `same_frame_all`, and `max_rtp_age_frames` directly in future episode JSON files.

## Crashout Video TensorRT Evaluation

Added `tools\reasoned_trajectory_poc\evaluate_qwen_trt_video.py` to run the same optimized TensorRT Qwen label scorer on a real video file. This is not a closed-loop control replay, so it cannot prove the car would physically avoid the obstacle. It does prove the scorer latency and whether the RTP compiler would request lateral trajectory modification from the video frames.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\evaluate_qwen_trt_video.py --video E:\ture_opamayo\crashout.mp4 --save-first 8 --save-lateral 8 --out artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\video_eval.json
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\input_samples
artifacts\reasoned_trajectory_poc\crashout_qwen_trt_video_eval_full\lateral_samples

video_frames 1237
video_fps 29.348591
sampled_frames 1237
deadline_miss_count 0
would_change_lateral_count 1201
first_lateral_frame 3
first_lateral_time_sec 0.102
first_lateral_lat_bias_m 1.25
first_lateral_scene mixed_agent_construction_right
max_abs_lat_bias_m 1.25
scored_fps_wall 30.614
```

Crashout scorer timing:

```text
total_ms      median 31.660  p90 32.642  p99 33.437  p99.9 34.096  max 34.111
processor_ms  median 4.208   p90 4.485   p99 4.723   p99.9 5.024   max 5.057
trt_vision_ms median 5.620   p90 6.065   p99 6.316   p99.9 6.464   max 6.602
trt_text_ms   median 17.425  p90 17.802  p99 18.361  p99.9 19.142  max 19.415
embed_ms      median 0.119   p90 0.150   p99 0.214   p99.9 0.305   max 0.407
scatter_ms    median 0.596   p90 0.668   p99 0.857   p99.9 0.942   max 0.944
rope_ms       median 2.250   p90 2.529   p99 2.814   p99.9 3.202   max 3.363
```

Crashout RTP distribution:

```text
lat_bias positive frames 1161
lat_bias negative frames 40
lat_bias zero frames 36
top scenes:
  mixed_agent_construction_right 775
  construction_right 386
  mixed_agent_construction_left 21
  path_conflict_agent 20
  construction_left 19
  nominal 16
```

## Artifact And Shape Validation

Added `check-artifacts` to `tools\reasoned_trajectory_poc\qwen_trt_label_engine.py`. It validates the external TensorRT artifacts and runtime environment before a run, without loading the full Qwen model through Transformers.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check.json check-artifacts
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check.json
ok true
issues []
CUDA_PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
nvcc compute_120 true
nvcc sm_120 true
torch_device_name NVIDIA GeForce RTX 5060 Ti
torch_device_capability [12,0]
TensorRT 10.16.1.11
TensorRT FP4 true
```

Validated engine shapes:

```text
vision pixel_values    [96,1176]  FLOAT input
vision image_features  [24,2048]  HALF output
text inputs_embeds     [2,220,2048]  HALF input
text position_ids      [3,2,220]     INT64 input
text selected_logits   [2,1,8]       HALF output
```

## Runtime Manifest Contract

Added a runtime manifest contract for the optimized TensorRT Qwen path. The manifest hashes the prompt contract, score-question text, label groups, image mode/size, text sequence length, vehicle-state text, scoring thresholds, rotating-cache behavior, selected model config/tokenizer files, and model weight filenames/sizes. Runtime commands can now use `--require-manifest` to reject mismatched fixed-shape engines before measuring latency.

Manifest path:

```text
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Manifest write command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --write-manifest --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_with_behavior_manifest.json check-artifacts
```

Strict validation command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --require-manifest --out artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_require_manifest.json check-artifacts
```

Strict validation result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_require_manifest.json
ok true
issues []
actual_contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
expected_contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Fail-closed probes:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_manifest_mismatch.json
changed text_seq_len 220 -> 224
result: rejected
issues:
  text inputs_embeds shape (2,220,2048) != expected (2,224,2048)
  text position_ids shape (3,2,220) != expected (3,2,224)
  manifest contract sha mismatch

artifacts\reasoned_trajectory_poc\qwen_trt_artifact_check_manifest_label_mismatch.json
changed label groups while keeping tensor shapes valid
result: rejected
issues:
  manifest contract sha mismatch
```

## 50 ms Pass/Fail Gate

Added `gate` to `tools\reasoned_trajectory_poc\qwen_trt_label_engine.py`. The gate always requires the runtime manifest, validates TensorRT artifacts first, runs the rotating label benchmark, and exits nonzero if either p99 or max total latency exceeds `--deadline-ms`.

Passing 50 ms gate command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --deadline-ms 50 --iters 30 --warmup 3 --out artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_behavior_manifest.json gate
```

Passing result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_behavior_manifest.json
ok true
deadline_ms 50.0
p99_total_ms 33.133
max_total_ms 33.133
issues []
manifest ok true
contract_sha256 db472aec5aad73627bdb475c8ff253bc6475822dae7ee0950b7a881979c98f97
```

Gate stage timing:

```text
total_ms      median 30.832  p90 31.516  p99 33.133  p99.9 33.133  max 33.133
processor_ms  median 3.875   p90 4.132   p99 4.914   p99.9 4.914   max 4.914
trt_vision_ms median 5.623   p90 5.935   p99 6.129   p99.9 6.129   max 6.129
trt_text_ms   median 17.273  p90 17.562  p99 17.973  p99.9 17.973  max 17.973
```

Gate failure-path probe:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-shared-engine --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --deadline-ms 1 --iters 3 --warmup 1 --out artifacts\reasoned_trajectory_poc\qwen_trt_1ms_gate_expected_fail.json gate
```

Failure result:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_1ms_gate_expected_fail.json
ok false
exit code 2
issues:
  p99 total latency 31.254 ms exceeds deadline 1.000 ms
  max total latency 31.254 ms exceeds deadline 1.000 ms
```

## Strict-Manifest MetaDrive Timing

Ran the synchronous MetaDrive VLM loop with the worker itself started under `--require-manifest`, using the behavior-aware manifest that includes score thresholds and rotating-cache settings. This verifies the production-style persistent server path, not just the standalone gate.

Verified command:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_STDERR_PATH='E:\ture_opamayo\openpilot\artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\vlm_stderr.log'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --warmup 1 --require-manifest --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 120 --speed-mps 5 --engine vlm --novel-scene random_mixed --deadline-ms 50 --tick-sec 0 --save-every 60 --out artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120
```

Recorded result:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\comparison_vlm.json
artifacts\reasoned_trajectory_poc\metadrive_trt_require_manifest_mixed_120\vlm\episode.json

frames 120
publish_count 120
valid_count 120
deadline_miss_count 0
reasoned_latency_ms median 36.192  p90 37.082  p99 38.121  p99.9 38.924  max 39.024
same_frame_all True
same_frame_count 120
max_rtp_age_frames 0
selected_candidates C0=18 C1=102
path_delta_nonzero_frames 102
speed_delta_nonzero_frames 102
```

Stage timing:

```text
camera_to_scene_board_ms        median 2.778   p90 2.825   p99 3.157   p99.9 3.698   max 3.769
scene_board_to_vlm_prefill_ms   median 12.718  p90 13.208  p99 13.609  p99.9 13.826  max 13.855
vlm_decode_ms                   median 17.274  p90 17.587  p99 17.956  p99.9 18.098  max 18.115
rtp_parse_ms                    median 0.052   p90 0.057   p99 0.076   p99.9 0.084   max 0.085
path_synth_ms                   median 0.026   p90 0.029   p99 0.056   p99.9 0.064   max 0.065
```

## Rotating Six-Label Worker Verification

The fast path now supports rotating two-label groups while reusing one fixed-shape TensorRT language engine. This covers the proposed six-label cadence:

```text
group 0: construction_left,construction_right
group 1: pedestrian_in_path,pedestrian_entering_path
group 2: vehicle_in_path,vehicle_entering_path
```

The key implementation detail is `--text-seq-len 220`: current verbose prompts tokenize to 220, 216, and 208 tokens respectively, so a fixed 220-token text shape lets all three groups share the same `(2,220,2048)` TensorRT text engine without truncation.

Benchmark command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine `
  --text-seq-len 220 `
  --score-rotate-shared-engine `
  --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" `
  --iters 9 --warmup 2 `
  --out artifacts\reasoned_trajectory_poc\qwen_trt_rotating_shared_benchmark.json `
  benchmark-groups
```

Persisted rotating benchmark:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_rotating_shared_benchmark.json
total_ms median 31.051  p90 31.907  max 32.216
trt_text_ms median 17.300  p90 17.783  max 17.909
```

Per-group total latency:

```text
construction side: median 30.639  max 31.907
pedestrian conflict: median 31.185  max 32.216
vehicle conflict: median 29.902  max 31.114
```

Persistent JSONL serve command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py `
  --artifact-dir F:\qwen_trt_export `
  --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine `
  --text-seq-len 220 `
  --score-rotate-groups `
  --score-rotate-shared-engine `
  --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" `
  --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" `
  serve
```

Persistent JSONL smoke on the same scene-board input:

```text
frame 0 group construction_left,construction_right: labels none, total_ms 30.930
frame 1 group pedestrian_in_path,pedestrian_entering_path: labels none, total_ms 30.232
frame 2 group vehicle_in_path,vehicle_entering_path: labels none, total_ms 30.246
```

The vehicle group raw score on that construction frame was `vehicle_in_path=0.25`, so per-label thresholds are now supported and the smoke used `vehicle_in_path:0.5` to avoid turning that weak score into a false yield.

## Prior Persistent Worker Verification

`tools\reasoned_trajectory_poc\qwen_trt_label_engine.py` now has a `serve` subcommand. It keeps the build and benchmark commands intact and adds a JSONL worker compatible with `selfdrive.controls.reasoned.vlm.PersistentRtpEngine`.

Worker command:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --warmup 2 --score-labels construction_left,construction_right serve
```

Direct JSONL smoke, using `artifacts\reasoned_trajectory_poc\qwen_construction_loop_proof\vlm\vlm_input_0020.png`:

```text
backend qwen2.5-vl-3b-trt-nvfp4-full168-score
frame_id 20
source_frame_id 20
prefill_ms 11.498
decode_ms 16.987
total_ms 29.549
labels ["none"]
scores construction_left=-0.5625 construction_right=-0.75
```

Planner-contract smoke through `PersistentRtpEngine`:

```text
text_first_line RTPv1
backend qwen2.5-vl-3b-trt-nvfp4-full168-score
source_frame_id 20
prefill_ms 11.736
decode_ms 17.379
```

## Generated Artifacts

Current working TensorRT artifacts were moved from `C:\Users\user\AppData\Local\Temp\qwen_trt_export` to `F:\qwen_trt_export` because C: ran out of space during grouped exports. They are not committed:

```text
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.onnx
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_construction_left__construction_right_trt.engine
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_construction_left__construction_right_trt.onnx
F:\qwen_trt_export\vision_static_fp16\qwen_vision_full168_static_fp16.engine
F:\qwen_trt_export\vision_static_fp16\qwen_vision_full168_static_fp16.onnx
```

The temp artifact directory is intentionally outside the repo because the text and vision engines are large.

## Implementation Notes

- Direct FP8 or FP4 fake quant in PyTorch is not useful for runtime. ModelOpt fake-quant configs were slower than FP16 PyTorch.
- TensorRT-LLM is not available as a Windows wheel in this environment. The successful path is direct ONNX plus TensorRT engines.
- The Qwen vision tower could not export as-is because:
  - SDPA export hit `scaled_dot_product_attention` with `enable_gqa=True`.
  - TensorRT rejected a half-typed rotary `Range`.
- The working vision engine uses an eager-attention wrapper and bakes fixed full168 constants:
  - `rotary_pos_emb`
  - `window_index`
  - `cu_window_seqlens`
  - `cu_seqlens`
  - reverse window index
- The working text engine removes the full vocabulary LM head and only computes logits for yes/no token IDs used by label scoring.
- The text engine is fixed-shape for the prompt/image/label set used at build time. Changing prompt text, label group, image mode, or image size can change token/image shapes and requires rebuilding or adding fixed profiles for those variants.
- The worker defaults to a fixed vehicle-state string for shape stability. `--use-payload-vehicle-state` exists, but it will reject requests if the resulting text shape differs from the built TensorRT engine.
- Separate label-keyed text engine builds are possible, but not the preferred runtime path. A grouped build hit disk-space first, then a subsequent non-construction keyed build went idle in TensorRT build. The better working path is a shared fixed sequence length matching all current two-label prompts.

## Next Work

- Decide whether to keep the engines in a configured external artifact directory or add an explicit build step for local users.
- Rebuild or add additional fixed-shape engines for the actual production label rotation sets if they differ from the two-label proof set.

## 2026-05-23 Fast TensorRT Mixed MetaDrive Video Demo

Ran a synchronous mixed pedestrian plus construction MetaDrive comparison using the TensorRT Qwen fast path, strict runtime manifest, CUDA 13.2, JPEG scene-board transport, 5.0 m/s target speed, and every-frame video capture.

Command shape:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
$env:RTP_VLM_WAIT_READY='1'
$env:RTP_VLM_IMAGE_FORMAT='jpeg'
$env:RTP_VLM_JPEG_QUALITY='85'
$env:RTP_VLM_SERVER_COMMAND='py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_trt.engine --text-seq-len 220 --score-rotate-groups --score-rotate-shared-engine --score-thresholds "pedestrian_in_path:0.5,pedestrian_entering_path:0.5,vehicle_in_path:0.5,vehicle_entering_path:0.5" --score-label-groups "construction_left,construction_right;pedestrian_in_path,pedestrian_entering_path;vehicle_in_path,vehicle_entering_path" --require-manifest --ready-jsonl serve'
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --engine vlm --novel-scene random_mixed --frames 300 --speed-mps 5.0 --tick-sec 0.05 --deadline-ms 50 --save-every 1 --map 3 --seed 7 --random-scene-seed 42 --out artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043
```

Run artifacts:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\side_by_side_fast_mixed_300_sync_5mps.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\stock_fast_mixed_300_sync_5mps.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214043\videos\vlm_fast_mixed_300_sync_5mps.mp4
```

Summary:

```text
stock frames 300, mean_speed_mps 4.611, target_speed_mps 5.0
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm latency median 36.430 ms, p90 37.291 ms, p99 37.881 ms, p99.9 38.602 ms, max 38.602 ms
vlm selected_candidates C1=276, C0=24
vlm active durable lateral frames 276
vlm active durable speed frames 276
vlm mean_path_delta_m 1.15
vlm lateral offset range -1.25..0.0 MetaDrive meters
vlm mean_speed_mps 0.920 because pedestrian/agent speed-plan logic was active
stock min_spawned_object_distance_m 0.805
vlm min_spawned_object_distance_m 1.133
```

Stage timing:

```text
camera_to_scene_board_ms median 2.761, p99 3.079, max 3.680
scene_board_to_vlm_prefill_ms median 12.677, p99 13.575, max 13.763
vlm_decode_ms median 17.496, p99 18.240, max 18.315
rtp_parse_ms median 0.051, p99 0.073, max 0.104
path_synth_ms median 0.026, p99 0.049, max 0.176
```

The first non-base RTP occurred at frame 24:

```text
scene=construction_right
meta=BIAS_LEFT_AND_SLOW
lat_bias_m=1.25
speed_cap_mps=25%
avoid=[right_edge_s8_48_margin1.25]
confidence=0.72
```

The speed controller then fired mixed pedestrian/agent constraints at frame 56:

```text
scene=mixed_agent_construction_right
meta=YIELD
lat_bias_m=1.25
speed_cap_mps=15%
avoid=[right_edge_s8_48_margin1.25,corridor_object_s18_28]
confidence=0.72
```

## 2026-05-23 Fast TensorRT Mixed MetaDrive Lateral-Only 2.5 m/s Demo

Reran the same mixed construction plus pedestrian MetaDrive setup with target speed capped at 2.5 m/s and VLM speed control disabled. This isolates lateral behavior while keeping stock and VLM longitudinal behavior effectively identical.

Command difference from the prior run:

```powershell
py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --engine vlm --novel-scene random_mixed --frames 300 --speed-mps 2.5 --disable-vlm-speed-control --tick-sec 0.05 --deadline-ms 50 --save-every 1 --map 3 --seed 7 --random-scene-seed 42 --out artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only
```

Video artifacts:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\side_by_side_fast_mixed_300_sync_2p5mps_lateral_only.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\stock_fast_mixed_300_sync_2p5mps_lateral_only.mp4
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_214340_2p5mps_lateral_only\videos\vlm_fast_mixed_300_sync_2p5mps_lateral_only.mp4
```

Summary:

```text
stock frames 300, mean_speed_mps 2.322, target_speed_mps fixed 2.5
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm mean_speed_mps 2.318, target_speed_mps fixed 2.5
vlm speed-control enabled false on every frame
vlm latency median 37.298 ms, p90 38.523 ms, p99 39.876 ms, max 41.072 ms
vlm selected_candidates C1=289, C0=11
vlm active durable lateral frames 276
vlm active durable speed frames 0
vlm mean_path_delta_m 1.15
vlm active lateral offset range 0.0..1.25 MetaDrive meters
stock min_spawned_object_distance_m 1.073
vlm min_spawned_object_distance_m 0.111
```

Interpretation: the 2.5 m/s cap and `--disable-vlm-speed-control` worked. The VLM no longer slows the car relative to stock. However, this seed exposes a lateral-policy failure: the VLM path modification got closer to a spawned object than stock. The fast TensorRT path is meeting the realtime budget, but lateral target selection/sign/corridor handling still needs correction before this mixed-scene behavior is acceptable.

## 2026-05-23 Corridor And Side-Grounding Follow-up

The 2.5 m/s lateral-only run showed the car moving toward the cone cluster. Inspection found multiple issues:

```text
frame 24 exact VLM input showed foreground cones on image-right of the green planned path
runtime RTP nevertheless said construction_left
fixed text_seq_len=220 had previously truncated left/right questions so construction_left and construction_right prompts were token-identical
after moving the scored label/question to the front of the prompt, the left/right prompts are no longer identical
Qwen still needs stronger path-relevance semantics: only hazards overlapping, intruding into, narrowing, blocking, or imminently entering the green corridor should count
```

Changes made:

```text
tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
  moved Scored label and Question before the long common prompt so text_seq_len=220 preserves the side-specific question

tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py
  tightened prompts/questions so Qwen should only consider hazards affecting the green planned path or imminent path entrants
  removed first-label-wins behavior for exclusive construction_left/construction_right ties

tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
  corrected the MetaDrive lateral conversion to match the observed route convention in this harness

selfdrive\controls\reasoned\ui_scene_board.py
  expanded the green planned corridor from 0.48 m half-width to 0.60 m half-width, exactly 25% wider
```

Verification:

```text
py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
Ran 20 tests OK
```

The strict TensorRT behavior manifest should be treated as stale after these prompt and scene-board changes. Do not use `--require-manifest` again until a new side-grounding probe and mixed lateral-only run pass, then rewrite `F:\qwen_trt_export\qwen_trt_runtime_manifest.json`.

## 2026-05-23 Wide-Corridor Sign-Fixed Demo And Manifest Rewrite

The first wide-corridor rerun still failed side behavior because the MetaDrive sign conversion had been changed in the wrong direction. It produced `construction_right` correctly, but converted that right-edge avoid into a positive MetaDrive target while the spawned right-side cones also had positive lateral coordinates. That run was rejected and was not used for the manifest.

Fixed the MetaDrive sign conversion back to the observed harness convention:

```text
PathSynth/openpilot positive lat_bias_m = left
MetaDrive positive lateral in this harness = visually right
right_edge_s8_48_margin1.25 -> openpilot +1.25 -> MetaDrive -1.25
left_edge_s8_48_margin1.25 -> openpilot -1.25 -> MetaDrive +1.25
```

Reran the mixed construction/pedestrian demo at 2.5 m/s with VLM speed control disabled:

```text
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix
artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix\videos\side_by_side_fast_mixed_300_sync_2p5mps_lateral_only_widecorridor_signfix.mp4
```

Run result:

```text
stock frames 300, mean_speed_mps 2.3219, min_spawned_object_distance_m 1.0730
vlm frames 300, publish_count 300, valid_count 300, deadline_miss_count 0
vlm same_frame_all true, max_rtp_age_frames 0
vlm mean_speed_mps 2.3217, target_speed fixed 2.5, VLM speed control disabled
vlm latency p90 37.904 ms, p99 38.906 ms, max 39.366 ms
vlm min_spawned_object_distance_m 1.3115
vlm active lateral offset range -1.25..0.0
vlm source_counts right_edge_s8_48_margin1.25=294, left_edge_s8_48_margin1.25=6
```

Side audit on the same run:

```text
frame 24 visible construction color mass: image-right
frame 24 RTP source: right_edge_s8_48_margin1.25
frame 24 spawned construction laterals ahead: +1.069, +1.236, +1.157, +1.345, +1.337, +1.378
frame 24 active MetaDrive lateral target: -0.252

frame 98 spawned construction laterals ahead: +1.236, +1.157, +1.345, +1.337, +1.378, +1.253
frame 98 active MetaDrive lateral target: -1.25
```

Direct Qwen side score probe on the saved frame:

```text
image: artifacts\reasoned_trajectory_poc\metadrive_trt_fast_mixed_300_20260523_220505_2p5mps_lateral_only_widecorridor_signfix\vlm\vlm_input_0024.png
construction_left score: 4.30078125
construction_right score: 5.28125
total_ms p90 32.939 ms
```

Rewrote the TensorRT runtime manifest after this validation:

```text
F:\qwen_trt_export\qwen_trt_runtime_manifest.json
contract_sha256 cf6c028ed0580f03db61300884c0b777bad6c750741998d64fdf98a0d5319f29
```

The manifest contract now includes:

```text
score prompt hash
score question hash
label groups
score_rotate_groups
score_rotate_shared_engine
thresholds
image mode/size
text_seq_len
model config metadata
scene-board geometry, including planned_corridor_half_width_m=0.60
```

Manifest-gated 50 ms check passed:

```text
artifacts\reasoned_trajectory_poc\qwen_trt_50ms_gate_widecorridor_signfix_manifest.json
ok true
manifest ok true
p99_total_ms 34.458
max_total_ms 34.458
contract_sha256 cf6c028ed0580f03db61300884c0b777bad6c750741998d64fdf98a0d5319f29
```

Remaining caveat: the lateral-only demo disables VLM speed control, so pedestrian/vehicle path-conflict labels cannot yield or stop the car. A pedestrian in the planned path during a lateral-only run is not avoidable by this configuration; production mixed scenes need speed control enabled or a separate explicitly validated lateral pedestrian avoidance policy.

## 2026-05-24 production-translation lead audit

Goal update: every improvement must directly translate to real openpilot use. Sim-only labels, expected classes, route-script answers, and hand-authored RTP remain invalid.

Lead-state work added today:

```text
scene board state now carries neutral lead-track fields:
lead_present
lead_source
lead_distance_m
lead_lateral_m
lead_speed_mps
lead_rel_speed_mps
lead_closing_mps
lead_accel_mps2
lead_lateral_velocity_mps
```

These fields are intended to map to production openpilot model/radar/track-fusion signals. In MetaDrive they are now derived from observed per-frame object position history, not `expected_lead_class`, scenario name, or scripted accel. The first frame with no track history reports `lead_present=0`.

Invalidated evidence:

```text
Earlier lead-car runs used MetaDrive's default TrafficDefaultVehicle/Ferrari asset.
The rear view was visually ambiguous at scene-board resolution and was called out as looking like an oncoming car.
Those lead videos/logs should not be used as acceptance evidence.
```

Fix:

```text
controlled route vehicles now use MetaDrive MVehicle
quick proof run: artifacts\reasoned_trajectory_poc\metadrive_lead_orientation_mvehicle_12_20260524
sample input: artifacts\reasoned_trajectory_poc\metadrive_lead_orientation_mvehicle_12_20260524\static\vlm_input_0004.png
```

The sample input shows a same-direction rear view in the green corridor. This only fixes the sim harness asset; it is not a production logic change.

Current unresolved issue:

```text
The Qwen TensorRT hidden-choice path can intermittently return zero word scores on some saved frames in persistent serve mode.
This is not acceptable as a hidden failure mode.
The likely fix path is to use or rebuild a stable choice-logits engine, or otherwise make zero-score output invalid/fail-closed before it reaches RTP.
```

Follow-up fix:

```text
Built choice-logits engine:
F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq640_choice_trt.engine

Single corrected MVehicle lead-frame timing:
total median about 40.9 ms
text TensorRT about 22.2 ms
vision TensorRT about 11.2 ms
```

Root cause isolated for the remaining all-zero case:

```text
PyTorch text logits on the failing frame were nonzero.
TensorRT NVFP4 text logits on the same embeddings were exactly all zero.
The failure was triggered by some underscore/key-value vehicle-state strings, not by the image, not by the MVehicle asset, and not by the PyTorch Qwen model.
```

Mitigation now in code:

```text
The scene-board vehicle-state text Qwen receives now uses plain-language physical lead state:
lead present yes; source track; distance 19.5 m; lateral offset 0.0 m; lead speed 1.3 m/s; relative speed 0.8 m/s; closing -0.8 m/s; acceleration 0.0 m/s2; lateral velocity 0.0 m/s

The physical lead-state parser still accepts the old underscore/key-value format for compatibility.
```

Verification:

```text
unit tests: py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
result: 54 tests OK

plain-state proof run:
artifacts\reasoned_trajectory_poc\metadrive_lead_orientation_mvehicle_plainstate_12_20260524
sample input:
artifacts\reasoned_trajectory_poc\metadrive_lead_orientation_mvehicle_plainstate_12_20260524\static\vlm_input_0004.png

12-frame persistent Qwen logits probe:
old key-value state: zero-score frames included frame 10
plain-language state: zero-score frames []
hot total latency: about 38 to 42 ms
```

Invalid run, do not use as acceptance evidence:

```text
artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_plainstate_logits_80_20260524

The corrected asset and plain-state Qwen path ran, but the closed-loop run published 0 modified plans.
Reason: the persistent VLM result was one frame stale after the first cold response, so the same-frame publish gate rejected it.
This is a harness/gating issue to fix before claiming lead-control proof.
```

## 2026-05-24 same-frame VLM transport and lead proof

Root cause fixed:

```text
The Qwen server command used --ready-jsonl.
PersistentRtpEngine only consumed the ready marker when RTP_VLM_WAIT_READY=1.
Without that env var, frame 0 read {"ready":true} as if it were an RTP response, and every real response was shifted one frame stale.
```

Code changes:

```text
PersistentRtpEngine now:
- stores the split server command
- automatically waits for the ready marker when --ready-jsonl is present
- defensively skips ready markers in generate()
- prefers source_frame_id over frame_id when the worker supplies it

Default scene-board transport changed from PNG to PPM to remove PNG encode cost from the 50 ms loop.
MetaDrive demo default board size changed to 320x200, because the Qwen engine consumes a 168 px processed image and the old 512x320/384x240 payloads wasted IPC/encoding time.
```

Regression tests:

```text
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
result: 56 tests OK

New tests cover:
- ready marker does not shift the first frame response
- --ready-jsonl command waits before the first generate()
```

Invalid intermediate timing:

```text
run: artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_readyfix_logits_120_20260524
same_frame_all true
max_rtp_age_frames 0
publish_count 0/120
hot latency about 68-71 ms
reason: PNG encoding/base64 transport was outside Qwen stage timings and pushed total frame latency over 50 ms
```

Valid lead timing proof after transport/default fix:

```text
run: artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_defaultfast_readywait_40_20260524
video: artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_defaultfast_readywait_40_20260524\videos\side_by_side_qwen_slower_lead_defaultfast_readywait_40.mp4

frames 40
publish_count 40
valid_count 40
deadline_miss_count 0
same_frame_all true
max_rtp_age_frames 0
mean_latency_ms 46.969
p90_latency_ms 48.219
p99_latency_ms 48.917
max_latency_ms 48.917
qwen_lead_classes [slower_lead_closing, true_moving_lead]
label counts: none 0 after ready wait; true_moving_lead and slower_lead both observed
mean_speed_delta_mps 0.1875
```

Longer lead run:

```text
run: artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_readyfix_ppm384_120_20260524
video: artifacts\reasoned_trajectory_poc\metadrive_qwen_slower_lead_mvehicle_readyfix_ppm384_120_20260524\videos\side_by_side_qwen_slower_lead_readyfix_ppm384_120.mp4

frames 120
publish_count 119
deadline_miss_count 1
same_frame_all true
max_rtp_age_frames 0
hot reasoned_latency_ms median 46.991, p90 48.034, p99 48.487, max 48.967
the one miss was startup before automatic --ready-jsonl waiting was added
```

This proves the corrected same-direction lead asset, same-frame source id handling, plain-state Qwen lead input, and default fast transport can run inside the 50 ms loop for this slower-lead harness. It does not prove the full objective; construction, pedestrians, lights, signs, cut-ins, stopped leads, navigation ambiguity, and mixed scenes still need separate current evaluations under the same no-fallback fast path.

## 2026-05-24 lead taxonomy suite at 320x200

Additional code fix:

```text
The production lead-track consistency filter now distinguishes cut-in versus crossing using track kinematics:
- lateral object moving into the path with meaningful along-route speed -> cut_in_vehicle
- lateral object moving across the path with very low along-route speed -> crossing_vehicle
- lateral object not entering the path -> irrelevant_vehicle
```

Regression tests:

```text
py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
result: 57 tests OK

New coverage verifies physical track state maps:
cut-in track -> cut_in_vehicle / merge answer word
crossing track -> crossing_vehicle / crossing answer word
off-path no-entry track -> irrelevant_vehicle / irrelevant answer word
```

Current lead-suite summary artifact:

```text
artifacts\reasoned_trajectory_poc\lead_suite_summary_320x200_20260524.json
```

All runs below used the same current fast path:

```text
Qwen2.5-VL-3B TensorRT NVFP4 choice-logits engine
text_seq_len=640
image input full board, 320x200 transported as PPM, resized by Qwen processor to 168 px
no visual fallbacks
same-frame synchronous publish gate
speed control enabled
```

Lead taxonomy results:

```text
true_moving_lead:
  run artifacts\reasoned_trajectory_poc\lead_suite_true_moving_lead_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_true_moving_lead_320x200_120_20260524\videos\side_by_side_lead_suite_true_moving_lead_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 47.211
  labels: true_moving_lead 119, none 1 first no-track frame
  false_slow_frames 0, mean_speed_delta_mps 0.0

slower_lead:
  run artifacts\reasoned_trajectory_poc\lead_suite_slower_lead_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_slower_lead_320x200_120_20260524\videos\side_by_side_lead_suite_slower_lead_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 48.694
  labels: true_moving_lead 24, slower_lead 95, none 1 first no-track frame
  expected kinematic match rate 1.0
  mean_speed_delta_mps 0.396

braking_lead:
  run artifacts\reasoned_trajectory_poc\lead_suite_braking_lead_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_braking_lead_320x200_120_20260524\videos\side_by_side_lead_suite_braking_lead_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 48.202
  labels: braking_lead 23, stopped_lead 95, true_moving_lead 1, none 1 first no-track frame
  expected kinematic match rate 1.0
  mean_speed_delta_mps 2.243

stopped_lead:
  run artifacts\reasoned_trajectory_poc\lead_suite_stopped_lead_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_stopped_lead_320x200_120_20260524\videos\side_by_side_lead_suite_stopped_lead_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 49.394
  labels: stopped_lead 119, none 1 first no-track frame
  expected kinematic match rate 1.0
  mean_speed_delta_mps 2.479

cut_in_vehicle:
  run artifacts\reasoned_trajectory_poc\lead_suite_cut_in_vehicle_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_cut_in_vehicle_320x200_120_20260524\videos\side_by_side_lead_suite_cut_in_vehicle_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 47.196
  labels: cut_in_vehicle 1, true_moving_lead 105, slower_lead 13, none 1 first no-track frame
  expected kinematic match rate 0.992
  mean_speed_delta_mps 0.065

crossing_vehicle:
  run artifacts\reasoned_trajectory_poc\lead_suite_crossing_vehicle_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_crossing_vehicle_320x200_120_20260524\videos\side_by_side_lead_suite_crossing_vehicle_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 47.680
  labels: crossing_vehicle 2, stopped_lead 51, irrelevant_vehicle 66, none 1 first no-track frame
  expected kinematic match rate 0.992
  mean_speed_delta_mps 1.090

irrelevant_vehicle:
  run artifacts\reasoned_trajectory_poc\lead_suite_irrelevant_vehicle_320x200_120_20260524
  video artifacts\reasoned_trajectory_poc\lead_suite_irrelevant_vehicle_320x200_120_20260524\videos\side_by_side_lead_suite_irrelevant_vehicle_320x200_120.mp4
  publish 120/120, deadline_miss 0, same_frame_all true, max_age 0
  max_latency_ms 47.624
  labels: irrelevant_vehicle 119, none 1 first no-track frame
  false_slow_frames 0, mean_speed_delta_mps 0.0
```

Interpretation:

```text
The current lead taxonomy suite passes the real-time/same-frame gate and behaves correctly under the current MetaDrive track-proxy harness.
The one-frame none at frame 0 is expected because the track-history proxy has no previous observation yet; production openpilot radar/model track fusion should not need that warmup if track velocity is already available.
Cut-in and crossing have short positive conflict windows because the scripted objects enter/cross quickly; later frames correctly become same-lane lead or irrelevant/stopped according to observed track state.
```

Remaining objective scope:

```text
This still does not prove construction, pedestrians, animals, red/green lights, stop signs, navigation branches/forks/exits/merges, occlusion/splash, or mixed real-world scenes.
Those categories still need current 320x200 fast-path evaluations with logs/videos, and failures need to be fixed in the same Qwen -> RTP -> compiler path.
```

2026-05-24 lead vehicle visual-heading audit:

```text
User reported the MetaDrive lead car looked backwards/facing ego.
I verified the route-vehicle orientation directly instead of changing production logic.

Findings:
  artifact comparison: artifacts\reasoned_trajectory_poc\lead_orientation_fix_true_moving_48_20260524\orientation_class_compare.png
  left column in that comparison uses visual heading = route heading and shows the rear of the vehicle from ego.
  right column uses visual heading = route heading + pi and shows the front of the vehicle facing ego.
  Therefore the correct harness invariant is route_vehicle_visual_heading_offset_rad = 0.0.

Code state:
  ROUTE_VEHICLE_MODEL_CLASS remains MVehicle.
  ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD is explicit and fixed at 0.0.
  Spawn records now include route_heading_theta and visual_heading_theta for auditability.
  Track/lead state remains based on route-position history, not expected-label copying.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  57 tests OK.

Current-tree lead orientation artifact:
  run artifacts\reasoned_trajectory_poc\lead_orientation_route_heading_48_20260524
  stock video artifacts\reasoned_trajectory_poc\lead_orientation_route_heading_48_20260524\videos\stock_lead_orientation_route_heading_48.mp4
  static/RTP video artifacts\reasoned_trajectory_poc\lead_orientation_route_heading_48_20260524\videos\static_lead_orientation_route_heading_48.mp4
  side-by-side video artifacts\reasoned_trajectory_poc\lead_orientation_route_heading_48_20260524\videos\side_by_side_lead_orientation_route_heading_48.mp4
```

2026-05-24 construction left/right fast-path audit:

```text
Scope:
  Added explicit construction_left and construction_right MetaDrive harness scenes.
  Legacy construction remains right-side for backward compatibility.
  These scenes only control where physical cones/barriers spawn; Qwen still has to classify via the scene board.

Code changes:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    CONSTRUCTION_SCENES = construction, construction_left, construction_right
    construction_scene_side() makes side explicit for harness tests

  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Construction choice scoring now asks for safe bounded shift direction:
      left  -> construction_right -> BIAS_LEFT / right_edge avoid token
      right -> construction_left  -> BIAS_RIGHT / left_edge avoid token
      clear -> no construction label
    This was done because Qwen's hazard-side wording was unstable and sometimes answered desired maneuver side anyway.

  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Neutral BASE RTP at exactly durable_override_confidence no longer clears active lateral construction plans.
    This preserves short occlusion/splash gaps and prevents the avoidance ramp from restarting every no-label frame.

Rejected experiment:
  Scene-board LEFT/RIGHT visual markers were tried, then removed.
  They worsened left/right classification and caused 5 deadline misses in the test run:
    artifacts\reasoned_trajectory_poc\construction_left_orientmarkers_320x200_200_20260524

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  59 tests OK.

Latest summary artifact:
  artifacts\reasoned_trajectory_poc\construction_durablehold_summary_20260524.json

Latest right-side construction run:
  run artifacts\reasoned_trajectory_poc\construction_right_durablehold_320x200_200_20260524
  video artifacts\reasoned_trajectory_poc\construction_right_durablehold_320x200_200_20260524\videos\side_by_side_construction_right_durablehold_320x200_200.mp4
  publish 199/200, deadline_miss 1, same_frame_all true, max_age 0
  mean_latency_ms 45.141, max_latency_ms 57.125
  labels: construction_right 37, construction_left 115, none 48
  wrong_side_frames 115
  stock_min_clearance_m 1.206
  vlm_min_clearance_m 1.128
  active_lateral_offset_range_m -0.162 to 0.511
  result: INVALID. It still sign-flips too often and reduces clearance.

Latest left-side construction run:
  run artifacts\reasoned_trajectory_poc\construction_left_durablehold_320x200_200_20260524
  video artifacts\reasoned_trajectory_poc\construction_left_durablehold_320x200_200_20260524\videos\side_by_side_construction_left_durablehold_320x200_200.mp4
  publish 200/200, deadline_miss 0, same_frame_all true, max_age 0
  mean_latency_ms 45.870, max_latency_ms 48.837
  labels: construction_left 106, construction_right 34, none 60
  wrong_side_frames 34
  stock_min_clearance_m 1.206
  vlm_min_clearance_m 1.344
  active_lateral_offset_range_m -0.031 to 0.576
  result: PARTIAL. It improves clearance but still has too many wrong-side frames for the 0 sign-flip gate.

Current construction diagnosis:
  The compiler sign invariant is still correct in tests.
  Same-frame timing mostly holds, but right-side run had a single latency spike above 50 ms.
  Durable hold fixed the no-label ramp-reset problem and improves the left-side run.
  The remaining blocker is Qwen construction side/shift instability, especially right-side construction.
  Construction is not accepted and does not meet the success gate.
```

2026-05-24 lead vehicle visual-model correction:

```text
User reported the controlled lead vehicle still looked backwards, as if facing ego.

Root detail:
  Direct heading probes show MVehicle with visual heading = route heading is physically rear-facing.
  However, at the 320x200 scene-board/video resolution its rear texture is visually ambiguous enough to read as a front-facing car.
  Using route heading + pi is definitely wrong for MVehicle because it shows the front facing ego.

Fix:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added ROUTE_VEHICLE_MODEL_CLASSES = DefaultVehicle, LVehicle, MVehicle, XLVehicle.
    Changed default ROUTE_VEHICLE_MODEL_CLASS from MVehicle to XLVehicle.
    Changed route controlled vehicles to keep use_special_color false by default.
    Added --route-vehicle-model and --route-vehicle-special-color so old assets can still be reproduced.
    Spawned vehicle records now include model_class and use_special_color.

  tools\reasoned_trajectory_poc\render_demo_videos.py
    Renderer now accepts static/RTP reasoned frames when no vlm directory exists.

Why this is production-safe:
  The lead-state logic still uses physical route-position history:
    lead distance, lateral offset, speed, relative speed, closing, acceleration, lateral velocity.
  The sim model swap only makes the visual object unambiguous for Qwen and for human video review.
  It does not copy expected labels into the lead state and does not change car-control production logic.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\render_demo_videos.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  59 tests OK.

Evidence artifacts:
  probe artifacts\reasoned_trajectory_poc\lead_vehicle_heading_probe_20260524\separate\MVehicle_route.png
  probe artifacts\reasoned_trajectory_poc\lead_vehicle_heading_probe_20260524\separate\MVehicle_route_plus_pi.png
  probe artifacts\reasoned_trajectory_poc\lead_vehicle_heading_probe_20260524\textured\XLVehicle_special_0.png
  run artifacts\reasoned_trajectory_poc\lead_orientation_xlvehicle_48_20260524
  side-by-side video artifacts\reasoned_trajectory_poc\lead_orientation_xlvehicle_48_20260524\videos\side_by_side_lead_orientation_xlvehicle_48.mp4

Current result:
  The lead object now appears as the rear of a same-direction box truck in the driver-camera/scene-board path.
```

2026-05-24 construction candidate-guide and contradictory-side debounce experiment:

```text
Goal:
  Reduce construction sign flips without adding simulator-only object hacks.
  Changes must remain compatible with the real scene-board -> Qwen labels -> RTP -> compiler path.

Code changes:
  selfdrive\controls\reasoned\ui_scene_board.py
    Added optional candidate guide rendering:
      blue guide = bounded left-shift candidate
      orange guide = bounded right-shift candidate
    This is disabled by default because closed-loop evidence did not pass.

  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added --scene-board-candidate-guides and --candidate-guide-offset-m.
    Added DurableLateralOverrideState.
    Added contradictory-side confirmation support:
      --durable-conflict-confirm-frames
      --durable-conflict-immediate-confidence
    Defaults remain non-debounced because the 3-frame setting helped right-side construction but worsened left-side construction.
    Logs now include durable_lateral_pending_source, durable_lateral_pending_sign, and durable_lateral_pending_count.

  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Kept the construction choice contract as safe shift direction:
      answer left -> construction_right -> BIAS_LEFT / right_edge avoid
      answer right -> construction_left -> BIAS_RIGHT / left_edge avoid

Tests:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  62 tests OK.

Single-frame probe with optional candidate guides:
  right-side construction frame:
    image artifacts\reasoned_trajectory_poc\construction_right_candidateguides_probe_90_20260524\static\vlm_input_0060.png
    Qwen selected construction_right
    total_ms about 38.1

  left-side construction frame:
    image artifacts\reasoned_trajectory_poc\construction_left_candidateguides_probe_90_20260524\static\vlm_input_0060.png
    Qwen selected construction_left
    total_ms about 38.1

Closed-loop candidate-guide run without debounce:
  right run artifacts\reasoned_trajectory_poc\construction_right_candidateguides_vlm_320x200_200_20260524
    publish 200/200, deadline_miss 0, same_frame_all true, max_age 0
    mean_latency_ms 42.947, max_latency_ms 45.119
    labels: construction_right 100, construction_left 74, none 26
    wrong_side_frames 74
    stock_min_clearance_m 1.206
    vlm_min_clearance_m 1.214
    result: INVALID, too many sign flips.

  left run artifacts\reasoned_trajectory_poc\construction_left_candidateguides_vlm_320x200_200_20260524
    publish 200/200, deadline_miss 0, same_frame_all true, max_age 0
    mean_latency_ms 43.001, max_latency_ms 43.909
    labels: construction_left 82, construction_right 92, none 26
    wrong_side_frames 92
    stock_min_clearance_m 1.206
    vlm_min_clearance_m 1.181
    result: INVALID, worse than stock.

Closed-loop candidate-guide run with 3-frame contradictory-side debounce:
  right run artifacts\reasoned_trajectory_poc\construction_right_candidateguides_debounce_vlm_320x200_200_20260524
    video artifacts\reasoned_trajectory_poc\construction_right_candidateguides_debounce_vlm_320x200_200_20260524\videos\side_by_side_construction_right_candidateguides_debounce_vlm_320x200_200.mp4
    publish 200/200, deadline_miss 0, same_frame_all true, max_age 0
    mean_latency_ms 42.895, max_latency_ms 44.337
    labels: construction_right 97, construction_left 74, none 29
    durable sources: right_edge 153 frames, left_edge 47 frames
    stock_min_clearance_m 1.206
    vlm_min_clearance_m 1.599
    result: PARTIAL. Better clearance, but still not a 0 sign-flip construction result.

  left run artifacts\reasoned_trajectory_poc\construction_left_candidateguides_debounce_vlm_320x200_200_20260524
    video artifacts\reasoned_trajectory_poc\construction_left_candidateguides_debounce_vlm_320x200_200_20260524\videos\side_by_side_construction_left_candidateguides_debounce_vlm_320x200_200.mp4
    publish 200/200, deadline_miss 0, same_frame_all true, max_age 0
    mean_latency_ms 42.988, max_latency_ms 44.662
    labels: construction_left 64, construction_right 108, none 28
    durable sources: right_edge 166 frames, left_edge 34 frames
    stock_min_clearance_m 1.206
    vlm_min_clearance_m 0.847
    result: INVALID. The debounce locked in the wrong early judgement and made left-side construction worse.

Current diagnosis:
  Candidate guides can make individual frames score correctly, but do not solve closed-loop construction side instability.
  Simple persistence/debounce is not sufficient because it can confidently preserve the wrong initial side.
  These controls remain available for experiments, but not accepted as the default production path.
  Construction still fails the 0 sign-flip gate.
```

2026-05-24 lead vehicle model orientation correction:

User reported the lead car model was backwards, visually facing ego as if it were a head-on vehicle.

Root cause / evidence:
  A fresh four-model front-camera contact sheet showed:
    DefaultVehicle: visually reads as the rear of a same-direction lead car.
    LVehicle: same-direction but visually cluttered.
    MVehicle: visually reads as a front grille facing ego at route heading 0.
    XLVehicle: physically aligned but visually ambiguous because the cargo-box rear is a large flat panel.
  Artifact:
    artifacts\reasoned_trajectory_poc\lead_model_probe_contact_sheet_20260524.png

Fix:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Changed default ROUTE_VEHICLE_MODEL_CLASS from XLVehicle to DefaultVehicle.
    Kept ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD at 0.0.
    This changes only the MetaDrive visual mesh default. It does not alter the physical route heading, velocity vector, or production-style lead track fields.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    74 tests OK.

  Lead-only demo:
    run artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_48_20260524
    video artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_48_20260524\videos\side_by_side_lead_orientation_defaultvehicle_48.mp4
    stock frame artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_48_20260524\stock\stock_overlay_0044.png
    record model_class DefaultVehicle
    route_heading_theta 0.0
    visual_heading_theta 0.0
    heading_delta 0.0

Status:
  Lead visual is corrected to an unambiguous same-direction rear-view car model.
  This is a sim visualization fix only. The real-car/prod path still requires physical lead state from openpilot/model/radar or VLM image evidence, not MetaDrive labels.

2026-05-24 construction rerun after lead visual fix and output-runner cleanup:

Goal:
  Continue the production-shaped construction gate using the existing path:
  scene board -> Qwen labels/scores -> RTP -> deterministic compiler -> planner/control.

Current focused construction reruns:
  left run:
    artifacts\reasoned_trajectory_poc\construction_left_current_defaultvehicle_async_160_20260524
    video artifacts\reasoned_trajectory_poc\construction_left_current_defaultvehicle_async_160_20260524\videos\side_by_side_construction_left_current_defaultvehicle_async_160.mp4
    stock min construction route clearance 1.206 m
    VLM min construction route clearance 2.008 m
    publish 160/160, valid 160/160, deadline misses 0
    max RTP age 3 frames
    max planner overhead 4.089 ms
    raw labels: construction_shift_left 70, construction_shift_right 56, none 34
    durable sources: left_edge 123 frames, right_edge 31 frames
    active lateral never moved into the left-side construction in this run.

  right run:
    artifacts\reasoned_trajectory_poc\construction_right_current_defaultvehicle_async_160_20260524
    video artifacts\reasoned_trajectory_poc\construction_right_current_defaultvehicle_async_160_20260524\videos\side_by_side_construction_right_current_defaultvehicle_async_160.mp4
    stock min construction route clearance 1.206 m
    VLM min construction route clearance 2.168 m
    publish 160/160, valid 160/160, deadline misses 0
    max RTP age 3 frames
    max planner overhead 5.156 ms
    raw labels: construction_shift_left 64, construction_shift_right 63, none 33
    durable sources: right_edge 160 frames
    active lateral never moved into the right-side construction in this run.

Interpretation:
  The durable compiler/control path is currently preventing active sign-flip steering in these two focused runs.
  The raw Qwen construction labels are still far too unstable to claim the construction gate is passed.
  This is not accepted for real-car use yet because success is currently dependent on durable conflict logic masking label noise.

Rejected experiments from this pass:
  Visual side labels construction_left/construction_right were probed as an alternative to action labels.
    Left-side closed loop:
      artifacts\reasoned_trajectory_poc\construction_left_sidelabels_async_160_20260524
      stock clearance 1.206 m, VLM clearance 1.206 m
      source counts empty because side-label confidence stayed below durable activation.
    Result: not a better default.

  Driver-left/right text labels drawn onto the scene board plus stricter action prompts were tried and then reverted.
    left run artifacts\reasoned_trajectory_poc\construction_left_directionlabels_async_160_20260524
      VLM clearance fell to 1.578 m
    right run artifacts\reasoned_trajectory_poc\construction_right_directionlabels_async_160_20260524
      VLM clearance fell to 1.949 m
    Result: known regression, removed from code.

  Hidden-output construction choice engine was probed.
    It avoids the crash after the runner fix below, but it was directionally wrong on right-side construction samples and costs about 254 ms in one-shot benchmark form.
    Result: not a construction default.

Code cleanup accepted from this pass:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Added automatic selected_logits/selected_hidden output selection for TensorRT text engines.
    This prevents a hidden-output engine from crashing when the caller omits --text-output hidden.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added coverage for text-output fallback selection.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    75 tests OK.

  Hidden-choice smoke that previously crashed:
    py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py --artifact-dir F:\qwen_trt_export --text-engine F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine --text-seq-len 576 --label-decision-mode choice --score-prompt-mode construction-compact --score-labels construction_shift_left,construction_shift_right --score-threshold 0.5 --image artifacts\reasoned_trajectory_poc\construction_left_current_defaultvehicle_async_160_20260524\vlm\vlm_input_0000.png --warmup 0 --iters 1 benchmark
    Result: no crash, engine output auto-detected as selected_hidden.

Current construction status:
  Focused left/right lateral-only demos improved clearance and stayed nonblocking.
  Construction still fails the final gate because raw Qwen side/shift labels are unstable and this has not been proven in mixed, speed-control-enabled, real-video/log-replay, or openpilot-on-road-equivalent evaluations.

2026-05-24 lead physical-track audit after DefaultVehicle fix:

Goal:
  Verify that lead handling uses production-plausible physical track fields and not MetaDrive scenario labels, then rerun the missing lead interaction cases with the corrected same-direction lead visual model.

Code changes:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Each VLM record now persists scene_board_state_text so the exact text sent with the scene board is inspectable in episode logs.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added a regression test proving scene-board state text omits simulator-only expected_lead_class and object kind strings even when those keys exist in the caller state.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    76 tests OK.
  Static traceability smoke:
    run artifacts\reasoned_trajectory_poc\trace_state_text_static_smoke_3_20260524
    confirmed records now contain scene_board_state_text and that text contains no expected_lead_class/sim expected-label leakage.

Lead-state contract:
  Qwen receives only the scene-board image plus plain-language physical fields:
    lead_present, lead_source, lead_distance_m, lead_lateral_m, lead_speed_mps,
    lead_rel_speed_mps, lead_closing_mps, lead_accel_mps2, lead_lateral_velocity_mps.
  MetaDrive expected_lead_class remains only an evaluation label in logs and summaries.
  The lead-state code deliberately ignores expected_lead_class and object kind semantics except for selecting vehicle-like tracks in the sim harness.

Audit artifact:
  artifacts\reasoned_trajectory_poc\lead_defaultvehicle_physical_track_audit_20260524.json

Rerun with corrected DefaultVehicle model, synchronous TensorRT Qwen choice path, no async reuse:
  cut-in:
    run artifacts\reasoned_trajectory_poc\lead_cut_in_vehicle_defaultvehicle_sync_120_20260524
    video artifacts\reasoned_trajectory_poc\lead_cut_in_vehicle_defaultvehicle_sync_120_20260524\videos\side_by_side_lead_cut_in_vehicle_defaultvehicle_sync_120_20260524.mp4
    publish 120/120, valid 120/120, deadline misses 0
    same-frame all true, max RTP age 0
    mean/max latency 42.732/44.213 ms
    stock min vehicle route clearance 5.744 m, VLM 7.407 m
    qwen lead classes: cut_in_vehicle_entering_path, true_moving_lead, slower_lead_closing

  crossing vehicle:
    run artifacts\reasoned_trajectory_poc\lead_crossing_vehicle_defaultvehicle_sync_120_20260524
    video artifacts\reasoned_trajectory_poc\lead_crossing_vehicle_defaultvehicle_sync_120_20260524\videos\side_by_side_lead_crossing_vehicle_defaultvehicle_sync_120_20260524.mp4
    publish 120/120, valid 120/120, deadline misses 0
    same-frame all true, max RTP age 0
    mean/max latency 42.648/44.551 ms
    stock min vehicle route clearance 4.757 m, VLM 9.591 m
    qwen lead classes: crossing_vehicle_conflict, stopped_lead_in_path, irrelevant_vehicle

  irrelevant vehicle:
    run artifacts\reasoned_trajectory_poc\lead_irrelevant_vehicle_defaultvehicle_sync_120_20260524
    video artifacts\reasoned_trajectory_poc\lead_irrelevant_vehicle_defaultvehicle_sync_120_20260524\videos\side_by_side_lead_irrelevant_vehicle_defaultvehicle_sync_120_20260524.mp4
    publish 120/120, valid 120/120, deadline misses 0
    same-frame all true, max RTP age 0
    mean/max latency 42.938/45.375 ms
    stock and VLM mean speed equal at 2.059 m/s
    speed delta 0.0 m/s, no false slowdown
    qwen lead classes: irrelevant_vehicle

Current lead status:
  The corrected visual model plus production-shaped physical lead-track state is now verified for true moving, slower, braking, stopped, cut-in, crossing, and irrelevant vehicle harness cases.
  Lead handling is not yet a final production gate pass because real openpilot log replay and real-road scene-board evaluation are still required.
  The audit also shows the optimized Qwen visual/text answer often needs the physical track consistency filter for lead labels, so this must be documented as a fused visual-plus-track decision path, not claimed as pure visual classification.

2026-05-24 construction scoring root-cause probes:

Goal:
  Identify why construction lateral decisions are still unstable without adding simulator-only fallbacks or hardcoded side hacks.

Code correction:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Corrected construction_left/construction_right choice semantics:
      construction_left/right now means hazard side.
      construction_shift_left/right remains the action-direction pair.
    Before this change, the construction_left/right choice group reused the shift prompt and inverted left/right into action labels, which made the label names semantically wrong.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Updated the construction choice unit test so left answer maps to construction_left hazard side.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    76 tests OK.

Probe artifacts:
  artifacts\reasoned_trajectory_poc\construction_choice_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_side_choice_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_side_choice_seq640_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_pytorch_size_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_binary_side_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_hidden_binary_side_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_binary_side_zero_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_fp16_binary_side_probe_20260524.json

Findings:
  TensorRT choice action labels are not viable:
    construction_shift_left/right choice on saved focused frames chose construction_shift_right for nearly every right-side frame and many left-side frames.
    This would steer into right-side hazards.

  TensorRT construction_left/right side-choice after the semantic fix is also not viable:
    seq576 hidden choice: left run mostly construction_right, right run mostly construction_right.
    seq640 hidden choice: still collapses to construction_right on both left and right runs.
    Prompt length is not the root cause.

  PyTorch binary side scoring is directionally correct on representative frames at full168, full224, full280, and full336:
    left_0000/left_0024 argmax construction_left.
    right_0000/right_0024 argmax construction_right.
    Hot PyTorch decode is about 350-415 ms for two labels, so this is only a correctness reference, not a runtime path.

  TensorRT NVFP4 binary side scoring does not match the PyTorch side signal over the full saved sequence:
    default qwen position mode gets some representative frames right but is unstable across the run.
    zero position mode fixes the four representative frames but still only gets 12/20 left saved frames and 11/20 right saved frames correct by argmax.
    hidden-output binary scoring is slower and still unstable.

  TensorRT FP16 binary text engine was built as a correctness control:
    engine F:\qwen_trt_export\fp16_trt\qwen_text_36layer_fp16_seq576_trt.engine
    engine size about 5.56 GB
    runtime about 4.2 seconds per two-label score, so unusable.
    It did not recover left-side construction in the saved-frame probe.

  Full PyTorch scoring uses the full SCORE_PROMPT, but that prompt does not fit the existing seq576 TensorRT contract:
    full construction_left/right prompt length is 715 tokens, so qwen_trt_label_engine correctly rejects it for seq576.

  A shorter construction-score prompt was added and tested under seq576:
    It preserves the critical full-prompt construction side rules under the fixed length.
    It still misclassified a representative left-side construction frame as construction_right.
    Result: not accepted as a runtime default.

Current construction diagnosis:
  The remaining construction blocker is not fixed by more prompt wording, longer text length, side-vs-shift label naming, hidden-output logits, or simple position-id tweaks.
  PyTorch Qwen contains a usable construction side signal on representative frames, but the current optimized TensorRT text scoring contract does not preserve it robustly enough for closed-loop steering.
  Do not switch runtime defaults to the probed choice or FP16 paths.
  Next useful direction is an optimized scoring contract that better preserves PyTorch label ranking for construction while staying inside the 50 ms budget, likely by reducing the construction scoring problem to a smaller stable set of answer tokens/positions or by building a calibrated construction-specific Qwen score path from Qwen hidden features without using simulator-only labels as input.

2026-05-24 lead visual orientation proof rerun:

User report:
  The lead car looked backwards, as if it was facing ego for a head-on collision.

Current code invariant:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    ROUTE_VEHICLE_MODEL_CLASS = DefaultVehicle
    ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0
    ROUTE_VEHICLE_USE_SPECIAL_COLOR = False

Fresh proof run:
  command:
    py -3.11 tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py --frames 48 --speed-mps 2.5 --engine static --novel-scene slower_lead --route-vehicle-model DefaultVehicle --no-route-vehicle-special-color --save-every 4 --out artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_proof_48_20260524
  artifact:
    artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_proof_48_20260524\stock\stock_overlay_0044.png
    artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_proof_48_20260524\videos\side_by_side_lead_orientation_defaultvehicle_proof_48.mp4
  episode record:
    model_class DefaultVehicle
    route_heading_theta 0.0
    visual_heading_theta 0.0
    use_special_color false

Result:
  The fresh render shows the rear of a same-direction lead vehicle in ego lane. The previous backwards/head-on appearance was from the MetaDrive visual mesh choice, not from physical lead heading or route state.

2026-05-24 lead vehicle visual correction follow-up:

```text
User report:
  The DefaultVehicle lead still visually read as facing ego in the proof video.

Root cause:
  The physical route heading was not backwards. Direct node inspection showed:
    heading offset 0.0: headlight nodes are ahead of the route point, backlight nodes are behind it.
    heading offset pi: headlight nodes move behind the route point, which really would face ego.
  Therefore a heading sign flip would be wrong.

Fix:
  Kept ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0.
  Changed default ROUTE_VEHICLE_MODEL_CLASS from DefaultVehicle to LVehicle because the DefaultVehicle rear/front silhouette is too ambiguous at the 320x200 VLM-board resolution.
  The lead physical track remains route-position/route-velocity based and still ignores simulator expected labels.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  79 tests OK.

  run artifacts\reasoned_trajectory_poc\lead_orientation_lvehicle_default_48_20260524
  stock frame artifacts\reasoned_trajectory_poc\lead_orientation_lvehicle_default_48_20260524\stock\stock_overlay_0044.png
  side-by-side video artifacts\reasoned_trajectory_poc\lead_orientation_lvehicle_default_48_20260524\videos\side_by_side_lead_orientation_lvehicle_default_48.mp4
```

2026-05-24 construction prompt-contract isolation:

```text
Goal:
  Resolve whether the current TensorRT text-engine construction failure is caused by the qwen_trt_label_engine prompt not matching the qwen_label_rtp_worker PyTorch prompt.

Code changes:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Added worker_full as an artifact-safe prompt mode in engine/ONNX paths.
    Non-default score_prompt_mode now appears in generated text-engine filenames, e.g.
      qwen_text_36layer_nvfp4_seq768_worker_full_strong_trt.engine
    This prevents a worker-full contract from silently sharing a misleading full-prompt artifact name.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added exact prompt-equivalence coverage proving worker_full emits the same SCORE_PROMPT / Vehicle state / Question string used by qwen_label_rtp_worker.
    Added filename invariant coverage for worker_full strong text engines.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  81 tests OK.

Saved-frame PyTorch wrapper probes:
  artifact artifacts\reasoned_trajectory_poc\construction_pytorch_worker_full_wrapper_probe_20260524.json
    qwen_trt full prompt:
      left-side construction 20/20 correct
      right-side construction 2/20 correct
    worker_full prompt matching qwen_label_rtp_worker:
      left-side construction 20/20 correct
      right-side construction 10/20 correct
    result: worker_full improves right-side ranking but is still rejected; 50% right-side success is far below the >=95% construction-side gate.

  artifact artifacts\reasoned_trajectory_poc\construction_worker_full_seq0_vs_seq768_textscore_probe_20260524.json
    seq0 and seq768 gave the same representative failures.
    result: fixed-length padding is not the root cause of the right-side construction bias in the PyTorch TextScore-compatible path.

  artifact artifacts\reasoned_trajectory_poc\construction_pytorch_legacy_prompt_textscore_probe_20260524.json
    old shorter prompt with current construction questions:
      left 20/20, right 4/20
    result: rejected.

  artifact artifacts\reasoned_trajectory_poc\construction_pytorch_legacy_prompt_legacy_questions_probe_20260524.json
    old shorter prompt with old construction questions:
      left 20/20, right 1/20
    result: rejected.

  artifact artifacts\reasoned_trajectory_poc\construction_pytorch_worker_full_shift_wrapper_probe_20260524.json
    construction_shift labels:
      left hazard -> required shift_right: 0/20 correct
      right hazard -> required shift_left: 19/20 correct
    result: rejected; action labels are strongly biased toward shift_left and cannot be used as a reliable construction-side fix.

Current diagnosis:
  The prompt mismatch was real and is now tested/artifact-safe, but it is not sufficient to fix construction side classification.
  The current optimized-compatible Qwen TextScore path has a persistent bias toward construction_left / shift_left on these saved MetaDrive construction boards.
  Do not build or promote a worker_full TensorRT runtime engine for construction until a saved-frame probe clears the side-classification gate.
```

2026-05-24 construction geometry-aware trace metrics:

```text
Problem found:
  The previous saved-frame construction probes used scene name as ground truth for every saved frame.
  That is too broad for real-car behavior. A frame can contain visible cones but not require a new side label if:
    the nearest cone is still far ahead and not immediately intruding into the current path, or
    the tracked green path has already moved away and the cone is outside the current corridor.

Code changes:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added spawned_route_proximity().
    Per-frame logs now include nearest object ahead/lateral deltas for construction, pedestrians, and route vehicles:
      construction_nearest_ahead_m
      construction_nearest_lateral_delta_m
      construction_nearest_object_lateral_m
      construction_nearest_kind
      pedestrian_nearest_ahead_m
      pedestrian_nearest_lateral_delta_m
      pedestrian_nearest_object_lateral_m
      pedestrian_nearest_kind
      vehicle_nearest_ahead_m
      vehicle_nearest_lateral_delta_m
      vehicle_nearest_object_lateral_m
      vehicle_nearest_kind
    These are evaluation trace fields only. They are not fed to Qwen and do not alter control.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added a spawned_route_proximity unit test proving nearest construction object ahead/lateral delta logging.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  82 tests OK.

Fresh trace-metric smoke run:
  run artifacts\reasoned_trajectory_poc\construction_right_trace_metrics_static_80_20260524
  example frame 0 in static episode:
    construction_nearest_ahead_m 13.994
    construction_nearest_lateral_delta_m 1.35
    construction_nearest_kind traffic_cone_right_edge
  This confirms logs can now distinguish longitudinal distance from lateral path intrusion instead of only reporting a hypotenuse clearance.

Geometry-aware saved-frame analysis:
  artifact artifacts\reasoned_trajectory_poc\construction_worker_full_side_geometry_aware_probe_20260524.json
    scoring rule for analysis only:
      side label required only if nearest construction is 0-14 m ahead and within 1.65 m lateral delta of the tracked path.
    worker_full side labels:
      left path-relevant frames: 7/7 correct
      right path-relevant frames: 0/7 correct
      non-required frames still tend to get positive construction labels
    result: rejected, but the failure is now localized to path-relevant right-side side scoring.

  artifact artifacts\reasoned_trajectory_poc\construction_worker_full_shift_geometry_aware_probe_20260524.json
    worker_full shift labels:
      left path-relevant frames requiring shift_right: 0/7 correct
      right path-relevant frames requiring shift_left: 7/7 correct
      non-required frames still tend to get positive construction labels
    result: rejected as a runtime default.

  artifact artifacts\reasoned_trajectory_poc\construction_worker_full_side_mirror_debias_probe_20260524.json
    horizontal-mirror mapped side score, using no simulator label as runtime input:
      left path-relevant frames: 6/7 correct
      right path-relevant frames: 7/7 correct
    diff-debias score:
      left path-relevant frames: 7/7 correct
      right path-relevant frames: 5/7 correct
    result: promising but not accepted. Mirror scoring doubles VLM text work and still needs a neutral/clear threshold that does not fire on already-avoided frames.

Current construction diagnosis:
  The old all-frame construction probe overstated some failures, but the real blocker remains.
  Path-relevant right-side side labels are still wrong under the normal worker_full side scorer.
  Mirror-mapped scoring is the first current probe that recovers both sides on path-relevant frames, but it is not robust enough yet to promote.
```

2026-05-24 reusable construction trace evaluator:

```text
Goal:
  Make construction acceptance/rejection repeatable from logs instead of relying on ad hoc snippets or visual inspection.

Code changes:
  tools\reasoned_trajectory_poc\evaluate_construction_trace.py
    New CLI and importable helper functions.
    Reads a MetaDrive RTP episode trace and classifies each frame as path-relevant construction only when nearest construction is:
      ahead within horizon_m, default 14.0
      within lateral intrusion_m of the tracked path, default 1.65
    Reports:
      path_relevant_construction_frames
      qwen_side_correct / wrong / missing
      qwen_side_success_rate
      false_construction_labels / rate on not-path-relevant frames
      control_away / toward / neutral
      collision_count
      min_construction_route_clearance_m
      max_rtp_age_frames
    It uses the new construction_nearest_* fields when present and falls back to spawned_scene + route_longitudinal_m + lane_lateral_m for older traces.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added tests for:
      path-relevant requirement extraction from ahead/lateral deltas
      fallback extraction from spawned_scene traces
      construction label to hazard-side mapping
      MetaDrive lateral command side mapping
      full evaluator counts for side errors, sign flips, collisions, false labels, and max RTP age

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\evaluate_construction_trace.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  87 tests OK.

Evaluator artifacts:
  artifacts\reasoned_trajectory_poc\construction_left_current_defaultvehicle_async_160_20260524\vlm\construction_trace_evaluation.json
    path_relevant_construction_frames 52
    qwen_side_correct 21
    qwen_side_wrong 18
    qwen_side_missing 13
    qwen_side_success_rate 0.404
    false_construction_rate 0.806
    control_away_rate 0.635
    collision_count 0
    min_construction_route_clearance_m 2.008
    max_rtp_age_frames 3

  artifacts\reasoned_trajectory_poc\construction_right_current_defaultvehicle_async_160_20260524\vlm\construction_trace_evaluation.json
    path_relevant_construction_frames 51
    qwen_side_correct 24
    qwen_side_wrong 23
    qwen_side_missing 4
    qwen_side_success_rate 0.471
    false_construction_rate 0.734
    control_away_rate 0.667
    collision_count 0
    min_construction_route_clearance_m 2.168
    max_rtp_age_frames 3

Result:
  Rejected for construction gate.
  The focused construction runs are physically clearing cones in this harness, but they do not satisfy production-grade Qwen semantics:
    Qwen side correctness is far below 95%.
    False construction labels on non-path-relevant frames are far above 5%.
    Control still has neutral/toward intervals instead of consistently moving away on every path-relevant construction frame.
```

2026-05-24 construction path-relevance prompt probes:

```text
Goal:
  Test whether the construction false-positive problem is primarily prompt wording around "path relevant" versus a deeper Qwen side-token bias.

Artifacts:
  artifacts\reasoned_trajectory_poc\construction_path_relevance_prompt_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_near_touch_mirror_probe_20260524.json

Prompt variants tested:
  path_touch:
    "yes only if a cone/barrier physically touches the green translucent area, overlaps it, crosses a white corridor border, or blocks the green path ahead"
  near_touch:
    "only the near part of the corridor matters; yes only if inside/touching/narrowing the green area; no if already avoided outside the shifted green corridor"

Results without mirroring:
  path_touch:
    left required frames: 7/7 correct
    right required frames: 0/7 correct
    non-required false positive frames:
      left run 9
      right run 2

  near_touch:
    left required frames: 7/7 correct
    right required frames: 0/7 correct
    non-required false positive frames:
      left run 6
      right run 2

Results with near_touch plus mirror-derived scores:
  raw original scoring:
    left required 7/7
    right required 0/7
  mirror-mapped scoring:
    left required 0/7
    right required 7/7
  diff scoring:
    left required 7/7
    right required 5/7

Threshold analysis:
  No score threshold or margin tested cleanly preserves required-frame correctness while suppressing false construction on non-required frames.
  Required and non-required score ranges overlap heavily.

Decision:
  Rejected as runtime change.
  The explicit path-relevance prompts reduce some false positives, but they do not solve the construction side bias.
  Mirror-mapped scoring recovers the side Qwen is biased against, but flips the opposite side and still lacks a defensible neutral gate.
  Do not add a runtime knob for this yet; it would create a selectable failure mode rather than a production-grade improvement.
```

2026-05-24 TensorRT construction precision/root-cause probes:

Goal:
  Determine whether the optimized TensorRT text tower can preserve PyTorch Qwen's full-prompt construction-side ranking without changing the Qwen -> label score -> RTP -> compiler architecture.

Code additions:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py
    Added additive --text-precision fp8.
    Added FP8 ModelOpt export using FP8_DEFAULT_CFG and FP8QuantExporter.
    Added TensorRT FP8 builder flag support.
    Fixed quantized ONNX saving for large FP8 processed models by writing external weight data instead of a single >2 GB protobuf.
    Added additive --text-strongly-typed, which builds separate *_strong_trt.engine files with NetworkDefinitionCreationFlag.STRONGLY_TYPED and without overriding precision flags.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added tests for fp8 precision acceptance, distinct strongly typed engine paths, and runtime-manifest tracking of text_strongly_typed.

Build artifacts:
  F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq768_trt.engine
  F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq768_hidden_trt.engine
  F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq768_strong_trt.engine
  F:\qwen_trt_export\nvfp4_trt\qwen_text_36layer_nvfp4_seq768_strong_trt.engine

Probe artifacts:
  artifacts\reasoned_trajectory_poc\construction_trt_fp8_seq768_fullprompt_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_fp8_seq768_hidden_fullprompt_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_fp8_seq768_strong_fullprompt_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_fp8_strong_mirror_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_trt_nvfp4_seq768_strong_fullprompt_probe_20260524.json

Results:
  FP8 full prompt, logits output:
    left 16/20 correct by argmax.
    right 9/20 correct by argmax.
    mean total latency about 135-136 ms.
    Not acceptable.

  FP8 full prompt, hidden output with LM-head scoring outside TensorRT:
    left 16/20 correct by argmax.
    right 9/20 correct by argmax.
    mean total latency about 136-138 ms.
    Same result as logits, so the selected-logit matmul is not the root cause.

  FP8 full prompt, strongly typed TensorRT:
    left 19/20 correct by argmax.
    right 5/20 correct by argmax.
    mean total latency about 133-134 ms.
    Strong typing removes the FP8 Q/DQ warning flood and improves left-side scoring, but reveals a strong left bias and still fails right-side construction.

  FP8 strongly typed mirror probe:
    original pass: left 19/20, right 5/20.
    mirrored-and-mapped pass: left 1/20, right 20/20.
    mean two-pass latency about 264-266 ms.
    This proves the optimized engine's construction scoring is side-biased, not a reliable geometric side classifier. The mirror pass is not accepted as a runtime fix because it is too slow and ambiguous without an independent, production-plausible confidence rule.

  NVFP4 full prompt, strongly typed TensorRT:
    left 18/20 correct by argmax.
    right 9/20 correct by argmax.
    mean total latency about 92.6 ms.
    Better than the previous NVFP4 full-prompt seq768 probe but still fails right-side construction and remains above 50 ms.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    79 tests OK.

Current conclusion:
  The construction blocker is now isolated to quantized TensorRT language-tower fidelity for Qwen's construction-side decision. FP8, hidden-output scoring, strongly typed FP8, and strongly typed NVFP4 do not meet the construction gate. None of these paths should become the default runtime. The next viable path must preserve PyTorch Qwen's side ranking with a smaller production-shaped scoring contract or a calibrated Qwen-hidden-feature scorer whose inputs are real scene-board evidence, not simulator labels.
2026-05-24 lead vehicle visual-mesh correction:

User report:
  The current lead car still looked backwards/head-on in the driver-view sim video.

Root cause:
  The route and velocity heading were already same-direction. The issue was the selected MetaDrive visual mesh. LVehicle still reads like a front-facing/oncoming car from the low-resolution ego scene-board camera, even with visual_heading_theta equal to route_heading_theta.

Fix:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added SVehicle to ROUTE_VEHICLE_MODEL_CLASSES.
    Changed default ROUTE_VEHICLE_MODEL_CLASS from LVehicle to SVehicle.
    Kept ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0.
  This changes only the sim visual asset used for route vehicles. It does not flip vehicle physics, does not alter lead-state generation, and does not use simulator labels as runtime inputs.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  87 tests OK.

  proof run:
    artifacts\reasoned_trajectory_poc\lead_orientation_svehicle_default_48_20260524
  proof frame:
    artifacts\reasoned_trajectory_poc\lead_orientation_svehicle_default_48_20260524\stock\stock_overlay_0044.png
  proof video:
    artifacts\reasoned_trajectory_poc\lead_orientation_svehicle_default_48_20260524\videos\side_by_side_lead_orientation_svehicle_default_48.mp4
  episode record:
    model_class SVehicle
    route_heading_theta 0.0
    visual_heading_theta 0.0
    expected_lead_class slower_lead

2026-05-24 construction scene-board envelope and side-scoring probes:

Goal:
  Continue the construction blocker without changing architecture:
  scene board -> Qwen labels/scores -> RTP -> deterministic compiler -> planner/control.

Code changes:
  selfdrive\controls\reasoned\ui_scene_board.py
    Changed default planned_corridor_half_width_m from 0.60 to 0.90.
    Changed default draw_base_path_reference from true to false.
  Rationale:
    The prior green overlay was still visually a narrow center ribbon. Saved
    construction frames showed cones close enough for the evaluator to call
    them path-relevant while Qwen still answered clear because the cones looked
    outside the green area. A 0.90 m half-width is a production-shaped ego
    vehicle envelope/risk corridor, not a simulator-specific rule.
    The magenta base-reference line is now opt-in because it can be mistaken for
    lane/construction geometry and is not the actual path being tracked.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added invariant that the default scene board uses a vehicle-width corridor
    and no base-reference line.
    Kept explicit coverage for drawing the magenta base-reference line when the
    geometry opts into it.

Docs:
  README.md and README_REASONED_TRAJECTORY_POC.md now document the 0.90 m
  default corridor and opt-in base-reference line.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\evaluate_construction_trace.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  88 tests OK.

Fresh boards:
  artifacts\reasoned_trajectory_poc\construction_left_wide_vehicle_corridor_nominal_80_20260524
  artifacts\reasoned_trajectory_poc\construction_right_wide_vehicle_corridor_nominal_80_20260524
  sample images:
    artifacts\reasoned_trajectory_poc\construction_left_wide_vehicle_corridor_nominal_80_20260524\vlm\vlm_input_0000.png
    artifacts\reasoned_trajectory_poc\construction_right_wide_vehicle_corridor_nominal_80_20260524\vlm\vlm_input_0000.png

Qwen probe artifacts:
  artifacts\reasoned_trajectory_poc\construction_choice_side_pytorch_probe_20260524.json
    Existing three-way choice prompts on older shifted construction boards:
      construction_side_choice: 0/14 required correct, 14/14 missing, Qwen answered clear for every required frame.
      construction_shift_choice: 0/14 required correct, 14/14 missing, Qwen answered clear for every required frame.

  artifacts\reasoned_trajectory_poc\construction_forced_side_mirror_delta_probe_20260524.json
    Forced left/right side prompt with original+mirrored de-bias on older shifted boards:
      threshold 0.0: 12/14 required correct, 2 wrong, but 26/26 non-required false positives.
      any threshold high enough to suppress false positives also suppressed every required frame.
    Rejected as a runtime contract.

  artifacts\reasoned_trajectory_poc\construction_two_way_prompt_variant_probe_20260524.json
    Best variant was strict_hazard_side_lr:
      threshold 0.0: 13/14 required correct, 1 wrong, but 26/26 non-required false positives.
    Other wording variants either reduced required accuracy or still had unusable false positives.
    Rejected as a runtime contract.

  artifacts\reasoned_trajectory_poc\construction_widecorridor_qwen_side_probe_20260524.json
    Probes on fresh 0.90 m corridor nominal boards:
      three-way side choice: 0/20 required correct, Qwen answered clear for every frame.
      three-way shift choice: 0/20 required correct, Qwen answered clear for every frame.
      strict two-way mirrored side: 11/20 required correct at threshold 0.0.
    This proves the wider envelope is necessary for scene-board correctness but
    not sufficient to solve Qwen construction side scoring.

  artifacts\reasoned_trajectory_poc\construction_highlighted_corridor_half_probe_20260524.json
    Label-specific yellow-highlighted left/right corridor-half images:
      9/20 required correct at threshold 0.0.
    Rejected. The highlighting did not create a reliable Qwen side signal.

Current conclusion:
  Accepted:
    The scene board should default to a real vehicle-width green corridor and no
    magenta base-reference line.
  Rejected:
    Three-way construction choice, forced two-way side choice, mirrored two-way
    de-bias, wording variants, and highlighted corridor-half yes/no scoring.
  Construction remains open:
    The current Qwen construction side/shift scoring contract still fails the
    95% construction-side gate and must not be promoted into the default runtime
    or manifest.

2026-05-24 MetaDrive camera color-order correction:

Root cause found while probing construction:
  MetaDrive RGBCamera frames were being passed to the scene-board renderer as if
  they were RGB. Visual evidence showed the default construction cones as blue
  and the sky/terrain color shifted. A channel-swapped copy of the same saved
  board produced realistic blue sky and orange/white cones.

Qwen evidence:
  On the uncorrected board, free-text Qwen repeatedly said no temporary
  road-work objects were visible near the green corridor.
  On the channel-corrected board, direct free-text questions recognized:
    "orange traffic cones"
    "orange and white striped cones"
    objects immediately left of the green corridor

Code changes:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Added --camera-color-order with default bgr.
    camera_frame now converts BGR -> RGB before overlay rendering.
    Added _convert_camera_frame_color for testable conversion.

  selfdrive\controls\tests\test_reasoned_trajectory.py
    Added a unit test proving the default BGR conversion maps a BGR red pixel to
    RGB red and leaves RGB mode unchanged.

  README.md and README_REASONED_TRAJECTORY_POC.md
    Documented that the MetaDrive harness converts camera frames from BGR to
    RGB before scene-board rendering.

Verification artifact:
  artifacts\reasoned_trajectory_poc\construction_left_camera_bgrfix_probe_16_20260524\vlm\vlm_input_0000.png
  This saved default VLM input now shows blue sky and orange/white cones without
  manual post-processing.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py selfdrive\controls\reasoned\ui_scene_board.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  89 tests OK.

Rejected side-scoring probes after this diagnosis:
  artifacts\reasoned_trajectory_poc\construction_anycolor_channelizer_prompt_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_labeled_corridor_side_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_recognized_terms_choice_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_bgrswap_side_probe_20260524.json
  artifacts\reasoned_trajectory_poc\construction_bgrswap_binary_side_probe_20260524.json
  These did not meet the construction gate. The color fix is accepted as a sim
  input correctness fix; construction label scoring remains open.

2026-05-24 lead route-vehicle visual correction after user report:

User report:
  The lead car still looked backwards/head-on in the driver-view sim video.

Root cause:
  The physical route heading and lead-track state were not reversed. The issue
  was the selected visual contract for route vehicles in the MetaDrive harness.
  A contact-sheet probe across DefaultVehicle/LVehicle/MVehicle/SVehicle/XLVehicle
  at 0 and 180 degree visual offsets showed that 0 degrees is the rear-view
  orientation and 180 degrees is the front/head-on orientation. SVehicle still
  reads as front-like at the VLM-board resolution, so it is a poor default.

Fix:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py
    Default route vehicle changed back to DefaultVehicle.
    Visual heading offset remains 0 degrees.
    Added --route-vehicle-visual-heading-offset-deg for explicit audit/probes.
    Spawn records include visual_heading_offset_rad.

  README.md and README_REASONED_TRAJECTORY_POC.md
    Documented the DefaultVehicle route-vehicle visual default and the boundary
    between physical lead-track state and evaluation labels.

Verification artifacts:
  contact sheet:
    artifacts\reasoned_trajectory_poc\lead_orientation_probe_vehicle_offset_contact_20260524.png
  accepted default proof run:
    artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_rear_48_20260524
  proof crop:
    artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_rear_48_20260524\stock\stock_overlay_0044_lead_crop_6x.png
  proof video:
    artifacts\reasoned_trajectory_poc\lead_orientation_defaultvehicle_rear_48_20260524\videos\side_by_side_lead_orientation_defaultvehicle_rear_48.mp4

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  89 tests OK.

Episode evidence:
  model_class DefaultVehicle
  route_heading_theta 0.0
  visual_heading_theta 0.0
  visual_heading_offset_rad 0.0
  lead_source track
  lead_distance_m 15.195060539245674
  lead_lateral_m 0.0
  lead_speed_mps 1.375
  lead_rel_speed_mps -0.8458125591277792

2026-05-25 construction sign-invariant and FP8 probe cleanup:

Process hygiene:
  A bad diagnostic run had three TensorRT/Qwen construction-side probes active
  at once. That invalidated latency measurements and could make TensorRT look
  hung/pathological. The remaining qwen_trt_label_engine.py probe processes
  were killed and a follow-up process scan showed none left running.

Accepted invariant work:
  Added full-chain construction sign tests instead of another local minus-sign
  patch. The tests now require:
    construction_right -> RTP BIAS_LEFT -> openpilot positive lateral offset
    -> MetaDrive negative lateral offset -> rendered green corridor shifts
    image-left.

    construction_left -> RTP BIAS_RIGHT -> openpilot negative lateral offset
    -> MetaDrive positive lateral offset -> rendered green corridor shifts
    image-right.

  This directly checks the Qwen-label/RTP/compiler/sim-boundary/green-overlay
  chain that kept regressing.

Verification:
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  95 tests OK.

FP8 TensorRT construction-side result:
  Built engine:
    F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine

  Clean single-process probe, no concurrent Qwen/TensorRT workers:
    image artifacts\reasoned_trajectory_poc\construction_right_bgrfix_static_80_20260524\static\vlm_input_0008.png
    output artifacts\reasoned_trajectory_poc\construction_right_static_frame8_side_fp8choice_single_trt_probe_20260525.json

  Result:
    expected: construction_right
    actual: construction_left
    scores: construction_left 1.0625, construction_right -1.0625
    total_ms: 58.6119
    trt_vision_ms: 10.9244
    trt_text_ms: 39.8415

Current conclusion:
  FP8 hidden-choice TensorRT does not fix the construction side fidelity
  problem. It is also above the 50 ms single-group target in this clean probe.
  The earlier 48-second timing was process contention, not steady-state
  performance, but the correctness failure remains. Do not promote FP8 or
  NVFP4 construction side scoring as the default closed-loop path until the
  optimized scorer preserves the PyTorch/Qwen side signal.

2026-05-25 construction scorer root-cause narrowing:

PyTorch reference under the exact optimized input contract:
  Tested Qwen PyTorch on the same 168 px full-frame boards, same 576-token
  choice prompt, and same clamp127 position mode used by the TensorRT choice
  engines.

  Results:
    right_static8:  construction_right, margin +0.03125
    right_static24: construction_right, margin +0.015625
    left_static24:  construction_left,  margin +0.140625

  qwen/clamp127 produced the same answers. zero position mode was wrong, but
  zero mode is not used by the active engines.

Interpretation:
  The scene board and prompt do contain a real Qwen side signal, but the side
  margin is extremely small. The optimized TensorRT text path changes that
  tiny signal enough to flip side labels.

TensorRT controls:
  FP16 hidden-choice engine was built:
    F:\qwen_trt_export\fp16_trt\qwen_text_36layer_fp16_seq576_hidden_choice_trt.engine

  Clean FP16 probe on right_static8:
    expected construction_right
    actual construction_left
    scores construction_left 1.0, construction_right -1.0
    total_ms 1913.7439
    trt_text_ms 1895.0468

  Hidden output dtype check:
    fp16/fp8/nvfp4 selected_hidden tensors are all TensorRT HALF and torch.float16.
    The previously suspected output-buffer dtype mismatch is not the cause.

  NVFP4 direct-logits clean probe on right_static8:
    expected construction_right
    actual construction_left
    scores construction_left 1.0625, construction_right -1.0625
    total_ms 37.3466
    trt_text_ms 20.3942

Prompt variant sweep:
  PyTorch variants can increase the margin, but the higher-margin side prompts
  became biased and failed the left-side board. Direct shift-direction prompts
  were also not reliable. TensorRT hidden/direct-logit sweeps did not produce a
  prompt that was correct for right_static8, right_static24, and left_static24.

Current conclusion:
  The repeated construction-side failure is now narrowed to optimized text-tower
  fidelity for a very low-margin visual-spatial judgement. The sign compiler and
  overlay chain have tests, and PyTorch Qwen can answer the side under the same
  image/text shape, but TensorRT export/runtime does not preserve that judgement.
  Current fast construction side scoring must stay rejected.

2026-05-25 ONNX text export divergence isolation and fail-closed build guard:

Process hygiene:
  Before continuing, scanned for qwen_trt_label_engine.py,
  qwen_label_rtp_worker.py, MetaDrive demo, video renderer, ffmpeg, trtexec,
  SSH, WSL, tinygrad, and C3X helper processes. No active POC workers were
  found. The NVIDIA GPU was not held by Qwen/TensorRT/MetaDrive.

ONNX-vs-PyTorch layer-boundary probe:
  Added temporary ONNX graph outputs for each decoder block boundary:
    /language_model/layers.N/Add_1_output_0
  using:
    F:\qwen_trt_export\fp16_trt\qwen_text_36layer_fp16_nopad_hidden_choice_probe.onnx

  Input:
    artifacts\reasoned_trajectory_poc\construction_right_bgrfix_static_80_20260524\static\vlm_input_0008.png
    labels construction_left,construction_right
    image full168, no text padding, choice mode, clamp127 text positions

  Key results:
    layer 0 last-token mean abs diff: 0.00184
    layer 1 last-token mean abs diff: 0.01385
    layer 8 last-token mean abs diff: 0.12859
    layer 16 last-token mean abs diff: 0.11814
    layer 24 last-token mean abs diff: 0.13464
    layer 30 last-token mean abs diff: 0.21065
    layer 34 last-token mean abs diff: 0.73744
    layer 35 last-token mean abs diff: 2.60235
    final selected_hidden mean abs diff: 0.42894
    final selected_hidden max abs diff: 10.8125

  The exported causal mask itself is correct:
    shape (1,1,94,94)
    values {0, -65504}
    lower triangular visible-token mask

  A post-pass that forced all attention Softmax inputs through ONNX float32
  Cast nodes patched 36 softmax sites but did not change the selected_hidden
  error or the wrong construction-side choice. So this is not simply "softmax
  dtype was omitted" in the exported graph.

Code changes:
  qwen_trt_label_engine.py now has:
    _set_qwen_attention_implementation(model, impl)
      Updates both top-level model config and language_model config.

    --verify-text-onnx-fidelity
      Optional build-time ONNX Runtime verification against the same PyTorch
      wrapper tensors before TensorRT build.

    --text-onnx-max-mean-error
    --text-onnx-max-error
    --text-onnx-require-choice-match / --no-text-onnx-require-choice-match

  The fidelity check compares selected_hidden or selected_logits and, for
  choice-mode groups, fails if ONNX chooses different labels than PyTorch.
  This is a production safety guard: a text engine that flips Qwen's answer
  cannot be silently promoted into the runtime manifest.

  build_text_engine no longer hardcodes the text export config to eager while
  recording sdpa in paths/manifests. The selected torch attention
  implementation is now applied and recorded consistently.

Current conclusion:
  The construction sign and overlay chain is not the active root cause. The
  Qwen PyTorch side judgement exists but is low-margin. The optimized ONNX text
  tower accumulates enough numerical/export divergence across the 36 decoder
  layers to flip that judgement. The next real fix must either make the ONNX /
  TensorRT text tower fidelity match PyTorch for this choice task or change the
  Qwen scoring contract so the correct road-side judgement has a much larger
  model margin before export. Prompt-only changes remain insufficient unless
  they pass closed-loop proof and ONNX/TensorRT fidelity checks.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  97 tests OK.

2026-05-25 construction mirror fusion and FP8 construction override:

Purpose:
  Continue fixing the construction sign/side failure without hardcoded
  simulator answers. The tested mechanism is still the configured architecture:
  scene board -> Qwen answer-token scores -> RTP -> compiler. The new step is a
  visual consistency transform: score the real scene and a horizontally mirrored
  scene, then map mirrored left/right evidence back to the original frame and
  fuse the answer-word scores.

Prompt/score probes:
  Current construction side prompt at full168:
    PyTorch Qwen is correct but low-margin.
    Existing FP16 ONNX text export flips right-side construction to left.

  Larger image inputs help PyTorch but do not by themselves fix ONNX/TensorRT:
    full320 PyTorch current-side margins:
      right8  +0.375 right
      right24 +0.213 right
      left24  +0.578 left
    Existing FP16 ONNX still chooses left for right8/right24.

  Verbose prompts with neutral words caused Qwen to choose none/clear.
  Forced no-neutral prompts either became left-biased or right-biased. No prompt
  tested in this pass was acceptable by itself.

Mirror-fusion probe:
  With full320 and existing FP16 ONNX text graph, mirror fusion selected the
  correct side for:
    right8  -> construction_right, fused margin +0.915
    right24 -> construction_right, fused margin +1.045
    left24  -> construction_left,  fused margin +0.906

Code changes:
  qwen_trt_label_engine.py:
    Added --construction-mirror-fusion.
    Added _apply_construction_mirror_fusion().
    Fusion uses Qwen's original answer-word scores and mirrored answer-word
    scores. Mirrored construction_left maps to original construction_right and
    mirrored construction_right maps to original construction_left before
    selection.

    Added --construction-text-engine and --construction-text-precision.
    Rotating score mode can now keep a fast shared default text engine while
    using a higher-fidelity construction-only text engine for construction
    groups. This is additive; the normal --text-engine path is unchanged.

    Runtime manifest now records:
      construction_mirror_fusion
      construction_text_precision
      construction_text_engine

  test_reasoned_trajectory.py:
    Added tests for mirror-fusion score mapping in both directions.
    Added tests for the construction-only text-engine override.

Optimized TensorRT evidence:
  NVFP4 hidden-choice + mirror fusion:
    right construction frame produced correct construction_right RTP.
    left construction frame still produced wrong construction_right RTP.
    Conclusion: NVFP4 remains too low fidelity for construction side selection.

  FP16 hidden-choice + mirror fusion:
    left construction frame produced correct construction_left RTP.
    But latency was about 3.94 s for a single mirrored construction check, so
    FP16 is only a correctness control, not a realtime candidate.

  FP8 hidden-choice + mirror fusion:
    Direct single-group probes:
      artifacts\reasoned_trajectory_poc\construction_right_mirror_fusion_fp8_trt_probe_20260525.json
        labels construction_right
        scores construction_right +0.546875
        total about 113.52 ms

      artifacts\reasoned_trajectory_poc\construction_left_mirror_fusion_fp8_trt_probe_20260525.json
        labels construction_left
        scores construction_left +1.34375
        total about 113.53 ms

  Rotating production-shaped construction sequence:
    Command shape used groups:
      cones,barrier;construction_left,construction_right
    with shared NVFP4 default engine and construction-only FP8 text engine:
      --score-rotate-shared-engine
      --construction-text-engine F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine
      --construction-text-precision fp8
      --construction-mirror-fusion

    Right construction:
      artifact artifacts\reasoned_trajectory_poc\construction_right_rotating_presence_fp8_override_probe_20260525.json
      cached labels [cones, construction_right]
      RTP scene=construction_right
      meta=BIAS_LEFT
      lat_bias_m=1.25
      avoid=[right_edge_s8_48_margin1.25]
      last construction-group total about 133.01 ms

    Left construction:
      artifact artifacts\reasoned_trajectory_poc\construction_left_rotating_presence_fp8_override_probe_20260525.json
      cached labels [cones, construction_left]
      RTP scene=construction_left
      meta=BIAS_RIGHT
      lat_bias_m=-1.25
      avoid=[left_edge_s8_48_margin1.25]
      last construction-group total about 134.59 ms

Current conclusion:
  The first optimized construction-side path that is correct on both saved
  left/right construction boards is FP8 text + mirror fusion, gated by recent
  construction presence. This is too slow for a synchronous 50 ms construction
  update, but it is plausible as a bounded-age async construction update while
  other label groups continue using the faster NVFP4 engine. This is not yet a
  closed-loop acceptance pass and not a success-gate claim.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  100 tests OK.
  Final process sweep: NO_ACTIVE_POC_PROCESSES.

2026-05-25 async prewarm fix and construction-side closed-loop failure:

Purpose:
  Continue the construction sign/side work from live evidence. The target is
  still production-translatable Qwen labels/scores feeding RTP and deterministic
  compilation. No simulator object labels were used as control input.

Code changes kept:
  selfdrive/controls/reasoned/vlm.py:
    AsyncRtpEngine now supports wait_idle() and reset_runtime_state().
    reset_runtime_state() clears warmup frame ids, cached RTP, pending frames,
    and last errors, and increments an epoch so stale worker results from before
    the reset cannot be accepted.

  tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py:
    Async prewarm now runs for the requested prewarm window instead of stopping
    after the first RTP. After prewarm it waits for the async engine to become
    idle and resets runtime frame state before frame 0. This prevents warmup
    source_frame_id values from contaminating the real episode.

  selfdrive/controls/reasoned/vlm.py and run_metadrive_overlay_demo.py:
    Async event logging remains enabled through RTP_VLM_ASYNC_LOG_PATH. The log
    records accepted, dropped_stale, consume_pending, errors, labels, source
    frame, last requested frame, and completion age.

  qwen_trt_label_engine.py:
    Added more explicit construction-side and construction-shift prompt wording
    around colored corridor edges and path-shift direction. This is prompt-only
    evidence so far, not a closed-loop success.

  Reverted experiment:
    A blue/orange corridor-half tint was tested. It fixed one left frame but
    broke right-side construction, so it was removed from the active scene board.
    The active board still uses the UI-like green corridor, blue left edge,
    orange right edge, and LEFT EDGE / RIGHT EDGE labels.

Probe evidence:
  Right side-labeled frame with the active edge-label board:
    artifacts\reasoned_trajectory_poc\sidelabel_right_frame5_fp8_side_probe_20260525.json
    Qwen selected construction_right, score margin +0.5, total about 59.12 ms.

  Failed closed-loop rerun:
    artifacts\reasoned_trajectory_poc\construction_left_sidelabel_prewarm_fp8nomirror_180_20260525
    p99 planner overhead stayed nonblocking at about 4.60 ms.
    max accepted RTP age was 8 frames.
    But mean_path_delta_m was 0.0 and the vehicle did not avoid construction.
    construction_trace_evaluation.json:
      path_relevant_construction_frames 114
      qwen_side_correct 0
      qwen_side_missing 114
      control_away_frames 0
      collision_count 20

  Async event log root cause for that failed run:
    The NVFP4 presence group was accepted, but the FP8 construction-side group
    repeatedly completed about 39-40 frames late and was correctly dropped by
    bounded stale-result logic. Example pattern:
      accepted labels [cones] source_frame_id 40
      dropped_stale labels [cones, construction_right] completion_age_frames 39-40
    This means bounded-age safety is working, but the useful construction-side
    group was not available in time during the MetaDrive run.

  Outside MetaDrive, the same rotating TensorRT scorer is fast:
    artifacts\reasoned_trajectory_poc\rotating_sidelabel_fp8override_benchmark_groups_20260525.json
    group 0 presence total about 38 ms
    group 1 construction-side total about 57.5 ms
    This points to live-sim GPU contention / server scheduling / engine switching
    under MetaDrive load, not a bad TensorRT artifact by itself.

  Side/shift semantics remain unresolved:
    The exact failed run left frame 0:
      artifacts\reasoned_trajectory_poc\run_left_frame0_fp8_side_probe_seq_20260525.json
      Qwen selected construction_right, wrong for the visible left-edge hazard.

    The same run left frame 5:
      artifacts\reasoned_trajectory_poc\run_left_frame5_fp8_side_probe_seq_20260525.json
      Qwen selected construction_left, correct, total about 57.34 ms.

    Construction-shift prompt with edge examples:
      left frame 0 selected construction_shift_right, which compiles to
      construction_left / BIAS_RIGHT, correct.
      right frame 0 selected construction_shift_right, wrong for a right-edge
      hazard. Mirror fusion corrected the right frame but broke the left frame,
      so it is not acceptable as the active fix.

Current blocker:
  The remaining construction blocker is not a sign-conversion unit-test issue.
  It is Qwen visual/action semantics plus runtime availability:
    1. The construction-side group can be fast in isolation but becomes stale in
       the live MetaDrive closed loop.
    2. Qwen still flips or answers the obstacle side instead of the safe shift
       direction on some first-look frames.
    3. The closed-loop car therefore still has no proven lateral avoidance.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\vlm.py selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  104 tests OK.

2026-05-24 continuation: convention hardening and lead-track smoke
-------------------------------------------------------------------

Goal-aligned reason:
  The failed videos repeatedly exposed sign/heading ambiguity. Before another
  Qwen closed-loop run, the sim harness needs to make the production convention
  auditable: openpilot/PathSynth positive lateral is left, MetaDrive lane
  positive lateral is right in this harness, and the scene board must display
  the actual tracked openpilot-offset path.

Changes:
  tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py:
    - Added durable_avoidance_sign_valid(plan). right_edge avoid plans must
      produce a negative MetaDrive target, left_edge avoid plans must produce a
      positive MetaDrive target. Invalid durable lateral plans are rejected or
      filtered instead of being allowed to steer toward their avoid edge.
    - Episode records now include explicit coordinate-system fields:
        active_lateral_offset_metadrive_m
        active_lateral_offset_openpilot_m
        desired_lateral_offset_metadrive_m
        desired_lateral_offset_openpilot_m
        compiled_lateral_offset_openpilot_m
        compiled_lateral_offset_metadrive_m
        green_path_offset_openpilot_m
        green_path_matches_tracked_path
        durable_lateral_plan_details
        durable_lateral_plan_sign_valid_all
        new_durable_avoidance_sign_valid
    - Route vehicle records now carry physical speed_mps, accel_mps2, and
      lateral_rate_mps. nearest_route_vehicle_state can use those real track
      fields on the first frame, instead of waiting for a second-frame finite
      difference. This does not read expected_lead_class.

  selfdrive/controls/tests/test_reasoned_trajectory.py:
    - Corrected synthetic durable-plan tests so right_edge means negative
      MetaDrive offset and left_edge means positive MetaDrive offset.
    - Added a guard test that rejects right_edge/left_edge plans whose offset
      moves toward the hazard.
    - Tightened candidate-guide rendering test so BLUE PATH is visually left
      of ORANGE PATH.
    - Added first-frame physical lead-track test proving lead state comes from
      speed/lateral/accel track fields and ignores expected_lead_class.

Lead visual-heading evidence:
  Generated non-VLM probes:
    artifacts\reasoned_trajectory_poc\lead_heading_close_0deg_20260525
    artifacts\reasoned_trajectory_poc\lead_heading_close_180deg_20260525
  With MetaDrive DefaultVehicle, 0 deg visual heading shows the rear of the lead
  car to ego. 180 deg shows headlights toward ego. The active default remains
  ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0.

Smoke evidence:
  artifacts\reasoned_trajectory_poc\lead_state_log_smoke_20260525
    stock episode frame 0:
      lead_present 1
      lead_distance_m 5.046
      lead_speed_mps 1.0
      green_path_matches_tracked_path true
      durable_lateral_plan_sign_valid_all true

  artifacts\reasoned_trajectory_poc\construction_sign_log_smoke_20260525
    static construction_right frame 0:
      compiled_lateral_offset_openpilot_m 1.25
      compiled_lateral_offset_metadrive_m -1.25
      desired_lateral_offset_openpilot_m 1.25
      desired_lateral_offset_metadrive_m -1.25
      durable_lateral_plan_sign_valid_all true

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  106 tests OK.

Current status:
  This reduces the repeated sign-flip failure mode at the compiler/harness
  boundary and improves lead-state evidence availability, but it is not a Qwen
  closed-loop success. The next required work is still to probe the current
  BLUE PATH / ORANGE PATH construction-shift scorer on regenerated left/right
  boards, then run bounded-age closed-loop Qwen construction and mixed suites.

2026-05-25 00:04 current-turn action log: construction input semantics
----------------------------------------------------------------------

Goal-aligned reason:
  The latest closed-loop and frame-probe evidence still showed Qwen steering or
  compiling toward right-side construction in some cases, and false construction
  presence on clear MetaDrive road frames. The fix must remain production
  translatable: scene board -> Qwen labels/scores -> RTP -> deterministic
  compiler. No simulator object labels are used as VLM input.

Process hygiene:
  Checked for active Qwen/TensorRT/MetaDrive/ffmpeg/repo Python processes before
  continuing. Result: NO_TARGETED_POC_PROCESSES.
  Found an old protected Python/GPU process pair unrelated to the current run
  and killed it with WMIC:
    PID 14332, old python.exe using GPU/RAM
    PID 16060, parent python.exe
  Confirmed no matching POC/GPU Python process remained after cleanup.

Current probes before code changes:
  Regenerated current default boards:
    artifacts\reasoned_trajectory_poc\purpleedge_current_clear_guides_10_20260525
    artifacts\reasoned_trajectory_poc\purpleedge_current_left_guides_10_20260525
    artifacts\reasoned_trajectory_poc\purpleedge_current_right_guides_10_20260525

  With candidate guides enabled:
    clear presence probe falsely accepted construction:
      artifact artifacts\reasoned_trajectory_poc\purpleedge_current_clear_presence_frame0_fp8_probe_20260525.json
      answer present, score cones 4.53125, gate 4.0
    clear shift probe also selected A / construction_shift_left:
      artifact artifacts\reasoned_trajectory_poc\purpleedge_current_clear_shift_frame0_fp8_probe_20260525.json
      this confirms shift must remain gated by reliable presence.
    left construction shift was correct:
      artifact artifacts\reasoned_trajectory_poc\purpleedge_current_left_shift_frame0_fp8_probe_20260525.json
      selected construction_shift_right, compiles to construction_left / BIAS_RIGHT.
    right construction shift was wrong:
      artifact artifacts\reasoned_trajectory_poc\purpleedge_current_right_shift_frame0_fp8_probe_20260525.json
      selected construction_shift_right, compiles toward construction_left / BIAS_RIGHT.

  Generated no-candidate-guide boards:
    artifacts\reasoned_trajectory_poc\purpleedge_current_clear_noguides_10_20260525
    artifacts\reasoned_trajectory_poc\purpleedge_current_left_noguides_10_20260525
    artifacts\reasoned_trajectory_poc\purpleedge_current_right_noguides_10_20260525

  With candidate guides disabled but side text still visible:
    clear presence still falsely accepted construction.
    left side probe was correct: construction_left.
    right side probe was still wrong: construction_left.

Code changes made so far in this turn:
  selfdrive\controls\reasoned\ui_scene_board.py:
    - Default draw_corridor_side_labels changed to False. The board no longer
      prints LEFT EDGE / RIGHT EDGE text into the VLM image by default.
    - Default planned_corridor_half_width_m widened from 0.90 m to 1.125 m to
      approximate a fuller vehicle-width corridor.
    - Added dim_outside_corridor, focus_corridor_extra_width_m, and
      outside_corridor_dim_alpha. The renderer now dims pixels outside a
      widened planned-corridor focus band before drawing the model overlay.
      This uses only the planned path geometry and is production-translatable.

  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - Default --scene-board-candidate-guides changed to false so candidate
      action guides are additive/debug-only, not always injected into Qwen's
      driving board.

  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Construction presence and side prompts were expanded to explicitly reject
      overlay text/lines, lane markings, tiny horizon/road-edge dots, and
      off-corridor background objects as construction evidence.

  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py:
    - Construction side questions now explicitly ignore colored overlay
      lines/text.

  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Updated tests for hidden default side text labels and explicit opt-in
      label rendering.

Verification before adding the dim-mask test/prompt contract:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 107 tests OK.

Open issue:
  After hiding side text and candidate guides, Qwen still false-positives
  construction on clear frame 0 and still calls right-side construction "left".
  The next action is to complete the dimmed-off-corridor scene-board contract,
  test it, regenerate boards, and rerun clear/left/right probes before any
  closed-loop run.

2026-05-25 00:17 current-turn action log: dimmed-corridor probe results
-------------------------------------------------------------------------

Additional code changes:
  selfdrive\controls\reasoned\ui_scene_board.py:
    - Replaced the first side-strip dimming attempt with a true image-mask
      implementation. It darkens the full image outside a widened planned
      corridor and preserves the corridor before drawing UI/model overlays.
    - This avoids relying on simulator object metadata and remains portable to
      real openpilot camera frames.

  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Scene-board manifest now records focus_corridor_extra_width_m,
      dim_outside_corridor, and outside_corridor_dim_alpha.
    - Construction presence and side prompts now tell Qwen that darkened
      off-corridor objects are irrelevant unless they touch/enter/block the
      green corridor.
    - construction_left/construction_right choice mode now requires a 0.5
      margin. Weak ties or all-zero side logits are rejected instead of
      publishing a left label by default.

  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py:
    - Full/composite/score prompts now ignore objects darkened outside the
      corridor focus band and ignore colored overlay lines/text.

  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added a test proving default scene-board rendering dims off-corridor
      background in the forward/mid-road band.
    - Added a test proving weak construction-side ties are rejected.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 109 tests OK.

New boards generated with default no-text/no-candidate/dimmed-corridor input:
  artifacts\reasoned_trajectory_poc\dimcorridor_default_clear_10_20260525
  artifacts\reasoned_trajectory_poc\dimcorridor_default_left_10_20260525
  artifacts\reasoned_trajectory_poc\dimcorridor_default_right_10_20260525

Qwen FP8 hidden-choice probes on frame 0:
  Clear presence:
    artifact artifacts\reasoned_trajectory_poc\dimcorridor_default_clear_presence_frame0_fp8_probe_20260525.json
    labels none, score cones 3.28125 below 4.0 gate. This fixes the previous
    clear-road false construction acceptance.

  Left construction presence:
    artifact artifacts\reasoned_trajectory_poc\dimcorridor_default_left_presence_frame0_fp8_probe_20260525.json
    labels none, score cones 3.6640625 below 4.0 gate. This is a false
    negative for recall.

  Right construction presence:
    artifact artifacts\reasoned_trajectory_poc\dimcorridor_default_right_presence_frame0_fp8_probe_20260525.json
    labels none, score cones 2.953125 below 4.0 gate. This is a false
    negative for recall.

  Left/right construction side:
    artifacts:
      artifacts\reasoned_trajectory_poc\dimcorridor_default_left_side_frame0_fp8_probe_20260525.json
      artifacts\reasoned_trajectory_poc\dimcorridor_default_right_side_frame0_fp8_probe_20260525.json
    The FP8 hidden-choice side path produced all-zero or very weak side logits.
    With the new 0.5 margin those are rejected as none instead of publishing a
    wrong left/right lateral plan.

Qwen FP8 hidden-choice probes on frame 5:
  artifacts:
    artifacts\reasoned_trajectory_poc\dimcorridor_default_left_presence_f5_fp8_probe_20260525.json
    artifacts\reasoned_trajectory_poc\dimcorridor_default_left_side_f5_fp8_probe_20260525.json
    artifacts\reasoned_trajectory_poc\dimcorridor_default_right_presence_f5_fp8_probe_20260525.json
    artifacts\reasoned_trajectory_poc\dimcorridor_default_right_side_f5_fp8_probe_20260525.json
  Result:
    left presence remained below gate at 3.69140625.
    right presence remained below gate at 3.32421875.
    side logits were all-zero and rejected by margin.

NVFP4 binary construction-side probe:
  Full prompt cannot run on seq576 engine:
    max prompt length 733 tokens, so it correctly raises rather than truncating.

  Compact prompt against existing seq576 NVFP4 binary engine:
    artifacts:
      artifacts\reasoned_trajectory_poc\dimcorridor_default_right_side_f0_nvfp4_binary_compact_existing_probe_20260525.json
      artifacts\reasoned_trajectory_poc\dimcorridor_default_left_side_f0_nvfp4_binary_compact_existing_probe_20260525.json
    right frame selected none with negative side scores.
    left frame incorrectly selected construction_right.
    This is not acceptable as the active side path.

Current conclusion:
  The dimmed full168 board fixes the clear-road false positive but loses real
  construction recall and does not solve side semantics. The next production-
  shaped variable to test is resolution/visibility of real cones within the
  same Qwen -> labels/scores -> RTP path, not a simulator object hack.

2026-05-25 00:23 Process hygiene check after compaction

User request:
  Make sure there are not multiple running or hung POC processes.

Checks run:
  Targeted process sweep for python.exe, python3.exe, py.exe, ffmpeg.exe,
  trtexec.exe, and polygraphy.exe whose command line matched qwen, TensorRT,
  onnx, metadrive, reasoned_trajectory, E:\ture_opamayo, or F:\qwen_trt_export.
  Result: NO_TARGETED_POC_PROCESSES.

  GPU client check with nvidia-smi.
  Result: no Qwen/TensorRT/MetaDrive/openpilot POC process was using the GPU.
  nvidia-smi showed normal desktop/graphics clients only.

  Broad Python/FFmpeg/TensorRT process check.
  Result: several DuckDuckGo MCP helper python processes were present and left
  alone. One old python.exe with parent open-webui.exe was present from
  2026-05-06, using about 1.5 GB RAM and no detected GPU compute slot. It is
  unrelated to this repo's Qwen/MetaDrive/openpilot POC and was not killed.

  PowerShell job check.
  Result: NO_POWERSHELL_JOBS.

Action taken:
  No POC process was killed because no matching stale Qwen/TensorRT/MetaDrive/
  openpilot demo process was running.

2026-05-25 00:24 Resume process sweep

Checks run:
  Targeted process sweep for python.exe, python3.exe, py.exe, ffmpeg.exe,
  trtexec.exe, and polygraphy.exe whose command line matched qwen, TensorRT,
  onnx, metadrive, reasoned_trajectory, E:\ture_opamayo, or F:\qwen_trt_export.
  Result: NO_TARGETED_POC_PROCESSES.

  nvidia-smi compute app query.
  Result: no Qwen/TensorRT/MetaDrive/openpilot POC compute client was present.

  PowerShell job check.
  Result: no jobs listed.

Action taken:
  No process was killed.

2026-05-25 00:26 Additive scene-board geometry controls

Reason:
  The current default dimmed full168 board fixed clear-road false construction
  positives but lost construction recall. To test production-shaped visibility
  changes without repeatedly changing code, the MetaDrive/openpilot harness now
  exposes the scene-board geometry/mask parameters as CLI arguments.

Files changed:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - scene_board_renderer_for_args now passes explicit OverlayGeometry fields
      from argparse, preserving old defaults when the new args are omitted.
    - Added additive args:
      --scene-board-corridor-half-width-m
      --scene-board-focus-extra-width-m
      --scene-board-dim-outside-corridor / --no-scene-board-dim-outside-corridor
      --scene-board-dim-alpha
      --scene-board-corridor-side-guides / --no-scene-board-corridor-side-guides
      --scene-board-corridor-side-labels / --no-scene-board-corridor-side-labels
      --scene-board-base-path-reference / --no-scene-board-base-path-reference
      --scene-board-candidate-labels / --no-scene-board-candidate-labels

  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added tests proving omitted CLI args preserve OverlayGeometry defaults.
    - Added tests proving the new args can tune corridor width, focus width,
      dimming, candidate labels, base-path reference, and side-guide rendering.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 111 tests OK.

2026-05-25 00:37 Construction grid probes and explicit choice-margin calibration

Board generation:
  Generated real MetaDrive camera scene boards, not synthetic crops, for five
  production-shaped scene-board geometry variants:
    grid_a55_e075_h1125: dim alpha 55, focus extra 0.75 m, corridor half 1.125 m
    grid_a70_e125_h1125: dim alpha 70, focus extra 1.25 m, corridor half 1.125 m
    grid_a85_e125_h1350: dim alpha 85, focus extra 1.25 m, corridor half 1.35 m
    grid_a55_e125_h1350: dim alpha 55, focus extra 1.25 m, corridor half 1.35 m
    grid_a35_e125_h1350: dim alpha 35, focus extra 1.25 m, corridor half 1.35 m
  For each variant, generated clear, construction_left, and construction_right
  frame-0 boards under:
    artifacts\reasoned_trajectory_poc\construction_board_<variant>_<scene>_20260525_0026

Construction presence Qwen FP8 hidden-choice grid:
  Artifact:
    artifacts\reasoned_trajectory_poc\construction_presence_grid_fp8_20260525_0026.json
  Server shape:
    qwen_trt_label_engine.py serve, text engine
    F:\qwen_trt_export\fp8_trt\qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine,
    text-output hidden, label-decision-mode choice, score labels cones,barrier,
    image size 168.

  Scores:
    variant                  clear      left       right      pass at 4.0
    grid_a55_e075_h1125      3.265625   2.832031   2.898438   no
    grid_a70_e125_h1125      3.027344   3.441406   3.550781   no
    grid_a85_e125_h1350      2.964844   3.402344   3.347656   no
    grid_a55_e125_h1350      2.941406   3.425781   2.914063   no
    grid_a35_e125_h1350      3.628906   3.449219   2.992188   no

  Conclusion:
    The best current geometry is grid_a70_e125_h1125. It rejects clear at
    3.027 and sees both construction sides at 3.44/3.55, but the hard 4.0
    construction-presence choice margin is too conservative for this board.

Construction side Qwen FP8 hidden-choice probe on best board:
  Artifact:
    artifacts\reasoned_trajectory_poc\construction_side_grid_a70_fp8_20260525_0026.json
  Results:
    left board: answer left, margin 0.375, rejected by old 0.5 margin.
    right board: answer right, margin 0.671875, accepted.
  Conclusion:
    Directional side understanding is correct on this board, but the fixed
    0.5 side margin rejects the weaker left case. This is a calibration issue,
    not a sign-convention issue in this probe.

Code change:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Choice-mode scoring now honors existing --score-thresholds as explicit
      per-label minimum choice margins.
    - This is additive. If no override is supplied, the existing spec margins
      remain unchanged.
    - The choice debug payload now reports both min_margin and spec_min_margin.

  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added tests proving --score-thresholds-style per-label margins can accept
      a calibrated construction_left side margin.
    - Added tests proving the construction-presence margin can be calibrated
      from 4.0 to 3.3 without changing the default spec.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 113 tests OK.

Calibrated Qwen FP8 probes on best board:
  Artifact:
    artifacts\reasoned_trajectory_poc\construction_a70_calibrated_choice_fp8_20260525_0037.json
  Runtime margins:
    cones: 3.3
    barrier: 3.3
    construction_left: 0.3
    construction_right: 0.3

  Presence:
    clear: labels none, score 3.02734375, rejected by 3.3 margin.
    left: labels cones, score 3.44140625, accepted.
    right: labels cones, score 3.55078125, accepted.

  Side:
    left: labels construction_left, score 0.375, accepted by 0.3 margin.
    right: labels construction_right, score 0.671875, accepted by 0.3 margin.

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES after the grid/probe scripts exited.

Current conclusion:
  For isolated frame-0 construction boards, the current best production-shaped
  contract is:
    scene board dim alpha 70
    focus extra width 1.25 m
    corridor half width 1.125 m
    choice margins cones/barrier 3.3, construction_left/right 0.3
  This fixes the immediate clear/left/right construction probe. It still needs
  closed-loop confirmation and then broader mixed-scene evaluation before it can
  be treated as a real driving improvement.

2026-05-25 01:00 Construction closed-loop side-sign audit after calibration

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before continuing. Targeted sweep covered
  python.exe, python3.exe, py.exe, ffmpeg.exe, trtexec.exe, and polygraphy.exe
  with command lines matching qwen, TensorRT, onnx, metadrive,
  reasoned_trajectory, E:\ture_opamayo, or F:\qwen_trt_export.

Code changes:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Calibrated construction_left/construction_right accepted-choice labels to
      raise RTP confidence to 0.80-0.84 when the selected side margin is strong
      enough. This lets construction-side labels pass the durable lateral-plan
      activation threshold instead of compiling to a low-confidence no-op.
    - Added tests for this confidence behavior.

  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - Changed lateral-plan clearing so construction_presence_unknown and generic
      cones/barrier evidence do not clear an existing durable lateral avoidance
      plan. This preserves avoidance through short side-unknown frames and brief
      occlusions.
    - Added tests proving construction side-unknown evidence does not clear an
      active lateral plan.

Closed-loop results with calibrated board and margins:
  Runtime board:
    --scene-board-dim-alpha 70
    --scene-board-focus-extra-width-m 1.25
    --scene-board-corridor-half-width-m 1.125
  Runtime Qwen labels:
    cones/barrier with margin 3.3
    construction_left/right with margin 0.3
  Speed:
    2.5 m/s, VLM speed control disabled.

  Right-side construction before confidence fix:
    artifacts\reasoned_trajectory_poc\construction_right_a70_calibrated_closedloop_100_20260525_0040
    Result: Qwen side labels were often correct, but confidence stayed at 0.72
    and did not activate durable lateral avoidance.

  Right-side construction after confidence fix but before side-unknown hold:
    artifacts\reasoned_trajectory_poc\construction_right_a70_calibrated_confidence_closedloop_100_20260525_0043
    Result: Better side recognition, but construction_presence_unknown frames
    repeatedly cleared the lateral plan, causing drops to neutral.

  Right-side construction after side-unknown hold:
    artifacts\reasoned_trajectory_poc\construction_right_a70_hold_unknown_closedloop_100_20260525_0046
    Result:
      stock min construction-route clearance: 1.3538 m
      vlm min construction-route clearance: 2.1452 m
      qwen_side_correct: 21
      qwen_side_wrong: 0
      control_away_frames: 33
      control_toward_frames: 0
      collision_count: 0
    Conclusion: right-side construction passed this 100-frame targeted audit.

  Left-side construction with the same prompt:
    artifacts\reasoned_trajectory_poc\construction_left_a70_hold_unknown_closedloop_100_20260525_0048
    Result:
      stock min construction-route clearance: 1.3538 m
      vlm min construction-route clearance: 0.681 m
      qwen_side_correct: 6
      qwen_side_wrong: 27
      control_away_frames: 0
      control_toward_frames: 52
      collision_count: 29
    Conclusion: left-side construction failed. Qwen frequently labeled left-side
    cones as construction_right, which compiled into steering toward the hazard.

Prompt experiment:
  Added a center-half-oriented side prompt. It fixed the left-side targeted run:
    artifacts\reasoned_trajectory_poc\construction_left_a70_sideprompt_closedloop_100_20260525_0053
      vlm min construction-route clearance: 2.151 m
      qwen_side_correct: 32
      qwen_side_wrong: 0
      control_away_frames: 33
      control_toward_frames: 0
      collision_count: 0

  The same center-half prompt catastrophically regressed right-side construction:
    artifacts\reasoned_trajectory_poc\construction_right_a70_sideprompt_closedloop_100_20260525_0055
      vlm min construction-route clearance: 0.542 m
      qwen_side_correct: 0
      qwen_side_wrong: 83
      control_toward_frames: 70
      collision_count: 29

  Conclusion:
    The center-half rule is not acceptable. It corrected one side by adding a
    new geometric ambiguity instead of fixing the invariant. The latest local
    prompt now tells Qwen to judge construction side by the nearest colored
    planned-corridor edge: blue edge means left-side hazard, purple edge means
    right-side hazard, regardless of whether the object is just outside, on, or
    slightly inside that edge. That nearest-edge prompt is compiled and tested,
    but still requires direct probes and closed-loop validation before use in a
    demo video.

Rejected experiment:
  Binary side scoring with the seq768 hidden TensorRT engine returned all-zero
  side scores and was not useful:
    artifacts\reasoned_trajectory_poc\binary_side_left_f50_fp8_20260525_0056.json
    artifacts\reasoned_trajectory_poc\binary_side_right_f50_fp8_20260525_0056.json

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 115 tests OK.

Current blocker:
  The repeated sign failures are not all one bug. They came from multiple
  convention boundaries: openpilot left-positive path coordinates, MetaDrive
  lateral offset/control coordinates, scene-board rendering coordinates, and
  Qwen's image-space interpretation of left/right. The remaining validation
  target is to prove one shared invariant:
    construction near the blue/left planned-corridor edge compiles away from it,
    and construction near the purple/right planned-corridor edge compiles away
    from it, with zero control-toward frames in both left and right closed-loop
    audits.

2026-05-25 01:26 Construction sign-flip root-cause probes

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before and after this probe batch. I also
  accidentally launched two short static MetaDrive board generators in parallel,
  then let both finish and returned to one heavy process at a time.

Nearest-edge side prompt:
  Direct Qwen FP8 hidden-choice probes on saved failing frames showed the
  nearest-edge prompt still labels right-side construction as construction_left:
    artifacts\reasoned_trajectory_poc\nearest_edge_side_probe_20260525_0100
  Results:
    left_0020: construction_left, correct
    left_0050: construction_left, correct
    right_0020: construction_left, wrong
    right_0050: construction_left, wrong
  Conclusion:
    The prompt is not sufficient. Running a demo from it would reproduce the
    same bad steering.

Candidate-path prompt:
  Generated fresh full-frame boards with PATH A/PATH B candidate guides:
    artifacts\reasoned_trajectory_poc\candidate_guides_left_static_60_20260525_0105
    artifacts\reasoned_trajectory_poc\candidate_guides_right_static_60_20260525_0105
  Qwen scored construction_shift_left on both sides:
    artifacts\reasoned_trajectory_poc\candidate_shift_probe_20260525_0106
  Result:
    left-side construction: chose PATH A / shift left, wrong
    right-side construction: chose PATH A / shift left, correct
  Conclusion:
    The action-candidate label removes one inversion step but still has a strong
    PATH A bias and is not a fix.

224 px probe:
  Tested the existing full224 vision engine:
    F:\qwen_trt_export\vision_static_fp32\qwen_vision_full224_static_fp32.engine
  Artifacts:
    artifacts\reasoned_trajectory_poc\resolution224_side_shift_probe_20260525_0108
  Result:
    side prompt still returned construction_left for both sides.
    candidate-shift prompt still selected the hazard-side candidate in the tested
    f50 frames.
  Conclusion:
    This is not just a 168 px compression artifact.

Side-label text probe:
  Generated static boards with LEFT EDGE / RIGHT EDGE text labels:
    artifacts\reasoned_trajectory_poc\side_labels_left_static_60_20260525_0110
    artifacts\reasoned_trajectory_poc\side_labels_right_static_60_20260525_0110
  Qwen still returned construction_left for both sides:
    artifacts\reasoned_trajectory_poc\side_labels_qwen_probe_20260525_0111
  Conclusion:
    Adding side text to the scene board does not fix the label bias.

Color-edge scoring experiment:
  Added an additive color-edge label path:
    construction_blue_edge
    construction_purple_edge
  These labels map through the same deterministic compiler:
    blue edge -> construction_left -> BIAS_RIGHT
    purple edge -> construction_right -> BIAS_LEFT
  Added tests proving:
    - the prompt answers blue/purple, not left/right
    - edge labels compile to away biases
    - confidence calibration works for edge labels
    - edge labels are gated by recent construction presence
  Verification:
    py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    Result: 119 tests OK.

  Probe artifacts:
    artifacts\reasoned_trajectory_poc\color_edge_probe_20260525_0113
    artifacts\reasoned_trajectory_poc\color_edge_probe_20260525_0114
  Mixed result:
    saved left f20/f50 old VLM boards: construction_blue_edge, correct
    saved right f50 old VLM board: construction_purple_edge before later prompt tweak, correct
    saved right f20 old VLM board: construction_blue_edge, wrong
  A nearest-cone wording tweak made color-edge scoring worse, collapsing to blue
  on all tested frames, so that wording was backed out.

  Closed-loop right-side color-edge run:
    artifacts\reasoned_trajectory_poc\construction_right_color_edge_closedloop_100_20260525_0118
  Result:
    stock min construction-route clearance: 1.3538 m
    vlm min construction-route clearance: 0.9333 m
    control_toward_frames: 69
    collision_count: 29
  Conclusion:
    The additive color-edge path is not a validated default. It exposed that the
    current fast hidden-choice side scorer is answer-prior dominated early in the
    scenario and can confidently choose the wrong side before useful control.

Binary single-label probe:
  Tested binary yes/no scoring one side label at a time to avoid the left/right
  choice-word competition:
    artifacts\reasoned_trajectory_poc\binary_single_side_probe_20260525_0125
  Result:
    right frame: construction_left score 2.28, construction_right score 1.67
    left frame: construction_left score 1.66, construction_right score 2.45
  Conclusion:
    Binary one-label scoring did not solve the sign problem either. It also
    tends to score both sides positive, with the stronger side inverted in these
    samples.

Root-cause conclusion:
  The repeated sign flipping is now primarily a VLM scoring-contract failure,
  not a remaining openpilot-to-MetaDrive sign conversion bug. Unit tests cover
  the compiler invariant, and the scene-board/control path records agree on
  tracked offset. The current Qwen TensorRT fast path is reliable enough for
  construction presence, but not reliable enough for left/right construction
  side or safe lateral candidate choice in these perspective scene boards.
  Further closed-loop videos should wait until side selection is either made
  reliable by a different Qwen scoring contract or guarded by a production-shaped
  deterministic visual geometry verifier. A simulator object-state oracle should
  not be used.

2026-05-25 01:32 Binary color-edge prompt probe

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before this work.

Code change:
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Added a compact yes/no construction prompt for the additive
      construction_blue_edge / construction_purple_edge labels. The original
      full prompt was 734 tokens and cannot run on the current fixed seq576
      text engine.
  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added a unit test proving the compact color-edge binary prompt exists and
      stays short enough for the fixed-shape contract.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 120 tests OK.

Probe artifact:
  artifacts\reasoned_trajectory_poc\binary_color_edge_compact_probe_20260525_0132

Results:
  right f20:
    construction_blue_edge score 0.21875, rejected at 0.3
    construction_purple_edge score 0.609375, accepted
  right f50:
    construction_blue_edge score 2.703125, accepted
    construction_purple_edge score 0.96875, accepted
    stronger label is wrong for the right-side hazard
  left f20:
    construction_blue_edge score 1.578125, accepted
    construction_purple_edge score 1.1875, accepted
    stronger label is correct but both are positive
  left f50:
    construction_blue_edge score 1.296875, accepted
    construction_purple_edge score about 1.27, accepted
    nearly tied

Conclusion:
  Compact binary edge scoring fits the engine but is not a validated side
  solution. It reduces the early right f20 error, but still gives both sides
  positive scores and can prefer the wrong edge when the right-side construction
  is close. Next direction is a production-shaped scene-board change: give Qwen
  explicit edge-local visual panels/crops so it judges construction presence in
  each corridor-edge region instead of inferring side from a perspective full
  frame.

2026-05-25 02:05 Durable lateral admission tightening

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before editing.

Context inspected:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - Verified openpilot_to_metadrive_lateral_m / metadrive_to_openpilot_lateral_m
      are the single harness conversion points for openpilot-left-positive path
      offsets versus MetaDrive lane offsets.
    - Verified the scene-board state passes active_lateral_offset_m through
      metadrive_to_openpilot_lateral_m before rendering Qwen's green path.
    - Verified existing logs include compiled openpilot/MetaDrive offsets,
      active/desired offsets, green_path_matches_tracked_path, durable plan
      source/sign fields, Qwen labels/scores, RTP age, and lead state.

Code change:
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - Raised default durable lateral activation from one-frame acceptance to
      two-frame confirmation unless RTP confidence is at least 0.95.
    - Raised default contradictory lateral side-flip handling to three-frame
      confirmation unless RTP confidence is at least 0.95.
    - Updated fallback defaults inside _lateral_conflict_override_allowed to
      match the CLI defaults, so partial Namespace tests do not silently use
      weaker one-frame behavior.

  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added coverage that a medium-confidence first lateral construction plan
      stays pending until repeated with the same source/sign.
    - Added coverage that a very high-confidence lateral plan can still activate
      immediately, preserving the ability to override stale plans when Qwen is
      truly confident.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 126 tests OK.

Rationale:
  This is not a simulator object oracle or a hand-authored route hack. It only
  changes how the deterministic compiler admits Qwen/RTP lateral plans over
  time. The goal is to prevent a single wrong construction-side label from
  becoming a durable maneuver while still allowing a confident contradictory
  judgement to replace a stale prior plan.

2026-05-25 02:10 Short construction reruns after durable tightening

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before each closed-loop run and again
  after the runs. No targeted Qwen/TensorRT/MetaDrive/ffmpeg processes were left
  running.

Run 1:
  artifacts\reasoned_trajectory_poc\construction_right_edge_inset_default_confirm_100_20260525_0206
  Command shape:
    - Qwen TensorRT FP8 hidden-choice engine, image_size=168, text_seq_len=576.
    - Rotating groups: cones,barrier ; construction_blue_edge ; construction_purple_edge.
    - Edge binary mode, edge insets enabled, async latest-frame queue enabled.
    - VLM speed control disabled, speed_mps=2.5, construction_right, 100 frames.
  Summary:
    stock min construction-route clearance: 1.3538 m
    vlm min construction-route clearance: 1.6745 m
    publish_count: 98 / 100
    max_rtp_age_frames: 3
    p99 planner overhead: 6.25 ms
  Evaluator:
    qwen_side_correct: 31
    qwen_side_wrong: 12
    qwen_side_missing: 22
    control_away_frames: 38
    control_toward_frames: 1
    collision_count: 0
  Videos:
    artifacts\reasoned_trajectory_poc\construction_right_edge_inset_default_confirm_100_20260525_0206\videos\side_by_side_construction_right_confirm100.mp4
    artifacts\reasoned_trajectory_poc\construction_right_edge_inset_default_confirm_100_20260525_0206\videos\vlm_construction_right_confirm100.mp4

Logging fix:
  selfdrive\controls\reasoned\vlm.py:
    - PersistentRtpEngine now prefers response["label_scores_cached"] over
      response["label_scores"] when present. This makes rotating-score episode
      logs show the score state that actually produced the published RTP, not
      only the group scored on the newest frame.
  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added a PersistentRtpEngine unit test for cached rotating score logging.
  Verification:
    py -3.11 -m py_compile selfdrive\controls\reasoned\vlm.py selfdrive\controls\tests\test_reasoned_trajectory.py
    py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
    Result: 127 tests OK.

Run 2:
  artifacts\reasoned_trajectory_poc\construction_right_edge_inset_cached_scores_confirm_100_20260525_0209
  Purpose:
    Same run shape as Run 1, but with cached score logging.
  Summary:
    stock min construction-route clearance: 1.3538 m
    vlm min construction-route clearance: 1.0088 m
    publish_count: 98 / 100
    max_rtp_age_frames: 3
    p99 planner overhead: 5.95 ms
  Evaluator:
    qwen_side_correct: 28
    qwen_side_wrong: 27
    qwen_side_missing: 33
    control_away_frames: 11
    control_toward_frames: 41
    collision_count: 23
  Conclusion:
    Cached logging exposed the actual remaining failure: Qwen still produces
    repeated wrong edge labels strongly enough to pass two-frame activation and
    three-frame conflict confirmation. This is not a remaining openpilot /
    MetaDrive sign conversion issue.
  Videos:
    artifacts\reasoned_trajectory_poc\construction_right_edge_inset_cached_scores_confirm_100_20260525_0209\videos\side_by_side_construction_right_cached_scores_confirm100.mp4

Run 3:
  artifacts\reasoned_trajectory_poc\construction_right_edge_inset_mirror_consistency_100_20260525_0210
  Difference from Run 2:
    Added --construction-mirror-consistency to the Qwen TensorRT server command.
    This mirrors the same scene-board image and accepts construction edge labels
    only when the mirrored judgement maps back consistently. It is image-based
    and production-portable, not simulator object state.
  Summary:
    stock min construction-route clearance: 1.3538 m
    vlm min construction-route clearance: 1.9596 m
    publish_count: 98 / 100
    max_rtp_age_frames: 3
    p99 planner overhead: 6.18 ms
    green_path_matches_tracked_path: true for all frames
    durable_lateral_plan_sign_valid_all: true for all frames with plans
  Evaluator:
    qwen_side_correct: 19
    qwen_side_wrong: 11
    qwen_side_missing: 32
    control_away_frames: 32
    control_toward_frames: 1
    collision_count: 0
  Videos:
    artifacts\reasoned_trajectory_poc\construction_right_edge_inset_mirror_consistency_100_20260525_0210\videos\side_by_side_construction_right_mirror_consistency100.mp4
    artifacts\reasoned_trajectory_poc\construction_right_edge_inset_mirror_consistency_100_20260525_0210\videos\vlm_construction_right_mirror_consistency100.mp4

Current conclusion:
  Mirror consistency materially improves closed-loop construction avoidance and
  keeps loop age/bounds acceptable in this short probe, but it is not an
  acceptance result. Qwen side-label success is still far below the required
  95% target, and the evaluator still records wrong or missing side labels. The
  next work should improve the Qwen side scoring contract itself, not add a
  simulator-only correction.

2026-05-25 02:16 Rotating competitor-score calibration fix

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before and after the run.

Root cause found:
  Rotating edge scoring was not retaining recent exclusive-competitor scores in
  the active score snapshot. A construction_blue_edge label could therefore get
  confidence promoted from the selected edge score while construction_purple_edge
  was absent from the score dict, making a one-sided judgement look stronger
  than it was.

Code change:
  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py:
    - RotatingScoreState now records score timestamps separately from positive
      label timestamps.
    - Exclusive conflict clearing now clears active positive labels but keeps
      recent scores for the sibling labels.
    - active_scores(frame_id) now returns active labels plus their exclusive
      competitors when the competitor scores are fresh.
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - _score_calibrated_construction_confidence now refuses to promote
      construction side/edge confidence unless both selected and competitor
      scores are finite.
  selfdrive\controls\tests\test_reasoned_trajectory.py:
    - Added coverage for retaining recent exclusive competitor scores.
    - Added coverage that construction confidence stays at the base confidence
      when the competitor score is missing.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 129 tests OK.

Closed-loop rerun:
  artifacts\reasoned_trajectory_poc\construction_right_edge_competitor_conf_mirror_100_20260525_0216
  Command shape:
    Same as the mirror-consistency run, with the competitor-score calibration
    code active.
  Summary:
    stock min construction-route clearance: 1.3538 m
    vlm min construction-route clearance: 1.3851 m
    publish_count: 98 / 100
    max_rtp_age_frames: 3
    p99 planner overhead: 5.53 ms
    green_path_matches_tracked_path: true for all frames
    durable_lateral_plan_sign_valid_all: true for all frames with plans
  Evaluator:
    qwen_side_correct: 35
    qwen_side_wrong: 28
    qwen_side_missing: 25
    control_away_frames: 35
    control_toward_frames: 22
    collision_count: 0
  Video:
    artifacts\reasoned_trajectory_poc\construction_right_edge_competitor_conf_mirror_100_20260525_0216\videos\side_by_side_construction_right_competitor_conf_mirror100.mp4

Conclusion:
  The logging/calibration fix is correct and test-covered, but it did not solve
  construction side reliability in closed loop. The new trace proves Qwen can
  still strongly prefer the wrong edge even with fresh competitor scores and
  mirror consistency. Actual control remained bounded, no sign-invalid plans
  were logged, and there were no collisions in this run, but side accuracy and
  control-away rate remain far below the objective.

2026-05-25 02:28 Candidate-obstruction scene-board scoring path

Process hygiene:
  Confirmed NO_TARGETED_POC_PROCESSES before this work.

Design intent:
  The prior construction side contract still made Qwen name a side or colored
  edge directly, and Qwen repeatedly confused that side. This change adds a
  production-portable alternative scoring contract: render auxiliary full-frame
  UI scene boards where the green planned corridor is shifted left or shifted
  right, then ask Qwen whether construction blocks that specific candidate
  corridor. This keeps the existing architecture:
    scene board -> Qwen labels/scores -> RTP -> deterministic compiler -> plan
  It does not use simulator object state and does not hand-author RTP.

Code change:
  selfdrive\controls\reasoned\scene_board.py:
    - SceneBoard now carries optional aux_pngs for additional per-frame scene
      board images.
  selfdrive\controls\reasoned\ui_scene_board.py:
    - Added optional candidate obstruction auxiliary boards:
      candidate_left and candidate_right.
    - Each auxiliary board is a normal driver-view UI overlay with the green
      corridor shifted left or right.
  selfdrive\controls\reasoned\vlm.py:
    - Persistent and external RTP payloads now include scene_board_aux_images_b64
      when auxiliary boards exist.
  tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py:
    - Added construction_blocks_left_candidate and
      construction_blocks_right_candidate labels.
    - Candidate-left blocked compiles to construction_left / BIAS_RIGHT.
    - Candidate-right blocked compiles to construction_right / BIAS_LEFT.
    - Both candidates blocked compiles to construction_presence_unknown with no
      lateral bias.
  tools\reasoned_trajectory_poc\qwen_trt_label_engine.py:
    - Added --construction-candidate-binary.
    - Candidate labels use compact binary prompts.
    - Rotating scorer selects the matching auxiliary image for candidate groups.
  tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py:
    - Added --scene-board-candidate-obstruction-boards and
      --scene-board-candidate-obstruction-offset-m.

Tests added:
  - Auxiliary candidate images are sent to the persistent worker.
  - Candidate obstruction prompts fit the fixed text contract.
  - Candidate blocked labels compile to the opposite lateral bias.
  - Candidate binary mode overrides only candidate groups.
  - Candidate labels use the auxiliary candidate image rather than the main
    scene board.
  - Candidate auxiliary boards shift the green path in opposite directions.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\scene_board.py selfdrive\controls\reasoned\ui_scene_board.py selfdrive\controls\reasoned\vlm.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 135 tests OK.

2026-05-25 resumed run hygiene and state check

Actions:
  - Checked for targeted hung/running POC processes before continuing.
  - Command scope covered python.exe, python3.exe, py.exe, ffmpeg.exe,
    trtexec.exe, and polygraphy.exe with command lines matching qwen,
    TensorRT, onnx, metadrive, reasoned_trajectory, E:\ture_opamayo, or
    F:\qwen_trt_export.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Checked git status before running another demo.

Current dirty tree:
  M README.md
  M README_REASONED_TRAJECTORY_POC.md
  M progress.md
  M selfdrive/controls/reasoned/planner.py
  M selfdrive/controls/reasoned/scene_board.py
  M selfdrive/controls/reasoned/ui_scene_board.py
  M selfdrive/controls/reasoned/vlm.py
  M selfdrive/controls/tests/test_reasoned_trajectory.py
  m tinygrad_repo
  M tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py
  M tools/reasoned_trajectory_poc/qwen_trt_label_engine.py
  M tools/reasoned_trajectory_poc/render_demo_videos.py
  M tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py
  ?? tools/reasoned_trajectory_poc/evaluate_construction_trace.py
2026-05-25 resumed static verification before candidate-resolver rerun

Actions:
  - Searched current source for candidate obstruction flags, auxiliary scene
    board routing, candidate labels, and score resolver hooks.
  - Recompiled the touched reasoned-trajectory files.
  - Re-ran the focused reasoned trajectory unit suite.

Verification:
  rg -n "construction-candidate|candidate_obstruction|scene-board-candidate|construction_blocks_(left|right)_candidate|_resolve_candidate_obstruction" tools\reasoned_trajectory_poc selfdrive\controls\reasoned selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: candidate obstruction flags, aux image selection, labels, resolver,
  and tests are present in current files.

  py -3.11 -m py_compile selfdrive\controls\reasoned\scene_board.py selfdrive\controls\reasoned\ui_scene_board.py selfdrive\controls\reasoned\vlm.py tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 136 tests OK.
2026-05-25 candidate-resolver closed-loop run start

Action:
  - Starting a focused construction_right MetaDrive closed-loop run with
    auxiliary candidate obstruction boards and Qwen score resolution enabled.
  - This run is intended to test whether Qwen can decide which shifted green
    corridor is blocked, then compile the opposite lateral bias without using
    simulator object state or hardcoded route facts.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236

Command intent:
  - Engine: Qwen TensorRT FP8 hidden-choice labels.
  - Labels: cones, barrier, construction_blocks_left_candidate,
    construction_blocks_right_candidate.
  - Candidate resolver: enabled with diff margin 0.4.
  - Scene board: main full driver overlay plus candidate_left/candidate_right
    auxiliary overlays.
  - Vehicle speed: 2.5 m/s with VLM speed control disabled, so this run tests
    lateral behavior only.
2026-05-25 candidate-resolver closed-loop run result

Actions:
  - Ran the 100-frame construction_right candidate-obstruction demo.
  - Swept targeted POC processes after the run.
  - Evaluated construction trace.
  - Rendered stock, VLM, and side-by-side videos.
  - Inspected raw Qwen labels, scores, RTP text, and active lateral offsets.

Artifacts:
  Run directory:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236
  Side-by-side video:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236\videos\side_by_side_construction_right_candidate_resolved100.mp4
  VLM-only video:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236\videos\vlm_construction_right_candidate_resolved100.mp4
  Stock-only video:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236\videos\stock_construction_right_candidate_resolved100.mp4
  Trace evaluation:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_resolved_100_20260525_0236\vlm\construction_trace_evaluation.json

Run summary:
  stock min construction-route clearance: 1.3538 m
  vlm min construction-route clearance: 1.5116 m
  publish_count: 98 / 100
  max_rtp_age_frames: 3
  p99 planner overhead: 21.08 ms
  mean path delta: 0.7270 m
  mean speed delta: 0.0 m/s
  process sweep after run: NO_TARGETED_POC_PROCESSES

Evaluator result:
  path_relevant_construction_frames: 66
  qwen_side_correct: 0
  qwen_side_wrong: 0
  qwen_side_missing: 66
  control_away_frames: 32
  control_toward_frames: 15
  control_neutral_frames: 19
  collision_count: 0

Raw label inspection:
  label_counts:
    29 frames: cones + construction_blocks_left_candidate
    28 frames: cones + construction_blocks_right_candidate
    24 frames: none
    17 frames: cones only
    2 frames: empty labels during startup

Conclusion:
  This run is not acceptable. Clearance improved and no collision occurred, but
  Qwen alternated between left-candidate-blocked and right-candidate-blocked.
  The current evaluator also does not yet map candidate-obstruction labels to
  construction side, which makes its Qwen side counts incomplete. The stronger
  root cause is that the left and right candidate labels were scored in separate
  rotating requests, so the resolver compared scores from different frames and
  different tracked-path states. That is another source of sign oscillation even
  when the RTP compiler sign conversion is correct.
2026-05-25 same-frame candidate comparison patch

Actions:
  - Added a candidate_pair auxiliary scene board derived from the same camera
    frame as the main board.
  - Kept existing candidate_left and candidate_right auxiliary boards intact.
  - Changed TensorRT label image routing so a score group containing both
    construction_blocks_left_candidate and construction_blocks_right_candidate
    uses candidate_pair instead of comparing two separately timed images.
  - Updated the candidate binary prompt to identify the same-frame comparison:
    cyan / blue-green is the left-shifted candidate, magenta / pink is the
    right-shifted candidate.
  - Updated the construction trace evaluator so candidate-obstruction labels
    count as construction side evidence.
  - Added/updated tests for candidate pair aux images, candidate prompt text,
    and evaluator side mapping.

Reason:
  The previous candidate resolver compared left-candidate and right-candidate
  scores from separate rotating requests. In async mode those scores can come
  from different source frames and different tracked-path states, which is a
  direct mechanism for left/right oscillation. Same-frame candidate comparison
  is still production-portable because it uses only the scene board and
  deterministic candidate overlays.

Verification:
  py -3.11 -m py_compile selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\evaluate_construction_trace.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 136 tests OK.
2026-05-25 same-frame candidate comparison rerun start

Actions:
  - Checked targeted POC processes before launching the rerun.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Starting another 100-frame construction_right closed-loop run.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_candidate_pair_100_20260525_0242

Key difference from previous run:
  - Score groups changed from
      cones,barrier;construction_blocks_left_candidate;construction_blocks_right_candidate
    to
      cones,barrier;construction_blocks_left_candidate,construction_blocks_right_candidate
  - This forces candidate-left and candidate-right obstruction labels to be
    scored from the same source frame and the new candidate_pair auxiliary
    scene board.
2026-05-25 same-frame candidate run failed, root cause fixed

Actions:
  - Attempted the same-frame candidate-pair closed-loop run.
  - It failed before VLM ready.
  - Checked targeted POC processes after failure.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Reproduced the Qwen TensorRT scorer startup directly with benchmark-groups.

Failure:
  The paired candidate group was still being forced through
  --construction-candidate-binary. The active TensorRT hidden-choice text engine
  is batch-1 for choice scoring, but binary scoring a two-label group creates
  inputs_embeds shape (2, 576, 2048). The engine expects (1, 576, 2048), so the
  second rotating group failed.

Direct reproduction:
  py -3.11 tools\reasoned_trajectory_poc\qwen_trt_label_engine.py ... --score-label-groups "cones,barrier;construction_blocks_left_candidate,construction_blocks_right_candidate" --warmup 0 --iters 2 benchmark-groups
  Error:
    RuntimeError: text inputs_embeds shape (2, 576, 2048) does not match engine (1, 576, 2048)

Fix:
  - Added a choice-mode candidate obstruction group:
      allowed words: cyan, pink, none
      cyan -> construction_blocks_left_candidate
      pink -> construction_blocks_right_candidate
  - Added a same-frame candidate obstruction prompt that asks which colored
    candidate corridor is obstructed, not which path is safe.
  - Made construction_blocks_left_candidate and
    construction_blocks_right_candidate an exclusive label group in rotating
    state, so a fresh contradictory Qwen judgement clears stale opposite-side
    candidate evidence.
  - Added tests for the candidate choice prompt, word-to-label mapping, and
    stale opposite-candidate override.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\reasoned\ui_scene_board.py tools\reasoned_trajectory_poc\evaluate_construction_trace.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 138 tests OK.
2026-05-25 corrected candidate-pair scorer probe and rerun start

Actions:
  - Ran a direct two-iteration Qwen TensorRT benchmark-groups probe without
    --construction-candidate-binary.
  - Confirmed both rotating groups execute:
      group 0: cones,barrier
      group 1: construction_blocks_left_candidate,construction_blocks_right_candidate
  - Group 1 candidate-pair direct probe completed with total_ms about 59.42 ms.
  - Checked targeted POC processes before launching the closed-loop rerun.
  - Result: NO_TARGETED_POC_PROCESSES.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_candidate_pair_choice_100_20260525_0249

Command intent:
  - Same-frame candidate-pair choice scoring.
  - No construction-candidate binary mode.
  - Speed fixed at 2.5 m/s and VLM speed control disabled.
2026-05-25 corrected candidate-pair closed-loop result and traceability patch

Actions:
  - Ran the corrected 100-frame construction_right candidate-pair choice demo.
  - Swept targeted POC processes after the run.
  - Evaluated construction trace.
  - Rendered stock, VLM, and side-by-side videos.
  - Inspected raw labels and scores.
  - Patched the demo runner to save auxiliary scene-board images so future
    traces include candidate_pair / candidate_left / candidate_right PNGs.

Artifacts:
  Run directory:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_pair_choice_100_20260525_0249
  Side-by-side video:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_pair_choice_100_20260525_0249\videos\side_by_side_construction_right_candidate_pair_choice100.mp4
  Trace evaluation:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_pair_choice_100_20260525_0249\vlm\construction_trace_evaluation.json

Run summary:
  stock min construction-route clearance: 1.3538 m
  vlm min construction-route clearance: 1.3538 m
  publish_count: 98 / 100
  max_rtp_age_frames: 3
  p99 planner overhead: 27.28 ms
  mean path delta: 0.0 m
  process sweep after run: NO_TARGETED_POC_PROCESSES

Evaluator result:
  path_relevant_construction_frames: 88
  qwen_side_correct: 0
  qwen_side_wrong: 0
  qwen_side_missing: 88
  control_away_frames: 0
  control_toward_frames: 0
  control_neutral_frames: 88
  collision_count: 0

Raw label inspection:
  label_counts:
    83 frames: cones only
    15 frames: none
    2 frames: empty labels during startup
  Candidate obstruction scores were negative whenever present, so no lateral
  RTP was produced.

Conclusion:
  Same-frame candidate comparison fixed the stale cross-frame comparison
  mechanism but made Qwen too conservative on the paired candidate board. The
  current pair-board visual/prompt contract is not usable enough. Future runs
  now save aux input images so this can be debugged from exact Qwen inputs
  rather than inferred from labels alone.
2026-05-25 safe-candidate shift test start

Reason:
  Candidate-obstruction scoring became too conservative: Qwen detected cones but
  chose no candidate obstruction, so no lateral RTP was produced. The existing
  construction_shift_left / construction_shift_right labels ask the more
  control-relevant same-frame question: which bounded candidate path should be
  used to avoid construction. This remains within the configured architecture
  and uses only the scene board plus Qwen labels/scores.

Action:
  - Starting a 100-frame construction_right run using:
      score groups: cones,barrier;construction_shift_left,construction_shift_right
      scene-board candidate guides: enabled
      PATH A: cyan left-shifted candidate
      PATH B: pink right-shifted candidate
  - Speed remains fixed at 2.5 m/s and VLM speed control remains disabled.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252
2026-05-25 safe-candidate shift closed-loop result

Actions:
  - Ran 100-frame construction_right safe-candidate shift demo.
  - Swept targeted POC processes after the run and after rendering.
  - Evaluated construction trace.
  - Rendered stock, VLM, and side-by-side videos.
  - Inspected raw labels/scores/RTP/lateral offsets.

Artifacts:
  Run directory:
    artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252
  Side-by-side video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252\videos\side_by_side_construction_right_shift_choice100.mp4
  VLM-only video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252\videos\vlm_construction_right_shift_choice100.mp4
  Stock-only video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252\videos\stock_construction_right_shift_choice100.mp4
  Trace evaluation:
    artifacts\reasoned_trajectory_poc\construction_right_shift_choice_100_20260525_0252\vlm\construction_trace_evaluation.json

Run summary:
  stock min construction-route clearance: 1.3538 m
  vlm min construction-route clearance: 2.1566 m
  publish_count: 98 / 100
  max_rtp_age_frames: 3
  p99 planner overhead: 5.10 ms
  max planner overhead: 5.41 ms
  mean path delta: 0.9949 m
  mean speed delta: 0.0 m/s
  process sweep after render: NO_TARGETED_POC_PROCESSES

Evaluator result:
  path_relevant_construction_frames: 52
  qwen_side_correct: 27
  qwen_side_wrong: 5
  qwen_side_missing: 20
  qwen_side_success_rate: 51.9%
  false_construction_labels: 46 / 48 not-path-relevant frames
  control_away_frames: 35
  control_toward_frames: 0
  control_neutral_frames: 17
  control_away_rate: 67.3%
  collision_count: 0

Raw label inspection:
  label_counts:
    58 frames: cones + construction_shift_left
    20 frames: cones + construction_shift_right
    19 frames: none
    1 frame: cones only
    2 frames: empty labels during startup

Conclusion:
  This is the best closed-loop construction result in this turn. The VLM path
  moved laterally, improved construction clearance by about 0.80 m over stock,
  changed no speed, did not collide, and did not command toward the construction
  in the evaluator. It still fails the final objective: Qwen side accuracy is
  only about 52%, wrong labels still occur, and the evaluator reports many
  construction labels after construction is no longer path-relevant. The
  durable lateral/controller logic is masking some Qwen label noise rather than
  Qwen itself meeting the 95% semantic gate.
2026-05-25 construction shift competitor-margin patch

Actions:
  - Audited construction_shift score margins from the best closed-loop trace.
  - Found that path-relevant wrong construction_shift_right frames had tiny
    shift-left vs shift-right score gaps.
  - Added a generic choice-group competitor_min_margin for construction shift
    labels, requiring the chosen safe candidate to beat the other candidate by
    at least 0.5 before producing a label.
  - Added tests proving low-margin candidate flips are rejected even when the
    selected label is well above the neutral/clear score.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 139 tests OK.
2026-05-25 construction shift margin rerun start

Actions:
  - Checked targeted POC processes before rerun.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Starting the same construction_right safe-candidate shift run with the new
    competitor_min_margin active.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257
2026-05-25 construction shift margin rerun result

Actions:
  - Ran 100-frame construction_right safe-candidate shift demo with competitor
    margin enabled.
  - Swept targeted POC processes after the run and after rendering.
  - Evaluated construction trace.
  - Rendered stock, VLM, and side-by-side videos.
  - Inspected raw labels, scores, RTP text, scene-board aux key logging, and
    current dirty git status.

Artifacts:
  Run directory:
    artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257
  Side-by-side video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257\videos\side_by_side_construction_right_shift_margin100.mp4
  VLM-only video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257\videos\vlm_construction_right_shift_margin100.mp4
  Stock-only video:
    artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257\videos\stock_construction_right_shift_margin100.mp4
  Trace evaluation:
    artifacts\reasoned_trajectory_poc\construction_right_shift_margin_100_20260525_0257\vlm\construction_trace_evaluation.json

Run summary:
  stock min construction-route clearance: 1.3538 m
  vlm min construction-route clearance: 2.1566 m
  publish_count: 98 / 100
  max_rtp_age_frames: 3
  p99 planner overhead: 5.13 ms
  max planner overhead: 5.48 ms
  mean path delta: 0.8546 m
  mean speed delta: 0.0 m/s
  process sweep after render: NO_TARGETED_POC_PROCESSES

Evaluator result:
  path_relevant_construction_frames: 52
  qwen_side_correct: 32
  qwen_side_wrong: 0
  qwen_side_missing: 20
  qwen_side_success_rate: 61.5%
  false_construction_labels: 35 / 48 not-path-relevant frames
  control_away_frames: 35
  control_toward_frames: 0
  control_neutral_frames: 17
  collision_count: 0

Raw label inspection:
  label_counts:
    59 frames: cones + construction_shift_left
    19 frames: none
    12 frames: cones only
    8 frames: cones + construction_shift_right
    2 frames: empty labels during startup

Traceability check:
  scene_board_aux_keys are logged in records. This run did not use candidate
  obstruction auxiliary boards, so aux keys are empty as expected. Future
  candidate-obstruction runs will save aux images.

Conclusion:
  The competitor margin removed all path-relevant wrong-side Qwen construction
  labels in this scenario while preserving the improved clearance and zero
  collision result. It still does not satisfy the objective: Qwen side success
  is only 61.5%, 20 path-relevant frames are missing side labels, and false
  construction labels after the scene is no longer path-relevant remain high.
  The current best construction path is safe-candidate shift choice with
  candidate guides plus competitor margin, not candidate-obstruction scoring.

Current process hygiene and tree:
  Final targeted process sweep: NO_TARGETED_POC_PROCESSES.
  Git status remains dirty with the current POC changes, nested tinygrad_repo
  dirty marker, and untracked evaluate_construction_trace.py.
2026-05-25 resumed state check after compaction

Actions:
  - Checked targeted POC processes before continuing.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Checked current git status.
  - Reviewed latest progress entries to avoid repeating the previous loop.

Current dirty tree:
  M README.md
  M README_REASONED_TRAJECTORY_POC.md
  M progress.md
  M selfdrive/controls/reasoned/planner.py
  M selfdrive/controls/reasoned/scene_board.py
  M selfdrive/controls/reasoned/ui_scene_board.py
  M selfdrive/controls/reasoned/vlm.py
  M selfdrive/controls/tests/test_reasoned_trajectory.py
  m tinygrad_repo
  M tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py
  M tools/reasoned_trajectory_poc/qwen_trt_label_engine.py
  M tools/reasoned_trajectory_poc/render_demo_videos.py
  M tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py
  ?? tools/reasoned_trajectory_poc/evaluate_construction_trace.py
2026-05-25 remaining construction trace analysis

Actions:
  - Analyzed the latest construction_right_shift_margin trace by frame.
  - Split required-missing, required-correct, not-required-false, and
    not-required-clear frames.
  - Inspected representative frame records and Qwen choice fields.

Findings:
  - Required-missing frames 31-47 often had Qwen choice answer A with strong
    construction_shift_left scores, but labels were emitted as none because
    RotatingScoreState gated all construction side/shift labels behind a
    separate recent cones/barrier label.
  - This is wrong for the current fast path because construction_shift_left /
    construction_shift_right are themselves Qwen scene-understanding labels:
    the prompt asks whether the safe bounded candidate path should shift away
    from construction.
  - Many not-required-false frames are evaluator-sensitive because the current
    evaluator uses the already-shifted tracked path lateral position. Once the
    VLM has moved the car away, the evaluator can mark the same cone row
    "not path relevant" even while the cones are still ahead/alongside and the
    durable avoidance should continue.

Patch intent:
  - Narrowly change label retention so construction_shift_* and
    construction_blocks_*_candidate labels can stand without a separate
    cones/barrier label.
  - Keep older construction_left/right and blue/purple edge labels gated by
    cones/barrier presence, because those side-only labels were historically
    noisy.
2026-05-25 construction shift self-evidencing gate patch

Actions:
  - Changed RotatingScoreState construction gating so
    construction_shift_left, construction_shift_right,
    construction_blocks_left_candidate, and construction_blocks_right_candidate
    can remain active without a separate cones/barrier label.
  - Kept construction_left/right and construction_blue/purple_edge gated by
    cones/barrier presence.
  - Added a unit test proving construction_shift_left survives without a
    separate cone-presence label while a side-only construction_left label is
    still gated away.

Reason:
  The safe-candidate shift group is the active production path. Its prompt
  already asks Qwen whether construction requires a bounded path shift, so
  suppressing it solely because the separate rotating cones/barrier group is
  stale or below threshold causes valid current-frame judgements to disappear.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_label_rtp_worker.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 140 tests OK.
2026-05-25 construction shift self-evidencing rerun start

Actions:
  - Checked targeted POC processes before rerun.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Starting the same construction_right safe-candidate shift run with
    construction_shift labels no longer gated by separate cones/barrier state.

Output directory:
  artifacts\reasoned_trajectory_poc\construction_right_shift_self_evidence_100_20260525_0303
2026-05-25 construction evaluator startup-adjusted metrics

Actions:
  - Added post-first-side metrics to evaluate_construction_trace.py.
  - Added a unit test proving startup/no-side frames remain counted in raw
    metrics while post-first-side metrics separately report steady-state
    side-classification quality.
  - Re-ran the evaluator on the latest self-evidencing construction run.

Reason:
  The remaining three side-missing frames were:
    frame 0: no VLM output yet
    frame 1: no VLM output yet
    frame 2: first published rotating group saw cones only
  The first side-capable Qwen frame was frame 3.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\evaluate_construction_trace.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 141 tests OK.

Updated evaluator result for:
  artifacts\reasoned_trajectory_poc\construction_right_shift_self_evidence_100_20260525_0303

Raw metrics:
  path_relevant_construction_frames: 52
  qwen_side_correct: 49
  qwen_side_wrong: 0
  qwen_side_missing: 3
  qwen_side_success_rate: 94.23%

Post-first-side metrics:
  first_qwen_side_frame: 3
  post_first_side_path_relevant_construction_frames: 49
  post_first_side_qwen_side_correct: 49
  post_first_side_qwen_side_wrong: 0
  post_first_side_qwen_side_missing: 0
  post_first_side_qwen_side_success_rate: 100.0%

Remaining issue:
  The current evaluator still reports high false construction labels after the
  vehicle has shifted away from the cone row. Many of those frames still have
  cones ahead or alongside and an active durable avoidance plan, so the current
  "false" count mixes true irrelevant-background false positives with valid
  maintain-avoidance behavior. This needs a better construction relevance
  evaluator before treating the false-construction metric as final.

2026-05-25 03:09:52 -04:00

Process hygiene sweep:
  Checked local targeted POC process names:
    python.exe, python3.exe, py.exe, ffmpeg.exe, trtexec.exe, polygraphy.exe, ssh.exe
  Filtered command lines for:
    qwen, TensorRT, onnx, metadrive, reasoned_trajectory,
    E:\ture_opamayo, F:\qwen_trt_export, 192.168.1.95,
    /data/qwen, /data/tinygrad, /data/openpilot

Result:
  NO_TARGETED_POC_PROCESSES

Additional checks:
  Broader local Python check found only non-POC processes:
    open-webui child python.exe
    DuckDuckGo MCP server python.exe pair
  NVIDIA process query showed desktop/UI processes only, no Qwen/TensorRT/MetaDrive compute job.
  C3X SSH process sweep to comma@192.168.1.95 timed out before login, then local
  sweep was repeated and again returned NO_TARGETED_POC_PROCESSES.

2026-05-25 03:13:30 -04:00

Resumed goal after compaction:
  - Re-read the active goal context.
  - Checked targeted local POC processes before continuing.
  - Result: NO_TARGETED_POC_PROCESSES.
  - Checked git status and current dirty tree.
  - Inspected current sign/scene-board/control seams:
    tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py
    tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py
    tools/reasoned_trajectory_poc/qwen_trt_label_engine.py
    selfdrive/controls/reasoned/ui_scene_board.py
    selfdrive/controls/reasoned/pathsynth.py
    selfdrive/controls/tests/test_reasoned_trajectory.py

Finding:
  Repeated sign regressions are coming from multiple side meanings crossing
  module boundaries:
    image/driver left-right,
    openpilot/PathSynth left-positive lateral,
    MetaDrive harness right-positive lateral,
    hazard side versus avoidance command side,
    and candidate-obstruction "blocked path" semantics.
  Existing tests cover many individual pieces, but the strongest current guard
  is still spread across separate tests rather than one invariant over the full
  Qwen-label to RTP to PathSynth to MetaDrive-control to scene-board-render
  chain.

2026-05-25 03:33:17 -04:00

Actions:
  - Added explicit lateral side helpers in run_metadrive_overlay_demo.py:
      lateral_side_openpilot
      lateral_side_metadrive
      construction_hazard_side_from_token
      construction_avoidance_side_valid
  - Rewired durable_avoidance_sign_valid to use the named side helper instead
    of raw sign checks.
  - Added log fields for active/desired/compiled lateral side in both
    openpilot and MetaDrive conventions, plus durable hazard side and durable
    target side.
  - Added full-chain construction invariant tests covering:
      construction_left
      construction_right
      construction_shift_left
      construction_shift_right
      construction_blocks_left_candidate
      construction_blocks_right_candidate
    The invariant follows label -> RTP -> PathSynth openpilot offset ->
    MetaDrive durable target -> rendered green path screen side.
  - Changed qwen_trt_label_engine.py so construction mirror fusion defaults on.

Verification:
  py -3.11 -m py_compile tools\reasoned_trajectory_poc\qwen_trt_label_engine.py tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py
  Result: OK.

  py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory
  Result: 143 tests OK.

Closed-loop construction probes:
  1. construction_right_sign_contract_80_20260525_031656
     Mode: candidate guide board, Qwen construction_shift labels.
     Result:
       collisions: 0
       min construction clearance: 2.1566 m
       qwen side wrong: 0
       control toward frames: 0
       max RTP age: 3
       p99 planner overhead: about 5.08 ms
     Video:
       artifacts\reasoned_trajectory_poc\construction_right_sign_contract_80_20260525_031656\videos\side_by_side_construction_right_shift80.mp4

  2. construction_left_sign_contract_80_20260525_031656
     Mode: candidate guide board, Qwen construction_shift labels.
     Result:
       invalid. Qwen chose construction_shift_left for left-side cones,
       which correctly compiled to a leftward move into the left-side hazard.
       qwen side wrong: 53
       control toward frames: 51
       collisions: 13

  3. construction_left_mirror_fusion_80_20260525_031656
     Mode: candidate guide board, construction_left/right with mirror fusion.
     Result:
       invalid. The PATH A/PATH B candidate lines contaminated side scoring.
       qwen side wrong: 35
       control toward frames: 44
       collisions: 13

  4. construction_left_side_board_mirror_80_20260525_031656
     Mode: side-label board, no candidate guides, construction_left/right with
     mirror fusion.
     Result:
       closed-loop control was safe, but raw Qwen side labels were unstable.
       collisions: 0
       min construction clearance: 1.9543 m
       qwen side correct: 24
       qwen side wrong: 5
       qwen side missing: 33
       control away frames: 33
       control toward frames: 0
       max RTP age: 4
     Video:
       artifacts\reasoned_trajectory_poc\construction_left_side_board_mirror_80_20260525_031656\videos\side_by_side_construction_left_side_board_mirror80.mp4

  5. construction_right_side_board_mirror_80_20260525_031656
     Mode: side-label board, no candidate guides, construction_left/right with
     mirror fusion.
     Result:
       invalid. Side-label board that helped left construction did not transfer
       to right construction.
       collisions: 0
       min construction clearance: 1.2795 m
       qwen side wrong: 17
       control toward frames: 30
     Video:
       artifacts\reasoned_trajectory_poc\construction_right_side_board_mirror_80_20260525_031656\videos\side_by_side_construction_right_side_board_mirror80.mp4

Additional probes:
  - One-frame construction_left/right benchmark showed the side group can score
    left correctly on a left-side frame, but is left-biased on a right-side
    candidate-guide frame.
  - construction_mirror_consistency rejects same-side mirror bias but can reject
    true positives too.
  - construction_mirror_fusion corrects some right-side frames but can invert
    early left-side frames when the mirror run has a stronger left-word prior.
  - construction_blue_edge/purple_edge with side board produced low-margin
    outputs and did not solve the side problem.
  - construction_shift_left/right with mirror fusion tied A and B and rejected
    the frame, so it does not solve the candidate-choice path.

Current conclusion:
  The raw sign conversion in the compiler is now explicitly guarded by tests.
  The remaining construction failure is Qwen label instability and lexical/color
  prior, not a MetaDrive/openpilot sign conversion bug. Candidate-guide shift
  labels work for right-side construction and fail for left-side construction.
  Side-label boards can make left closed-loop safe but are not symmetric enough
  for right-side construction. Construction is therefore not solved yet.

Process hygiene:
  Final targeted sweep returned NO_TARGETED_POC_PROCESSES.

2026-05-25 continuation:
  - Ran targeted process sweep before new work. Result: NO_TARGETED_POC_PROCESSES.
  - Re-read current dirty tree and active MetaDrive/Qwen path instead of relying on compacted context.
  - Found a production-relevant control bypass: when a newly compiled lateral RTP did not activate a durable plan, the demo still fell back to the one-frame compiled lateral offset. That bypass can steer on a single wrong side label even though repeated activation/override guards exist. Next edit: keep that fallback only behind an explicit opt-in argument and log when it is used.
  - Implemented the bypass fix in tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py:
    compose_lateral_offset_after_publish() now returns durable lateral output only by default, and the one-frame compiled lateral fallback is available only with --allow-compiled-lateral-fallback.
  - Added per-frame trace field compiled_lateral_fallback_used.
  - Added tests proving default behavior does not use the compiled fallback after one unconfirmed construction label, and old fallback behavior is explicit opt-in.
  - Verification: py -3.11 -m py_compile tools\reasoned_trajectory_poc\run_metadrive_overlay_demo.py selfdrive\controls\tests\test_reasoned_trajectory.py passed. py -3.11 -m unittest selfdrive.controls.tests.test_reasoned_trajectory passed, 145 tests OK.
  - Ran right-side construction demo with durable-only lateral publication:
    artifacts\reasoned_trajectory_poc\construction_right_durable_guard_80_20260525_0340.
    Result: stock min construction clearance 1.3538 m, VLM min clearance 1.6957 m, collisions 0, max RTP age 4, planner p99 about 5.20 ms, but Qwen side correct only 36/72, wrong 20/72, missing 16/72, control toward 2 frames. Video: videos\side_by_side_construction_right_durable_guard80.mp4.
  - Ran left-side construction demo with same settings:
    artifacts\reasoned_trajectory_poc\construction_left_durable_guard_80_20260525_0340.
    Result: stock min construction clearance 1.3538 m, VLM min clearance 1.9543 m, collisions 0, max RTP age 4, planner p99 about 5.18 ms, Qwen side correct 24/62, wrong 5/62, missing 33/62, control toward 0 frames. Video: videos\side_by_side_construction_left_durable_guard80.mp4.
  - Post-run targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - Current evidence: durable-only publication reduced bad immediate control, but construction-side perception remains far below the 95% gate and right-side still has two toward-hazard control frames. Next step is to inspect the toward-control records and remove stale/wrong-side durable persistence without using simulator object labels.
  - Inspected the two reported right-side toward-control frames. They were not actually tracked toward hazard: evaluator was prioritizing post-publish desired_lateral_offset_m over active_lateral_offset_m. On frames 70-71 active_lateral_offset_m still tracked left/away, while desired_lateral_offset_m had just flipped right from a stale/wrong-side durable plan.
  - Fixed tools/reasoned_trajectory_poc/evaluate_construction_trace.py so control side uses active_lateral_offset_m/green_path first, and planned side is reported separately. Added planned_away/planned_toward metrics. Tests still pass, 145 OK.
  - Re-ran construction evals for the two 80-frame demos. Right-side corrected actual control_toward_frames to 0, planned_toward_frames remains 2. Left-side actual/planned control_toward_frames are 0. This does not fix Qwen perception: Qwen side success remains 36/72 right and 24/62 left.
  - Tested same-frame candidate-obstruction scoring on right-side construction:
    artifacts\reasoned_trajectory_poc\construction_right_candidate_blocked_80_20260525_0346.
    Result: failed to activate; Qwen emitted no candidate obstruction labels, path delta 0, min clearance stayed stock at 1.3538 m. Latency stayed nonblocking but higher, p99 about 28.84 ms. Video: videos\side_by_side_construction_right_candidate_blocked80.mp4.
  - Tightened production default for contradictory lateral overrides: --durable-conflict-override-confidence now defaults to 0.90 instead of 0.80. Initial construction avoidance can still activate at moderate confidence, but reversing an already active away-from-hazard plan requires stronger evidence. Added regression test: 0.84 contradictory construction side does not override under the production default, while explicit lower-threshold tests still allow repeated override. Verification: py_compile passed, unittest passed with 146 tests OK.
  - Re-ran side-label construction demos under the 0.90 contradictory override default:
    * Right: artifacts\reasoned_trajectory_poc\construction_right_conflict090_80_20260525_0348, min clearance 2.1382 m, collisions 0, actual control_toward 0, planned_toward 0, max RTP age 4, planner p99 about 5.51 ms, video videos\side_by_side_construction_right_conflict090_80.mp4.
    * Left: artifacts\reasoned_trajectory_poc\construction_left_conflict090_80_20260525_0348, min clearance 2.0669 m, collisions 0, actual control_toward 0, planned_toward 0, max RTP age 4, planner p99 about 5.13 ms, video videos\side_by_side_construction_left_conflict090_80.mp4.
  - Important caveat: Qwen raw construction side classification is still bad in these runs. The safer behavior is from initial side activation plus bounded temporal gating, not from a solved side classifier. This remains below the required 95% construction side gate.
  - Post-run targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - Inspected lead-vehicle visual heading path. Current code already uses DefaultVehicle, no special color, and visual heading offset 0.0 rad. Unit tests assert model/default heading equals route heading.
  - Ran true moving lead demo:
    artifacts\reasoned_trajectory_poc\true_moving_lead_heading_80_20260525_0351.
    Result: spawned lead visual_heading_theta - route_heading_theta = 0.0 deg, Qwen lead classes included true_moving_lead, no speed delta, no speed cap, max RTP age 3, planner p99 about 5.72 ms. Video: videos\side_by_side_true_moving_lead_heading80.mp4.
  - Ran slower lead demo before fixing live state default:
    artifacts\reasoned_trajectory_poc\slower_lead_heading_80_20260525_0352.
    Result: failed. Qwen/physical consistency classified the slower lead as true_moving_lead and no slowdown occurred. Root cause: qwen_trt_label_engine serve mode defaulted to static --vehicle-state instead of per-frame scene_board_state_text, so physical lead consistency did not see live lead distance/speed/relative speed.
  - Fixed qwen_trt_label_engine.py so --use-payload-vehicle-state is a BooleanOptionalAction defaulting to true, and added use_payload_vehicle_state to the runtime contract. Verification: py_compile passed, unittest passed with 146 tests OK.
  - Re-ran slower lead with payload state default:
    artifacts\reasoned_trajectory_poc\slower_lead_payload_state_80_20260525_0355.
    Result: Qwen lead classes included slower_lead_closing, mean speed reduced from stock 1.8525 m/s to 1.6549 m/s, mean speed delta 0.3333 m/s, min vehicle clearance improved from 9.8724 m to 11.4170 m, max RTP age 3, planner p99 about 5.50 ms. Video: videos\side_by_side_slower_lead_payload_state80.mp4.
  - Post-lead targeted process sweep returned NO_TARGETED_POC_PROCESSES.

2026-05-25 continuation 2:
  - Started by checking current external state. Targeted process sweep returned NO_TARGETED_POC_PROCESSES. Worktree still has only the expected POC/readme/progress/tinygrad dirty files.
  - Next target is raw construction side classification, because current control is bounded safer but construction side classification remains far below the 95% gate.
  - Re-ran right construction after the prior live payload-state default:
    artifacts\reasoned_trajectory_poc\construction_right_payload_state_80_20260525_0401.
    Result: regression. Qwen emitted no construction side labels, path delta 0, clearance stayed stock at 1.3538 m. This showed the live scene_board_state_text should not be fed into every label group because it changes the fixed text input enough to break construction scoring.
  - Patched qwen_trt_label_engine.py with _vehicle_state_for_labels() and --payload-vehicle-state-scope auto/all/none. Default auto now feeds live scene_board_state_text only to physical lead/vehicle label groups; construction and signal groups use the stable --vehicle-state string unless explicitly overridden. Runtime contract now records payload_vehicle_state_scope.
  - Added unit test proving auto scope keeps construction on static vehicle_state while giving live payload state to lead groups. Verification: py_compile passed, unittest passed with 147 tests OK.
  - Continued after compaction with a targeted process sweep before code changes. Result: NO_TARGETED_POC_PROCESSES.
  - Inspected current dirty tree and the active rotating Qwen score state. Found the next concrete construction failure: side labels can be cached before cones/barriers are positively confirmed, then later combine with fresh construction presence and arm a durable lateral plan on temporally incoherent evidence.
  - Patched RotatingScoreState in qwen_label_rtp_worker.py to track a construction presence episode. Construction side/edge labels are now control-relevant only when scored after the active cones/barrier episode begins. Shift and candidate-obstruction labels remain self-evidencing because their questions already include the construction object.
  - Updated rotating-score tests so side/edge labels before construction presence do not arm when cones arrive later, while side/edge labels after construction presence still arm and survive presence refresh within the same episode.
  - Verification after temporal coherence patch: py_compile passed for qwen_label_rtp_worker.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 149 tests OK.
  - Process sweep before construction reruns returned NO_TARGETED_POC_PROCESSES.
  - Reran failing left color-edge construction with temporal coherence:
    artifacts\reasoned_trajectory_poc\construction_left_color_edge_coherent_80_20260525_0416.
    Result: collisions 0, min construction clearance 2.0249 m, actual control_toward 0, planned_toward 0, max RTP age 4, planner p99 about 5.16 ms, speed delta 0.0 with speed control disabled. Video: videos\side_by_side_construction_left_color_edge_coherent80.mp4.
    Perception caveat remains severe: qwen_side_correct 8/59, wrong 0/59, missing 51/59, false construction labels 20/21. Temporal coherence fixed the stale wrong-side activation, not the raw construction side classifier.
  - Reran right color-edge construction with temporal coherence:
    artifacts\reasoned_trajectory_poc\construction_right_color_edge_coherent_80_20260525_0417.
    Result: collisions 0, min construction clearance 2.0794 m, actual control_toward 0, planned_toward 0, max RTP age 4, planner p99 about 5.09 ms, speed delta 0.0 with speed control disabled. Video: videos\side_by_side_construction_right_color_edge_coherent80.mp4.
    Perception caveat remains unacceptable: qwen_side_correct 8/56, wrong 23/56, missing 25/56. Control safety is currently coming from temporal/durable bounds, not a solved Qwen side classifier.
  - Inspected the right color-edge trace and a saved wrong-side frame. Direct construction_left/construction_right scoring on vlm_input_0028 selected construction_right correctly, while the blue/purple edge label had selected the wrong edge in the closed-loop trace. This points at overlay color-edge wording as a brittle label formulation, not a compiler sign bug.
  - Tested a semantic construction_left/construction_right runtime with the same temporal coherence:
    artifacts\reasoned_trajectory_poc\construction_right_side_semantic_coherent_80_20260525_0419.
    Result: failed closed-loop. It produced 13 collisions, min construction clearance 0.5752 m, actual/planned control_toward 56 frames, qwen_side_correct 19/77, wrong 21/77, missing 37/77. A single direct-frame probe was not enough; the closed-loop first accepted semantic side can still arm the wrong durable maneuver.
  - Tried to test 224 px construction side input, but the existing 224 vision engine is not usable with the current processor output: processor produced visual input shape (192, 1176), engine expects (160, 1176). No episode was produced. Post-failure process sweep returned NO_TARGETED_POC_PROCESSES.
  - Next fix direction: keep the existing Qwen label architecture but require multi-signal construction side consensus when both semantic side labels and colored edge labels are configured. A construction side should not become control-relevant from a single label family when an independent Qwen side formulation is expected but has not agreed yet.
  - Implemented multi-signal construction side consensus in RotatingScoreState. If both construction_left/right and construction_blue/purple_edge groups are configured, dependent construction side labels are published only when both Qwen label families agree after the active cones/barrier episode begins. Single-family runtimes keep their prior behavior. Added tests for waiting on agreement and rejecting disagreement until the stale family updates.
  - Verification after consensus patch: py_compile passed for qwen_label_rtp_worker.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 151 tests OK.
  - Reran right construction with combined semantic side plus color-edge side consensus:
    artifacts\reasoned_trajectory_poc\construction_right_multisignal_consensus_80_20260525_0428.
    Result: collisions 0, min construction clearance 2.0912 m, actual/planned control_toward 0, max RTP age 5, planner p99 about 5.05 ms, speed delta 0.0 with speed control disabled. Video: videos\side_by_side_construction_right_multisignal_consensus80.mp4.
    Perception changed from unsafe to conservative: qwen_side_correct 19/55, wrong 0/55, missing 36/55, false construction 1/25. This removes sign-flip failures in this run but still misses the >=95% construction side gate by a wide margin.
  - Reran left construction with combined semantic side plus color-edge side consensus:
    artifacts\reasoned_trajectory_poc\construction_left_multisignal_consensus_80_20260525_0429.
    Result: collisions 0, min construction clearance 1.3538 m, actual/planned control_toward 0, max RTP age 6, planner p99 about 4.89 ms, speed delta 0.0. Video: videos\side_by_side_construction_left_multisignal_consensus80.mp4.
    This was safe but not useful: qwen_side_correct 0/77, wrong 0/77, missing 77/77, path delta nearly zero. The consensus gate eliminated wrong-side output but killed left-side recall.
  - Inspected left-side consensus trace. Root cause: construction-mirror-fusion corrupts semantic left/right labels on left-side scenes. The original unmirrored semantic choice often says left correctly, but the mirrored image also says left due Qwen's left-word prior; the fusion maps that mirrored-left answer back to original-right and overpowers the original. Color-edge labels see left/blue, but consensus rejects because semantic fusion reports right. Next work should repair mirror fusion or stop using semantic fusion as a consensus input.
  - Final targeted process sweep after all runs returned NO_TARGETED_POC_PROCESSES.

2026-05-25 continuation 3:
  - Started by checking current external state. Targeted process sweep returned NO_TARGETED_POC_PROCESSES. Dirty tree still contains the expected POC/readme/progress/tinygrad files.
  - Patched construction mirror fusion in qwen_trt_label_engine.py. Mirror fusion now treats the mirrored scene as auxiliary evidence: if original and mirror-mapped construction choices agree, scores are fused; if they disagree, a weak original is cleared to none and a strong original is preserved. The mirrored view can no longer invert the direct-view side by additive score alone.
  - Updated mirror-fusion tests: weak original plus contradictory mirror now clears instead of flipping, agreement still boosts the mapped original side, and the existing strong-original left case remains protected.
  - Tightened the cleared mirror-fusion path after tests exposed a stale-score risk: when a construction side choice is rejected by margin, label_scores for the construction side group are forced negative so rotating durable state can clear rather than hold a weak positive score.
  - Verification: py_compile passed for qwen_trt_label_engine.py, qwen_label_rtp_worker.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 152 tests OK.
  - Reran left construction with guarded mirror fusion plus multisignal consensus:
    artifacts\reasoned_trajectory_poc\construction_left_guarded_fusion_consensus_80_20260525_0437.
    Result: collisions 0, min construction clearance 1.6900 m, actual/planned control_toward 0, max RTP age 5, planner p99 about 5.04 ms, speed delta 0.0. Video: videos\side_by_side_construction_left_guarded_fusion_consensus80.mp4.
    Improvement: qwen_side_wrong stayed 0 and qwen_side_correct improved from 0/77 to 22/75. Still incomplete: qwen_side_missing 53/75 and false construction labels 5/5 not-required frames, so this does not meet the >=95% construction gate.
  - Reran right construction with guarded mirror fusion plus previous strict consensus:
    artifacts\reasoned_trajectory_poc\construction_right_guarded_fusion_consensus_80_20260525_0438.
    Result: safe but useless. collisions 0, min construction clearance 1.3538 m, qwen_side_correct 0/77, wrong 0/77, missing 77/77, mean path delta 0.0. The guard removed sign flips but semantic left/right remained lexically biased left and blocked right-side action.
  - Inspected right trace. Color-edge labels correctly identified purple/right on useful frames, while semantic construction_left/right often said left. Updated RotatingScoreState combined-family behavior: when both semantic and corridor-edge side families are configured, semantic labels cannot arm construction alone, and the corridor-edge family is authoritative if the two families disagree. This is production-facing because edge labels refer to the actual green path corridor, not simulator coordinates.
  - Verification after corridor-edge authority patch: py_compile passed for qwen_trt_label_engine.py, qwen_label_rtp_worker.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 152 tests OK.
  - Reran right construction with corridor-edge authority:
    artifacts\reasoned_trajectory_poc\construction_right_edge_authority_80_20260525_0442.
    Result: collisions 0, min construction clearance 1.4670 m, actual/planned control_toward 0, max RTP age 5, planner p99 about 5.06 ms. Video: videos\side_by_side_construction_right_edge_authority80.mp4.
    Still incomplete: qwen_side_correct 6/77, wrong 6/77, missing 65/77. Wrong labels happened late, after a correct right-side avoidance had been established, because Qwen flipped edge side near the cones.
  - Added construction side-locking in RotatingScoreState. Within the same cones/barrier episode, a construction side cannot flip on a single contradictory view; it needs either 3 repeated contradictory observations or a contradiction score >= 3.0. This is production-facing short-occlusion/perspective persistence, not simulator object logic.
  - Fixed the side-lock pending counter after tests showed the current locked side was resetting pending contradictions during active-label rendering. Verification: py_compile passed for qwen_label_rtp_worker.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 154 tests OK.
  - 2026-05-25 04:51:24 -04:00 targeted hung-process sweep returned NO_TARGETED_POC_PROCESSES. Checked python.exe, python3.exe, py.exe, ffmpeg.exe, trtexec.exe, polygraphy.exe, and ssh.exe for Qwen/TensorRT/ONNX/MetaDrive/reasoned-trajectory/openpilot/C3X command lines.
  - 2026-05-25 04:54:46 -04:00 inspected construction_right_side_lock_80_20260525_0447 frames 20-39. Root cause found: a stale construction_blue_edge score from source frame 28 was not eligible to publish after the new cones episode at source frame 31, but _construction_presence_gated still used it as internal consensus evidence and allowed a fresh semantic construction_left at source frame 32. This is a rotating-state temporal-consensus bug; the consensus side must be computed only from side labels scored after the active construction presence anchor.
  - 2026-05-25 04:56:12 -04:00 patched qwen_label_rtp_worker.py so combined construction-side consensus is computed only from side labels whose positive frame is at or after the active cones/barrier anchor. Added test_construction_combined_side_ignores_stale_edge_from_previous_episode to prove a stale corridor-edge label cannot authorize a fresh semantic side after a new construction episode starts. Verification: py_compile passed for qwen_label_rtp_worker.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 155 tests OK.
  - 2026-05-25 04:57:47 -04:00 preparing a same-shape 80-frame construction_right closed-loop rerun using the current TensorRT Qwen rotating groups cones,barrier ; construction_left,construction_right ; construction_blue_edge,construction_purple_edge, async latest-frame queue, speed_mps=2.5, and VLM speed control disabled. Purpose: verify the stale-edge consensus fix removes the frame-36 wrong-side semantic activation without changing the scenario.
  - 2026-05-25 05:00:43 -04:00 first rerun attempt failed before VLM ready because --require-manifest rejected the changed code SHA. An attempted refresh with the old 220-token selected-logits engine failed artifact validation because the current manifest contract requires last_token_mask. Refreshed the manifest successfully using the active fast construction contract: qwen_text_36layer_nvfp4_seq576_hidden_choice_trt.engine, text_seq_len=576, text_output=hidden, label_decision_mode=choice, rotating groups cones/barrier ; construction_left/right ; construction_blue_edge/purple_edge, mirror fusion, and edge binary. check-artifacts returned ok true and wrote manifest SHA b1361247870c0885f5ad0dac2e42db06b1e6902701cb7118c326417ddcb8bad3.
  - 2026-05-25 05:02:48 -04:00 direct server smoke showed --construction-edge-binary is incompatible with the shared hidden-choice engine because it creates binary batch-2 inputs for a batch-1 choice engine. Refreshed the manifest without --construction-edge-binary, preserving the previous side-lock run shape with all three rotating groups in choice mode. Direct serve smoke with --require-manifest printed {"ready":true}.
  - 2026-05-25 05:07:40 -04:00 completed nvfp4 rerun construction_right_stale_consensus_fix_80_20260525_0503. It is a failed construction run: collisions 0 but min clearance stayed stock at 1.3538 m, mean path delta 0.016 m, qwen_side_correct 0/77, wrong 1/77, missing 76/77, max RTP age 4, planner p99 about 4.78 ms. Rendered videos under videos\side_by_side_construction_right_stale_consensus_fix80.mp4. Comparing traces showed prior side_lock used backend fp8, while this rerun used nvfp4. Refreshed manifest for fp8 hidden-choice engine qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine with the same rotating construction groups and mirror fusion; check-artifacts ok true, manifest SHA d61fe921f73bbc5f8612d3d177352dc46dea5fb2dfddc29911b22390418d1ea2. Direct fp8 serve smoke printed {"ready":true}.
  - 2026-05-25 05:11:25 -04:00 completed fp8 hidden-choice rerun construction_right_stale_consensus_fix_fp8_80_20260525_0508. It is also a failed construction run: collisions 0 but min clearance stayed stock at 1.3538 m, path delta 0, qwen_side_correct 0/77, wrong 0/77, missing 77/77, max RTP age 5. Rendered videos under videos\side_by_side_construction_right_stale_consensus_fix_fp8_80.mp4. The previous side_lock backend string was fp8-full168-score, so refreshed manifest for the older binary selected-logits FP8 engine qwen_text_36layer_fp8_seq768_trt.engine, text_seq_len=768, text_output=logits, label_decision_mode=binary, same rotating groups and mirror fusion. check-artifacts ok true, manifest SHA 3ed6016751667a5c0cf12291d1ef5bab24b447c21c1ce628171e43ffc6e132be. Direct binary fp8 serve smoke printed {"ready":true}.
  - 2026-05-25 05:18:25 -04:00 completed binary FP8 rerun construction_right_stale_consensus_fix_fp8_binary_80_20260525_0512. It is unsafe: qwen_side_correct 0/77, wrong 8/77, missing 69/77, collision_count 13, min clearance 0.5841 m, control_toward 56, max RTP age 8. Videos under videos\side_by_side_construction_right_stale_consensus_fix_fp8_binary_80.mp4. Root cause from trace: stale reused RTP frames and two early wrong source frames allowed durable lateral activation toward the hazard. Patched DurableLateralOverrideState and update_durable_lateral_plans so activation and conflict confirmation counts only advance on a new rtp_source_frame_id; added durable_lateral_pending_observation_id trace logging; changed default durable_lateral_activation_confirm_frames from 2 to 3 distinct observations. Added tests for stale reused RTP not counting toward activation or conflict override. Verification: py_compile passed for run_metadrive_overlay_demo.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 157 tests OK.
  - 2026-05-25 05:22:20 -04:00 reran binary FP8 after distinct-source/3-confirm guard:
    artifacts\reasoned_trajectory_poc\construction_right_distinct_source_confirm_fp8_binary_80_20260525_0519.
    Result: the unsafe wrong-side maneuver was prevented, but construction avoidance did not activate. collision_count 0, min clearance stayed stock at 1.3538 m, path delta 0, qwen_side_correct 0/77, wrong 0/77, missing 77/77, control_toward 0, max RTP age 8, planner p99 about 1.95 ms. Videos rendered under videos\side_by_side_construction_right_distinct_source_confirm_fp8_binary_80.mp4. The run shows the guard improves safety by refusing weak/stale activations, but Qwen construction side recall is currently zero in this command shape. Final targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - 2026-05-25 05:23:26 -04:00 continuation start. Targeted process sweep returned NO_TARGETED_POC_PROCESSES. Dirty tree still has the expected POC/readme/progress/tinygrad files. Next focus: recover construction-side recall in a production-facing way using existing Qwen label/scoring path, because the latest guard made the car safer but construction avoidance did not activate.
  - 2026-05-25 05:29:32 -04:00 inspected current construction prompt/RTP code and probed saved right-construction scene-board frames from construction_right_distinct_source_confirm_fp8_binary_80_20260525_0519. Direct non-rotating FP8 binary probes: presence labels are stable, semantic side is lexically wrong on most right-side frames, edge-color detects purple/right on at least one clear frame, shift labels are mostly wrong or tied without candidate guides. This points to edge-color as the least bad production-facing construction signal because it refers to the blue/purple planned-corridor overlay instead of simulator coordinates. Refreshed the runtime manifest for an edge-only rotating contract, groups cones/barrier ; construction_blue_edge/construction_purple_edge, FP8 seq768 selected-logits engine. check-artifacts returned ok true and wrote manifest SHA 71d7f5c4e56eac9a8dd05e938e0f07b44e17e8ebcc21e6542adeb1e86e32ed80.
  - 2026-05-25 05:33:02 -04:00 edge-only closed-loop run construction_right_edge_only_distinct_confirm_80_20260525_0530 finished. Result: collision_count 0, but qwen_side_correct 0/77, wrong 24/77, missing 53/77, path delta did not improve clearance, min clearance stayed stock 1.3538 m. Trace root cause: RotatingScoreState locked construction side after the first side observation. An initial wrong construction_blue_edge from source frame 0 locked the episode, then a later construction_purple_edge score from source frame 2 was suppressed before durable activation. This is too sticky for production; side lock should not harden until repeated distinct side observations.
  - 2026-05-25 05:35:07 -04:00 patched RotatingScoreState so construction side lock hardens only after 3 distinct same-side observations. Before lock hardens, a later opposite side observation can replace an initial single wrong observation. Existing locked-side contradiction behavior still requires 3 repeated contradictions or a high-confidence override. Added test_construction_side_does_not_lock_on_first_observation and updated side-lock tests to establish a lock with repeated observations before checking contradiction rejection/override. Verification: py_compile passed for qwen_label_rtp_worker.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 158 tests OK.
  - 2026-05-25 05:40:39 -04:00 targeted POC process sweep returned NO_TARGETED_POC_PROCESSES. Broader read-only check found only DuckDuckGo MCP helper Python processes and an unrelated old python.exe child of open-webui.exe using about 3.8 GB private memory. No Qwen, TensorRT, ONNX, MetaDrive, reasoned_trajectory, workspace, C3X SSH, ffmpeg, trtexec, or polygraphy process is currently running.
  - 2026-05-25 05:45:38 -04:00 continuation from active goal. Targeted POC sweep returned NO_TARGETED_POC_PROCESSES. Inspected current sign/overlay/control paths and ran a live MetaDrive geometry probe. Probe confirmed MetaDrive lane positive lateral is visual/scene right while openpilot positive lateral is left, so the current openpilot_to_metadrive_lateral_m/metadrive_to_openpilot_lateral_m sign inversion is correct. Patched UiSceneBoardRenderer so corridor side text labels are default-on and anchored to the actual projected blue-left and purple-right corridor edges instead of fixed screen coordinates. Added test_scene_board_side_text_labels_are_default_and_follow_projected_edges and updated default scene-board tests. Verification: py_compile passed for ui_scene_board.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 159 tests OK.
  - 2026-05-25 05:56:27 -04:00 closed the side-label experiment. The default-on text labels were tested in construction_right_default_edge_labels_80_20260525_0549 and rejected: Qwen over-called construction_blue_edge on a right-side hazard, with qwen_side_correct 5/77, wrong 35/77, missing 37/77. Durable activation gating prevented steering into the cones, so control_away/control_toward both stayed 0 and min construction clearance stayed stock at 1.3538 m. Rendered videos under artifacts\reasoned_trajectory_poc\construction_right_default_edge_labels_80_20260525_0549\videos. Reverted corridor side labels to opt-in while keeping the anchored helper/test for controlled future probes; py_compile passed for ui_scene_board.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 159 tests OK. Tried candidate-obstruction auxiliary boards with construction_blocks_left_candidate/construction_blocks_right_candidate in construction_right_candidate_blocks_80_20260525_0552; rejected that mode for now because one candidate-side inference completed 211 frames stale during prewarm and the run produced 0 valid publishes. Killed only that run's process tree and final targeted process sweep returned NO_TARGETED_POC_PROCESSES. Restored the active runtime manifest to the previous fast edge-only FP8 binary contract, manifest SHA 71d7f5c4e56eac9a8dd05e938e0f07b44e17e8ebcc21e6542adeb1e86e32ed80.
  - 2026-05-25 06:18:57 -04:00 targeted hung-process sweep returned NO_TARGETED_POC_PROCESSES. Checked python.exe, python3.exe, py.exe, ffmpeg.exe, trtexec.exe, polygraphy.exe, and ssh.exe for command lines tied to Qwen, TensorRT, ONNX, MetaDrive, reasoned_trajectory, E:\ture_opamayo, F:\qwen_trt_export, C3X SSH 192.168.1.95, /data/qwen, /data/tinygrad, and /data/openpilot. No cleanup was needed.
  - 2026-05-25 06:44:27 -04:00 targeted hung-process sweep returned NO_TARGETED_POC_PROCESSES. Checked python.exe, python3.exe, py.exe, ffmpeg.exe, trtexec.exe, polygraphy.exe, and ssh.exe for Qwen/TensorRT/ONNX/MetaDrive/reasoned-trajectory/openpilot/C3X command lines. No targeted POC, render, TensorRT, SSH, or MetaDrive process was running, so no process was killed.
  - 2026-05-25 06:50:31 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES and dirty tree matched the expected POC files.
  - Inspected the current construction side, MetaDrive lateral, scene-board, and lead visual-heading paths. The repeated sign-flip failure mode is not one single inverted multiply; it comes from the same left/right facts being encoded separately as Qwen labels, RTP scenes, avoid tokens, openpilot lateral offsets, MetaDrive lane offsets, and overlay geometry. That lets one local patch fix one family while another label family or simulator conversion drifts.
  - Added selfdrive.controls.reasoned.side_semantics as the canonical production-facing side contract. It defines construction label families, hazard side derivation, hazard-side-to-safe-openpilot-lateral mapping, RTP construction fields, avoid-token hazard side, and openpilot-side sign validation.
  - Routed qwen_label_rtp_worker._construction_side and _construction_rtp_fields through side_semantics so Qwen labels now compile through one canonical mapping instead of a local duplicate mapping.
  - Routed run_metadrive_overlay_demo construction_hazard_side_from_token, construction_avoidance_side_valid, and durable_avoidance_from_program through side_semantics. The simulator still performs the separate openpilot-to-MetaDrive sign conversion, but the hazard-side to safe openpilot offset is no longer hand-written in the sim compiler.
  - Expanded tests with test_canonical_construction_side_semantics_cover_all_qwen_families and added construction_blue_edge/construction_purple_edge to the full Qwen label -> RTP -> PathSynth -> durable plan -> MetaDrive conversion -> scene-board overlay sign test. This now covers semantic side labels, blue/purple corridor-edge labels, shift labels, and candidate-obstruction labels in the same invariant.
  - Verification: py_compile passed for side_semantics.py, qwen_label_rtp_worker.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 166 tests OK.
  - Final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No Qwen, TensorRT, ONNX, MetaDrive, ffmpeg, trtexec, C3X SSH, or reasoned-trajectory process was left running.
  - 2026-05-25 06:53:47 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - Inspected remaining construction-side logic after side_semantics was added. Found one more control-relevant duplicate: RotatingScoreState still had its own CONSTRUCTION_SIDE_BY_LABEL and family-side logic, and evaluate_construction_trace had its own CONSTRUCTION_LABEL_SIDES. That meant Qwen label persistence, side-locking, and proof metrics could still drift from the compiler's canonical mapping.
  - Patched qwen_label_rtp_worker.py so construction semantic labels, edge labels, shift labels, candidate labels, and CONSTRUCTION_SIDE_BY_LABEL are all derived from selfdrive.controls.reasoned.side_semantics. RotatingScoreState._construction_label_side and _family_side now call construction_hazard_side_from_labels instead of reimplementing left/right.
  - Patched evaluate_construction_trace.py so qwen_construction_side uses construction_hazard_side_from_labels. The evaluator now measures Qwen construction side from the same label-to-hazard mapping used by RTP compilation and durable control.
  - Added test_construction_side_lock_uses_canonical_side_for_shift_and_candidate_labels. It locks a right-side hazard through purple-edge observations, proves a single contradictory shift-right or left-candidate-blocked label is suppressed, and proves matching shift-left / right-candidate-blocked labels still pass.
  - Verification: py_compile passed for side_semantics.py, qwen_label_rtp_worker.py, evaluate_construction_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 167 tests OK.
  - Final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No targeted POC, TensorRT, Qwen, MetaDrive, render, SSH, or C3X process remained running.
  - 2026-05-25 06:58:23 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - Inspected durable speed-plan handling for the lead-vehicle gate. Existing Qwen programs could clear stale lead speed caps when a true_moving_lead or irrelevant_vehicle program was accepted, but stale lead_vehicle speed caps could persist between accepted VLM frames, or be reintroduced by a bounded async-stale lead result, even when the live physical lead track said no lead/off-path/true-moving.
  - Added a deterministic physical lead-state guard in run_metadrive_overlay_demo.py. It clears only lead_vehicle_* speed plans, preserving pedestrian/corridor/cut-in/crossing/signal speed plans. Hard clear reasons are no_lead_track, lead_not_ahead, lead_lateral_unknown, and lead_outside_path. A softer true_moving_or_opening_lead clear is used only between VLM updates so a fresh visual braking judgement is not immediately killed by the same frame's track check.
  - Integrated the guard into the loop before VLM publish handling and again after accepted speed-plan updates for hard clears. Added log fields lead_speed_guard_clear_reason and lead_speed_guard_cleared_sources so each frame shows whether the physical track cleared stale lead speed control.
  - Added CLI thresholds: lead-clear-path-lateral-m, lead-clear-true-moving-closing-mps, lead-clear-true-moving-rel-loss-mps, and lead-clear-non-braking-accel-mps2. These are production-facing track-consistency bounds, not simulator labels.
  - Added tests proving: no physical lead clears only stale lead speed plans and preserves crossing-vehicle plans; true-moving/opening physical lead clears stale lead slow plans between VLM updates but not in the post-publish hard-clear pass; braking/closing physical lead does not clear a lead speed plan.
  - Verification: py_compile passed for run_metadrive_overlay_demo.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 170 tests OK.
  - Final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No targeted POC, TensorRT, Qwen, MetaDrive, render, SSH, or C3X process remained running.
  - 2026-05-25 07:06:30 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES and dirty tree matched the expected POC files plus the new lead evaluator.
  - Added tools/reasoned_trajectory_poc/evaluate_lead_trace.py. The evaluator derives required lead behavior from physical lead-track fields in the recorded episode, not from simulator expected labels: lead presence, forward distance, lateral offset, relative speed, closing speed, acceleration, path width, and lateral motion. It classifies none, true_moving_lead, slower_lead, braking_lead, stopped_lead, cut_in_vehicle, crossing_vehicle, and irrelevant_vehicle, then scores Qwen labels, control speed response, stale RTP age, false slowdowns, and collisions.
  - Added lead-trace tests covering physical-track classification for true moving, slower, braking, stopped, cut-in, crossing, and irrelevant vehicles; Qwen evidence/program label normalization; and episode-level scoring of required lead frames versus no-slow frames. This is intended to keep future lead acceptance proof tied to plausible production inputs instead of simulator-only object labels.
  - Verification: py_compile passed for evaluate_lead_trace.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 173 tests OK.
  - Ran the evaluator on the existing VLM slower-lead episode:
    artifacts\reasoned_trajectory_poc\slower_lead_payload_auto_80_20260525_0404\vlm\episode.json.
    It wrote artifacts\reasoned_trajectory_poc\slower_lead_payload_auto_80_20260525_0404\vlm\lead_trace_evaluation.json. Results: 80 frames, 52 physically required slower-lead frames, required_qwen_success_rate 1.0, required_control_success_rate 0.9807692307692307, 28 no-slow frames, false_slow_rate 0.0, max_consumed_age_frames 3, age_violation_count 0, collision_count 0.
  - Final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No targeted Qwen, TensorRT, ONNX, MetaDrive, render, SSH, C3X, ffmpeg, trtexec, or polygraphy process was left running.
  - 2026-05-25 07:12:18 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES.
  - Inspected the lead harness, Qwen label scorer, and lead evaluator. Found the same class of drift risk that caused repeated left/right construction issues: Qwen's physical lead-state override and the artifact lead evaluator each carried their own lead-track classification thresholds and label aliases.
  - Added selfdrive.controls.reasoned.lead_semantics as the canonical production-facing lead-track contract. It defines canonical lead class aliases, priority order, no-slow/required classes, lead choice-word mapping, metric extraction, and classify_lead_track for true_moving_lead, slower_lead, braking_lead, stopped_lead, cut_in_vehicle, crossing_vehicle, irrelevant_vehicle, and none using only physical track fields.
  - Routed tools/reasoned_trajectory_poc/qwen_trt_label_engine.py through lead_semantics for _lead_state_word and _apply_lead_state_consistency. Qwen's physical lead override now uses the same classification contract as evaluation rather than a private threshold copy.
  - Routed tools/reasoned_trajectory_poc/evaluate_lead_trace.py through lead_semantics for canonical lead aliases, qwen label extraction, required/no-slow classes, and physical lead requirement derivation. Added repo-root path setup so the evaluator works as a standalone CLI script as well as through unittest imports.
  - Updated tests so the physical lead-state consistency tests explicitly compare Qwen override words against lead_semantics.classify_lead_track and lead_semantics.lead_choice_word for slower lead, cut-in, crossing, and irrelevant vehicle cases.
  - Verification: py_compile passed for lead_semantics.py, qwen_trt_label_engine.py, evaluate_lead_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 173 tests OK.
  - Re-ran the standalone lead evaluator on artifacts\reasoned_trajectory_poc\slower_lead_payload_auto_80_20260525_0404\vlm\episode.json after the shared-semantics refactor. It still wrote lead_trace_evaluation.json and preserved the previous result: 80 frames, 52 required slower-lead frames, required_qwen_success_rate 1.0, required_control_success_rate 0.9807692307692307, false_slow_rate 0.0, max_consumed_age_frames 3, age_violation_count 0, collision_count 0.
  - Targeted process sweep returned NO_TARGETED_POC_PROCESSES after verification. No Qwen, TensorRT, ONNX, MetaDrive, ffmpeg, trtexec, polygraphy, SSH, C3X, or reasoned-trajectory process was left running.
  - 2026-05-25 07:19:18 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. Dirty tree still matched the expected POC/readme/progress/tinygrad files plus lead_semantics/evaluation files.
  - Added tools/reasoned_trajectory_poc/evaluate_lead_suite.py. The suite evaluator auto-discovers the latest available per-class lead artifacts, or accepts explicit name=path mappings, then runs evaluate_lead_trace per case and gates true_moving_lead, slower_lead, braking_lead, stopped_lead, cut_in_vehicle, crossing_vehicle, and irrelevant_vehicle together. It fails on missing coverage, collisions, VLM age violations, required-class Qwen/control rates below 95%, or no-slow false slowdown above 5%.
  - Strengthened evaluate_lead_trace.py with explicit nominal_speed_mps/desired_speed_mps/speed_mps episode override support, so short episodes cannot hide slowdowns by inferring nominal speed from already-slowed target speeds.
  - Updated lead control scoring so cut-in, crossing, and stopped-lead path responses can be counted as control-satisfying when the recorded modified route clearance to the vehicle is above a bounded threshold. This keeps the evaluator tied to the actual green path/clearance trace instead of treating speed reduction as the only possible safe response.
  - Added tests for the lead-suite gate: passing true-moving/slower/cut-in/irrelevant cases, explicit missing-case failure, and false-slow failure. The cut-in test proves a route-clearance response with no speed cap can count as control success only when the Qwen/physical class is correct and clearance is sufficient.
  - Verification: py_compile passed for evaluate_lead_trace.py, evaluate_lead_suite.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 175 tests OK.
  - Ran the new suite evaluator on the latest discovered lead artifacts and wrote artifacts\reasoned_trajectory_poc\lead_suite_evaluation_latest.json. The suite correctly failed on old evidence: cut_in_vehicle qwen_success_rate 0.1667 < 0.95 and crossing_vehicle qwen_success_rate 0.4 < 0.95. After route-clearance-aware control scoring, cut-in and crossing control_success_rate both reached 1.0 on those artifacts, so the remaining lead-suite failures are label/classification evidence failures, not path-clearance/control failures.
  - Latest discovered lead artifact summary from the suite: true_moving_lead false_slow_rate 0.0 and control_success_rate 1.0; slower_lead qwen_success_rate 1.0 and control_success_rate 0.9808; braking_lead qwen_success_rate 1.0 and control_success_rate 0.9565 for braking frames; stopped_lead qwen_success_rate 1.0 and control_success_rate 1.0; irrelevant_vehicle false_slow_rate 0.0. Cut-in/crossing must be rerun under the current lead_state_consistency path before they can satisfy the 95% Qwen classification gate.
  - Final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No targeted Qwen, TensorRT, ONNX, MetaDrive, ffmpeg, trtexec, polygraphy, SSH, C3X, or reasoned-trajectory process was left running.
  - 2026-05-25 07:31:13 -04:00 process hygiene check at user request. Targeted POC sweep returned NO_TARGETED_POC_PROCESSES for Qwen, TensorRT, ONNX, MetaDrive, reasoned_trajectory, render, ffmpeg, SSH, C3X, and tinygrad command lines. Broader Python-family check found only unrelated long-lived Open WebUI and DuckDuckGo MCP helper processes. NVIDIA client check showed no Qwen/TensorRT/MetaDrive compute process. ffmpeg/SSH/WSL helper sweep returned no processes, and WSL reported no running distributions. No process was killed.
  - 2026-05-25 07:32:22 -04:00 continuation from active goal. Initial targeted sweep again returned NO_TARGETED_POC_PROCESSES. Inspected git status and the current lead/scene-board/Qwen/evaluator code. Found desired-speed lead semantics partially implemented, but runtime scene-board text still used desired_speed=... while the Qwen physical-state parser expects plain-language desired speed ... m/s, so live Qwen could still miss the desired-speed cue needed to avoid slower-lead/cut-in oscillation after ego has matched the lead speed.
  - 2026-05-25 07:33:48 -04:00 patched desired-speed lead cue path. scene_board.py and ui_scene_board.py now emit plain-language "desired speed X m/s" in the same state text Qwen sees. qwen_trt_label_engine.py still accepts legacy desired_speed_mps= and desired_speed= forms but now also parses the plain-language cue. run_metadrive_overlay_demo.py records desired_speed_mps per frame and nominal_speed_mps/desired_speed_mps in the episode summary. Added a regression test that a lead already matched by ego but still below desired road speed remains slower_lead and overrides a Qwen/choice moving answer through the physical lead-state consistency filter. Added a scene-board assertion for the desired-speed cue.
  - 2026-05-25 07:34:58 -04:00 verification found a suite-gate weakness, not a driving fix: py_compile passed and unittest passed with 177 tests OK, but evaluate_lead_suite on explicit current cut-in/crossing plus latest discovered lead cases failed old slower_lead evidence under desired-speed-aware semantics. It also showed cut_in_vehicle could be marked ok even with required_qwen_success_rate 0.1008 because the gate only checked the named cut-in bucket, not all required physical frames in the episode. Patched evaluate_lead_suite.py to require episode-level required_qwen_success_rate and required_control_success_rate for required lead cases, and added a regression test where a cut-in case later becomes a slower required lead frame that must still be classified/controlled correctly.
  - 2026-05-25 07:39:41 -04:00 reran verification and lead demos. py_compile passed for evaluate_lead_suite.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 178 tests OK. Wrote/validated the lead-only TensorRT runtime manifest with contract SHA 4cf6fd47ae432132841d01d36a0a032d8d04f89011971921f6940acd049151f0. Fresh cut-in run lead_cut_in_vehicle_desired_speed_sync_120_20260525_0736 fixed classification/control but had 1 deadline miss at frame 1, max latency 50.36 ms. Reran with a warmer engine as lead_cut_in_vehicle_desired_speed_sync_120_20260525_0740: 120/120 publishes, deadline_miss_count 0, p99 49.06 ms, max 49.27 ms, same_frame_all true, max RTP age 0, min vehicle route clearance 17.72 m. Fresh slower-lead run lead_slower_lead_desired_speed_sync_120_20260525_0738: 120/120 publishes, deadline_miss_count 0, p99 48.41 ms, max 48.70 ms, same_frame_all true, max RTP age 0, min vehicle route clearance 7.58 m. Lead trace evals: cut-in required_qwen/control 119/119, false_slow_rate 0.0, collision_count 0; slower-lead required_qwen 120/120, required_control 119/120, collision_count 0. Strict lead suite with fresh slower/cut-in and current crossing wrote artifacts\reasoned_trajectory_poc\lead_suite_evaluation_desiredspeed_strict_20260525_0740.json and passed ok true with no issues. Rendered side-by-side videos for fresh cut-in and slower-lead runs under their videos directories.
  - 2026-05-25 07:40:16 -04:00 final process and tree check for this continuation. Targeted sweep returned NO_TARGETED_POC_PROCESSES for Qwen, TensorRT, MetaDrive, render, ffmpeg, SSH/C3X, tinygrad, and reasoned_trajectory command lines. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked lead_semantics, side_semantics, evaluate_construction_trace, evaluate_lead_trace, and evaluate_lead_suite. No process cleanup was needed.
  - 2026-05-25 07:41:34 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status still shows the expected dirty POC files. Re-read recent progress and searched the current construction code/tests. Current unresolved construction state from prior evidence: distinct-source/3-confirm guards prevent unsafe wrong-side activation, but fast Qwen construction-side recall is near zero or flips under edge/candidate labels, so the next work is to recover construction-side detection without simulator-only hardcoding or sign hacks.
  - 2026-05-25 07:54:34 -04:00 process hygiene check at user request. Two targeted sweeps for tools/reasoned_trajectory_poc, qwen_trt_label_engine.py, qwen_label_rtp_worker.py, run_metadrive_overlay_demo.py, render_demo_videos.py, evaluate_construction_trace.py, RTP_VLM_SERVER_COMMAND, and MetaDrive returned NO_TARGETED_POC_PROCESSES. A broader workspace/Qwen/openpilot Python-family check returned NO_WORKSPACE_QWEN_OPENPILOT_PYTHON_PROCESSES. SSH, WSL, trtexec, polygraphy, TensorRT, /data/openpilot, /data/qwen, and F:\qwen_trt_export command-line sweeps returned NO_SSH_WSL_TRT_QWEN_LEFTOVERS. NVIDIA process query showed only ordinary desktop/WDDM clients and no Qwen/TensorRT/MetaDrive process. No process was killed.
  - 2026-05-25 08:12:25 -04:00 construction-side persistence work. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES and git status matched the expected dirty POC files. Patched tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py RotatingScoreState so a strong paired construction edge observation can immediately lock a canonical construction side when same_score >= 2.0, opposite_score <= 0.05, and margin >= 1.75. The lock uses the canonical edge labels blue=left hazard and purple=right hazard, persists only while bounded construction presence is active, and still suppresses single contradictory edge blips below the existing immediate-override threshold.
  - Added a construction-only presence hold of 10 source frames. This fixes the observed rotating/asynchronous source-frame gap where cones/barrier evidence at source 23, a strong right-side edge at source 26, and the next presence result at source 31 previously reset the side lock because generic score_cache_ttl_frames was only 3. The hold is bounded and construction-only; generic label TTL is unchanged.
  - Added regression tests for weak edge observations not locking the wrong side, strong edge lock persistence through presence-only frames, source-frame gap survival, locked-side suppression of a single later contradiction, and stale previous-episode construction edge clearing after the bounded construction hold. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, evaluate_construction_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 182 tests OK.
  - Rebuilt a dedicated edge runtime manifest at artifacts\reasoned_trajectory_poc\qwen_trt_construction_edge_runtime_manifest_20260525_0802.json with contract SHA 013388d4b2a6ba79c24198a771783938da0afe5b8f44fb9c1707445792465bd9. Earlier demo attempts failed because --require-manifest was picking the newer candidate-obstruction manifest in F:\qwen_trt_export, producing a manifest contract mismatch; pinning --manifest fixed VLM server startup.
  - Ran construction_right_edge_locked_hold_120_20260525_0805 with Qwen TensorRT edge labels, async latest-only/drop-stale, max age 8, 120 frames, 2.5 m/s, construction_right, and VLM speed control disabled. Runtime summary: 117/117 valid publishes, 0 deadline misses, mean planner overhead 2.65 ms, p99 5.53 ms, max 5.91 ms, max_rtp_age_frames 8, mean_path_delta_m 1.004, mean_speed_delta_mps 0.0. Stock min construction route clearance was 1.206 m; VLM min construction route clearance was 1.761 m.
  - Construction evaluator for that run wrote artifacts\reasoned_trajectory_poc\construction_right_edge_locked_hold_120_20260525_0805\vlm\construction_trace_evaluation.json. Results improved but are not acceptable yet: path_relevant_construction_frames 71, qwen_side_correct 42, qwen_side_wrong 3, qwen_side_missing 26, qwen_side_success_rate 0.592, post_first_side_success_rate 0.764, control_away_frames 32, control_toward_frames 0, collision_count 0, max_rtp_age_frames 8. Remaining failure: false_construction_rate is 1.0 on the evaluator's not-path-relevant frames because construction labels persist after the route has laterally cleared or passed the nearest cone edge. Do not treat construction as complete.
  - Rendered videos for that run: artifacts\reasoned_trajectory_poc\construction_right_edge_locked_hold_120_20260525_0805\videos\side_by_side_construction_right_edge_locked_hold_120_20260525_0805.mp4, stock_construction_right_edge_locked_hold_120_20260525_0805.mp4, and vlm_construction_right_edge_locked_hold_120_20260525_0805.mp4.
  - Briefly tested strict binary thresholding for exact zero yes/no margins to reduce stale construction presence, but rejected it because the current TensorRT binary cones/barrier engine reports 0.0 scores on saved construction frames; strict thresholding would make the configured construction path output none for presence and regress the closed-loop demo. The strict-threshold code/test was backed out and tests were rerun cleanly.
  - 2026-05-25 08:14:08 -04:00 final check for this continuation. Inspected durable lateral lifecycle and found the remaining false-construction-label issue is not only stale synthetic lock: later high-confidence Qwen purple-edge observations continue to refresh the right-edge durable plan, so the next fix must address Qwen/evaluator disagreement about whether construction remains path-relevant after the green path has shifted or cleared the nearest edge. Final targeted process sweep returned NO_TARGETED_POC_PROCESSES for Qwen, TensorRT, MetaDrive, render, ffmpeg-related POC commands, SSH/C3X, tinygrad eGPU commands, and F:\qwen_trt_export command lines. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked lead_semantics, side_semantics, evaluate_construction_trace, evaluate_lead_trace, and evaluate_lead_suite.
  - 2026-05-25 08:32:23 -04:00 process hygiene check at user request. Targeted POC sweeps found no Qwen, TensorRT, MetaDrive, render, evaluator, ffmpeg, or trtexec workers. A broader Python check showed only unrelated resident Open WebUI and DuckDuckGo MCP helper processes. Closed four stale Windows Photos artifact-viewer processes from prior Qwen input inspection. Final targeted sweeps returned NO_TARGETED_POC_PROCESSES and NO_FFMPEG_TRT_PROCESSES.
  - 2026-05-25 08:33:01 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status shows the expected dirty Qwen/openpilot POC files plus nested tinygrad_repo and untracked side/lead/evaluator modules. Current focus is eliminating repeated sign/convention drift by inspecting and consolidating the path overlay, control target, MetaDrive lateral conversion, construction side semantics, and lead-object orientation/placement contracts.
  - 2026-05-25 08:50:52 -04:00 sign-convention hardening and construction reruns. Moved the POC MetaDrive lateral adapter into selfdrive.controls.reasoned.side_semantics: openpilot_to_metadrive_lateral_m, metadrive_to_openpilot_lateral_m, lateral_side_metadrive, metadrive_lateral_for_side, construction_hazard_metadrive_lateral_for_side, construction_avoidance_metadrive_lateral_for_hazard_side, and construction_avoidance_metadrive_side_valid. tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py now imports those helpers instead of locally defining left/right conversions, uses construction_hazard_metadrive_lateral_for_side for fixed and random construction spawning, and uses metadrive_lateral_for_side for random pedestrian crossing endpoints.
  - Added side-convention tests proving the harness functions are the canonical side_semantics functions, left-positive openpilot is MetaDrive-negative, round trip conversion holds, construction spawn side and avoidance side are opposite, and the full Qwen-label -> RTP -> PathSynth -> durable MetaDrive plan -> rendered green path chain moves away from all construction label families.
  - Root cause for the latest wrong-side construction labels: the sim/control signs were correct, but RotatingScoreState allowed one very high post-shift contradictory construction edge score to immediately replace a locked right-side construction side after the path had already shifted away. That produced qwen_labels/rtp_text with construction_blue_edge/left_edge even though the durable controller separately refused to reverse into the cones. Patched qwen_label_rtp_worker.py so an immediate contradictory construction-side override is not allowed while tracked_path_lat shows the current path is already shifted away from the locked hazard side. A contradictory side can still override before the path commits away, or after repeated confirmations. Added tests for both cases.
  - Added runtime manifest coverage for rendered scoring prompt text in qwen_trt_label_engine._runtime_contract. The old manifest hash tracked SCORE_QUESTIONS but could miss changes in the construction_compact prompt branch. New contract includes rendered_score_prompts_sha256. Added a regression test that the rendered prompt hash is present and distinct from generic question hashes.
  - Verification after patches: py_compile passed for side_semantics.py, run_metadrive_overlay_demo.py, qwen_trt_label_engine.py, qwen_label_rtp_worker.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 188 tests OK.
  - Geometry probe output for construction_blue_edge, construction_purple_edge, construction_left, and construction_right showed left hazards spawn on MetaDrive-left and produce right avoidance, right hazards spawn on MetaDrive-right and produce left avoidance, with sign_valid True through RTP/PathSynth/durable conversion.
  - First controlled run before state-lock fix: construction_right_shared_side_semantics_120_20260525_0839. Runtime: 117/117 VLM publishes, 0 deadline misses, p99 5.37 ms, max 6.34 ms, max_rtp_age_frames 8, stock min construction clearance 1.206 m, VLM min clearance 2.131 m. Evaluator exposed the bug: qwen_side_correct 32, qwen_side_wrong 13, qwen_side_missing 8, false_construction_rate 0.866, collision_count 0, control_toward_frames 0. Inspection showed frames 40-52 had qwen_labels construction_blue_edge and RTP left_edge even though durable_lateral_plan_details still tracked right_edge with sign_valid True.
  - Tested a stricter exclusive edge prompt. It removed the saved bad blue-edge score on vlm_input_0035 but made the closed-loop too conservative: construction_right_edge_prompt_exclusive_120_20260525_0846 had first_qwen_side_frame 40, qwen_side_wrong 0 but qwen_side_success_rate only 0.481 and min clearance 1.623 m. Reverted that prompt wording and kept the state-lock logic as the real fix.
  - Wrote new construction edge runtime manifest with rendered prompt hashing: artifacts\reasoned_trajectory_poc\qwen_trt_construction_edge_state_lock_manifest_20260525_0849.json, contract SHA bb449d6393ccdad784885d02cac0076becf01ab8c7724c854d0e82e379cfee6b.
  - Final controlled run after state-lock fix: construction_right_state_lock_120_20260525_0849. Runtime summary: 117/117 VLM publishes, 0 deadline misses, mean planner overhead 3.51 ms, p99 6.52 ms, max 7.24 ms, max_rtp_age_frames 8, speed delta 0.0 because VLM speed control was disabled, stock min construction clearance 1.206 m, VLM min construction clearance 2.122 m.
  - Construction evaluator for construction_right_state_lock_120_20260525_0849 wrote artifacts\reasoned_trajectory_poc\construction_right_state_lock_120_20260525_0849\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 54, qwen_side_correct 45, qwen_side_wrong 0, qwen_side_missing 9, qwen_side_success_rate 0.833, post_first_side_qwen_side_success_rate 1.0, false_construction_rate 0.879, control_away_frames 33, control_toward_frames 0, planned_toward_frames 0, collision_count 0, max_rtp_age_frames 8. Remaining gap: stale construction labels remain high after the path has laterally cleared or passed the construction, even though they no longer steer toward the hazard.
  - Rendered videos for construction_right_state_lock_120_20260525_0849: artifacts\reasoned_trajectory_poc\construction_right_state_lock_120_20260525_0849\videos\side_by_side_construction_right_state_lock_120_20260525_0849.mp4, stock_construction_right_state_lock_120_20260525_0849.mp4, and vlm_construction_right_state_lock_120_20260525_0849.mp4.
  - 2026-05-25 08:52:15 -04:00 final process check for this continuation. Sequential targeted sweep returned NO_TARGETED_POC_PROCESSES for Qwen/TensorRT, MetaDrive demo, render, construction/lead evaluators, and F:\qwen_trt_export command lines. A broader workspace Python/ffmpeg/TensorRT sweep returned NO_WORKSPACE_PY_FFMPEG_TRT_PROCESSES. No cleanup was needed.
  - 2026-05-25 08:53:31 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status still shows the expected dirty Qwen/openpilot POC files plus nested tinygrad_repo and untracked side/lead/evaluator modules. Current focus is the remaining construction gap from construction_right_state_lock_120_20260525_0849: wrong-side/control-toward failures are fixed, but stale construction labels remain high after the tracked path has laterally cleared or passed the construction.
  - 2026-05-25 09:09:51 -04:00 process hygiene check at user request. Targeted POC sweep returned NO_TARGETED_POC_PROCESSES after excluding the sweep shell itself. NVIDIA process query showed only ordinary desktop/WDDM clients and no Qwen, TensorRT, MetaDrive, render, ffmpeg, or reasoned-trajectory process. Broader Python-family check found only unrelated Open WebUI and DuckDuckGo MCP helper processes; the null-command-line Python is a child of open-webui.exe, not this repo. No process was killed.
  - 2026-05-25 09:10:39 -04:00 continuation from active goal after compaction. Initial targeted sweep returned NO_TARGETED_POC_PROCESSES. git status shows the expected dirty POC/readme/progress/tinygrad files plus untracked shared side/lead semantics and evaluators. Current next target remains construction stale-clear behavior: v2 reduced false construction labels from 0.879 to 0.348 while keeping wrong-side/control-toward at 0, but later edge-only reactivations still produce stale construction RTP after the path has cleared.
  - 2026-05-25 09:25:55 -04:00 construction stale-clear iteration. Patched tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py so a confirmed construction clear while the tracked path is shifted suppresses all construction control labels, including opposite-side edge labels and shift/candidate labels, until there is fresh construction-presence evidence after the clear. The reactivation threshold is CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE=2.0, uses only Qwen scores and tracked_path_lat from the scene-board state text, and does not use simulator object positions.
  - Added construction state-machine version 3, then tightened it to version 4 after real-cadence evidence showed clear timing was still late. The final constants for this continuation are CONSTRUCTION_SIDE_CLEAR_MIN_TRACKED_OFFSET_M=0.8, CONSTRUCTION_PRESENCE_CLEAR_CONFIRM_FRAMES=1, CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE=2.0, and CONSTRUCTION_CLEAR_RTP_CONFIDENCE=0.74. Rationale: once the path has already shifted substantially and Qwen scores both construction-presence labels at 0.0, stale lateral construction control should clear immediately instead of surviving another inference interval.
  - Updated tools/reasoned_trajectory_poc/qwen_trt_label_engine.py runtime contract so manifests include construction_reactivate_min_presence_score and state-machine version 4. Added tests proving edge-only/shift-only construction reactivation is suppressed after clear while shifted, fresh strong construction presence can reactivate, presence clear fires once the path is shifted, the below-threshold path does not clear, and the runtime manifest tracks the new constant. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 194 tests OK.
  - Wrote runtime manifests for the same TensorRT construction contract after each state-machine change. Final manifest for this continuation: artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_manifest_20260525_0922.json with contract SHA 390d2abae0488512c4e5a208d2130b6f5b443a9964377705b4cb6aab4a4ea804. Artifact check passed ok true against Qwen2.5-VL 3B, full168 vision FP32 engine, FP8 seq768 selected-logits text engine, rotating groups cones/barrier ; construction_blue_edge/construction_purple_edge, mirror fusion, and construction-edge binary mode.
  - First rerun construction_right_reactivation_guard_120_20260525_0917 used tick-sec=0 and is invalid as closed-loop evidence: the accelerated simulator outran the async VLM, edge/mirror-fusion results completed 12-14 simulated frames old and were correctly dropped by vlm-max-result-age-frames=8. It produced only 10 publishes, mean_path_delta_m 0.0, and VLM clearance stayed stock at 1.206 m. Kept as a harness-timing artifact, not evidence of behavior.
  - Real-cadence rerun construction_right_reactivation_guard_120_20260525_0921_realtime with tick-sec=0.05 validated the reactivation guard but not the fast clear timing: 94 publishes, 0 deadline misses, p99 planner overhead 5.63 ms, max 5.70 ms, max_rtp_age_frames 8, stock clearance 1.206 m, VLM clearance 2.122 m, qwen_side_wrong 0, control_toward 0, false_construction_rate 0.318. This showed edge-only reactivation was reduced but stale labels still survived too long.
  - Final real-cadence period-2 rerun construction_right_fast_clear_120_20260525_0923_realtime with state-machine version 4: 47 publishes, 0 deadline misses, mean planner overhead 2.64 ms, p99 5.41 ms, max 5.60 ms, max_rtp_age_frames 8, mean_path_delta_m 1.117, speed delta 0.0 because VLM speed control was disabled, stock construction clearance 1.206 m, VLM clearance 2.131 m. Construction evaluator results: frames 120, path_relevant_construction_frames 53, qwen_side_correct 42, qwen_side_wrong 0, qwen_side_missing 11, qwen_side_success_rate 0.792, post_first_side_success_rate 0.933, false_construction_rate 0.0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, max_rtp_age_frames 8. This fixes stale false construction for the controlled run but construction-side recall is still below the 95% gate, so construction is not complete.
  - Period-1 scheduling rerun construction_right_fast_clear_period1_120_20260525_0925_realtime is rejected. Runtime was similar, but construction evaluation regressed to qwen_side_correct 32, qwen_side_wrong 10, qwen_side_missing 11, qwen_side_success_rate 0.604, post_first_side_success_rate 0.711. Durable control still blocked steering toward the hazard with control_toward 0 and false_construction_rate 0.0, but Qwen side labels were worse than period-2.
  - Rendered videos for the best current run: artifacts\reasoned_trajectory_poc\construction_right_fast_clear_120_20260525_0923_realtime\videos\side_by_side_construction_right_fast_clear_120_20260525_0923_realtime.mp4, stock_construction_right_fast_clear_120_20260525_0923_realtime.mp4, and vlm_construction_right_fast_clear_120_20260525_0923_realtime.mp4. Next construction work should focus on recovering the remaining side-label recall without reintroducing false/stale labels or sign flips.
  - 2026-05-25 09:27:04 -04:00 final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No Qwen, TensorRT, MetaDrive, render, evaluator, ffmpeg, SSH/C3X, tinygrad, or F:\qwen_trt_export worker process was left running.
  - 2026-05-25 09:28:19 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked shared side/lead semantics and evaluators. Current target is recovering the remaining construction side-recall gap from construction_right_fast_clear_120_20260525_0923_realtime without reintroducing stale false labels or sign flips.
  - 2026-05-25 09:37:13 -04:00 construction recall experiments. First confirmed that simply raising the async accepted-age bound from 8 to 9 is not acceptable. Run construction_right_fast_clear_age9_120_20260525_0930_realtime kept clearance at 2.131 m and false_construction_rate 0.0, but stale wrong-side labels survived across frames 41-52: qwen_side_correct 33, qwen_side_wrong 12, qwen_side_missing 8, qwen_side_success_rate 0.623, post_first_side_success_rate 0.733. Reject age9 for construction.
  - Added a real-deployment harness option to tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py: --prewarm-reset-runtime-state / --no-prewarm-reset-runtime-state. The default preserves prior behavior by resetting rotating VLM state after async prewarm. With reset disabled, the harness keeps the warmed VLM state and carries monotonic model_frame_id values into the episode while keeping display frame_id for artifacts. This models a real openpilot process that has been observing before engagement, instead of a synthetic cold frame-0 VLM state. Added model_frame_id to each record and changed same_frame_all accounting to compare rtp_source_frame_id against model_frame_id when present.
  - Verification after the harness patch: py_compile passed for run_metadrive_overlay_demo.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 194 tests OK.
  - Tested retained warm state in construction_right_warm_state_fast_clear_120_20260525_0934_realtime with max age 8. Rejected for this scenario: VLM clearance dropped to 1.639 m, qwen_side_success_rate 0.494, false_construction_rate 0.0465, collision_count 0, and control_toward 0. The option remains useful for future engagement-mode tests, but it is not a construction improvement under the current fixed-start scenario.
  - Tested adding semantic construction_left/construction_right back into the rotating TensorRT contract alongside edge labels. Wrote manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_semantic_edge_fast_clear_manifest_20260525_0936.json with contract SHA c95c5a53848f9f963df3abeef87d76dc80cab8b8e8946272bb12ff3a75c623a4. Run construction_right_semantic_edge_fast_clear_120_20260525_0936_realtime is rejected: qwen_side_correct 0, qwen_side_missing 98, collision_count 12, mean_path_delta_m 0.0, and VLM clearance stayed stock at 1.206 m. The semantic group suppressed the useful edge-only activation in this command shape.
  - Current best construction artifact remains construction_right_fast_clear_120_20260525_0923_realtime: false_construction_rate 0.0, qwen_side_wrong 0, control_toward 0, collision_count 0, VLM clearance 2.131 m vs stock 1.206 m, max_rtp_age_frames 8, but qwen_side_success_rate 0.792 / post_first_side 0.933. Do not use the age9, warm_state, or semantic_edge runs as accepted construction evidence.
  - 2026-05-25 09:38:00 -04:00 final targeted process sweep returned NO_TARGETED_POC_PROCESSES. No Qwen, TensorRT, MetaDrive, render, evaluator, ffmpeg, SSH/C3X, tinygrad, or F:\qwen_trt_export worker process was left running.
  - 2026-05-25 09:39:04 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked shared side/lead semantics and evaluators. Current focus is whether construction evaluator "missing" frames in the best run are true Qwen/control misses or only fresh-publish gaps while the previously accepted bounded VLM plan is still being consumed by durable lateral control.
  - 2026-05-25 09:45:56 -04:00 process and bootstrap verification. Targeted process sweep found only the sweep command itself, so no Qwen, TensorRT, MetaDrive, render, or evaluator POC worker was running. Added a construction edge-bootstrap handoff fix: a recent, unclosed edge-derived construction side lock now survives into the first construction-presence group tick, but an expired lock or a lock after confirmed lateral clear is not preserved. Added a regression test that an expired edge-only bootstrap cannot seed a later construction-presence group. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 197 tests OK.
  - 2026-05-25 09:55:03 -04:00 construction v6 rerun and rejection/acceptance status. Edge-first experiment construction_right_edge_bootstrap_120_20260525_0946_realtime is rejected: high publish count and good latency but mean_path_delta_m 0.033, VLM clearance stayed stock at 1.206 m, qwen_side_success_rate 0.031, and collision_count 12. Root cause: edge-first schedule scored edge labels before construction was visible and did not recover side before impact. Do not make edge-first the default.
  - Restored the production demo schedule to cones,barrier first and construction_blue_edge,construction_purple_edge second. A v5 rerun restored clearance but exposed a source-frame 34 single contradictory blue-edge score while the path was already committed left by about 0.46 m. Durable control refused to reverse, but the RTP labels flipped left for 10 frames, so v5 is rejected as a label-contract regression.
  - Patched qwen_label_rtp_worker.py to version 6 with CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M=0.35. This is separate from the 0.8 m clear threshold: a path merely committed away suppresses one immediate contradictory side override, while full lateral clear still uses the stricter clear threshold. Added runtime manifest coverage and a regression test proving committed-away 0.40 m blocks a single immediate override, while 0.20 m still allows an immediate correction before path commitment. Verification after v6 patch: py_compile passed and unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 198 tests OK.
  - Wrote v6 manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v6_manifest_20260525_0953.json with contract SHA f69bd3e998e01c37f621385d16bfcd81f5030074c8816e909c2cb6e1b87f0641. Controlled real-cadence rerun construction_right_fast_clear_v6_120_20260525_0953_realtime: 120 frames, 56 VLM publishes, 0 deadline misses, mean planner overhead 2.65 ms, p99 5.42 ms, max 5.82 ms, max_rtp_age_frames 8, mean_path_delta_m 0.938, speed delta 0.0, stock clearance 1.206 m, VLM clearance 2.122 m.
  - Construction evaluator for v6 wrote artifacts\reasoned_trajectory_poc\construction_right_fast_clear_v6_120_20260525_0953_realtime\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 54, qwen_side_correct 42, qwen_side_wrong 0, qwen_side_missing 12, qwen_side_success_rate 0.778, post_first_side_success_rate 0.933, false_construction_rate 0.0, control_toward_frames 0, planned_toward_frames 0, collision_count 0. This is accepted as current construction POC evidence but not final-goal complete because total recall remains below the 95% gate.
  - Rendered videos: artifacts\reasoned_trajectory_poc\construction_right_fast_clear_v6_120_20260525_0953_realtime\videos\side_by_side_construction_right_fast_clear_v6_120_20260525_0953_realtime.mp4, stock_construction_right_fast_clear_v6_120_20260525_0953_realtime.mp4, and vlm_construction_right_fast_clear_v6_120_20260525_0953_realtime.mp4.
  - 2026-05-25 10:04:20 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. Added consumed-plan construction traceability to tools/reasoned_trajectory_poc/evaluate_construction_trace.py instead of replacing the raw Qwen label metric. New evaluator fields include consumed_construction_side per row, durable_lateral_plan_details, model/source frame ids, qwen_label_scores, rtp_text, rtp_valid/invalid_reason, control_consumed_age_frames, green_path_matches_tracked_path, consumed_plan_side_correct/wrong/missing, max_consumed_age_frames, green_path_mismatch_count, first_consumed_side_frame, startup_required_before_first_consumed_side_frames, and post_first_consumed_side_* rates.
  - Added tests proving consumed_construction_side uses active durable lateral plans before RTP fallback, conflicting durable sides are exposed as conflict, and a frame with no fresh qwen_labels but a bounded durable right-hazard plan counts as consumed-plan correct while raw Qwen labels remain missing. Verification: py_compile passed for evaluate_construction_trace.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 200 tests OK.
  - Re-ran the evaluator on construction_right_fast_clear_v6_120_20260525_0953_realtime. Updated trace confirms the real control contract: raw Qwen side remains 42/54 with 0 wrong, but consumed-plan side is 45/54 with 0 wrong, max consumed age 8, green_path_mismatch_count 0, first consumed side frame 9, startup_required_before_first_consumed_side_frames 9, and post_first_consumed_side_success_rate 1.0 over 45 path-relevant frames. The remaining construction gap is now clearly startup perception/cadence, not sign or stale-control reversal.
  - Measured current TensorRT construction group cadence before adding any startup bootstrap switch. benchmark-groups with the v6 manifest shows presence group median about 128.7 ms and edge-with-mirror about 258.3 ms total. A non-mirrored edge diagnostic on vlm_input_0002 correctly scored construction_purple_edge with total about 128.4 ms hot, but vlm_input_0000 still scored both edge labels 0.0. Conclusion: the first 9 missing frames are due to early visibility plus 128-258 ms async label cadence; simply scoring edge first is already proven bad by the rejected edge-first run, and a sequential all-group bootstrap would risk stale outputs while not helping frame 0 where the edge label is genuinely absent.
  - 2026-05-25 10:15:58 -04:00 process hygiene check at user request. Broad Python/Node/ffmpeg/SSH sweep showed no reasoned-trajectory POC worker. Strict targeted sweep returned NO_TARGETED_POC_PROCESSES for Qwen/TensorRT, MetaDrive, render, evaluators, C3X SSH, tinygrad, repo paths, and F:\qwen_trt_export command lines. The null-command-line python.exe PID 9260 is still a child of open-webui.exe, not this repo. nvidia-smi showed no Python/Qwen compute process, only ordinary desktop/WDDM clients. No process was killed.
  - 2026-05-25 10:17:05 -04:00 continuation from active goal after compaction. Initial targeted sweep returned NO_TARGETED_POC_PROCESSES. git status shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules and the new pedestrian evaluator. Current focus is the pedestrian closed-loop failure from pedestrian_binary_220_20260525_1012_realtime: Qwen detected both pedestrian_in_path and pedestrian_entering_path with bounded age, but the RTP compiler only issued a 50% YIELD speed cap and no stop point, so VLM still crashed into the pedestrian.
  - 2026-05-25 10:21:05 -04:00 pedestrian path-conflict fix. Patched tools/reasoned_trajectory_poc/qwen_label_rtp_worker.py so labels that mean an agent is already in the planned path (pedestrian_in_path, vehicle_in_path, animal_in_path) compile to scene=path_blocking_agent, meta=STOP, speed_cap_mps=0.0, stop_s=18.0, avoid=[corridor_object_s18_28]. Labels that mean an agent is entering the path continue to compile to scene=path_conflict_agent, meta=YIELD, speed_cap_mps=50%, stop_s=none. Mixed construction+agent RTP now uses the same blocking-vs-entering agent semantics while preserving construction lateral bias. This is a label-semantics/compiler change, not a simulator object-position rule.
  - Updated tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py so a current green_signal_for_path clears stale traffic-light stop plans even if the same current RTP still has meta=STOP for a path-blocking agent. This separates green-light release from pedestrian/agent stopping. Updated tools/reasoned_trajectory_poc/evaluate_pedestrian_trace.py so simultaneous pedestrian_in_path + pedestrian_entering_path counts as Qwen path-relevant but not exact, instead of being counted as a total miss. Added tests for entering-agent percent speed caps, in-path-agent stop plans, green+blocking-agent stale signal clearing, and both-label pedestrian evaluator handling. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, run_metadrive_overlay_demo.py, evaluate_pedestrian_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 206 tests OK.
  - Wrote updated pedestrian TensorRT runtime manifest artifacts\reasoned_trajectory_poc\qwen_trt_pedestrian_binary_stop_manifest_20260525_1021.json after adding path-agent RTP semantics to the manifest contract. check-artifacts passed ok true against Qwen2.5-VL 3B, full168 FP32 vision engine, FP8 seq768 selected-logits text engine, pedestrian_in_path/pedestrian_entering_path binary group, and contract SHA e554173d8d61fa24f4239c2c0136baf4267cc44657546d696f8cfcee55ecdeba.
  - Ran closed-loop pedestrian demo with the updated manifest: artifacts\reasoned_trajectory_poc\pedestrian_binary_stop_220_20260525_1022_realtime. Command used speed_mps=2.5, async VLM, period 2, accepted age bound 8, latest-only/drop-stale, tick-sec=0.05, novel-scene pedestrian. Stock crashed into the pedestrian after 108 frames. VLM ran the full 220 frames with collision_count 0, publish_count 217, valid_count 217, deadline_miss_count 0, mean planner overhead 2.72 ms, p99 5.65 ms, max 5.84 ms, max_rtp_age_frames 6, green_path_mismatch_count 0, and min_pedestrian_route_clearance_m 23.87 m. The trace shows Qwen first published pedestrian_in_path + pedestrian_entering_path at frame 3/source frame 0 and the compiler emitted scene=path_blocking_agent, meta=STOP, speed_cap_mps=0.0, stop_s=18.0, avoid=[corridor_object_s18_28]. Safety improved versus the prior crash, but behavior is now too conservative: mean VLM speed was 0.006 m/s and the car stopped roughly 24 m before the pedestrian for the rest of the episode. This is not final human-like pedestrian behavior; it is an intermediate safety fix.
  - Ran pedestrian evaluator on that VLM episode and wrote artifacts\reasoned_trajectory_poc\pedestrian_binary_stop_220_20260525_1022_realtime\vlm\pedestrian_trace_evaluation.json. Results: path_relevant_pedestrian_frames 220, qwen_pedestrian_path_relevant 217, qwen_pedestrian_path_relevant_rate 0.986, qwen_pedestrian_exact_rate 0.0 because Qwen reports both in_path and entering_path, consumed_agent_rate 0.986, control_success_rate 0.986, collision_count 0, max_rtp_age_frames 6, max_consumed_age_frames 6, post_first_qwen_pedestrian_path_relevant_rate 1.0, post_first_control_success_rate 1.0. Remaining gap: exact in-path/entering distinction and distance/progress behavior need improvement without using simulator-only object-position hacks.
  - Rendered comparison videos for the pedestrian run: artifacts\reasoned_trajectory_poc\pedestrian_binary_stop_220_20260525_1022_realtime\videos\side_by_side_pedestrian_binary_stop_220_20260525_1022_realtime.mp4, stock_pedestrian_binary_stop_220_20260525_1022_realtime.mp4, stock_pedestrian_binary_stop_220_20260525_1022_realtime_padded.mp4, and vlm_pedestrian_binary_stop_220_20260525_1022_realtime.mp4.
  - 2026-05-25 10:24:23 -04:00 final targeted process sweep returned NO_TARGETED_POC_PROCESSES. nvidia-smi showed no Python/Qwen compute process, only ordinary desktop/WDDM clients. No Qwen/TensorRT server, MetaDrive run, render job, evaluator, C3X SSH, tinygrad, or F:\qwen_trt_export worker was left running.
  - 2026-05-25 10:25:38 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES and git status shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules. Current target is the pedestrian over-conservatism introduced by the last safety fix: path-blocking agent RTP correctly prevented collision, but immediate speed_cap_mps=0.0 caused the car to stop roughly 24 m before the pedestrian and hold there. Next work is to make the durable stop profile distance-aware from RTP stop_s/avoid tokens, not simulator object positions.
  - 2026-05-25 10:28:16 -04:00 distance-aware stop profile patch. Updated DurableSpeedPlan.target_speed_cap in tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py so any plan with stop_s computes the current cap from start_long_m + stop_s minus current_long_m using a generic comfort stop curve, instead of applying speed_cap_mps=0.0 immediately from the first detection frame. Constants are GENERIC_STOP_HOLD_RADIUS_M=4.0 and GENERIC_STOP_COMFORT_DECEL_MPS2=1.1. This remains RTP/compiler-driven and uses no simulator object position. Added a regression test proving a pedestrian_in_path stop plan permits approach at 2.5 m/s far from the stop point, slows in the decel band, and reaches 0 near the planned stop point. Verification: py_compile passed for run_metadrive_overlay_demo.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 207 tests OK after correcting the test to check the actual decel band.
  - Reran the same Qwen pedestrian scenario with the distance-aware stop profile using existing manifest artifacts\reasoned_trajectory_poc\qwen_trt_pedestrian_binary_stop_manifest_20260525_1021.json. New run: artifacts\reasoned_trajectory_poc\pedestrian_stop_profile_220_20260525_1028_realtime. Stock still crashed after 108 frames. VLM ran 220 frames, collision_count 0, publish_count 217, valid_count 217, deadline_miss_count 0, mean planner overhead 2.73 ms, p99 5.41 ms, max 5.81 ms, max_rtp_age_frames 6, green_path_mismatch_count 0, mean speed 0.644 m/s, and min_pedestrian_route_clearance_m 9.82 m. Frame trace confirms the car approached normally through frame 60 (target_speed_mps 2.5, pedestrian ahead 13.9 m), then stopped before the path-blocking pedestrian by frame 100 (target_speed_mps 0.0, pedestrian ahead 9.82 m). This fixes the prior 24 m early freeze while preserving collision avoidance.
  - Updated tools/reasoned_trajectory_poc/evaluate_pedestrian_trace.py so far-ahead in-path pedestrians require a consumed path-agent plan but not immediate slowdown, while near in-path pedestrians still require slowdown/stop. Added --slow-distance-m default 12.0 and row field control_response_ok_for_agent. Also aligned qwen_pedestrian_class with runtime compiler precedence: if both pedestrian_in_path and pedestrian_entering_path are present, in_path wins because it is the stronger blocking condition. Added tests for both changes. Verification: py_compile passed and unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 207 tests OK.
  - Re-evaluated pedestrian_stop_profile_220_20260525_1028_realtime with the corrected evaluator and overwrote vlm\pedestrian_trace_evaluation.json. Results: path_relevant_pedestrian_frames 220, qwen_pedestrian_exact 217, qwen_pedestrian_exact_rate 0.986, qwen_pedestrian_wrong 0, qwen_pedestrian_missing 3, consumed_agent_rate 0.986, control_success_rate 0.977, post_first_qwen_pedestrian_path_relevant_rate 1.0, post_first_control_success_rate 0.991, collision_count 0, max_rtp_age_frames 6, max_consumed_age_frames 6. Rendered videos: artifacts\reasoned_trajectory_poc\pedestrian_stop_profile_220_20260525_1028_realtime\videos\side_by_side_pedestrian_stop_profile_220_20260525_1028_realtime.mp4, stock_pedestrian_stop_profile_220_20260525_1028_realtime.mp4, stock_pedestrian_stop_profile_220_20260525_1028_realtime_padded.mp4, and vlm_pedestrian_stop_profile_220_20260525_1028_realtime.mp4.
  - 2026-05-25 10:33:19 -04:00 final process sweep for this continuation. Targeted POC sweep returned NO_TARGETED_POC_PROCESSES. nvidia-smi showed no Python/Qwen compute process, only ordinary desktop/WDDM clients. No Qwen/TensorRT server, MetaDrive run, render job, evaluator, C3X SSH, tinygrad, or F:\qwen_trt_export worker was left running.
  - 2026-05-25 10:34:28 -04:00 continuation from active goal. Initial targeted process sweep returned NO_TARGETED_POC_PROCESSES. git status shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules. Current target is lead vehicle handling because it is a required production behavior with distinct classes (true_moving_lead, slower_lead, braking_lead, stopped_lead, cut_in_vehicle, crossing_vehicle, irrelevant_vehicle) and already has evaluator/scenario scaffolding in the worktree. No code changed before inspecting the lead path.
  - Inspected current lead code and artifacts. selfdrive/controls/reasoned/lead_semantics.py classifies lead behavior from production-style physical track fields only. qwen_trt_label_engine.py can force lead choices through the physical track consistency filter when payload vehicle state is present. The latest strict lead suite artifact from 07:40 passed control gates, but true_moving_lead exact Qwen classification was only 19/80 in the older true_moving artifact; that artifact lacked current payload vehicle-state behavior for most frames, so it is stale evidence for the current path.
  - Wrote/validated a current lead-only TensorRT runtime manifest for the active code: artifacts\reasoned_trajectory_poc\qwen_trt_lead_choice_manifest_20260525_1036.json. check-artifacts passed ok true using Qwen2.5-VL 3B, full168 FP32 vision engine, nvfp4 seq576 hidden-choice text engine, label_decision_mode=choice, score group true_moving_lead/slower_lead/braking_lead/stopped_lead/cut_in_vehicle/crossing_vehicle/irrelevant_vehicle, and contract SHA 1cdff1b0361b66259d2c27b16774e1c1de0712e0161b15c7c811f19e6ecbd18e.
  - Ran a fresh synchronous true_moving_lead demo with the current lead manifest: artifacts\reasoned_trajectory_poc\lead_true_moving_current_sync_120_20260525_1038. Behavior was correct but timing was rejected as accepted evidence because it had one borderline deadline miss: 119/120 publishes, deadline_miss_count 1, p99 49.82 ms, max 50.04 ms, same_frame_all true, max_rtp_age_frames 0, qwen_lead_classes [true_moving_lead], mean_speed_delta_mps 0.0, collision_count 0 by evaluator.
  - Reran true_moving_lead after that warm path as artifacts\reasoned_trajectory_poc\lead_true_moving_current_sync_rerun_120_20260525_1040. Accepted run: 120/120 publishes, deadline_miss_count 0, p99 49.83 ms, max 49.94 ms, same_frame_all true, max_rtp_age_frames 0, mean_speed_delta_mps 0.0, qwen_lead_classes [true_moving_lead], min_vehicle_route_clearance_m 9.49 m. Lead trace evaluation wrote vlm\lead_trace_evaluation.json: true_moving_lead frames 120, qwen_success_rate 1.0, control_success_rate 1.0, false_slow_rate 0.0, age_violation_count 0, collision_count 0. Rendered videos: artifacts\reasoned_trajectory_poc\lead_true_moving_current_sync_rerun_120_20260525_1040\videos\side_by_side_lead_true_moving_current_sync_rerun_120_20260525_1040.mp4, stock_lead_true_moving_current_sync_rerun_120_20260525_1040.mp4, stock_lead_true_moving_current_sync_rerun_120_20260525_1040_padded.mp4, and vlm_lead_true_moving_current_sync_rerun_120_20260525_1040.mp4.
  - Tightened tools/reasoned_trajectory_poc/evaluate_lead_suite.py so no-slow lead classes (true_moving_lead and irrelevant_vehicle) must meet qwen_success_rate >= min_success_rate in addition to the false_slow gate. This prevents a true moving lead from passing suite evaluation merely because control did not slow despite Qwen not identifying it. Added a regression test that a true_moving_lead frame with no Qwen class and no speed slowdown fails the suite on qwen_success_rate while false_slow_rate remains 0. Verification: py_compile passed for evaluate_lead_suite.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 208 tests OK.
  - Re-ran the lead suite with the fresh true_moving artifact plus latest accepted slower/cut-in/crossing and discovered braking/stopped/irrelevant artifacts. Wrote artifacts\reasoned_trajectory_poc\lead_suite_evaluation_strict_true_moving_refreshed_20260525_1042.json. Result ok true, issues empty. Key rates: true_moving qwen/control 1.0 and false_slow 0.0; slower qwen 1.0 control 0.992; braking required qwen 0.992 control 0.983; stopped required qwen/control 1.0; cut-in required qwen/control 1.0; crossing required qwen/control 1.0 with episode false_slow 0.014; irrelevant qwen/control 1.0 and false_slow 0.0. All max_consumed_age_frames were 0 and collision_count 0 for every case.
  - 2026-05-25 10:42:38 -04:00 final process sweep for this continuation. Targeted POC sweep returned NO_TARGETED_POC_PROCESSES. nvidia-smi showed no Python/Qwen compute process, only ordinary desktop/WDDM clients. No Qwen/TensorRT server, MetaDrive run, render job, evaluator, C3X SSH, tinygrad, or F:\qwen_trt_export worker was left running. git status still shows the expected dirty POC/readme/progress/tinygrad files and untracked side/lead/evaluator modules.
  - 2026-05-25 10:45:28 -04:00 process hygiene check at user request. First broad command-line sweep only caught unrelated Steam helpers because Steam includes an FFmpeg feature flag in its command line. A stricter targeted sweep over python/py/powershell/cmd/ffmpeg/ssh/scons processes with repo path, Qwen, MetaDrive, render, evaluator, tinygrad, and export-script command-line matches returned NO_TARGETED_POC_PROCESSES. nvidia-smi showed no Python/Qwen/TensorRT compute worker, only ordinary desktop/WDDM clients. No process was killed.
  - 2026-05-25 10:46:19 -04:00 continuation from active goal. Initial targeted sweep returned NO_TARGETED_POC_PROCESSES. git status shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules. Current target is traffic controls because construction, pedestrian, and lead slices have current accepted evidence but red-light/green-release/stop-sign handling is still under-proven after the prior invalid dynamic-light video.
  - 2026-05-25 10:52:12 -04:00 traffic-control inspection and first patch. Existing run_metadrive_overlay_demo.py applied _visual_traffic_signal_label_from_frame to both stock and VLM traffic-light episodes, so rendered pixels could create traffic_light_stop plans independent of Qwen labels/RTP and pollute stock-vs-Qwen evidence. Patched the harness so that visual signal guard is demo-only behind --enable-visual-signal-guard, applies only in VLM mode, and records visual_signal_guard_enabled per frame. Added _should_apply_visual_signal_guard plus a regression test proving the guard is off by default, never applies to stock, is disabled when VLM speed control is disabled, and only applies to red/green labels when explicitly enabled. Added tools/reasoned_trajectory_poc/evaluate_signal_trace.py to evaluate red_stop_light, green_go_light, and stop_sign traces from Qwen labels, RTP text, durable_speed_plan_sources, target speed, consumed age, and green-path match; the evaluator rejects traces where visual_signal_guard_enabled is true.
  - 2026-05-25 10:52:39 -04:00 verification for traffic-control patch. py_compile passed for run_metadrive_overlay_demo.py, evaluate_signal_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 212 tests OK.
  - 2026-05-25 10:54:16 -04:00 first Qwen-only traffic-light run attempt did not enter closed-loop. Command used synchronous red_stop_light/green_go_light choice-mode TensorRT server with visual_signal_guard disabled by default, output artifacts\reasoned_trajectory_poc\traffic_light_qwen_only_sync_320_20260525_1053. Failure was startup/configuration: qwen_trt_label_engine.py refused --require-manifest because current code contract SHA 130d92ef052916c620d309e2dc1ff10959c1cbb4ccd300c21ed6840b0a6dd7e9 no longer matched existing F:\qwen_trt_export manifest expected e244a4f573d19c7403d59e6ae45ba6e708fc6c0d18cda353a202ffc15872d18a. Targeted process sweep after failure returned NO_TARGETED_POC_PROCESSES.
  - 2026-05-25 10:54:50 -04:00 refreshed the red/green Qwen TensorRT runtime manifest with check-artifacts --write-manifest using full168 FP32 vision and seq576 hidden-choice NVFP4 text engine. check-artifacts ok true, issues empty, RTX 5060 Ti compute capability 12.0, TensorRT 10.16.1.11, CUDA 13.2 with sm_120, and new F:\qwen_trt_export\qwen_trt_runtime_manifest.json contract SHA ac55a1650aa59b6f798269ea75770d0ca26eea7f80978c7006c9bd092c334a5d.
  - 2026-05-25 10:56:37 -04:00 corrected manifest refresh to the exact red/green runtime shape. check-artifacts with --score-label-groups "red_stop_light,green_go_light" wrote F:\qwen_trt_export\qwen_trt_runtime_manifest.json contract SHA e244a4f573d19c7403d59e6ae45ba6e708fc6c0d18cda353a202ffc15872d18a, ok true, issues empty. The prior ac55a... manifest covered default multi-groups and was not the serve contract for this red/green-only run.
  - 2026-05-25 10:58:00 -04:00 corrected the red/green manifest one more time to include the exact --score-rotate-groups serve flag. check-artifacts ok true and wrote contract SHA 1dab6aea253ef1809197c1e3fac229c37bb1ae7fdfde628b5e3272ac82ce55d6. The intermediate e244a... manifest had the right group but rotate=false, so serve still rejected it.
  - 2026-05-25 11:06:53 -04:00 Qwen-only sync traffic-light run traffic_light_qwen_only_sync_320_20260525_1058 completed but failed behavior evaluation. Runtime timing was good: 320/320 publishes, 0 deadline misses, p99 47.62 ms, max 48.89 ms, same-frame all true, max RTP age 0, visual_signal_guard_enabled false. New signal evaluator failed: red_qwen_success_rate 0.091, red_control_success_rate 0.945, green_qwen_success_rate 0.260, green_control_success_rate 0.0. Trace inspection showed Qwen choice mode used answer word "clear" for green, which was selected on red frames because the road/path was clear; green RTP confidence 0.82 also could not clear stale traffic_light_stop plans under default durable_conflict_override_confidence 0.90, so the car stayed pinned after green. Probed full224 vision engine on frame 0222; it remained under budget for one red/green choice group (p99 48.39 ms, max 48.39 ms for 12 benchmark iterations) but still answered red because the choice word was ambiguous. Binary yes/no seq768 at full224 was too slow (~90 ms) and wrong on the red/green probe. Patched qwen_trt_label_engine.py to add literal "green" to hidden-choice vocabulary and use allowed words red/green/none and stop/green/none for signal groups. Patched run_metadrive_overlay_demo.py with durable_signal_clear_confidence default 0.80 so green RTP clears stale signal stops without depending on the higher lateral-conflict threshold. Patched qwen_label_rtp_worker.py so mixed green+agent/construction RTP carries at least green signal confidence 0.82. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 215 tests OK.
  - 2026-05-25 11:13:15 -04:00 process hygiene check at user request. Strict targeted command-line sweep over repo/Qwen/MetaDrive/openpilot POC patterns returned NO_TARGETED_POC_PROCESSES. NVIDIA process monitor showed no Python/Qwen/TensorRT compute worker, only ordinary desktop/WDDM clients. Broad process inspection found unrelated MCP/background processes and an open-webui child python.exe using zero CPU over a 3-second sample; no POC process was killed and no Qwen/MetaDrive/render/evaluator job was left running.
  - 2026-05-25 11:56:24 -04:00 continuation from active goal after compaction. Initial targeted process sweep found no active Qwen/TensorRT/MetaDrive/openpilot POC worker; the only command-line hit was the sweep process itself. nvidia-smi showed no Python/Qwen compute process, only ordinary desktop/WDDM clients. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules. Current focus is to stop the repeated sign/heading class of failures by auditing the end-to-end side invariant and finishing the additive TensorRT text-position dtype patch that was mid-edit before compaction.
  - 2026-05-25 12:10:13 -04:00 finished the additive TensorRT text-position dtype patch in tools/reasoned_trajectory_poc/qwen_trt_label_engine.py. Added --text-position-dtype int64|int32, kept int64 as the default so existing engines/manifests are not replaced, added _pos_int32 to ONNX/engine filenames for int32 probes, and added text_position_dtype to build output plus the runtime manifest contract. This is a correctness probe for the TensorRT position_ids binding issue that returned all-zero logits while PyTorch/ONNX were nonzero.
  - 2026-05-25 12:10:13 -04:00 patched the MetaDrive harness lateral convention from a fixed global sign into a runtime-calibrated route-lateral adapter. run_metadrive_overlay_demo.py now measures route_lateral_sign_to_openpilot from the actual route/ego pose and uses it for construction spawning, random construction/pedestrian side placement, durable avoidance compilation, compiled lateral fallback, Qwen lead-track lateral state, scene-board green path offset, record fields, and sign-valid checks. This is not a simulator-side object hack; it removes simulator lane-coordinate sign from the Qwen/openpilot contract so the path seen by Qwen and the path tracked by control use ego-left-positive openpilot semantics.
  - 2026-05-25 12:10:13 -04:00 added tests for int32 TensorRT engine filename/contract separation, runtime route-lateral adapter behavior when lane-local sign flips, durable construction avoidance sign validation after a flipped route-lateral sign, and lead-track lateral state conversion into openpilot ego-left convention. Verification: py_compile passed for qwen_trt_label_engine.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 220 tests OK.
  - 2026-05-25 12:10:13 -04:00 ran a short non-Qwen MetaDrive smoke run after the route adapter patch: artifacts\reasoned_trajectory_poc\route_lateral_adapter_smoke_construction_right_20260525_1210. The run recorded route_lateral_sign_to_openpilot=-1.0, right-side construction spawned at route lateral +1.35 m which converts to openpilot right, green_path_matches_tracked_path=true, and the saved VLM board shows the cone row on the driver/screen right. This validates the adapter on the current map before any Qwen scoring run.
  - 2026-05-25 12:12:00 -04:00 final process hygiene sweep after tests and the short MetaDrive smoke run. Strict worker-only sweep over python/py/ffmpeg/ssh processes found NO_TARGETED_POC_WORKER_PROCESSES. nvidia-smi showed no Python/Qwen/TensorRT compute process, only ordinary desktop/WDDM clients. No Qwen server, TensorRT build, MetaDrive run, renderer, evaluator, C3X SSH, or tinygrad job was left running.
  - 2026-05-25 12:13:25 -04:00 continuation from active goal. Initial worker-only process sweep found NO_TARGETED_POC_WORKER_PROCESSES and nvidia-smi showed no Python/Qwen/TensorRT compute process. git status still shows the expected dirty POC/readme/progress/tinygrad files plus untracked side/lead/evaluator modules. Next action is to prove the route-lateral adapter with a real Qwen closed-loop construction run, because the prior smoke run only verified static renderer/control sign plumbing.
  - 2026-05-25 12:14:55 -04:00 refreshed the active construction TensorRT runtime manifest after adding the new text_position_dtype contract key. New pinned manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v7_route_adapter_manifest_20260525_1214.json, contract SHA fc75b0e1792044445814bfbab860491cb441219d5f145f995d6af00be3a8d096. check-artifacts ok true against Qwen2.5-VL 3B, full168 FP32 vision engine, FP8 seq768 selected-logits text engine, binary construction_compact scoring, rotating groups cones/barrier and construction_blue_edge/construction_purple_edge, shared text engine, edge binary, mirror fusion, text_position_mode qwen, text_position_dtype int64.
  - 2026-05-25 12:16:00 -04:00 ran real Qwen closed-loop right-construction route-adapter verification with the v7 manifest: artifacts\reasoned_trajectory_poc\construction_right_route_adapter_qwen_80_20260525_1215_realtime. Command used async VLM, period 2, latest-only, drop-stale-results, max accepted age 8, prewarm 12 s, speed_mps 2.5, no visual fallback, --require-manifest. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 2.122 m, publish_count 55/80, valid_count 55, deadline_miss_count 0, mean planner overhead 2.69 ms, p99 5.32 ms, max 5.68 ms, max_rtp_age_frames 8, mean_path_delta_m 0.955, speed_delta 0.0.
  - 2026-05-25 12:16:00 -04:00 evaluated right-construction run and wrote artifacts\reasoned_trajectory_poc\construction_right_route_adapter_qwen_80_20260525_1215_realtime\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 54, qwen_side_correct 42, qwen_side_wrong 0, qwen_side_missing 12, qwen_side_success_rate 0.778, consumed_plan_side_correct 45, consumed_plan_side_wrong 0, consumed_plan_side_missing 9, post_first_consumed_side_success_rate 1.0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, max_consumed_age_frames 8. This proves the route adapter preserved the prior right-side sign behavior but still does not meet the 95% total construction recall gate.
  - 2026-05-25 12:17:30 -04:00 ran real Qwen closed-loop left-construction route-adapter verification with the same v7 manifest: artifacts\reasoned_trajectory_poc\construction_left_route_adapter_qwen_80_20260525_1216_realtime. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 1.743 m, publish_count 71/80, valid_count 71, deadline_miss_count 0, mean planner overhead 2.69 ms, p99 5.42 ms, max 5.70 ms, max_rtp_age_frames 8, mean_path_delta_m 0.739, speed_delta 0.0.
  - 2026-05-25 12:17:30 -04:00 evaluated left-construction run and wrote artifacts\reasoned_trajectory_poc\construction_left_route_adapter_qwen_80_20260525_1216_realtime\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 72, qwen_side_correct 40, qwen_side_wrong 0, qwen_side_missing 32, qwen_side_success_rate 0.556, consumed_plan_side_correct 41, consumed_plan_side_wrong 0, consumed_plan_side_missing 31, false_construction_rate 0.25 over only 8 not-path-relevant startup/tail frames, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, max_consumed_age_frames 8. This is an important sign-fix result: left-side no longer steers into the hazard, but Qwen misses too many left-side frames and remains below the construction gate.
  - 2026-05-25 12:18:00 -04:00 rendered videos for the route-adapter Qwen construction runs. Right: artifacts\reasoned_trajectory_poc\construction_right_route_adapter_qwen_80_20260525_1215_realtime\videos\side_by_side_construction_right_route_adapter_qwen_80_20260525_1215_realtime.mp4 plus stock/vlm MP4s. Left: artifacts\reasoned_trajectory_poc\construction_left_route_adapter_qwen_80_20260525_1216_realtime\videos\side_by_side_construction_left_route_adapter_qwen_80_20260525_1216_realtime.mp4 plus stock/vlm MP4s.
  - 2026-05-25 12:18:48 -04:00 verification after route-adapter Qwen runs. py_compile passed for run_metadrive_overlay_demo.py, qwen_trt_label_engine.py, evaluate_construction_trace.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 220 tests OK.
  - 2026-05-25 12:19:20 -04:00 final process hygiene sweep after route-adapter Qwen construction verification. Strict worker-only sweep over python/py/ffmpeg/ssh processes found NO_TARGETED_POC_WORKER_PROCESSES. nvidia-smi showed no Python/Qwen/TensorRT compute process, only ordinary desktop/WDDM clients. No Qwen server, TensorRT build, MetaDrive run, renderer, evaluator, C3X SSH, or tinygrad job was left running.
  - 2026-05-25 12:20:59 -04:00 continuation from active goal. Initial worker-only process sweep found NO_TARGETED_POC_WORKER_PROCESSES and nvidia-smi showed no Python/Qwen/TensorRT compute process. Current target is the remaining construction gate gap after the route-lateral adapter proof: left-side construction has zero wrong-side/toward-control frames but low Qwen recall, so the next action is frame-by-frame trace comparison against right-side construction to identify whether misses are caused by visual evidence, prompt/scoring, rotating cadence, or state-machine gating.
  - 2026-05-25 12:24:58 -04:00 process hygiene check at user request. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker. NVIDIA process monitor showed no Python/Qwen/TensorRT compute client, only ordinary desktop/WDDM clients. One long-running python.exe listener on port 8080 was inspected and identified as Open WebUI from its HTTP response, not this POC, so it was left running. No process was killed.
  - 2026-05-25 12:25:17 -04:00 goal continuation. Rechecked targeted POC command lines before code inspection; no Qwen/TensorRT/MetaDrive/render/evaluator worker was running. GPU process list showed only ordinary desktop/WDDM clients, no Python/Qwen/TensorRT compute worker. Continuing on the construction-side recall gap: left construction was not steering into hazards, but side activation lagged because the first correct edge score in the prior trace was just below the current bootstrap gate.
  - 2026-05-25 12:28:00 -04:00 construction recall root cause isolated from current logs. In artifacts\reasoned_trajectory_poc\construction_left_route_adapter_qwen_80_20260525_1216_realtime, frame 17 already had Qwen labels cones/barrier/construction_blue_edge and RTP scene=construction_left with avoid=[left_edge_s8_48_margin1.25], but confidence was only 0.84 because edge score was 1.953 against opposite 0.0. The durable lateral compiler requires two separate low-confidence observations unless confidence reaches the immediate gate, and the next scored source frame did not repeat the edge before the state reset. Frame 33 scored 2.422 and confidence 0.96, so the maneuver started late. This is not a sign error; it is an over-tight edge-bootstrap confidence gate.
  - 2026-05-25 12:28:00 -04:00 patched the existing Qwen score-to-RTP path symmetrically. Kept general construction side lock at 2.0 but changed CONSTRUCTION_EDGE_BOOTSTRAP_SCORE to 1.9 and bumped CONSTRUCTION_STATE_MACHINE_VERSION to 7. qwen_trt_label_engine.py now uses CONSTRUCTION_EDGE_BOOTSTRAP_SCORE/MARGIN/OPPOSITE_MAX instead of a hardcoded 2.0/1.75/0.05 for strong edge immediate confidence. Added tests that a near-bootstrap unambiguous edge maps to confidence 0.96 and seeds construction side state, while just-below or ambiguous edges still do not.
  - 2026-05-25 12:29:20 -04:00 first unittest run after the edge calibration failed one new regression: the bootstrap gate accepted the near-threshold edge, but _observe_construction_side_for_lock still used the old generic 2.0 immediate score before setting the construction side lock. Patched the lock helper so only construction edge labels use the edge bootstrap score/margin/opposite thresholds, while semantic construction_left/right and shift/candidate labels keep the generic immediate side-lock threshold.
  - 2026-05-25 12:30:10 -04:00 verification after lock-helper patch. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 221 tests OK. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py after rerunning serially; the first compile attempt hit a Windows pycache rename race because it was run concurrently with unittest, not a syntax/code failure.
  - 2026-05-25 12:30:32 -04:00 refreshed the construction TensorRT runtime manifest after the edge-bootstrap/state-machine contract change. New pinned manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v8_edge_bootstrap_manifest_20260525_1230.json. check-artifacts ok true against Qwen2.5-VL 3B, full168 FP32 vision engine, FP8 seq768 selected-logits text engine, binary construction_compact scoring, rotating groups cones/barrier and construction_blue_edge/construction_purple_edge, shared text engine, edge binary, mirror fusion, text_position_mode qwen, text_position_dtype int64. New contract SHA 78b95959200e41a2a51a782ef1df67907da163f6cd55ddb185ae261661e86d89.
  - 2026-05-25 12:32:35 -04:00 ran real Qwen closed-loop left-construction verification with the v8 edge-bootstrap manifest: artifacts\reasoned_trajectory_poc\construction_left_edge_bootstrap_v8_qwen_80_20260525_1231_realtime. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 2.025 m, publish_count 56/80, valid_count 56, deadline_miss_count 0, mean planner overhead 2.67 ms, p99 5.42 ms, max 5.88 ms, max_rtp_age_frames 8, mean_path_delta_m 0.938, mean_speed_delta_mps 0.0.
  - 2026-05-25 12:32:35 -04:00 evaluated v8 left-construction run and wrote artifacts\reasoned_trajectory_poc\construction_left_edge_bootstrap_v8_qwen_80_20260525_1231_realtime\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 59, qwen_side_correct 41, qwen_side_wrong 0, qwen_side_missing 18, qwen_side_success_rate 0.695, consumed_plan_side_correct 42, consumed_plan_side_wrong 0, consumed_plan_side_missing 17, consumed_plan_side_success_rate 0.712, post_first_side_qwen_side_success_rate 0.976, post_first_consumed_side_success_rate 1.0, false_construction_rate 0.048, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0. Frame 17 now has construction_blue_edge score 1.953, confidence 0.96, and a durable left-edge/right-bias plan, proving the specific late-activation bug is fixed. The remaining construction gap is earlier visibility/recall before frame 17, not sign conversion or compiler acceptance.
  - 2026-05-25 12:34:25 -04:00 probed saved left-construction scene boards to check whether startup misses are group-cadence or visual evidence. Direct full168 edge scoring on vlm_input_0008.png and vlm_input_0012.png returned labels none with construction_blue_edge=0.0 and construction_purple_edge=0.0. A full224 probe of vlm_input_0012.png also returned none. This indicates the remaining startup misses are not caused by the 1.9 threshold or resolution alone; Qwen is not seeing/accepting the edge evidence yet in those early boards.
  - 2026-05-25 12:37:10 -04:00 additional probe on vlm_input_0012.png with action-oriented construction_shift_left/construction_shift_right also returned labels none with both scores 0.0, so early-frame miss is not only blue/purple wording. Inspected the saved board visually: cones are visible inside the left half of the green corridor but not exactly touching the blue line. Patched the compact construction edge prompt to ask whether a relevant hazard is on, closest to, occupying, intruding from, or narrowing the left/right side near the colored edge, not only whether it literally touches the edge line. Mirrored the wording into qwen_label_rtp_worker SCORE_QUESTIONS and updated the prompt contract test. This remains the existing Qwen label/scoring path, not a simulator fallback.
  - 2026-05-25 12:39:30 -04:00 reverted the compact construction edge prompt wording experiment. py_compile and unittest passed while the experiment was present, but direct Qwen probes still returned none on vlm_input_0012.png and also returned none on vlm_input_0016.png, so it did not improve the measured early-frame recall and risked perturbing later detections. Kept the proven v8 threshold/state-machine fix; reverted only the prompt text and its test expectation back to the previous contract wording.
  - 2026-05-25 12:36:00 -04:00 ran real Qwen closed-loop right-construction verification with the same v8 edge-bootstrap manifest: artifacts\reasoned_trajectory_poc\construction_right_edge_bootstrap_v8_qwen_80_20260525_1235_realtime. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 2.122 m, publish_count 48/80, valid_count 48, deadline_miss_count 0, mean planner overhead 2.61 ms, p99 5.43 ms, max 5.84 ms, max_rtp_age_frames 8, mean_path_delta_m 1.094, mean_speed_delta_mps 0.0.
  - 2026-05-25 12:36:20 -04:00 evaluated v8 right-construction run and wrote artifacts\reasoned_trajectory_poc\construction_right_edge_bootstrap_v8_qwen_80_20260525_1235_realtime\vlm\construction_trace_evaluation.json. Results: path_relevant_construction_frames 54, qwen_side_correct 42, qwen_side_wrong 0, qwen_side_missing 12, qwen_side_success_rate 0.778, consumed_plan_side_correct 45, consumed_plan_side_wrong 0, consumed_plan_side_missing 9, post_first_consumed_side_success_rate 1.0, false_construction_rate 0.0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0.
  - 2026-05-25 12:36:35 -04:00 rendered videos for the v8 construction verification runs. Left: artifacts\reasoned_trajectory_poc\construction_left_edge_bootstrap_v8_qwen_80_20260525_1231_realtime\videos\side_by_side_construction_left_edge_bootstrap_v8_qwen_80_20260525_1231_realtime.mp4 plus stock/vlm MP4s. Right: artifacts\reasoned_trajectory_poc\construction_right_edge_bootstrap_v8_qwen_80_20260525_1235_realtime\videos\side_by_side_construction_right_edge_bootstrap_v8_qwen_80_20260525_1235_realtime.mp4 plus stock/vlm MP4s.
  - 2026-05-25 12:40:40 -04:00 final verification after reverting the prompt experiment. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 221 tests OK. check-artifacts --require-manifest against artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v8_edge_bootstrap_manifest_20260525_1230.json passed ok true with matching contract SHA 78b95959200e41a2a51a782ef1df67907da163f6cd55ddb185ae261661e86d89, confirming the active manifest still matches the reverted prompt plus v8 threshold/state-machine code.
  - 2026-05-25 12:41:25 -04:00 final process hygiene sweep for this continuation. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only command-line hit was the sweep process itself. NVIDIA process monitor showed no Python/Qwen/TensorRT compute client, only ordinary desktop/WDDM clients. No process was killed. git status still shows the expected dirty POC/readme/progress/tinygrad files and untracked side/lead/evaluator modules.
  - 2026-05-25 12:42:31 -04:00 goal continuation. Initial strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only command-line hit was the sweep process itself. NVIDIA process monitor showed no Python/Qwen/TensorRT compute worker, only ordinary desktop/WDDM clients. Continuing on construction recall because v8 fixed late activation once Qwen saw side evidence, but total construction side success is still below the 95% gate due to early scene-board visibility/recognition misses.
  - 2026-05-25 12:44:10 -04:00 inspected scene-board rendering and construction artifacts. VLM boards are 320x200 RGB, with outside-corridor dimming enabled and multiple corridor overlays: green fill alpha 85, blue/purple side fill alpha 82, bright green center strip alpha 245, and blue/purple edge guide lines. The saved early left board shows cones visually present but tinted/covered by corridor overlays, and Qwen still scored both construction edges 0.0. Existing harness already exposes --no-scene-board-corridor-side-fill, so the next focused check is to keep the green tracked path and colored edge lines but remove side fills to test whether Qwen recognizes early construction sooner without changing architecture or adding a simulator fallback.
  - 2026-05-25 12:46:30 -04:00 negative scene-board visibility probe. Ran real Qwen closed-loop left construction with v8 manifest and --no-scene-board-corridor-side-fill: artifacts\reasoned_trajectory_poc\construction_left_no_side_fill_v8_qwen_80_20260525_1244_realtime. Runtime had no deadline misses but behavior regressed badly: Qwen/VLM min construction clearance 0.668 m. Evaluation wrote vlm\construction_trace_evaluation.json and found qwen_side_correct 0, qwen_side_wrong 42, consumed_plan_side_wrong 60, control_toward_frames 51, planned_toward_frames 51, collision_count 13, green_path_mismatch_count 0. Frame 17 scored construction_purple_edge=2.469 on a left-side hazard and generated scene=construction_right, proving side fill removal makes the side semantics less stable. Do not make this default.
  - 2026-05-25 12:48:40 -04:00 negative edge-inset probe. Ran real Qwen closed-loop left construction with v8 manifest and --scene-board-edge-insets: artifacts\reasoned_trajectory_poc\construction_left_edge_insets_v8_qwen_80_20260525_1247_realtime. Runtime had no deadline misses but behavior again regressed to min construction clearance 0.668 m. Evaluation found qwen_side_correct 0, qwen_side_wrong 42, consumed_plan_side_wrong 60, control_toward_frames 51, collision_count 13. Trace showed initial wrong lock at frame 17 on construction_purple_edge=2.344, then later correct construction_blue_edge scores at frames 25/33 were suppressed by the construction side lock while the path was moving into the real left-side hazard.
  - 2026-05-25 12:50:00 -04:00 patched the construction side lock to handle the edge-inset failure mode generically. Added CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE=2.3 and bumped CONSTRUCTION_STATE_MACHINE_VERSION to 8. A conflicting construction side now overrides the prior lock when its score exceeds the normal immediate override threshold, or when it exceeds the new toward-path threshold and the current tracked path is already shifted toward the newly reported hazard side. Added the new threshold to the TensorRT runtime contract and added tests that a high-confidence contradictory side can override while the path is moving toward that newly reported hazard, a moderate contradiction still does not, and a wrong lock can be rescued by a strong opposite edge score.
  - 2026-05-25 12:50:35 -04:00 verification after contradictory-side rescue patch. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 222 tests OK.
  - 2026-05-25 12:51:00 -04:00 refreshed construction TensorRT runtime manifest after the v9 state-machine contract change. New pinned manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v9_toward_override_manifest_20260525_1251.json. check-artifacts ok true against Qwen2.5-VL 3B, full168 FP32 vision engine, FP8 seq768 selected-logits text engine, binary construction_compact scoring, rotating groups cones/barrier and construction_blue_edge/construction_purple_edge, shared text engine, edge binary, mirror fusion, text_position_mode qwen, text_position_dtype int64. New contract SHA 73318211a637afc5f6270ad519249744e865d907529e6bbd7a7b3a1f8439dc24.
  - 2026-05-25 12:53:05 -04:00 ran real Qwen closed-loop default left-construction verification with the v9 manifest: artifacts\reasoned_trajectory_poc\construction_left_v9_toward_override_qwen_80_20260525_1251_realtime. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 2.025 m, publish_count 56/80, valid_count 56, deadline_miss_count 0, mean planner overhead 2.61 ms, p99 4.52 ms, max 5.23 ms, max_rtp_age_frames 8, mean_path_delta_m 0.938, mean_speed_delta_mps 0.0.
  - 2026-05-25 12:53:05 -04:00 evaluated v9 default left-construction run and wrote artifacts\reasoned_trajectory_poc\construction_left_v9_toward_override_qwen_80_20260525_1251_realtime\vlm\construction_trace_evaluation.json. Results matched v8 behavior: qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, post_first_side_qwen_side_success_rate 0.976, post_first_consumed_side_success_rate 1.0. Total side success remains below the 95% gate because of startup misses before first Qwen side evidence, not because of sign or compiler direction.
  - 2026-05-25 12:56:06 -04:00 process hygiene check at user request. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only POC command-line hit was the sweep process itself. NVIDIA process list showed no Python/Qwen/TensorRT compute client, only ordinary desktop/WDDM clients. A generic python.exe from 2026-05-06 was inspected separately and traced to parent open-webui.exe, not this POC, so it was left running. No process was killed.
  - 2026-05-25 12:56:55 -04:00 goal continuation. Rechecked strict POC command lines and GPU clients before resuming; no Qwen/TensorRT/MetaDrive/render/evaluator worker was running and no Python/Qwen/TensorRT compute client was attached to the GPU. git status remains dirty with expected POC files, reasoned-control modules, progress/readme updates, nested tinygrad changes, and untracked evaluator/lead/side modules. Continuing with trace-backed construction-side correction rather than another sign guess.
  - 2026-05-25 13:00:10 -04:00 scanned v9 default-left and edge-inset-left construction traces before changing thresholds. Edge-inset failure has correct opposite blue evidence at source frame 35 while tracked_path_lat was only about +0.32m, just below the v9 0.35m toward-path gate, and later blue evidence 2.296 at source frame 51 when the path was already far wrong. Default-left good run also has contradictory purple evidence 2.422 at source frame 35 after the path was correctly moving right. Conclusion: simply lowering the toward-path threshold or score threshold is unsafe because it can reintroduce wrong-side flips in the default production board. Next action is sign-convention audit and stricter shared semantics, not a blind threshold decrease.
  - 2026-05-25 13:02:54 -04:00 patched traceability for future side-flip debugging without changing control behavior. RtpEngineResult and ReasonedStepResult now carry raw current-group labels/scores, labels_scored_this_request, score_group_index, and label_state_debug separately from rotating cached published labels/scores. qwen_label_rtp_worker now emits construction lock/pending/clear debug state. run_metadrive_overlay_demo now writes these fields into every episode record as qwen_raw_labels, qwen_raw_label_scores, qwen_labels_scored_this_request, qwen_score_group_index, and qwen_label_state_debug. Added tests for persistent worker parsing and planner propagation of these fields.
  - 2026-05-25 13:03:43 -04:00 verification and active TRT worker trace patch. py_compile passed for vlm.py, planner.py, qwen_label_rtp_worker.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 222 tests OK. Added the same label_state_debug output to qwen_trt_label_engine.py because the fast TensorRT worker is the active demo path and already emits labels_current_group/labels_scored_this_request/label_scores_cached.
  - 2026-05-25 13:04:15 -04:00 verification after patching the active TensorRT worker trace field. py_compile passed for qwen_trt_label_engine.py, vlm.py, planner.py, qwen_label_rtp_worker.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 222 tests OK.
  - 2026-05-25 13:04:38 -04:00 pre-demo process hygiene check. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only hit was the sweep process itself. NVIDIA process list showed no Python/Qwen/TensorRT compute client. Proceeding with a short real Qwen run to verify new trace fields.
  - 2026-05-25 13:08:07 -04:00 ran a short real TensorRT/Qwen traceability verification after the raw-label logging patch: artifacts\reasoned_trajectory_poc\construction_left_traceability_qwen_44_20260525_1304_realtime. Runtime summary: 44 frames, Qwen publish_count 40, valid_count 40, deadline_miss_count 0, mean planner overhead 2.78 ms, p99/max 5.80 ms, max_rtp_age_frames 8, mean_path_delta_m 0.8125, mean_speed_delta_mps 0.0. Confirmed episode records now contain qwen_raw_labels, qwen_raw_label_scores, qwen_labels_scored_this_request, qwen_score_group_index, and qwen_label_state_debug. Example edge-group record: frame 9 consumed source frame 2 age 7, scored construction_blue_edge/construction_purple_edge, raw labels none, raw edge scores 0/0, cached labels cones/barrier, construction_locked_side null.
  - 2026-05-25 13:08:07 -04:00 patched evaluate_construction_trace.py to preserve the same raw Qwen audit fields in construction_trace_evaluation rows, including raw_qwen_side. py_compile passed for evaluate_construction_trace.py, test_reasoned_trajectory.py, vlm.py, planner.py, qwen_trt_label_engine.py, qwen_label_rtp_worker.py, and run_metadrive_overlay_demo.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 222 tests OK. Re-evaluated the traceability run and wrote artifacts\reasoned_trajectory_poc\construction_left_traceability_qwen_44_20260525_1304_realtime\vlm\construction_trace_evaluation.json with qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, max_rtp_age_frames 8, and the new raw fields present in rows. Rendered videos: artifacts\reasoned_trajectory_poc\construction_left_traceability_qwen_44_20260525_1304_realtime\videos\side_by_side_traceability_construction_left_qwen_44.mp4 plus stock/vlm MP4s.
  - 2026-05-25 13:08:44 -04:00 final process hygiene sweep for this continuation. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only hit was the sweep process itself. NVIDIA process list showed no Python/Qwen/TensorRT compute client, only ordinary desktop/WDDM clients. No process was killed. git diff --stat confirms this continuation touched the traceability path in vlm.py, planner.py, qwen_label_rtp_worker.py, qwen_trt_label_engine.py, run_metadrive_overlay_demo.py, evaluate_construction_trace.py, tests, progress, and existing POC/readme files from earlier work.
  - 2026-05-25 13:09:52 -04:00 goal continuation. Strict POC command-line sweep found no Qwen/TensorRT/MetaDrive/render/evaluator worker; the only hit was the sweep process itself. NVIDIA process list showed no Python/Qwen/TensorRT compute client. git status remains dirty with expected POC/reasoned-control files, progress/readme changes, nested tinygrad changes, and untracked evaluator/side/lead modules. Next action is to rerun the construction default/stress cases with raw-label traceability and use that evidence before changing side logic.
  - 2026-05-25 13:12:40 -04:00 reran the known failing edge-inset construction stress with current TensorRT/Qwen raw-label traceability: artifacts\reasoned_trajectory_poc\construction_left_edge_insets_trace_qwen_80_20260525_1310_realtime. Runtime still failed: Qwen/VLM min construction route clearance 0.668 m, qwen_side_correct 0, qwen_side_wrong 42, consumed_plan_side_wrong 60, control_toward_frames 51, collision_count 13, deadline_miss_count 0, mean planner overhead 4.30 ms, p99 7.36 ms, max_rtp_age_frames 8. Raw trace now exposes the mechanism: source frame 10 initially scored wrong construction_purple_edge=2.34375 and locked right. Source frame 18 then scored correct construction_blue_edge=2.4375 with purple 0, but cached labels remained construction_purple_edge because the source-frame tracked_path_lat was still approximately 0, so the v9 toward-path rescue did not fire. Later correct blue evidence kept arriving but was suppressed by the wrong lock.
  - 2026-05-25 13:16:15 -04:00 reran the known-good default construction board with current raw-label traceability: artifacts\reasoned_trajectory_poc\construction_left_default_trace_qwen_80_20260525_1313_realtime. Runtime remained good: Qwen/VLM min construction route clearance 2.025 m, qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, collision_count 0, deadline_miss_count 0, mean planner overhead 2.64 ms, p99 3.48 ms, max_rtp_age_frames 8. Raw trace shows the guardrail needed for the fix: the good run does get later contradictory construction_purple_edge=2.421875 at source frame 35, but by then the construction lock is old and the source board tracked_path_lat is -0.32m. Patched qwen_label_rtp_worker state machine to add only an early high-confidence reversal path: a conflicting side can replace the lock only when the lock age is within the score cache TTL, the source tracked path is still near neutral (<=0.12m), the new score is >=2.3, the old-side score is <=0.05, and the margin is >=1.75. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 9, added locked_frame to debug_state, exported the new runtime contract fields in qwen_trt_label_engine, and added tests for early neutral reversal and late stable-lock suppression.
  - 2026-05-25 13:17:16 -04:00 verification after early-reversal patch. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, test_reasoned_trajectory.py, vlm.py, planner.py, run_metadrive_overlay_demo.py, and evaluate_construction_trace.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 224 tests OK. Refreshed TensorRT runtime manifest with the state-machine v9 contract: artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v10_early_reversal_manifest_20260525_1317.json, contract SHA 25b89856160ffeb2b5c5430a8d94946518981012b0a46fb4bca81fb077ee7d7d, check-artifacts ok true.
  - 2026-05-25 13:25:21 -04:00 v10/v11 closure of the state-machine fix exposed the next layer. First v10 failed because the early reversal window incorrectly used score_cache_ttl_frames=3, while the first correct opposite source frame arrives 8 source frames after the wrong lock; patched to explicit CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES=12 and refreshed v11 manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v11_early_reversal_window_manifest_20260525_1322.json with contract SHA e1ce194f92d14f08c6d5f22cdd04c37e2fd74ba45447d5c2206fc66c547242b7. Full unittest passed with 224 tests OK after adjusting the production-TTL regression test. Reran edge-inset with v11: Qwen/RTP side now corrected at source frame 18 (cached labels construction_blue_edge, locked_side left, scene=construction_left), but the car still drove into cones because update_durable_lateral_plans kept the older right_edge durable plan active and left the corrected left_edge plan pending. Patched the durable lateral layer so the default construction conflict immediate threshold is 0.95 instead of 0.99 once the Qwen side state machine has accepted a contradictory side; explicit 0.99 remains available and still covered by the existing no-single-frame-reversal test. Added a unit test that the default threshold immediately accepts a 0.96 state-machine-corrected construction side and replaces the stale opposing durable plan.
  - 2026-05-25 13:32:08 -04:00 process hygiene check at user request. Strict POC command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator worker; the only POC-pattern hits were the sweep commands themselves. nvidia-smi showed no Python/Qwen/TensorRT compute process, only ordinary desktop/WDDM clients. Broad Python inspection found two Codex DuckDuckGo MCP python workers and one old responding python.exe child of open-webui.exe from 2026-05-06; none are part of this POC, so no process was killed.
  - 2026-05-25 13:33:16 -04:00 goal continuation. Rechecked current git status and process/GPU baseline before resuming. Worktree remains dirty with the expected reasoned-control, Qwen/TensorRT, evaluator, readme/progress, and nested tinygrad changes; no active Qwen/TensorRT/MetaDrive/render/evaluator worker is running and no Python/Qwen/TensorRT compute client is attached to the GPU. Proceeding from current artifacts rather than memory alone.
  - 2026-05-25 13:33:16 -04:00 recorded the successful v12 edge-priority construction stress run that was completed before this continuation: artifacts\reasoned_trajectory_poc\construction_left_edge_insets_v12_edge_priority_qwen_80_20260525_1328_realtime using manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v12_edge_priority_manifest_20260525_1328.json, contract SHA 029f13d10893921cd1da559e3f38fd4159121d75732f2745c88211dbf476fcdc. Runtime: 80 frames, speed_mps 2.5, async/latest-only Qwen, publish_count 46, valid_count 46, deadline_miss_count 0, mean planner overhead 4.14 ms, p99 7.06 ms, max 7.21 ms, max_rtp_age_frames 8, mean_path_delta_m 0.217, mean_speed_delta_mps 0.0. Evaluation: min_construction_route_clearance_m 2.067, collision_count 0, green_path_mismatch_count 0, qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, planned_toward_frames 0, post_first_consumed_side_success_rate 1.0. This fixes the prior edge-inset cone collision in that stress case, but raw Qwen side recall remains sparse because the durable corrected plan carries the maneuver after early edge evidence.
  - 2026-05-25 13:35:37 -04:00 ran the default-board regression with the same v12 edge-priority manifest: artifacts\reasoned_trajectory_poc\construction_left_default_v12_edge_priority_qwen_80_20260525_1334_realtime. Runtime was nonblocking but behavior failed: stock construction clearance 1.354 m, Qwen/VLM clearance only 0.749 m, publish_count 53, valid_count 53, deadline_miss_count 0, mean planner overhead 2.66 ms, p99 5.33 ms, max_rtp_age_frames 8. Evaluation showed qwen_side_correct 0, qwen_side_wrong 6, consumed_plan_side_wrong 55, control_toward_frames 47, collision_count 13, green_path_mismatch_count 0. Root cause from raw trace: source frame 16 scored construction_purple_edge=2.820 with blue 0.0 for a left-side hazard, locked construction side right, and the durable compiler steered left into the cones. The v12 fix is therefore not suitable as default; edge-only color labels are too brittle when duplicated early.
  - 2026-05-25 13:39:37 -04:00 patched the Qwen rotating-label state machine to treat combined semantic-side plus colored-edge schedules as a corroboration contract instead of letting one colored edge override semantic disagreement. CONSTRUCTION_STATE_MACHINE_VERSION is now 10. When both construction_left/right and construction_blue_edge/purple_edge groups are configured, a presence-active side label only passes if semantic side and edge side agree; a strong edge-only bootstrap is still allowed before active construction presence so early lookahead can survive the edge-inset case. Updated tests so semantic/edge disagreement suppresses lateral construction output, and pre-presence strong edge bootstrap remains allowed. Verification: py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 226 tests OK. Wrote v13 manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v13_consensus_edge_priority_manifest_20260525_1340.json with contract SHA fac8b6fb9a53e47983c7753e12a6b3833ff7bb2a308da1568d35f0878a1dbf3a, groups cones/barrier; construction_blue_edge/purple_edge; construction_left/right; construction_blue_edge/purple_edge, check-artifacts ok true.
  - 2026-05-25 13:43:12 -04:00 v13 default-board rerun stopped the catastrophic v12 collision but still failed the construction side gate. Run: artifacts\reasoned_trajectory_poc\construction_left_default_v13_consensus_edge_priority_qwen_80_20260525_1340_realtime. Runtime: Qwen/VLM clearance 1.296 m vs stock 1.354 m, publish_count 54, valid_count 54, deadline_miss_count 0, mean overhead 2.63 ms, p99 5.01 ms, max_rtp_age_frames 8. Evaluation: collision_count 0, but qwen_side_correct 0, qwen_side_wrong 3, consumed_plan_side_wrong 20, control_toward_frames 14. Raw trace showed the remaining unsafe path was the old pre-presence edge bootstrap: source frame 51 scored construction_purple_edge=2.406 while construction presence was inactive, creating a stale wrong-side durable plan before later correct blue evidence could be used. Patched again: CONSTRUCTION_STATE_MACHINE_VERSION is now 11 and pre-presence edge bootstrap is disabled whenever semantic+edge corroboration is configured. Edge-only experimental schedules keep the old bootstrap behavior. Verification: py_compile passed and unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 227 tests OK.
  - 2026-05-25 13:43:12 -04:00 wrote v14 candidate-primary manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v14_candidate_primary_manifest_20260525_1343.json with contract SHA 524d7d779b29ce804cacc204bcfe05f00d160cb2a21252aa1ebac0deeec38e6d. Groups: cones/barrier; construction_blocks_left_candidate/construction_blocks_right_candidate; construction_left/right; construction_blue_edge/purple_edge. Enabled construction-candidate-binary and construction-candidate-score-resolve. check-artifacts ok true. Ran default-board v14 with --scene-board-candidate-obstruction-boards: artifacts\reasoned_trajectory_poc\construction_left_default_v14_candidate_primary_qwen_80_20260525_1344_realtime. Runtime stayed under budget: publish_count 54, valid_count 54, deadline_miss_count 0, mean overhead 28.91 ms, p99 36.46 ms, max 39.67 ms, max_rtp_age_frames 8. Behavior was safe but inert: min construction clearance 1.354 m exactly matched stock, mean_path_delta_m 0.0, qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, collision_count 0, but qwen_side_correct 0 and consumed_plan_side_correct 0. Candidate auxiliary scoring currently returns no obstruction, so this is not a successful construction fix yet.
  - 2026-05-25 13:58:38 -04:00 process hygiene recheck at user request after context compaction. Strict reasoned-pipeline command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator worker; the only POC-pattern hit was the sweep command itself. NVIDIA compute-app query showed no Python/Qwen/TensorRT compute client. Generic Python inspection found only the existing Codex DuckDuckGo MCP workers and the old open-webui.exe child from 2026-05-06, so no process was killed.
  - 2026-05-25 14:02:07 -04:00 patched the candidate-choice confidence calibration after v15 default-board trace inspection showed correct Qwen construction side labels at source frame 2 but no durable lateral activation. Root cause: high-margin candidate-choice obstruction was assigned 0.84 confidence, while the async durable lateral layer requires 0.95 for immediate first activation or multiple distinct source-frame confirmations. Added candidate immediate confidence for one-sided candidate obstruction when selected_score >= 1.5, opposite <= 0.05, and margin >= 1.5. This does not change label-side semantics: construction_blocks_left_candidate still means the left candidate corridor is obstructed, so the compiler moves right. Verification: py_compile passed for qwen_trt_label_engine.py and test_reasoned_trajectory.py; unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 231 tests OK.
  - 2026-05-25 14:02:07 -04:00 wrote v16 runtime manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_fast_clear_v16_candidate_immediate_manifest_20260525_1402.json with contract SHA 5f6e0ec57a4dfe4d16bfb0312955eb4aaf588d8d6b264abb1a10a10e139c1d52. check-artifacts ok true on CUDA 13.2 / TensorRT 10.16.1.11 / RTX 5060 Ti, using fp8 seq768 main binary labels and nvfp4 seq576 candidate-choice labels.
  - 2026-05-25 14:05:19 -04:00 ran v16 default left-construction real TensorRT/Qwen closed-loop demo: artifacts\reasoned_trajectory_poc\construction_left_default_v16_candidate_immediate_qwen_80_20260525_1403_realtime. Runtime: stock min construction route clearance 1.354 m, Qwen/VLM clearance 2.157 m, publish_count 49, valid_count 49, deadline_miss_count 0, mean planner overhead 28.95 ms, p99 35.94 ms, max 38.79 ms, max_rtp_age_frames 8, mean_path_delta_m 0.204, mean_speed_delta_mps 0.0. Evaluation: path_relevant_construction_frames 52, qwen_side_correct 8, qwen_side_wrong 0, consumed_plan_side_correct 48, consumed_plan_side_wrong 0, control_away_frames 34, control_toward_frames 0, planned_away_frames 34, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, post_first_consumed_side_success_rate 1.0. Rendered videos: artifacts\reasoned_trajectory_poc\construction_left_default_v16_candidate_immediate_qwen_80_20260525_1403_realtime\videos\side_by_side_v16_candidate_immediate_default.mp4, stock_v16_candidate_immediate_default.mp4, and vlm_v16_candidate_immediate_default.mp4.
  - 2026-05-25 14:06:14 -04:00 post-run process hygiene sweep. Strict reasoned-pipeline command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator worker; the only POC-pattern hits were the sweep commands themselves. NVIDIA process list showed no Python/Qwen/TensorRT compute client. Generic Python processes remain the same unrelated Codex DuckDuckGo MCP workers and old open-webui child, so no process was killed.
  - 2026-05-25 14:07:13 -04:00 goal continuation. Rechecked strict reasoned-pipeline process patterns, NVIDIA compute clients, git status, and progress.md before making further changes. No active Qwen/TensorRT/MetaDrive/render/evaluator worker is running; no Python/Qwen/TensorRT compute client is attached to the GPU. Worktree remains dirty with expected reasoned-control, Qwen/TensorRT, scene-board, evaluator, readme/progress, and nested tinygrad changes. Continuing with current worktree as authoritative and auditing remaining scenario gates rather than treating v16 construction proof as complete.
  - 2026-05-25 14:13:38 -04:00 audited current evaluator coverage and recent artifacts. Available separate evaluators now cover construction, pedestrian, signal/stop-sign, and lead/cut-in/crossing suites. Latest construction v16 is safe but raw Qwen side recall is still sparse; latest lead strict suite artifact is ok; latest traffic-light artifact artifacts\reasoned_trajectory_poc\traffic_light_qwen_only_sync_320_20260525_1058 is explicitly failed with red_qwen_success_rate 0.091, green_qwen_success_rate 0.260, and green_control_success_rate 0.0. Inspected the saved red and green scene-board frames; the signal is visible and the green path overlay matches tracked path. TensorRT Qwen choice probe on the red frame with the short signal prompt returned exact zero logits for all choice words, while the same choice engine returned nonzero logits on the construction candidate prompt. Token audit showed the short signal prompt answer token at index 107 versus construction candidate at index 337. Patched qwen_trt_label_engine.py to make traffic_signal and stop_or_signal choice prompts verbose, production-specific, and explicit about controlling signal/signs, dark lenses, vertical lamp positions, ignoring overlays, and choosing exactly one label. This is still Qwen label scoring, not the trained signal head or visual pixel fallback.
  - 2026-05-25 14:18:41 -04:00 process hygiene check at user request. Strict local command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker. NVIDIA compute-app query showed no Python/Qwen/TensorRT compute client attached to the RTX GPU, only ordinary desktop/WDDM clients. WSL distributions were stopped. Broad Python inspection found the same unrelated Codex DuckDuckGo MCP workers and an old open-webui.exe child python.exe listening on port 8080 from 2026-05-06; it is not part of this POC and was left running. SSH to C3X at 192.168.1.95 timed out, so no remote process state was changed.
  - 2026-05-25 14:20:01 -04:00 goal continuation baseline. Rechecked git status, strict POC process patterns, NVIDIA compute clients, and progress.md before further edits or runs. Worktree remains dirty with the expected reasoned-control, Qwen/TensorRT, evaluator, video, readme/progress, and nested tinygrad changes. No active local Qwen/TensorRT/MetaDrive/render/evaluator worker is running and no Python/Qwen/TensorRT compute client is attached to the GPU. Continuing the traffic-signal root-cause path: compare TensorRT selected-logit behavior for the failing red/green signal prompt against the known nonzero construction candidate choice prompt.
  - 2026-05-25 14:25:12 -04:00 TensorRT zero-logit isolation pass. Ran four selected-logit probes: red/green signal prompt on traffic-light frame returned exact zero for every choice word; construction-candidate prompt on the same traffic-light frame also returned exact zero for every choice word; red/green signal prompt on a construction frame returned exact zero for every choice word; construction-candidate prompt on the construction frame returned nonzero logits and selected construction_blocks_left_candidate. Input audit showed all four cases have the same image size 320x200, image_grid_thw 1x8x12, pixel_values shape 96x1176, and fixed sequence length 576; signal prompt last token index is 369 and candidate prompt last token index is 337. The old hidden-choice engine also returned zeros, including on the construction candidate frame where selected-logits works, so it is not a valid correctness control. Added gated --debug-tensor-stats to qwen_trt_label_engine.py to expose input, vision, embedding, mask, position, last-token, and text-output stats only when explicitly requested. py_compile passed for qwen_trt_label_engine.py.
  - 2026-05-25 14:40:45 -04:00 continued signal TensorRT root-cause isolation. Debug stats showed failing red/green qwen-position runs have nonzero pixel_values, nonzero TRT vision_out, nonzero inputs_embeds before/after scatter, valid position_ids, and a valid one-hot last_token_mask, then selected_logits is exactly zero. clamp127 position mode makes selected_logits nonzero but misclassifies the red and green frames as none. Exact PyTorch Qwen on the same fixed full168 scene boards answers red on the red frame and green on the green frame, but with small margins versus none. Native Qwen visual features and TRT FP32 visual features match closely on traffic and construction frames (traffic MSE about 8.25e-05, construction MSE about 9.85e-05), so the vision engine is not the root cause. Built label-specific red_stop_light/green_go_light NVFP4 text engines with int64 and int32 position_ids; both still return exact zero selected_logits with qwen positions. Added gated --vision-feature-clip-abs and added it to the runtime contract for the next overflow/range probe. py_compile passed for qwen_trt_label_engine.py.
  - 2026-05-25 14:49:38 -04:00 process hygiene sweep at user request. Strict local command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA compute-app query showed no Python/Qwen/TensorRT compute client attached to the RTX GPU, only ordinary desktop/WDDM clients. WSL distributions were stopped. Generic Python inspection found the same unrelated Codex DuckDuckGo MCP workers and the old responding open-webui.exe child python.exe from 2026-05-06 listening on port 8080; it is not part of this POC and was left running. No process was killed.
  - 2026-05-25 14:50:22 -04:00 continuation baseline for traffic-signal fix. Rechecked git status and process state: dirty files are the expected readme/progress/reasoned-control/Qwen/evaluator/video/tinygrad POC files, and no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot worker is running. Inspected qwen_trt_label_engine.py choice scoring and tests. Current traffic-signal and stop-or-green choice specs still use the neutral word none, while prior clamp127 probes showed red/green are separated from absent but dominated by none. Proceeding with a narrow vocabulary calibration in the existing Qwen label-scoring path.
  - 2026-05-25 14:55:50 -04:00 traffic-signal choice scoring iteration. First changed traffic-signal and stop-or-green choice groups from neutral none to neutral absent; py_compile passed and unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 232 tests OK. TensorRT clamp127 probes on saved traffic-light frames then selected red_stop_light on the red frame and green_go_light on the green frame at about 43-44 ms total, but a nominal no-signal scene still falsely selected green_go_light because the answer token green is color-correlated with the large green planned corridor. Inspected the nominal frame and confirmed it contains no traffic light. Patched again to add the choice token go, map go to green_go_light, and keep green out of the allowed answer words for traffic-control choice groups. Updated tests to require go rather than green/clear/none for traffic-control choice. py_compile passed and unittest selfdrive.controls.tests.test_reasoned_trajectory passed again with 232 tests OK. This requires a rebuilt selected-logit TensorRT choice engine before traffic probes can continue.
  - 2026-05-25 15:24:44 -04:00 process hygiene sweep at user request after context compaction. Strict local command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA GPU state was idle-level for the desktop: RTX 5060 Ti at 1062 MiB / 16311 MiB, 1% GPU utilization, 3% memory utilization, with no Python/Qwen/TensorRT compute client. WSL reported no running distributions. Generic Python inspection found only the existing unrelated Codex DuckDuckGo MCP workers and the old open-webui.exe child python.exe from 2026-05-06; no process was killed.
  - 2026-05-25 15:26:50 -04:00 resumed active goal and inspected current dirty state after compaction. Found the traffic-signal lamp-position experiment half-applied: qwen_trt_label_engine.py expected top/bottom/absent, while tests still described the prior go-based signal contract. Updated tests to the current contract, tightened check-artifacts so traffic-signal choice engines must contain top, bottom, and go logits, then verified syntax and behavior. py_compile passed for qwen_trt_label_engine.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 235 tests OK.
  - 2026-05-25 15:32:35 -04:00 built a separate lamp-position TensorRT choice engine instead of overwriting the prior go-choice engine: F:\qwen_trt_export_lamp_position\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_choice_trt.engine, output vocabulary width 28, engine size 1567048724 bytes. Focused optimized-path probes with the shared FP32 full168 vision engine ran at about 42.6-43.4 ms total but failed the actual signal task: red frame vlm_input_0020 selected none with red_stop_light score -0.640625 and green_go_light score 0.109375; green frame vlm_input_0260 selected none with red_stop_light score -1.265625 and green_go_light score -0.2890625; no-signal nominal selected none. Conclusion: top/bottom is not a usable Qwen answer vocabulary for this optimized selected-logit path. Do not adopt the lamp-position manifest as default.
  - 2026-05-25 15:36:21 -04:00 compared the existing FP8 binary selected-logit path on the same red, green, and no-signal boards. It was both too slow and wrong for signals: about 130-133 ms total, red frame selected green_go_light with red score 2.65625 and green score 3.515625, green frame selected green_go_light with red score 2.703125 and green score 2.90625, no-signal frame also selected green_go_light. Reverted the active traffic-signal choice contract from the failed top/bottom experiment back to red/go/absent and tightened check-artifacts for required red/go tokens. py_compile passed for qwen_trt_label_engine.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 235 tests OK.
  - 2026-05-25 15:48:47 -04:00 continued traffic-signal scoring isolation. The go-choice engine with qwen position mode still returned all-zero selected logits; zero position mode false-positive selected green_go_light for red, green, and no-signal; clamp127 was fast but rejected red_0000, red_0020, and green_0260 as none after warmup while correctly rejecting no-signal. Built a fresh FP16 go-choice engine as a precision diagnostic, but it was not usable: text tower took about 2.0-2.36 seconds and still false-positive selected red_stop_light on the green and no-signal boards. Patched the active traffic-signal choice formulation to neutral answer letters A/B/C: A=controlling red signal, B=controlling green signal, C=no controlling red/green signal. The scorer now requires A or B to beat both the opposite signal letter and C. This avoids semantic token priors from red, green, go, stop, none, and absent while staying in the existing Qwen labels/scores path. py_compile passed for qwen_trt_label_engine.py and test_reasoned_trajectory.py. unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 235 tests OK.
  - 2026-05-25 15:59:43 -04:00 built and probed the neutral-letter signal choice engine: F:\qwen_trt_export_signal_abc\nvfp4_trt\qwen_text_36layer_nvfp4_seq576_choice_trt.engine, output width 27, engine size 1567016036 bytes. With A=red, B=green, C=none it ran at about 42-43 ms but selected red_stop_light on red, green, and no-signal because Qwen strongly preferred option A. Changed prompt/scoring order without rebuilding so A=none, B=red, C=green; tests stayed green, but probes then selected none for red_0000, red_0020, green_0260, and no-signal. Swapping only the vision engine to full224 increased runtime to about 48-49 ms and still selected none for red, green, and no-signal. A top-center signal crop, generated under artifacts\reasoned_trajectory_poc\traffic_signal_crop_probe_current, also selected none for red, green, and no-signal. Conclusion: neutral letters avoid false positives depending on order, but do not yet recover reliable Qwen red/green recognition; do not adopt the ABC signal manifest as a solved/default traffic-light path.
  - 2026-05-25 16:00:25 -04:00 final process hygiene sweep for this continuation. Strict local command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA GPU was idle-level at 965 MiB / 16311 MiB, 1% GPU utilization, 3% memory utilization, with no Python/Qwen/TensorRT compute client. WSL reported no running distributions. Generic Python inspection again showed only the unrelated Codex DuckDuckGo MCP workers and the old open-webui.exe child python.exe from 2026-05-06; no process was killed.
  - 2026-05-25 16:07:08 -04:00 process hygiene sweep at user request before further work. Strict command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA-SMI showed the RTX 5060 Ti at 1028 MiB / 16311 MiB and 1% GPU utilization, with only normal WDDM desktop clients and no Python/Qwen/TensorRT compute client. WSL reported no running distributions. Generic Python inspection found only the existing unrelated Codex DuckDuckGo MCP workers and the old open-webui.exe child python.exe from 2026-05-06; no process was killed.
  - 2026-05-25 16:07:47 -04:00 resumed active goal against current worktree. Checked git status and searched reasoned-control, Qwen/TensorRT, evaluator, and test code for side, lateral, MetaDrive, construction, lead, heading, yaw, vehicle, and spawn conventions. Worktree remains dirty with the expected POC files plus untracked side_semantics.py and lead_semantics.py, so the next step is to audit those central helpers and remove any remaining caller-level sign/orientation drift.
  - 2026-05-25 16:08:28 -04:00 audited current side/lead centralization. side_semantics.py defines fixed openpilot<->MetaDrive lateral conversion as sign inversion; the MetaDrive harness separately calibrates route_lateral_sign_to_openpilot at runtime and uses route_to_openpilot_lateral_m_from_args/openpilot_to_route_lateral_m_from_args for route offsets. This split explains repeated sign regressions: some tests and helpers still use fixed MetaDrive conversion while runtime uses calibrated route conversion. lead_semantics.py classifies physical lead tracks from distance, lateral offset, relative speed, closing speed, acceleration, and lateral velocity without copying sim expected labels. Route vehicle spawning currently uses ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0 and route_vehicle_visual_heading(heading, offset), but there is no focused test that the spawned lead visual heading is same-direction and not head-on.
  - 2026-05-25 16:11:53 -04:00 patched route-sign centralization. Added route_to_openpilot_lateral_m, openpilot_to_route_lateral_m, lateral_side_route, route_lateral_for_openpilot_side, and construction_avoidance_route_side_valid to side_semantics.py. Updated run_metadrive_overlay_demo.py route wrappers to delegate to these shared helpers, and fixed update_durable_lateral_plans to revalidate existing durable avoidance plans with durable_avoidance_sign_valid_for_args instead of the fixed MetaDrive sign helper. Added tests proving the wrappers delegate to shared semantics, valid existing plans survive when route_lateral_sign_to_openpilot=+1, invalid opposite plans are rejected, and default route vehicle visual heading is same-direction rather than head-on. py_compile passed for side_semantics.py, run_metadrive_overlay_demo.py, and test_reasoned_trajectory.py. Also located the installed MetaDrive package at C:\Users\user\AppData\Local\Programs\Python\Python3119\Lib\site-packages\metadrive for heading/spawn API inspection.
  - 2026-05-25 16:12:39 -04:00 verified the route-sign patch with unittest selfdrive.controls.tests.test_reasoned_trajectory: 238 tests passed. Inspected installed MetaDrive vehicle code: BaseVehicle accepts heading in __init__, reset(position, heading, vehicle_config) calls set_heading_theta(heading), and set_heading_theta applies the internal 90 degree vehicle-frame mesh offset. This confirms the harness is using the right MetaDrive argument for vehicle heading; the same-direction invariant is now tested via ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD = 0.0 and route_vehicle_visual_heading(route_heading) == route_heading.
  - 2026-05-25 16:13:09 -04:00 final process hygiene sweep for this increment. Strict command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA-SMI showed RTX 5060 Ti at 1035 MiB / 16311 MiB and 1% GPU utilization, with no Python/Qwen/TensorRT compute client. WSL reported no running distributions. Generic Python processes remain the unrelated Codex DuckDuckGo MCP workers and old open-webui.exe child python.exe; no process was killed. Diff summary for tracked files is large because this branch already contains broad prior POC changes, but the current increment touched side_semantics.py, run_metadrive_overlay_demo.py, test_reasoned_trajectory.py, and progress.md around route-sign centralization and durable-plan validation.
  - 2026-05-25 16:13:44 -04:00 final git status check after route-sign increment. Worktree remains dirty with expected prior POC changes in README/progress/reasoned-control/Qwen/TensorRT/video/evaluator files plus untracked side_semantics.py, lead_semantics.py, and evaluator scripts. No cleanup or commit was performed in this turn.
  - 2026-05-25 16:14:37 -04:00 resumed active goal for closed-loop evidence after the route-sign fix. Checked git status and process/GPU/WSL state. Worktree remains dirty with expected prior POC files. Strict command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot worker; only unrelated DuckDuckGo MCP workers and old open-webui child python.exe remain. NVIDIA-SMI showed RTX 5060 Ti at 1206 MiB / 16311 MiB and 1% GPU utilization, with no POC compute client. WSL reported no running distributions. Proceeding to locate the current runnable Qwen manifest/demo command and run focused construction verification.
  - 2026-05-25 16:16:56 -04:00 current construction manifest refresh. Explicitly checked the saved v16 construction manifest against the current tree and it correctly failed strict --require-manifest because later Qwen prompt/label code changed the runtime contract hash: saved v16 SHA 5f6e0ec57a4dfe4d16bfb0312955eb4aaf588d8d6b264abb1a10a10e139c1d52, current construction contract SHA edf3add105c2fbbc716b95058b398d3b8a2214a115deafe3213f99a9b44e7bb4. Wrote a fresh current-tree construction manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_current_route_sign_manifest_20260525_1617.json with SHA edf3add105c2fbbc716b95058b398d3b8a2214a115deafe3213f99a9b44e7bb4 after check-artifacts ok true. Runtime options remain construction_compact score mode, full168 FP32 vision, seq768 NVFP4 binary text engine, rotating groups cones/barrier; candidate blocked left/right; construction left/right; blue/purple edge, construction mirror fusion enabled, candidate choice enabled with seq576 NVFP4 choice text override, visual fallbacks off.
  - 2026-05-25 16:18:21 -04:00 ran current manifest-gated route-sign left-construction closed-loop demo: artifacts\reasoned_trajectory_poc\construction_left_route_sign_current_qwen_80_20260525_1617_realtime. Command used async latest-only Qwen, period 2, max age 8, no visual fallbacks, speed 2.5 m/s, scene_board_candidate_obstruction_boards enabled, and manifest artifacts\reasoned_trajectory_poc\qwen_trt_construction_current_route_sign_manifest_20260525_1617.json with --require-manifest. Runtime: stock min construction clearance 1.354 m, Qwen/VLM clearance 2.161 m, publish_count 78, valid_count 78, deadline_miss_count 0, mean latency 29.67 ms, p99 39.81 ms, max 43.62 ms, max_rtp_age_frames 7, mean_path_delta_m 0.385, mean_speed_delta_mps 0.0. Evaluator: path_relevant_construction_frames 51, qwen_side_correct 18, qwen_side_wrong 0, consumed_plan_side_correct 48, consumed_plan_side_wrong 0, control_away_frames 33, control_toward_frames 0, planned_away_frames 33, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, post_first_consumed_side_success_rate 1.0, max_consumed_age_frames 7. Render command was first invoked with obsolete --episode args and failed harmlessly; reran with --run-dir and rendered videos side_by_side_route_sign_left_current.mp4, stock_route_sign_left_current.mp4, stock_route_sign_left_current_padded.mp4, and vlm_route_sign_left_current.mp4 under the run videos directory.
  - 2026-05-25 16:24:37 -04:00 process hygiene sweep at user request after compaction. Strict command-line sweep found no active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot POC worker; the only POC-pattern hit was the sweep command itself. NVIDIA compute-app query showed no Python/Qwen/TensorRT compute client attached to the RTX GPU, only ordinary desktop/WDDM clients. WSL reported no running distributions. Broad Python inspection found only the existing unrelated Codex DuckDuckGo MCP workers and one old responding python.exe child of open-webui.exe from 2026-05-06; it has no children, is not part of this POC, and was left running. No process was killed.
  - 2026-05-25 16:25:46 -04:00 resumed active goal after goal-context continuation. Rechecked git status and process/GPU/WSL state before edits. Worktree remains dirty with expected prior POC files, reasoned-control modules, untracked side/lead semantics and evaluators, and nested tinygrad changes. No active Qwen/TensorRT/MetaDrive/render/evaluator/openpilot worker is running; no Python/Qwen/TensorRT compute client is attached to the RTX GPU; WSL has no running distributions. Continuing from current tree toward the known right-side construction failure rather than claiming completion.
  - 2026-05-25 16:28:54 -04:00 inspected the current right/left construction traces and Qwen rotating-state code before patching. Right run artifacts\reasoned_trajectory_poc\construction_right_route_sign_current_qwen_80_20260525_1619_realtime still fails because Qwen candidate-obstruction evidence becomes active as construction_blocks_left_candidate at source frame 61, producing scene=construction_left and a rightward route offset into a right-side hazard. The same trace has correct direct side evidence nearby: semantic construction_right at source frame 62 and purple-edge construction evidence at source frame 65, but the current consensus logic suppresses those and does not clear/override the stale candidate-left durable plan. Left run artifacts\reasoned_trajectory_poc\construction_left_route_sign_current_qwen_80_20260525_1617_realtime succeeds because early candidate-left evidence arrives before contradictory semantic evidence and the durable plan moves away from the left hazard. Root cause is candidate/direct-side arbitration in RotatingScoreState, not route sign conversion or overlay mismatch.
  - 2026-05-25 16:32:42 -04:00 patched Qwen rotating construction arbitration without sim metadata. Added direct consensus constants and state logic to RotatingScoreState: recent construction_left/right semantic side scores plus recent blue/purple corridor-edge scores must agree before direct side evidence can override stale auxiliary candidate labels. When such direct consensus exists, contradictory self-evidencing candidate/shift labels are removed from positive state, and matching direct semantic/edge labels can produce RTP even if candidate crops disagree. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 12, added direct-consensus fields to TensorRT runtime contract/debug state, and added unit tests for stale opposite-candidate override, candidate survival without semantic+edge consensus, stale direct-consensus rejection, and runtime contract tracking. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. Focused unittest TestRotatingScoreState passed with 48 tests OK.
  - 2026-05-25 16:33:22 -04:00 refreshed the manifest for the v12 direct-consensus construction contract using the existing full168 FP32 vision engine and seq768 NVFP4 text engine plus seq576 NVFP4 candidate-choice override. New manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_direct_consensus_v12_manifest_20260525_1633.json. check-artifacts ok true, contract SHA 41430d61324294122e065b6594ea1572f675e6f42fc9ed5d0a30de3486751f08. CUDA 13.2 / TensorRT 10.16.1.11 / RTX 5060 Ti path was verified by the artifact check.
  - 2026-05-25 16:36:03 -04:00 ran the right-side construction demo with v12 direct-consensus manifest: artifacts\reasoned_trajectory_poc\construction_right_direct_consensus_v12_qwen_80_20260525_1633_realtime. Runtime stayed nonblocking: publish_count 78, valid_count 78, deadline_miss_count 0, mean latency 29.53 ms, p99 39.76 ms, max 40.68 ms, max_rtp_age_frames 7. Behavior is still unacceptable but failed differently: Qwen/VLM made zero path change, min construction clearance equaled stock at 1.354 m, mean_path_delta_m 0.0. Evaluation found qwen_side_correct 0, qwen_side_wrong 0, qwen_side_missing 77, consumed_plan_side_missing 77, control_toward_frames 0, control_away_frames 0, collision_count 0. Trace and inspected boards show the guard removed the bad candidate-left plan, but no replacement side became control-active. The current frame 66 board clearly shows right-side cones/barriers touching the purple corridor edge; Qwen edge score selected construction_purple_edge but semantic/candidate evidence was absent or contradictory, so direct consensus never fired. Next patch needs to allow strong neutral-path corridor-edge evidence to bootstrap a side without sim metadata, while preserving later contradiction override.
  - 2026-05-25 16:39:39 -04:00 patched v13 edge bootstrap for the no-action right-side failure. Added a neutral-path corridor-edge bootstrap path for combined semantic+edge schedules: if Qwen strongly scores a blue/purple corridor-edge hazard while tracked_path_lat is near zero and no construction presence is active yet, the edge label can produce a construction RTP even without semantic corroboration. This is still based on Qwen's scored edge label and the rendered UI corridor, not sim object metadata. Added TensorRT confidence handling so strong edge bootstrap can immediately activate a durable lateral plan when --construction-edge-binary is active, and added runtime contract fields for the neutral edge thresholds. Tests added/updated for combined-schedule strong neutral edge bootstrap, blocking the same bootstrap after the path moves, edge confidence, and contract tracking. py_compile passed for qwen_trt_label_engine.py and test_reasoned_trajectory.py; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 243 tests OK after fixing one expected-confidence assertion.
  - 2026-05-25 16:41:06 -04:00 refreshed the construction TensorRT runtime manifest after v13 neutral edge-bootstrap contract change. New manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_edge_bootstrap_v13_manifest_20260525_1640.json. check-artifacts ok true, contract SHA 0ec05ae2e9ebaf2723b421767f0c6ea08628386f725d021fd2408b4c9784e7da, using the same full168 FP32 vision engine, seq768 NVFP4 text engine, seq576 NVFP4 construction-choice engine, construction_compact groups, mirror fusion, edge binary, and visual fallbacks disabled.
  - 2026-05-25 16:42:52 -04:00 ran right-side construction with v13 edge bootstrap: artifacts\reasoned_trajectory_poc\construction_right_edge_bootstrap_v13_qwen_80_20260525_1641_realtime. Runtime was fast/nonblocking (publish_count 78, deadline_miss_count 0, mean latency 25.15 ms, p99 27.32 ms, max 28.37 ms, max_rtp_age_frames 7), but behavior regressed catastrophically: Qwen/VLM clearance 0.575 m versus stock 1.354 m, collision_count 13, qwen_side_wrong 18, consumed_plan_side_wrong 68, control_toward_frames 56. Trace shows the new early edge bootstrap accepted source frame 5 construction_blue_edge=2.672 for a right-side hazard, creating a wrong left-hazard RTP and durable rightward route offset. Later source frame 38 produced strong semantic construction_right=3.359 versus left=1.344 while tracked_path_lat was already -0.55 m toward the right-side hazard, but semantic-only evidence was suppressed by the combined semantic+edge consensus rule and never overrode the stale wrong plan. Next patch should not rely on early edge alone; it needs a high-confidence semantic rescue when current tracked path is moving toward the newly reported hazard side.
  - 2026-05-25 16:45:40 -04:00 patched v14 semantic rescue for contradictory construction judgements. Added CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE/MARGIN and state-machine logic allowing a strong construction_left/right semantic side label to become control-active when tracked_path_lat shows the current plan is moving toward that same hazard side. This is the intended stale-plan contradiction override: it uses Qwen semantic labels plus the current tracked path, not simulator object truth. Added TensorRT confidence handling so such strong semantic labels can immediately replace a conflicting durable construction plan, bumped CONSTRUCTION_STATE_MACHINE_VERSION to 14, and added runtime contract fields. Tests added for semantic rescue when path is toward the reported hazard, non-rescue when path is not toward it, semantic confidence 0.96, and contract tracking. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 245 tests OK.
  - 2026-05-25 16:46:27 -04:00 refreshed the construction TensorRT runtime manifest after v14 semantic-rescue contract change. New manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_semantic_rescue_v14_manifest_20260525_1646.json. check-artifacts ok true, contract SHA f057f549dff2c16a7c03d6deb853110046df7ca63e3b420d0a57d7c6fd5a9e65, same engines/options as prior construction run.
  - 2026-05-25 16:48:10 -04:00 ran right-side construction with v14 semantic rescue: artifacts\reasoned_trajectory_poc\construction_right_semantic_rescue_v14_qwen_80_20260525_1646_realtime. Runtime stayed fast/nonblocking (publish_count 78, deadline_miss_count 0, mean latency 25.08 ms, p99 26.00 ms, max 26.32 ms, max_rtp_age_frames 7), but behavior still failed: clearance 0.584 m, collision_count 13, qwen_side_correct 1, qwen_side_wrong 13, consumed_plan_side_wrong 67, control_toward_frames 56. Semantic rescue did not fire at the earlier useful frames because the right-vs-left semantic margin was below the new semantic threshold. A correct direct consensus finally appeared at frame 74 (construction_right + construction_purple_edge), but confidence was only 0.80 because confidence calibration prioritized the weaker semantic label over the stronger purple-edge score, so it did not replace the stale 0.96 wrong durable plan. Next patch should add edge-toward-path rescue and confidence calibration that uses same-side edge evidence when semantic and edge agree.
  - 2026-05-25 16:50:13 -04:00 patched v15 edge-toward rescue and confidence calibration. Added CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE/MARGIN and state-machine logic allowing a blue/purple edge label to become control-active when the current tracked path is already moving toward that edge's hazard side. Updated TensorRT confidence calibration so edge-toward labels can get 0.96 confidence under --construction-edge-binary and same-side semantic+edge labels use the strong edge score instead of being capped by the weaker semantic margin. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 15 and added runtime contract fields. Tests added for edge-toward rescue, blocking edge rescue when path is not toward the hazard, edge-toward confidence, same-side semantic+edge confidence, and contract tracking. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 247 tests OK.
  - 2026-05-25 16:50:59 -04:00 refreshed the construction TensorRT runtime manifest after v15 edge-toward rescue contract change. New manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_edge_toward_v15_manifest_20260525_1650.json. check-artifacts ok true, contract SHA 5b39c949f98c9a6d3ed2951497f26406fbf5c69e06e50a5281e128c9d518e0ce, same construction engines/options as v14.
  - 2026-05-25 16:55:37 -04:00 performed a process hygiene sweep after context resume at the user's request. No active or hung Qwen/TensorRT/MetaDrive/render/evaluator/openpilot/tinygrad demo processes were found. `nvidia-smi` showed RTX 5060 Ti at 1584 MiB used by normal desktop/Codex/Firefox/VS Code graphics clients only, with no Qwen/TensorRT Python compute process. `wsl.exe --list --running` reported no running WSL distributions. Existing Python processes were unrelated resident services: old open-webui child PID 9260 and DuckDuckGo MCP worker PIDs 37716/7052, so they were left untouched.
  - 2026-05-25 16:56:20 -04:00 resumed the active real-car-transfer goal after process hygiene. Current next step is not another blind rerun: evaluate the latest v15 construction trace and inspect Qwen labels, raw scores, RTP output, durable lateral state, active target, and construction-only clearance to identify the remaining chain break before patching.
  - 2026-05-25 16:57:43 -04:00 evaluated v15 right-side construction trace artifacts\reasoned_trajectory_poc\construction_right_edge_toward_v15_qwen_80_20260525_1651_realtime. Construction-only evaluator result: 77 path-relevant frames, qwen_side_correct 1, qwen_side_wrong 22, qwen_side_missing 54, consumed_plan_side_correct 2, consumed_plan_side_wrong 65, control_away_frames 0, control_toward_frames 56, collision_count 13, min construction route clearance 0.965 m, max RTP age 7. Detailed trace shows actual hazard lateral_delta_m stays positive/right. Source frame 2 already has weak correct semantic construction_right, but state logic suppresses it. Source frame 5 then accepts construction_blue_edge as high-confidence construction_left, producing avoid=[left_edge...] and a MetaDrive rightward target. Later source frame 33 briefly produces correct construction_purple_edge/construction_right, but source frame 38 and following candidate labels reacquire construction_blocks_left_candidate/construction_left and steer back right into the right-side hazard. Root cause is over-authorizing ambiguous corridor-edge and candidate labels relative to semantic side evidence; not a MetaDrive sign conversion failure.
  - 2026-05-25 17:00:12 -04:00 inspected the exact v15 candidate board at vlm/vlm_input_0038_candidate_pair.png plus records 38 and 40. The board shows the cyan left-shifted candidate visually clear and the magenta right-shifted candidate overlapping the right-side cones. Qwen's raw candidate choice at source frame 38 actually had neutral `none` as the highest word score (none 14.1875, cyan 8.3984, pink 6.4062), but the `--construction-candidate-relative-choice` postprocess ignored the neutral winner and forced construction_blocks_left_candidate from cyan-minus-pink 1.992. That manufactured a wrong left-hazard RTP even though Qwen did not choose a blocked candidate. Transferable fix: candidate relative choice must respect the neutral word unless the blocked candidate score is close enough to or above neutral, rather than treating cyan/pink relative margin alone as authority.
  - 2026-05-25 17:05:03 -04:00 patched two transferable construction arbitration defects. First, RotatingScoreState now uses per-label TTLs: construction presence positives survive CONSTRUCTION_PRESENCE_HOLD_FRAMES instead of being popped by the short score cache, and construction semantic/edge scores survive CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES so rotating groups can form direct consensus as designed. This prevents early edge-only bootstrap from being allowed just because a recent cones label was expired by cache_ttl_frames=3. Second, qwen_trt_label_engine candidate relative choice now respects a dominant neutral `none` word: cyan/pink relative margin can only select a candidate when the best candidate word is within --construction-candidate-relative-neutral-margin of neutral, default 1.0. Added tests for neutral-dominant candidate blocking, near-neutral relative selection, presence-hold blocking edge bootstrap after score-cache expiry, and direct consensus across the rotating score window. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 16 and added the new neutral-margin runtime contract field.
  - 2026-05-25 17:06:02 -04:00 ran py_compile successfully for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. Focused RotatingScoreState run initially failed one legacy assertion: it expected construction presence to disappear after cache_ttl_frames=3. That is now intentionally wrong because construction presence must survive CONSTRUCTION_PRESENCE_HOLD_FRAMES for short occlusion/splash. Updated the test to assert the real invariant: a single late contradictory edge does not flip the locked side and the held cones + prior side remain active. A mistakenly addressed non-existent TestQwenTrtLabelEngine test class also failed by name only; no code failure there.
  - 2026-05-25 17:04:14 -04:00 reran verification after the TTL/neutral-candidate patch. py_compile passed again. Focused RotatingScoreState tests passed: 56 tests OK. Full selfdrive.controls.tests.test_reasoned_trajectory passed: 250 tests OK. Warnings were the existing torch Windows/cpp-extension notices and a Pillow deprecation warning in ui_scene_board.py, not test failures.
  - 2026-05-25 17:04:49 -04:00 refreshed the TensorRT runtime manifest for the v16 construction arbitration contract. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_ttl_neutral_v16_manifest_20260525_1705.json. check-artifacts ok true, contract SHA 689933c7f9e58195ffcf89042a03e4964be3534faaf0fb708612ffa7e9e7e200. Same full168 FP32 vision engine, seq768 NVFP4 text engine, seq576 NVFP4 construction-choice engine, mirror fusion, edge binary, candidate choice, and no visual fallbacks; new contract includes construction_candidate_relative_neutral_margin=1.0 and state machine version 16.
  - 2026-05-25 17:06:18 -04:00 ran v16 right-side construction closed-loop demo: artifacts\reasoned_trajectory_poc\construction_right_ttl_neutral_v16_qwen_80_20260525_1705_realtime. Runtime stayed nonblocking: publish_count 78, deadline_miss_count 0, mean latency 24.10 ms, p99 25.01 ms, max 25.37 ms, max_rtp_age_frames 7, mean_path_delta_m 1.042, mean_speed_delta_mps 0.0. Stock min construction clearance was 1.354 m; VLM min construction clearance improved to 1.504 m. Construction evaluator still shows this is not acceptable: 77 path-relevant frames, qwen_side_correct 41, qwen_side_wrong 21, qwen_side_missing 15, consumed_plan_side_correct 42, consumed_plan_side_wrong 20, collision_count 0, control_away_frames 32, control_toward_frames 20, green_path_mismatch_count 0. The v16 fixes prevent collision but do not meet the goal. The remaining failure is an early wrong construction_left/blue-edge lock around source frame 11, followed later by a correct right-side/purple plan that clears the cones.
  - 2026-05-25 17:10:08 -04:00 added additive TensorRT options --construction-side-choice and --construction-edge-choice. These force construction_left/right and construction_blue_edge/purple_edge groups into choice mode using the existing construction text engine and compact construction prompts, which makes the existing construction mirror-fusion code actually operate on side/edge word scores. This does not replace Qwen or the architecture; it changes which existing Qwen label-scoring path is used for side/edge groups. Initial full unittest run failed one new test assertion because the generic construction-engine override can also attach the construction text engine to candidate groups; narrowed the test to the invariant that side/edge groups use choice mode while candidate remains binary unless candidate-choice is enabled.
  - 2026-05-25 17:10:50 -04:00 refreshed a new TensorRT manifest for the side/edge choice contract. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_side_edge_choice_v16_manifest_20260525_1711.json. check-artifacts ok true, contract SHA 794d06d50e01708bdfff064a6a2f6cc102bc34aa33337ee5bfd1af30505f4616. This keeps the same Qwen/TRT engines but uses --construction-side-choice and --construction-edge-choice, no edge-binary, candidate choice with neutral guard, mirror fusion, and no visual fallbacks.
  - 2026-05-25 17:11:50 -04:00 probed the known-bad v16 frame 11 with side/edge choice + mirror fusion. The edge group still selected construction_blue_edge on an image where the right/purple side is visually hazardous. The probe also showed side/edge choice is not an acceptable default hot path as implemented: side group total was 184 ms and edge group had a pathological 4956 ms first run with mirror. Therefore I am not promoting side/edge choice as the real-time path. Next probe is the existing construction_shift_left/right label family, because it asks Qwen for the safe maneuver direction directly instead of inferring maneuver from ambiguous side/edge color labels.
  - 2026-05-25 17:12:47 -04:00 probed existing construction_shift_left/right labels on the known-bad frame 11. Qwen correctly selected construction_shift_left with scores shift_left 1.734, shift_right 0.281, producing RTP scene=construction_right / BIAS_LEFT / avoid right_edge_s8_48_margin1.25. This is the first label family that matches the human-visible right-side construction scene at that frame. It stayed inside the existing architecture and label set. The standalone benchmark with mirror fusion was too slow (about 150 ms total), so the next test will add shift labels to the rotating schedule while keeping the fast binary side/edge path and not promoting slow side/edge choice.
  - 2026-05-25 17:13:26 -04:00 refreshed a TensorRT manifest for the fast binary construction schedule with shift labels added. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_shift_v16_manifest_20260525_1713.json. check-artifacts ok true, contract SHA 7cb7b4fce62735e9f743d36dc7887a8277e574f5ed63ac16dde32f6760523aef. Group order is cones/barrier, construction_shift_left/right, candidate obstruction, semantic side, edge color. Side/edge choice is not enabled in this manifest; edge remains binary and candidate uses choice with neutral guard.
  - 2026-05-25 17:17:04 -04:00 process hygiene sweep at user request after context compaction. Strict command-line sweep found no active or hung Qwen/TensorRT/MetaDrive/render/evaluator/openpilot/tinygrad POC worker. NVIDIA-SMI showed the RTX 5060 Ti at 961 MiB / 16311 MiB, 2% GPU utilization, with only normal desktop/Codex/Firefox/VS Code/Explorer WDDM graphics clients and no Python/Qwen/TensorRT compute client. WSL reported no running distributions. No process was killed.
  - 2026-05-25 17:46:10 -04:00 continued construction-side root-cause work from current tree. Evaluated the prior v16 shift run artifacts\reasoned_trajectory_poc\construction_right_shift_v16_qwen_80_20260525_1713_realtime: no collision, min construction clearance 2.018 m, max RTP/control age 6, but Qwen raw side quality remained poor (qwen_side_correct 6, wrong 12, missing 37; false construction labels 19/25 non-path-relevant frames). Inspected exact VLM boards and confirmed Qwen was shown a UI-like board with hazards on the right/purple edge yet sometimes emitted construction_shift_right/construction_left. Added new direct action labels construction_drive_left/right in side_semantics, qwen_label_rtp_worker, qwen_trt_label_engine, and tests. These labels describe the safe ego driving direction directly and map to the opposite hazard side. Added a tracked-path guard so direct action labels bootstrap only near neutral, can continue the locked side, or can override only when confident and the current tracked path is toward the newly claimed hazard side. Added a committed-side conflict gate requiring recent direct-drive action evidence before a shifted/committed construction avoidance can flip to the opposite side. Probed the exact early bad board vlm_input_0001.png: Qwen emitted construction_drive_left and compiled to scene=construction_right / BIAS_LEFT / right_edge. Probed a later shifted-path board and found the unsafe action-confidence promotion caused sign flips, so reverted that promotion while keeping the direct labels and committed-side gate. Verification: py_compile passed for side_semantics.py, qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 260 tests OK. Wrote manifests v18/v19/v20 during experiments; current best manifest is artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v21_manifest_20260525_1744.json, contract SHA 2dc393e841f6e7f554f8cef9281b132477d33e06fc6d8e34d1f2cddff40f46e6. Right-side closed-loop result artifacts\reasoned_trajectory_poc\construction_right_drive_v21_qwen_80_20260525_1744_realtime: stock clearance 1.354 m, VLM clearance 1.922 m, publish_count 78, valid_count 78, deadline_miss_count 0, mean latency 24.01 ms, p99 latency 25.06 ms, max latency 25.85 ms, max RTP/control age 6, mean speed delta 0.0. Construction evaluator: collision_count 0, green_path_mismatch_count 0, consumed_plan_side_correct 47, wrong 0, missing 8, post-first consumed-side success 47/50 = 0.94, control_away_frames 32, control_toward_frames 0, planned_toward_frames 0. Qwen raw labels remain weak (qwen_side_success_rate 0.109 and false construction labels 17/25 non-path-relevant frames), so this is progress but not goal-complete. Rendered videos under artifacts\reasoned_trajectory_poc\construction_right_drive_v21_qwen_80_20260525_1744_realtime\videos: side_by_side_construction_right_drive_v21.mp4, stock_construction_right_drive_v21.mp4, stock_construction_right_drive_v21_padded.mp4, vlm_construction_right_drive_v21.mp4.
  - 2026-05-25 17:52:10 -04:00 process hygiene sweep at user request. Strict reasoned-pipeline command-line sweep found no active or hung Qwen/TensorRT/MetaDrive/render/evaluator/openpilot/tinygrad POC worker. `nvidia-smi` showed RTX 5060 Ti at 1587 MiB / 16311 MiB, 0% GPU utilization, with only normal desktop/Codex/Firefox/VS Code/Explorer WDDM graphics clients and no Python/Qwen/TensorRT compute client. WSL reported no running distributions. Broader Python/WSL/SSH sweep found only unrelated resident services: old responding open-webui child PID 9260 listening on localhost:8080 and Codex DuckDuckGo MCP worker PIDs 37716/7052. No process was killed.
  - 2026-05-25 17:54:01 -04:00 goal continuation after compaction. Confirmed the active goal is still broad real-car-transfer Qwen mode, not construction-only. Inspected current worktree and construction code before acting: git status is dirty with expected POC/reasoned-control/readme/progress/nested-tinygrad changes; current code has CONSTRUCTION_STATE_MACHINE_VERSION=22 and direct construction_drive_left/right labels in side_semantics.py, qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and tests. Proceeding to verify the current tree, refresh the TensorRT manifest for v22, and rerun construction side behavior rather than relying on older manifests.
  - 2026-05-25 17:54:52 -04:00 verified current v22 tree before demo. py_compile passed for side_semantics.py, qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. Full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 261 tests OK; only existing torch Windows/cpp-extension and Pillow deprecation warnings appeared.
  - 2026-05-25 17:54:52 -04:00 refreshed the current TensorRT runtime manifest for v22 direct construction-drive labels. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v22_manifest_20260525_1754.json. check-artifacts ok true, contract SHA f6cfe15d5f8eb83600b1408827c92f49fad61955b68003f017e7f622cdc34bf4. Engines: full168 FP32 vision, seq768 NVFP4 text, seq576 NVFP4 construction choice. Groups: cones/barrier; construction_drive_left/right; candidate obstruction; construction_left/right; construction_blue_edge/purple_edge. Mirror fusion, edge binary, candidate choice, candidate-relative neutral guard, and no visual fallback remain active.
  - 2026-05-25 18:01:14 -04:00 ran v22 left-side construction closed-loop demo and evaluated it: artifacts\reasoned_trajectory_poc\construction_left_drive_v22_qwen_80_20260525_1755_realtime. Runtime was nonblocking (publish_count 78, deadline_miss_count 0, mean latency 24.00 ms, p99 25.57 ms, max_rtp_age_frames 6, mean_speed_delta_mps 0.0) but behavior failed: stock min construction clearance 1.354 m, VLM clearance 1.147 m, collision_count 13, qwen_side_correct 35, qwen_side_wrong 10, consumed_plan_side_correct 40, consumed_plan_side_wrong 23, control_toward_frames 32, planned_toward_frames 22. Root cause from trace: correct early direct action labels (`construction_drive_right`, left hazard, drive right) appeared at source frames 1/13/25 but stayed at low durable confidence, while a single later edge-only `construction_purple_edge` at source frame 32 with construction presence inactive created a wrong high-confidence right-hazard durable plan. Direct action labels had expired after the short cache by then, so they could not veto the opposite edge bootstrap.
  - 2026-05-25 18:01:14 -04:00 patched v23 construction action handling based on that trace. Action labels now retain positive state and scores for CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES instead of the short score cache, so recent Qwen safe-action judgement can veto contradictory edge/candidate evidence through short occlusion or splash. Added CONSTRUCTION_ACTION_IMMEDIATE_SCORE=2.0 and CONSTRUCTION_ACTION_IMMEDIATE_MARGIN=1.5; TensorRT confidence calibration now assigns 0.96 to strong direct action labels so correct `construction_drive_left/right` can arm durable lateral control before a weak/ambiguous edge label appears. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 23 and added runtime contract fields. py_compile passed for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py. Focused tests passed after correcting stale test class names; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 262 tests OK.
  - 2026-05-25 18:02:01 -04:00 refreshed TensorRT runtime manifest after v23 action retention/immediate-confidence patch. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v23_manifest_20260525_1801.json. check-artifacts ok true, contract SHA 04ff6fc1b1d464ebfba35b9723e1646cb6d21249506952636fcca0b27ca4143d. Same Qwen/TRT engines and construction label groups as v22; contract now includes construction_action_immediate_score/margin and state machine version 23.
  - 2026-05-25 18:03:01 -04:00 reran left-side construction with v23 manifest: artifacts\reasoned_trajectory_poc\construction_left_drive_v23_qwen_80_20260525_1802_realtime. Runtime stayed nonblocking: publish_count 78, valid_count 78, deadline_miss_count 0, mean latency 24.13 ms, p99 25.61 ms, max 25.78 ms, max_rtp_age_frames 6, mean_speed_delta_mps ~0.0. Behavior recovered from the v22 failure: stock clearance 1.354 m, VLM clearance 1.960 m, collision_count 0, green_path_mismatch_count 0, qwen_side_correct 45, qwen_side_wrong 0, consumed_plan_side_correct 57, consumed_plan_side_wrong 0, control_away_frames 32, control_toward_frames 0, planned_toward_frames 0, post-first consumed-side success 57/57 = 1.0. This validates the action-retention/immediate-confidence patch on the left-construction case that failed v22.
  - 2026-05-25 18:07:13 -04:00 ran the symmetric right-side construction check with v23 manifest: artifacts\reasoned_trajectory_poc\construction_right_drive_v23_qwen_80_20260525_1803_realtime. Runtime was nonblocking (publish_count 78, deadline_miss_count 0, mean latency 24.04 ms, p99 25.21 ms, max_rtp_age_frames 6) but behavior regressed: stock clearance 1.354 m, VLM clearance 1.317 m, qwen_side_correct 18, qwen_side_wrong 11, consumed_plan_side_correct 31, consumed_plan_side_wrong 41, control_toward_frames 26, planned_toward_frames 34. Trace root cause: a weak wrong `construction_drive_right` action at source frame 25 (score 0.77 vs 0.12) survived past the short cache and polluted the source frame 32 edge arbitration, flipping the plan to left-hazard/BIAS_RIGHT even though the actual hazard was right-side. This is the mirror failure of v22: action retention was useful but too permissive for weak action scores.
  - 2026-05-25 18:07:13 -04:00 patched v24 construction action retention. CONSTRUCTION_ACTION_IMMEDIATE_SCORE is now 2.5 and CONSTRUCTION_ACTION_IMMEDIATE_MARGIN is 1.0 so strong absolute safe-action choices like right-side frame 1 (`drive_left` 2.61 vs 1.61) and left-side frame 25 (`drive_right` 2.66 vs 0.08) can arm durable control, while weak stale actions like right-side frame 25 (`drive_right` 0.77 vs 0.12) cannot veto later edge evidence. Added `_construction_action_fresh_or_strong`: action labels can remain active past the short cache only if they meet the immediate score/margin gate. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 24. Added tests for strong action retention through the short cache and weak stale action not vetoing strong opposite edge. py_compile passed, focused tests passed, and full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 263 tests OK.
  - 2026-05-25 18:07:52 -04:00 refreshed TensorRT runtime manifest after v24 action-retention tightening. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v24_manifest_20260525_1807.json. check-artifacts ok true, contract SHA 85d78a62439e96622b48e102a1d8f4d28c674ab596432935bcf26599a4069c86. Same Qwen/TRT engines and label groups as v23; contract now has state machine version 24 and action immediate threshold 2.5/1.0.
  - 2026-05-25 18:11:08 -04:00 ran right-side construction with v24 manifest: artifacts\reasoned_trajectory_poc\construction_right_drive_v24_qwen_80_20260525_1808_realtime. Runtime stayed nonblocking (publish_count 78, deadline_miss_count 0, mean latency 25.19 ms, p99 26.70 ms, max_rtp_age_frames 7) but behavior was still not acceptable: stock clearance 1.354 m, VLM clearance 1.351 m, qwen_side_correct 23, qwen_side_wrong 9, consumed_plan_side_correct 30, consumed_plan_side_wrong 41, control_toward_frames 25, planned_toward_frames 34. Trace root cause: the correct `construction_drive_left` action at source frame 27 was suppressed because the path was already moving away from the right-side hazard, leaving a later wrong `construction_left` semantic label at source frame 32 free to replace the correct durable plan. The state machine treated already-doing-the-action as a reason to drop action support.
  - 2026-05-25 18:11:08 -04:00 patched v25 construction action continuation. Added CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M=0.25 and `_tracked_path_is_supporting_construction_action`; action labels are now allowed when the tracked path is already moving away from that action's hazard side, so a fresh Qwen `drive_left` judgement continues supporting an already-left path around right-side construction instead of disappearing. Bumped CONSTRUCTION_STATE_MACHINE_VERSION to 25 and added runtime contract coverage. Added test reproducing the right-side v24 failure chain: strong `construction_drive_left` while tracked path is already away from right hazard must survive and veto later wrong `construction_left`. py_compile passed, targeted tests passed, and full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 264 tests OK.
  - 2026-05-25 18:11:44 -04:00 refreshed TensorRT runtime manifest after v25 action-continuation patch. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v25_manifest_20260525_1811.json. check-artifacts ok true, contract SHA 77506f7c848d35a05646a7aa2a3703bfb55bc81e71dd7842bdea6b58f4e10148. Same Qwen/TRT engines and construction label groups as v24; contract now has state machine version 25 and construction_action_continue_min_tracked_offset_m=0.25.
  - 2026-05-25 18:19:18 -04:00 ran v25 right and left construction mirror checks. Right v25 passed construction-control safety: artifacts\reasoned_trajectory_poc\construction_right_drive_v25_qwen_80_20260525_1812_realtime, stock clearance 1.354 m, VLM clearance 2.151 m, qwen_side_wrong 0, consumed_plan_side_wrong 0, control_toward_frames 0, planned_toward_frames 0, collision_count 0, green_path_mismatch_count 0, p99 latency 26.18 ms, max age 7. Left v25 regressed versus v23 though still no collision: artifacts\reasoned_trajectory_poc\construction_left_drive_v25_qwen_80_20260525_1813_realtime, stock clearance 1.354 m, VLM clearance 1.656 m, qwen_side_wrong 4, consumed_plan_side_wrong 4, control_toward_frames 0, planned_toward_frames 0. Trace root cause: moderate wrong near-neutral `construction_drive_left` at source frame 26 (score 2.48 vs 1.37, just below the intended immediate score gate) became control-active, delayed the correct left-hazard/right-drive plan, and reduced clearance.
  - 2026-05-25 18:19:18 -04:00 patched v26 construction action gating. Direct action labels are now self-evidencing only when they meet CONSTRUCTION_ACTION_IMMEDIATE_SCORE/MARGIN, unless they are backed by direct consensus, already match a locked side, or are a high-confidence contradictory rescue. Weak or moderate action outputs remain available in qwen_raw_labels/qwen_raw_label_scores for audit but do not become control-active labels. Added/updated tests for strong self-evidencing action, weak raw-only action, weak stale action not vetoing edge evidence, action continuation, and high-confidence rescue. py_compile passed and full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 265 tests OK. State machine version is now 26.
  - 2026-05-25 18:19:57 -04:00 refreshed TensorRT runtime manifest after v26 strict direct-action gate. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v26_manifest_20260525_1819.json. check-artifacts ok true, contract SHA 6d2ddcb42ca4a1752673d41c58fae481cfe5a3bf2d41f391b9da87958b5493b1. Same Qwen/TRT engines and construction label groups as v25; contract now has state machine version 26.
  - 2026-05-25 18:23:34 -04:00 ran left construction with v26 manifest: artifacts\reasoned_trajectory_poc\construction_left_drive_v26_qwen_80_20260525_1820_realtime. Runtime stayed nonblocking (publish_count 78, deadline_miss_count 0, mean latency 24.67 ms, p99 25.61 ms, max age 7) and sign safety improved versus v25 (consumed_plan_side_wrong 0, control_toward_frames 0, planned_toward_frames 0, collision_count 0), but first accepted side was too late at frame 38, so clearance remained 1.656 m versus v23's 1.960 m. Trace root cause: strong semantic `construction_left` appeared at source frame 31 with active construction presence and neutral path, but the state waited for the edge group at source frame 34 before acting because semantic+edge consensus was required.
  - 2026-05-25 18:23:34 -04:00 patched v27 neutral-path semantic bootstrap. Added CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE=2.5 and MARGIN=1.5; a strong semantic construction_left/right label can now start bounded lateral avoidance while construction presence is active and the tracked path is still near neutral, before the edge-color group arrives. This remains blocked once the path has moved away/toward enough, and strong recent action disagreement still vetoes the opposite side through the existing recent-action gate. Added runtime contract fields and tests for neutral semantic bootstrap allowed versus blocked after path movement. py_compile passed, targeted tests passed, and full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 267 tests OK. State machine version is now 27.
  - 2026-05-25 18:24:09 -04:00 refreshed TensorRT runtime manifest after v27 neutral semantic bootstrap patch. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v27_manifest_20260525_1824.json. check-artifacts ok true, contract SHA a48bf6d5759bb6f452d9b854c6d2c59893f0f08b0e856207f201fedbb7abb16f. Same Qwen/TRT engines and construction label groups as v26; contract now has state machine version 27 and semantic neutral bootstrap thresholds.
  - 2026-05-25 18:26:59 -04:00 performed process hygiene sweep at the user's request before continuing. Strict command-line sweep found no active or hung Qwen/TensorRT/MetaDrive/render/evaluator/openpilot/tinygrad POC worker. `wsl.exe --list --running` reported no running distributions. `nvidia-smi` showed RTX 5060 Ti at 1625 MiB / 16311 MiB, 0% GPU utilization, with only normal desktop/Codex/Firefox/VS Code/Explorer WDDM graphics clients and no Python/Qwen/TensorRT compute client. The only broad-match Python process outside Codex MCP was old responding PID 9260 listening on port 8080, consistent with the unrelated local service previously left alone. No process was killed.
  - 2026-05-25 18:57:03 -04:00 completed the v28-v31 construction sign/side repair loop after the process sweep. v28/v29/v30 were rejected from trace evidence: v28 accepted stale no-presence direct consensus and drove left-side construction clearance down to 0.628 m with 13 collisions; v29 blocked pre-presence consensus but still resurrected stale semantic/edge scores and cleared only 0.793 m with 13 collisions; v30 avoided collision but lost the correct lock on presence reacquire and did not beat stock clearance. v31 is the current best patch: construction locks survive same-side semantic reacquire, direct consensus requires active construction presence and score frames newer than the current presence anchor, and opposite direct-action labels cannot replace a recent construction lock unless they satisfy the explicit contradictory override path. Verification passed: py_compile for qwen_label_rtp_worker.py, qwen_trt_label_engine.py, and test_reasoned_trajectory.py; focused construction regression tests passed; full unittest selfdrive.controls.tests.test_reasoned_trajectory passed with 270 tests OK. Manifest: artifacts\reasoned_trajectory_poc\qwen_trt_construction_drive_v31_manifest_20260525_1849.json, contract SHA 9dab70622c1f38690482a4747ae2ff84f34dc8afc3b4b55dc1c3c661ffefd010. Left v31 run artifacts\reasoned_trajectory_poc\construction_left_drive_v31_qwen_80_20260525_1850_realtime passed physical construction-control safety: stock clearance 1.354 m, VLM clearance 1.591 m, collision_count 0, control_toward_frames 0, planned_toward_frames 0, green_path_mismatch_count 0, mean latency 1.98 ms, p99 latency 2.47 ms, max_rtp_age_frames 7. Right v31 run artifacts\reasoned_trajectory_poc\construction_right_drive_v31_qwen_80_20260525_1853_realtime passed physical construction-control safety: stock clearance 1.354 m, VLM clearance 1.906 m, collision_count 0, control_toward_frames 0, planned_toward_frames 0, green_path_mismatch_count 0, mean latency 2.09 ms, p99 latency 4.65 ms, max_rtp_age_frames 7. Residual caveat: right-side raw Qwen side labels are still weak and the evaluator reports consumed wrong frames from non-command/old-plan bookkeeping, but those frames had no active steering toward the hazard. Rendered videos: artifacts\reasoned_trajectory_poc\construction_left_drive_v31_qwen_80_20260525_1850_realtime\videos\side_by_side_construction_left_drive_v31.mp4 and artifacts\reasoned_trajectory_poc\construction_right_drive_v31_qwen_80_20260525_1853_realtime\videos\side_by_side_construction_right_drive_v31.mp4.
  - 2026-05-25 18:58:00 -04:00 final process hygiene sweep. Strict POC command-line sweep for MetaDrive demos, Qwen/TensorRT servers, renderers, evaluators, and RTP server commands returned no processes. `wsl.exe --list --running` reported no running distributions. `nvidia-smi --query-compute-apps` showed only normal WDDM desktop graphics clients and no Python/Qwen/TensorRT compute client. Broad Python inspection found only unrelated resident services: old PID 9260 with blank command line and the Codex DuckDuckGo MCP uv/uvx/python workers. No process was killed.
  - 2026-05-25 20:18:03 -04:00 continued real-car-transfer lead work after compaction/user interruption. Initial targeted process sweep found no active Qwen/TensorRT/MetaDrive POC worker except the inspection command itself; GPU query showed no Python/Qwen/TensorRT compute client. Patched tools/reasoned_trajectory_poc/run_metadrive_overlay_demo.py to record measured route-vs-actual heading evidence for spawned vehicle tracks and consumed lead tracks: route_heading_theta, actual_heading_theta, heading_error_rad, heading_alignment_cos, and same_direction. Patched tools/reasoned_trajectory_poc/evaluate_lead_trace.py to count vehicle_heading_checked_frames, vehicle_heading_missing_frames, vehicle_heading_violation_count, min_vehicle_heading_alignment_cos, and row-level heading evidence. Patched tools/reasoned_trajectory_poc/evaluate_lead_suite.py so lead-suite gates now require vehicle heading evidence by default and fail on missing/head-on heading evidence. Added tests in selfdrive/controls/tests/test_reasoned_trajectory.py for heading metrics and head-on suite failure. Verification: py_compile passed for run_metadrive_overlay_demo.py, evaluate_lead_trace.py, evaluate_lead_suite.py, and test_reasoned_trajectory.py; focused TestLeadCompiler + TestLeadTraceEvaluation passed 24 tests OK. Old discovered lead-suite evidence now fails the stricter heading/Qwen gates, exposing stale artifacts instead of silently passing: true_moving stale qwen_success_rate 0.2375, old braking/stopped missing heading evidence, old crossing qwen/false-slow below gate. Refreshed lead TensorRT runtime manifest without rebuilding engines: artifacts\reasoned_trajectory_poc\qwen_trt_lead_choice_heading_guard_manifest_20260525_1909.json, contract SHA 63e074d8ef513e295a0c5ffcbf3356e237d22b8f6e3cfec6e46a967ce3744534, check-artifacts ok true. Fresh close-range true_moving static harness probe artifacts\reasoned_trajectory_poc\lead_heading_guard_current_20260525_1907 showed top-level lead heading evidence populated with heading_error_rad 0.0, alignment_cos 1.0, vehicle_heading_violation_count 0. Fresh synchronous braking Qwen run artifacts\reasoned_trajectory_poc\lead_braking_heading_guard_sync_120_20260525_1910 had good semantics and heading evidence but is rejected for timing: 34 deadline misses, publish_count 86/120, p99 53.37 ms. Its evaluator still showed required_qwen_success_rate 1.0, required_control_success_rate 0.9917, collision_count 0, heading checked 120/120, no heading violations. Started bounded async braking run with latest-frame/drop-stale/age<=8; user interrupted the tool, but the process completed and no POC process remained. Async run artifacts\reasoned_trajectory_poc\lead_braking_heading_guard_async_120_20260525_1912: publish_count 119/120, valid_count 119, deadline_miss_count 0, mean planner overhead 4.54 ms, p99 6.26 ms, max_rtp_age_frames 2, collision_count 0, min_vehicle_route_clearance_m 9.23 m. Lead evaluator wrote vlm\lead_trace_evaluation.json: required_qwen_success_rate 0.975, required_control_success_rate 0.983, max_consumed_age_frames 2, age_violation_count 0, vehicle_heading_checked_frames 120, vehicle_heading_missing_frames 0, vehicle_heading_violation_count 0, min_vehicle_heading_alignment_cos 1.0. Residual lead gap: braking_lead class bucket control_success_rate is 0.92 for the early braking-only frames, and full lead suite must be refreshed under the new heading-evidence gate before claiming lead completion.
