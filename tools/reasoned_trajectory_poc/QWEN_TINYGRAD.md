# Qwen2.5-VL Tinygrad Port Smoke

This is an additive proof path for running Qwen2.5-VL components without the
CUDA/Transformers worker. It targets C3X native Linux plus the AMD eGPU dock via
upstream tinygrad `DEV=USB+AMD`.

## What Exists

`qwen2_5_vl_tinygrad.py` ports the Qwen2.5-VL visual tower structure to
tinygrad:

- patch embedding
- visual rotary embedding
- windowed/full visual attention
- visual MLP blocks
- patch merger to 2048-dimensional LLM image-token embeddings
- visual-only safetensors export for C3X smoke tests
- FP16 export for avoiding slow BF16 execution/conversion on the C3X path
- cached grid tensors and optional TinyJit hot-forward execution

It does not yet implement the full Qwen2.5-VL language decoder with multimodal
RoPE and image-token insertion. The current CUDA worker is still the only full
Qwen2.5-VL RTP generator.

## Local Export

From the openpilot repo root on the PC:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen2_5_vl_tinygrad.py `
  --export-visual-subset artifacts\reasoned_trajectory_poc\qwen2_5_vl_visual_32block_fp16.safetensors `
  --max-vision-blocks -1 `
  --export-weight-dtype float16
```

For a cheaper patch-embed plus merger-only smoke:

```powershell
py -3.11 tools\reasoned_trajectory_poc\qwen2_5_vl_tinygrad.py `
  --export-visual-subset artifacts\reasoned_trajectory_poc\qwen2_5_vl_visual_1block_fp16.safetensors `
  --max-vision-blocks 1 `
  --export-weight-dtype float16
```

## C3X Smoke

Copy the script, `config.json`, and exported visual subset to the C3X:

```powershell
ssh comma@192.168.1.95 "rm -rf /data/qwen2_5_vl_tinygrad_smoke && mkdir -p /data/qwen2_5_vl_tinygrad_smoke"
scp tools\reasoned_trajectory_poc\qwen2_5_vl_tinygrad.py comma@192.168.1.95:/data/qwen2_5_vl_tinygrad_smoke/qwen2_5_vl_tinygrad.py
scp models\vlm\qwen2_5_vl_3b_instruct\config.json comma@192.168.1.95:/data/qwen2_5_vl_tinygrad_smoke/config.json
scp artifacts\reasoned_trajectory_poc\qwen2_5_vl_visual_32block_fp16.safetensors comma@192.168.1.95:/data/qwen2_5_vl_tinygrad_smoke/visual_32block_fp16.safetensors
```

Run against upstream tinygrad on the C3X:

```bash
cd /data/qwen2_5_vl_tinygrad_smoke
env QWEN_TINYGRAD_REPO=/data/tinygrad_usb_test \
  PYTHONPATH=/data/tinygrad_usb_test \
  DEV=USB+AMD \
  python3 qwen2_5_vl_tinygrad.py \
    --model-dir /data/qwen2_5_vl_tinygrad_smoke \
    --visual-safetensors /data/qwen2_5_vl_tinygrad_smoke/visual_32block_fp16.safetensors \
    --grid-thw 1,12,12 \
    --max-vision-blocks -1 \
    --device USB+AMD \
    --repeat-forwards 8 \
    --use-jit \
    --input-mode ramp \
    --jit-batch-size 1024 \
    --hard-exit
```

## Measured C3X Results

On C3X native Linux with the AMD USB eGPU dock, using exported FP16 weights,
cached grid tensors, nonzero ramp input, and TinyJit:

```text
1 visual block, 56px-equivalent grid:
loaded_tensors: 18
loaded_bytes: 115,832,176
output_shape: [4, 2048]
forward_ms_all: [11,811, 4,212, 15.9, 16.3, 16.9, 16.7, 18.0, 17.1]

4 visual blocks, 56px-equivalent grid:
loaded_tensors: 54
loaded_bytes: 234,045,376
output_shape: [4, 2048]
forward_ms_all: [17,365, 9,011, 25.7, 24.7, 29.4, 31.7, 30.0, 29.9]

32 visual blocks, 56px-equivalent grid, pre-sync timing harness:
loaded_tensors: 390
loaded_bytes: 1,337,368,576
output_shape: [4, 2048]
forward_ms_all: [62,632, 51,845, 52.3, 50.9, 51.2, 51.2]

32 visual blocks, 168px-equivalent grid, synchronized timing harness, default
tinygrad JIT graph split:
loaded_tensors: 390
loaded_bytes: 1,337,368,576
output_shape: [36, 2048]
load_ms: 258,620
forward_ms_all: [140,260, 91,596, 129.3, 136.2, 135.4, 130.5, 126.8, 133.9]

32 visual blocks, 168px-equivalent grid, synchronized timing harness,
JIT_BATCH_SIZE=1024:
loaded_tensors: 390
loaded_bytes: 1,337,368,576
output_shape: [36, 2048]
load_ms: 257,925
forward_ms_all: [131,667, 86,738, 120.8, 117.0, 116.9, 117.9, 118.5, 117.9]

32 visual blocks, 280px-equivalent grid, synchronized timing harness:
loaded_tensors: 390
loaded_bytes: 1,337,368,576
output_shape: [100, 2048]
load_ms: 255,019
forward_ms_all: [231,471, 154,431, 277.6, 272.8, 271.0, 271.4, 276.5, 272.3]

32 visual blocks, 384px-equivalent grid, pre-sync timing harness:
loaded_tensors: 390
loaded_bytes: 1,337,368,576
output_shape: [196, 2048]
load_ms: 240,069
forward_ms_all: [383,087, 241,203, 250.6, 562.7, 556.6]

32 visual blocks, shape traps, synchronized timing harness:
grid 1,14,14: output_shape [49, 2048], hot forward 528-533 ms
grid 1,18,18: output_shape [81, 2048], hot forward 356-371 ms
```

This proves the C3X dock can load and execute real Qwen2.5-VL visual weights
through tinygrad `USB+AMD`. It also shows that fresh-process measurements are
dominated by USB/device load and first-forward compile/capture costs. The fixed
hot visual path is orders of magnitude faster than the original naive smoke.

The current eGPU speed-matched default is `grid_thw=1,12,12`. With explicit
device synchronization, that runs the full 32-block Qwen2.5-VL visual tower in
about 117-121 ms hot on the C3X AMD USB eGPU path after setting
`JIT_BATCH_SIZE=1024`. That beats the previously measured PC-side Qwen label
probe of about 212-215 ms for the measured vision side of the pipeline.

The 280px-equivalent `1,20,20` grid is a higher-resolution option, but its hot
visual pass is about 271-278 ms on the eGPU. Odd-looking intermediate shapes are
not automatically faster; `1,14,14` and `1,18,18` compile into much worse kernels
on this tinygrad AMD path.

## Root Cause Found

The first real hot-path bottleneck was tinygrad graph splitting, not raw GPU
math. At `grid_thw=1,12,12`, a 4-block proxy originally captured as two JIT
batches:

```text
batched 32: about 3.3-3.8 ms
batched 52: about 11.9-12.2 ms
wall forward: about 46-48 ms
```

After increasing `JIT_BATCH_SIZE` to 1024, the same 4-block graph captured as one
batch:

```text
batched 84: about 15.2-15.8 ms
wall forward: about 29-32 ms
```

So the fix removes avoidable USB/tinygrad graph-launch overhead. It does not
change the model math.

The remaining blocker is not the visual tower anymore. The current tinygrad path
still needs the Qwen2.5-VL language decoder, multimodal RoPE, and image-token
insertion before it can replace the CUDA/Transformers RTP worker end to end.
