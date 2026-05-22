# C-RADIOv4 MLX

`cradio-mlx` is a standalone Apple Silicon runtime for NVIDIA C-RADIOv4 image
embeddings using MLX, with PyTorch/MPS retained as the reference backend.

C-RADIOv4 is an embedding model, not an end-to-end chat model. The package exposes
summary embeddings and spatial embeddings for downstream classifiers, retrieval systems,
dense vision tasks, or projector/VLM integrations.

## Model Sources and Licensing

Official checkpoints live on Hugging Face:

- `nvidia/C-RADIOv4-SO400M`: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- `nvidia/C-RADIOv4-H`: https://huggingface.co/nvidia/C-RADIOv4-H

Published quantized MLX bundles live here:

- `StephanST/C-radiov4_quantized`: https://huggingface.co/StephanST/C-radiov4_quantized

Bundle paths in that repository:

| Path | Model | Format | Recommended use |
| --- | --- | --- | --- |
| `so400m/8bit-affine` | C-RADIOv4-SO400M | 8-bit affine, group size 64 | Compact/high-precision, not a throughput tier |
| `h/8bit-affine` | C-RADIOv4-H | 8-bit affine, group size 64 | Compact/high-precision, not a throughput tier |
| `so400m/cider-w8a8` | C-RADIOv4-SO400M | Cider W8A8, per-channel | M5+ compact/sometimes faster |
| `h/cider-w8a8` | C-RADIOv4-H | Cider W8A8, per-channel | M5+ compact/faster |
| `so400m/cider-w8a8-g128` | C-RADIOv4-SO400M | Cider W8A8, group size 128 | M5+ balanced precision/speed |
| `h/cider-w8a8-g128` | C-RADIOv4-H | Cider W8A8, group size 128 | M5+ balanced precision/speed |
| `so400m/cider-w8a8-p9999` | C-RADIOv4-SO400M | Cider W8A8, 99.99 percentile clip | M5+ fastest experimental |
| `h/cider-w8a8-p9999` | C-RADIOv4-H | Cider W8A8, 99.99 percentile clip | M5+ fastest experimental |
| `so400m/mxfp8` | C-RADIOv4-SO400M | `mxfp8`, group size 32 | Experimental/lower precision, not recommended |
| `h/mxfp8` | C-RADIOv4-H | `mxfp8`, group size 32 | Experimental/lower precision, not recommended |

Pinned revisions used for current parity and benchmark runs:

- SO400M: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- H: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`

This repository code is MIT licensed. NVIDIA model weights and any converted model
bundles remain governed by the NVIDIA Open Model License Agreement. See
[MODEL_NOTICE.md](MODEL_NOTICE.md) for provenance and redistribution notes.

## Current Status

Implemented:

- native MLX GPU forward path for `nvidia/C-RADIOv4-SO400M`
- native MLX GPU forward path for `nvidia/C-RADIOv4-H`
- PyTorch/MPS reference runner for parity capture and fallback
- CLI/API for embedding, benchmarking, device inspection, and output comparison
- fp32 and bf16 parity checks against PyTorch/MPS on the smoke image
- safetensors key/shape audit for the HF-to-MLX mapping pass
- self-contained MLX bundle writer for local Hugging Face checkpoint directories
- 8-bit affine and `mxfp8` MLX packed-weight quantization for supported linear layers
- optional Cider W8A8 runtime bundles that use packed int8 weights plus online int8
  activation quantization on Apple M5+ INT8 kernels
- optional SmoothQuant-style calibration scales for Cider experiments without
  dequantizing weights back to dense bf16
- fused MLX fast attention and layernorm kernels in the native forward path
- experimental patched-Cider fusion targets for `layernorm -> W8A8 linear` and
  `GELU -> W8A8 fc2`
- fixed-shape Core ML fast-kill conversion and benchmark tooling for SO400M/H
- broad MLX benchmark matrix by model, resolution, batch size, dtype, and quantization mode
- local-checkpoint parity tests that skip when checkpoints are absent
- unit tests for shapes, manifests, preprocessing, CLI behavior, metrics, and MLX helpers

## Performance

Local benchmark environment:

- Apple M5 Max, `mlx` default device `Device(gpu, 0)`
- `mlx==0.31.2`; Cider W8A8 tests used Cider `0.7.0` from
  https://github.com/Mininglamp-AI/cider
- PyTorch backend device `mps`
- speed matrix images copied from
  `/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images`
  into ignored `reports/local_waldo_crops/`
- current fair speed matrix below: `data/golden_images/smoke.jpg`, no output
  materialization, 6 warmups, 12 measured repeats, one benchmark process at a time.
  The Cider rows use the faster of compiled and non-compiled forward for that cell.
- smoke parity fixture: `data/golden_images/smoke.jpg`

### 512px Batch-1 Speed

| Model | Artifact | p50 latency | Throughput | Notes |
| --- | --- | ---: | ---: | --- |
| C-RADIOv4-SO400M | bf16 | 32.7 ms | 30.6 images/s | dense MLX baseline |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 49.6 ms | 20.2 images/s | packed MLX weight-only runtime; high precision |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8` | 32.5 ms | 30.8 images/s | real W8A8 runtime; smaller, roughly matches bf16 here |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-g128` | 31.3 ms | 32.0 images/s | balanced Cider W8A8 |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-p9999` + `ln+mlp` | 28.5 ms | 35.1 images/s | fastest Cider W8A8; experimental patch |
| C-RADIOv4-SO400M | Core ML fixed-shape | 25.1 ms | 39.8 images/s | proof only; missed 20% gate vs fused Cider |
| C-RADIOv4-H | bf16 | 53.7 ms | 18.6 images/s | strong baseline |
| C-RADIOv4-H | HF `h/8bit-affine` | 74.2 ms | 13.5 images/s | packed MLX weight-only runtime; high precision |
| C-RADIOv4-H | HF `h/cider-w8a8` | 47.1 ms | 21.2 images/s | real W8A8 runtime; modestly faster than bf16 |
| C-RADIOv4-H | HF `h/cider-w8a8-g128` | 48.3 ms | 20.7 images/s | balanced Cider W8A8 |
| C-RADIOv4-H | HF `h/cider-w8a8-p9999` + `ln+mlp` | 42.7 ms | 23.4 images/s | fastest Cider W8A8; experimental patch |
| C-RADIOv4-H | Core ML fixed-shape | 31.4 ms | 31.8 images/s | proof passed speed/precision gate |

### Best Matrix Throughput

| Model | Artifact | Resolution | Batch | p50 batch latency | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 256 | 4 | 27.6 ms | 144.9 images/s |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 256 | 4 | 35.0 ms | 114.3 images/s |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8` | 256 | 4 | 27.1 ms | 147.8 images/s |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-p9999` + `ln+mlp` | 256 | 4 | 23.5 ms | 170.1 images/s |
| C-RADIOv4-H | bf16 | 256 | 4 | 42.0 ms | 95.2 images/s |
| C-RADIOv4-H | HF `h/8bit-affine` | 256 | 4 | 59.0 ms | 67.8 images/s |
| C-RADIOv4-H | HF `h/cider-w8a8` | 256 | 4 | 37.4 ms | 106.9 images/s |
| C-RADIOv4-H | HF `h/cider-w8a8-p9999` + `ln+mlp` | 256 | 4 | 32.7 ms | 122.3 images/s |

MLX `0.31.2` is installed and current. Its newer `mxfp8`/`nvfp4` path is available, and
`quantize_input=True` is documented only for `mxfp8` and `nvfp4` linear layers. For this
ViT workload, weight-only `mxfp8` was smaller but lower precision and slower than bf16;
`nvfp4`/`mxfp4` did not meet precision gates in smoke checks. The useful performance win
came from MLX fast attention/layernorm plus compilation, not from the quantized matmul
formats.

There is no supported dequantize-at-load model path. If a bundle expands low-bit weights
back to dense bf16 before inference, treat it as storage compression only and do not use it
as a runtime quantized model. The published affine and `mxfp8` artifacts are different:
they keep weights packed and call `mx.quantized_matmul`, but they are still weight-only
paths. Activations, attention, layernorm, GELU, residual traffic, and image patching remain
bf16. On this ViT encoder, those weight-only formats reduce storage and runtime weight
memory but do not improve throughput.

Cider W8A8 is different: it quantizes activations online and runs int8 weight by int8
activation kernels on Apple M5+ INT8 TensorOps. That is the first local path here that is
both genuinely weight/activation low-bit at runtime and sometimes faster. The gains are
still modest rather than 10x because C-RADIOv4 is not an LLM decode workload: it has large
token matrices, attention and normalization remain outside the int8 kernels, and the custom
linear kernels do not fuse whole transformer blocks. Local 2-bit/4-bit Qwen models feel
more dramatic because decoder LLM serving is often memory-bandwidth-bound, repeatedly
streams the same large linear weights, and uses inference stacks built specifically around
low-bit decode. This repo follows the same rule: use low-bit models only where the
Apple/MLX kernel layer actually consumes low-bit weights during inference.

Experimental Cider patches are included under `cider_patches/`. They add native
`layernorm -> W8A8 linear` and `GELU -> W8A8 fc2` Metal primitives for the p99.99 Cider
bundles and are enabled with `--cider-fusion auto|required --cider-fusion-targets ln+mlp`.
The MLP-stage patch passed the fast-kill gate: SO400M p99.99 `512x512` batch 1 moved from
32.54 ms unfused to 28.47 ms with `ln+mlp`; H moved from 47.24 ms to 42.68 ms. Smoke
fused-vs-unfused cosine at 512px stayed above 0.9987/0.9990 for SO400M summary/spatial and
0.9986/0.9971 for H. This is the best low-level win found so far, but it is still not a
10x path because attention, the GEMMs themselves, residual traffic, patching, and output
materialization still dominate the full encoder.

Core ML fixed-shape conversion was also tested. SO400M Core ML `ALL` reached 25.12 ms at
512px batch 1 with excellent cosine but missed the planned 20% speed gate versus the fused
Cider baseline. H Core ML `ALL` reached 31.42 ms and passed the gate. `ALL` and
`CPU_AND_GPU` were effectively identical, so this is not evidence of an ANE-specific win.

W4A8 was also tested. It cut bundle and active weight memory further, but it was slower
end-to-end and failed precision gates, so it is not a supported artifact.

The larger-batch check did not reveal hidden throughput wins for Cider. At 512px batch
8/16, bf16 remained best for SO400M, while H Cider variants were only roughly comparable.
For this implementation, batch 1-4 is the useful latency/throughput zone.

SmoothQuant-style Cider calibration was implemented and tested with alpha `0.5` on the 12
WALDO crop calibration set. It preserves the runtime low-bit contract: weights remain
packed and Cider still consumes low-bit weights/activations at matmul time. It improves
precision, but it is not a speed tier because it adds an activation rescale before every
Cider linear.

| Model | Variant | Summary cosine mean/min | Spatial cosine mean/min | 512px batch-1 p50 |
| --- | --- | ---: | ---: | ---: |
| SO400M | Cider g128 + SmoothQuant | 0.999373 / 0.998303 | 0.999501 / 0.998993 | 31.9 ms |
| SO400M | Cider p99.99 + SmoothQuant | 0.999302 / 0.998743 | 0.999278 / 0.998793 | 30.1 ms |
| H | Cider g128 + SmoothQuant | 0.999352 / 0.999218 | 0.998830 / 0.998373 | 48.8 ms |
| H | Cider p99.99 + SmoothQuant | 0.999159 / 0.999066 | 0.998521 / 0.998092 | 49.1 ms |

Compared with the non-SmoothQuant Cider rows, this is a precision recovery option, not a
throughput improvement. It is therefore not published as a default HF model variant.

### Quantized Precision

Quantized formats versus the bf16 MLX bundle at `512x512`:

| Artifact | Format | Images | Summary cosine mean/min | Spatial cosine mean/min |
| --- | --- | ---: | ---: | ---: |
| HF `so400m/8bit-affine` | 8-bit affine | 12 WALDO crops | 0.999907 / 0.999868 | 0.999930 / 0.999876 |
| HF `h/8bit-affine` | 8-bit affine | 12 WALDO crops | 0.999899 / 0.999878 | 0.999830 / 0.999764 |
| HF `so400m/mxfp8` | `mxfp8` | 12 WALDO crops | 0.989820 / 0.950717 | 0.993502 / 0.977879 |
| HF `h/mxfp8` | `mxfp8` | 12 WALDO crops | 0.990217 / 0.974710 | 0.988696 / 0.976071 |

Smoke-image 8-bit versus bf16 precision at `512x512`:

| Model | Format | Summary cosine | Spatial cosine |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | 8-bit affine | 0.999879 | 0.999778 |
| C-RADIOv4-SO400M | Cider W8A8 | 0.998164 | 0.998889 |
| C-RADIOv4-SO400M | Cider W8A8 g128 | 0.998630 | 0.998837 |
| C-RADIOv4-SO400M | Cider W8A8 p99.99 | 0.998565 | 0.998815 |
| C-RADIOv4-H | 8-bit affine | 0.999886 | 0.999615 |
| C-RADIOv4-H | Cider W8A8 | 0.997202 | 0.996210 |
| C-RADIOv4-H | Cider W8A8 g128 | 0.997877 | 0.996008 |
| C-RADIOv4-H | Cider W8A8 p99.99 | 0.997006 | 0.995913 |

Rejected W4A8 smoke precision at `256x256`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M Cider W4A8 | 0.913606 | 0.979707 |
| C-RADIOv4-H Cider W4A8 | 0.850901 | 0.817213 |

Parity against PyTorch/MPS at `512x512`, `bfloat16`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M MLX vs PyTorch/MPS | 0.999939 | 0.999773 |
| C-RADIOv4-H MLX vs PyTorch/MPS | 0.999858 | 0.999494 |

### Bundle Size

| Model | bf16 bundle | 8-bit affine | Cider W8A8 | Rejected Cider W4A8 |
| --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 1.6 GB | 517 MB | 468 MB | 271 MB |
| C-RADIOv4-H | 2.4 GB | 758 MB | 685 MB | 384 MB |

Cider W8A8 g128 bundle sizes are 480 MB for SO400M and 702 MB for H. The p99.99
clipped bundles are the same size as per-channel W8A8.

Approximate MLX active memory immediately after loading weights:

| Model | bf16 | 8-bit affine | Cider W8A8 | Rejected Cider W4A8 |
| --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 863 MB | 507 MB | 453 MB | 247 MB |
| C-RADIOv4-H | 1.30 GB | 754 MB | 676 MB | 361 MB |

## Low-Level Acceleration Work

This was the follow-on pass after the first quantized artifacts landed. The goal was to
test two concrete bets instead of continuing to chase generic "quantization should be
faster" assumptions:

1. Fuse the Cider W8A8 MLP path so `GELU(mlp.fc1)` feeds `mlp.fc2` without round-tripping
   through a separate MLX GELU op before activation quantization.
2. Run a Core ML fixed-shape proof to see whether Apple's model compiler could beat the
   best MLX/Cider path enough to justify a separate backend.

Both bets were implemented, benchmarked, and gated. The short version is:

- The fastest low-bit runtime path is now Cider p99.99 with
  `--cider-fusion required --cider-fusion-targets ln+mlp`.
- The MLP fusion is a real low-level win, but it is a 1.08x to 1.19x win, not a 10x win.
- Core ML is precise and fast for fixed `512x512` batch-1 inputs. It is worth keeping as a
  proof for C-RADIOv4-H, but it is not yet a production backend in this repo.
- There is no evidence from these runs that Core ML `ALL` is using ANE in a meaningfully
  different way than `CPU_AND_GPU`; the timings were effectively identical.

### What Changed

The MLX runtime now exposes a `cider_fusion_targets` setting through the Python API and
CLI. Valid values are:

| Target | Effect |
| --- | --- |
| `ln` | Fuse `norm1 -> attn.qkv` and `norm2 -> mlp.fc1`; this is the default target. |
| `mlp` | Fuse only `GELU(mlp.fc1) -> mlp.fc2`. |
| `ln+mlp` | Use both low-level fused paths. |

The Cider integration has three modes:

| Mode | Behavior |
| --- | --- |
| `off` | Do not call patched Cider fusion ops. |
| `auto` | Use patched ops if installed; silently fall back otherwise. |
| `required` | Fail fast if the installed Cider package does not provide the requested fused op. |

The shipped Cider patch set is:

| Patch | Adds | Used by |
| --- | --- | --- |
| `cider_patches/0001-layernorm-w8a8-linear.patch` | `cider.ops.layernorm_perchannel_linear(...)` | `ln` and `ln+mlp` |
| `cider_patches/0002-gelu-w8a8-linear.patch` | `cider.ops.gelu_perchannel_linear(...)` | `mlp` and `ln+mlp` |

The fused hooks are intentionally limited to per-channel Cider W8A8 bundles. They reject
SmoothQuant scales and per-group W8A8 in `required` mode because the patched Cider ops
implemented here only cover the p99.99 per-channel path that was benchmarked.

### Install Patched Cider

Use a separate Python 3.12 environment for Cider because it builds a native MLX extension:

```sh
python3.12 -m venv .venv-cider
source .venv-cider/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,cider]"
```

If testing the experimental fused path, apply both patches to a Cider checkout based on
commit `01b8f9c0e65a54375e50eab9480ca2ff6a1a0d6e`, then reinstall Cider:

```sh
git clone https://github.com/Mininglamp-AI/cider /tmp/cider-src
git -C /tmp/cider-src checkout 01b8f9c0e65a54375e50eab9480ca2ff6a1a0d6e
git -C /tmp/cider-src apply --3way \
  /path/to/c-radio_v4_MLX/cider_patches/0001-layernorm-w8a8-linear.patch
git -C /tmp/cider-src apply \
  /path/to/c-radio_v4_MLX/cider_patches/0002-gelu-w8a8-linear.patch
python -m pip install -e /tmp/cider-src
```

Run with `--cider-fusion required` when benchmarking. That prevents accidentally measuring
the unfused fallback because the wrong Cider package was imported.

### Reproduce Cider Fusion Benchmarks

SO400M p99.99 Cider W8A8 with both fusion targets:

```sh
python scripts/benchmark_matrix.py \
  --variant so400m \
  --checkpoint bundles/hf-c-radiov4-quantized/so400m/cider-w8a8-p9999 \
  --images data/golden_images/smoke.jpg \
  --image-sizes 256 512 \
  --batch-sizes 1 4 \
  --warmups 3 \
  --repeats 8 \
  --dtype bfloat16 \
  --cider-fusion required \
  --cider-fusion-targets ln+mlp \
  --summary reports/experiments/mlp-fusion-so400m-ln-mlp.json
```

C-RADIOv4-H p99.99 Cider W8A8 with both fusion targets:

```sh
python scripts/benchmark_matrix.py \
  --variant h \
  --checkpoint bundles/hf-c-radiov4-quantized/h/cider-w8a8-p9999 \
  --images data/golden_images/smoke.jpg \
  --image-sizes 256 512 \
  --batch-sizes 1 4 \
  --warmups 3 \
  --repeats 8 \
  --dtype bfloat16 \
  --cider-fusion required \
  --cider-fusion-targets ln+mlp \
  --summary reports/experiments/mlp-fusion-h-ln-mlp.json
```

The current fused-versus-unfused results, using `data/golden_images/smoke.jpg` and
materialized outputs, are:

| Model | Resolution | Batch | Off p50 | `mlp` p50 | `ln+mlp` p50 | Best speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SO400M | 256 | 1 | 11.33 ms | 10.60 ms | 10.31 ms | 1.099x |
| SO400M | 256 | 4 | 27.48 ms | 24.09 ms | 23.51 ms | 1.169x |
| SO400M | 512 | 1 | 32.54 ms | 28.99 ms | 28.47 ms | 1.143x |
| SO400M | 512 | 4 | 119.66 ms | 102.97 ms | 100.79 ms | 1.187x |
| H | 256 | 1 | 15.64 ms | 14.76 ms | 14.53 ms | 1.077x |
| H | 256 | 4 | 37.42 ms | 32.76 ms | 32.71 ms | 1.144x |
| H | 512 | 1 | 47.24 ms | 42.84 ms | 42.68 ms | 1.107x |
| H | 512 | 4 | 169.33 ms | 144.61 ms | 143.67 ms | 1.179x |

Fused p99.99 Cider precision against unfused p99.99 Cider at `512x512`:

| Model | Target | Summary cosine | Spatial cosine |
| --- | --- | ---: | ---: |
| SO400M | `mlp` | 0.998928 | 0.999143 |
| SO400M | `ln+mlp` | 0.998735 | 0.999090 |
| H | `mlp` | 0.998615 | 0.997137 |
| H | `ln+mlp` | 0.998612 | 0.997095 |

The precision drift is expected: LayerNorm and GELU move into Cider FP16 Metal kernels
immediately before W8A8 activation quantization. The drift is small enough for an
experimental speed tier, but downstream application metrics should still decide whether
`ln+mlp` is acceptable for production.

### Reproduce Core ML Fast-Kill

Install the optional Core ML dependency:

```sh
python -m pip install -e ".[reference,coreml]"
```

The script first attempts TorchScript tracing. The RADIO patch generator shape logic
currently makes tracing fail with:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

The fallback path uses `torch.export(...).run_decompositions({})`, which converted both
SO400M and H to Core ML ML Program packages.

SO400M:

```sh
python scripts/coreml_fastkill.py \
  --checkpoint checkpoints/c-radiov4-so400m \
  --model-id nvidia/C-RADIOv4-SO400M \
  --variant so400m \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --batch-size 1 \
  --baseline-p50-ms 28.47 \
  --compute-units ALL CPU_AND_GPU \
  --warmups 3 \
  --repeats 8 \
  --package reports/experiments/so400m-512-b1.mlpackage \
  --out reports/experiments/coreml-so400m-fastkill.json
```

C-RADIOv4-H:

```sh
python scripts/coreml_fastkill.py \
  --checkpoint checkpoints/c-radiov4-h \
  --model-id nvidia/C-RADIOv4-H \
  --variant h \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --batch-size 1 \
  --baseline-p50-ms 42.68 \
  --compute-units ALL CPU_AND_GPU \
  --warmups 3 \
  --repeats 8 \
  --package reports/experiments/h-512-b1.mlpackage \
  --out reports/experiments/coreml-h-fastkill.json
```

Current Core ML results:

| Model | Compute unit | p50 | p95 | Summary cosine | Spatial cosine | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SO400M | `ALL` | 25.12 ms | 25.25 ms | 0.999998 | 0.999995 | kill |
| SO400M | `CPU_AND_GPU` | 25.01 ms | 25.26 ms | 0.999998 | 0.999995 | kill |
| H | `ALL` | 31.42 ms | 31.83 ms | 0.999998 | 0.999991 | continue |
| H | `CPU_AND_GPU` | 31.73 ms | 31.90 ms | 0.999998 | 0.999991 | continue |

The gate was "continue only if Core ML is at least 20% faster than the fused Cider
baseline while preserving cosine." SO400M was precise and absolutely faster, but it did
not clear the 20% threshold versus `28.47 ms`. H cleared the threshold versus `42.68 ms`.
Because `ALL` and `CPU_AND_GPU` were effectively the same, the result should be described
as a Core ML fixed-shape GPU/compiler win, not as an ANE win.

### Final Performance Decision

The repo now has four distinct model tiers:

| Tier | Use when | Do not expect |
| --- | --- | --- |
| bf16 MLX | You want the strongest stable baseline and simplest deployment. | Smaller bundles. |
| 8-bit affine MLX | You want compact, very high-cosine bundles using packed weights. | Higher throughput on this ViT. |
| Cider W8A8 p99.99 `ln+mlp` | You are on Apple M5+, accept experimental patched Cider, and want the fastest low-bit MLX path found so far. | 10x speedups or LLM-style low-bit gains. |
| Core ML fixed-shape | You can tolerate a separate fixed-shape proof path, especially for C-RADIOv4-H. | A documented ANE-specific speedup or dynamic production backend here. |

The main reason quantization does not produce "super-duper" throughput here is that this
is a dense ViT encoder. The model runs large full-sequence matrix operations once per
image; it is not repeatedly decoding one token at a time from enormous weight matrices.
Weight-only quantization reduces bundle and active weight memory, but attention, layernorm,
GELU, residual traffic, patch projection, activation movement, output materialization, and
MLX scheduling remain in the hot path. Cider W8A8 helps because it is a true runtime
low-bit path, but it only covers selected linears, not the full transformer block.

## Bootstrap

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,reference]"
```

For a lighter dev setup that avoids PyTorch, install:

```sh
python -m pip install -e ".[dev]"
```

Cider W8A8 runtime bundles require Apple M5+ hardware and Python `>=3.12`, because Cider
builds a native MLX extension:

```sh
python3.12 -m venv .venv-cider
source .venv-cider/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,cider]"
```

For long-running local services, use compiled MLX forward after warmup:

```sh
cradio-mlx embed \
  --backend mlx-so400m \
  --checkpoint bundles/c-radiov4-so400m-bf16 \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --compile \
  --save-npz embedding.npz
```

The first call pays MLX compilation overhead for that input shape; repeated calls at the
same resolution are the intended use.

For Cider fusion experiments, use `--cider-fusion auto` to opt into a Cider fork that
provides fused low-level ops while keeping the current path as fallback. Use
`--cider-fusion-targets ln`, `mlp`, or `ln+mlp` to isolate targets, and
`--cider-fusion required` in benchmark jobs when absence of native fused ops should fail
fast.

## Download Checkpoints

The model weights are not committed. Download them from Hugging Face at pinned revisions:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/C-RADIOv4-SO400M",
    revision="c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
    local_dir="checkpoints/c-radiov4-so400m",
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "README.md",
        "*.py",
    ],
)

snapshot_download(
    repo_id="nvidia/C-RADIOv4-H",
    revision="0057b339059c0b9e1b4ba996f975410ebbfdfcc8",
    local_dir="checkpoints/c-radiov4-h",
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "README.md",
        "*.py",
    ],
)
PY
```

Download the published quantized MLX bundles:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="StephanST/C-radiov4_quantized",
    local_dir="bundles/hf-c-radiov4-quantized",
    allow_patterns=[
        "README.md",
        "so400m/8bit-affine/*",
        "h/8bit-affine/*",
        "so400m/cider-w8a8/*",
        "h/cider-w8a8/*",
        "so400m/cider-w8a8-g128/*",
        "h/cider-w8a8-g128/*",
        "so400m/cider-w8a8-p9999/*",
        "h/cider-w8a8-p9999/*",
        "so400m/mxfp8/*",
        "h/mxfp8/*",
    ],
)
PY
```

Example checkpoint paths after download:

- `bundles/hf-c-radiov4-quantized/so400m/8bit-affine`
- `bundles/hf-c-radiov4-quantized/h/8bit-affine`
- `bundles/hf-c-radiov4-quantized/so400m/cider-w8a8`
- `bundles/hf-c-radiov4-quantized/h/cider-w8a8`
- `bundles/hf-c-radiov4-quantized/so400m/cider-w8a8-g128`
- `bundles/hf-c-radiov4-quantized/h/cider-w8a8-g128`
- `bundles/hf-c-radiov4-quantized/so400m/cider-w8a8-p9999`
- `bundles/hf-c-radiov4-quantized/h/cider-w8a8-p9999`
- `bundles/hf-c-radiov4-quantized/so400m/mxfp8`
- `bundles/hf-c-radiov4-quantized/h/mxfp8`

## CLI

Inspect expected spatial shape:

```sh
cradio-mlx spatial-shape --image-size 512
```

Run the native MLX SO400M path:

```sh
cradio-mlx embed \
  --backend mlx-so400m \
  --checkpoint checkpoints/c-radiov4-so400m \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz reports/so400m-smoke-512-mlx-bf16.npz
```

Run the native MLX H path:

```sh
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint checkpoints/c-radiov4-h \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz reports/h-smoke-512-mlx-bf16.npz
```

Run the PyTorch/MPS reference path:

```sh
cradio-mlx embed \
  --model-id nvidia/C-RADIOv4-H \
  --revision 0057b339059c0b9e1b4ba996f975410ebbfdfcc8 \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz reports/h-smoke-512-pytorch-bf16.npz
```

Benchmark native MLX:

```sh
cradio-mlx mlx-benchmark \
  --variant h \
  --checkpoint checkpoints/c-radiov4-h \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --report reports/h-smoke-512-mlx-bf16-benchmark.json
```

Benchmark PyTorch/MPS:

```sh
cradio-mlx pytorch-benchmark \
  --model-id nvidia/C-RADIOv4-H \
  --revision 0057b339059c0b9e1b4ba996f975410ebbfdfcc8 \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --report reports/h-smoke-512-pytorch-bf16-benchmark.json
```

Check local acceleration backends:

```sh
cradio-mlx device-info
```

Create a self-contained bundle from a local Hugging Face checkpoint directory:

```sh
cradio-mlx convert \
  --hf-path checkpoints/c-radiov4-h \
  --mlx-path bundles/c-radiov4-h-bf16 \
  --model-id nvidia/C-RADIOv4-H \
  --revision 0057b339059c0b9e1b4ba996f975410ebbfdfcc8 \
  --dtype bfloat16
```

Quantize a bundle with 8-bit affine MLX packing:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-8bit \
  --bits 8 \
  --group-size 64 \
  --mode affine
```

Build an experimental `mxfp8` bundle:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-mxfp8 \
  --bits 8 \
  --group-size 32 \
  --mode mxfp8
```

Build optional Cider W8A8 runtime bundles on Apple M5+ with Python `>=3.12`:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-cider-w8a8 \
  --bits 8 \
  --group-size 0 \
  --mode cider-w8a8
```

Build the balanced g128 or fastest p99.99 Cider variants:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-cider-w8a8-g128 \
  --bits 8 \
  --group-size 128 \
  --mode cider-w8a8

cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-cider-w8a8-p9999 \
  --bits 8 \
  --group-size 0 \
  --mode cider-w8a8 \
  --clip-percentile 99.99
```

Hugging Face upload notes and model-card README templates are in
[docs/huggingface_upload.md](docs/huggingface_upload.md).

Uploaded quantized bundles are published at:

https://huggingface.co/StephanST/C-radiov4_quantized

Run the fast-kernel compiled benchmark matrix:

```sh
python scripts/benchmark_matrix.py \
  --images reports/local_waldo_crops/*.jpg \
  --image-sizes 256 512 768 \
  --batch-sizes 1 2 4 \
  --warmups 2 \
  --repeats 5 \
  --no-materialize \
  --compile \
  --summary reports/benchmark-matrix-fast-compiled.json
```

Add `--include-cider` to include optional Cider W8A8 cases in the matrix.

Run local parity tests. These skip automatically if the local checkpoints are absent:

```sh
python -m pytest tests/test_local_parity.py -q
```

Compare embedding outputs:

```sh
cradio-mlx compare \
  --reference reports/h-smoke-512-bf16-pytorch.npz \
  --candidate reports/h-smoke-512-bf16-mlx.npz \
  --out reports/h-smoke-512-bf16-pytorch-vs-mlx.json
```

Audit local checkpoint tensor keys:

```sh
cradio-mlx audit-weights \
  --hf-path checkpoints/c-radiov4-h \
  --out reports/h-weight-audit.json
```

## Python API

```python
from cradio_mlx import MLXHEncoder, MLXSO400MEncoder

so400m = MLXSO400MEncoder.load("checkpoints/c-radiov4-so400m", dtype="bfloat16")
so400m_result = so400m.encode_image("data/golden_images/smoke.jpg", image_size=512)

h = MLXHEncoder.load("checkpoints/c-radiov4-h", dtype="bfloat16")
h_result = h.encode_image("data/golden_images/smoke.jpg", image_size=512)
```

## Development Contract

The package stays embedding-first:

- summary output: `(B, C)`
- spatial output: `(B, T, D)`
- spatial grid reshape: `(B, D, H, W)`
- supported image sizes: 256 to 2048, multiples of 16
- first production tier: `bfloat16`
- supported quantized tiers: 8-bit affine for high precision, Cider W8A8 for Apple M5+
  compact runtime experiments, both with precision gates before use
