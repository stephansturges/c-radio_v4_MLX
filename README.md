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
| C-RADIOv4-SO400M | bf16 | 32.7 ms | 30.6 images/s | fastest 512px SO400M tier measured |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 49.6 ms | 20.2 images/s | packed MLX weight-only runtime; high precision |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8` | 32.5 ms | 30.8 images/s | real W8A8 runtime; smaller, roughly matches bf16 here |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-g128` | 31.3 ms | 32.0 images/s | balanced Cider W8A8 |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-p9999` | 29.8 ms | 33.5 images/s | fastest Cider W8A8; more drift |
| C-RADIOv4-H | bf16 | 53.7 ms | 18.6 images/s | strong baseline |
| C-RADIOv4-H | HF `h/8bit-affine` | 74.2 ms | 13.5 images/s | packed MLX weight-only runtime; high precision |
| C-RADIOv4-H | HF `h/cider-w8a8` | 47.1 ms | 21.2 images/s | real W8A8 runtime; modestly faster than bf16 |
| C-RADIOv4-H | HF `h/cider-w8a8-g128` | 48.3 ms | 20.7 images/s | balanced Cider W8A8 |
| C-RADIOv4-H | HF `h/cider-w8a8-p9999` | 43.7 ms | 22.9 images/s | fastest Cider W8A8; more drift |

### Best Matrix Throughput

| Model | Artifact | Resolution | Batch | p50 batch latency | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 256 | 4 | 27.6 ms | 144.9 images/s |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 256 | 4 | 35.0 ms | 114.3 images/s |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8` | 256 | 4 | 27.1 ms | 147.8 images/s |
| C-RADIOv4-SO400M | HF `so400m/cider-w8a8-p9999` | 256 | 4 | 24.8 ms | 161.4 images/s |
| C-RADIOv4-H | bf16 | 256 | 4 | 42.0 ms | 95.2 images/s |
| C-RADIOv4-H | HF `h/8bit-affine` | 256 | 4 | 59.0 ms | 67.8 images/s |
| C-RADIOv4-H | HF `h/cider-w8a8` | 256 | 4 | 37.4 ms | 106.9 images/s |
| C-RADIOv4-H | HF `h/cider-w8a8-p9999` | 256 | 4 | 33.8 ms | 118.3 images/s |

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
