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
| `so400m/8bit-affine` | C-RADIOv4-SO400M | 8-bit affine, group size 64 | Compact/high-precision |
| `h/8bit-affine` | C-RADIOv4-H | 8-bit affine, group size 64 | Compact/high-precision |
| `so400m/mxfp8` | C-RADIOv4-SO400M | `mxfp8`, group size 32 | Experimental/lower precision |
| `h/mxfp8` | C-RADIOv4-H | `mxfp8`, group size 32 | Experimental/lower precision |

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
- fused MLX fast attention and layernorm kernels in the native forward path
- broad MLX benchmark matrix by model, resolution, batch size, dtype, and quantization mode
- local-checkpoint parity tests that skip when checkpoints are absent
- unit tests for shapes, manifests, preprocessing, CLI behavior, metrics, and MLX helpers

## Performance

Local benchmark environment:

- Apple Silicon, `mlx` default device `Device(gpu, 0)`
- PyTorch backend device `mps`
- speed matrix images copied from
  `/Users/stephansturges/Pictures/WALDO/WALDO_new_data_for_v4/cropped images`
  into ignored `reports/local_waldo_crops/`
- speed matrix: MLX fast kernels, compiled forward only, no output materialization,
  2 warmups, 5 measured repeats
- smoke parity fixture: `data/golden_images/smoke.jpg`

### 512px Batch-1 Speed

| Model | Artifact | p50 latency | Throughput | Notes |
| --- | --- | ---: | ---: | --- |
| C-RADIOv4-SO400M | bf16 | 32.4 ms | 30.9 images/s | best speed path at same output contract |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 47.1 ms | 21.2 images/s | packed low-bit runtime; compact/high-precision |
| C-RADIOv4-SO400M | HF `so400m/mxfp8` | 49.8 ms | 20.1 images/s | packed low-bit runtime; experimental/lower precision |
| C-RADIOv4-H | bf16 | 45.6 ms | 22.0 images/s | best speed path at same output contract |
| C-RADIOv4-H | HF `h/8bit-affine` | 58.8 ms | 17.0 images/s | packed low-bit runtime; compact/high-precision |
| C-RADIOv4-H | HF `h/mxfp8` | 52.6 ms | 19.0 images/s | packed low-bit runtime; experimental/lower precision |

### Best Matrix Throughput

| Model | Artifact | Resolution | Batch | p50 batch latency | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 256 | 4 | 27.2 ms | 146.8 images/s |
| C-RADIOv4-SO400M | HF `so400m/8bit-affine` | 256 | 4 | 34.9 ms | 114.5 images/s |
| C-RADIOv4-SO400M | HF `so400m/mxfp8` | 256 | 4 | 45.3 ms | 88.3 images/s |
| C-RADIOv4-H | bf16 | 256 | 4 | 35.3 ms | 113.4 images/s |
| C-RADIOv4-H | HF `h/8bit-affine` | 256 | 4 | 47.4 ms | 84.3 images/s |
| C-RADIOv4-H | HF `h/mxfp8` | 256 | 4 | 37.9 ms | 105.6 images/s |

MLX `0.31.2` is installed and current. Its newer `mxfp8`/`nvfp4` path is available, and
`quantize_input=True` is documented only for `mxfp8` and `nvfp4` linear layers. For this
ViT workload, weight-only `mxfp8` was smaller but lower precision and slower than bf16;
`nvfp4`/`mxfp4` did not meet precision gates in smoke checks. The useful performance win
came from MLX fast attention/layernorm plus compilation, not from the quantized matmul
formats.

The quantized HF artifacts run packed low-bit weights at runtime. They improve storage size
and runtime weight memory, not end-to-end throughput. C-RADIOv4 is a dense ViT encoder:
attention, layernorm, GELU, residual traffic, image patching, and activation movement remain
in bf16 even when linear weights are packed. At these 512px batch-1 and small-batch shapes,
Apple GPU bf16 dense matmul is already highly optimized, while weight-only quantized matmul
adds per-group scale/bias handling and cannot fuse away the rest of the transformer block.
If the goal is fastest local inference and the bf16 model fits, the quantized bundles are
not the right runtime tier today. Keep them only when compact storage or lower runtime
weight memory matters.

The current 10x-class speed lever is workload-level: compiled bf16 plus lower resolution
and batching gives SO400M about 6.8 ms per image at 256px batch 4, versus the earlier
PyTorch/MPS 512px batch-1 baseline of 120 ms. At the same 512px batch-1 contract, no
honest 10x speedup was found; bf16 remains the fastest measured mode.

### Quantized Precision

Quantized formats versus the bf16 MLX bundle at `512x512`:

| Artifact | Format | Images | Summary cosine mean/min | Spatial cosine mean/min |
| --- | --- | ---: | ---: | ---: |
| HF `so400m/8bit-affine` | 8-bit affine | 12 WALDO crops | 0.999907 / 0.999868 | 0.999930 / 0.999876 |
| HF `h/8bit-affine` | 8-bit affine | 12 WALDO crops | 0.999899 / 0.999878 | 0.999830 / 0.999764 |
| HF `so400m/mxfp8` | `mxfp8` | 12 WALDO crops | 0.989820 / 0.950717 | 0.993502 / 0.977879 |
| HF `h/mxfp8` | `mxfp8` | 12 WALDO crops | 0.990217 / 0.974710 | 0.988696 / 0.976071 |

Smoke-image 8-bit versus bf16 precision at `512x512`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M | 0.999879 | 0.999778 |
| C-RADIOv4-H | 0.999886 | 0.999615 |

Parity against PyTorch/MPS at `512x512`, `bfloat16`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M MLX vs PyTorch/MPS | 0.999939 | 0.999773 |
| C-RADIOv4-H MLX vs PyTorch/MPS | 0.999858 | 0.999494 |

### Bundle Size

| Model | bf16 bundle | 8-bit affine bundle | mxfp8 bundle |
| --- | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 1.6 GB | 517 MB | 479 MB |
| C-RADIOv4-H | 2.4 GB | 758 MB | 702 MB |

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
        "so400m/mxfp8/*",
        "h/mxfp8/*",
    ],
)
PY
```

Example checkpoint paths after download:

- `bundles/hf-c-radiov4-quantized/so400m/8bit-affine`
- `bundles/hf-c-radiov4-quantized/h/8bit-affine`
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
- supported quantized tier: 8-bit affine, with precision gates before use
