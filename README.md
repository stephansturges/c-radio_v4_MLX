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
- bundle manifest read/write and bundle inspection utilities
- manifest-only conversion bootstrap for local checkpoint directories
- unit tests for shapes, manifests, preprocessing, CLI behavior, metrics, and MLX helpers

Still open:

- self-contained MLX bundle writer
- 8-bit affine quantization
- broader resolution and batch benchmark matrix
- automated parity tests that run only when local checkpoints are present

## Performance

Local benchmark environment:

- Apple Silicon, `mlx` default device `Device(gpu, 0)`
- PyTorch backend device `mps`
- image size `512x512`, batch size 1, dtype `bfloat16`
- smoke fixture: `data/golden_images/smoke.jpg`
- 1 warmup, 3 measured repeats

| Model | Backend | Device | p50 latency | p95 latency | Load time | Output shapes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| C-RADIOv4-SO400M | MLX | GPU | 46.5 ms | 46.7 ms | 0.29 s | summary `(1, 2304)`, spatial `(1, 1024, 1152)` |
| C-RADIOv4-SO400M | PyTorch | MPS | 120.0 ms | 120.7 ms | 1.65 s | summary `(1, 2304)`, spatial `(1, 1024, 1152)` |
| C-RADIOv4-H | MLX | GPU | 59.9 ms | 59.9 ms | 0.41 s | summary `(1, 2560)`, spatial `(1, 1024, 1280)` |
| C-RADIOv4-H | PyTorch | MPS | 165.8 ms | 167.3 ms | 2.43 s | summary `(1, 2560)`, spatial `(1, 1024, 1280)` |

Parity against PyTorch/MPS at `512x512`, `bfloat16`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M MLX vs PyTorch/MPS | 0.999939 | 0.999773 |
| C-RADIOv4-H MLX vs PyTorch/MPS | 0.999858 | 0.999494 |

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

Create a manifest-only bundle scaffold from a local Hugging Face checkpoint directory:

```sh
cradio-mlx convert \
  --hf-path checkpoints/c-radiov4-h \
  --mlx-path bundles/c-radiov4-h-bf16 \
  --model-id nvidia/C-RADIOv4-H \
  --revision 0057b339059c0b9e1b4ba996f975410ebbfdfcc8 \
  --dtype bfloat16 \
  --manifest-only
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
- next production target: 8-bit affine quantization
