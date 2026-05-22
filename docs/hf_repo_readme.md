---
license: other
license_name: nvidia-open-model-license
library_name: mlx
pipeline_tag: image-feature-extraction
base_model:
  - nvidia/C-RADIOv4-SO400M
  - nvidia/C-RADIOv4-H
tags:
  - mlx
  - c-radio
  - vision
  - embeddings
  - quantized
  - apple-silicon
---

# C-RADIOv4 Quantized MLX Bundles

This repository contains self-contained MLX bundles for NVIDIA C-RADIOv4 image embedding
models on Apple Silicon.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Repository Structure

| Path | Model | Format | Status |
| --- | --- | --- | --- |
| `so400m/8bit-affine` | `nvidia/C-RADIOv4-SO400M` | 8-bit affine, group size 64 | Compact/high-precision |
| `h/8bit-affine` | `nvidia/C-RADIOv4-H` | 8-bit affine, group size 64 | Compact/high-precision |
| `so400m/mxfp8` | `nvidia/C-RADIOv4-SO400M` | `mxfp8`, group size 32 | Experimental/lower precision |
| `h/mxfp8` | `nvidia/C-RADIOv4-H` | `mxfp8`, group size 32 | Experimental/lower precision |

Each subdirectory contains:

- `model.safetensors`
- `manifest.json`
- upstream config and preprocessor metadata
- a subdirectory `README.md` with model-specific provenance, measurements, and usage

## Source Models

- `nvidia/C-RADIOv4-SO400M`: https://huggingface.co/nvidia/C-RADIOv4-SO400M
  - Revision: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- `nvidia/C-RADIOv4-H`: https://huggingface.co/nvidia/C-RADIOv4-H
  - Revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`

## Accuracy Summary

Measured against local bf16 MLX bundles at `512x512` on 12 WALDO crop images.

| Bundle | Summary cosine mean/min | Spatial cosine mean/min |
| --- | ---: | ---: |
| `so400m/8bit-affine` | 0.999913 / 0.999885 | 0.999927 / 0.999881 |
| `h/8bit-affine` | 0.999907 / 0.999884 | 0.999828 / 0.999761 |
| `so400m/mxfp8` | 0.989676 / 0.949449 | 0.993379 / 0.978096 |
| `h/mxfp8` | 0.990272 / 0.974978 | 0.988784 / 0.976665 |

The 8-bit affine bundles are the recommended compact/high-precision artifacts. The
`mxfp8` bundles are included for experimentation and are lower precision in these checks.

## Speed Summary

Fast-kernel compiled-forward MLX measurements on Apple Silicon at `512x512`, batch 1:

| Bundle | Runtime | p50 latency | Throughput |
| --- | --- | ---: | ---: |
| `so400m/8bit-affine` | packed | 47.1 ms | 21.2 images/s |
| `so400m/8bit-affine` | dequantize at load | 32.4 ms | 30.9 images/s |
| `h/8bit-affine` | packed | 58.8 ms | 17.0 images/s |
| `h/8bit-affine` | dequantize at load | 45.5 ms | 22.0 images/s |
| `so400m/mxfp8` | packed | 49.8 ms | 20.1 images/s |
| `so400m/mxfp8` | dequantize at load | 32.5 ms | 30.8 images/s |
| `h/mxfp8` | packed | 52.6 ms | 19.0 images/s |
| `h/mxfp8` | dequantize at load | 45.4 ms | 22.0 images/s |

`packed` keeps weights low-bit during inference and reduces runtime weight memory, but it
is slower than dense bf16 on this ViT encoder. `dequantize at load` expands the compact
artifact to bf16 weights once during load, then uses the dense MLX kernels; it recovers
bf16-class throughput while using bf16 runtime weight memory.

## Usage

Install the implementation package from the GitHub repository, then point `--checkpoint` at
one of the downloaded subdirectories:

```sh
cradio-mlx embed \
  --backend mlx-so400m \
  --checkpoint so400m/8bit-affine \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --quantized-runtime dequantize \
  --save-npz embedding.npz
```

Use `--backend mlx-h` for the H model:

```sh
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint h/8bit-affine \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --quantized-runtime dequantize \
  --save-npz embedding.npz
```

## License

The implementation code in `c-radio_v4_MLX` is MIT licensed. The model weights and these
converted bundles are governed by NVIDIA's Open Model License Agreement, not by the MIT
license. Preserve NVIDIA provenance and license terms when redistributing these bundles.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
