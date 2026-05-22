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
| `so400m/8bit-affine` | `nvidia/C-RADIOv4-SO400M` | 8-bit affine, group size 64 | Compact/high-precision, not a throughput tier |
| `h/8bit-affine` | `nvidia/C-RADIOv4-H` | 8-bit affine, group size 64 | Compact/high-precision, not a throughput tier |
| `so400m/cider-w8a8` | `nvidia/C-RADIOv4-SO400M` | Cider W8A8, per-channel | M5+ compact/runtime low-bit |
| `h/cider-w8a8` | `nvidia/C-RADIOv4-H` | Cider W8A8, per-channel | M5+ compact/runtime low-bit |
| `so400m/cider-w8a8-g128` | `nvidia/C-RADIOv4-SO400M` | Cider W8A8, group size 128 | M5+ balanced precision/speed |
| `h/cider-w8a8-g128` | `nvidia/C-RADIOv4-H` | Cider W8A8, group size 128 | M5+ balanced precision/speed |
| `so400m/cider-w8a8-p9999` | `nvidia/C-RADIOv4-SO400M` | Cider W8A8, 99.99 percentile clip | M5+ fastest experimental |
| `h/cider-w8a8-p9999` | `nvidia/C-RADIOv4-H` | Cider W8A8, 99.99 percentile clip | M5+ fastest experimental |
| `so400m/mxfp8` | `nvidia/C-RADIOv4-SO400M` | `mxfp8`, group size 32 | Experimental/lower precision, not recommended |
| `h/mxfp8` | `nvidia/C-RADIOv4-H` | `mxfp8`, group size 32 | Experimental/lower precision, not recommended |

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
| `so400m/8bit-affine` | 0.999907 / 0.999868 | 0.999930 / 0.999876 |
| `h/8bit-affine` | 0.999899 / 0.999878 | 0.999830 / 0.999764 |
| `so400m/mxfp8` | 0.989820 / 0.950717 | 0.993502 / 0.977879 |
| `h/mxfp8` | 0.990217 / 0.974710 | 0.988696 / 0.976071 |

The 8-bit affine bundles are the recommended compact/high-precision artifacts, not the
throughput tier. They are packed weight-only MLX artifacts and are not dequantized back to
dense bf16 at load time. Cider W8A8 is the real weight/activation low-bit runtime path for
Apple M5+ machines and trades a little more embedding drift for lower memory and modest
speedups in some cells. The `mxfp8` bundles are included for experimentation and are lower
precision in these checks.

Smoke-image Cider W8A8 precision versus local bf16 MLX at `512x512`:

| Bundle | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| `so400m/cider-w8a8` | 0.998164 | 0.998889 |
| `h/cider-w8a8` | 0.997202 | 0.996210 |

WALDO 12-image Cider W8A8 precision versus local bf16 MLX at `512x512`:

| Bundle | Summary cosine mean/min | Spatial cosine mean/min |
| --- | ---: | ---: |
| `so400m/cider-w8a8-g128` | 0.998808 / 0.998460 | 0.999269 / 0.998657 |
| `h/cider-w8a8-g128` | 0.997935 / 0.997436 | 0.997821 / 0.996704 |
| `so400m/cider-w8a8-p9999` | 0.998638 / 0.998112 | 0.999240 / 0.998586 |
| `h/cider-w8a8-p9999` | 0.997642 / 0.996978 | 0.997628 / 0.996634 |

## Speed Summary

MLX measurements on Apple M5 Max at `512x512`, batch 1:

| Bundle | p50 latency | Throughput |
| --- | ---: | ---: |
| `so400m/8bit-affine` | 49.6 ms | 20.2 images/s |
| `h/8bit-affine` | 74.2 ms | 13.5 images/s |
| `so400m/cider-w8a8` | 32.5 ms | 30.8 images/s |
| `h/cider-w8a8` | 47.1 ms | 21.2 images/s |
| `so400m/cider-w8a8-g128` | 31.3 ms | 32.0 images/s |
| `h/cider-w8a8-g128` | 48.3 ms | 20.7 images/s |
| `so400m/cider-w8a8-p9999` | 29.8 ms | 33.5 images/s |
| `h/cider-w8a8-p9999` | 43.7 ms | 22.9 images/s |
| `so400m/mxfp8` | 49.8 ms | 20.1 images/s |
| `h/mxfp8` | 52.6 ms | 19.0 images/s |

There are no supported dequantize-at-load artifacts in this repository. The MLX affine and
`mxfp8` bundles keep weights packed but are weight-only, so they prioritize compact storage
and lower runtime weight memory over throughput. Cider W8A8 is the faster
weight/activation low-bit runtime path found so far, but it requires Apple M5+ hardware
and the optional Cider package.

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
  --save-npz embedding.npz
```

The g128 variants are the balanced Cider choice. The p99.99 variants are faster and
slightly lower precision; validate them against downstream task metrics before replacing
bf16 or g128.

Use `--backend mlx-h` for the H model:

```sh
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint h/8bit-affine \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz embedding.npz
```

Cider W8A8 bundles require Python `>=3.12`, Apple M5+ hardware, and Cider:

```sh
python -m pip install "cider @ git+https://github.com/Mininglamp-AI/cider.git"
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint h/cider-w8a8 \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz embedding.npz
```

## License

The implementation code in `c-radio_v4_MLX` is MIT licensed. The model weights and these
converted bundles are governed by NVIDIA's Open Model License Agreement, not by the MIT
license. Preserve NVIDIA provenance and license terms when redistributing these bundles.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
