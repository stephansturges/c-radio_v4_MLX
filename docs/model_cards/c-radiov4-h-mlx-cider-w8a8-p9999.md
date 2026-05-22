---
license: other
license_name: nvidia-open-model-license
base_model: nvidia/C-RADIOv4-H
library_name: mlx
pipeline_tag: image-feature-extraction
tags:
  - mlx
  - c-radio
  - vision
  - embeddings
  - quantized
  - cider
  - apple-silicon
---

# C-RADIOv4-H MLX Cider W8A8 p99.99

Fast experimental Apple M5+ W8A8 runtime bundle for `nvidia/C-RADIOv4-H`.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-H
- Upstream revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`
- Local bundle path: `bundles/c-radiov4-h-cider-w8a8-p9999`

## Format

- Runtime: MLX plus Cider
- Quantization: W8A8, per-channel, 99.99 percentile weight clipping
- Required hardware: Apple M5 or newer
- Required package: https://github.com/Mininglamp-AI/cider
- Bundle size observed locally: 685 MB

## Measured Accuracy

Against local bf16 MLX at `512x512`:

| Data | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| Smoke image | 0.997006 | 0.995913 |
| 12 WALDO crops mean/min | 0.997642 / 0.996978 | 0.997628 / 0.996634 |

## Measured Speed

Apple M5 Max, `mlx==0.31.2`, Cider `0.7.0`, compiled forward, no output materialization:

| Resolution | Batch | p50 latency | Throughput |
| ---: | ---: | ---: | ---: |
| 256x256 | 1 | 14.5 ms | 69.1 images/s |
| 256x256 | 4 | 33.8 ms | 118.3 images/s |
| 512x512 | 1 | 43.7 ms | 22.9 images/s |
| 512x512 | 4 | 174.8 ms | 22.9 images/s |

This is the fastest tested H Cider variant, but it trades away more embedding precision
than g128. Validate downstream metrics before using it as a default.

## Usage

```sh
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint /path/to/this/bundle \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --save-npz embedding.npz
```

## License

The implementation code in `c-radio_v4_MLX` is MIT licensed. The model weights and this
converted bundle are governed by NVIDIA's Open Model License Agreement.
