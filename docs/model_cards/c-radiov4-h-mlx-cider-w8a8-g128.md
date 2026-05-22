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

# C-RADIOv4-H MLX Cider W8A8 g128

Balanced Apple M5+ W8A8 runtime bundle for `nvidia/C-RADIOv4-H`.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-H
- Upstream revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`
- Local bundle path: `bundles/c-radiov4-h-cider-w8a8-g128`

## Format

- Runtime: MLX plus Cider
- Quantization: W8A8, int8 activations, int8 weights, group size 128
- Required hardware: Apple M5 or newer
- Required package: https://github.com/Mininglamp-AI/cider
- Bundle size observed locally: 702 MB

## Measured Accuracy

Against local bf16 MLX at `512x512`:

| Data | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| Smoke image | 0.997877 | 0.996008 |
| 12 WALDO crops mean/min | 0.997935 / 0.997436 | 0.997821 / 0.996704 |

## Measured Speed

Apple M5 Max, `mlx==0.31.2`, Cider `0.7.0`, compiled forward, no output materialization:

| Resolution | Batch | p50 latency | Throughput |
| ---: | ---: | ---: | ---: |
| 256x256 | 1 | 14.8 ms | 67.8 images/s |
| 256x256 | 4 | 35.9 ms | 111.3 images/s |
| 512x512 | 1 | 48.3 ms | 20.7 images/s |
| 512x512 | 4 | 168.7 ms | 23.7 images/s |

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
