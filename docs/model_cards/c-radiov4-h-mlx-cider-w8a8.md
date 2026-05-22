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

# C-RADIOv4-H MLX Cider W8A8

This is a self-contained MLX bundle converted from `nvidia/C-RADIOv4-H` for Apple
Silicon image embeddings.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-H
- Upstream revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`
- Converted bundle path in the local repo: `bundles/c-radiov4-h-cider-w8a8`

## Format

- Runtime: MLX plus Cider
- Quantization: W8A8, per-channel int8 weights and online int8 activation quantization
- Required hardware: Apple M5 or newer
- Required package: https://github.com/Mininglamp-AI/cider
- Quantized tensors: 129
- Copied tensors: 261
- Bundle size observed locally: 685 MB

## Measured Accuracy

Against the local bf16 MLX bundle on `data/golden_images/smoke.jpg`:

| Image size | Summary cosine | Spatial cosine |
| ---: | ---: | ---: |
| 256x256 | 0.998008 | 0.997055 |
| 512x512 | 0.997202 | 0.996210 |

## Measured Speed

Apple M5 Max, `mlx==0.31.2`, Cider `0.7.0`, no output materialization:

| Resolution | Batch | p50 latency | Throughput |
| ---: | ---: | ---: | ---: |
| 256x256 | 1 | 15.5 ms | 64.6 images/s |
| 256x256 | 4 | 37.4 ms | 106.9 images/s |
| 512x512 | 1 | 47.1 ms | 21.2 images/s |
| 512x512 | 4 | 179.6 ms | 22.3 images/s |

## Usage

Install Cider in a Python `>=3.12` environment, then run:

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
converted bundle are governed by NVIDIA's Open Model License Agreement, not by the MIT
license. Preserve NVIDIA provenance and license terms when redistributing this bundle.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
