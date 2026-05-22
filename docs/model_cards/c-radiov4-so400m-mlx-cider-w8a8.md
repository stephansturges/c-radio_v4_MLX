---
license: other
license_name: nvidia-open-model-license
base_model: nvidia/C-RADIOv4-SO400M
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

# C-RADIOv4-SO400M MLX Cider W8A8

This is a self-contained MLX bundle converted from `nvidia/C-RADIOv4-SO400M` for
Apple Silicon image embeddings.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- Upstream revision: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- Converted bundle path in the local repo: `bundles/c-radiov4-so400m-cider-w8a8`

## Format

- Runtime: MLX plus Cider
- Quantization: W8A8, per-channel int8 weights and online int8 activation quantization
- Required hardware: Apple M5 or newer
- Required package: https://github.com/Mininglamp-AI/cider
- Quantized tensors: 109
- Copied tensors: 221
- Bundle size observed locally: 468 MB

## Measured Accuracy

Against the local bf16 MLX bundle on `data/golden_images/smoke.jpg`:

| Image size | Summary cosine | Spatial cosine |
| ---: | ---: | ---: |
| 256x256 | 0.998193 | 0.999393 |
| 512x512 | 0.998164 | 0.998889 |

## Measured Speed

Apple M5 Max, `mlx==0.31.2`, Cider `0.7.0`, no output materialization:

| Resolution | Batch | p50 latency | Throughput |
| ---: | ---: | ---: | ---: |
| 256x256 | 1 | 10.9 ms | 91.6 images/s |
| 256x256 | 4 | 27.1 ms | 147.8 images/s |
| 512x512 | 1 | 32.5 ms | 30.8 images/s |
| 512x512 | 4 | 119.5 ms | 33.5 images/s |

## Usage

Install Cider in a Python `>=3.12` environment, then run:

```sh
cradio-mlx embed \
  --backend mlx-so400m \
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
