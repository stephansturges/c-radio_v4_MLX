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

# C-RADIOv4-SO400M MLX Cider W8A8 g128

Balanced Apple M5+ W8A8 runtime bundle for `nvidia/C-RADIOv4-SO400M`.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- Upstream revision: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- Local bundle path: `bundles/c-radiov4-so400m-cider-w8a8-g128`

## Format

- Runtime: MLX plus Cider
- Quantization: W8A8, int8 activations, int8 weights, group size 128
- Required hardware: Apple M5 or newer
- Required package: https://github.com/Mininglamp-AI/cider
- Bundle size observed locally: 480 MB

## Measured Accuracy

Against local bf16 MLX at `512x512`:

| Data | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| Smoke image | 0.998630 | 0.998837 |
| 12 WALDO crops mean/min | 0.998808 / 0.998460 | 0.999269 / 0.998657 |

## Measured Speed

Apple M5 Max, `mlx==0.31.2`, Cider `0.7.0`, compiled forward, no output materialization:

| Resolution | Batch | p50 latency | Throughput |
| ---: | ---: | ---: | ---: |
| 256x256 | 1 | 10.2 ms | 98.4 images/s |
| 256x256 | 4 | 26.5 ms | 150.8 images/s |
| 512x512 | 1 | 31.3 ms | 32.0 images/s |
| 512x512 | 4 | 112.8 ms | 35.5 images/s |

## Usage

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
converted bundle are governed by NVIDIA's Open Model License Agreement.
