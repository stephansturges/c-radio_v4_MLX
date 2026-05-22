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
  - apple-silicon
---

# C-RADIOv4-H MLX 8-bit Affine

This is a self-contained MLX bundle converted from `nvidia/C-RADIOv4-H` for Apple
Silicon image embeddings.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-H
- Upstream revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`
- Converted bundle path in the local repo: `bundles/c-radiov4-h-8bit`

## Format

- Runtime: MLX
- Quantization: 8-bit affine
- Group size: 64
- Quantized tensors: 129
- Copied tensors: 261
- Padded tensors: 0
- Bundle size observed locally: 758 MB

## Measured Accuracy

Against the local bf16 MLX bundle at `512x512` on 12 WALDO crop images:

| Metric | Mean | Min |
| --- | ---: | ---: |
| Summary cosine | 0.999907 | 0.999884 |
| Spatial cosine | 0.999828 | 0.999761 |

Smoke-image 8-bit versus bf16 at `512x512`:

| Metric | Cosine |
| --- | ---: |
| Summary | 0.999886 |
| Spatial | 0.999615 |

## Measured Speed

Fast-kernel compiled-forward MLX measurements at `512x512`, batch 1:

| Runtime | p50 latency | Throughput |
| --- | ---: | ---: |
| packed | 58.8 ms | 17.0 images/s |
| dequantize at load | 45.5 ms | 22.0 images/s |

`packed` keeps weights low-bit at runtime but is slower for this ViT encoder. Use
`--quantized-runtime dequantize` when latency matters; it expands weights to bf16 at load.

## Usage

```sh
cradio-mlx embed \
  --backend mlx-h \
  --checkpoint /path/to/this/bundle \
  --image image.jpg \
  --image-size 512 \
  --dtype bfloat16 \
  --quantized-runtime dequantize \
  --save-npz embedding.npz
```

## License

The implementation code in `c-radio_v4_MLX` is MIT licensed. The model weights and this
converted bundle are governed by NVIDIA's Open Model License Agreement, not by the MIT
license. Preserve NVIDIA provenance and license terms when redistributing this bundle.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
