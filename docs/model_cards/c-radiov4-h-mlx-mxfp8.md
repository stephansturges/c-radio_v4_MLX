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
  - mxfp8
  - experimental
  - apple-silicon
---

# C-RADIOv4-H MLX mxfp8

This is an experimental self-contained MLX bundle converted from `nvidia/C-RADIOv4-H`
for Apple Silicon image embeddings.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-H
- Upstream revision: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`
- Converted bundle path in the local repo: `bundles/c-radiov4-h-mxfp8`

## Format

- Runtime: MLX
- Quantization: `mxfp8`
- Group size: 32
- Quantized tensors: 129
- Copied tensors: 261
- Bundle size observed locally: 702 MB

## Measured Accuracy

Against the local bf16 MLX bundle at `512x512` on 12 WALDO crop images:

| Metric | Mean | Min |
| --- | ---: | ---: |
| Summary cosine | 0.990272 | 0.974978 |
| Spatial cosine | 0.988784 | 0.976665 |

This is lower precision than the 8-bit affine bundle. Treat this as experimental.

## Measured Speed

Fast-kernel compiled-forward MLX measurements at `512x512`, batch 1:

| Runtime | p50 latency | Throughput |
| --- | ---: | ---: |
| packed | 52.6 ms | 19.0 images/s |
| dequantize at load | 45.4 ms | 22.0 images/s |

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
license.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
