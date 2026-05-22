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
  - apple-silicon
---

# C-RADIOv4-SO400M MLX 8-bit Affine

This is a self-contained MLX bundle converted from `nvidia/C-RADIOv4-SO400M` for
Apple Silicon image embeddings.

Implementation repository:

https://github.com/stephansturges/c-radio_v4_MLX

## Source

- Upstream model: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- Upstream revision: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- Converted bundle path in the local repo: `bundles/c-radiov4-so400m-8bit`

## Format

- Runtime: MLX
- Quantization: 8-bit affine
- Group size: 64
- Quantized tensors: 109
- Copied tensors: 221
- Padded tensors: 27
- Bundle size observed locally: 517 MB

## Measured Accuracy

Against the local bf16 MLX bundle at `512x512` on 12 WALDO crop images:

| Metric | Mean | Min |
| --- | ---: | ---: |
| Summary cosine | 0.999907 | 0.999868 |
| Spatial cosine | 0.999930 | 0.999876 |

Smoke-image 8-bit versus bf16 at `512x512`:

| Metric | Cosine |
| --- | ---: |
| Summary | 0.999879 |
| Spatial | 0.999778 |

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
converted bundle are governed by NVIDIA's Open Model License Agreement, not by the MIT
license. Preserve NVIDIA provenance and license terms when redistributing this bundle.

NVIDIA Open Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
