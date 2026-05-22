# Hugging Face Upload Notes

The local model bundles are ignored by git and live under `bundles/`.

Primary compact/high-precision bundles to upload:

- `bundles/c-radiov4-so400m-8bit`
- `bundles/c-radiov4-h-8bit`

Experimental `mxfp8` bundles, only upload if you want to publish the lower-precision
tradeoff explicitly:

- `bundles/c-radiov4-so400m-mxfp8`
- `bundles/c-radiov4-h-mxfp8`

Recommended Hugging Face repository names:

- `stephansturges/c-radiov4-so400m-mlx-8bit-affine`
- `stephansturges/c-radiov4-h-mlx-8bit-affine`
- `stephansturges/c-radiov4-so400m-mlx-mxfp8`
- `stephansturges/c-radiov4-h-mlx-mxfp8`

Each bundle directory contains:

- `model.safetensors`
- `manifest.json`
- copied upstream config/processor metadata
- `README.md` model card

Committed copies of the model cards live in `docs/model_cards/`. The current cards have
also been copied into the local bundle directories above so `huggingface-cli upload` will
include them.

Upload example:

```sh
huggingface-cli upload \
  stephansturges/c-radiov4-so400m-mlx-8bit-affine \
  bundles/c-radiov4-so400m-8bit \
  . \
  --repo-type model
```

The model cards must preserve upstream NVIDIA provenance. The repository code is MIT
licensed, but the model weights and converted bundles remain governed by the NVIDIA Open
Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf

The implementation repository is:

https://github.com/stephansturges/c-radio_v4_MLX
