# Hugging Face Upload Notes

The uploaded Hugging Face repository is:

https://huggingface.co/StephanST/C-radiov4_quantized

The local model bundles are ignored by git and live under `bundles/`.

Primary compact/high-precision bundles to upload:

- `bundles/c-radiov4-so400m-8bit`
- `bundles/c-radiov4-h-8bit`

Apple M5+ runtime low-bit bundles to upload:

- `bundles/c-radiov4-so400m-cider-w8a8`
- `bundles/c-radiov4-h-cider-w8a8`

Experimental `mxfp8` bundles, only upload if you want to publish the lower-precision
tradeoff explicitly:

- `bundles/c-radiov4-so400m-mxfp8`
- `bundles/c-radiov4-h-mxfp8`

Remote repository structure:

- `so400m/8bit-affine`
- `h/8bit-affine`
- `so400m/cider-w8a8`
- `h/cider-w8a8`
- `so400m/mxfp8`
- `h/mxfp8`

Each bundle directory contains:

- `model.safetensors`
- `manifest.json`
- copied upstream config/processor metadata
- `README.md` model card

Committed copies of the root model card and per-bundle model cards live in:

- `docs/hf_repo_readme.md`
- `docs/model_cards/`

The current per-bundle cards have also been copied into the local bundle directories above
so `hf upload` will include them.

Upload example:

```sh
hf upload StephanST/C-radiov4_quantized docs/hf_repo_readme.md README.md --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-so400m-8bit so400m/8bit-affine --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-h-8bit h/8bit-affine --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-so400m-cider-w8a8 so400m/cider-w8a8 --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-h-cider-w8a8 h/cider-w8a8 --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-so400m-mxfp8 so400m/mxfp8 --type model
hf upload StephanST/C-radiov4_quantized bundles/c-radiov4-h-mxfp8 h/mxfp8 --type model
```

Uploaded commits:

- Root model card: https://huggingface.co/StephanST/C-radiov4_quantized/commit/069d8fb7a9f39d1d9a6ba53e2fff184884a7d650
- `so400m/8bit-affine`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/d7f8c24dcef67a9953b1078c2ce4ad552d6214b7
- `h/8bit-affine`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/0a6fca512bd75c9973984e633fbc7dd43b32061e
- `so400m/mxfp8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/2d7d1174c7e09c64fc77d25df40bd010a41df5e8
- `h/mxfp8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/8bf68bec75ecb24ef1ef9e12457d228ff3b20944

Uploaded packed-runtime documentation refresh:

- Root model card: https://huggingface.co/StephanST/C-radiov4_quantized/commit/ce1c84b371c783c1497e357e8a462d62c208bec2
- `so400m/8bit-affine`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/2508229e8e60e7b346f010be208abd971fc8e02b
- `h/8bit-affine`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/029a4fb3ea51495a685939ea75da31a0b5b254e6
- `so400m/mxfp8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/87a03f5353acf256eea8058efda472cfb3350405
- `h/mxfp8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/ae3a673d621eb7cb98a3b6e3f3b0a0f07ecccc6e

Uploaded Cider W8A8 runtime bundles:

- Root model card: https://huggingface.co/StephanST/C-radiov4_quantized/commit/3a7fe7204a0d7a3558d0796d124449b67e46a89c
- `so400m/cider-w8a8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/039a7d2ed412819c4eed47600623237c9d1d7117
- `h/cider-w8a8`: https://huggingface.co/StephanST/C-radiov4_quantized/commit/67437d9b33369a532cf080376ab0733827ea931f

The model cards must preserve upstream NVIDIA provenance. The repository code is MIT
licensed, but the model weights and converted bundles remain governed by the NVIDIA Open
Model License Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf

The implementation repository is:

https://github.com/stephansturges/c-radio_v4_MLX
