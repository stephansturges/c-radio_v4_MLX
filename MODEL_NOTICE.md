# Model Provenance and License Notice

This repository contains implementation code for running NVIDIA C-RADIOv4 checkpoints on
Apple Silicon with MLX. It does not redistribute NVIDIA model weights.

Official model sources:

- `nvidia/C-RADIOv4-SO400M`: https://huggingface.co/nvidia/C-RADIOv4-SO400M
  - pinned revision used for current tests: `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`
- `nvidia/C-RADIOv4-H`: https://huggingface.co/nvidia/C-RADIOv4-H
  - pinned revision used for current tests: `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`

Both Hugging Face model cards identify the model license as
`nvidia-open-model-license`. The governing model terms are the NVIDIA Open Model License
Agreement:

https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf

The repository code is licensed under MIT. That MIT license applies to this implementation
code only; it does not relicense NVIDIA checkpoints, model weights, model configuration,
or converted model bundles.

If you distribute NVIDIA model weights or a converted/derivative MLX model bundle, preserve
the upstream model provenance, include the NVIDIA license agreement as required by the
model terms, and include this attribution notice with model copies:

```text
Licensed by NVIDIA Corporation under the NVIDIA Open Model License
```
