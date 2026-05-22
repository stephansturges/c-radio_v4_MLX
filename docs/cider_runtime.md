# Cider W8A8 Runtime

Cider W8A8 is the only quantized path in this repo that currently runs both low-bit
weights and low-bit activations at runtime. It stores per-channel int8 weights and calls
Cider's Apple M5+ MLX custom kernels, which quantize activations online to int8 and run
INT8 TensorOps.

Source:

- https://github.com/Mininglamp-AI/cider

Runtime requirements:

- Apple M5 or newer
- Python `>=3.12`
- `mlx>=0.31`
- Cider installed from source or GitHub

Install example:

```sh
python3.12 -m venv .venv-cider
source .venv-cider/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,cider]"
```

Build bundles:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-so400m-bf16 \
  --out bundles/c-radiov4-so400m-cider-w8a8 \
  --bits 8 \
  --group-size 0 \
  --mode cider-w8a8

cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-cider-w8a8 \
  --bits 8 \
  --group-size 0 \
  --mode cider-w8a8
```

Current fair benchmark summary on Apple M5 Max:

| Model | Format | 256px b4 | 512px b1 | 512px b4 |
| --- | --- | ---: | ---: | ---: |
| SO400M | bf16 | 27.6 ms / 144.9 img/s | 32.7 ms / 30.6 img/s | 111.3 ms / 35.9 img/s |
| SO400M | Cider W8A8 | 27.1 ms / 147.8 img/s | 32.5 ms / 30.8 img/s | 119.5 ms / 33.5 img/s |
| H | bf16 | 42.0 ms / 95.2 img/s | 53.7 ms / 18.6 img/s | 187.0 ms / 21.4 img/s |
| H | Cider W8A8 | 37.4 ms / 106.9 img/s | 47.1 ms / 21.2 img/s | 179.6 ms / 22.3 img/s |

Smoke-image `512x512` precision versus local bf16 MLX:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| SO400M Cider W8A8 | 0.998164 | 0.998889 |
| H Cider W8A8 | 0.997202 | 0.996210 |

Approximate active MLX memory immediately after loading weights:

| Model | bf16 | 8-bit affine | Cider W8A8 |
| --- | ---: | ---: | ---: |
| SO400M | 863 MB | 507 MB | 453 MB |
| H | 1.30 GB | 754 MB | 676 MB |

Why it is not a 10x speedup:

- C-RADIOv4 is a dense ViT encoder, not a decoder LLM.
- Attention, layernorm, GELU, residual additions, token reshapes, and image patching remain
  outside the W8A8 kernels.
- Cider accelerates linear layers, but those kernels are not fused into complete
  transformer blocks.
- MLX bf16 dense matmul on Apple GPU is already very strong for these matrix sizes.

Cider W4A8 was tested and rejected. It reduced bundle size further, but it was slower
end-to-end and failed precision gates on the smoke image.
