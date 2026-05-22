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

Useful variants:

- `group_size=0`: per-channel W8A8, smallest W8A8 metadata.
- `group_size=128`: balanced W8A8; slightly larger than per-channel, better precision,
  and faster on SO400M in the current matrix.
- `group_size=0 --clip-percentile 99.99`: fastest tested W8A8 variant, with more
  embedding drift. Treat it as experimental and validate downstream.

Current fair benchmark summary on Apple M5 Max:

| Model | Format | 256px b4 | 512px b1 | 512px b4 |
| --- | --- | ---: | ---: | ---: |
| SO400M | bf16 | 27.6 ms / 144.9 img/s | 32.7 ms / 30.6 img/s | 111.3 ms / 35.9 img/s |
| SO400M | Cider W8A8 per-channel | 27.1 ms / 147.8 img/s | 32.5 ms / 30.8 img/s | 119.5 ms / 33.5 img/s |
| SO400M | Cider W8A8 g128 | 26.5 ms / 150.8 img/s | 31.3 ms / 32.0 img/s | 112.8 ms / 35.5 img/s |
| SO400M | Cider W8A8 p99.99 | 24.8 ms / 161.4 img/s | 29.8 ms / 33.5 img/s | 106.4 ms / 37.6 img/s |
| H | bf16 | 42.0 ms / 95.2 img/s | 53.7 ms / 18.6 img/s | 187.0 ms / 21.4 img/s |
| H | Cider W8A8 per-channel | 37.4 ms / 106.9 img/s | 47.1 ms / 21.2 img/s | 179.6 ms / 22.3 img/s |
| H | Cider W8A8 g128 | 35.9 ms / 111.3 img/s | 48.3 ms / 20.7 img/s | 168.7 ms / 23.7 img/s |
| H | Cider W8A8 p99.99 | 33.8 ms / 118.3 img/s | 43.7 ms / 22.9 img/s | 174.8 ms / 22.9 img/s |

Smoke-image `512x512` precision versus local bf16 MLX:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| SO400M Cider W8A8 | 0.998164 | 0.998889 |
| SO400M Cider W8A8 g128 | 0.998630 | 0.998837 |
| SO400M Cider W8A8 p99.99 | 0.998565 | 0.998815 |
| H Cider W8A8 | 0.997202 | 0.996210 |
| H Cider W8A8 g128 | 0.997877 | 0.996008 |
| H Cider W8A8 p99.99 | 0.997006 | 0.995913 |

WALDO 12-image `512x512` precision versus local bf16 MLX:

| Model | Variant | Summary mean/min | Spatial mean/min |
| --- | --- | ---: | ---: |
| SO400M | g128 | 0.998808 / 0.998460 | 0.999269 / 0.998657 |
| SO400M | p99.99 | 0.998638 / 0.998112 | 0.999240 / 0.998586 |
| H | g128 | 0.997935 / 0.997436 | 0.997821 / 0.996704 |
| H | p99.99 | 0.997642 / 0.996978 | 0.997628 / 0.996634 |

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
