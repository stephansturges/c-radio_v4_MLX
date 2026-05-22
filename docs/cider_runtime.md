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

## Fused LayerNorm + Linear Hook

The runtime has an experimental fused-op integration point for:

- `norm1 -> attn.qkv`
- `norm2 -> mlp.fc1`

Enable it with:

```sh
cradio-mlx mlx-benchmark \
  --checkpoint bundles/c-radiov4-so400m-cider-w8a8-p9999 \
  --variant so400m \
  --image data/golden_images/smoke.jpg \
  --image-size 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --cider-fusion auto \
  --report reports/experiments/fusion-auto.json
```

Modes:

- `off`: current production path.
- `auto`: use fused Cider ops when installed, otherwise fall back to the current path.
- `required`: fail if the installed Cider package does not provide the fused ops.

Current upstream Cider `0.7.0` does not expose the fused functions. With
`--cider-fusion required`, the runtime fails explicitly with:

```text
Cider fusion is required, but cider.ops.layernorm_perchannel_linear is not available.
```

This repo includes a tested Cider patch at
`cider_patches/0001-layernorm-w8a8-linear.patch`, based on Cider commit
`01b8f9c0e65a54375e50eab9480ca2ff6a1a0d6e`. The patch adds:

- `cider.ops.layernorm_perchannel_linear(...)`

The current p99.99 Cider bundles are per-channel W8A8, so this is enough for the tested
SO400M/H fused path. A future per-group Cider bundle would also need
`cider.ops.layernorm_pergroup_linear(...)`.

Apply the patch in a Cider checkout, then reinstall Cider:

```sh
git apply /path/to/c-radio_v4_MLX/cider_patches/0001-layernorm-w8a8-linear.patch
python -m pip install -e .
```

This hook is intentionally narrow: it only exercises the first native-kernel target without
changing bundle format or removing the unfused fallback.

Smoke precision of fused versus unfused Cider p99.99 at `512x512`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| SO400M | 0.998747 | 0.998902 |
| H | 0.998525 | 0.996893 |

The precision drift comes from doing the fused LayerNorm path inside the Cider FP16 Metal
kernel before W8A8 activation quantization. Keep `--cider-fusion required` experimental
until downstream task metrics accept this drift.

Fused versus unfused Cider p99.99 speed, `data/golden_images/smoke.jpg`, materialized
outputs, 2 warmups, 5 repeats:

| Model | Resolution | Batch | Off p50 | Fused p50 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| SO400M | 256 | 1 | 11.35 ms | 10.93 ms | 1.038x |
| SO400M | 512 | 1 | 32.32 ms | 31.69 ms | 1.020x |
| SO400M | 768 | 1 | 92.53 ms | 91.51 ms | 1.011x |
| SO400M | 512 | 4 | 119.19 ms | 117.85 ms | 1.011x |
| H | 256 | 1 | 15.69 ms | 15.67 ms | 1.001x |
| H | 512 | 1 | 47.10 ms | 46.95 ms | 1.003x |
| H | 768 | 1 | 131.05 ms | 133.85 ms | 0.979x |
| H | 512 | 4 | 168.89 ms | 168.88 ms | 1.000x |

With MLX `mx.compile` on SO400M `512x512`, batch 1 improved from 29.65 ms unfused to
29.17 ms fused, and batch 4 improved from 106.66 ms to 104.70 ms. The patch is therefore
a small, real low-level improvement, not a 10x path: it only removes two LayerNorm launches
per transformer block, while attention, MLP second projections, GELU, residual traffic,
patching, and output materialization still dominate the full encoder.

## Segment Profiling

Use the intrusive segment profiler to decide whether a fused kernel is worth keeping:

```sh
python scripts/profile_runtime_segments.py \
  --checkpoint bundles/c-radiov4-so400m-cider-w8a8-p9999 \
  --variant so400m \
  --images data/golden_images/smoke.jpg \
  --image-size 256 \
  --batch-size 1 \
  --warmups 1 \
  --repeats 1 \
  --dtype bfloat16 \
  --cider-fusion auto \
  --out reports/experiments/profile-segments-smoke-cider-p9999.json
```

This profiler forces `mx.eval` inside the graph, so it is diagnostic rather than a
production latency number. In the SO400M `256x256` smoke run, layernorms plus `attn.qkv`
and `mlp.fc1` linears accounted for roughly 22 ms of the forced-sync profile, which is why
`layernorm -> W8A8 linear` remains the first fused kernel target.
