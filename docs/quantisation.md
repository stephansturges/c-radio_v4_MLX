# Quantization

The quantization order is:

1. `bfloat16` reference bundle
2. 8-bit affine production bundle
3. Cider W8A8 Apple M5+ runtime bundle
4. 6-bit affine candidate
5. 5-bit affine candidate
6. 4-bit and `mxfp*` experimental bundles

Every quantized bundle must pass embedding parity gates before it is described as supported.

## Implemented Formats

Implemented:

- MLX `mx.quantize` packing for patch, attention, and MLP linear weights
- `affine`, `mxfp8`, `mxfp4`, and `nvfp4` safetensors layouts
- optional Cider W8A8 safetensors layout with int8 weights and online int8 activation
  quantization through Cider's M5+ MLX kernels
- zero-padding for SO400M linear dimensions that are not divisible by the group size
- self-contained quantized bundles with `qweight`, `qscales`, optional `qbiases`, mode
  metadata, and manifest stats
- quantized runtime loading through `mx.quantized_matmul` or Cider W8A8 kernels

The runtime path keeps quantized weights packed. There is no dequantize-at-load production
mode in this repo; if weights are expanded back to bf16, the result is not a low-bit
runtime model.

Build both local 8-bit bundles with:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-so400m-bf16 \
  --out bundles/c-radiov4-so400m-8bit \
  --bits 8 \
  --group-size 64 \
  --mode affine

cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-8bit \
  --bits 8 \
  --group-size 64 \
  --mode affine
```

Build experimental `mxfp8` bundles with:

```sh
cradio-mlx quantize \
  --model bundles/c-radiov4-so400m-bf16 \
  --out bundles/c-radiov4-so400m-mxfp8 \
  --bits 8 \
  --group-size 32 \
  --mode mxfp8

cradio-mlx quantize \
  --model bundles/c-radiov4-h-bf16 \
  --out bundles/c-radiov4-h-mxfp8 \
  --bits 8 \
  --group-size 32 \
  --mode mxfp8
```

Build optional Cider W8A8 runtime bundles with Python `>=3.12`, Apple M5+ hardware,
and https://github.com/Mininglamp-AI/cider installed:

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

Current 512px precision versus bf16 on 12 WALDO crops:

| Model | Format | Summary cosine mean/min | Spatial cosine mean/min |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | 8-bit affine | 0.999907 / 0.999868 | 0.999930 / 0.999876 |
| C-RADIOv4-H | 8-bit affine | 0.999899 / 0.999878 | 0.999830 / 0.999764 |
| C-RADIOv4-SO400M | mxfp8 | 0.989820 / 0.950717 | 0.993502 / 0.977879 |
| C-RADIOv4-H | mxfp8 | 0.990217 / 0.974710 | 0.988696 / 0.976071 |

Smoke-image Cider W8A8 precision versus bf16 at `512x512`:

| Model | Summary cosine | Spatial cosine |
| --- | ---: | ---: |
| C-RADIOv4-SO400M | 0.998164 | 0.998889 |
| C-RADIOv4-H | 0.997202 | 0.996210 |

Cider W4A8 was tested but is not supported. It reduced SO400M/H bundles to 271 MB /
384 MB, but failed smoke precision at `256x256` with SO400M summary/spatial cosine
`0.913606` / `0.979707` and H `0.850901` / `0.817213`.

Current bundle sizes:

| Model | bf16 | 8-bit affine | Cider W8A8 | mxfp8 |
| --- | ---: | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 1.6 GB | 517 MB | 468 MB | 479 MB |
| C-RADIOv4-H | 2.4 GB | 758 MB | 685 MB | 702 MB |

Current packed-runtime `512x512`, batch-1 speed:

| Model | Format | p50 latency | Throughput |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 32.7 ms | 30.6 images/s |
| C-RADIOv4-SO400M | 8-bit affine packed | 49.6 ms | 20.2 images/s |
| C-RADIOv4-SO400M | Cider W8A8 packed | 32.5 ms | 30.8 images/s |
| C-RADIOv4-H | bf16 | 53.7 ms | 18.6 images/s |
| C-RADIOv4-H | 8-bit affine packed | 74.2 ms | 13.5 images/s |
| C-RADIOv4-H | Cider W8A8 packed | 47.1 ms | 21.2 images/s |

`mxfp4` and `nvfp4` smoke checks did not pass precision gates. Keep bf16 as the
performance tier, 8-bit affine as the compact high-precision/low-runtime-weight-memory
tier, Cider W8A8 as the Apple M5+ compact runtime tier, and `mxfp8` as experimental
unless future MLX kernels or calibration change these results.
