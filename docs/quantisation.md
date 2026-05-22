# Quantization

The quantization order is:

1. `bfloat16` reference bundle
2. 8-bit affine production bundle
3. 6-bit affine candidate
4. 5-bit affine candidate
5. 4-bit and `mxfp*` experimental bundles

Every quantized bundle must pass embedding parity gates before it is described as supported.

## Implemented Formats

Implemented:

- MLX `mx.quantize` packing for patch, attention, and MLP linear weights
- `affine`, `mxfp8`, `mxfp4`, and `nvfp4` safetensors layouts
- zero-padding for SO400M linear dimensions that are not divisible by the group size
- self-contained quantized bundles with `qweight`, `qscales`, optional `qbiases`, mode
  metadata, and manifest stats
- packed quantized runtime loading through `mx.quantized_matmul`
- dequantize-at-load runtime loading that expands compact bundles to bf16 dense weights and
  then uses the dense MLX kernels

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

## Runtime Modes

The quantized bundles support two runtime modes:

- `--quantized-runtime packed`: keep weights packed and run `mx.quantized_matmul`. This is
  the low-runtime-weight-memory path, but it is slower than bf16 for these transformer
  layers on the measured Apple Silicon GPU.
- `--quantized-runtime dequantize`: load the compact bundle, dequantize each packed linear
  weight once to bf16, then run the dense MLX path. This recovers bf16-class throughput but
  uses bf16 runtime weight memory.

At `512x512`, batch 1, compiled forward:

| Model | Format | Runtime | p50 latency | Throughput |
| --- | --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | dense | 32.4 ms | 30.9 images/s |
| C-RADIOv4-SO400M | 8-bit affine | packed | 47.1 ms | 21.2 images/s |
| C-RADIOv4-SO400M | 8-bit affine | dequantize at load | 32.4 ms | 30.9 images/s |
| C-RADIOv4-H | bf16 | dense | 45.6 ms | 21.9 images/s |
| C-RADIOv4-H | 8-bit affine | packed | 58.8 ms | 17.0 images/s |
| C-RADIOv4-H | 8-bit affine | dequantize at load | 45.5 ms | 22.0 images/s |

Current 512px dequantized-runtime precision versus bf16 on 12 WALDO crops:

| Model | Format | Summary cosine mean/min | Spatial cosine mean/min |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | 8-bit affine | 0.999913 / 0.999885 | 0.999927 / 0.999881 |
| C-RADIOv4-H | 8-bit affine | 0.999907 / 0.999884 | 0.999828 / 0.999761 |
| C-RADIOv4-SO400M | mxfp8 | 0.989676 / 0.949449 | 0.993379 / 0.978096 |
| C-RADIOv4-H | mxfp8 | 0.990272 / 0.974978 | 0.988784 / 0.976665 |

Current bundle sizes:

| Model | bf16 | 8-bit affine | mxfp8 |
| --- | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 1.6 GB | 517 MB | 479 MB |
| C-RADIOv4-H | 2.4 GB | 758 MB | 702 MB |

`mxfp4` and `nvfp4` smoke checks did not pass precision gates. Keep bf16 or dequantized
8-bit affine as the performance tier, packed 8-bit affine as the compact runtime-memory
tier, and `mxfp8` as experimental unless future MLX kernels or calibration change these
results.
