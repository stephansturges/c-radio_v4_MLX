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
- quantized runtime loading through `mx.quantized_matmul`

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

Current 512px precision versus bf16 on 12 WALDO crops:

| Model | Format | Summary cosine mean/min | Spatial cosine mean/min |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | 8-bit affine | 0.999907 / 0.999868 | 0.999930 / 0.999876 |
| C-RADIOv4-H | 8-bit affine | 0.999899 / 0.999878 | 0.999830 / 0.999764 |
| C-RADIOv4-SO400M | mxfp8 | 0.989820 / 0.950717 | 0.993502 / 0.977879 |
| C-RADIOv4-H | mxfp8 | 0.990217 / 0.974710 | 0.988696 / 0.976071 |

Current bundle sizes:

| Model | bf16 | 8-bit affine | mxfp8 |
| --- | ---: | ---: | ---: |
| C-RADIOv4-SO400M | 1.6 GB | 517 MB | 479 MB |
| C-RADIOv4-H | 2.4 GB | 758 MB | 702 MB |

`mxfp4` and `nvfp4` smoke checks did not pass precision gates. Keep bf16 as the
performance tier, 8-bit affine as the compact high-precision tier, and `mxfp8` as
experimental unless future MLX kernels or calibration change these results.
