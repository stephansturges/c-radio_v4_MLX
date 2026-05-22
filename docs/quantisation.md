# Quantization

The quantization order is:

1. `bfloat16` reference bundle
2. 8-bit affine production bundle
3. 6-bit affine candidate
4. 5-bit affine candidate
5. 4-bit and `mxfp*` experimental bundles

Every quantized bundle must pass embedding parity gates before it is described as supported.
