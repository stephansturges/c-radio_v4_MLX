# Benchmarking

Benchmarks should report:

- cold load time
- warm latency p50 and p95
- images per second
- peak resident memory
- model, dtype, quantization mode, image size, and batch size
- host machine class

The default benchmark image size is `512`, with sweep points at `256`, `384`, `512`,
`768`, `1024`, `1536`, and `2048`.

## Current Matrix Results

The current reproducible matrix is generated with:

```sh
python scripts/benchmark_matrix.py \
  --images reports/local_waldo_crops/*.jpg \
  --image-sizes 256 512 768 \
  --batch-sizes 1 2 4 \
  --warmups 2 \
  --repeats 5 \
  --no-materialize \
  --compile \
  --summary reports/benchmark-matrix-fast-compiled.json
```

Local `512x512`, batch-1, fast-kernel compiled-forward results on Apple Silicon:

| Model | Bundle | p50 latency | Throughput |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 32.4 ms | 30.9 images/s |
| C-RADIOv4-SO400M | 8-bit affine packed | 47.1 ms | 21.2 images/s |
| C-RADIOv4-SO400M | mxfp8 packed | 49.8 ms | 20.1 images/s |
| C-RADIOv4-H | bf16 | 45.6 ms | 22.0 images/s |
| C-RADIOv4-H | 8-bit affine packed | 58.8 ms | 17.0 images/s |
| C-RADIOv4-H | mxfp8 packed | 52.6 ms | 19.0 images/s |

The quantized bundles are actual packed low-bit runtime bundles. They reduce bundle size
and runtime weight memory, but they are slower than bf16 on MLX GPU for this ViT encoder.
Weight-only `mxfp8` is also materially lower precision. Use fast-kernel compiled bf16 for
latency-sensitive local inference when the model fits.

## Remaining Speed Levers Checked

The remaining low-risk kernel substitutions did not produce a useful speedup:

- MLX `conv2d` patch embedding matches the current patch embed numerically but was slightly
  slower than the existing im2patch plus dense matmul path at 512px.
- `mlx.nn.gelu` was effectively neutral versus the local exact GELU expression.
- `mlx.nn.gelu_approx` was slower in this graph and introduced measurable drift.
- `mlx.nn.gelu_fast_approx` was marginally faster in isolation but failed precision
  expectations, with SO400M smoke-output cosine around `0.97` summary / `0.98` spatial.

The next non-trivial fixed-contract latency work would require custom fused Metal kernels
or a model-level approximation decision. The safe production path remains fast-kernel
compiled bf16.
