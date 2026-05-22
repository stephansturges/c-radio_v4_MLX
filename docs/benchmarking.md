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

Local `512x512`, batch-1 results on Apple M5 Max. Measurements used
`data/golden_images/smoke.jpg`, no output materialization, 6 warmups, 12 measured repeats,
and one benchmark process at a time. The bf16 and 8-bit affine rows use compiled forward;
Cider W8A8 is reported from the faster of compiled/non-compiled runs for that cell.

| Model | Bundle | p50 latency | Throughput |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | bf16 | 32.7 ms | 30.6 images/s |
| C-RADIOv4-SO400M | 8-bit affine packed | 49.6 ms | 20.2 images/s |
| C-RADIOv4-SO400M | Cider W8A8 packed | 32.5 ms | 30.8 images/s |
| C-RADIOv4-H | bf16 | 53.7 ms | 18.6 images/s |
| C-RADIOv4-H | 8-bit affine packed | 74.2 ms | 13.5 images/s |
| C-RADIOv4-H | Cider W8A8 packed | 47.1 ms | 21.2 images/s |

The MLX affine and `mxfp8` bundles are actual packed low-bit runtime bundles, but they are
weight-only. They reduce bundle size and runtime weight memory while leaving activations
and the rest of the transformer block in bf16, so they are slower than bf16 on this ViT
encoder. Cider W8A8 is the useful low-bit runtime path found so far because it uses int8
weights and online int8 activation quantization on M5+ INT8 kernels. The gain is modest,
not 10x, because attention, norms, GELU, residual traffic, and custom-kernel dispatch still
remain outside a fused low-bit transformer block.

Rejected Cider W4A8 measurements: SO400M `512x512` batch 1 was 42.0 ms and H was 61.2 ms,
and both failed precision gates. Do not publish W4A8 as supported without calibration or
layer-exclusion work.

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
