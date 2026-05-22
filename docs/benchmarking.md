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
| C-RADIOv4-SO400M | 8-bit affine | 51.6 ms | 19.4 images/s |
| C-RADIOv4-SO400M | mxfp8 | 62.5 ms | 16.0 images/s |
| C-RADIOv4-H | bf16 | 51.9 ms | 19.3 images/s |
| C-RADIOv4-H | 8-bit affine | 73.2 ms | 13.7 images/s |
| C-RADIOv4-H | mxfp8 | 63.5 ms | 15.8 images/s |

8-bit affine currently reduces bundle size and preserves precision but is slower than
bf16 on MLX GPU. Weight-only `mxfp8` is also slower than bf16 and materially lower
precision for this model. Use fast-kernel compiled bf16 for latency-sensitive local
inference.
