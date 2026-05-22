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

## Current Smoke Results

Local `512x512`, batch-1, `bfloat16` results on Apple Silicon:

| Model | Backend | p50 latency | p95 latency |
| --- | --- | ---: | ---: |
| C-RADIOv4-SO400M | MLX GPU | 46.5 ms | 46.7 ms |
| C-RADIOv4-SO400M | PyTorch MPS | 120.0 ms | 120.7 ms |
| C-RADIOv4-H | MLX GPU | 59.9 ms | 59.9 ms |
| C-RADIOv4-H | PyTorch MPS | 165.8 ms | 167.3 ms |
