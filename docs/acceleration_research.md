# Acceleration Research

This document tracks the current speedup levers for the MLX C-RADIOv4 runtime.

## Findings

1. MLX weight-only quantization is not a throughput path for this model.
   `mx.quantized_matmul` keeps weights packed, but activations and the rest of each
   transformer block remain bf16. It is useful for compact storage and lower runtime
   weight memory. A dequantize-at-load bundle is not useful here and should be treated as
   storage compression only, not as runtime quantization.

2. Cider W8A8 is the current low-bit runtime path worth using on Apple M5+.
   Cider adds online activation quantization and INT8 TensorOps kernels that MLX does not
   currently expose for general ViT matrix shapes.

3. Cider W8A8 g128 is the balanced variant.
   On the 12 WALDO crops at `512x512`, it is more precise than per-channel W8A8 while
   keeping most of the speed advantage. It costs a little more bundle size because of the
   per-group scale table.

4. Cider W8A8 p99.99 is the fastest tested variant.
   It uses Cider's percentile weight clipping. p99.99 retained usable smoke and WALDO
   cosine values; p99.9 was much faster but degraded embeddings too much.

5. Cider W4A8 is not viable yet for these bundles.
   It is smaller, but current precision and end-to-end speed are both worse.

6. Larger batch sizes are not a hidden win.
   Batch 8/16 improved raw throughput for bf16 in some cells, but Cider did not beat bf16
   for SO400M at 512px large batches. Keep latency-oriented batches around 1-4 unless the
   application can tolerate the queueing delay.

7. Qwen-style 2-bit/4-bit local speedups do not transfer automatically.
   Decoder LLM inference is usually more weight-bandwidth-bound and has mature low-bit
   decode paths. C-RADIOv4 image encoding runs large ViT matrix batches plus attention,
   layernorm, GELU, residual traffic, and image patching. If the low-bit kernel only
   covers isolated linears, the remaining block cost limits the speedup.

8. SmoothQuant-style Cider calibration improves precision, not speed.
   Alpha `0.5` calibration on the 12 WALDO crop set raised Cider cosine values, but adds
   an activation rescale before each Cider linear. SO400M p99.99 moved from 29.8 ms to
   30.1 ms at `512x512` batch 1; H p99.99 moved from 43.7 ms to 49.1 ms. Keep it as an
   experiment for downstream precision recovery, not as a throughput artifact.

## Current Best Modes

| Use case | Mode |
| --- | --- |
| Highest precision and simple deployment | bf16 |
| Compact high-precision weights | 8-bit affine |
| Balanced low-bit runtime on Apple M5+ | Cider W8A8 g128 |
| Fastest tested low-bit runtime on Apple M5+ | Cider W8A8 p99.99 |
| Lowest memory regardless of quality | Not supported; W4A8 failed precision |

## Remaining Work

- AWQ/QuaRot or layer-exclusion calibration for Cider W8A8/W4A8. SmoothQuant alpha `0.5`
  was tested and improved W8A8 precision, but not speed.
- ViTDet/windowed attention mode. The C-RADIOv4 report highlights ViTDet for high-resolution
  efficiency, but the current local configs have `vitdet_window_size: null`. A correct MLX
  implementation would need a PyTorch reference run with ViTDet enabled and separate parity
  gates because it changes the attention pattern.
- Fused custom Metal blocks. The plausible kernels are layernorm+linear, linear+bias+GELU,
  and MLP fusion. They are materially harder than using Cider linears and should only be
  pursued after profiling shows the target block dominates.
- Runtime integration with a crop queue. If Tator can batch independent crops, use benchmark
  data to choose a batch size per model/resolution rather than assuming one universal batch.

## Sources

- MLX `quantized_matmul` documentation: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- MLX custom Metal kernel documentation: https://ml-explore-mlx.mintlify.app/cpp/metal-kernels
- Cider W8A8/W4A8 runtime: https://github.com/Mininglamp-AI/cider
- C-RADIOv4 tech report: https://arxiv.org/abs/2601.17237
- QAttn mixed-precision ViT kernels: https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Kluska_QAttn_Efficient_GPU_Kernels_for_Mixed-precision_Vision_Transformers_CVPRW_2024_paper.pdf
