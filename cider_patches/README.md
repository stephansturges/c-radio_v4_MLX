# Cider Runtime Patches

These patches are for the optional Cider dependency used by the M5+ W8A8 bundles.
They are not applied by default when installing `cradio-mlx`.

## Fused LayerNorm / GELU + W8A8 Linear

Patches:

```sh
cider_patches/0001-layernorm-w8a8-linear.patch
cider_patches/0002-gelu-w8a8-linear.patch
```

Base tested Cider commit:

```text
01b8f9c0e65a54375e50eab9480ca2ff6a1a0d6e
```

Apply and install:

```sh
git clone https://github.com/Mininglamp-AI/cider.git /tmp/cider-src
cd /tmp/cider-src
git checkout 01b8f9c0e65a54375e50eab9480ca2ff6a1a0d6e
git apply --3way /path/to/c-radio_v4_MLX/cider_patches/0001-layernorm-w8a8-linear.patch
git apply /path/to/c-radio_v4_MLX/cider_patches/0002-gelu-w8a8-linear.patch
python -m pip install -e .
```

The patch adds `cider.ops.layernorm_perchannel_linear(...)`, which fuses
LayerNorm and activation quantization into one Metal kernel before calling Cider's
existing INT8 TensorOps GEMM.

The second patch adds `cider.ops.gelu_perchannel_linear(...)`, which fuses exact-ish
GELU and activation quantization before the same W8A8 GEMM. The current C-RADIOv4
p99.99 Cider bundles are per-channel W8A8, so these ops cover the tested fast path.

Use it with:

```sh
cradio-mlx mlx-benchmark \
  --checkpoint bundles/c-radiov4-so400m-cider-w8a8-p9999 \
  --variant so400m \
  --image data/golden_images/smoke.jpg \
  --image-sizes 512 \
  --batch-size 1 \
  --dtype bfloat16 \
  --cider-fusion required \
  --cider-fusion-targets ln+mlp \
  --report reports/experiments/fused-ln-w8a8.json
```
