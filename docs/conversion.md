# Conversion

The converter writes self-contained local MLX bundles from Hugging Face checkpoint
directories. It validates the source checkpoint, records provenance, copies available
config/processor metadata, copies `model.safetensors`, hashes bundle files, and writes a
manifest with model ID, revision, dtype, variant, and NVIDIA model-license provenance.

The next checkpoint is the SO400M `bfloat16` weight-map audit:

- enumerate source safetensors keys
- map them to MLX module parameter names
- instantiate the MLX architecture
- load weights and compare layer outputs against PyTorch
- promote the converter from manifest-only to weight-producing

Current SO400M implementation status:

- safetensors audit works for the official checkpoint
- native MLX SO400M fp32 forward works at 256 and 512
- native MLX SO400M bf16 forward works at 512
- 512px fp32 parity against PyTorch/MPS: summary cosine above `0.999`, spatial cosine above `0.995`
- 512px bf16 parity against PyTorch/MPS: summary cosine above `0.999`, spatial cosine above `0.995`

Current H implementation status:

- safetensors audit works for the official checkpoint
- native MLX H bf16 forward works at 512
- 512px bf16 parity against PyTorch/MPS: summary cosine above `0.999`, spatial cosine above `0.999`

The native MLX loaders can run from either the original checkpoint directories or the
self-contained bundle directories.

Start the audit with:

```sh
cradio-mlx audit-weights \
  --hf-path checkpoints/c-radiov4-so400m \
  --out reports/so400m-weight-audit.json
```

Capture a golden PyTorch reference output with:

```sh
cradio-mlx capture-reference \
  --model-id nvidia/C-RADIOv4-SO400M \
  --revision <pinned-sha> \
  --image data/golden_images/example.jpg \
  --out data/expected_metadata/so400m
```
