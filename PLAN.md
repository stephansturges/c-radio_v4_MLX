# Standalone MLX C-RADIOv4 Implementation Plan

## Inferred Goal

Build and ship a standalone Apple Silicon package, `cradio-mlx`, that converts official
NVIDIA C-RADIOv4 Hugging Face checkpoints into reproducible local MLX bundles and serves
summary plus spatial image embeddings through a CLI and Python API.

This is not primarily a chat runtime. C-RADIOv4 is an embedding model, so the first product
contract is feature extraction parity against the official PyTorch implementation. A
VLM-style prompt adapter can be added later as a downstream integration convenience.

## Scope Boundaries

In scope:

- `nvidia/C-RADIOv4-SO400M` and `nvidia/C-RADIOv4-H`
- PyTorch MPS reference runner
- MLX bundle format with config, processor, metadata, manifest, and weights
- summary embeddings, spatial embeddings, and grid reshape helpers
- `bfloat16` first, 8-bit affine second
- benchmark and parity harnesses
- provenance, license, and revision metadata in every bundle

Out of scope for the MVP:

- end-to-end text generation
- projector training
- lower-bit quantization as a default path
- depending on `trust_remote_code=True` at runtime

## Milestones

### 1. Bootstrap

Deliverables:

- Python package skeleton
- CLI entrypoint
- model/config presets
- bundle manifest format
- docs and test harness

Acceptance:

- `python -m compileall src tests` passes
- shape and manifest tests pass
- CLI can inspect spatial dimensions and bundle manifests

### 2. PyTorch Reference Runner

Deliverables:

- revision-pinned Hugging Face loader
- MPS-first device selection with CPU fallback
- golden output capture for summary and spatial tensors
- preprocessing comparison hooks

Acceptance:

- official checkpoint loads from a pinned revision
- golden images produce persisted reference outputs
- output shapes match `(B, C)` and `(B, T, D)`

### 3. SO400M MLX Core

Deliverables:

- MLX module definitions for the SO400M architecture
- safetensors weight loader
- HF-to-MLX key mapping audit
- summary and spatial forward pass

Acceptance:

- SO400M `bfloat16` loads locally from the pinned Hugging Face checkpoint directory
- summary cosine similarity is at least `0.999` against PyTorch reference
- spatial cosine similarity is at least `0.995` against PyTorch reference

Status: complete for direct checkpoint loading and self-contained bundle loading.

### 4. H Model Support

Deliverables:

- C-RADIOv4-H config support
- expanded weight mapping
- larger-model memory and latency smoke tests

Acceptance:

- H `bfloat16` loads locally from an MLX bundle
- same parity thresholds as SO400M pass on golden images

### 5. Bundle Writer and Converter

Deliverables:

- revision-pinned local checkpoint ingestion
- config and processor capture
- MLX weights written as safetensors
- provenance manifest and model card stub

Acceptance:

- converted bundle runs without `trust_remote_code=True`
- manifest contains source model ID, revision, dtype, license/provenance fields, and file hashes

### 6. Quantization

Deliverables:

- 8-bit affine quantization
- regression thresholds by model and image size
- optional lower-bit experimental paths after parity gates exist

Acceptance:

- 8-bit bundles pass defined parity gates
- lower-bit bundles are marked experimental unless they meet the same gates

### 7. Benchmarks and Release

Deliverables:

- benchmark matrix by model, dtype, image size, batch size, and machine class
- CLI JSON reports
- PyPI packaging
- Hugging Face upload workflow for bundles
- Apple Silicon CI smoke path

Acceptance:

- benchmark reports include cold load, warm p50/p95 latency, throughput, and memory
- release artifacts preserve upstream licenses and revision metadata

## Immediate Work Queue

1. Add calibrated quantization experiments if a deployment requires smaller-than-affine
   bundles.
2. Investigate custom Metal kernels only if fixed-contract 512px latency needs to improve
   beyond the fused MLX fast-kernel path.
3. Add CI artifact publishing for benchmark JSON summaries.
4. Add optional Hugging Face upload tooling for self-contained bundles.
