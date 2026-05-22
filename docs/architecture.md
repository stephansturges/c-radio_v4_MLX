# Architecture

`cradio-mlx` is organized around a reference-first conversion flow:

1. Download an official C-RADIOv4 checkpoint from Hugging Face at a pinned revision.
2. Run the official PyTorch model once as the source of truth.
3. Capture preprocessing, config, summary outputs, and spatial outputs.
4. Convert weights into an MLX-native model implementation.
5. Save a self-contained bundle with model weights, processor metadata, and provenance.
6. Run parity and benchmark gates before publishing a bundle.

Runtime inference must not require `trust_remote_code=True`. That flag belongs only in the
controlled reference/conversion path.
