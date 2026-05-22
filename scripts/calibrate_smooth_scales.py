#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors import safe_open

import cradio_mlx.mlx_so400m as runtime
from cradio_mlx.mlx_so400m import MLXHEncoder, MLXSO400MEncoder
from cradio_mlx.quantize import _should_quantize_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect SmoothQuant input scales for Cider W8A8/W4A8 quantization."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", choices=["so400m", "h"], required=True)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--min-scale", type=float, default=1e-3)
    parser.add_argument("--max-scale", type=float, default=1e3)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    activation_maxima = collect_activation_maxima(
        checkpoint=args.checkpoint,
        variant=args.variant,
        images=args.images,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    scales, report = build_smooth_scales(
        checkpoint=args.checkpoint,
        activation_maxima=activation_maxima,
        alpha=args.alpha,
        eps=args.eps,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **scales)

    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "variant": args.variant,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "images": [str(path) for path in args.images],
            "alpha": args.alpha,
            "eps": args.eps,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "out": str(args.out),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def collect_activation_maxima(
    checkpoint: Path,
    variant: str,
    images: list[Path],
    image_size: int,
    batch_size: int,
    dtype: str,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    encoder_cls = MLXHEncoder if variant == "h" else MLXSO400MEncoder
    encoder = encoder_cls.load(checkpoint, dtype=dtype)

    activation_maxima: dict[str, np.ndarray] = {}
    original_linear = runtime._linear

    def collecting_linear(x: mx.array, weights: dict[str, mx.array], prefix: str) -> mx.array:
        flat = mx.reshape(mx.abs(x.astype(mx.float32)), (-1, x.shape[-1]))
        current = mx.max(flat, axis=0)
        mx.eval(current)
        current_np = np.asarray(current)
        previous = activation_maxima.get(prefix)
        activation_maxima[prefix] = (
            current_np if previous is None else np.maximum(previous, current_np)
        )
        return original_linear(x, weights, prefix)

    runtime._linear = collecting_linear
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            summary, spatial = encoder.forward(runtime._load_rescaled_images(batch, image_size))
            mx.eval(summary, spatial)
    finally:
        runtime._linear = original_linear

    return activation_maxima


def build_smooth_scales(
    checkpoint: Path,
    activation_maxima: dict[str, np.ndarray],
    alpha: float,
    eps: float,
    min_scale: float,
    max_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model_path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    scales: dict[str, np.ndarray] = {}
    missing_activation_prefixes: list[str] = []
    with safe_open(model_path, framework="numpy") as handle:
        for key in handle.keys():
            weight = handle.get_tensor(key)
            if not _should_quantize_key(key, weight):
                continue
            prefix = key[: -len(".weight")]
            activation_max = activation_maxima.get(prefix)
            if activation_max is None:
                missing_activation_prefixes.append(prefix)
                continue

            weight_max = np.max(np.abs(weight.astype(np.float32)), axis=0)
            scale = (activation_max + eps) ** alpha / (weight_max + eps) ** (1.0 - alpha)
            scale = np.where(np.isfinite(scale), scale, 1.0)
            scale = np.clip(scale, min_scale, max_scale).astype(np.float32)
            scales[prefix] = scale

    return scales, {
        "state": "complete",
        "activation_prefixes": len(activation_maxima),
        "scale_tensors": len(scales),
        "missing_activation_prefixes": missing_activation_prefixes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
