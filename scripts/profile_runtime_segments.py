#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx

import cradio_mlx.mlx_so400m as runtime
from cradio_mlx.mlx_so400m import MLXHEncoder, MLXSO400MEncoder


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile C-RADIOv4 MLX runtime segments with forced synchronization."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", choices=["so400m", "h"], required=True)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cider-fusion", choices=["off", "auto", "required"], default="off")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    payload = profile_segments(
        checkpoint=args.checkpoint,
        variant=args.variant,
        images=args.images,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dtype=args.dtype,
        warmups=args.warmups,
        repeats=args.repeats,
        cider_fusion=args.cider_fusion,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def profile_segments(
    checkpoint: Path,
    variant: str,
    images: list[Path],
    image_size: int,
    batch_size: int,
    dtype: str,
    warmups: int,
    repeats: int,
    cider_fusion: str,
) -> dict[str, Any]:
    encoder_cls = MLXHEncoder if variant == "h" else MLXSO400MEncoder
    encoder = encoder_cls.load(checkpoint, dtype=dtype, cider_fusion=cider_fusion)
    batch_images = _cycle_to_batch(images, batch_size)
    pixel_values = runtime._load_rescaled_images(batch_images, image_size)
    mx.eval(pixel_values)

    for _ in range(warmups):
        summary, spatial = encoder.forward(pixel_values)
        mx.eval(summary, spatial)

    stats: dict[str, list[float]] = defaultdict(list)
    original_linear = runtime._linear
    original_layer_norm = runtime._layer_norm
    original_gelu = runtime._gelu_exact
    original_sdpa = runtime.mx_fast.scaled_dot_product_attention

    def timed_linear(x, weights, prefix):
        start = perf_counter()
        y = original_linear(x, weights, prefix)
        mx.eval(y)
        elapsed = perf_counter() - start
        stats[f"linear:{_linear_family(prefix)}"].append(elapsed)
        stats[f"linear_prefix:{prefix}"].append(elapsed)
        return y

    def timed_layer_norm(x, weight, bias, eps=1e-6):
        start = perf_counter()
        y = original_layer_norm(x, weight, bias, eps)
        mx.eval(y)
        stats["layer_norm"].append(perf_counter() - start)
        return y

    def timed_gelu(x):
        start = perf_counter()
        y = original_gelu(x)
        mx.eval(y)
        stats["gelu"].append(perf_counter() - start)
        return y

    def timed_sdpa(*args, **kwargs):
        start = perf_counter()
        y = original_sdpa(*args, **kwargs)
        mx.eval(y)
        stats["scaled_dot_product_attention"].append(perf_counter() - start)
        return y

    runtime._linear = timed_linear
    runtime._layer_norm = timed_layer_norm
    runtime._gelu_exact = timed_gelu
    runtime.mx_fast.scaled_dot_product_attention = timed_sdpa
    end_to_end: list[float] = []
    try:
        for _ in range(repeats):
            start = perf_counter()
            summary, spatial = encoder.forward(pixel_values)
            mx.eval(summary, spatial)
            end_to_end.append(perf_counter() - start)
    finally:
        runtime._linear = original_linear
        runtime._layer_norm = original_layer_norm
        runtime._gelu_exact = original_gelu
        runtime.mx_fast.scaled_dot_product_attention = original_sdpa

    return {
        "benchmark_state": "complete",
        "note": (
            "Segment timings force mx.eval inside the graph and are diagnostic, "
            "not production latency."
        ),
        "checkpoint": str(checkpoint),
        "variant": variant,
        "dtype": dtype,
        "image_size": image_size,
        "batch_size": batch_size,
        "warmups": warmups,
        "repeats": repeats,
        "cider_fusion": cider_fusion,
        "end_to_end": _summarize(end_to_end),
        "segments": {name: _summarize(values) for name, values in sorted(stats.items())},
    }


def _summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    count = len(ordered)
    total = sum(ordered)
    return {
        "count": count,
        "total_seconds": total,
        "mean_seconds": total / count,
        "p50_seconds": ordered[count // 2],
        "p95_seconds": ordered[min(count - 1, int(count * 0.95))],
    }


def _linear_family(prefix: str) -> str:
    if prefix.endswith(".attn.qkv"):
        return "attn.qkv"
    if prefix.endswith(".attn.proj"):
        return "attn.proj"
    if prefix.endswith(".mlp.fc1"):
        return "mlp.fc1"
    if prefix.endswith(".mlp.fc2"):
        return "mlp.fc2"
    if prefix.endswith(".patch_generator.embedder"):
        return "patch_embed"
    return "other"


def _cycle_to_batch(paths: list[Path], batch_size: int) -> list[Path]:
    if not paths:
        raise ValueError("at least one image is required")
    return [paths[index % len(paths)] for index in range(batch_size)]


if __name__ == "__main__":
    raise SystemExit(main())
