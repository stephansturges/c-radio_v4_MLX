from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from cradio_mlx.bundles.manifest import BundleManifest


@dataclass(frozen=True)
class BenchmarkRequest:
    model: Path
    image_size: int
    batch_sizes: list[int]
    report: Path


@dataclass(frozen=True)
class PyTorchBenchmarkRequest:
    model_id: str
    image: Path
    report: Path
    revision: str | None = None
    image_size: int | tuple[int, int] = 256
    dtype: str = "float32"
    device: str = "auto"
    warmups: int = 1
    repeats: int = 3


@dataclass(frozen=True)
class MLXSO400MBenchmarkRequest:
    checkpoint: Path
    image: Path | list[Path]
    report: Path
    revision: str | None = None
    variant: str = "so400m"
    image_size: int | tuple[int, int] | list[int | tuple[int, int]] = 512
    dtype: str = "bfloat16"
    batch_size: int | list[int] = 1
    warmups: int = 1
    repeats: int = 3
    materialize_outputs: bool = True
    compile_forward: bool = False


def write_benchmark_stub(request: BenchmarkRequest) -> Path:
    start = perf_counter()
    manifest = BundleManifest.load(request.model)
    elapsed = perf_counter() - start
    payload: dict[str, Any] = {
        "benchmark_state": "manifest_only",
        "note": "MLX forward benchmark will be enabled after the model implementation lands.",
        "model": str(request.model),
        "model_id": manifest.model_id,
        "revision": manifest.revision,
        "dtype": manifest.dtype,
        "quantization": manifest.quantization,
        "image_size": request.image_size,
        "batch_sizes": request.batch_sizes,
        "manifest_load_seconds": elapsed,
    }
    request.report.parent.mkdir(parents=True, exist_ok=True)
    with request.report.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return request.report


def write_pytorch_benchmark(request: PyTorchBenchmarkRequest) -> Path:
    from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner

    runner = PyTorchReferenceRunner.from_model_id(
        request.model_id,
        revision=request.revision,
        dtype=request.dtype,
        device=request.device,
    )

    load_start = perf_counter()
    runner.load()
    load_seconds = perf_counter() - load_start

    for _ in range(request.warmups):
        runner.encode(request.image, image_size=request.image_size)
        _synchronize_torch()

    latencies: list[float] = []
    result = None
    for _ in range(request.repeats):
        start = perf_counter()
        result = runner.encode(request.image, image_size=request.image_size)
        _synchronize_torch()
        latencies.append(perf_counter() - start)

    if result is None:
        raise ValueError("repeats must be at least 1")

    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95 = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))]

    payload: dict[str, Any] = {
        "benchmark_state": "complete",
        "backend": "pytorch",
        "accelerated": result.metadata["accelerated"],
        "device": result.metadata["device"],
        "model_id": request.model_id,
        "revision": request.revision,
        "dtype": request.dtype,
        "image": str(request.image),
        "image_size": result.image_size,
        "grid_h": result.grid_h,
        "grid_w": result.grid_w,
        "summary_shape": tuple(result.summary.shape),
        "spatial_shape": tuple(result.spatial.shape),
        "load_seconds": load_seconds,
        "warmups": request.warmups,
        "repeats": request.repeats,
        "latencies_seconds": latencies,
        "latency_p50_seconds": p50,
        "latency_p95_seconds": p95,
    }
    request.report.parent.mkdir(parents=True, exist_ok=True)
    with request.report.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return request.report


def write_mlx_so400m_benchmark(request: MLXSO400MBenchmarkRequest) -> Path:
    import mlx.core as mx

    from cradio_mlx.mlx_so400m import (
        MLXHEncoder,
        MLXSO400MEncoder,
        _load_rescaled_images,
        _to_numpy,
    )

    load_start = perf_counter()
    encoder_cls = MLXHEncoder if request.variant == "h" else MLXSO400MEncoder
    encoder = encoder_cls.load(
        request.checkpoint,
        dtype=request.dtype,
        revision=request.revision,
    )
    forward = mx.compile(encoder.forward) if request.compile_forward else encoder.forward
    load_seconds = perf_counter() - load_start

    image_paths = _as_path_list(request.image)
    image_sizes = _as_list(request.image_size)
    batch_sizes = _as_list(request.batch_size)

    if request.repeats < 1:
        raise ValueError("repeats must be at least 1")

    rows: list[dict[str, Any]] = []
    for image_size in image_sizes:
        for batch_size in batch_sizes:
            batch_images = _cycle_to_batch(image_paths, int(batch_size))
            pixel_values = _load_rescaled_images(batch_images, image_size)
            mx.eval(pixel_values)

            for _ in range(request.warmups):
                summary, spatial = forward(pixel_values)
                mx.eval(summary, spatial)
                if request.materialize_outputs:
                    _to_numpy(summary)
                    _to_numpy(spatial)

            latencies: list[float] = []
            summary = spatial = None
            for _ in range(request.repeats):
                start = perf_counter()
                summary, spatial = forward(pixel_values)
                mx.eval(summary, spatial)
                if request.materialize_outputs:
                    _to_numpy(summary)
                    _to_numpy(spatial)
                latencies.append(perf_counter() - start)

            if summary is None or spatial is None:
                raise ValueError("repeats must be at least 1")

            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies) // 2]
            p95 = sorted_latencies[
                min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))
            ]
            if isinstance(image_size, int):
                height = width = image_size
            else:
                height, width = image_size
            grid_h = height // encoder.spec.patch_size
            grid_w = width // encoder.spec.patch_size
            rows.append(
                {
                    "image_size": (height, width),
                    "batch_size": int(batch_size),
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "summary_shape": tuple(summary.shape),
                    "spatial_shape": tuple(spatial.shape),
                    "warmups": request.warmups,
                    "repeats": request.repeats,
                    "latencies_seconds": latencies,
                    "latency_p50_seconds": p50,
                    "latency_p95_seconds": p95,
                    "images_per_second_p50": int(batch_size) / p50,
                }
            )

    payload: dict[str, Any] = {
        "benchmark_state": "complete",
        "backend": "mlx",
        "accelerated": "gpu" in str(mx.default_device()).lower(),
        "device": str(mx.default_device()),
        "model_id": encoder.config.model_id,
        "revision": encoder.config.revision,
        "variant": encoder.config.variant,
        "dtype": request.dtype,
        "load_seconds": load_seconds,
        "images": [str(path) for path in image_paths],
        "materialize_outputs": request.materialize_outputs,
        "compile_forward": request.compile_forward,
        "rows": rows,
    }
    if len(rows) == 1:
        payload.update(rows[0])
    request.report.parent.mkdir(parents=True, exist_ok=True)
    with request.report.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return request.report


def _synchronize_torch() -> None:
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _as_path_list(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, list):
        return value
    return [value]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _cycle_to_batch(paths: list[Path], batch_size: int) -> list[Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not paths:
        raise ValueError("at least one image is required")
    return [paths[index % len(paths)] for index in range(batch_size)]
