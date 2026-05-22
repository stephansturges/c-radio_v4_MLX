#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner
from cradio_mlx.imaging import load_rgb_image
from cradio_mlx.metrics import cosine_similarity

COMPUTE_UNITS = ("ALL", "CPU_AND_GPU", "CPU_ONLY")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast-kill Core ML conversion and benchmark proof for C-RADIOv4."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--variant", choices=["so400m", "h"], required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-p50-ms", type=float)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compute-units", nargs="+", choices=COMPUTE_UNITS, default=["ALL"])
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    package = args.package or (
        args.out.parent / f"{args.variant}-{args.image_size}-b{args.batch_size}.mlpackage"
    )
    payload = run_fastkill(
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        variant=args.variant,
        image=args.image,
        image_size=args.image_size,
        batch_size=args.batch_size,
        package=package,
        out=args.out,
        baseline_p50_ms=args.baseline_p50_ms,
        warmups=args.warmups,
        repeats=args.repeats,
        compute_units=args.compute_units,
        skip_convert=args.skip_convert,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def run_fastkill(
    checkpoint: Path,
    model_id: str,
    variant: str,
    image: Path,
    image_size: int,
    batch_size: int,
    package: Path,
    out: Path,
    baseline_p50_ms: float | None,
    warmups: int,
    repeats: int,
    compute_units: list[str],
    skip_convert: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if repeats < 1:
        raise ValueError("--repeats must be at least 1")

    payload: dict[str, Any] = {
        "benchmark_state": "started",
        "checkpoint": str(checkpoint),
        "model_id": model_id,
        "variant": variant,
        "image": str(image),
        "image_size": image_size,
        "batch_size": batch_size,
        "package": str(package),
        "out": str(out),
        "baseline_p50_ms": baseline_p50_ms,
        "warmups": warmups,
        "repeats": repeats,
        "compute_units": compute_units,
    }

    try:
        import coremltools as ct
        import torch
    except Exception as exc:
        return _failed(payload, "dependency_unavailable", exc)

    try:
        runner = PyTorchReferenceRunner.from_local_checkpoint(
            checkpoint,
            model_id=model_id,
            dtype="float32",
            device="cpu",
        ).load()
        pixel_values = _pixel_values(runner, image, image_size, batch_size, torch)
        reference = _reference_outputs(runner, pixel_values, torch)
        payload["reference_shapes"] = {
            "summary": list(reference["summary"].shape),
            "spatial": list(reference["spatial"].shape),
        }
    except Exception as exc:
        return _failed(payload, "pytorch_reference_failed", exc)

    if not skip_convert or not package.exists():
        try:
            convert_result = _convert_to_coreml(
                runner=runner,
                pixel_values=pixel_values,
                package=package,
                ct=ct,
                torch=torch,
            )
            payload.update(convert_result)
        except Exception as exc:
            return _failed(payload, "conversion_failed", exc)
    else:
        payload["convert_seconds"] = None

    rows: list[dict[str, Any]] = []
    for unit in compute_units:
        rows.append(
            _benchmark_unit(
                package=package,
                unit=unit,
                pixel_values=pixel_values,
                reference=reference,
                warmups=warmups,
                repeats=repeats,
                ct=ct,
            )
        )

    payload["rows"] = rows
    payload["gate"] = _evaluate_gate(rows, baseline_p50_ms, variant)
    payload["benchmark_state"] = "complete"
    return payload


def _pixel_values(
    runner: PyTorchReferenceRunner,
    image: Path,
    image_size: int,
    batch_size: int,
    torch: Any,
) -> Any:
    images = [load_rgb_image(image) for _ in range(batch_size)]
    processor = runner._processor
    if processor is None:
        raise RuntimeError("PyTorch runner did not initialize its processor")
    pixel_values = processor(
        images=images,
        return_tensors="pt",
        do_resize=True,
        size={"height": image_size, "width": image_size},
    ).pixel_values
    return pixel_values.to(dtype=torch.float32, device="cpu")


def _reference_outputs(
    runner: PyTorchReferenceRunner,
    pixel_values: Any,
    torch: Any,
) -> dict[str, np.ndarray]:
    model = runner._model
    if model is None:
        raise RuntimeError("PyTorch runner did not initialize its model")
    model = model.eval().cpu()
    with torch.inference_mode():
        summary, spatial = model(pixel_values)
    return {
        "summary": summary.detach().float().cpu().numpy(),
        "spatial": spatial.detach().float().cpu().numpy(),
    }


def _convert_to_coreml(
    runner: PyTorchReferenceRunner,
    pixel_values: Any,
    package: Path,
    ct: Any,
    torch: Any,
) -> dict[str, Any]:
    model = runner._model
    if model is None:
        raise RuntimeError("PyTorch runner did not initialize its model")

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values: Any) -> tuple[Any, Any]:
            summary, spatial = self.wrapped(pixel_values)
            return summary, spatial

    wrapped = Wrapper(model.eval().cpu())
    package.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, pixel_values, strict=False)
    attempts: list[dict[str, str]] = []
    mlmodel = None
    convert_method = None
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="pixel_values", shape=tuple(pixel_values.shape))],
            outputs=[ct.TensorType(name="summary"), ct.TensorType(name="spatial")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
        convert_method = "torchscript_trace"
    except Exception as exc:
        attempts.append(
            {
                "method": "torchscript_trace",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    if mlmodel is None and hasattr(torch, "export"):
        try:
            exported = torch.export.export(wrapped, (pixel_values,))
            if hasattr(exported, "run_decompositions"):
                exported = exported.run_decompositions({})
            mlmodel = ct.convert(
                exported,
                inputs=[ct.TensorType(name="pixel_values", shape=tuple(pixel_values.shape))],
                outputs=[ct.TensorType(name="summary"), ct.TensorType(name="spatial")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS13,
                compute_precision=ct.precision.FLOAT16,
            )
            convert_method = "torch_export"
        except Exception as exc:
            attempts.append(
                {
                    "method": "torch_export",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if mlmodel is None:
        raise RuntimeError(f"Core ML conversion failed for all methods: {attempts}")
    if package.exists():
        import shutil

        shutil.rmtree(package)
    mlmodel.save(str(package))
    return {
        "convert_seconds": perf_counter() - start,
        "convert_method": convert_method,
        "convert_attempt_failures": attempts,
    }


def _benchmark_unit(
    package: Path,
    unit: str,
    pixel_values: Any,
    reference: dict[str, np.ndarray],
    warmups: int,
    repeats: int,
    ct: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"compute_unit": unit, "benchmark_state": "started"}
    try:
        load_start = perf_counter()
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, unit))
        row["load_seconds"] = perf_counter() - load_start
        inputs = {"pixel_values": pixel_values.detach().cpu().numpy().astype(np.float32)}

        for _ in range(warmups):
            model.predict(inputs)

        latencies: list[float] = []
        outputs = None
        for _ in range(repeats):
            start = perf_counter()
            outputs = model.predict(inputs)
            latencies.append(perf_counter() - start)

        if outputs is None:
            raise ValueError("repeats must be at least 1")
        summary = _select_output(outputs, "summary")
        spatial = _select_output(outputs, "spatial")
        row.update(
            {
                "benchmark_state": "complete",
                "latencies_seconds": latencies,
                "latency_p50_seconds": _percentile(latencies, 0.50),
                "latency_p95_seconds": _percentile(latencies, 0.95),
                "summary_shape": list(summary.shape),
                "spatial_shape": list(spatial.shape),
                "summary_cosine": cosine_similarity(reference["summary"], summary),
                "spatial_cosine": cosine_similarity(reference["spatial"], spatial),
            }
        )
    except Exception as exc:
        row.update(
            {
                "benchmark_state": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
    return row


def _select_output(outputs: dict[str, Any], preferred: str) -> np.ndarray:
    if preferred in outputs:
        return np.asarray(outputs[preferred])
    if len(outputs) == 2:
        key = sorted(outputs)[0 if preferred == "summary" else 1]
        return np.asarray(outputs[key])
    raise KeyError(f"Core ML output {preferred!r} not found; keys={sorted(outputs)}")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * percentile))]


def _evaluate_gate(
    rows: list[dict[str, Any]],
    baseline_p50_ms: float | None,
    variant: str,
) -> dict[str, Any]:
    complete = [row for row in rows if row.get("benchmark_state") == "complete"]
    thresholds = {
        "so400m": {"summary": 0.998, "spatial": 0.998},
        "h": {"summary": 0.996, "spatial": 0.996},
    }[variant]
    all_row = next((row for row in complete if row["compute_unit"] == "ALL"), None)
    gpu_row = next((row for row in complete if row["compute_unit"] == "CPU_AND_GPU"), None)
    precision_pass = bool(
        all_row
        and all_row["summary_cosine"] >= thresholds["summary"]
        and all_row["spatial_cosine"] >= thresholds["spatial"]
    )
    speed_pass: bool | None = None
    if baseline_p50_ms is not None and all_row:
        speed_pass = all_row["latency_p50_seconds"] * 1000.0 <= baseline_p50_ms * 0.80
    ane_inferred: bool | None = None
    if all_row and gpu_row:
        ane_inferred = all_row["latency_p50_seconds"] <= gpu_row["latency_p50_seconds"] * 0.85
    return {
        "precision_pass": precision_pass,
        "speed_pass": speed_pass,
        "ane_inferred": ane_inferred,
        "baseline_required_speed_ms": baseline_p50_ms * 0.80
        if baseline_p50_ms is not None
        else None,
        "decision": "continue" if precision_pass and speed_pass else "kill",
    }


def _failed(payload: dict[str, Any], state: str, exc: Exception) -> dict[str, Any]:
    payload.update(
        {
            "benchmark_state": state,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "gate": {"decision": "kill"},
        }
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
