from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cradio_mlx.bundles.manifest import BundleManifest

AFFINE_BITS = {4, 5, 6, 8}
QUANTIZATION_MODES = {"affine", "mxfp4", "mxfp8", "nvfp4"}


@dataclass(frozen=True)
class QuantizationRequest:
    model: Path
    out: Path
    bits: int | None = 8
    group_size: int = 64
    mode: str = "affine"
    dry_run: bool = False


def validate_quantization(bits: int | None, group_size: int, mode: str) -> None:
    if mode not in QUANTIZATION_MODES:
        known = ", ".join(sorted(QUANTIZATION_MODES))
        raise ValueError(f"unsupported quantization mode={mode!r}; known: {known}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if mode == "affine" and bits not in AFFINE_BITS:
        raise ValueError(f"affine quantization bits must be one of {sorted(AFFINE_BITS)}")


def quantize_bundle(request: QuantizationRequest) -> Path:
    validate_quantization(request.bits, request.group_size, request.mode)
    source_manifest = BundleManifest.load(request.model)
    request.out.mkdir(parents=True, exist_ok=True)

    quantized = BundleManifest(
        model_id=source_manifest.model_id,
        revision=source_manifest.revision,
        variant=source_manifest.variant,
        dtype=source_manifest.dtype,
        patch_size=source_manifest.patch_size,
        preferred_resolution=source_manifest.preferred_resolution,
        max_resolution=source_manifest.max_resolution,
        quantization={
            "mode": request.mode,
            "bits": request.bits,
            "group_size": request.group_size,
            "state": "planned" if request.dry_run else "pending_implementation",
        },
        source_files=source_manifest.source_files,
        license=source_manifest.license,
        extra={
            "source_bundle": str(request.model),
            "conversion_state": "quantization_manifest_only",
        },
    )
    manifest_path = quantized.save(request.out)

    if not request.dry_run:
        raise NotImplementedError(
            "Quantized weight writing is not implemented yet. Re-run with --dry-run to create "
            "the planned quantized bundle manifest."
        )

    return manifest_path
