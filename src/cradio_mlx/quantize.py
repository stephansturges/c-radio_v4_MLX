from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.convert import METADATA_FILES, sha256_file

AFFINE_BITS = {4, 5, 6, 8}
QUANTIZATION_MODES = {"affine", "mxfp4", "mxfp8", "nvfp4"}
QUANTIZATION_MODE_CODES = {
    "affine": 0,
    "mxfp4": 1,
    "mxfp8": 2,
    "nvfp4": 3,
}


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
    if mode == "mxfp8" and (bits != 8 or group_size != 32):
        raise ValueError("mxfp8 quantization requires bits=8 and group_size=32")
    if mode == "mxfp4" and (bits != 4 or group_size != 32):
        raise ValueError("mxfp4 quantization requires bits=4 and group_size=32")
    if mode == "nvfp4" and (bits != 4 or group_size != 16):
        raise ValueError("nvfp4 quantization requires bits=4 and group_size=16")


def quantize_bundle(request: QuantizationRequest) -> Path:
    validate_quantization(request.bits, request.group_size, request.mode)
    source_manifest = BundleManifest.load(request.model)
    source_dir = request.model if request.model.is_dir() else request.model.parent
    source_weights = source_dir / "model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)

    request.out.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        source = source_dir / filename
        if source.exists() and source.is_file():
            shutil.copy2(source, request.out / filename)

    target_weights = request.out / "model.safetensors"
    quantization_stats = quantize_safetensors(
        source_weights,
        target_weights,
        group_size=request.group_size,
        bits=request.bits,
        mode=request.mode,
    )

    source_files = {
        path.name: sha256_file(path)
        for path in sorted(request.out.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    quantization_state = "dry_run_wrote_weights" if request.dry_run else "packed_weights"

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
            "state": quantization_state,
        },
        source_files=source_files,
        license=source_manifest.license,
        extra={
            "source_bundle": str(request.model),
            "conversion_state": "quantized_self_contained",
            "weights_file": "model.safetensors",
            "quantization_stats": quantization_stats,
        },
    )
    return quantized.save(request.out)


def quantize_safetensors(
    source: Path,
    target: Path,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> dict[str, int]:
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    tensors = {}
    stats = {
        "quantized_tensors": 0,
        "copied_tensors": 0,
        "padded_tensors": 0,
        "padded_features": 0,
    }
    with safe_open(source, framework="numpy") as handle:
        for key in handle.keys():
            array = handle.get_tensor(key)
            if _should_quantize_key(key, array):
                matrix = mx.array(array)
                original_in_features = int(array.shape[-1])
                pad_features = (-original_in_features) % group_size
                if pad_features:
                    padding = mx.zeros((*matrix.shape[:-1], pad_features), dtype=matrix.dtype)
                    matrix = mx.concatenate([matrix, padding], axis=-1)
                    stats["padded_tensors"] += 1
                    stats["padded_features"] += pad_features

                quantized = mx.quantize(
                    matrix,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                mx.eval(*quantized)
                prefix = key[: -len(".weight")]
                qweight = quantized[0]
                scales = quantized[1]
                tensors[f"{prefix}.qweight"] = np.asarray(qweight)
                tensors[f"{prefix}.qscales"] = np.asarray(scales)
                if len(quantized) == 3:
                    tensors[f"{prefix}.qbiases"] = np.asarray(quantized[2].astype(mx.float32))
                tensors[f"{prefix}.qbits"] = np.array([bits], dtype=np.int32)
                tensors[f"{prefix}.qgroup_size"] = np.array([group_size], dtype=np.int32)
                tensors[f"{prefix}.qmode_code"] = np.array(
                    [QUANTIZATION_MODE_CODES[mode]],
                    dtype=np.int32,
                )
                tensors[f"{prefix}.qin_features"] = np.array(
                    [original_in_features],
                    dtype=np.int32,
                )
                tensors[f"{prefix}.qpadded_in_features"] = np.array(
                    [matrix.shape[-1]],
                    dtype=np.int32,
                )
                stats["quantized_tensors"] += 1
            else:
                tensors[key] = array
                stats["copied_tensors"] += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, target)
    return stats


def _should_quantize_key(key: str, array) -> bool:
    if not key.endswith(".weight") or array.ndim != 2:
        return False
    if key == "radio_model.model.patch_generator.embedder.weight":
        return True
    return (
        ".attn.qkv.weight" in key
        or ".attn.proj.weight" in key
        or ".mlp.fc1.weight" in key
        or ".mlp.fc2.weight" in key
    )
