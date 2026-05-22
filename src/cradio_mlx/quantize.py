from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.convert import METADATA_FILES, sha256_file

AFFINE_BITS = {4, 5, 6, 8}
CIDER_GROUP_SIZES = {0, 64, 128, 256}
QUANTIZATION_MODES = {"affine", "mxfp4", "mxfp8", "nvfp4", "cider-w8a8", "cider-w4a8"}
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
    clip_percentile: float | None = None
    smooth_scales: Path | None = None
    dry_run: bool = False


def validate_quantization(
    bits: int | None,
    group_size: int,
    mode: str,
    clip_percentile: float | None = None,
) -> None:
    if mode not in QUANTIZATION_MODES:
        known = ", ".join(sorted(QUANTIZATION_MODES))
        raise ValueError(f"unsupported quantization mode={mode!r}; known: {known}")
    if clip_percentile is not None and not (0.0 < clip_percentile <= 100.0):
        raise ValueError("clip_percentile must be in the range (0, 100]")
    if mode in {"cider-w8a8", "cider-w4a8"}:
        required_bits = 8 if mode == "cider-w8a8" else 4
        if bits != required_bits:
            raise ValueError(f"{mode} quantization requires bits={required_bits}")
        if group_size not in CIDER_GROUP_SIZES:
            raise ValueError(f"{mode} group_size must be one of {sorted(CIDER_GROUP_SIZES)}")
        if mode == "cider-w4a8" and group_size != 0:
            raise ValueError("cider-w4a8 currently requires group_size=0")
        if clip_percentile is not None and (mode != "cider-w8a8" or group_size != 0):
            raise ValueError("clip_percentile is only supported for cider-w8a8 group_size=0")
        return
    if clip_percentile is not None:
        raise ValueError("clip_percentile is only supported for cider-w8a8")
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
    if request.smooth_scales is not None and request.mode not in {"cider-w8a8", "cider-w4a8"}:
        raise ValueError("smooth_scales is only supported for Cider runtime quantization modes")
    validate_quantization(
        request.bits,
        request.group_size,
        request.mode,
        clip_percentile=request.clip_percentile,
    )
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
    smooth_scales = _load_smooth_scales(request.smooth_scales)
    quantization_stats = quantize_safetensors(
        source_weights,
        target_weights,
        group_size=request.group_size,
        bits=request.bits,
        mode=request.mode,
        clip_percentile=request.clip_percentile,
        smooth_scales=smooth_scales,
    )

    source_files = {
        path.name: sha256_file(path)
        for path in sorted(request.out.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    quantization_state = (
        "dry_run_wrote_weights"
        if request.dry_run
        else (
            "packed_w8a8_runtime"
            if request.mode == "cider-w8a8"
            else "packed_w4a8_runtime"
            if request.mode == "cider-w4a8"
            else "packed_weights"
        )
    )
    quantization = {
        "mode": request.mode,
        "bits": request.bits,
        "group_size": request.group_size,
        "state": quantization_state,
    }
    if request.mode in {"cider-w8a8", "cider-w4a8"}:
        quantization.update(
            {
                "runtime": "cider",
                "weight_bits": 8 if request.mode == "cider-w8a8" else 4,
                "activation_bits": 8,
                "scheme": "symmetric_per_channel"
                if request.group_size == 0
                else "symmetric_per_group",
            }
        )
        if request.clip_percentile is not None:
            quantization["clip_percentile"] = request.clip_percentile
        if request.smooth_scales is not None:
            quantization["smooth_scales"] = str(request.smooth_scales)

    quantized = BundleManifest(
        model_id=source_manifest.model_id,
        revision=source_manifest.revision,
        variant=source_manifest.variant,
        dtype=source_manifest.dtype,
        patch_size=source_manifest.patch_size,
        preferred_resolution=source_manifest.preferred_resolution,
        max_resolution=source_manifest.max_resolution,
        quantization=quantization,
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
    clip_percentile: float | None = None,
    smooth_scales: dict[str, Any] | None = None,
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
        "smoothed_tensors": 0,
    }
    with safe_open(source, framework="numpy") as handle:
        for key in handle.keys():
            array = handle.get_tensor(key)
            if _should_quantize_key(key, array):
                original_in_features = int(array.shape[-1])
                prefix = key[: -len(".weight")]
                if mode in {"cider-w8a8", "cider-w4a8"}:
                    pad_group_size = 2 if mode == "cider-w4a8" else group_size
                    matrix, pad_features = _pad_numpy_matrix(array, pad_group_size)
                    if pad_features:
                        stats["padded_tensors"] += 1
                        stats["padded_features"] += pad_features
                    smooth_scale = _smooth_scale_for_prefix(
                        smooth_scales,
                        prefix,
                        original_in_features,
                        matrix.shape[-1],
                    )
                    if smooth_scale is not None:
                        matrix = matrix * smooth_scale.reshape(1, -1)
                        tensors[f"{prefix}.cider_input_scale"] = smooth_scale
                        stats["smoothed_tensors"] += 1
                    if mode == "cider-w8a8":
                        qweight, scales = _quantize_cider_w8a8(
                            matrix,
                            group_size,
                            clip_percentile=clip_percentile,
                        )
                        tensors[f"{prefix}.cider_weight"] = qweight
                        tensors[f"{prefix}.cider_scale"] = scales
                        tensors[f"{prefix}.cider_group_size"] = np.array(
                            [group_size],
                            dtype=np.int32,
                        )
                    else:
                        qweight, scales = _quantize_cider_w4a8(matrix)
                        tensors[f"{prefix}.cider4_weight"] = qweight
                        tensors[f"{prefix}.cider4_scale"] = scales
                        tensors[f"{prefix}.cider4_group_size"] = np.array(
                            [group_size],
                            dtype=np.int32,
                        )
                    tensors[f"{prefix}.cider_in_features"] = np.array(
                        [original_in_features],
                        dtype=np.int32,
                    )
                    tensors[f"{prefix}.cider_padded_in_features"] = np.array(
                        [matrix.shape[-1]],
                        dtype=np.int32,
                    )
                    tensors[f"{prefix}.cider_out_features"] = np.array(
                        [array.shape[0]],
                        dtype=np.int32,
                    )
                else:
                    matrix = mx.array(array)
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


def _pad_numpy_matrix(array, group_size: int):
    import numpy as np

    if group_size == 0:
        return array.astype(np.float32), 0
    original_in_features = int(array.shape[-1])
    pad_features = (-original_in_features) % group_size
    if not pad_features:
        return array.astype(np.float32), 0
    padding = np.zeros((*array.shape[:-1], pad_features), dtype=array.dtype)
    return np.concatenate([array, padding], axis=-1).astype(np.float32), pad_features


def _load_smooth_scales(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import numpy as np

    with np.load(path) as handle:
        return {key: handle[key].astype(np.float32) for key in handle.files}


def _smooth_scale_for_prefix(
    smooth_scales: dict[str, Any] | None,
    prefix: str,
    original_in_features: int,
    padded_in_features: int,
) -> Any | None:
    if smooth_scales is None or prefix not in smooth_scales:
        return None
    import numpy as np

    scale = np.asarray(smooth_scales[prefix], dtype=np.float32)
    if scale.shape != (original_in_features,):
        raise ValueError(
            f"smooth scale for {prefix!r} has shape {scale.shape}, "
            f"expected {(original_in_features,)}"
        )
    if padded_in_features > original_in_features:
        scale = np.pad(
            scale,
            (0, padded_in_features - original_in_features),
            mode="constant",
            constant_values=1.0,
        )
    return scale.astype(np.float32)


def _quantize_cider_w8a8(
    matrix,
    group_size: int,
    clip_percentile: float | None = None,
):
    if group_size == 0:
        try:
            from cider.ops import quantize_weight_int8
        except ImportError as exc:
            raise RuntimeError(
                "cider-w8a8 quantization requires the optional Cider package"
            ) from exc
        qweight, scales = quantize_weight_int8(matrix, clip_percentile=clip_percentile)
        return qweight, scales
    return _symmetric_quantize_pergroup(matrix, group_size)


def _quantize_cider_w4a8(matrix):
    try:
        from cider.ops import pack_weight_int4
    except ImportError as exc:
        raise RuntimeError("cider-w4a8 quantization requires the optional Cider package") from exc
    return pack_weight_int4(matrix.T.copy())


def _symmetric_quantize_pergroup(matrix, group_size: int):
    import numpy as np

    out_features, in_features = matrix.shape
    if in_features % group_size:
        raise ValueError(f"in_features={in_features} is not divisible by group_size={group_size}")
    groups = in_features // group_size
    grouped = matrix.reshape(out_features, groups, group_size)
    group_max = np.max(np.abs(grouped), axis=2)
    scales = group_max / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    qweight = np.clip(
        np.round(grouped / scales[:, :, None]),
        -128,
        127,
    ).astype(np.int8)
    return qweight.reshape(out_features, in_features), scales.astype(np.float32).T.copy()
