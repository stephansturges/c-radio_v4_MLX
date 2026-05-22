from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.config import get_model_config

METADATA_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
)


@dataclass(frozen=True)
class ConversionRequest:
    hf_path: Path
    mlx_path: Path
    model_id: str
    revision: str
    dtype: str = "bfloat16"
    manifest_only: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_metadata_files(hf_path: Path, mlx_path: Path) -> dict[str, str]:
    source_files: dict[str, str] = {}
    for filename in METADATA_FILES:
        source = hf_path / filename
        if not source.exists() or not source.is_file():
            continue
        target = mlx_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_files[filename] = sha256_file(target)
    return source_files


def convert_checkpoint(request: ConversionRequest) -> Path:
    if not request.hf_path.exists():
        raise FileNotFoundError(request.hf_path)
    if not request.hf_path.is_dir():
        raise NotADirectoryError(request.hf_path)

    model_cfg = get_model_config(request.model_id)
    request.mlx_path.mkdir(parents=True, exist_ok=True)
    source_files = copy_metadata_files(request.hf_path, request.mlx_path)
    conversion_state = "manifest_only"

    if not request.manifest_only:
        source_weights = request.hf_path / "model.safetensors"
        if not source_weights.exists():
            raise FileNotFoundError(source_weights)
        target_weights = request.mlx_path / "model.safetensors"
        shutil.copy2(source_weights, target_weights)
        source_files["model.safetensors"] = sha256_file(target_weights)
        conversion_state = "self_contained"

    manifest = BundleManifest(
        model_id=request.model_id,
        revision=request.revision,
        variant=model_cfg.variant,
        dtype=request.dtype,
        patch_size=model_cfg.patch_size,
        preferred_resolution=model_cfg.preferred_resolution,
        max_resolution=model_cfg.max_resolution,
        source_files=source_files,
        license="nvidia-open-model-license",
        extra={
            "conversion_state": conversion_state,
            "source_hf_path": str(request.hf_path),
            "weights_file": "model.safetensors" if not request.manifest_only else None,
        },
    )
    return manifest.save(request.mlx_path)
