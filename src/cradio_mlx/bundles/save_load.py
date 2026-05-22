from __future__ import annotations

from pathlib import Path
from typing import Any

from cradio_mlx.bundles.manifest import BundleManifest


def inspect_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    if not bundle_path.is_dir():
        raise NotADirectoryError(bundle_path)

    manifest_path = bundle_path / "manifest.json"
    manifest = BundleManifest.load(manifest_path).to_dict() if manifest_path.exists() else None
    files = sorted(
        p.relative_to(bundle_path).as_posix()
        for p in bundle_path.rglob("*")
        if p.is_file()
    )

    return {
        "path": str(bundle_path),
        "has_manifest": manifest is not None,
        "manifest": manifest,
        "files": files,
    }
