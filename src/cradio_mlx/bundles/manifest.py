from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class BundleManifest:
    model_id: str
    revision: str
    variant: str
    dtype: str
    patch_size: int = 16
    preferred_resolution: int = 512
    max_resolution: int = 2048
    quantization: dict[str, Any] | None = None
    source_files: dict[str, str] = field(default_factory=dict)
    license: str | None = None
    created_at: str = field(default_factory=_now_utc)
    manifest_version: int = MANIFEST_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> BundleManifest:
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix != ".json":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "manifest.json"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output_path
