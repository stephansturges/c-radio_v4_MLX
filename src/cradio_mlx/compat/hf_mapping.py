from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorRecord:
    file: str
    key: str
    shape: tuple[int, ...]
    dtype: str


def audit_safetensors(hf_path: str | Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("weight audit requires `safetensors`") from exc

    root = Path(hf_path)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    records: list[TensorRecord] = []
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="numpy") as handle:
            for key in sorted(handle.keys()):
                tensor = handle.get_tensor(key)
                records.append(
                    TensorRecord(
                        file=path.name,
                        key=key,
                        shape=tuple(int(dim) for dim in tensor.shape),
                        dtype=str(tensor.dtype),
                    )
                )

    return {
        "path": str(root),
        "safetensors_files": sorted(path.name for path in root.glob("*.safetensors")),
        "tensor_count": len(records),
        "records": [asdict(record) for record in records],
    }


def write_weight_audit(hf_path: str | Path, out: str | Path) -> Path:
    payload = audit_safetensors(hf_path)
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
