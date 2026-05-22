from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingComparison:
    summary_cosine: float
    spatial_cosine: float
    summary_shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_cosine": self.summary_cosine,
            "spatial_cosine": self.spatial_cosine,
            "summary_shape": self.summary_shape,
            "spatial_shape": self.spatial_shape,
        }


def cosine_similarity(a: Any, b: Any) -> float:
    import numpy as np

    left = np.asarray(a, dtype="float64").reshape(-1)
    right = np.asarray(b, dtype="float64").reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")

    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return float(np.dot(left, right) / denom)


def compare_embedding_npz(reference: str | Path, candidate: str | Path) -> EmbeddingComparison:
    import numpy as np

    with np.load(reference) as ref, np.load(candidate) as cand:
        ref_summary = ref["summary"]
        cand_summary = cand["summary"]
        ref_spatial = ref["spatial"]
        cand_spatial = cand["spatial"]

    if ref_summary.shape != cand_summary.shape:
        raise ValueError(f"summary shape mismatch: {ref_summary.shape} != {cand_summary.shape}")
    if ref_spatial.shape != cand_spatial.shape:
        raise ValueError(f"spatial shape mismatch: {ref_spatial.shape} != {cand_spatial.shape}")

    return EmbeddingComparison(
        summary_cosine=cosine_similarity(ref_summary, cand_summary),
        spatial_cosine=cosine_similarity(ref_spatial, cand_spatial),
        summary_shape=tuple(int(dim) for dim in ref_summary.shape),
        spatial_shape=tuple(int(dim) for dim in ref_spatial.shape),
    )


def write_embedding_comparison(
    reference: str | Path,
    candidate: str | Path,
    out: str | Path,
) -> Path:
    comparison = compare_embedding_npz(reference, candidate)
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
