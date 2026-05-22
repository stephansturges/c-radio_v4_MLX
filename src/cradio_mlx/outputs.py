from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingResult:
    summary: Any
    spatial: Any
    grid_h: int
    grid_w: int
    patch_size: int
    image_size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def spatial_as_grid(self) -> Any:
        return spatial_as_grid(self.spatial, self.grid_h, self.grid_w)


def spatial_as_grid(spatial: Any, grid_h: int, grid_w: int) -> Any:
    """Convert spatial embeddings from (B, T, D) to (B, D, H, W)."""
    import numpy as np

    arr = np.asarray(spatial)
    if arr.ndim != 3:
        raise ValueError(f"spatial embeddings must have rank 3, got shape={arr.shape}")

    batch, tokens, dim = arr.shape
    expected_tokens = grid_h * grid_w
    if tokens != expected_tokens:
        raise ValueError(
            f"spatial token count {tokens} does not match grid {grid_h}x{grid_w}="
            f"{expected_tokens}"
        )

    return arr.reshape(batch, grid_h, grid_w, dim).transpose(0, 3, 1, 2)
