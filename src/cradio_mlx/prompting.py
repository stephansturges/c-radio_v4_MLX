from __future__ import annotations

from typing import Any

from cradio_mlx.outputs import EmbeddingResult


def package_vlm_prompt(
    prompt: str,
    image_embeddings: EmbeddingResult,
    projector: str | None = None,
) -> dict[str, Any]:
    summary_shape = tuple(getattr(image_embeddings.summary, "shape", ()))
    spatial_shape = tuple(getattr(image_embeddings.spatial, "shape", ()))
    return {
        "prompt": prompt,
        "projector": projector,
        "embedding_contract": {
            "summary_shape": summary_shape,
            "spatial_shape": spatial_shape,
            "grid_h": image_embeddings.grid_h,
            "grid_w": image_embeddings.grid_w,
            "patch_size": image_embeddings.patch_size,
            "image_size": image_embeddings.image_size,
        },
        "metadata": image_embeddings.metadata,
    }
