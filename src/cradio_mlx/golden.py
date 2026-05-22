from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner


@dataclass(frozen=True)
class ReferenceCaptureRequest:
    model_id: str
    image: Path
    out: Path
    revision: str | None = None
    image_size: int | tuple[int, int] = 512
    dtype: str = "bfloat16"
    device: str = "auto"


def capture_reference_outputs(request: ReferenceCaptureRequest) -> dict[str, Path]:
    import numpy as np

    result = PyTorchReferenceRunner.from_model_id(
        request.model_id,
        revision=request.revision,
        dtype=request.dtype,
        device=request.device,
    ).encode(request.image, image_size=request.image_size)

    request.out.mkdir(parents=True, exist_ok=True)
    stem = request.image.stem
    tensors_path = request.out / f"{stem}.npz"
    metadata_path = request.out / f"{stem}.json"

    np.savez_compressed(tensors_path, summary=result.summary, spatial=result.spatial)
    metadata = {
        "model_id": request.model_id,
        "revision": request.revision,
        "image": str(request.image),
        "image_size": result.image_size,
        "grid_h": result.grid_h,
        "grid_w": result.grid_w,
        "patch_size": result.patch_size,
        "summary_shape": tuple(result.summary.shape),
        "spatial_shape": tuple(result.spatial.shape),
        "backend": result.metadata,
    }
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    metadata_path.write_text(metadata_text, encoding="utf-8")

    return {
        "tensors": tensors_path,
        "metadata": metadata_path,
    }


def save_embedding_result_npz(result, path: str | Path) -> Path:
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, summary=result.summary, spatial=result.spatial)
    return output_path
