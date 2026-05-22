from __future__ import annotations

from pathlib import Path

import pytest

from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner
from cradio_mlx.metrics import cosine_similarity
from cradio_mlx.mlx_so400m import MLXHEncoder, MLXSO400MEncoder

ROOT = Path(__file__).resolve().parents[1]
SMOKE_IMAGE = ROOT / "data" / "golden_images" / "smoke.jpg"


@pytest.mark.parametrize(
    ("model_id", "checkpoint", "encoder_cls", "summary_min", "spatial_min"),
    [
        (
            "nvidia/C-RADIOv4-SO400M",
            ROOT / "checkpoints" / "c-radiov4-so400m",
            MLXSO400MEncoder,
            0.999,
            0.998,
        ),
        (
            "nvidia/C-RADIOv4-H",
            ROOT / "checkpoints" / "c-radiov4-h",
            MLXHEncoder,
            0.999,
            0.998,
        ),
    ],
)
def test_local_checkpoint_mlx_matches_pytorch_reference(
    model_id,
    checkpoint,
    encoder_cls,
    summary_min,
    spatial_min,
):
    if not (checkpoint / "model.safetensors").exists():
        pytest.skip(f"local checkpoint is not present: {checkpoint}")

    try:
        reference = PyTorchReferenceRunner.from_local_checkpoint(
            checkpoint,
            model_id=model_id,
            dtype="bfloat16",
            device="auto",
        ).encode(SMOKE_IMAGE, image_size=256)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    candidate = encoder_cls.load(checkpoint, dtype="bfloat16").encode_image(
        SMOKE_IMAGE,
        image_size=256,
    )

    assert cosine_similarity(reference.summary, candidate.summary) >= summary_min
    assert cosine_similarity(reference.spatial, candidate.spatial) >= spatial_min
