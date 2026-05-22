from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from cradio_mlx.compat.pytorch_ref import PyTorchReferenceRunner
from cradio_mlx.outputs import EmbeddingResult

BackendName = Literal["pytorch"]


class CRadioEncoder:
    """User-facing accelerated encoder facade.

    The current production backend is PyTorch on MPS/CPU/CUDA. The standalone MLX backend
    remains the next implementation target, but this facade gives callers a stable API now.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        backend: BackendName = "pytorch",
        dtype: str = "bfloat16",
        device: str = "auto",
    ):
        if backend != "pytorch":
            raise ValueError("only the `pytorch` backend is implemented right now")
        self.model_id = model_id
        self.revision = revision
        self.backend = backend
        self.dtype = dtype
        self.device = device
        self._runner = PyTorchReferenceRunner.from_model_id(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str | None = None,
        backend: BackendName = "pytorch",
        dtype: str = "bfloat16",
        device: str = "auto",
    ) -> CRadioEncoder:
        return cls(model_id, revision=revision, backend=backend, dtype=dtype, device=device)

    def load(self) -> CRadioEncoder:
        self._runner.load()
        return self

    def encode_image(
        self,
        image: str | Path | Any,
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        return self._runner.encode(image, image_size=image_size)

    def encode_batch(
        self,
        images: list[str | Path | Any],
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        return self._runner.encode(images, image_size=image_size)
