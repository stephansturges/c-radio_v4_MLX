from __future__ import annotations

from pathlib import Path
from typing import Any

from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.outputs import EmbeddingResult


class CRadioMLX:
    def __init__(self, bundle_path: str | Path, manifest: BundleManifest):
        self.bundle_path = Path(bundle_path)
        self.manifest = manifest

    @classmethod
    def load(cls, path: str | Path) -> CRadioMLX:
        bundle_path = Path(path)
        manifest = BundleManifest.load(bundle_path)
        return cls(bundle_path, manifest)

    def encode_image(
        self,
        image: str | Path | Any,
        image_size: int | tuple[int, int] = 512,
        return_summary: bool = True,
        return_spatial: bool = True,
    ) -> EmbeddingResult:
        del image, image_size, return_summary, return_spatial
        raise NotImplementedError(
            "MLX C-RADIOv4 forward pass is not implemented yet. "
            "Use `cradio-mlx pytorch-ref` for the reference backend while the MLX "
            "architecture and weight mapping are being built."
        )

    def encode_batch(
        self,
        images: list[str | Path | Any],
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        del images, image_size
        raise NotImplementedError(
            "MLX C-RADIOv4 batch encoding is waiting on the model implementation."
        )

    def spatial_as_grid(self, result: EmbeddingResult) -> Any:
        return result.spatial_as_grid()
