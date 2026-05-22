from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelVariant = Literal["so400m", "h"]


@dataclass(frozen=True)
class CRadioConfig:
    model_id: str
    variant: ModelVariant
    patch_size: int = 16
    min_resolution: int = 256
    preferred_resolution: int = 512
    max_resolution: int = 2048
    parameter_count: int | None = None

    def validate_image_size(self, image_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(image_size, int):
            height = width = image_size
        else:
            height, width = image_size

        for label, value in (("height", height), ("width", width)):
            if value < self.min_resolution or value > self.max_resolution:
                msg = (
                    f"{label}={value} is outside the supported range "
                    f"{self.min_resolution}-{self.max_resolution}"
                )
                raise ValueError(msg)
            if value % self.patch_size != 0:
                raise ValueError(f"{label}={value} must be a multiple of {self.patch_size}")

        return height, width

    def grid_shape(self, image_size: int | tuple[int, int]) -> tuple[int, int]:
        height, width = self.validate_image_size(image_size)
        return height // self.patch_size, width // self.patch_size

    def spatial_tokens(self, image_size: int | tuple[int, int]) -> int:
        grid_h, grid_w = self.grid_shape(image_size)
        return grid_h * grid_w


MODEL_PRESETS: dict[str, CRadioConfig] = {
    "nvidia/C-RADIOv4-SO400M": CRadioConfig(
        model_id="nvidia/C-RADIOv4-SO400M",
        variant="so400m",
        parameter_count=400_000_000,
    ),
    "nvidia/C-RADIOv4-H": CRadioConfig(
        model_id="nvidia/C-RADIOv4-H",
        variant="h",
        parameter_count=653_000_000,
    ),
}


def get_model_config(model_id: str) -> CRadioConfig:
    try:
        return MODEL_PRESETS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"unsupported C-RADIOv4 model_id={model_id!r}; known: {known}") from exc
