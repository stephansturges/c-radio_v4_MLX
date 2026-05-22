from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(Path(image)).convert("RGB")


def resolve_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    return image_size


def resize_image(image: Image.Image, image_size: int | tuple[int, int]) -> Image.Image:
    height, width = resolve_image_size(image_size)
    return image.resize((width, height), Image.Resampling.BICUBIC)


def preprocess_clip_numpy(
    image: str | Path | Image.Image,
    image_size: int | tuple[int, int] = 512,
    mean: tuple[float, float, float] = CLIP_MEAN,
    std: tuple[float, float, float] = CLIP_STD,
) -> Any:
    """Return a CLIP-style float32 tensor with shape (1, 3, H, W)."""
    import numpy as np

    pil_image = resize_image(load_rgb_image(image), image_size)
    arr = np.asarray(pil_image).astype("float32") / 255.0
    arr = (arr - np.asarray(mean, dtype="float32")) / np.asarray(std, dtype="float32")
    chw = arr.transpose(2, 0, 1)
    return chw[None, ...]
