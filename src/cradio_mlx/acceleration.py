from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AccelerationReport:
    mlx_available: bool
    mlx_default_device: str | None
    mlx_gpu: bool
    torch_available: bool
    torch_mps_available: bool | None
    torch_cuda_available: bool | None
    preferred_torch_device: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_acceleration() -> AccelerationReport:
    mlx_available = False
    mlx_default_device: str | None = None
    mlx_gpu = False

    try:
        import mlx.core as mx

        mlx_available = True
        default_device = mx.default_device()
        mlx_default_device = str(default_device)
        mlx_gpu = "gpu" in mlx_default_device.lower()
    except Exception:
        pass

    torch_available = False
    torch_mps_available: bool | None = None
    torch_cuda_available: bool | None = None
    preferred_torch_device: str | None = None

    try:
        import torch

        torch_available = True
        torch_mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        torch_cuda_available = bool(torch.cuda.is_available())
        if torch_mps_available:
            preferred_torch_device = "mps"
        elif torch_cuda_available:
            preferred_torch_device = "cuda"
        else:
            preferred_torch_device = "cpu"
    except Exception:
        pass

    return AccelerationReport(
        mlx_available=mlx_available,
        mlx_default_device=mlx_default_device,
        mlx_gpu=mlx_gpu,
        torch_available=torch_available,
        torch_mps_available=torch_mps_available,
        torch_cuda_available=torch_cuda_available,
        preferred_torch_device=preferred_torch_device,
    )
