from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cradio_mlx.config import get_model_config
from cradio_mlx.imaging import load_rgb_image
from cradio_mlx.outputs import EmbeddingResult

TORCH_DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass(frozen=True)
class PyTorchReferenceConfig:
    model_id: str
    model_config_id: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False


class PyTorchReferenceRunner:
    def __init__(self, config: PyTorchReferenceConfig):
        self.config = config
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
    ) -> PyTorchReferenceRunner:
        return cls(
            PyTorchReferenceConfig(
                model_id=model_id,
                revision=revision,
                dtype=dtype,
                device=device,
            )
        )

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_id: str,
        dtype: str = "bfloat16",
        device: str = "auto",
    ) -> PyTorchReferenceRunner:
        return cls(
            PyTorchReferenceConfig(
                model_id=str(checkpoint_path),
                model_config_id=model_id,
                dtype=dtype,
                device=device,
                local_files_only=True,
            )
        )

    def load(self) -> PyTorchReferenceRunner:
        try:
            import torch
            from transformers import AutoModel, CLIPImageProcessor
        except ImportError as exc:
            msg = "PyTorch reference mode requires `pip install -e .[reference]`"
            raise RuntimeError(msg) from exc

        device = self._resolve_device(torch)
        torch_dtype = self._resolve_dtype(torch)
        kwargs = {"local_files_only": self.config.local_files_only}
        if self.config.revision:
            kwargs["revision"] = self.config.revision

        self._processor = CLIPImageProcessor.from_pretrained(self.config.model_id, **kwargs)
        local_path = Path(self.config.model_id)
        if self.config.local_files_only and local_path.exists():
            self._model = self._load_local_model(local_path, torch_dtype)
        else:
            self._model = AutoModel.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                dtype=torch_dtype,
                **kwargs,
            )
        self._model = self._model.eval().to(device)
        self._device = device
        return self

    def encode(
        self,
        images: str | Path | Any | list[str | Path | Any],
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        try:
            import torch
        except ImportError as exc:
            msg = "PyTorch reference mode requires `pip install -e .[reference]`"
            raise RuntimeError(msg) from exc

        if self._model is None or self._processor is None:
            self.load()

        if isinstance(image_size, int):
            height = width = image_size
        else:
            height, width = image_size

        model_cfg = get_model_config(self.config.model_config_id or self.config.model_id)
        grid_h, grid_w = model_cfg.grid_shape((height, width))
        batch_images = images if isinstance(images, list) else [images]
        pil_images = [load_rgb_image(image) for image in batch_images]

        processor_kwargs = {
            "images": pil_images,
            "return_tensors": "pt",
            "do_resize": True,
            "size": {"height": height, "width": width},
        }
        pixel_values = self._processor(**processor_kwargs).pixel_values
        dtype = self._resolve_dtype(torch)
        pixel_values = pixel_values.to(device=self._device, dtype=dtype)

        with torch.inference_mode():
            summary, spatial = self._model(pixel_values)

        return EmbeddingResult(
            summary=summary.float().cpu().numpy(),
            spatial=spatial.float().cpu().numpy(),
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=model_cfg.patch_size,
            image_size=(height, width),
            metadata={
                "backend": "pytorch",
                "device": str(self._device),
                "accelerated": str(self._device) != "cpu",
                "model_id": self.config.model_config_id or self.config.model_id,
                "checkpoint": self.config.model_id,
                "revision": self.config.revision,
                "dtype": self.config.dtype,
            },
        )

    def _resolve_device(self, torch: Any) -> Any:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_dtype(self, torch: Any) -> Any:
        dtype_name = TORCH_DTYPES.get(self.config.dtype, self.config.dtype)
        try:
            return getattr(torch, dtype_name)
        except AttributeError as exc:
            known = ", ".join(sorted(TORCH_DTYPES))
            msg = f"unsupported torch dtype={self.config.dtype!r}; known: {known}"
            raise ValueError(msg) from exc

    def _load_local_model(self, checkpoint_path: Path, torch_dtype: Any) -> Any:
        import hashlib
        import importlib.util
        import sys
        import types

        package = f"_cradio_local_{hashlib.sha1(str(checkpoint_path).encode()).hexdigest()[:12]}"
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__file__ = str(checkpoint_path / "__init__.py")
            module.__path__ = [str(checkpoint_path)]
            sys.modules[package] = module

        module_name = f"{package}.hf_model"
        if module_name not in sys.modules:
            hf_model_path = checkpoint_path / "hf_model.py"
            spec = importlib.util.spec_from_file_location(module_name, hf_model_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not import local RADIO model from {hf_model_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        model_cls = sys.modules[module_name].RADIOModel
        return model_cls.from_pretrained(
            checkpoint_path,
            dtype=torch_dtype,
            local_files_only=True,
        )
