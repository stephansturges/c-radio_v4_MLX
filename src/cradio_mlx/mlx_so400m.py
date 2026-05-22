from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.core.fast as mx_fast

from cradio_mlx.imaging import load_rgb_image, resize_image
from cradio_mlx.outputs import EmbeddingResult

SO400M_EMBED_DIM = 1152
SO400M_DEPTH = 27
SO400M_NUM_HEADS = 16
SO400M_PATCH_SIZE = 16
SO400M_NUM_CLS_TOKENS = 3
SO400M_NUM_PREFIX_TOKENS = 10
SO400M_SUMMARY_IDXS = (0, 1)
SO400M_MAX_GRID = 128
H_EMBED_DIM = 1280
H_DEPTH = 32
H_NUM_HEADS = 16
H_PATCH_SIZE = 16
H_NUM_CLS_TOKENS = 3
H_NUM_PREFIX_TOKENS = 10
H_SUMMARY_IDXS = (0, 1)
H_MAX_GRID = 128


@dataclass(frozen=True)
class MLXRadioSpec:
    model_id: str
    revision: str
    variant: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int
    num_cls_tokens: int
    num_prefix_tokens: int
    summary_idxs: tuple[int, ...]
    max_grid: int

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


SO400M_SPEC = MLXRadioSpec(
    model_id="nvidia/C-RADIOv4-SO400M",
    revision="c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
    variant="so400m",
    embed_dim=SO400M_EMBED_DIM,
    depth=SO400M_DEPTH,
    num_heads=SO400M_NUM_HEADS,
    patch_size=SO400M_PATCH_SIZE,
    num_cls_tokens=SO400M_NUM_CLS_TOKENS,
    num_prefix_tokens=SO400M_NUM_PREFIX_TOKENS,
    summary_idxs=SO400M_SUMMARY_IDXS,
    max_grid=SO400M_MAX_GRID,
)

H_SPEC = MLXRadioSpec(
    model_id="nvidia/C-RADIOv4-H",
    revision="0057b339059c0b9e1b4ba996f975410ebbfdfcc8",
    variant="h",
    embed_dim=H_EMBED_DIM,
    depth=H_DEPTH,
    num_heads=H_NUM_HEADS,
    patch_size=H_PATCH_SIZE,
    num_cls_tokens=H_NUM_CLS_TOKENS,
    num_prefix_tokens=H_NUM_PREFIX_TOKENS,
    summary_idxs=H_SUMMARY_IDXS,
    max_grid=H_MAX_GRID,
)

VARIANT_SPECS = {
    "so400m": SO400M_SPEC,
    "h": H_SPEC,
}
QUANTIZATION_CODE_MODES = {
    0: "affine",
    1: "mxfp4",
    2: "mxfp8",
    3: "nvfp4",
}
QUANTIZED_RUNTIME_MODES = {"packed", "dequantize"}
QUANTIZED_WEIGHT_SUFFIXES = (
    ".qweight",
    ".qscales",
    ".qbiases",
    ".qbits",
    ".qgroup_size",
    ".qmode_code",
    ".qin_features",
    ".qpadded_in_features",
)


@dataclass(frozen=True)
class MLXSO400MConfig:
    model_id: str = "nvidia/C-RADIOv4-SO400M"
    revision: str = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
    dtype: str = "float32"
    patch_size: int = SO400M_PATCH_SIZE
    variant: str = "so400m"
    quantized_runtime: str = "packed"


class MLXRadioEncoder:
    """Native MLX C-RADIOv4 forward path for audited checkpoint variants.

    This implements the common CPE ViT path used by SO400M and H. The variant spec is
    derived from the pinned official Hugging Face checkpoints.
    """

    def __init__(self, weights: dict[str, mx.array], config: MLXSO400MConfig, spec: MLXRadioSpec):
        self.weights = weights
        self.config = config
        self.spec = spec
        self.dtype = _mlx_dtype(config.dtype)
        self._pos_cache: dict[tuple[int, int], mx.array] = {}

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        variant: str = "so400m",
        quantized_runtime: str = "packed",
    ) -> MLXRadioEncoder:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("MLX C-RADIOv4 loading requires safetensors") from exc

        try:
            spec = VARIANT_SPECS[variant]
        except KeyError as exc:
            known = ", ".join(sorted(VARIANT_SPECS))
            raise ValueError(f"unknown C-RADIOv4 MLX variant={variant!r}; known: {known}") from exc

        revision = revision or spec.revision
        quantized_runtime = _validate_quantized_runtime(quantized_runtime)

        root = Path(checkpoint_path)
        model_path = root / "model.safetensors" if root.is_dir() else root
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        target_dtype = _mlx_dtype(dtype)
        weights: dict[str, mx.array] = {}
        with safe_open(model_path, framework="numpy") as handle:
            for key in handle.keys():
                array = mx.array(handle.get_tensor(key))
                if _should_cast_loaded_weight(key):
                    array = array.astype(target_dtype)
                weights[key] = array
        has_quantized_weights = any(key.endswith(".qweight") for key in weights)
        if quantized_runtime == "dequantize":
            weights = _dequantize_loaded_weights(weights, target_dtype)
        mx.eval(list(weights.values()))
        config = MLXSO400MConfig(
            model_id=spec.model_id,
            revision=revision,
            dtype=dtype,
            patch_size=spec.patch_size,
            variant=spec.variant,
            quantized_runtime=quantized_runtime if has_quantized_weights else "dense",
        )
        return cls(weights, config, spec)

    def encode_image(
        self,
        image: str | Path | Any,
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        pixel_values = _load_rescaled_image(image, image_size)
        summary, spatial = self.forward(pixel_values)
        mx.eval(summary, spatial)

        if isinstance(image_size, int):
            height = width = image_size
        else:
            height, width = image_size

        return EmbeddingResult(
            summary=_to_numpy(summary),
            spatial=_to_numpy(spatial),
            grid_h=height // self.spec.patch_size,
            grid_w=width // self.spec.patch_size,
            patch_size=self.spec.patch_size,
            image_size=(height, width),
            metadata={
                "backend": "mlx",
                "device": str(mx.default_device()),
                "accelerated": "gpu" in str(mx.default_device()).lower(),
                "model_id": self.config.model_id,
                "revision": self.config.revision,
                "dtype": self.config.dtype,
                "variant": self.config.variant,
                "quantized_runtime": self.config.quantized_runtime,
            },
        )

    def encode_batch(
        self,
        images: list[str | Path | Any],
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        pixel_values = _load_rescaled_images(images, image_size)
        summary, spatial = self.forward(pixel_values)
        mx.eval(summary, spatial)

        if isinstance(image_size, int):
            height = width = image_size
        else:
            height, width = image_size

        return EmbeddingResult(
            summary=_to_numpy(summary),
            spatial=_to_numpy(spatial),
            grid_h=height // self.spec.patch_size,
            grid_w=width // self.spec.patch_size,
            patch_size=self.spec.patch_size,
            image_size=(height, width),
            metadata={
                "backend": "mlx",
                "device": str(mx.default_device()),
                "accelerated": "gpu" in str(mx.default_device()).lower(),
                "model_id": self.config.model_id,
                "revision": self.config.revision,
                "dtype": self.config.dtype,
                "variant": self.config.variant,
                "quantized_runtime": self.config.quantized_runtime,
                "batch_size": len(images),
            },
        )

    def forward(self, pixel_values: mx.array) -> tuple[mx.array, mx.array]:
        x = pixel_values.astype(self.dtype)
        mean = self.weights["radio_model.input_conditioner.norm_mean"].astype(self.dtype)
        std = self.weights["radio_model.input_conditioner.norm_std"].astype(self.dtype)
        x = (x - mean) / std

        x = self._patch_generator(x)
        for index in range(self.spec.depth):
            x = self._block(x, index)

        all_summary = x[:, : self.spec.num_cls_tokens, :]
        summary_parts = [all_summary[:, idx, :] for idx in self.spec.summary_idxs]
        summary = mx.concatenate(summary_parts, axis=-1)
        spatial = x[:, self.spec.num_prefix_tokens :, :]
        return summary, spatial

    def _patch_generator(self, x: mx.array) -> mx.array:
        patches = _im2patches(x, self.spec.patch_size)
        patches = _linear(
            patches,
            self.weights,
            "radio_model.model.patch_generator.embedder",
        )

        _, _, height, width = x.shape
        pos_embed = self._pos_embed(height // self.spec.patch_size, width // self.spec.patch_size)
        patches = patches + pos_embed.astype(self.dtype)

        token = self.weights["radio_model.model.patch_generator.cls_token.token"].astype(self.dtype)
        token = mx.broadcast_to(
            token[None, :, :],
            (patches.shape[0], token.shape[0], token.shape[1]),
        )
        return mx.concatenate([token, patches], axis=1)

    def _pos_embed(self, grid_h: int, grid_w: int) -> mx.array:
        cache_key = (grid_h, grid_w)
        cached = self._pos_cache.get(cache_key)
        if cached is not None:
            return cached

        pos = self.weights["radio_model.model.patch_generator.pos_embed"]
        if grid_h == self.spec.max_grid and grid_w == self.spec.max_grid:
            self._pos_cache[cache_key] = pos
            return pos

        pos_np = _to_numpy(pos).reshape(
            1,
            self.spec.max_grid,
            self.spec.max_grid,
            self.spec.embed_dim,
        )
        resized = _resize_pos_embed_align_corners_false(pos_np, grid_h, grid_w)
        resized = resized.reshape(1, grid_h * grid_w, self.spec.embed_dim)
        out = mx.array(resized).astype(self.dtype)
        mx.eval(out)
        self._pos_cache[cache_key] = out
        return out

    def _block(self, x: mx.array, index: int) -> mx.array:
        prefix = f"radio_model.model.blocks.{index}"
        norm1 = _layer_norm(
            x,
            self.weights[f"{prefix}.norm1.weight"],
            self.weights[f"{prefix}.norm1.bias"],
        )
        x = x + self._attention(norm1, prefix)
        norm2 = _layer_norm(
            x,
            self.weights[f"{prefix}.norm2.weight"],
            self.weights[f"{prefix}.norm2.bias"],
        )
        x = x + self._mlp(norm2, prefix)
        return x

    def _attention(self, x: mx.array, prefix: str) -> mx.array:
        batch, tokens, _ = x.shape
        qkv = _linear(
            x,
            self.weights,
            f"{prefix}.attn.qkv",
        )
        qkv = mx.reshape(qkv, (batch, tokens, 3, self.spec.num_heads, self.spec.head_dim))
        q = mx.transpose(qkv[:, :, 0, :, :], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1, :, :], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2, :, :], (0, 2, 1, 3))

        x = mx_fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.spec.head_dim**-0.5,
        )
        x = mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (batch, tokens, self.spec.embed_dim))
        return _linear(
            x,
            self.weights,
            f"{prefix}.attn.proj",
        )

    def _mlp(self, x: mx.array, prefix: str) -> mx.array:
        x = _linear(
            x,
            self.weights,
            f"{prefix}.mlp.fc1",
        )
        x = _gelu_exact(x)
        return _linear(
            x,
            self.weights,
            f"{prefix}.mlp.fc2",
        )


class MLXSO400MEncoder(MLXRadioEncoder):
    """Native MLX C-RADIOv4-SO400M forward path."""

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        quantized_runtime: str = "packed",
    ) -> MLXSO400MEncoder:
        encoder = MLXRadioEncoder.load(
            checkpoint_path,
            dtype=dtype,
            revision=revision,
            variant="so400m",
            quantized_runtime=quantized_runtime,
        )
        return cls(encoder.weights, encoder.config, encoder.spec)


class MLXHEncoder(MLXRadioEncoder):
    """Native MLX C-RADIOv4-H forward path."""

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        quantized_runtime: str = "packed",
    ) -> MLXHEncoder:
        encoder = MLXRadioEncoder.load(
            checkpoint_path,
            dtype=dtype,
            revision=revision,
            variant="h",
            quantized_runtime=quantized_runtime,
        )
        return cls(encoder.weights, encoder.config, encoder.spec)


def _linear(x: mx.array, weights: dict[str, mx.array], prefix: str) -> mx.array:
    qweight = weights.get(f"{prefix}.qweight")
    if qweight is not None:
        group_size = int(_scalar_item(weights[f"{prefix}.qgroup_size"]))
        padded_meta = weights.get(f"{prefix}.qpadded_in_features")
        padded_in_features = (
            int(_scalar_item(padded_meta))
            if padded_meta is not None
            else int(weights[f"{prefix}.qscales"].shape[-1]) * group_size
        )
        mode_code = weights.get(f"{prefix}.qmode_code")
        mode = (
            QUANTIZATION_CODE_MODES[int(_scalar_item(mode_code))]
            if mode_code is not None
            else "affine"
        )
        if x.shape[-1] < padded_in_features:
            pad_width = padded_in_features - x.shape[-1]
            padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
            x = mx.concatenate([x, padding], axis=-1)
        y = mx.quantized_matmul(
            x,
            qweight,
            weights[f"{prefix}.qscales"],
            weights.get(f"{prefix}.qbiases"),
            group_size=group_size,
            bits=int(_scalar_item(weights[f"{prefix}.qbits"])),
            mode=mode,
        )
        bias = weights.get(f"{prefix}.bias")
        if bias is not None:
            y = y + bias
        return y

    weight = weights[f"{prefix}.weight"]
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        return mx.addmm(bias, x, mx.transpose(weight))
    return x @ mx.transpose(weight)


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6) -> mx.array:
    return mx_fast.layer_norm(x, weight.astype(x.dtype), bias.astype(x.dtype), eps)


def _gelu_exact(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / (2.0**0.5)))


def _im2patches(x: mx.array, patch_size: int) -> mx.array:
    batch, channels, height, width = x.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    x = mx.reshape(x, (batch, channels, grid_h, patch_size, grid_w, patch_size))
    x = mx.transpose(x, (0, 2, 4, 1, 3, 5))
    return mx.reshape(x, (batch, grid_h * grid_w, channels * patch_size * patch_size))


def _load_rescaled_image(image: str | Path | Any, image_size: int | tuple[int, int]) -> mx.array:
    return _load_rescaled_images([image], image_size)


def _load_rescaled_images(
    images: list[str | Path | Any],
    image_size: int | tuple[int, int],
) -> mx.array:
    import numpy as np

    arrays = []
    for image in images:
        pil_image = resize_image(load_rgb_image(image), image_size)
        arr = np.asarray(pil_image).astype("float32") / 255.0
        arrays.append(arr.transpose(2, 0, 1))
    return mx.array(np.stack(arrays, axis=0))


def _resize_pos_embed_align_corners_false(pos: Any, out_h: int, out_w: int) -> Any:
    import numpy as np

    _, in_h, in_w, _ = pos.shape
    y0, y1, wy = _resize_indices(in_h, out_h)
    x0, x1, wx = _resize_indices(in_w, out_w)

    top = pos[:, y0, :, :] * (1.0 - wy)[None, :, None, None] + pos[:, y1, :, :] * wy[
        None, :, None, None
    ]
    out = top[:, :, x0, :] * (1.0 - wx)[None, None, :, None] + top[:, :, x1, :] * wx[
        None, None, :, None
    ]
    return out.astype(np.float32)


def _resize_indices(in_size: int, out_size: int) -> tuple[Any, Any, Any]:
    import numpy as np

    scale = in_size / out_size
    real = (np.arange(out_size, dtype=np.float32) + 0.5) * scale - 0.5
    real = np.maximum(real, 0.0)
    idx0 = np.floor(real).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, in_size - 1)
    weight = real - idx0
    return idx0, idx1, weight.astype(np.float32)


def _mlx_dtype(name: str) -> Any:
    if name in {"float32", "fp32"}:
        return mx.float32
    if name in {"bfloat16", "bf16"}:
        return mx.bfloat16
    if name in {"float16", "fp16"}:
        return mx.float16
    raise ValueError(f"unsupported MLX dtype={name!r}")


def _to_numpy(array: mx.array) -> Any:
    import numpy as np

    return np.asarray(array.astype(mx.float32))


def _should_cast_loaded_weight(key: str) -> bool:
    if key == "radio_model.summary_idxs" or "input_conditioner" in key:
        return False
    if key.endswith(QUANTIZED_WEIGHT_SUFFIXES):
        return False
    return True


def _validate_quantized_runtime(value: str) -> str:
    if value not in QUANTIZED_RUNTIME_MODES:
        known = ", ".join(sorted(QUANTIZED_RUNTIME_MODES))
        raise ValueError(f"unsupported quantized_runtime={value!r}; known: {known}")
    return value


def _dequantize_loaded_weights(
    weights: dict[str, mx.array],
    target_dtype: Any,
) -> dict[str, mx.array]:
    """Expand packed quantized linear weights once so dense MLX kernels handle inference."""
    out = dict(weights)
    prefixes = sorted(key[: -len(".qweight")] for key in weights if key.endswith(".qweight"))
    dense_weights: list[mx.array] = []

    for prefix in prefixes:
        mode_code = weights.get(f"{prefix}.qmode_code")
        mode = (
            QUANTIZATION_CODE_MODES[int(_scalar_item(mode_code))]
            if mode_code is not None
            else "affine"
        )
        dense = mx.dequantize(
            weights[f"{prefix}.qweight"],
            weights[f"{prefix}.qscales"],
            weights.get(f"{prefix}.qbiases"),
            group_size=int(_scalar_item(weights[f"{prefix}.qgroup_size"])),
            bits=int(_scalar_item(weights[f"{prefix}.qbits"])),
            mode=mode,
            dtype=target_dtype,
        )

        in_features_meta = weights.get(f"{prefix}.qin_features")
        if in_features_meta is not None:
            in_features = int(_scalar_item(in_features_meta))
            if dense.shape[-1] != in_features:
                dense = dense[..., :in_features]

        dense = dense.astype(target_dtype)
        out[f"{prefix}.weight"] = dense
        dense_weights.append(dense)
        for suffix in QUANTIZED_WEIGHT_SUFFIXES:
            out.pop(f"{prefix}{suffix}", None)

    if dense_weights:
        mx.eval(*dense_weights)
    return out


def _scalar_item(value: mx.array) -> int | float:
    return value.item() if hasattr(value, "item") else value
