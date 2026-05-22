from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
CIDER_FUSION_MODES = {"off", "auto", "required"}
CIDER_FUSION_TARGETS = {"ln", "mlp", "ln+mlp"}


@dataclass(frozen=True)
class MLXSO400MConfig:
    model_id: str = "nvidia/C-RADIOv4-SO400M"
    revision: str = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
    dtype: str = "float32"
    patch_size: int = SO400M_PATCH_SIZE
    variant: str = "so400m"
    cider_fusion: str = "off"
    cider_fusion_targets: str = "ln"


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
        self.cider_fusion = _validate_cider_fusion_mode(config.cider_fusion)
        self.cider_fusion_targets = _validate_cider_fusion_targets(
            config.cider_fusion_targets
        )
        self._pos_cache: dict[tuple[int, int], mx.array] = {}
        self._compiled_forward: Any | None = None

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        variant: str = "so400m",
        compile_forward: bool = False,
        cider_fusion: str = "off",
        cider_fusion_targets: str = "ln",
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
        mx.eval(list(weights.values()))
        config = MLXSO400MConfig(
            model_id=spec.model_id,
            revision=revision,
            dtype=dtype,
            patch_size=spec.patch_size,
            variant=spec.variant,
            cider_fusion=_validate_cider_fusion_mode(cider_fusion),
            cider_fusion_targets=_validate_cider_fusion_targets(cider_fusion_targets),
        )
        encoder = cls(weights, config, spec)
        if compile_forward:
            encoder.compile_forward()
        return encoder

    def compile_forward(self) -> None:
        self._compiled_forward = mx.compile(self.forward)

    def _run_forward(self, pixel_values: mx.array) -> tuple[mx.array, mx.array]:
        forward = self._compiled_forward or self.forward
        return forward(pixel_values)

    def encode_image(
        self,
        image: str | Path | Any,
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        pixel_values = _load_rescaled_image(image, image_size)
        summary, spatial = self._run_forward(pixel_values)
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
                "cider_fusion": self.cider_fusion,
                "cider_fusion_targets": self.cider_fusion_targets,
            },
        )

    def encode_batch(
        self,
        images: list[str | Path | Any],
        image_size: int | tuple[int, int] = 512,
    ) -> EmbeddingResult:
        pixel_values = _load_rescaled_images(images, image_size)
        summary, spatial = self._run_forward(pixel_values)
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
                "batch_size": len(images),
                "cider_fusion": self.cider_fusion,
                "cider_fusion_targets": self.cider_fusion_targets,
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
        qkv = None
        if self._uses_cider_fusion_target("ln"):
            qkv = self._fused_layer_norm_linear(
                x,
                self.weights[f"{prefix}.norm1.weight"],
                self.weights[f"{prefix}.norm1.bias"],
                f"{prefix}.attn.qkv",
            )
        if qkv is None:
            norm1 = _layer_norm(
                x,
                self.weights[f"{prefix}.norm1.weight"],
                self.weights[f"{prefix}.norm1.bias"],
            )
            x = x + self._attention(norm1, prefix)
        else:
            x = x + self._attention_from_qkv(qkv, prefix)

        fc1 = None
        if self._uses_cider_fusion_target("ln"):
            fc1 = self._fused_layer_norm_linear(
                x,
                self.weights[f"{prefix}.norm2.weight"],
                self.weights[f"{prefix}.norm2.bias"],
                f"{prefix}.mlp.fc1",
            )
        if fc1 is None:
            norm2 = _layer_norm(
                x,
                self.weights[f"{prefix}.norm2.weight"],
                self.weights[f"{prefix}.norm2.bias"],
            )
            x = x + self._mlp(norm2, prefix)
        else:
            x = x + self._mlp_from_fc1(fc1, prefix)
        return x

    def _attention(self, x: mx.array, prefix: str) -> mx.array:
        qkv = _linear(
            x,
            self.weights,
            f"{prefix}.attn.qkv",
        )
        return self._attention_from_qkv(qkv, prefix)

    def _attention_from_qkv(self, qkv: mx.array, prefix: str) -> mx.array:
        batch, tokens, _ = qkv.shape
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
        return self._mlp_from_fc1(x, prefix)

    def _mlp_from_fc1(self, x: mx.array, prefix: str) -> mx.array:
        if self._uses_cider_fusion_target("mlp"):
            fused = self._fused_gelu_linear(x, f"{prefix}.mlp.fc2")
            if fused is not None:
                return fused
        x = _gelu_exact(x)
        return _linear(
            x,
            self.weights,
            f"{prefix}.mlp.fc2",
        )

    def _fused_layer_norm_linear(
        self,
        x: mx.array,
        norm_weight: mx.array,
        norm_bias: mx.array,
        linear_prefix: str,
    ) -> mx.array | None:
        return _fused_cider_layer_norm_linear(
            x,
            self.weights,
            linear_prefix,
            norm_weight,
            norm_bias,
            cider_fusion=self.cider_fusion,
        )

    def _fused_gelu_linear(self, x: mx.array, linear_prefix: str) -> mx.array | None:
        return _fused_cider_gelu_linear(
            x,
            self.weights,
            linear_prefix,
            cider_fusion=self.cider_fusion,
        )

    def _uses_cider_fusion_target(self, target: str) -> bool:
        if self.cider_fusion == "off":
            return False
        if self.cider_fusion_targets == "ln+mlp":
            return target in {"ln", "mlp"}
        return self.cider_fusion_targets == target


class MLXSO400MEncoder(MLXRadioEncoder):
    """Native MLX C-RADIOv4-SO400M forward path."""

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        compile_forward: bool = False,
        cider_fusion: str = "off",
        cider_fusion_targets: str = "ln",
    ) -> MLXSO400MEncoder:
        encoder = MLXRadioEncoder.load(
            checkpoint_path,
            dtype=dtype,
            revision=revision,
            variant="so400m",
            cider_fusion=cider_fusion,
            cider_fusion_targets=cider_fusion_targets,
        )
        wrapped = cls(encoder.weights, encoder.config, encoder.spec)
        if compile_forward:
            wrapped.compile_forward()
        return wrapped


class MLXHEncoder(MLXRadioEncoder):
    """Native MLX C-RADIOv4-H forward path."""

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        dtype: str = "float32",
        revision: str | None = None,
        compile_forward: bool = False,
        cider_fusion: str = "off",
        cider_fusion_targets: str = "ln",
    ) -> MLXHEncoder:
        encoder = MLXRadioEncoder.load(
            checkpoint_path,
            dtype=dtype,
            revision=revision,
            variant="h",
            cider_fusion=cider_fusion,
            cider_fusion_targets=cider_fusion_targets,
        )
        wrapped = cls(encoder.weights, encoder.config, encoder.spec)
        if compile_forward:
            wrapped.compile_forward()
        return wrapped


def _linear(x: mx.array, weights: dict[str, mx.array], prefix: str) -> mx.array:
    cider_weight = weights.get(f"{prefix}.cider_weight")
    if cider_weight is not None:
        return _cider_linear(x, weights, prefix, cider_weight)

    cider4_weight = weights.get(f"{prefix}.cider4_weight")
    if cider4_weight is not None:
        return _cider_w4a8_linear(x, weights, prefix, cider4_weight)

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


def _cider_linear(
    x: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
    cider_weight: mx.array,
) -> mx.array:
    try:
        from cider.ops import perchannel_linear, pergroup_linear
    except ImportError as exc:
        raise RuntimeError(
            "This bundle uses cider-w8a8 runtime quantization, but the optional "
            "Cider package is not installed."
        ) from exc
    _assert_cider_available()

    padded_meta = weights.get(f"{prefix}.cider_padded_in_features")
    padded_in_features = (
        int(_scalar_item(padded_meta)) if padded_meta is not None else int(cider_weight.shape[-1])
    )
    out_features_meta = weights.get(f"{prefix}.cider_out_features")
    out_features = (
        int(_scalar_item(out_features_meta))
        if out_features_meta is not None
        else cider_weight.shape[0]
    )
    if x.shape[-1] < padded_in_features:
        pad_width = padded_in_features - x.shape[-1]
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)
    x = _apply_cider_input_scale(x, weights, prefix)

    orig_shape = x.shape
    x_2d = mx.reshape(x, (-1, padded_in_features))
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        bias = bias.astype(mx.float16)
    group_size = int(_scalar_item(weights[f"{prefix}.cider_group_size"]))
    scale = weights[f"{prefix}.cider_scale"]
    if group_size == 0:
        y = perchannel_linear(x_2d, cider_weight, scale, bias)
    else:
        y = pergroup_linear(x_2d, cider_weight, scale, group_size, bias)
    return mx.reshape(y, (*orig_shape[:-1], out_features))


def _cider_w4a8_linear(
    x: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
    cider4_weight: mx.array,
) -> mx.array:
    try:
        from cider.ops import w4a8_linear
    except ImportError as exc:
        raise RuntimeError(
            "This bundle uses cider-w4a8 runtime quantization, but the optional "
            "Cider package is not installed."
        ) from exc
    _assert_cider_available()

    padded_meta = weights.get(f"{prefix}.cider_padded_in_features")
    padded_in_features = (
        int(_scalar_item(padded_meta))
        if padded_meta is not None
        else int(cider4_weight.shape[0] * 2)
    )
    out_features_meta = weights.get(f"{prefix}.cider_out_features")
    out_features = (
        int(_scalar_item(out_features_meta))
        if out_features_meta is not None
        else cider4_weight.shape[1]
    )
    if x.shape[-1] < padded_in_features:
        pad_width = padded_in_features - x.shape[-1]
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)
    x = _apply_cider_input_scale(x, weights, prefix)

    orig_shape = x.shape
    x_2d = mx.reshape(x, (-1, padded_in_features))
    y = w4a8_linear(x_2d, cider4_weight, weights[f"{prefix}.cider4_scale"])
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return mx.reshape(y, (*orig_shape[:-1], out_features))


def _fused_cider_layer_norm_linear(
    x: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
    norm_weight: mx.array,
    norm_bias: mx.array,
    cider_fusion: str = "off",
    eps: float = 1e-6,
) -> mx.array | None:
    cider_fusion = _validate_cider_fusion_mode(cider_fusion)
    if cider_fusion == "off":
        return None

    cider_weight = weights.get(f"{prefix}.cider_weight")
    if cider_weight is None:
        if cider_fusion == "required":
            raise RuntimeError(
                f"Cider fusion is required, but {prefix!r} is not a cider-w8a8 linear."
            )
        return None
    if weights.get(f"{prefix}.cider_input_scale") is not None:
        if cider_fusion == "required":
            raise RuntimeError(
                f"Cider fusion for {prefix!r} does not support SmoothQuant input scales yet."
            )
        return None

    try:
        from cider import ops as cider_ops
    except ImportError as exc:
        if cider_fusion == "required":
            raise RuntimeError(
                "Cider fusion is required, but the optional Cider package is not installed."
            ) from exc
        return None

    group_size = int(_scalar_item(weights[f"{prefix}.cider_group_size"]))
    op_name = "layernorm_perchannel_linear" if group_size == 0 else "layernorm_pergroup_linear"
    op = getattr(cider_ops, op_name, None)
    if op is None:
        if cider_fusion == "required":
            raise RuntimeError(
                f"Cider fusion is required, but cider.ops.{op_name} is not available. "
                "Install a Cider build with fused LN+linear kernels."
            )
        return None
    _assert_cider_available()

    padded_meta = weights.get(f"{prefix}.cider_padded_in_features")
    padded_in_features = (
        int(_scalar_item(padded_meta)) if padded_meta is not None else int(cider_weight.shape[-1])
    )
    out_features_meta = weights.get(f"{prefix}.cider_out_features")
    out_features = (
        int(_scalar_item(out_features_meta))
        if out_features_meta is not None
        else cider_weight.shape[0]
    )
    if x.shape[-1] < padded_in_features:
        pad_width = padded_in_features - x.shape[-1]
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)

    orig_shape = x.shape
    x_2d = mx.reshape(x, (-1, padded_in_features))
    linear_bias = weights.get(f"{prefix}.bias")
    if linear_bias is not None:
        linear_bias = linear_bias.astype(mx.float16)
    else:
        linear_bias = mx.zeros((out_features,), dtype=mx.float16)
    scale = weights[f"{prefix}.cider_scale"]
    if group_size == 0:
        y = op(
            x_2d,
            norm_weight,
            norm_bias,
            cider_weight,
            scale,
            linear_bias,
            eps,
        )
    else:
        y = op(
            x_2d,
            norm_weight,
            norm_bias,
            cider_weight,
            scale,
            linear_bias,
            group_size,
            eps,
        )
    if y.dtype != x.dtype:
        y = y.astype(x.dtype)
    return mx.reshape(y, (*orig_shape[:-1], out_features))


def _fused_cider_gelu_linear(
    x: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
    cider_fusion: str = "off",
) -> mx.array | None:
    cider_fusion = _validate_cider_fusion_mode(cider_fusion)
    if cider_fusion == "off":
        return None

    cider_weight = weights.get(f"{prefix}.cider_weight")
    if cider_weight is None:
        if cider_fusion == "required":
            raise RuntimeError(
                f"Cider MLP fusion is required, but {prefix!r} is not a cider-w8a8 linear."
            )
        return None
    if weights.get(f"{prefix}.cider_input_scale") is not None:
        if cider_fusion == "required":
            raise RuntimeError(
                f"Cider MLP fusion for {prefix!r} does not support SmoothQuant input scales."
            )
        return None

    try:
        from cider import ops as cider_ops
    except ImportError as exc:
        if cider_fusion == "required":
            raise RuntimeError(
                "Cider MLP fusion is required, but the optional Cider package is not installed."
            ) from exc
        return None

    group_size = int(_scalar_item(weights[f"{prefix}.cider_group_size"]))
    if group_size != 0:
        if cider_fusion == "required":
            raise RuntimeError(
                "Cider MLP fusion currently supports per-channel W8A8 bundles only."
            )
        return None
    op = getattr(cider_ops, "gelu_perchannel_linear", None)
    if op is None:
        if cider_fusion == "required":
            raise RuntimeError(
                "Cider MLP fusion is required, but cider.ops.gelu_perchannel_linear "
                "is not available. Install a Cider build with GELU+linear kernels."
            )
        return None
    _assert_cider_available()

    padded_meta = weights.get(f"{prefix}.cider_padded_in_features")
    padded_in_features = (
        int(_scalar_item(padded_meta)) if padded_meta is not None else int(cider_weight.shape[-1])
    )
    out_features_meta = weights.get(f"{prefix}.cider_out_features")
    out_features = (
        int(_scalar_item(out_features_meta))
        if out_features_meta is not None
        else cider_weight.shape[0]
    )
    if x.shape[-1] < padded_in_features:
        pad_width = padded_in_features - x.shape[-1]
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)

    orig_shape = x.shape
    x_2d = mx.reshape(x, (-1, padded_in_features))
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        bias = bias.astype(mx.float16)
    else:
        bias = mx.zeros((out_features,), dtype=mx.float16)
    y = op(x_2d, cider_weight, weights[f"{prefix}.cider_scale"], bias)
    if y.dtype != x.dtype:
        y = y.astype(x.dtype)
    return mx.reshape(y, (*orig_shape[:-1], out_features))


@lru_cache(maxsize=1)
def _assert_cider_available() -> None:
    from cider import is_available

    if not is_available():
        raise RuntimeError(
            "This bundle uses cider-w8a8 runtime quantization, but Cider's Apple "
            "M5+ INT8 kernels are not available on this machine."
        )


def _apply_cider_input_scale(
    x: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
) -> mx.array:
    input_scale = weights.get(f"{prefix}.cider_input_scale")
    if input_scale is None:
        return x
    return x / input_scale.astype(x.dtype)


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


def _validate_cider_fusion_mode(mode: str) -> str:
    if mode not in CIDER_FUSION_MODES:
        known = ", ".join(sorted(CIDER_FUSION_MODES))
        raise ValueError(f"unsupported cider_fusion={mode!r}; known: {known}")
    return mode


def _validate_cider_fusion_targets(targets: str) -> str:
    if targets not in CIDER_FUSION_TARGETS:
        known = ", ".join(sorted(CIDER_FUSION_TARGETS))
        raise ValueError(f"unsupported cider_fusion_targets={targets!r}; known: {known}")
    return targets


def _to_numpy(array: mx.array) -> Any:
    import numpy as np

    return np.asarray(array.astype(mx.float32))


def _should_cast_loaded_weight(key: str) -> bool:
    if key == "radio_model.summary_idxs" or "input_conditioner" in key:
        return False
    quantized_suffixes = (
        ".qweight",
        ".qscales",
        ".qbiases",
        ".qbits",
        ".qgroup_size",
        ".qmode_code",
        ".qin_features",
        ".qpadded_in_features",
        ".cider_weight",
        ".cider_scale",
        ".cider4_weight",
        ".cider4_scale",
        ".cider4_group_size",
        ".cider_group_size",
        ".cider_in_features",
        ".cider_padded_in_features",
        ".cider_out_features",
        ".cider_input_scale",
    )
    if key.endswith(quantized_suffixes):
        return False
    return True


def _scalar_item(value: mx.array) -> int | float:
    return value.item() if hasattr(value, "item") else value
