import numpy as np

from cradio_mlx.mlx_so400m import (
    _apply_cider_input_scale,
    _assert_cider_available,
    _fused_cider_gelu_linear,
    _fused_cider_layer_norm_linear,
    _im2patches,
    _resize_pos_embed_align_corners_false,
    _validate_cider_fusion_mode,
    _validate_cider_fusion_targets,
)


def test_im2patches_matches_channel_first_patch_order():
    import mlx.core as mx

    x = mx.array(np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2))
    patches = np.asarray(_im2patches(x, patch_size=2))

    assert patches.shape == (1, 1, 12)
    assert patches[0, 0].tolist() == list(range(12))


def test_resize_pos_embed_preserves_requested_shape():
    pos = np.arange(1 * 4 * 4 * 2, dtype=np.float32).reshape(1, 4, 4, 2)

    resized = _resize_pos_embed_align_corners_false(pos, 2, 3)

    assert resized.shape == (1, 2, 3, 2)


def test_apply_cider_input_scale_divides_last_dimension():
    import mlx.core as mx

    x = mx.array([[[2.0, 6.0, 12.0]]], dtype=mx.float32)
    out = _apply_cider_input_scale(
        x,
        {"layer.cider_input_scale": mx.array([2.0, 3.0, 4.0], dtype=mx.float32)},
        "layer",
    )

    assert np.asarray(out).tolist() == [[[1.0, 2.0, 3.0]]]


def test_validate_cider_fusion_mode_rejects_unknown():
    import pytest

    assert _validate_cider_fusion_mode("auto") == "auto"
    with pytest.raises(ValueError, match="unsupported cider_fusion"):
        _validate_cider_fusion_mode("maybe")


def test_validate_cider_fusion_targets_rejects_unknown():
    import pytest

    assert _validate_cider_fusion_targets("ln+mlp") == "ln+mlp"
    with pytest.raises(ValueError, match="unsupported cider_fusion_targets"):
        _validate_cider_fusion_targets("attention")


def test_fused_cider_layer_norm_linear_auto_falls_back_when_op_missing(monkeypatch):
    import types

    import mlx.core as mx

    fake_ops = types.SimpleNamespace()
    fake_cider = types.SimpleNamespace(ops=fake_ops, is_available=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "cider", fake_cider)
    monkeypatch.setitem(__import__("sys").modules, "cider.ops", fake_ops)
    _assert_cider_available.cache_clear()

    out = _fused_cider_layer_norm_linear(
        mx.zeros((1, 2, 4), dtype=mx.float16),
        {
            "layer.cider_weight": mx.zeros((3, 4), dtype=mx.int8),
            "layer.cider_scale": mx.ones((3,), dtype=mx.float32),
            "layer.cider_group_size": mx.array([0], dtype=mx.int32),
        },
        "layer",
        mx.ones((4,), dtype=mx.float16),
        mx.zeros((4,), dtype=mx.float16),
        cider_fusion="auto",
    )

    assert out is None
    _assert_cider_available.cache_clear()


def test_fused_cider_layer_norm_linear_required_needs_fused_cider(monkeypatch):
    import types

    import mlx.core as mx
    import pytest

    fake_ops = types.SimpleNamespace()
    fake_cider = types.SimpleNamespace(ops=fake_ops, is_available=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "cider", fake_cider)
    monkeypatch.setitem(__import__("sys").modules, "cider.ops", fake_ops)
    _assert_cider_available.cache_clear()

    with pytest.raises(RuntimeError, match="Cider fusion is required"):
        _fused_cider_layer_norm_linear(
            mx.zeros((1, 2, 4), dtype=mx.float16),
            {
                "layer.cider_weight": mx.zeros((3, 4), dtype=mx.int8),
                "layer.cider_scale": mx.ones((3,), dtype=mx.float32),
                "layer.cider_group_size": mx.array([0], dtype=mx.int32),
            },
            "layer",
            mx.ones((4,), dtype=mx.float16),
            mx.zeros((4,), dtype=mx.float16),
            cider_fusion="required",
        )
    _assert_cider_available.cache_clear()


def test_fused_cider_layer_norm_linear_calls_ops_wrapper(monkeypatch):
    import types

    import mlx.core as mx

    calls = {}

    def layernorm_perchannel_linear(x, norm_weight, norm_bias, weight, scale, bias, eps):
        calls["args"] = (x, norm_weight, norm_bias, weight, scale, bias, eps)
        return mx.ones((2, 3), dtype=mx.float16)

    fake_ops = types.SimpleNamespace(layernorm_perchannel_linear=layernorm_perchannel_linear)
    fake_cider = types.SimpleNamespace(ops=fake_ops, is_available=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "cider", fake_cider)
    monkeypatch.setitem(__import__("sys").modules, "cider.ops", fake_ops)
    _assert_cider_available.cache_clear()

    out = _fused_cider_layer_norm_linear(
        mx.zeros((1, 2, 4), dtype=mx.float16),
        {
            "layer.cider_weight": mx.zeros((3, 4), dtype=mx.int8),
            "layer.cider_scale": mx.ones((3,), dtype=mx.float32),
            "layer.cider_group_size": mx.array([0], dtype=mx.int32),
        },
        "layer",
        mx.ones((4,), dtype=mx.float16),
        mx.zeros((4,), dtype=mx.float16),
        cider_fusion="required",
    )

    assert out.shape == (1, 2, 3)
    assert calls["args"][-1] == 1e-6
    _assert_cider_available.cache_clear()


def test_fused_cider_gelu_linear_calls_ops_wrapper(monkeypatch):
    import types

    import mlx.core as mx

    calls = {}

    def gelu_perchannel_linear(x, weight, scale, bias):
        calls["args"] = (x, weight, scale, bias)
        return mx.ones((2, 3), dtype=mx.float16)

    fake_ops = types.SimpleNamespace(gelu_perchannel_linear=gelu_perchannel_linear)
    fake_cider = types.SimpleNamespace(ops=fake_ops, is_available=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "cider", fake_cider)
    monkeypatch.setitem(__import__("sys").modules, "cider.ops", fake_ops)
    _assert_cider_available.cache_clear()

    out = _fused_cider_gelu_linear(
        mx.zeros((1, 2, 4), dtype=mx.float16),
        {
            "layer.cider_weight": mx.zeros((3, 4), dtype=mx.int8),
            "layer.cider_scale": mx.ones((3,), dtype=mx.float32),
            "layer.cider_group_size": mx.array([0], dtype=mx.int32),
        },
        "layer",
        cider_fusion="required",
    )

    assert out.shape == (1, 2, 3)
    assert calls["args"][0].shape == (2, 4)
    _assert_cider_available.cache_clear()


def test_fused_cider_gelu_linear_required_needs_op(monkeypatch):
    import types

    import mlx.core as mx
    import pytest

    fake_ops = types.SimpleNamespace()
    fake_cider = types.SimpleNamespace(ops=fake_ops, is_available=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "cider", fake_cider)
    monkeypatch.setitem(__import__("sys").modules, "cider.ops", fake_ops)
    _assert_cider_available.cache_clear()

    with pytest.raises(RuntimeError, match="gelu_perchannel_linear"):
        _fused_cider_gelu_linear(
            mx.zeros((1, 2, 4), dtype=mx.float16),
            {
                "layer.cider_weight": mx.zeros((3, 4), dtype=mx.int8),
                "layer.cider_scale": mx.ones((3,), dtype=mx.float32),
                "layer.cider_group_size": mx.array([0], dtype=mx.int32),
            },
            "layer",
            cider_fusion="required",
        )
    _assert_cider_available.cache_clear()
