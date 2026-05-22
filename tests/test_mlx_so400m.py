import numpy as np

from cradio_mlx.mlx_so400m import (
    _dequantize_loaded_weights,
    _im2patches,
    _linear,
    _resize_pos_embed_align_corners_false,
    _validate_quantized_runtime,
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


def test_dequantize_loaded_weights_restores_dense_linear_shape():
    import mlx.core as mx

    prefix = "radio_model.model.blocks.0.mlp.fc2"
    weight = np.linspace(-1.0, 1.0, num=4 * 65, dtype=np.float32).reshape(4, 65)
    padded_weight = np.pad(weight, ((0, 0), (0, 63)))
    qweight, qscales, qbiases = mx.quantize(
        mx.array(padded_weight),
        group_size=64,
        bits=8,
        mode="affine",
    )
    mx.eval(qweight, qscales, qbiases)
    weights = {
        f"{prefix}.qweight": qweight,
        f"{prefix}.qscales": qscales,
        f"{prefix}.qbiases": qbiases,
        f"{prefix}.qbits": mx.array([8]),
        f"{prefix}.qgroup_size": mx.array([64]),
        f"{prefix}.qmode_code": mx.array([0]),
        f"{prefix}.qin_features": mx.array([65]),
        f"{prefix}.qpadded_in_features": mx.array([128]),
        f"{prefix}.bias": mx.array(np.linspace(-0.2, 0.2, num=4, dtype=np.float32)),
    }

    dense_weights = _dequantize_loaded_weights(weights, mx.bfloat16)

    assert f"{prefix}.weight" in dense_weights
    assert f"{prefix}.qweight" not in dense_weights
    assert dense_weights[f"{prefix}.weight"].shape == (4, 65)

    x = mx.array(np.linspace(-0.25, 0.25, num=2 * 65, dtype=np.float32).reshape(2, 65))
    packed_out = _linear(x.astype(mx.bfloat16), weights, prefix)
    dense_out = _linear(x.astype(mx.bfloat16), dense_weights, prefix)
    mx.eval(packed_out, dense_out)

    np.testing.assert_allclose(
        np.asarray(packed_out.astype(mx.float32)),
        np.asarray(dense_out.astype(mx.float32)),
        rtol=1e-2,
        atol=1e-2,
    )


def test_validate_quantized_runtime_rejects_unknown_mode():
    import pytest

    assert _validate_quantized_runtime("packed") == "packed"
    assert _validate_quantized_runtime("dequantize") == "dequantize"
    with pytest.raises(ValueError, match="unsupported quantized_runtime"):
        _validate_quantized_runtime("magic")
