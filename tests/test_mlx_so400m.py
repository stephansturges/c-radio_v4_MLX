import numpy as np

from cradio_mlx.mlx_so400m import (
    _im2patches,
    _resize_pos_embed_align_corners_false,
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
