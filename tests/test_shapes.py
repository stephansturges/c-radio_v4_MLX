import pytest

from cradio_mlx.config import get_model_config


def test_default_h_grid_shape():
    config = get_model_config("nvidia/C-RADIOv4-H")

    assert config.grid_shape(512) == (32, 32)
    assert config.spatial_tokens(512) == 1024


def test_non_square_grid_shape():
    config = get_model_config("nvidia/C-RADIOv4-SO400M")

    assert config.grid_shape((512, 768)) == (32, 48)
    assert config.spatial_tokens((512, 768)) == 1536


def test_image_size_must_be_patch_aligned():
    config = get_model_config("nvidia/C-RADIOv4-H")

    with pytest.raises(ValueError, match="multiple of 16"):
        config.grid_shape(513)


def test_image_size_range_is_enforced():
    config = get_model_config("nvidia/C-RADIOv4-H")

    with pytest.raises(ValueError, match="outside the supported range"):
        config.grid_shape(128)
