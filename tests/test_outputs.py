import numpy as np
import pytest

from cradio_mlx.outputs import EmbeddingResult, spatial_as_grid


def test_spatial_as_grid_reshapes_tokens_to_channels_first_grid():
    spatial = np.arange(2 * 4 * 3).reshape(2, 4, 3)

    grid = spatial_as_grid(spatial, grid_h=2, grid_w=2)

    assert grid.shape == (2, 3, 2, 2)
    assert grid[0, 0].tolist() == [[0, 3], [6, 9]]


def test_spatial_token_mismatch_raises():
    spatial = np.zeros((1, 5, 3))

    with pytest.raises(ValueError, match="does not match grid"):
        spatial_as_grid(spatial, grid_h=2, grid_w=2)


def test_embedding_result_delegates_grid_conversion():
    result = EmbeddingResult(
        summary=np.zeros((1, 3)),
        spatial=np.zeros((1, 4, 3)),
        grid_h=2,
        grid_w=2,
        patch_size=16,
        image_size=(32, 32),
    )

    assert result.spatial_as_grid().shape == (1, 3, 2, 2)
