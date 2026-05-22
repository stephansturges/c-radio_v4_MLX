import numpy as np

from cradio_mlx.outputs import EmbeddingResult
from cradio_mlx.prompting import package_vlm_prompt


def test_package_vlm_prompt_records_embedding_contract():
    result = EmbeddingResult(
        summary=np.zeros((1, 4)),
        spatial=np.zeros((1, 16, 8)),
        grid_h=4,
        grid_w=4,
        patch_size=16,
        image_size=(64, 64),
        metadata={"backend": "test"},
    )

    package = package_vlm_prompt("describe this", result, projector="demo")

    assert package["prompt"] == "describe this"
    assert package["projector"] == "demo"
    assert package["embedding_contract"]["summary_shape"] == (1, 4)
    assert package["embedding_contract"]["spatial_shape"] == (1, 16, 8)
    assert package["metadata"]["backend"] == "test"
