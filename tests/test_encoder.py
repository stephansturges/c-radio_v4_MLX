import pytest

from cradio_mlx.encoder import CRadioEncoder
from cradio_mlx.mlx_so400m import H_SPEC, SO400M_SPEC


def test_encoder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="only the `pytorch` backend"):
        CRadioEncoder("nvidia/C-RADIOv4-SO400M", backend="mlx")  # type: ignore[arg-type]


def test_mlx_variant_specs_match_audited_dimensions():
    assert SO400M_SPEC.embed_dim == 1152
    assert SO400M_SPEC.depth == 27
    assert SO400M_SPEC.num_heads == 16

    assert H_SPEC.embed_dim == 1280
    assert H_SPEC.depth == 32
    assert H_SPEC.num_heads == 16
