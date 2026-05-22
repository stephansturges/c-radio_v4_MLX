import json

import pytest

from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.quantize import QuantizationRequest, quantize_bundle, validate_quantization


def test_validate_affine_bits():
    validate_quantization(bits=8, group_size=64, mode="affine")

    with pytest.raises(ValueError, match="affine quantization bits"):
        validate_quantization(bits=3, group_size=64, mode="affine")


def test_quantize_dry_run_writes_planned_manifest(tmp_path):
    source = tmp_path / "source"
    BundleManifest(
        model_id="nvidia/C-RADIOv4-H",
        revision="abc123",
        variant="h",
        dtype="bfloat16",
    ).save(source)

    manifest_path = quantize_bundle(
        QuantizationRequest(
            model=source,
            out=tmp_path / "quantized",
            bits=8,
            group_size=64,
            mode="affine",
            dry_run=True,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["quantization"]["bits"] == 8
    assert manifest["quantization"]["mode"] == "affine"
    assert manifest["quantization"]["state"] == "planned"
