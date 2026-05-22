import json

import numpy as np
from safetensors.numpy import save_file

from cradio_mlx.convert import ConversionRequest, convert_checkpoint


def test_manifest_only_conversion_copies_metadata(tmp_path):
    hf_path = tmp_path / "hf"
    hf_path.mkdir()
    (hf_path / "config.json").write_text('{"patch_size": 16}\n', encoding="utf-8")
    (hf_path / "preprocessor_config.json").write_text('{"image_size": 512}\n', encoding="utf-8")

    manifest_path = convert_checkpoint(
        ConversionRequest(
            hf_path=hf_path,
            mlx_path=tmp_path / "bundle",
            model_id="nvidia/C-RADIOv4-H",
            revision="abc123",
            manifest_only=True,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "nvidia/C-RADIOv4-H"
    assert manifest["variant"] == "h"
    assert manifest["extra"]["conversion_state"] == "manifest_only"
    assert "config.json" in manifest["source_files"]
    assert (tmp_path / "bundle" / "preprocessor_config.json").exists()


def test_conversion_copies_weights_for_self_contained_bundle(tmp_path):
    hf_path = tmp_path / "hf"
    hf_path.mkdir()
    (hf_path / "config.json").write_text('{"patch_size": 16}\n', encoding="utf-8")
    save_file({"x": np.zeros((1,), dtype=np.float32)}, hf_path / "model.safetensors")

    manifest_path = convert_checkpoint(
        ConversionRequest(
            hf_path=hf_path,
            mlx_path=tmp_path / "bundle",
            model_id="nvidia/C-RADIOv4-H",
            revision="abc123",
            manifest_only=False,
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["extra"]["conversion_state"] == "self_contained"
    assert manifest["extra"]["weights_file"] == "model.safetensors"
    assert "model.safetensors" in manifest["source_files"]
    assert (tmp_path / "bundle" / "model.safetensors").exists()
