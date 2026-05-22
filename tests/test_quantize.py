import json

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

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
    save_file(
        {
            "radio_model.model.blocks.0.attn.qkv.weight": np.ones((64, 64), dtype=np.float32),
            "radio_model.model.blocks.0.attn.qkv.bias": np.ones((64,), dtype=np.float32),
            "radio_model.model.blocks.0.norm1.weight": np.ones((64,), dtype=np.float32),
        },
        source / "model.safetensors",
    )

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
    assert manifest["quantization"]["state"] == "dry_run_wrote_weights"

    with safe_open(tmp_path / "quantized" / "model.safetensors", framework="numpy") as handle:
        keys = set(handle.keys())

    assert "radio_model.model.blocks.0.attn.qkv.qweight" in keys
    assert "radio_model.model.blocks.0.attn.qkv.weight" not in keys
    assert "radio_model.model.blocks.0.attn.qkv.bias" in keys


def test_quantize_pads_inner_dimension_for_group_size(tmp_path):
    source = tmp_path / "source"
    BundleManifest(
        model_id="nvidia/C-RADIOv4-SO400M",
        revision="abc123",
        variant="so400m",
        dtype="bfloat16",
    ).save(source)
    save_file(
        {
            "radio_model.model.blocks.0.mlp.fc2.weight": np.ones((64, 80), dtype=np.float32),
        },
        source / "model.safetensors",
    )

    manifest_path = quantize_bundle(
        QuantizationRequest(
            model=source,
            out=tmp_path / "quantized",
            bits=8,
            group_size=64,
            mode="affine",
        )
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = manifest["extra"]["quantization_stats"]
    assert stats["quantized_tensors"] == 1
    assert stats["padded_tensors"] == 1
    assert stats["padded_features"] == 48

    with safe_open(tmp_path / "quantized" / "model.safetensors", framework="numpy") as handle:
        assert handle.get_tensor("radio_model.model.blocks.0.mlp.fc2.qin_features").tolist() == [80]
        assert handle.get_tensor(
            "radio_model.model.blocks.0.mlp.fc2.qpadded_in_features"
        ).tolist() == [128]


def test_quantize_mxfp8_writes_scale_only_mode_metadata(tmp_path):
    source = tmp_path / "source"
    BundleManifest(
        model_id="nvidia/C-RADIOv4-H",
        revision="abc123",
        variant="h",
        dtype="bfloat16",
    ).save(source)
    save_file(
        {
            "radio_model.model.blocks.0.attn.proj.weight": np.ones((64, 64), dtype=np.float32),
        },
        source / "model.safetensors",
    )

    quantize_bundle(
        QuantizationRequest(
            model=source,
            out=tmp_path / "mxfp8",
            bits=8,
            group_size=32,
            mode="mxfp8",
        )
    )

    with safe_open(tmp_path / "mxfp8" / "model.safetensors", framework="numpy") as handle:
        keys = set(handle.keys())
        assert "radio_model.model.blocks.0.attn.proj.qweight" in keys
        assert "radio_model.model.blocks.0.attn.proj.qscales" in keys
        assert "radio_model.model.blocks.0.attn.proj.qbiases" not in keys
        assert handle.get_tensor("radio_model.model.blocks.0.attn.proj.qmode_code").tolist() == [
            2
        ]
