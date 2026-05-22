import numpy as np
from safetensors.numpy import save_file

from cradio_mlx.compat.hf_mapping import audit_safetensors, write_weight_audit


def test_audit_safetensors_records_keys_shapes_and_dtypes(tmp_path):
    save_file(
        {
            "encoder.block.weight": np.zeros((2, 3), dtype=np.float32),
            "encoder.norm.bias": np.zeros((3,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )

    report = audit_safetensors(tmp_path)

    assert report["tensor_count"] == 2
    assert report["records"][0]["key"] == "encoder.block.weight"
    assert report["records"][0]["shape"] == (2, 3)


def test_write_weight_audit(tmp_path):
    save_file({"x": np.zeros((1,), dtype=np.float32)}, tmp_path / "model.safetensors")

    path = write_weight_audit(tmp_path, tmp_path / "report.json")

    assert path.exists()
    assert "tensor_count" in path.read_text(encoding="utf-8")
