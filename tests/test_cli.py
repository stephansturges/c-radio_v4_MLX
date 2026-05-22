import json

from cradio_mlx.cli import main


def test_spatial_shape_cli(capsys):
    assert main(["spatial-shape", "--image-size", "512"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["grid_h"] == 32
    assert payload["grid_w"] == 32
    assert payload["spatial_tokens"] == 1024


def test_device_info_cli(capsys):
    assert main(["device-info"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "mlx_available" in payload
    assert "preferred_torch_device" in payload
