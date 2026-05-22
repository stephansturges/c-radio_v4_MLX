from cradio_mlx.acceleration import inspect_acceleration


def test_inspect_acceleration_returns_stable_keys():
    report = inspect_acceleration().to_dict()

    assert set(report) == {
        "mlx_available",
        "mlx_default_device",
        "mlx_gpu",
        "torch_available",
        "torch_mps_available",
        "torch_cuda_available",
        "preferred_torch_device",
    }
    assert isinstance(report["mlx_available"], bool)
