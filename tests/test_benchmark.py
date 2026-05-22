import json

from cradio_mlx.benchmark import BenchmarkRequest, write_benchmark_stub
from cradio_mlx.bundles.manifest import BundleManifest


def test_write_benchmark_stub(tmp_path):
    bundle = tmp_path / "bundle"
    BundleManifest(
        model_id="nvidia/C-RADIOv4-H",
        revision="abc123",
        variant="h",
        dtype="bfloat16",
    ).save(bundle)

    report = write_benchmark_stub(
        BenchmarkRequest(
            model=bundle,
            image_size=512,
            batch_sizes=[1, 2],
            report=tmp_path / "report.json",
        )
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["benchmark_state"] == "manifest_only"
    assert payload["batch_sizes"] == [1, 2]
