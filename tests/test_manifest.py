from cradio_mlx.bundles.manifest import BundleManifest
from cradio_mlx.bundles.save_load import inspect_bundle


def test_manifest_round_trip(tmp_path):
    manifest = BundleManifest(
        model_id="nvidia/C-RADIOv4-H",
        revision="abc123",
        variant="h",
        dtype="bfloat16",
    )

    path = manifest.save(tmp_path / "bundle")
    loaded = BundleManifest.load(path)

    assert loaded.model_id == "nvidia/C-RADIOv4-H"
    assert loaded.revision == "abc123"
    assert loaded.variant == "h"
    assert loaded.dtype == "bfloat16"


def test_inspect_bundle_reports_manifest(tmp_path):
    BundleManifest(
        model_id="nvidia/C-RADIOv4-H",
        revision="abc123",
        variant="h",
        dtype="bfloat16",
    ).save(tmp_path / "bundle")

    info = inspect_bundle(tmp_path / "bundle")

    assert info["has_manifest"] is True
    assert "manifest.json" in info["files"]
