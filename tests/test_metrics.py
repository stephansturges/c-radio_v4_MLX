import json

import numpy as np
import pytest

from cradio_mlx.metrics import compare_embedding_npz, cosine_similarity, write_embedding_comparison


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_compare_embedding_npz(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, summary=np.ones((1, 2)), spatial=np.ones((1, 4, 2)))
    np.savez(candidate, summary=np.ones((1, 2)), spatial=np.ones((1, 4, 2)))

    comparison = compare_embedding_npz(reference, candidate)

    assert comparison.summary_cosine == pytest.approx(1.0)
    assert comparison.spatial_cosine == pytest.approx(1.0)


def test_write_embedding_comparison(tmp_path):
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, summary=np.ones((1, 2)), spatial=np.ones((1, 4, 2)))
    np.savez(candidate, summary=np.ones((1, 2)), spatial=np.ones((1, 4, 2)))

    path = write_embedding_comparison(reference, candidate, tmp_path / "comparison.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["summary_cosine"] == pytest.approx(1.0)
    assert payload["spatial_cosine"] == pytest.approx(1.0)
