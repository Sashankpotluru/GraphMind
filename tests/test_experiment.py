import pytest


def test_complete_rrf_preserves_tied_zero_candidates():
    from benchmarks.run_retrieval import ranked, rrf

    sparse = {"z": 0, "b": 0, "a": 2}
    dense = {"z": 0.4, "b": -0.3, "a": 0.1}
    fused = rrf(sparse, dense, 0.25, 10)
    assert ranked(sparse) == ["a", "b", "z"]
    assert fused == pytest.approx(
        {"a": 0.75 / 11 + 0.25 / 12, "b": 0.75 / 12 + 0.25 / 13, "z": 0.75 / 13 + 0.25 / 11}
    )
    assert ranked({"b": 0, "a": 0}) == ["a", "b"]


def test_grid_selection_uses_primary_secondary_then_declared_order():
    from benchmarks.run_retrieval import choose

    assert choose([(1, 0.9), (2, 0.1), (2, 0.7), (2, 0.7)]) == 2


def test_dense_dot_replay_uses_full_stored_values():
    import numpy as np

    from benchmarks.models import cosine_scores

    q = np.array([0.6, 0.8], dtype="<f4")
    mat = np.array([[1, 0], [0, 1], [-1, 0]], dtype="<f4")
    assert cosine_scores(["b", "c", "a"], mat, q) == pytest.approx({"b": 0.6, "c": 0.8, "a": -0.6})
