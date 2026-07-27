"""The Python surface: LOF the estimator, lof the function."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_data

import culof
from culof import LOF, lof


def test_device_is_reported() -> None:
    assert culof.cuda_available()
    assert culof.device_info()


def test_scores_separate_outliers(data: np.ndarray) -> None:
    scores = lof(data, 20)
    assert scores.shape == (len(data),)
    assert scores.dtype == np.float32
    assert scores[:25].min() > np.median(scores[25:]) * 2


def test_fit_predict_returns_labels(data: np.ndarray) -> None:
    labels = LOF(n_neighbors=20, contamination=0.025).fit_predict(data)
    assert set(np.unique(labels)) <= {-1, 1}
    assert abs((labels == -1).sum() - 25) <= 3


def test_negative_outlier_factor_is_negated(data: np.ndarray) -> None:
    model = LOF(n_neighbors=20).fit(data)
    np.testing.assert_allclose(-model.negative_outlier_factor_, lof(data, 20))


def test_score_samples_without_argument_uses_fit(data: np.ndarray) -> None:
    model = LOF(n_neighbors=20).fit(data)
    np.testing.assert_array_equal(model.score_samples(), model.negative_outlier_factor_)


def test_predict_flags_lowest_scores(data: np.ndarray) -> None:
    model = LOF(n_neighbors=20, contamination=0.05).fit(data)
    flagged = model.predict() == -1
    assert (model.negative_outlier_factor_[flagged] < model.offset_).all()


def test_unfitted_access_raises() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        LOF().predict()
    with pytest.raises(ValueError, match="not fitted"):
        LOF().score_samples()


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_n_neighbors(bad: int) -> None:
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        LOF(n_neighbors=bad)


@pytest.mark.parametrize("bad", [0.0, 0.9, 1.0, -0.1])
def test_rejects_bad_contamination(bad: float) -> None:
    with pytest.raises(ValueError, match="contamination"):
        LOF(contamination=bad)


@pytest.mark.parametrize("bad", [100, 500])
def test_rejects_n_neighbors_above_sample_count(bad: int) -> None:
    X = make_data(100, 2, seed=1)
    with pytest.raises(ValueError, match="requires at least"):
        LOF(n_neighbors=bad).fit(X)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        lof(np.zeros(10, dtype=np.float32), 3)


def test_rejects_non_finite_input() -> None:
    X = np.ones((100, 2), dtype=np.float32)
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        lof(X, 10)


def test_accepts_lists() -> None:
    X = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [0.05, 0.2]]
    assert lof(X, 2).shape == (5,)


def test_repr_is_informative() -> None:
    expected = "LOF(n_neighbors=7, contamination='auto', normalize=False)"
    assert repr(LOF(n_neighbors=7)) == expected


def test_repeated_calls_are_bit_identical(data: np.ndarray) -> None:
    """Guards the prefix-sum gather; atomics there would make this flaky."""
    first = lof(data, 20)
    for _ in range(3):
        np.testing.assert_array_equal(first, lof(data, 20))


def test_tiling_does_not_change_results(data: np.ndarray, monkeypatch) -> None:
    """Tile height is an implementation detail, not a result-changing knob.

    To a few ULP rather than exactly: cuBLAS selects a different SGEMM kernel per
    tile shape, reordering the dot-product accumulation.
    """
    reference = lof(data, 20)
    for rows in ("1", "13", "128"):
        monkeypatch.setenv("CULOF_TILE_ROWS", rows)
        np.testing.assert_allclose(reference, lof(data, 20), rtol=1e-5)
