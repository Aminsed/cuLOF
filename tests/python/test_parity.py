"""Agreement with scikit-learn.

This is the contract the README advertises, so it is asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_data, max_rel_err, sklearn_lof
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from culof import LOF, lof

# scikit-learn computes in float64, cuLOF in float32. A few 1e-5 of relative
# disagreement is the expected cost; see test_near_ties below for the boundary.
REL_TOL = 2e-4


@pytest.mark.parametrize(
    ("n", "d", "k"),
    [
        (500, 2, 20),
        (2000, 2, 20),
        (2000, 8, 20),
        (3000, 16, 10),
        (1500, 64, 35),
        (1000, 3, 1),
        (1000, 3, 100),
    ],
)
def test_values_match(n: int, d: int, k: int) -> None:
    X = make_data(n, d, seed=n + d + k, n_outliers=max(1, n // 100))
    assert max_rel_err(lof(X, k), sklearn_lof(X, k)) < REL_TOL


@pytest.mark.parametrize("k", [33, 64, 128, 256])
def test_k_above_32(k: int) -> None:
    """The previous implementation raised for any k > 32."""
    X = make_data(1000, 4, seed=k)
    assert max_rel_err(lof(X, k), sklearn_lof(X, k)) < REL_TOL


def test_class_is_a_drop_in_replacement() -> None:
    """Same call, same labels, same attributes as scikit-learn's estimator."""
    X = make_data(3000, 6, seed=7, n_outliers=30)

    theirs = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    ours = LOF(n_neighbors=20, contamination=0.01)

    np.testing.assert_array_equal(ours.fit_predict(X), theirs.fit_predict(X))
    assert max_rel_err(-ours.negative_outlier_factor_, -theirs.negative_outlier_factor_) < REL_TOL
    assert ours.offset_ == pytest.approx(theirs.offset_, rel=1e-3)
    assert ours.n_neighbors_ == theirs.n_neighbors_


def test_auto_contamination_matches_sklearn_offset() -> None:
    X = make_data(1500, 4, seed=11, n_outliers=15)
    assert LOF().fit(X).offset_ == LocalOutlierFactor().fit(X).offset_ == -1.5


def test_ranking_matches_sklearn() -> None:
    """Ordering is what actually decides which points are flagged."""
    X = make_data(4000, 6, seed=13, n_outliers=40)
    got, want = lof(X, 20), sklearn_lof(X, 20)
    assert set(np.argsort(-got)[:40]) == set(np.argsort(-want)[:40])
    assert np.corrcoef(got, want)[0, 1] > 0.9999


def test_near_ties_are_the_only_disagreement() -> None:
    """Pin down exactly where float32 stops being able to agree with float64.

    When a point's k-th and (k+1)-th neighbours are closer together than float32
    can resolve, which one is "the k-th" is genuinely ambiguous, and the two
    libraries may pick differently. That point's score then differs by up to
    about 1%.

    The assertions below state the useful guarantee: the bulk agrees tightly,
    disagreement is rare, every disagreeing point sits on such a near-tie, and
    the outlier ranking is unaffected.
    """
    k, n = 20, 20_000
    X = make_data(n, 8, seed=99, n_outliers=n // 100)

    got, want = lof(X, k), sklearn_lof(X, k)
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-9)

    assert np.median(rel) < 1e-5
    assert np.percentile(rel, 99) < 1e-4

    disagreeing = np.flatnonzero(rel > 1e-4)
    assert len(disagreeing) < n // 100

    dist, _ = NearestNeighbors(n_neighbors=k + 2).fit(X).kneighbors(X)
    gap = (dist[:, k + 1] - dist[:, k]) / np.maximum(dist[:, k], 1e-12)
    if len(disagreeing):
        assert gap[disagreeing].max() < 1e-3, "disagreement away from a near-tie"

    top = n // 100
    assert set(np.argsort(-got)[:top]) == set(np.argsort(-want)[:top])
