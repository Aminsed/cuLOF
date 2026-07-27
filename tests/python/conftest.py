"""Shared fixtures and helpers for the Python test suite."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor


def sklearn_lof(X: np.ndarray, k: int) -> np.ndarray:
    """Reference LOF values, in the same sign convention as ``culof.lof``."""
    return -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_


def max_rel_err(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9)))


def make_data(n: int, d: int, seed: int, n_outliers: int = 0) -> np.ndarray:
    """Gaussian points with the first ``n_outliers`` rows scattered widely.

    The outliers are spread over a box rather than displaced by a shared offset:
    a group of points sharing one offset forms its own dense cluster, and LOF
    correctly scores such points near 1.0.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if n_outliers:
        X[:n_outliers] = rng.uniform(-15, 15, size=(n_outliers, d)).astype(np.float32)
    return X


@pytest.fixture
def data() -> np.ndarray:
    """1,000 points in 3-D with 25 planted outliers in the first rows."""
    return make_data(1000, 3, seed=42, n_outliers=25)
