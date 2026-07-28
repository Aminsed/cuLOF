"""cuLOF - Local Outlier Factor on the GPU.

:class:`LOF` mirrors :class:`sklearn.neighbors.LocalOutlierFactor`: same
constructor arguments, same methods, same attributes, same sign conventions.
Covers the transductive fit_predict workflow; see the README for what the
scikit-learn estimator does that this does not.

    >>> from culof import LOF
    >>> labels = LOF(n_neighbors=20).fit_predict(X)   # -1 outlier, 1 inlier

:func:`lof` is the direct path when you want the LOF value itself rather than
scikit-learn's negated convention.

    >>> from culof import lof
    >>> scores = lof(X, k=20)                         # ~1.0 normal, higher = more anomalous
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import _culof
from ._version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = ["LOF", "__version__", "cuda_available", "device_info", "lof"]

cuda_available = _culof.cuda_available
device_info = _culof.device_info


def _validate(X: ArrayLike, k: int) -> NDArray[np.float32]:
    arr = np.ascontiguousarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D array of shape (n_samples, n_features), got shape {arr.shape}"
        )
    n_samples, n_features = arr.shape
    if n_features < 1:
        raise ValueError("need at least one feature")
    if k < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {k}")
    if k >= n_samples:
        raise ValueError(f"n_neighbors={k} requires at least {k + 1} samples, got {n_samples}")
    # Checked here rather than in C++: one vectorised pass over the array beats a
    # scalar loop, and the error surfaces before anything touches the GPU.
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or infinity")
    _check_representable(arr, n_features)
    return arr


def _check_representable(arr: NDArray[np.float32], n_features: int) -> None:
    """Reject input whose squared distances would overflow float32.

    Distances are accumulated in float32. Centring bounds each coordinate by
    twice the input maximum, so a squared distance is at most
    ``n_features * (4 * max|x|)**2``. Past FLT_MAX that saturates to infinity,
    every selection key ties at infinity, and the scores come back NaN -- which
    the estimator reads as "inlier" for every point. Checked in logs so the
    check itself cannot overflow.
    """
    if arr.size == 0:
        return
    max_abs = float(np.max(np.abs(arr)))
    if max_abs == 0.0:
        return
    log_bound = math.log(n_features) + 2.0 * math.log(4.0 * max_abs)
    if log_bound > math.log(float(np.finfo(np.float32).max)):
        raise ValueError(
            f"input magnitude {max_abs:.3g} overflows float32 when squared "
            f"({n_features} features): distances would saturate to infinity and "
            f"every score would be NaN. Rescale the data, or pass normalize=True "
            f"to z-score each feature first."
        )


def lof(X: ArrayLike, k: int = 20, normalize: bool = False) -> NDArray[np.float32]:
    """Local Outlier Factor of every row of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Converted to C-contiguous float32 if it is not already.
    k : int, default=20
        Neighbours per point, excluding the point itself. Must satisfy
        ``1 <= k <= n_samples - 1``. Selection is independent of ``k``;
        storage and the density/score passes are O(n_samples * k).
    normalize : bool, default=False
        Z-score each feature before computing distances. scikit-learn does not,
        so leave it off when comparing.

    Returns
    -------
    ndarray of shape (n_samples,)
        Around 1.0 for a point as densely surrounded as its neighbours, larger
        for one in a comparatively sparse region.
    """
    scores = _culof.lof(_validate(X, k), k, normalize)
    # Backstop for anything the magnitude bound in _validate does not catch.
    # Returning NaN here is indistinguishable from "no outliers" downstream.
    if not np.isfinite(scores).all():
        raise ValueError(
            "LOF produced non-finite scores; this indicates float32 overflow or "
            "underflow in the distance computation. Rescale the data, or pass "
            "normalize=True."
        )
    return scores


class LOF:
    """Unsupervised outlier detection with Local Outlier Factor, on the GPU.

    Covers the transductive ``fit_predict`` workflow of
    :class:`sklearn.neighbors.LocalOutlierFactor`, with the same sign
    conventions. It is not a full estimator: there is no ``novelty`` mode,
    no ``get_params``/``set_params``, and no metric selection.

    Parameters
    ----------
    n_neighbors : int, default=20
        Neighbours used to measure local density. Must satisfy
        ``1 <= n_neighbors <= n_samples - 1``.
    contamination : {'auto'} or float, default='auto'
        Expected proportion of outliers, used to set :attr:`offset_`. ``'auto'``
        uses a threshold of -1.5, matching scikit-learn.
    normalize : bool, default=False
        Z-score each feature before computing distances.

    Attributes
    ----------
    negative_outlier_factor_ : ndarray of shape (n_samples,)
        Negated LOF of the training samples. The *more negative*, the more
        abnormal -- scikit-learn's convention.
    offset_ : float
        Threshold on :attr:`negative_outlier_factor_` used by :meth:`predict`.
    n_neighbors_ : int
        The value of ``n_neighbors`` actually used.

    Notes
    -----
    Like scikit-learn's, this estimator is transductive: it scores the set it is
    given as a whole. There is no meaningful prediction for unseen points, and
    passing new data recomputes from scratch over just that data.

    Examples
    --------
    >>> import numpy as np
    >>> from culof import LOF
    >>> X = np.random.default_rng(0).standard_normal((10_000, 8))
    >>> labels = LOF(n_neighbors=20, contamination=0.01).fit_predict(X)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: str | float = "auto",
        normalize: bool = False,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if contamination != "auto" and not 0.0 < float(contamination) <= 0.5:
            raise ValueError(f"contamination must be 'auto' or in (0, 0.5], got {contamination!r}")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.normalize = normalize

    def fit(self, X: ArrayLike, y: object = None) -> LOF:
        """Compute the LOF of every sample in ``X``. Returns ``self``."""
        del y  # present for scikit-learn API compatibility
        scores = lof(X, self.n_neighbors, self.normalize)
        self.n_neighbors_ = self.n_neighbors
        self.negative_outlier_factor_ = -scores
        self.offset_ = (
            -1.5
            if self.contamination == "auto"
            else float(
                np.percentile(self.negative_outlier_factor_, 100.0 * float(self.contamination))
            )
        )
        return self

    def fit_predict(self, X: ArrayLike, y: object = None) -> NDArray[np.intp]:
        """Fit and return a label per sample: ``-1`` for outliers, ``1`` for inliers."""
        del y
        return self.fit(X).predict()

    def score_samples(self) -> NDArray[np.float32]:
        """Opposite of the LOF, for the fitted data. The *lower*, the more abnormal.

        Matching scikit-learn, this is negated relative to the published LOF
        definition. Use :func:`culof.lof` for the value itself.

        This takes no ``X``. It previously accepted one and recomputed LOF over
        that array alone, scoring the new points against each other rather than
        against the fitted population, then applied the offset learned during
        ``fit``. That is a different quantity from scikit-learn's novelty mode
        and it looked like a valid answer, so the argument is gone rather than
        silently misleading.
        """
        return self._check_fitted()

    def predict(self) -> NDArray[np.intp]:
        """Label the fitted samples ``-1`` (outlier) or ``1`` (inlier)."""
        return np.where(self.score_samples() < self._check_offset(), -1, 1)

    def _check_fitted(self) -> NDArray[np.float32]:
        if not hasattr(self, "negative_outlier_factor_"):
            raise ValueError(
                "this LOF instance is not fitted yet - call fit() or fit_predict() first"
            )
        return self.negative_outlier_factor_

    def _check_offset(self) -> float:
        self._check_fitted()
        return self.offset_

    def __repr__(self) -> str:
        return (
            f"LOF(n_neighbors={self.n_neighbors}, "
            f"contamination={self.contamination!r}, normalize={self.normalize})"
        )
