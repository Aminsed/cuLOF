"""One test per defect that shipped in an earlier release.

Kept separate from the parity and API suites so it stays obvious why each of
these exists, and so that removing one is a deliberate act.
"""

from __future__ import annotations

import numpy as np
from conftest import sklearn_lof

import culof
from culof import lof


def test_output_is_not_a_constant() -> None:
    """pybind11 < 2.12 built against NumPy >= 2 returned stride-0 arrays.

    Every element aliased element 0, so the library silently produced one
    repeated value for every input. CMake now enforces the version floor; this
    guards the observable symptom.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 2)).astype(np.float32)
    X[:5] += 8.0

    scores = lof(X, 20)
    assert scores.strides == (scores.itemsize,), "returned array has degenerate strides"
    assert scores.std() > 0.0
    assert len(np.unique(scores)) > 400


def test_non_contiguous_input() -> None:
    """The old binding tested only strides[1], then passed the raw pointer, so a
    row-sliced view was read as though it were contiguous."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((2000, 3)).astype(np.float32)
    base[:20] += 8.0

    view = base[::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(lof(view, 20), lof(np.ascontiguousarray(view), 20))


def test_fortran_order_input() -> None:
    rng = np.random.default_rng(2)
    X = np.asfortranarray(rng.standard_normal((800, 4)).astype(np.float32))
    np.testing.assert_allclose(lof(X, 20), lof(np.ascontiguousarray(X), 20))


def test_float64_input_is_converted() -> None:
    rng = np.random.default_rng(3)
    X64 = rng.standard_normal((600, 3))
    np.testing.assert_allclose(lof(X64, 15), lof(X64.astype(np.float32), 15), rtol=1e-5)


def test_duplicate_points_do_not_collapse() -> None:
    """With many exact duplicates the old code returned a single repeated score."""
    rng = np.random.default_rng(4)
    X = np.repeat(rng.standard_normal((50, 2)).astype(np.float32), 20, axis=0)
    X[:5] += 5.0

    scores = lof(X, 20)
    assert np.isfinite(scores).all()
    assert len(np.unique(scores)) > 1
    assert np.max(np.abs(scores - sklearn_lof(X, 20))) < 1e-2


def test_normalize_path_is_correct() -> None:
    """normalize=True used to memcpy a device pointer with cudaMemcpyHostToDevice."""
    rng = np.random.default_rng(5)
    X = (rng.standard_normal((2000, 4)) * [1.0, 50.0, 0.1, 5.0]).astype(np.float32)
    X[:20] += 8.0

    Z = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    got, want = lof(X, 20, normalize=True), sklearn_lof(Z, 20)
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9)) < 2e-4


def test_constant_feature_with_normalize() -> None:
    """Dividing a zero-variance column by its standard deviation used to produce
    inf or NaN."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((500, 3)).astype(np.float32)
    X[:, 1] = 7.0
    assert np.isfinite(lof(X, 20, normalize=True)).all()


def test_far_from_origin_is_numerically_stable() -> None:
    """The GEMM distance identity cancels badly when |x|^2 dwarfs |x-y|^2.

    cuLOF centres internally to hold that in check. The comparison is against
    scikit-learn on the *same* shifted array, so the float32 quantisation of the
    shifted coordinates -- a property of the input, not of cuLOF -- hits both
    sides equally.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1500, 3)).astype(np.float32)
    X[:15] = rng.uniform(-15, 15, size=(15, 3)).astype(np.float32)
    shifted = (X + 5000.0).astype(np.float32)

    got, want = lof(shifted, 20), sklearn_lof(shifted, 20)
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-9)) < 2e-4


def test_version_has_one_source() -> None:
    """__version__ once disagreed between setup.py, the binding and __init__."""
    import contextlib
    from importlib.metadata import PackageNotFoundError, version

    assert culof.__version__
    # PackageNotFoundError just means the tests are running against a source
    # tree rather than an installed distribution; there is nothing to compare.
    with contextlib.suppress(PackageNotFoundError):
        assert version("culof") == culof.__version__
