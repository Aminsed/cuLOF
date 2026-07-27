#!/usr/bin/env python3
"""Detect outliers, then check the result against scikit-learn.

python examples/basic.py
"""

from __future__ import annotations

import time

import numpy as np

import culof
from culof import LOF


def make_data(n: int = 20_000, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three Gaussian clusters plus 1% of points scattered over a wide box."""
    rng = np.random.default_rng(seed)
    n_out = n // 100
    centers = rng.uniform(-8, 8, size=(3, d))
    labels = rng.integers(0, 3, size=n - n_out)

    X = np.vstack(
        [
            centers[labels] + rng.standard_normal((n - n_out, d)),
            rng.uniform(-20, 20, size=(n_out, d)),
        ]
    ).astype(np.float32)

    truth = np.zeros(n, dtype=bool)
    truth[n - n_out :] = True
    return X, truth


def main() -> None:
    if not culof.cuda_available():
        raise SystemExit("This example needs a CUDA-capable GPU.")

    print(f"cuLOF {culof.__version__} on {culof.device_info()}\n")

    X, truth = make_data()
    print(f"{X.shape[0]:,} samples x {X.shape[1]} features, {truth.sum()} planted outliers")

    # Creating the CUDA context costs ~110 ms and happens once per process.
    # Time it separately rather than charging it to the first call, which would
    # be the single biggest distortion in a small benchmark like this.
    t0 = time.perf_counter()
    culof.lof(X[:64], 5)
    print(f"one-time CUDA context init: {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    model = LOF(n_neighbors=20, contamination=float(truth.mean()))

    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0

    flagged = labels == -1
    recall = (flagged & truth).sum() / truth.sum()
    print(f"scored in {elapsed * 1000:.1f} ms")
    print(f"flagged {flagged.sum()} points, recall {recall:.1%}\n")

    scores = -model.negative_outlier_factor_
    print("highest-scoring points:")
    for rank, i in enumerate(np.argsort(-scores)[:5], 1):
        kind = "planted outlier" if truth[i] else "inlier"
        print(f"  {rank}. index {i:6d}  LOF {scores[i]:7.3f}  {kind}")

    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        return

    reference = LocalOutlierFactor(n_neighbors=20, contamination=float(truth.mean()))
    t0 = time.perf_counter()
    ref_labels = reference.fit_predict(X)
    ref_elapsed = time.perf_counter() - t0

    ref_scores = -reference.negative_outlier_factor_
    rel = np.abs(scores - ref_scores) / np.maximum(np.abs(ref_scores), 1e-9)

    print(f"\nscikit-learn took {ref_elapsed * 1000:.1f} ms ({ref_elapsed / elapsed:.1f}x slower)")
    print(f"identical labels:           {np.array_equal(labels, ref_labels)}")
    print(f"median relative difference: {np.median(rel):.2e}")


if __name__ == "__main__":
    main()
