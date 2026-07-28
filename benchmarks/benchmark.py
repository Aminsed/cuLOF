#!/usr/bin/env python3
"""Benchmark cuLOF against scikit-learn's LocalOutlierFactor.

Every timed configuration is also checked for agreement with scikit-learn, so
the accuracy column and the speed column come from the same run -- a speedup on
a wrong answer is not a speedup.

Usage
-----
    python benchmarks/benchmark_lof.py                      # default sweep
    python benchmarks/benchmark_lof.py --max-n 200000       # push the size sweep
    python benchmarks/benchmark_lof.py --json results.json  # machine-readable

Notes
-----
CUDA context creation costs a few hundred milliseconds and happens once per
process. It is measured and reported separately rather than amortised into the
per-call numbers, which would flatter the GPU on small inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

import culof

# Agreement limits for the check below. float32 against float64 moves a
# score by ~1% only where the k-th and (k+1)-th neighbours are closer than
# float32 can separate; the ranking of the top 1% has never moved.
MAX_REL_ERR = 0.05
MIN_RANK_AGREEMENT = 1.0

@dataclass
class Row:
    # Which sweep produced this row. Both sweeps share the (20k, d=8) point, so
    # the figures need this to tell them apart.
    sweep: str
    n_samples: int
    n_features: int
    k: int
    sklearn_s: float
    culof_s: float
    speedup: float
    max_rel_err: float
    rank_agreement: float


def make_data(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Gaussian clusters with 1% uniformly scattered outliers."""
    rng = np.random.default_rng(seed)
    n_out = max(1, n // 100)
    centers = rng.uniform(-8, 8, size=(4, d))
    labels = rng.integers(0, 4, size=n - n_out)
    inliers = (centers[labels] + rng.standard_normal((n - n_out, d))).astype(np.float32)
    outliers = rng.uniform(-20, 20, size=(n_out, d)).astype(np.float32)
    return np.vstack([outliers, inliers]).astype(np.float32)


def timeit(fn, repeats: int) -> float:
    """Fastest of `repeats` calls, after one warm-up.

    The minimum rather than the mean or median, for the reason timeit's own
    documentation gives: runs slower than the fastest are slower because of
    interference from other processes, not because the code varies. Both
    libraries are measured the same way, so the comparison stays fair, and the
    result is far more reproducible on a machine that is doing anything else.
    """
    fn()
    return min(_time_once(fn) for _ in range(repeats))


def _time_once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_one(sweep: str, n: int, d: int, k: int, repeats: int, n_jobs: int) -> Row:
    X = make_data(n, d)

    def run_sklearn() -> np.ndarray:
        m = LocalOutlierFactor(n_neighbors=k, n_jobs=n_jobs)
        m.fit(X)
        return -m.negative_outlier_factor_

    def run_culof() -> np.ndarray:
        return culof.lof(X, k)

    t_sk = timeit(run_sklearn, repeats)
    t_cu = timeit(run_culof, repeats)

    s_sk, s_cu = run_sklearn(), run_culof()
    rel = float(np.max(np.abs(s_cu - s_sk) / np.maximum(np.abs(s_sk), 1e-9)))

    top = max(1, n // 100)
    agree = len(set(np.argsort(-s_cu)[:top]) & set(np.argsort(-s_sk)[:top])) / top

    # Enforced, not merely reported. A speedup on a wrong answer is not a
    # speedup, and until this raised, a badly wrong build would still have
    # printed a table and written its JSON.
    if not np.isfinite(rel) or rel > MAX_REL_ERR or agree < MIN_RANK_AGREEMENT:
        raise RuntimeError(
            f"disagreement with scikit-learn at n={n}, d={d}, k={k}: "
            f"max relative error {rel:.3e} (limit {MAX_REL_ERR:.0e}), "
            f"top-1% rank agreement {agree:.3f} (limit {MIN_RANK_AGREEMENT:.2f})"
        )

    return Row(sweep, n, d, k, t_sk, t_cu, t_sk / t_cu, rel, agree)


def gpu_name() -> str:
    try:
        return culof.device_info()
    except Exception:  # pragma: no cover
        return "unknown"


def cpu_name() -> str:
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover
        pass
    return platform.processor() or "unknown"


ROW_FMT = "{n:>8} {d:>4} {sk:>12.4f} {cu:>11.4f} {sp:>8.1f}x {err:>12.2e} {agr:>12.0%}"
HEADER = (
    f"{'n':>8} {'d':>4} {'sklearn (s)':>12} {'cuLOF (s)':>11} "
    f"{'speedup':>9} {'max rel err':>12} {'top-1% agree':>13}"
)


def show(r: Row) -> None:
    print(
        ROW_FMT.format(
            n=r.n_samples,
            d=r.n_features,
            sk=r.sklearn_s,
            cu=r.culof_s,
            sp=r.speedup,
            err=r.max_rel_err,
            agr=r.rank_agreement,
        )
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--max-n", type=int, default=50_000)
    p.add_argument("--n-jobs", type=int, default=-1, help="cores for scikit-learn")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    if not culof.cuda_available():
        print("No CUDA device available.", file=sys.stderr)
        return 1

    # One-time CUDA context cost, measured before anything else touches the GPU.
    warm = np.zeros((64, 2), dtype=np.float32)
    warm[:, 0] = np.arange(64)
    t0 = time.perf_counter()
    culof.lof(warm, 5)
    init_s = time.perf_counter() - t0

    print(f"GPU : {gpu_name()}")
    print(f"CPU : {cpu_name()}")
    print(f"cuLOF {culof.__version__}, k={args.k}, best of {args.repeats} runs")
    print(f"one-time CUDA context init: {init_s * 1000:.0f} ms (excluded below)\n")

    sizes = [
        s
        for s in (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
        if s <= args.max_n
    ]

    rows: list[Row] = []

    print("Scaling with sample count (d = 8)")
    print(HEADER)
    print("-" * len(HEADER))
    for n in sizes:
        r = bench_one("samples", n, 8, args.k, args.repeats, args.n_jobs)
        rows.append(r)
        show(r)

    n_dim_sweep = min(20_000, args.max_n)
    print(f"\nScaling with dimensionality (n = {n_dim_sweep:,})")
    print(HEADER)
    print("-" * len(HEADER))
    for d in (2, 8, 32, 128):
        r = bench_one("features", n_dim_sweep, d, args.k, args.repeats, args.n_jobs)
        rows.append(r)
        show(r)

    if args.json:
        payload = {
            "gpu": gpu_name(),
            "cpu": cpu_name(),
            "culof_version": culof.__version__,
            "k": args.k,
            "repeats": args.repeats,
            "init_seconds": init_s,
            "rows": [asdict(r) for r in rows],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
