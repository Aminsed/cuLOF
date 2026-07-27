#!/usr/bin/env python3
"""Regenerate every figure in img/.

Performance figures are drawn from the JSON emitted by
``benchmarks/benchmark_lof.py``, so the plots and the README table are always
the same numbers from the same run. Qualitative figures are computed here.

Usage
-----
    python benchmarks/benchmark_lof.py --max-n 200000 --json bench.json
    python scripts/generate_figures.py --results bench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import LocalOutlierFactor

import culof

# NVIDIA green for the GPU series, a muted slate for the CPU baseline.
CULOF = "#76B900"
SKLEARN = "#546778"
ACCENT = "#C8102E"
GRID = "#DDE2E6"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#8A9299",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.color": "#3C4348",
        "ytick.color": "#3C4348",
        "axes.labelcolor": "#20262A",
        "text.color": "#20262A",
    }
)

IMG = Path(__file__).resolve().parent.parent / "img"


def _tidy(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _annotate_env(fig: plt.Figure, meta: dict) -> None:
    fig.text(
        0.5,
        -0.035,
        f"{meta.get('gpu', 'GPU')}  vs  {meta.get('cpu', 'CPU')} "
        f"(scikit-learn, all cores)   |   cuLOF {meta.get('culof_version', '')}, "
        f"k={meta.get('k', '')}, best of {meta.get('repeats', '')} runs",
        ha="center",
        fontsize=8,
        color="#68727A",
    )


# --------------------------------------------------------------- performance


def _sweep(meta: dict, name: str) -> list[dict]:
    """Rows from one sweep.

    Both sweeps contain the (20k, d=8) configuration, so filtering on shape
    alone would duplicate it. Older result files predate the `sweep` field, so
    fall back to shape with a de-duplication.
    """
    rows = meta["rows"]
    if any("sweep" in r for r in rows):
        return [r for r in rows if r.get("sweep") == name]

    key = "n_features" if name == "samples" else "n_samples"
    fixed = 8 if name == "samples" else 20_000
    seen: dict[int, dict] = {}
    for r in rows:
        if r[key] == fixed:
            seen[r["n_samples" if name == "samples" else "n_features"]] = r
    return list(seen.values())


def figure_speed(meta: dict) -> None:
    rows = _sweep(meta, "samples")
    rows.sort(key=lambda r: r["n_samples"])
    n = np.array([r["n_samples"] for r in rows])
    sk = np.array([r["sklearn_s"] for r in rows])
    cu = np.array([r["culof_s"] for r in rows])
    sp = sk / cu

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

    a.loglog(n, sk, "o-", color=SKLEARN, lw=2, ms=6, label="scikit-learn (24-core CPU)")
    a.loglog(n, cu, "o-", color=CULOF, lw=2.4, ms=6, label="cuLOF (RTX A6000)")
    a.set_xlabel("samples")
    a.set_ylabel("time (s)")
    a.set_title("Runtime, 8 features")
    a.legend(loc="upper left")
    _tidy(a)

    b.semilogx(n, sp, "o-", color=CULOF, lw=2.4, ms=6)
    b.axhline(1.0, color=ACCENT, ls="--", lw=1, alpha=0.7)
    b.text(n[0], 1.06, "parity", color=ACCENT, fontsize=8, va="bottom")
    for x, y in zip(n, sp):
        b.annotate(
            f"{y:.0f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#3C4348",
        )
    b.set_xlabel("samples")
    b.set_ylabel("speedup (x)")
    b.set_title("Speedup over scikit-learn")
    b.set_ylim(0, max(sp) * 1.25)
    _tidy(b)

    _annotate_env(fig, meta)
    fig.tight_layout()
    fig.savefig(IMG / "benchmark_speed.png")
    plt.close(fig)


def figure_dims(meta: dict) -> None:
    rows = _sweep(meta, "features")
    rows.sort(key=lambda r: r["n_features"])
    if not rows:
        return
    d = np.array([r["n_features"] for r in rows])
    sk = np.array([r["sklearn_s"] for r in rows])
    cu = np.array([r["culof_s"] for r in rows])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    idx = np.arange(len(d))
    w = 0.38
    a.bar(idx - w / 2, sk, w, color=SKLEARN, label="scikit-learn")
    a.bar(idx + w / 2, cu, w, color=CULOF, label="cuLOF")
    a.set_xticks(idx)
    a.set_xticklabels([str(x) for x in d])
    a.set_xlabel("features")
    a.set_ylabel("time (s)")
    a.set_title("Runtime at 20,000 samples")
    a.legend()
    _tidy(a)

    b.bar(idx, sk / cu, 0.55, color=CULOF)
    b.axhline(1.0, color=ACCENT, ls="--", lw=1, alpha=0.7)
    for i, v in enumerate(sk / cu):
        b.annotate(
            f"{v:.1f}x",
            (i, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color="#3C4348",
        )
    b.set_xticks(idx)
    b.set_xticklabels([str(x) for x in d])
    b.set_xlabel("features")
    b.set_ylabel("speedup (x)")
    b.set_title("Speedup vs feature count")
    _tidy(b)

    fig.tight_layout()
    fig.text(
        0.5,
        -0.03,
        "scikit-learn switches between KD-tree and brute force by dimensionality, "
        "so its CPU baseline is not monotonic in d.",
        ha="center",
        fontsize=8,
        color="#68727A",
    )
    fig.text(
        0.5,
        -0.08,
        f"{meta.get('gpu', 'GPU')}  vs  {meta.get('cpu', 'CPU')} "
        f"(scikit-learn, all cores)   |   cuLOF {meta.get('culof_version', '')}, "
        f"k={meta.get('k', '')}, best of {meta.get('repeats', '')} runs",
        ha="center",
        fontsize=8,
        color="#68727A",
    )
    fig.savefig(IMG / "benchmark_dimensions.png")
    plt.close(fig)


# ------------------------------------------------------------------ accuracy


def figure_accuracy() -> None:
    rng = np.random.default_rng(7)
    n, d, k = 20_000, 8, 20
    n_out = n // 100
    centers = rng.uniform(-8, 8, size=(4, d))
    lab = rng.integers(0, 4, size=n - n_out)
    X = np.vstack(
        [
            rng.uniform(-20, 20, size=(n_out, d)),
            centers[lab] + rng.standard_normal((n - n_out, d)),
        ]
    ).astype(np.float32)

    got = culof.lof(X, k)
    want = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-9)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

    a.scatter(want, got, s=5, alpha=0.25, color=CULOF, edgecolors="none")
    lim = [min(want.min(), got.min()) * 0.98, max(want.max(), got.max()) * 1.02]
    a.plot(lim, lim, ls="--", lw=1, color=ACCENT)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("scikit-learn LOF score")
    a.set_ylabel("cuLOF score")
    a.set_title(f"Score agreement (n={n:,}, d={d}, k={k})")
    a.annotate(
        f"Pearson r = {np.corrcoef(got, want)[0, 1]:.6f}",
        (0.05, 0.92),
        xycoords="axes fraction",
        fontsize=9,
    )
    _tidy(a)

    srt = np.sort(rel)
    cdf = np.arange(1, len(srt) + 1) / len(srt)
    b.semilogx(np.maximum(srt, 1e-9), cdf * 100, color=CULOF, lw=2)
    for q in (np.median(rel), np.percentile(rel, 99)):
        b.axvline(q, color=SKLEARN, ls=":", lw=1)
    b.set_xlabel("relative difference vs scikit-learn")
    b.set_ylabel("percent of points below")
    b.set_title("Agreement distribution")
    b.set_ylim(0, 101)
    stats = (
        f"median   {np.median(rel):.1e}\n"
        f"p99      {np.percentile(rel, 99):.1e}\n"
        f"max      {rel.max():.1e}\n"
        f"above 1e-4  {np.mean(rel > 1e-4):.2%} of points"
    )
    b.text(
        0.03,
        0.97,
        stats,
        transform=b.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color="#3C4348",
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.45"},
    )
    _tidy(b)

    fig.text(
        0.5,
        -0.05,
        "cuLOF computes distances in float32, scikit-learn in float64. The tail "
        "is points whose k-th and (k+1)-th neighbour are closer together than "
        "float32 can resolve;\nthere the choice of k-th neighbour is genuinely "
        "ambiguous. Outlier ranking is unaffected.",
        ha="center",
        fontsize=8,
        color="#68727A",
    )
    fig.tight_layout()
    fig.savefig(IMG / "accuracy.png")
    plt.close(fig)


# --------------------------------------------------------------- qualitative


def figure_detection() -> None:
    from matplotlib.colors import LogNorm

    rng = np.random.default_rng(3)
    k = 20

    def with_outliers(X: np.ndarray, m: int, span: float):
        out = rng.uniform(-span, span, size=(m, 2))
        both = np.vstack([X, out]).astype(np.float32)
        truth = np.zeros(len(both), dtype=bool)
        truth[len(X) :] = True
        return both, truth

    blobs, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=1)
    moons, _ = make_moons(n_samples=600, noise=0.06, random_state=1)
    dense = np.vstack(
        [
            rng.standard_normal((400, 2)) * 0.3,
            rng.standard_normal((200, 2)) * 1.6 + np.array([6.0, 0.0]),
        ]
    )

    datasets = [
        ("Blobs", *with_outliers(blobs, 40, 12.0)),
        ("Two moons", *with_outliers(moons, 40, 2.6)),
        ("Varying density", *with_outliers(dense, 40, 9.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))
    notes = []
    for col, (name, X, truth) in enumerate(datasets):
        cu = culof.lof(X, k)
        sk = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
        n_out = int(truth.sum())
        flagged = np.argsort(-cu)[:n_out]

        top_cu = set(np.argsort(-cu)[:n_out])
        top_sk = set(np.argsort(-sk)[:n_out])
        notes.append(
            f"{name}: recall {len(set(flagged) & set(np.flatnonzero(truth))) / n_out:.0%}"
            f", same points as scikit-learn: {'yes' if top_cu == top_sk else 'no'}"
        )

        # Ground truth
        ax = axes[0, col]
        ax.scatter(
            X[~truth, 0], X[~truth, 1], s=12, color="#B9C2C9", edgecolors="none", label="inlier"
        )
        ax.scatter(
            X[truth, 0], X[truth, 1], s=22, color=ACCENT, edgecolors="none", label="planted outlier"
        )
        ax.set_title(f"{name} - ground truth")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=1.2)

        # cuLOF scores. LOF concentrates near 1.0 with a long tail, so a log
        # colour scale is the only way to see structure among the inliers.
        ax = axes[1, col]
        sc = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=np.maximum(cu, 1.0),
            s=14,
            cmap="viridis",
            norm=LogNorm(vmin=1.0, vmax=float(np.percentile(cu, 99.5))),
            edgecolors="none",
        )
        ax.scatter(
            X[flagged, 0], X[flagged, 1], s=78, facecolors="none", edgecolors=ACCENT, linewidths=1.0
        )
        ax.set_title(f"cuLOF score, top {n_out} circled")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

        for r in (0, 1):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            axes[r, col].grid(False)

    fig.suptitle("Local Outlier Factor on the GPU", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.text(0.5, 0.022, "   |   ".join(notes), ha="center", fontsize=8, color="#3C4348")
    fig.text(
        0.5,
        0.002,
        f"k={k}. Colour is log-scaled: LOF concentrates just above 1.0 with a long "
        f"tail, so a linear scale would render every inlier the same shade.",
        ha="center",
        fontsize=8,
        color="#68727A",
    )
    fig.savefig(IMG / "detection.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        type=str,
        default=None,
        help="JSON from benchmark_lof.py; skips perf figures if omitted",
    )
    args = ap.parse_args()

    IMG.mkdir(exist_ok=True)

    if args.results:
        meta = json.loads(Path(args.results).read_text())
        figure_speed(meta)
        figure_dims(meta)
        print("wrote img/benchmark_speed.png, img/benchmark_dimensions.png")

    figure_accuracy()
    print("wrote img/accuracy.png")
    figure_detection()
    print("wrote img/detection.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
