<h1 align="center">cuLOF</h1>

<p align="center">
  <strong>Local Outlier Factor on the GPU.</strong><br>
  Exact LOF scores, matching scikit-learn's sign conventions,<br>
  for the <code>fit_predict</code> workflow.
</p>

<p align="center">
  <a href="https://github.com/Aminsed/cuLOF/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Aminsed/cuLOF/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="CUDA 11.0+" src="https://img.shields.io/badge/CUDA-11.0%2B-76B900">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB">
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-blue"></a>
</p>

---

```diff
- from sklearn.neighbors import LocalOutlierFactor as LOF
+ from culof import LOF

  labels = LOF(n_neighbors=20, contamination=0.01).fit_predict(X)
```

That transductive workflow — score every point against the dataset it came
in with — is the one cuLOF covers, with the same sign conventions and the same
`negative_outlier_factor_` and `offset_` attributes.

**It is not a drop-in replacement for the estimator.** `algorithm`,
`leaf_size`, `metric`, `p`, `metric_params`, `novelty` and `n_jobs` are not
accepted; sparse and precomputed input are not supported; and `get_params`,
`set_params`, `kneighbors` and `decision_function` do not exist, so it will not
go into a `Pipeline` or a grid search. There is no novelty mode: `predict` and
`score_samples` operate on the fitted data only.

Agreement on what it does cover is close rather than exact — see
[Accuracy](#accuracy) for the measured distribution and where float32 ties can
move a score.

## Performance

![Speedup over scikit-learn](https://raw.githubusercontent.com/Aminsed/cuLOF/main/img/benchmark_speed.png)

| samples | scikit-learn | cuLOF | speedup |
|--------:|-------------:|------:|--------:|
| 1,000   | 0.013 s  | 0.0002 s | **63×** |
| 5,000   | 0.096 s  | 0.0030 s | **32×** |
| 20,000  | 0.659 s  | 0.0221 s | **30×** |
| 50,000  | 2.723 s  | 0.0953 s | **29×** |
| 200,000 | 16.66 s  | 1.409 s  | **12×** |

RTX A6000 vs 24-thread i9-12900K, 8 features, k=20, best of 7 runs, one-time
CUDA context initialisation excluded. Reproduce with:

```bash
python benchmarks/benchmark.py --max-n 200000 --json bench.json
```

Every row is also checked against scikit-learn in the same run — a speedup on a
wrong answer is not a speedup.

## Why it is fast

**Distances are one GEMM.** Expanding `‖a-b‖² = ‖a‖² + ‖b‖² - 2a·b` turns the
O(n²d) distance computation into a single cuBLAS `SGEMM`, which runs at a large
fraction of peak FP32 instead of the few percent a hand-written distance loop
reaches. The identity is numerically delicate — it cancels when `‖a‖²` dwarfs
`‖a-b‖²` — so cuLOF centres the data first, which leaves every distance
unchanged while making that term as small as it can be.

**k-NN by selection, not sorting.** LOF needs the *set* of k neighbours and the
k-distance; it never needs them ordered. So cuLOF selects. For non-negative
IEEE-754 floats the integer ordering of the bit pattern is the numeric ordering,
so a radix select on the raw bits is exact — and the k-th key it produces *is*
the k-distance, for free. Three passes of 11 bits pin it down; a fourth gathers
the neighbours.

|                | insertion sort (before) | radix select (now) |
|----------------|-------------------------|--------------------|
| maximum k      | 32, a fixed register array | unbounded |
| work per row   | O(n·k)                  | O(n) per pass |
| memory access  | stride-n across lanes   | fully coalesced |

**The n×n matrix is never materialised.** Rows are processed in tiles sized to
free device memory, so memory grows linearly in n for fixed k. The neighbour
arrays are a separate O(n·k). At n = 200,000 a dense float32 distance matrix
would be 149 GiB.

## Accuracy

![Agreement with scikit-learn](https://raw.githubusercontent.com/Aminsed/cuLOF/main/img/accuracy.png)

cuLOF computes distances in float32 and scikit-learn in float64, so agreement is
measured rather than assumed. At n = 20,000, d = 8, k = 20: **median relative
difference 4.5×10⁻⁷**, 99th percentile 2.4×10⁻⁶, and identical top-1% rankings in
every benchmarked configuration.

The tail belongs to points whose k-th and (k+1)-th neighbours are closer
together than float32 can resolve. There, which neighbour is "the k-th" is
genuinely ambiguous and the two libraries may choose differently; that point's
score can move by up to ~1%. This affects 0.27% of points, and never changed
which points were flagged.
[`test_near_ties_are_the_only_disagreement`](tests/python/test_parity.py) asserts
exactly this.

![Detection on three datasets](https://raw.githubusercontent.com/Aminsed/cuLOF/main/img/detection.png)

## Install

```bash
pip install culof
```

Needs a CUDA toolkit (11.0+) with `nvcc` on `PATH`, CMake 3.18+, and an NVIDIA
GPU of compute capability 7.0 or newer. The build detects your GPU's
architecture automatically. See [docs/install.md](docs/install.md) for offline
builds, fat binaries, and troubleshooting.

## API

```python
from culof import LOF, lof

# The estimator: mirrors sklearn.neighbors.LocalOutlierFactor
model = LOF(n_neighbors=20, contamination="auto", normalize=False)
labels = model.fit_predict(X)  # -1 outlier, 1 inlier
model.negative_outlier_factor_  # negated LOF, sklearn's convention
model.offset_  # the threshold predict() applies

# The function: the LOF value itself, ~1.0 for a normal point
scores = lof(X, k=20)
```

`k` has no upper bound beyond `n_samples - 1`. The *selection* scan is
independent of k, but neighbour indices and distances are O(n·k) and both
downstream kernels loop over all k, so large k does cost time and memory. At
k = 20 the O(n²d) distance stage dominates and moderate changes to k barely
register. Full walkthrough in [docs/usage.md](docs/usage.md).

## Limitations

- **Brute force.** cuLOF computes all n² distances (measured: n^1.94 over
  n ≥ 50,000). scikit-learn uses a KD-tree in low dimensions, which never looks
  at most pairs (n^1.31 over the same range). Better asymptotics beat a constant
  hardware factor eventually, so the low-dimensional speedup decays as roughly
  n^-0.64. This is dimension-specific:
  at d = 128, where KD-trees collapse and scikit-learn falls back to brute force,
  both scale as n^1.9 and the ratio is flat at ~15× from n = 5,000 to 100,000.
  Treat ~15× as the steady-state hardware win and the low-d figures as additionally
  reflecting tree-build overhead scikit-learn pays and cuLOF does not.
  [Measured tables and the reasoning.](docs/implementation.md#why-brute-force-and-why-the-speedup-narrows)
- **Below ~1,000 points**, the one-off ~110 ms CUDA context initialisation
  dominates and scikit-learn is the better tool.
- **Transductive**, exactly like scikit-learn's: it scores the set it is given.
  There is no `predict` for unseen points.
- **float32 only**, with the precision boundary characterised above.

## Documentation

| | |
|---|---|
| [docs/install.md](docs/install.md) | Requirements, architectures, troubleshooting |
| [docs/usage.md](docs/usage.md) | Choosing k, interpreting scores, migration |
| [docs/implementation.md](docs/implementation.md) | Algorithm, kernels, determinism, memory |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development, testing, benchmarking |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Testing

```bash
pip install -e ".[test]" && pytest tests/python     # needs a GPU
cmake -B build -DCULOF_BUILD_TESTS=ON && cmake --build build -j
./build/culof_tests
```

The C++ suite checks every stage against an independent double-precision CPU
implementation of LOF; the Python suite checks agreement with scikit-learn and
carries one regression test per defect that shipped in an earlier release.

## References

Breunig, Kriegel, Ng, Sander. *LOF: Identifying Density-Based Local Outliers.*
SIGMOD 2000.

## License

MIT — see [LICENSE](LICENSE).
