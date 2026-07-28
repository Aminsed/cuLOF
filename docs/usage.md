# Usage

## Migrating from scikit-learn

```diff
- from sklearn.neighbors import LocalOutlierFactor as LOF
+ from culof import LOF

  model = LOF(n_neighbors=20, contamination=0.01)
  labels = model.fit_predict(X)            # -1 outlier, 1 inlier
  scores = model.negative_outlier_factor_  # more negative = more abnormal
```

Constructor arguments, methods, attributes and sign conventions all match, so
nothing downstream needs to change. The one addition is `normalize=True`, which
z-scores each feature before computing distances; scikit-learn has no
equivalent, so leave it off when comparing results directly.

## Two ways in

```python
from culof import LOF, lof

# The estimator, for pipelines and for scikit-learn's conventions.
labels = LOF(n_neighbors=20, contamination=0.01).fit_predict(X)

# The function, when you want the LOF value itself.
scores = lof(X, k=20)  # ~1.0 for a normal point, higher = more anomalous
```

The two differ only in sign and framing: `model.negative_outlier_factor_` is
exactly `-lof(X, k)`.

## Interpreting the score

LOF is the ratio between the local density around a point's neighbours and the
local density around the point itself.

| `lof()` value | meaning |
|---|---|
| ≈ 1.0 | as densely surrounded as its neighbours — normal |
| 1.5 – 2 | noticeably sparser neighbourhood — worth a look |
| > 2 | much sparser than its neighbours — a strong candidate |

There is no universal cutoff, which is why `contamination` exists:

```python
LOF(contamination="auto")  # threshold at -1.5, scikit-learn's default
LOF(contamination=0.01)  # flag the most extreme 1%
```

`model.offset_` is the resulting threshold on `negative_outlier_factor_`.

## Choosing n_neighbors

This is the parameter that matters. It sets the scale at which density is
measured.

- **Too small** (below ~10): scores get noisy, and small groups of duplicated
  points start to look like dense clusters.
- **Too large**: genuinely small clusters are absorbed into their surroundings
  and stop looking anomalous.
- **20** is a sound default, and is scikit-learn's.

One consequence worth internalising: a clump of `m` mutually close points where
`m > n_neighbors` scores near 1.0, because those points are each other's
neighbours. LOF measures *local* density, so a dense cluster of outliers is not
an outlier. If you expect anomalies to arrive in groups, set `n_neighbors`
larger than the group size.

`n_neighbors` may be anything from 1 to `n_samples - 1`. The selection scan
itself is independent of k, but the neighbour indices and distances are O(n·k)
and both the density and score kernels loop over all k, so a large value is not
free. At k = 20 the O(n²d) distance stage dominates and moderate changes barely
register; at k in the thousands, memory and post-processing both grow.

## Input handling

Anything array-like is accepted and converted to C-contiguous float32:

```python
lof(df.to_numpy(), k=20)  # pandas
lof(X_float64, k=20)  # converted
lof(np.asfortranarray(X), k=20)  # converted
lof(X[::2], k=20)  # non-contiguous view, handled
```

Passing float32 that is already C-contiguous avoids one copy. `NaN` and infinity
raise `ValueError` rather than silently producing nonsense.

## Memory and large inputs

Memory grows linearly with `n`: the distance matrix is computed in tiles and
never materialised. In practice you run out of time before memory.

If you share the GPU with another process, cap the scratch tile:

```bash
CULOF_TILE_ROWS=256 python your_script.py
```

Smaller values use less memory and are slightly slower. The result does not
change.

## Performance notes

- The first call in a process pays a one-off CUDA context initialisation of
  roughly 110 ms. Benchmark from the second call onwards.
- Below about 1,000 points that fixed cost dominates, and scikit-learn is the
  better tool.
- The GIL is released during computation, so calls from a worker thread do not
  block other Python threads.

## Reproducibility

For a given input and configuration, cuLOF returns bit-identical scores on every
run. The neighbour gather uses a prefix sum rather than atomics specifically to
guarantee this — see
[implementation.md](implementation.md#determinism) for why that is not automatic.
