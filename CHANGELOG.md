# Changelog

## 0.2.0

A correctness release. Version 0.1.1 returned a single repeated value for every
input on any system with NumPy 2.x, which is to say it did not work.

### Fixed

- **Results were a constant.** `python/CMakeLists.txt` pinned pybind11 v2.10.4,
  which predates NumPy 2.0's change to the array descriptor layout. Built
  against NumPy ≥ 2, every returned array had a stride of 0, so all elements
  aliased element 0. The kernels were computing correct values; the binding
  discarded them. CMake now requires pybind11 ≥ 2.12.
- **Every build targeted sm_52.** `project(LANGUAGES CUDA)` seeds
  `CMAKE_CUDA_ARCHITECTURES`, so the architecture-detection function's
  `if(DEFINED ...)` guard always returned early. Detection now uses `nvidia-smi`
  and can be overridden with `CULOF_CUDA_ARCHITECTURES` or `CUDAARCHS`.
- **`normalize=True` was undefined behaviour.** It issued a
  `cudaMemcpyHostToDevice` on a pointer that was already on the device, and
  called `cudaHostRegister` on device memory.
- **Non-contiguous input silently gave wrong answers.** The binding tested only
  `strides[1]` before passing the raw pointer, so a row-sliced view such as
  `X[::2]` was read as though it were contiguous.
- **`k > 32` raised `cudaErrorInvalidValue`.** The selection kernel held
  candidates in a fixed 32-entry register array.
- Zero-variance features produced `inf`/`NaN` under `normalize=True`.
- `--use_fast_math` was applied globally, contradicting the documented aim of
  matching scikit-learn numerically.
- `BUILD_TESTS=ON` failed to configure: `tests/CMakeLists.txt` copied a
  `tests/data` directory that does not exist.
- The build downloaded pybind11 from GitHub on every configure, because it
  tested for a *target* rather than calling `find_package`.
- `__version__` disagreed across `setup.py`, the binding and `__init__.py`. It
  now has one source, `culof/_version.py`.

### Changed

- **`LOF` is now a drop-in replacement for
  `sklearn.neighbors.LocalOutlierFactor`**: `n_neighbors`, `contamination`,
  `fit`, `fit_predict` (returning ±1 labels), `score_samples`, `predict`,
  `negative_outlier_factor_`, `offset_`. The previous `set_k` / `set_normalize`
  / `set_threshold` / `get_outliers` surface is gone, as is the pure-Python
  `LOF` class that `__init__.py` defined and then immediately overwrote with the
  C++ one.
- `culof.lof(X, k)` returns the LOF value itself, for callers who do not want
  scikit-learn's negated convention.
- Distances now go through a single cuBLAS `SGEMM` on centred data, rather than
  an elementwise kernel accumulating in float64 — a 1:64 rate on consumer
  Ampere.
- k-NN is an exact radix select on the float bit patterns rather than a
  per-thread insertion sort: no k limit, coalesced reads, O(n) per pass.
- The n×n distance matrix is no longer materialised. Tiling keeps memory linear
  in n, so n = 200,000 now runs where it previously needed 149 GiB.
- Neighbour output is deterministic: the gather uses a block-wide prefix sum
  rather than atomics, so repeated runs are bit-identical.
- Packaging moved to `pyproject.toml` with scikit-build-core; `setup.py` is
  gone. matplotlib is no longer a runtime dependency.
- Minimum Python is 3.9 (was 3.6, end-of-life since 2021).

### Added

- `CULOF_TILE_ROWS` to override the distance-tile height.
- `culof.cuda_available()` and `culof.device_info()`.
- GitHub Actions CI across CUDA 11.8/12.4 and Python 3.9–3.12.
- A C++ suite checking every stage against an independent double-precision CPU
  implementation, and a Python suite checking scikit-learn parity plus one
  regression test per defect above.
- Benchmarks that verify agreement with scikit-learn in the same run that times
  it, and figures generated from that run's output so the plots and the README
  cannot drift apart.

### Removed

- The duplicated `python/` package tree, which shipped a second copy of the
  tests and a top-level module literally named `python`.
- Three mutually contradictory sets of benchmark numbers (the README claimed
  6.75×, one figure 4.6×, another 1.57×).
- The `culof::LOF` C++ class, which stored three configuration values and
  forwarded. `culof::lof()` is the C++ API.

## 0.1.1

Initial PyPI release.
