# Contributing

## Setup

```bash
git clone https://github.com/Aminsed/cuLOF.git
cd cuLOF
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,bench]"
```

Requires a CUDA toolkit (11.0+) with `nvcc` on `PATH`, CMake 3.18+, and an
NVIDIA GPU.

## Tests

```bash
pytest tests/python                                    # needs a GPU

cmake -B build -DCULOF_BUILD_TESTS=ON
cmake --build build -j
./build/culof_tests
```

The Python suite checks agreement with scikit-learn and carries one regression
test per defect that shipped in an earlier release. The C++ suite checks each
pipeline stage against an independent double-precision CPU implementation of
LOF, so a wrong kernel fails rather than merely a crashing one.

## What CI covers, and what it does not

GitHub-hosted runners have no NVIDIA GPU. CI proves the project builds across
CUDA 11.8/12.4 and Python 3.9–3.12, imports cleanly without a device, has no
version skew, and lints.

**Kernel correctness is not verified by CI.** Run both suites on real hardware
before opening a pull request, and state which GPU and CUDA version you used.

## Benchmarks and figures

```bash
python benchmarks/benchmark.py --max-n 200000 --json bench.json
python benchmarks/figures.py --results bench.json
```

Figures in `img/` are generated from that JSON, so the plots and the README
table are necessarily the same run. If you change a number in the README,
regenerate the figures from the same file. Earlier releases of this project
carried three mutually contradictory sets of benchmark numbers because the
README, one figure and another figure were each edited by hand.

Every benchmark row also reports agreement with scikit-learn. A speedup on a
wrong answer is not a speedup, so if `max rel err` moves, find out why before
reporting the timing.

## Layout

```
include/          public header (culof.h) and internal stage interfaces
src/cuda/         kernels and the pybind11 binding
src/culof/        the Python package
tests/cpp/        GoogleTest suite against a CPU reference
tests/python/     pytest suite against scikit-learn
benchmarks/       timing harness and the figure generator
```

The Python package lives under `src/` deliberately. With it at the repository
root, running `python -m pytest` from the root imports the source directory
instead of the installed package, the native extension is missing, and
collection fails with a confusing "circular import" error.

## Style

- Python: `ruff check .` and `ruff format .`, line length 100.
- C++/CUDA: 4-space indent, roughly 90 columns, `snake_case` functions,
  `kCamelCase` compile-time constants.
- Comments explain *why*. The what is already in the code.

## Before changing the kernels

- **Determinism is a feature, not an accident.** The gather uses a block-wide
  prefix sum rather than `atomicAdd` so neighbours are emitted in index order.
  Atomics would be marginally faster and would silently make results
  irreproducible, because float addition is not associative.
- **No `--use_fast_math`.** The library's contract is agreement with
  scikit-learn to within float32 rounding; fast-math's reciprocal and
  square-root approximations break it for a few percent of throughput.
- **Selection is memory-bound.** Count sweeps of the distance tile before
  optimising anything else. There are currently four.
- **The k-th smallest key is the k-distance.** The radix select produces it as a
  by-product, so nothing downstream needs a sorted neighbour list. Any change
  that reintroduces sorting is doing more work than the problem requires.
- **Centring is load-bearing.** It leaves distances unchanged but keeps the GEMM
  identity `‖a-b‖² = ‖a‖² + ‖b‖² - 2a·b` from cancelling.

## Reporting a bug

Include:

```bash
python -c "import culof; print(culof.__version__, culof.device_info())"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
nvcc --version
```

plus the array shape, `n_neighbors`, and what you expected. If the results look
wrong, compare against `sklearn.neighbors.LocalOutlierFactor` on the same input
and include both.
