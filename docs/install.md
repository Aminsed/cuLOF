# Installation

## Requirements

| | minimum |
|---|---|
| NVIDIA GPU | compute capability 7.0 (Volta) or newer |
| CUDA Toolkit | 11.0 |
| CMake | 3.18 |
| C++ compiler | C++17 |
| Python | 3.9 |
| pybind11 | 2.12 (build time; installed automatically) |

cuLOF is a CUDA extension. It is compiled against your local toolkit at install
time, so `nvcc` must be on `PATH` (or `CUDACXX` must point at it) before you
install.

## From source

```bash
git clone https://github.com/Aminsed/cuLOF.git
cd cuLOF
pip install .
```

Editable install for development:

```bash
pip install -e ".[test]"
```

## From PyPI

```bash
pip install culof
```

The PyPI artefact is a source distribution: pip will compile it, so the same
toolchain requirements apply.

## Choosing CUDA architectures

By default the build detects the architecture of the GPU in the machine and
compiles only for that, which is fastest to build and to run. Override it when
building on a machine that will not run the code — a container, a CI job, or a
build host with a different GPU:

```bash
# One target
CULOF_CUDA_ARCHITECTURES=86 pip install .

# A fat binary covering several
CULOF_CUDA_ARCHITECTURES="70;80;86;89" pip install .
```

The standard CMake `CUDAARCHS` environment variable is honoured too. If no GPU
is visible and nothing is specified, the build falls back to a fat binary
covering Volta through Hopper.

## Verifying the install

```bash
python -c "import culof; print(culof.__version__, culof.device_info())"
```

Expected output resembles:

```
0.2.0 NVIDIA RTX A6000 (sm_86, 84 SMs, 48550 MiB)
```

Then run the test suite, which needs a GPU:

```bash
pip install -e ".[test]"
pytest tests/python
```

## Building the C++ library and its tests

```bash
cmake -B build -DCULOF_BUILD_TESTS=ON
cmake --build build -j
./build/culof_tests
```

## Troubleshooting

**`nvcc` not found.** Install the CUDA toolkit and put it on `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

**`no kernel image is available for execution on the device`.** The binary was
built for a different architecture than the GPU running it. Rebuild with
`CULOF_CUDA_ARCHITECTURES` set to your GPU's compute capability, which
`nvidia-smi --query-gpu=compute_cap --format=csv` reports.

**`CUDA error ... out of memory`.** cuLOF sizes its scratch tile from free
device memory, but another process may take memory after that measurement.
Reduce the tile explicitly:

```bash
CULOF_TILE_ROWS=256 python your_script.py
```

**Results are all the same number.** That symptom comes from pybind11 older
than 2.12 built against NumPy 2.x. The build now refuses those versions; if you
see it, you are running a stale binary — rebuild from a clean tree.

**CMake picks the wrong Python.** Pass the interpreter explicitly:

```bash
cmake -B build -DPython_EXECUTABLE=$(which python)
```
