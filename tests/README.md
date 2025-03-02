# Tests

This directory contains the unit tests for the CUDA-accelerated LOF algorithm implementation.

## Files

### CUDA Implementation Tests

- `test_distances.cu` - Tests for the distance computation kernels
- `test_knn.cu` - Tests for the K-nearest neighbors search implementation
- `test_lof.cu` - Tests for the LOF score computation

### Python Integration Tests

- `test_consistency.py` - Tests for ensuring consistent results across different runs

### Data

The `data/` directory contains test datasets used for evaluating the correctness and performance of the implementation.

## Running Tests

### CUDA Tests

The CUDA tests are built as part of the CMake build process when the `BUILD_TESTS` option is enabled:

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make
ctest
```

### Python Tests

The Python tests can be run using pytest:

```bash
cd python
python -m pytest test_*.py
```

## Test Coverage

The tests cover:

1. **Correctness** - Ensuring the CUDA implementation produces correct results
2. **Numerical Stability** - Testing the handling of edge cases, NaN values, and numerical precision
3. **Performance** - Benchmarking the implementation against scikit-learn
4. **Memory Management** - Testing for memory leaks and proper resource cleanup
5. **API Compatibility** - Ensuring the Python API is compatible with scikit-learn's LOF implementation 