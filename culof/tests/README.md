# Python Tests

This directory contains tests for the Python bindings to the CUDA-accelerated LOF implementation.

## Test Files

- `test_cuda_lof.py` - Basic functionality tests for the CUDA LOF implementation
- `test_sklearn_comparison.py` - Tests comparing results with scikit-learn's implementation
- `test_numerical_accuracy.py` - Tests for numerical accuracy and stability
- `test_edge_cases.py` - Tests for handling edge cases and error conditions

## Running Tests

To run all tests:

```bash
cd python
python -m pytest tests
```

To run a specific test:

```bash
cd python
python -m pytest tests/test_cuda_lof.py
```

## Test Coverage

The tests ensure that:

1. The CUDA LOF implementation produces correct results
2. Results are comparable to scikit-learn's implementation
3. The implementation handles edge cases gracefully
4. Numerical precision is maintained

## Adding New Tests

When adding new tests:

1. Follow the naming convention `test_*.py` for test files
2. Group related tests in the same file
3. Use descriptive test names
4. Include documentation for each test case 