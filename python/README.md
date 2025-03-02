# Python Interface

This directory contains the Python bindings and example code for the CUDA-accelerated LOF algorithm.

## Files

### lof_binding.cpp

This file contains the Python bindings implemented using pybind11. It exposes the CUDA-accelerated LOF implementation to Python, providing a scikit-learn-compatible interface.

### __init__.py

This file initializes the Python module and defines the `LOF` class, which provides a scikit-learn compatible interface.

### example.py

This file contains example usage of the CUDA-accelerated LOF algorithm and compares its performance with scikit-learn's implementation. It includes:
- Performance benchmarks across different dataset sizes
- Visualization of outlier detection results
- Examples of handling different data types and configurations

### Test Files

- `test_cuda_lof.py` - Basic tests for the CUDA LOF implementation
- `test_sklearn_comparison.py` - Tests comparing results with scikit-learn's implementation
- `test_numerical_accuracy.py` - Tests for numerical accuracy and stability
- `test_edge_cases.py` - Tests for handling edge cases and error conditions

## Usage

Basic usage:

```python
import numpy as np
import cuda_lof

# Create synthetic dataset with outliers
X = np.random.randn(1000, 2).astype(np.float32)
outliers = np.random.uniform(low=-5, high=5, size=(10, 2)).astype(np.float32)
X = np.vstack([X, outliers])

# Create LOF detector
lof = cuda_lof.LOF(k=20)

# Fit and predict (returns 1 for inliers, -1 for outliers)
labels = lof.fit_predict(X)

# Get raw LOF scores (higher values indicate more likely outliers)
scores = lof.score_samples(X)
```

For more detailed examples, see `example.py`. 