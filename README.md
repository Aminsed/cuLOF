# cuLOF: CUDA-Accelerated Local Outlier Factor

A CUDA-accelerated implementation of the Local Outlier Factor (LOF) algorithm for anomaly detection. This implementation is designed to be compatible with scikit-learn's LOF interface while providing significant speedups for larger datasets.

## ⚠️ Important Installation Note

This package requires CUDA and will **not install correctly without a properly configured CUDA environment**. Due to CUDA dependencies, installation from PyPI requires compilation against your local CUDA installation.

## Installation

### Prerequisites

- CUDA Toolkit 11.0+
- Python 3.6+
- NumPy
- scikit-learn (for comparison)
- C++14 compliant compiler
- CMake 3.18+

### Installing from PyPI

The source distribution is available on PyPI:

```bash
# Install dependencies first
pip install numpy scikit-learn matplotlib

# Then install culof (requires CUDA toolkit)
pip install culof
```

**⚠️ Note**: Since this is a CUDA extension, you **must** have the CUDA toolkit installed on your system before installing the package. The PyPI package is a source distribution that will be compiled during installation.

If you encounter compilation errors, follow the "Installing from Source" instructions below.

### Installing from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Aminsed/cuLOF.git
cd cuLOF

# Option 1: Development installation
pip install -e .

# Option 2: Build from source
python setup.py install
```

### Conda Installation (Alternative)

If you're having trouble with the PyPI installation, consider using conda:

```bash
# Install dependencies
conda install -c conda-forge numpy scikit-learn cmake cudatoolkit>=11.0

# Install culof from source
pip install git+https://github.com/Aminsed/cuLOF.git
```

## Troubleshooting Installation

If you encounter issues during installation:

1. **CUDA Toolkit**: Ensure CUDA toolkit is properly installed and in your PATH
   ```bash
   nvcc --version
   ```

2. **CUDA Version**: Check that your CUDA version is 11.0 or higher

3. **C++ Compiler**: Verify you have a modern C++ compiler (gcc 7+, MSVC 19.14+, clang 5+)
   ```bash
   g++ --version
   ```

4. **CMake**: Make sure CMake 3.18+ is installed
   ```bash
   cmake --version
   ```

5. **Build Output**: If installation fails, check the build output for specific errors
   ```bash
   # Install with verbose output
   pip install -v culof
   ```

6. For specific errors, check the project issues or create a new one at our [GitHub repository](https://github.com/Aminsed/cuLOF/issues)

## Usage

Basic Python usage:

```python
import numpy as np
from sklearn.datasets import make_blobs
from culof import LOF

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
X = np.vstack([X, outliers]).astype(np.float32)  # Cast to float32 for optimal performance

# Create and configure LOF detector
lof = LOF()
lof.set_k(20)                # Set the number of neighbors (default: 20)
lof.set_normalize(True)      # Optional: normalize the data
lof.set_threshold(1.5)       # Optional: set custom threshold (default: 1.5)

# Compute LOF scores
scores = lof.fit_predict(X)  

# Get outliers
outliers = lof.get_outliers(scores)
print(f"Detected {len(outliers)} outliers out of {len(X)} samples")
```

### Alternative Usage Pattern

```python
# Initialize with configuration
lof = LOF()
lof.set_k(20)

# Compute LOF scores and get results directly 
scores = lof.fit_predict(X)

# Print detection results
print(f"Detected {len(lof.get_outliers(scores))} outliers out of {len(X)} samples")
print(f"Top 5 highest LOF scores: {sorted(scores, reverse=True)[:5]}")
```

## Documentation

For more detailed information, check out these resources:

- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Usage Guide](docs/usage_guide.md) - Comprehensive guide with examples
- [Example Script](docs/example.py) - Ready-to-run example script
- [Technical Optimizations](docs/technical_optimizations.md) - Implementation details and optimizations
- [Benchmark Results](docs/benchmark_results.md) - Performance comparisons

## Performance

This CUDA implementation achieves significant speedups compared to scikit-learn's implementation, especially for larger datasets:

| Dataset Size | scikit-learn (s) | CUDA LOF (s) | Speedup |
|--------------|------------------|--------------|---------|
| 1,050        | 0.007614         | 0.049126     | 0.15x   |
| 2,065        | 0.022725         | 0.008887     | 2.56x   |
| 4,065        | 0.075397         | 0.020063     | 3.76x   |
| 8,002        | 0.261650         | 0.048288     | 5.42x   |
| 15,750       | 0.931842         | 0.137987     | 6.75x   |

Note: Performance may vary depending on your GPU and system configuration. The CUDA implementation has some overhead for small datasets but provides significant speedups for larger datasets.

## API Reference

### LOF Class

```python
class LOF:
    """Local Outlier Factor implementation accelerated with CUDA."""
```

#### Methods

- `set_k(k: int)`: Set the number of neighbors to use for LOF computation.
- `set_normalize(normalize: bool)`: Set whether to normalize the input data before computation.
- `set_threshold(threshold: float)`: Set the threshold for outlier detection.
- `fit_predict(X: np.ndarray)`: Compute LOF scores for the input data.
- `get_outliers(scores: np.ndarray)`: Get the indices of outliers based on LOF scores.

## Important Notes

- For optimal performance, input data should be of type `np.float32`.
- The `culof` package API differs slightly from scikit-learn's LOF implementation.
- The package is optimized for CUDA and will not work without a CUDA-capable GPU.

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++14 compliant compiler
- Python 3.6+ with NumPy and scikit-learn (for comparison and testing)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The LOF algorithm was originally proposed by Breunig et al. in "LOF: Identifying Density-Based Local Outliers" (2000).
- This implementation builds upon ideas from the scikit-learn implementation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to our [GitHub repository](https://github.com/Aminsed/cuLOF). 