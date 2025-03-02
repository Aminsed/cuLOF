# Installation Instructions

This document provides detailed instructions for installing the CUDA-accelerated LOF implementation.

## Prerequisites

- CUDA Toolkit 11.0 or newer
- CMake 3.18 or newer
- C++ compiler with C++14 support (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.6 or newer (for Python bindings)
- NumPy, scikit-learn (for comparison and testing)

## Installation Methods

### PyPI Installation (Recommended)

The `culof` package is available on PyPI as a source distribution:

```bash
# Install dependencies first
pip install numpy scikit-learn matplotlib

# Install culof from PyPI
pip install culof
```

**Note**: Since this is a CUDA extension, you **must** have the CUDA toolkit installed on your system before installing the package. The package will be compiled during installation using your local CUDA setup.

### System-specific Prerequisites

### Linux

```bash
# Install CUDA Toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Install build tools
sudo apt install cmake build-essential

# Install Python dependencies
pip install numpy scikit-learn matplotlib
```

### macOS

Note: CUDA is not officially supported on macOS from CUDA 11.0 onwards. These instructions are for legacy systems.

```bash
# Install CUDA Toolkit
# Download from NVIDIA website and follow the instructions

# Install build tools using Homebrew
brew install cmake

# Install Python dependencies
pip install numpy scikit-learn matplotlib
```

### Windows

1. Download and install CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Download and install CMake from the [CMake website](https://cmake.org/download/)
3. Install Visual Studio with C++ development tools
4. Install Python dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib
   ```

## Building from Source

### Option 1: Using pip

The simplest way to install from source is using pip:

```bash
pip install git+https://github.com/Aminsed/cuLOF.git
```

### Option 2: Manual build

1. Clone the repository:
   ```bash
   git clone https://github.com/Aminsed/cuLOF.git
   cd cuLOF
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Or install directly:
   ```bash
   python setup.py install
   ```

## Verifying Installation

### Python Package

Verify the Python package works:

```python
import culof
import numpy as np

# Check package version
print(f"Using cuLOF version: {culof.__version__}")

# Generate sample data
X = np.random.randn(100, 2).astype(np.float32)

# Create LOF detector
lof = culof.LOF()
lof.set_k(10)

# Compute outlier scores
scores = lof.fit_predict(X)
outliers = lof.get_outliers(scores)

print(f"Detected {len(outliers)} outliers out of {len(X)} points")
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure that CUDA is properly installed and that the `CUDA_HOME` environment variable is set:
   ```bash
   export CUDA_HOME=/usr/local/cuda  # Adjust path as needed
   export PATH=$PATH:$CUDA_HOME/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
   ```

2. **Build fails with CUDA errors**: Check compatibility between your CUDA Toolkit version and the compiler:
   ```bash
   nvcc --version
   g++ --version  # or your compiler
   ```

3. **Python ImportError**: Check that the installation completed successfully:
   ```bash
   # Try reinstalling with verbose output
   pip install -v culof
   ```

4. **Runtime CUDA errors**:
   - Check that your GPU is supported by the CUDA version you're using
   - Update your NVIDIA drivers to the latest version
   - Run `nvidia-smi` to verify that your GPU is detected and functioning

5. **AttributeError during installation**: This typically indicates a missing or incorrectly configured CUDA environment:
   ```
   AttributeError: 'NoneType' object has no attribute 'name'
   ```
   Make sure you have CUDA toolkit installed and properly configured.

For further assistance, please open an issue on the [GitHub repository](https://github.com/Aminsed/cuLOF/issues). 