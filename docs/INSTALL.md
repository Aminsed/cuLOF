# Installation Instructions

This document provides detailed instructions for installing the CUDA-accelerated LOF implementation (cuLOF).

## Prerequisites

- CUDA Toolkit 11.0 or newer
- CMake 3.18 or newer
- C++ compiler with C++14 support (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.6 or newer (for Python bindings)
- NumPy, scikit-learn (for comparison and testing)

## System-specific Prerequisites

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

The simplest way to install is using pip, which will automatically build the package:

```bash
pip install git+https://github.com/Aminsed/cuLOF.git
```

### Option 2: Manual build

1. Clone the repository:
   ```bash
   git clone git@github.com:Aminsed/cuLOF.git
   cd cuLOF
   ```

2. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

3. Configure with CMake:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   
   Additional configuration options:
   - `-DBUILD_TESTS=ON` - Build test suite
   - `-DPYTHON_BINDINGS=OFF` - Disable Python bindings
   - `-DCUDA_ARCH=70` - Specify CUDA architecture (default is based on detected GPU)

4. Build the project:
   ```bash
   make -j4  # Use 4 cores for compilation
   ```

5. Install (optional):
   ```bash
   make install
   ```

6. For Python bindings, install the package:
   ```bash
   cd ..  # Return to project root
   pip install -e .
   ```

## Verifying Installation

### C++ Library

If you've built the test suite, you can run the tests:

```bash
cd build
ctest
```

### Python Package

Verify the Python package works:

```python
import cuda_lof
import numpy as np

# Generate sample data
X = np.random.randn(100, 2).astype(np.float32)

# Create LOF detector
lof = cuda_lof.LOF(k=10)

# Compute outlier scores
scores = lof.fit_predict(X)

print(f"Detected {(scores == -1).sum()} outliers out of {len(X)} points")
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

3. **Python ImportError**: Ensure that the compiled module is in your Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/cuLOF/build/python
   ```

4. **Runtime CUDA errors**:
   - Check that your GPU is supported by the CUDA version you're using
   - Update your NVIDIA drivers to the latest version
   - Run `nvidia-smi` to verify that your GPU is detected and functioning

For further assistance, please open an issue on the [GitHub repository](https://github.com/Aminsed/cuLOF). 