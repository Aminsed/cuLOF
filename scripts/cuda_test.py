#!/usr/bin/env python3
"""
Simple test to check if CUDA is available and working.
"""
import os
import sys

try:
    import numpy as np
    print("NumPy version:", np.__version__)
except ImportError:
    print("NumPy not found")

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch not found")

try:
    import cupy as cp
    print("CuPy version:", cp.__version__)
    print("CUDA available in CuPy:", cp.cuda.is_available())
    if cp.cuda.is_available():
        print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
        print("CUDA device name:", cp.cuda.runtime.getDeviceProperties(0)['name'])
except ImportError:
    print("CuPy not found")

# Check CUDA installation
print("\nCUDA installation:")
os.system("nvcc --version")
os.system("nvidia-smi")

# Check environment variables
print("\nCUDA environment variables:")
for var in ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
    print(f"{var}={os.environ.get(var, 'Not set')}") 