#!/usr/bin/env python3
"""
Simple test script to debug CUDA LOF import issues.
"""
import os
import sys

# Add the build directory to the path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "python")
sys.path.insert(0, build_dir)

try:
    print(f"Python path: {sys.path}")
    print(f"Looking for module in: {build_dir}")
    
    # Try to import the module
    import _cuda_lof
    print("Module imported successfully!")
    print(f"LOF class: {_cuda_lof.LOF}")
    
    # Create a simple dataset
    import numpy as np
    X = np.array([
        [1.0, 1.0],
        [1.2, 0.8],
        [0.9, 1.1],
        [0.8, 0.8],
        [5.0, 5.0]  # Outlier
    ], dtype=np.float32)
    
    # Create LOF detector
    lof = _cuda_lof.LOF(k=3)
    
    # Compute LOF scores
    scores = lof.fit_predict(X)
    print(f"LOF scores: {scores}")
    
    # Get outliers using the module function
    outliers = _cuda_lof.get_outliers(scores, 1.5)
    print(f"Outliers: {outliers}")
    
except ImportError as e:
    print(f"Import error: {e}")
    
    # Check if CUDA is available
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available to check CUDA")
    
    # List the shared library
    print("\nShared library details:")
    os.system(f"ls -la {build_dir}")
    os.system(f"ldd {os.path.join(build_dir, '_cuda_lof.cpython-311-x86_64-linux-gnu.so')}") 