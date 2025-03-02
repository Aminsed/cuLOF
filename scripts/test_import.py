#!/usr/bin/env python3
"""
Simple test script to verify we can import the CUDA module.
"""

import os
import sys

# Print Python version and path
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Python path:", sys.path)

# Try to import numpy directly
try:
    import numpy
    print("Successfully imported numpy!")
    print("Numpy version:", numpy.__version__)
except ImportError as e:
    print("Error importing numpy:", e)

# Add the current directory to the path
sys.path.append(os.path.abspath('.'))

try:
    import culof
    print("Successfully imported culof package!")
    print("Version:", culof.__version__)
    
    # Try to create an LOF instance
    lof = culof.LOF(k=20)
    print("Successfully created LOF instance!")
    
    # Try to import the CUDA module directly
    try:
        from culof import _cuda_lof
        print("Successfully imported _cuda_lof module!")
    except ImportError as e:
        print("Error importing _cuda_lof:", e)
        
except ImportError as e:
    print("Error importing culof:", e) 