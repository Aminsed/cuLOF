#!/usr/bin/env python
import numpy as np
import sys
import os
import time

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))

try:
    print("Attempting to import _cuda_lof...")
    import _cuda_lof
    print("Module imported successfully!")
    
    # Try to create a LOF instance
    lof = _cuda_lof.LOF(n_neighbors=5)
    print("LOF instance created with k=5")
    
    # Generate a small dataset
    X = np.random.rand(100, 2).astype(np.float32)
    print(f"Generated dataset with shape: {X.shape}")
    
    # Fit and compute scores
    print("Fitting LOF model...")
    start_time = time.time()
    lof.fit(X)
    fit_time = time.time() - start_time
    print(f"Fit completed in {fit_time:.6f} seconds")
    
    # Get scores
    print("Computing LOF scores...")
    scores = np.array(lof.get_scores())
    print(f"Score shape: {scores.shape}")
    print(f"Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    # Try to get outliers
    print("Getting outliers...")
    try:
        outliers = lof.get_outliers(threshold=1.5)
        print(f"Found {len(outliers)} outliers")
    except Exception as e:
        print(f"Error getting outliers: {str(e)}")
    
    print("Test completed successfully!")
    
except ImportError as e:
    print(f"ImportError: {str(e)}")
    print("Module paths in sys.path:")
    for path in sys.path:
        print(f"  {path}")
except Exception as e:
    print(f"Unexpected error: {str(e)}") 