#!/usr/bin/env python
import numpy as np
import sys
import os
import time
from sklearn.neighbors import LocalOutlierFactor

# Set up paths
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))

# Try to import our CUDA LOF implementation
try:
    import _cuda_lof
    print("CUDA LOF module imported successfully!")
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {str(e)}")
    print("CUDA LOF implementation not found.")
    CUDA_AVAILABLE = False

# Generate a simple dataset
np.random.seed(42)
n_samples = 200
n_features = 2
X = np.random.rand(n_samples, n_features).astype(np.float32)
print(f"Generated dataset with shape: {X.shape}")

# Add some outliers
X[0:5] = X[0:5] * 5
print("Added outliers to the dataset")

# Compare scikit-learn and CUDA LOF
k = 10
print(f"Using k={k} neighbors")

# Scikit-learn LOF
print("\nRunning scikit-learn LOF...")
start_time = time.time()
sklearn_lof = LocalOutlierFactor(n_neighbors=k, algorithm='brute', metric='euclidean')
sklearn_lof.fit(X)
sklearn_scores = -sklearn_lof.negative_outlier_factor_  # sklearn returns negated scores
sklearn_time = time.time() - start_time
print(f"scikit-learn LOF completed in {sklearn_time:.6f} seconds")
print(f"scikit-learn score range: min={sklearn_scores.min():.4f}, max={sklearn_scores.max():.4f}, mean={sklearn_scores.mean():.4f}")

# CUDA LOF
if CUDA_AVAILABLE:
    print("\nRunning CUDA LOF...")
    start_time = time.time()
    lof = _cuda_lof.LOF(k=k)
    # Use fit_predict which returns the LOF scores
    cuda_scores = np.array(lof.fit_predict(X))
    cuda_time = time.time() - start_time
    print(f"CUDA LOF completed in {cuda_time:.6f} seconds")
    print(f"CUDA score range: min={cuda_scores.min():.4f}, max={cuda_scores.max():.4f}, mean={cuda_scores.mean():.4f}")
    
    # Compare results
    print("\nComparing results:")
    if sklearn_scores.shape == cuda_scores.shape:
        print(f"Shapes match: {sklearn_scores.shape}")
        
        # Calculate differences
        abs_diff = np.abs(sklearn_scores - cuda_scores)
        rel_diff = abs_diff / (np.abs(sklearn_scores) + 1e-10)
        
        print(f"Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
        print(f"Max relative difference: {np.max(rel_diff):.6e}")
        print(f"Mean relative difference: {np.mean(rel_diff):.6e}")
        
        # Check if values are close
        rtol = 1e-5
        atol = 1e-8
        values_close = np.allclose(sklearn_scores, cuda_scores, rtol=rtol, atol=atol)
        print(f"Values close within tolerance (rtol={rtol}, atol={atol}): {values_close}")
        
        # Outlier agreement
        threshold = np.percentile(sklearn_scores, 95)  # Using 95th percentile as threshold
        sklearn_outliers = sklearn_scores > threshold
        cuda_outliers = cuda_scores > threshold
        outlier_agreement = np.mean(sklearn_outliers == cuda_outliers)
        print(f"Outlier agreement: {outlier_agreement:.4f}")
        
        # Speedup
        speedup = sklearn_time / cuda_time if cuda_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Shapes don't match: sklearn={sklearn_scores.shape}, cuda={cuda_scores.shape}")
    
    # Try to get outliers
    print("\nGetting outliers from CUDA LOF...")
    try:
        outliers = lof.get_outliers(cuda_scores)
        print(f"Found {len(outliers)} outliers with threshold=1.5")
    except Exception as e:
        print(f"Error getting outliers: {str(e)}")
else:
    print("\nSkipping CUDA LOF comparison as the module is not available.")

print("\nTest completed!") 