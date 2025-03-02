#!/usr/bin/env python3
"""
Edge case tests for CUDA LOF implementation.
This script tests the CUDA LOF implementation against scikit-learn's implementation
on various edge cases and potentially problematic datasets.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from time import time

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cuda_lof
except ImportError:
    try:
        import _cuda_lof as cuda_lof
    except ImportError:
        print("Error: Could not import CUDA LOF module. Make sure it's built and installed.")
        print("Try running: python setup.py install")
        sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)

def compare_lof_scores(X, n_neighbors=20, verbose=True):
    """
    Compare LOF scores between sklearn and CUDA implementation.
    
    Returns:
        tuple: (sklearn_scores, cuda_scores, max_diff, matches_within_tolerance)
    """
    # Compute sklearn LOF scores
    sklearn_lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm='brute')
    sklearn_lof.fit(X)
    sklearn_scores = -sklearn_lof.negative_outlier_factor_  # Convert to positive scores
    
    # Compute CUDA LOF scores
    cuda_lof_instance = cuda_lof.LOF(k=n_neighbors)
    cuda_scores = cuda_lof_instance.fit_predict(X)
    
    # Calculate differences
    diffs = sklearn_scores - cuda_scores
    max_diff = np.max(np.abs(diffs))
    rtol, atol = 1e-3, 1e-3  # Relative and absolute tolerance
    matches_within_tolerance = np.allclose(sklearn_scores, cuda_scores, rtol=rtol, atol=atol)
    
    if verbose:
        print(f"Dataset shape: {X.shape}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Matches within tolerance (rtol={rtol}, atol={atol}): {matches_within_tolerance}")
    
    return sklearn_scores, cuda_scores, max_diff, matches_within_tolerance

def test_outlier_clusters(size=100, visualize=True):
    """Test with a dataset containing outlier clusters."""
    print("\n=== Testing outlier clusters ===")
    
    # Create main cluster
    X_main = np.random.normal(0, 1, size=(size, 2))
    
    # Create small distant clusters (outliers)
    X_outlier1 = np.random.normal(10, 0.5, size=(10, 2))
    X_outlier2 = np.random.normal(-10, 0.5, size=(10, 2))
    
    # Combine all points
    X = np.vstack([X_main, X_outlier1, X_outlier2]).astype(np.float32)
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=20)
    
    if visualize and X.shape[1] == 2:
        plt.figure(figsize=(15, 5))
        
        # Plot data
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
        plt.title('Dataset with Outlier Clusters')
        plt.grid(True, alpha=0.3)
        
        # Plot sklearn scores
        plt.subplot(1, 3, 2)
        sc = plt.scatter(X[:, 0], X[:, 1], c=sklearn_scores, cmap='plasma', alpha=0.7)
        plt.colorbar(sc, label='LOF Score')
        plt.title('scikit-learn LOF Scores')
        plt.grid(True, alpha=0.3)
        
        # Plot CUDA scores
        plt.subplot(1, 3, 3)
        sc = plt.scatter(X[:, 0], X[:, 1], c=cuda_scores, cmap='plasma', alpha=0.7)
        plt.colorbar(sc, label='LOF Score')
        plt.title('CUDA LOF Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("lof_outlier_clusters.png")
    
    return matches

def test_high_dimensional_data(n_dims=50, size=200):
    """Test with high-dimensional data."""
    print("\n=== Testing high-dimensional data ===")
    
    # Generate high-dimensional data
    X = np.random.normal(0, 1, size=(size, n_dims)).astype(np.float32)
    
    # Add some outliers with larger magnitude
    X[-20:, :] = np.random.normal(0, 5, size=(20, n_dims)).astype(np.float32)
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=20)
    
    return matches

def test_identical_points(size=100):
    """Test with identical points in the dataset."""
    print("\n=== Testing identical points ===")
    
    # Create dataset with some identical points
    X = np.random.normal(0, 1, size=(size, 3)).astype(np.float32)
    
    # Make some points identical
    identical_indices = np.random.choice(size, 10, replace=False)
    for i in range(1, len(identical_indices)):
        X[identical_indices[i]] = X[identical_indices[0]]
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=20)
    
    return matches

def test_varying_density(size=200, visualize=True):
    """Test with a dataset containing varying densities."""
    print("\n=== Testing varying density ===")
    
    # Create dataset with varying densities
    X1 = np.random.normal(0, 0.5, size=(size//2, 2))  # Dense cluster
    X2 = np.random.normal(5, 2, size=(size//2, 2))    # Sparse cluster
    X = np.vstack([X1, X2]).astype(np.float32)
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=20)
    
    if visualize and X.shape[1] == 2:
        plt.figure(figsize=(15, 5))
        
        # Plot data
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
        plt.title('Dataset with Varying Density')
        plt.grid(True, alpha=0.3)
        
        # Plot sklearn scores
        plt.subplot(1, 3, 2)
        sc = plt.scatter(X[:, 0], X[:, 1], c=sklearn_scores, cmap='plasma', alpha=0.7)
        plt.colorbar(sc, label='LOF Score')
        plt.title('scikit-learn LOF Scores')
        plt.grid(True, alpha=0.3)
        
        # Plot CUDA scores
        plt.subplot(1, 3, 3)
        sc = plt.scatter(X[:, 0], X[:, 1], c=cuda_scores, cmap='plasma', alpha=0.7)
        plt.colorbar(sc, label='LOF Score')
        plt.title('CUDA LOF Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("lof_varying_density.png")
    
    return matches

def test_small_k_large_data(size=1000):
    """Test with a large dataset but small k value."""
    print("\n=== Testing small k with large dataset ===")
    
    # Generate large dataset
    X = np.random.normal(0, 1, size=(size, 5)).astype(np.float32)
    
    # Use small k value
    k = 5
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=k)
    
    return matches

def test_large_k_small_data(size=50):
    """Test with a small dataset but large k value."""
    print("\n=== Testing large k with small dataset ===")
    
    # Generate small dataset
    X = np.random.normal(0, 1, size=(size, 3)).astype(np.float32)
    
    # Use large k value relative to dataset size
    k = size // 2
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=k)
    
    return matches

def test_extreme_values():
    """Test with extreme values in the dataset."""
    print("\n=== Testing extreme values ===")
    
    # Create dataset with some extreme values
    X = np.random.normal(0, 1, size=(100, 3)).astype(np.float32)
    
    # Add some extreme values
    X[-10:, :] = np.random.normal(0, 1, size=(10, 3)) * 1e6
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, matches = compare_lof_scores(X, n_neighbors=20)
    
    return matches

def run_all_edge_case_tests():
    """Run all edge case tests and report results."""
    tests = [
        ("Outlier Clusters", test_outlier_clusters),
        ("High-Dimensional Data", test_high_dimensional_data),
        ("Identical Points", test_identical_points),
        ("Varying Density", test_varying_density),
        ("Small k with Large Data", test_small_k_large_data),
        ("Large k with Small Data", test_large_k_small_data),
        ("Extreme Values", test_extreme_values)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        start_time = time()
        passed = test_func()
        end_time = time()
        
        results.append({
            'name': name,
            'passed': passed,
            'time': end_time - start_time
        })
    
    # Print summary
    print("\n=== Edge Case Test Results ===")
    print("{:<25} {:<10} {:<10}".format("Test", "Passed", "Time (s)"))
    print("-" * 45)
    
    all_passed = True
    for result in results:
        print("{:<25} {:<10} {:<10.4f}".format(
            result['name'], "Yes" if result['passed'] else "No", result['time']))
        if not result['passed']:
            all_passed = False
    
    print("\nOverall result:", "PASSED" if all_passed else "FAILED")
    
    return all_passed, results

if __name__ == "__main__":
    all_passed, results = run_all_edge_case_tests()
    plt.show()  # Show all figures
    
    # Exit with appropriate status code
    sys.exit(0 if all_passed else 1) 