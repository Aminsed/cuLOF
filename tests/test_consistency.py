#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import sys
import os
import time
import traceback

# Print debugging information
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))
print("sys.path:", sys.path)

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print("Added parent directory to sys.path:", parent_dir)
print("Updated sys.path:", sys.path)

# Add build/python directory to path
build_python_dir = os.path.join(parent_dir, 'build', 'python')
if os.path.exists(build_python_dir):
    sys.path.append(build_python_dir)
    print("Added build/python directory to sys.path:", build_python_dir)
    print("Updated sys.path:", sys.path)

# Try to import our CUDA LOF implementation
try:
    print("Attempting to import _cuda_lof...")
    import _cuda_lof
    print("Module imported successfully!")
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {str(e)}")
    print("CUDA LOF implementation not found. Make sure to build it first.")
    CUDA_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error importing _cuda_lof: {str(e)}")
    traceback.print_exc()
    CUDA_AVAILABLE = False


def generate_dataset(n_samples=1000, n_features=2, contamination=0.05, random_state=42):
    """Generate a synthetic dataset with outliers."""
    rng = np.random.RandomState(random_state)
    
    # Generate inliers
    n_inliers = int(n_samples * (1 - contamination))
    X_inliers = 0.3 * rng.randn(n_inliers, n_features)
    
    # Generate outliers
    n_outliers = n_samples - n_inliers
    X_outliers = rng.uniform(low=-4, high=4, size=(n_outliers, n_features))
    
    # Combine inliers and outliers
    X = np.vstack([X_inliers, X_outliers])
    
    # Shuffle the data
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X = X[indices]
    
    # True outlier labels (for reference)
    y_true = np.zeros(n_samples, dtype=int)
    y_true[n_inliers:] = 1
    y_true = y_true[indices]
    
    return X, y_true


def test_lof_consistency(X, k=20, rtol=1e-5, atol=1e-8, print_stats=True):
    """
    Test consistency between scikit-learn LOF and our CUDA LOF implementation.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input data
    k : int, default=20
        Number of neighbors to use
    rtol : float, default=1e-5
        Relative tolerance for numpy.allclose
    atol : float, default=1e-8
        Absolute tolerance for numpy.allclose
    print_stats : bool, default=True
        Whether to print detailed statistics about the comparison
        
    Returns:
    --------
    is_consistent : bool
        True if the implementations are consistent within tolerance
    stats : dict
        Dictionary with statistics about the comparison
    """
    if not CUDA_AVAILABLE:
        print("CUDA LOF implementation not available. Skipping consistency test.")
        return False, {}
    
    # 1. Compute LOF scores using scikit-learn
    start_time = time.time()
    sklearn_lof = LocalOutlierFactor(n_neighbors=k, algorithm='brute', metric='euclidean')
    sklearn_lof.fit(X)
    sklearn_scores = -sklearn_lof.negative_outlier_factor_  # sklearn returns negated scores
    sklearn_time = time.time() - start_time
    
    # 2. Compute LOF scores using our CUDA implementation
    X_float32 = X.astype(np.float32)  # Convert to float32 for CUDA
    start_time = time.time()
    lof = _cuda_lof.LOF(n_neighbors=k)
    lof.fit(X_float32)
    cuda_scores = np.array(lof.get_scores())
    cuda_time = time.time() - start_time
    
    # 3. Compare the results
    # Check if shapes match
    shape_match = sklearn_scores.shape == cuda_scores.shape
    
    # Calculate differences
    abs_diff = np.abs(sklearn_scores - cuda_scores)
    rel_diff = abs_diff / (np.abs(sklearn_scores) + 1e-10)
    
    # Check if values are close within tolerance
    values_close = np.allclose(sklearn_scores, cuda_scores, rtol=rtol, atol=atol)
    
    # Outlier agreement - check if both methods identify the same points as outliers
    # using the contamination threshold from the scores distribution
    if shape_match:
        threshold = np.percentile(sklearn_scores, 95)  # Using 95th percentile as threshold
        sklearn_outliers = sklearn_scores > threshold
        cuda_outliers = cuda_scores > threshold
        outlier_agreement = np.mean(sklearn_outliers == cuda_outliers)
    else:
        outlier_agreement = 0.0
    
    # Results and statistics
    is_consistent = shape_match and values_close
    
    # Prepare statistics dictionary
    stats = {
        "shape_match": shape_match,
        "values_close": values_close,
        "outlier_agreement": outlier_agreement,
        "max_abs_diff": np.max(abs_diff) if shape_match else float('inf'),
        "mean_abs_diff": np.mean(abs_diff) if shape_match else float('inf'),
        "max_rel_diff": np.max(rel_diff) if shape_match else float('inf'),
        "mean_rel_diff": np.mean(rel_diff) if shape_match else float('inf'),
        "sklearn_time": sklearn_time,
        "cuda_time": cuda_time,
        "speedup": sklearn_time / cuda_time if cuda_time > 0 else 0
    }
    
    # Print statistics if requested
    if print_stats:
        print("\n=== LOF Implementation Consistency Test ===")
        print(f"Dataset shape: {X.shape}")
        print(f"Number of neighbors: {k}")
        print(f"Shape match: {shape_match}")
        print(f"Values close within tolerance: {values_close}")
        print(f"Outlier agreement: {outlier_agreement:.4f}")
        
        if shape_match:
            print(f"Max absolute difference: {stats['max_abs_diff']:.6e}")
            print(f"Mean absolute difference: {stats['mean_abs_diff']:.6e}")
            print(f"Max relative difference: {stats['max_rel_diff']:.6e}")
            print(f"Mean relative difference: {stats['mean_rel_diff']:.6e}")
        
        print(f"scikit-learn time: {sklearn_time:.6f} seconds")
        print(f"CUDA time: {cuda_time:.6f} seconds")
        print(f"Speedup: {stats['speedup']:.2f}x")
        
        if is_consistent:
            print("Result: Implementations are consistent within tolerance!")
        else:
            print("Result: Implementations are NOT consistent!")
    
    return is_consistent, stats


def visualize_comparison(X, k=20):
    """
    Visualize and compare the LOF scores from scikit-learn and CUDA implementations.
    Only works for 2D data.
    """
    if X.shape[1] != 2:
        print("Visualization only works for 2D data")
        return
    
    if not CUDA_AVAILABLE:
        print("CUDA LOF implementation not available. Skipping visualization.")
        return
    
    # Compute LOF scores
    sklearn_lof = LocalOutlierFactor(n_neighbors=k, algorithm='brute', metric='euclidean')
    sklearn_lof.fit(X)
    sklearn_scores = -sklearn_lof.negative_outlier_factor_
    
    lof = _cuda_lof.LOF(n_neighbors=k)
    lof.fit(X.astype(np.float32))
    cuda_scores = np.array(lof.get_scores())
    
    # Create threshold for outlier detection (95th percentile)
    threshold = np.percentile(sklearn_scores, 95)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot scikit-learn LOF scores
    sc1 = axes[0].scatter(X[:, 0], X[:, 1], c=sklearn_scores, cmap='viridis', 
                         s=30, edgecolors='k', linewidths=0.5)
    axes[0].set_title('scikit-learn LOF Scores')
    plt.colorbar(sc1, ax=axes[0])
    
    # Plot CUDA LOF scores
    sc2 = axes[1].scatter(X[:, 0], X[:, 1], c=cuda_scores, cmap='viridis', 
                         s=30, edgecolors='k', linewidths=0.5)
    axes[1].set_title('CUDA LOF Scores')
    plt.colorbar(sc2, ax=axes[1])
    
    # Plot differences
    diff = cuda_scores - sklearn_scores
    sc3 = axes[2].scatter(X[:, 0], X[:, 1], c=diff, cmap='coolwarm', 
                         s=30, edgecolors='k', linewidths=0.5)
    axes[2].set_title('Score Differences (CUDA - scikit-learn)')
    plt.colorbar(sc3, ax=axes[2])
    
    # Set limits and labels
    for ax in axes:
        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('lof_comparison.png')
    print("Visualization saved to 'lof_comparison.png'")


def test_edge_cases():
    """Test edge cases to ensure robust implementation."""
    print("\n=== Testing Edge Cases ===")
    
    if not CUDA_AVAILABLE:
        print("CUDA LOF implementation not available. Skipping edge case tests.")
        return
    
    # Case 1: Very small dataset
    print("\nCase 1: Very small dataset (10 points)")
    X_small = np.random.rand(10, 2).astype(np.float32)
    test_lof_consistency(X_small, k=5)
    
    # Case 2: Single feature
    print("\nCase 2: Single feature")
    X_1d = np.random.rand(100, 1).astype(np.float32)
    test_lof_consistency(X_1d, k=10)
    
    # Case 3: Many features
    print("\nCase 3: High-dimensional data")
    X_highdim = np.random.rand(100, 50).astype(np.float32)
    test_lof_consistency(X_highdim, k=10)
    
    # Case 4: All identical points
    print("\nCase 4: All identical points")
    X_identical = np.ones((50, 2), dtype=np.float32)
    test_lof_consistency(X_identical, k=10)
    
    # Case 5: Very different scales for features
    print("\nCase 5: Features with different scales")
    X_scales = np.random.rand(100, 2).astype(np.float32)
    X_scales[:, 1] *= 1000  # Second feature is 1000x larger
    test_lof_consistency(X_scales, k=10)
    
    # Case 6: Different k values
    for k in [5, 15, 30]:
        print(f"\nCase 6: Using k={k}")
        X = np.random.rand(200, 2).astype(np.float32)
        test_lof_consistency(X, k=k)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Running LOF consistency tests...")
    
    # Generate synthetic dataset
    X, y_true = generate_dataset(n_samples=1000, n_features=2, contamination=0.05)
    
    # Test consistency with default parameters
    is_consistent, stats = test_lof_consistency(X)
    
    # If data is 2D, create visualization
    if X.shape[1] == 2 and CUDA_AVAILABLE:
        visualize_comparison(X)
    
    # Test edge cases
    test_edge_cases()
    
    # Performance test with larger datasets
    if CUDA_AVAILABLE:
        print("\n=== Performance Tests ===")
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            print(f"\nTesting with {size} samples:")
            X_perf, _ = generate_dataset(n_samples=size, n_features=2)
            _, stats = test_lof_consistency(X_perf)
            
    print("\nAll tests completed!") 