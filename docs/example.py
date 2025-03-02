#!/usr/bin/env python
"""
Example usage of CUDA-accelerated LOF implementation
and comparison with scikit-learn's LOF implementation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Ensure img directory exists
img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'img')
os.makedirs(img_dir, exist_ok=True)

# Import our CUDA-accelerated LOF
try:
    import cuda_lof
    CUDA_AVAILABLE = True
except ImportError:
    try:
        import _cuda_lof as cuda_lof
        CUDA_AVAILABLE = True
    except ImportError:
        print("CUDA LOF module not found. Run 'python setup.py install' first.")
        CUDA_AVAILABLE = False

def generate_dataset(n_samples=1000, n_outliers=50, n_features=2, centers=3, random_state=42):
    """Generate a synthetic dataset with outliers."""
    # Generate clusters
    X, _ = make_blobs(n_samples=n_samples - n_outliers, 
                      centers=centers,
                      n_features=n_features, 
                      random_state=random_state)
    
    # Generate outliers
    rng = np.random.RandomState(random_state)
    outliers_range = np.max(np.abs(X)) * 2
    outliers = rng.uniform(-outliers_range, outliers_range, 
                          size=(n_outliers, n_features))
    
    # Combine inliers and outliers
    X = np.vstack([X, outliers])
    
    # Create ground truth labels (1 for outliers, 0 for inliers)
    y = np.zeros(n_samples, dtype=int)
    y[n_samples - n_outliers:] = 1
    
    # Shuffle the data
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    return X, y

def plot_results(X, y_true, y_scores_sklearn, y_scores_cuda=None, name="lof_comparison"):
    """Plot the results of LOF detection."""
    fig, axes = plt.subplots(1, 2 if y_scores_cuda is not None else 1, figsize=(12, 5))
    
    # If only one plot, make axes a list for consistent indexing
    if y_scores_cuda is None:
        axes = [axes]
    
    # Plot sklearn results
    axes[0].scatter(X[:, 0], X[:, 1], c=y_scores_sklearn, cmap='viridis', 
                   alpha=0.7, edgecolors='k', s=50)
    axes[0].set_title('scikit-learn LOF Scores')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Mark true outliers with red circles
    outliers = np.where(y_true == 1)[0]
    axes[0].scatter(X[outliers, 0], X[outliers, 1], s=80, 
                   facecolors='none', edgecolors='r', linewidths=2)
    
    # Plot CUDA results if available
    if y_scores_cuda is not None:
        axes[1].scatter(X[:, 0], X[:, 1], c=y_scores_cuda, cmap='viridis', 
                       alpha=0.7, edgecolors='k', s=50)
        axes[1].set_title('CUDA LOF Scores')
        axes[1].set_xlabel('Feature 1')
        
        # Mark true outliers with red circles
        axes[1].scatter(X[outliers, 0], X[outliers, 1], s=80, 
                       facecolors='none', edgecolors='r', linewidths=2)
    
    plt.tight_layout()
    
    # Save to img directory
    img_path = os.path.join(img_dir, f"{name}.png")
    plt.savefig(img_path)
    print(f"Saved visualization to {img_path}")
    plt.close()

def compare_performance(sizes=[1000, 5000, 10000, 20000]):
    """Compare performance between scikit-learn and CUDA LOF."""
    sklearn_times = []
    cuda_times = []
    
    for size in sizes:
        print(f"Testing with {size} samples...")
        
        # Generate dataset
        X, y = generate_dataset(n_samples=size, n_outliers=int(size * 0.05), 
                               n_features=2, centers=3)
        
        # Time scikit-learn LOF
        start_time = time.time()
        sklearn_lof = LocalOutlierFactor(n_neighbors=20)
        sklearn_lof.fit_predict(X)
        sklearn_time = time.time() - start_time
        sklearn_times.append(sklearn_time)
        print(f"  scikit-learn LOF: {sklearn_time:.4f} seconds")
        
        # Time CUDA LOF
        if CUDA_AVAILABLE:
            try:
                start_time = time.time()
                lof = cuda_lof.LOF(k=20)
                lof.fit_predict(X)
                cuda_time = time.time() - start_time
                cuda_times.append(cuda_time)
                print(f"  CUDA LOF: {cuda_time:.4f} seconds")
                print(f"  Speedup: {sklearn_time/cuda_time:.2f}x")
            except RuntimeError as e:
                print(f"  CUDA LOF failed: {e}")
                cuda_times.append(None)
        else:
            print("  CUDA LOF not available")
            cuda_times.append(None)
    
    # Filter out None values for plotting
    valid_indices = [i for i, t in enumerate(cuda_times) if t is not None]
    valid_sizes = [sizes[i] for i in valid_indices]
    valid_sklearn_times = [sklearn_times[i] for i in valid_indices]
    valid_cuda_times = [cuda_times[i] for i in valid_indices]
    
    # Create performance plot
    plt.figure(figsize=(10, 6))
    
    # Plot execution times
    plt.subplot(1, 2, 1)
    plt.plot(valid_sizes, valid_sklearn_times, 'o-', label='scikit-learn LOF')
    plt.plot(valid_sizes, valid_cuda_times, 's-', label='CUDA LOF')
    plt.xlabel('Dataset Size (samples)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('LOF Execution Time Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    speedups = [s/c for s, c in zip(valid_sklearn_times, valid_cuda_times)]
    plt.plot(valid_sizes, speedups, 'D-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Dataset Size (samples)')
    plt.ylabel('Speedup (scikit-learn time / CUDA time)')
    plt.title('CUDA LOF Speedup')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save to img directory
    img_path = os.path.join(img_dir, "lof_performance_comparison.png")
    plt.savefig(img_path)
    print(f"Saved performance comparison to {img_path}")
    plt.close()

def generate_and_test_different_datasets():
    """Generate and test LOF on different kinds of datasets."""
    from sklearn.datasets import make_moons, make_circles
    
    # Test on moons dataset
    X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)
    # Add some outliers
    outliers = np.random.uniform(-2, 2, size=(50, 2))
    X = np.vstack([X, outliers])
    y_true = np.zeros(1050)
    y_true[1000:] = 1
    test_dataset(X, y_true, "lof_moons")
    
    # Test on circles dataset
    X, _ = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X = StandardScaler().fit_transform(X)
    # Add some outliers
    outliers = np.random.uniform(-2, 2, size=(50, 2))
    X = np.vstack([X, outliers])
    y_true = np.zeros(1050)
    y_true[1000:] = 1
    test_dataset(X, y_true, "lof_circles")
    
    # Test on blobs dataset with varying density
    X1, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.5, random_state=42)
    X2, _ = make_blobs(n_samples=700, centers=[[3, 3]], cluster_std=1.0, random_state=42)
    X = np.vstack([X1, X2])
    X = StandardScaler().fit_transform(X)
    y_true = np.zeros(1000)
    test_dataset(X, y_true, "lof_varying_density")
    
    # Test on outlier clusters
    X, _ = make_blobs(n_samples=950, centers=3, cluster_std=0.5, random_state=42)
    outlier_cluster, _ = make_blobs(n_samples=50, centers=[[8, 8]], cluster_std=0.5, random_state=42)
    X = np.vstack([X, outlier_cluster])
    X = StandardScaler().fit_transform(X)
    y_true = np.zeros(1000)
    y_true[950:] = 1
    test_dataset(X, y_true, "lof_outlier_clusters")

def test_dataset(X, y_true, name):
    """Test LOF on a specific dataset."""
    print(f"Testing on {name} dataset...")
    X = X.astype(np.float32)
    
    # scikit-learn LOF
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred_sklearn = clf.fit_predict(X)
    scores_sklearn = -clf.negative_outlier_factor_
    
    # CUDA LOF
    if CUDA_AVAILABLE:
        try:
            lof = cuda_lof.LOF(k=20)
            scores_cuda = lof.fit_predict(X)
            
            # Create score comparison plot
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(scores_sklearn, bins=50, alpha=0.5, label='scikit-learn')
            plt.hist(scores_cuda, bins=50, alpha=0.5, label='CUDA')
            plt.title('LOF Score Distribution')
            plt.xlabel('LOF Score')
            plt.ylabel('Count')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(scores_sklearn, scores_cuda)
            plt.plot([min(scores_sklearn), max(scores_sklearn)], 
                     [min(scores_sklearn), max(scores_sklearn)], 'r--')
            plt.title('LOF Score Comparison')
            plt.xlabel('scikit-learn LOF Scores')
            plt.ylabel('CUDA LOF Scores')
            
            plt.tight_layout()
            score_img_path = os.path.join(img_dir, f"{name}_score_comparison.png")
            plt.savefig(score_img_path)
            print(f"Saved score comparison to {score_img_path}")
            plt.close()
            
            # Plot spatial representation
            plot_results(X, y_true, scores_sklearn, scores_cuda, name)
        except Exception as e:
            print(f"Error running CUDA LOF on {name} dataset: {e}")
            # Plot only scikit-learn results
            plot_results(X, y_true, scores_sklearn, None, name)
    else:
        # Plot only scikit-learn results
        plot_results(X, y_true, scores_sklearn, None, name)

def main():
    # Generate a synthetic dataset
    X, y_true = generate_dataset(n_samples=1000, n_outliers=50)
    X = X.astype(np.float32)  # Use float32 for CUDA
    
    print(f"Dataset shape: {X.shape}")
    
    # scikit-learn LOF
    print("Running scikit-learn LOF...")
    start_time = time.time()
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred_sklearn = clf.fit_predict(X)
    sklearn_time = time.time() - start_time
    print(f"scikit-learn LOF time: {sklearn_time:.4f} seconds")
    
    # Get negative outlier factor (higher means more outlier-like)
    scores_sklearn = -clf.negative_outlier_factor_
    
    # CUDA LOF
    if CUDA_AVAILABLE:
        print("Running CUDA LOF...")
        start_time = time.time()
        lof = cuda_lof.LOF(k=20)
        scores_cuda = lof.fit_predict(X)
        cuda_time = time.time() - start_time
        print(f"CUDA LOF time: {cuda_time:.4f} seconds")
        print(f"Speedup: {sklearn_time / cuda_time:.2f}x")
        
        # Plot results
        plot_results(X, y_true, scores_sklearn, scores_cuda)
        
        # Compare with different dataset sizes
        compare_performance()
        
        # Generate and test on different datasets
        generate_and_test_different_datasets()
    else:
        # Plot scikit-learn results only
        plot_results(X, y_true, scores_sklearn)

if __name__ == "__main__":
    main() 