#!/usr/bin/env python3
"""
Numerical accuracy tests for CUDA LOF implementation.
This script performs detailed numerical comparisons between scikit-learn and CUDA LOF,
and generates visualizations of the differences.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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

def generate_dataset(n_samples=500, n_features=2, n_outliers=50, dataset_type='blobs'):
    """Generate a dataset with outliers for testing."""
    if dataset_type == 'blobs':
        # Generate clustered data
        X, _ = make_blobs(n_samples=n_samples-n_outliers, 
                         centers=3, 
                         n_features=n_features, 
                         random_state=42)
    elif dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples-n_outliers, noise=0.05, random_state=42)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples-n_outliers, noise=0.05, random_state=42, factor=0.5)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Generate outliers
    outlier_range = np.max(np.abs(X)) * 2
    outliers = np.random.uniform(-outlier_range, outlier_range, 
                               size=(n_outliers, n_features))
    
    # Combine inliers and outliers
    X = np.vstack([X, outliers])
    
    # True outlier labels
    y = np.zeros(n_samples, dtype=int)
    y[n_samples-n_outliers:] = 1
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)  # Use float32 for CUDA
    
    return X, y

def compare_lof_scores(X, n_neighbors=20, verbose=True):
    """
    Compare LOF scores between sklearn and CUDA implementation.
    
    Returns:
        tuple: (sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff)
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
    mean_diff = np.mean(np.abs(diffs))
    std_diff = np.std(diffs)
    
    if verbose:
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Standard deviation of differences: {std_diff:.6f}")
        
        # Count scores that differ by more than thresholds
        thresholds = [0.001, 0.01, 0.1]
        for threshold in thresholds:
            count = np.sum(np.abs(diffs) > threshold)
            percent = count / len(diffs) * 100
            print(f"Scores differing by more than {threshold}: {count} ({percent:.2f}%)")
    
    return sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff

def plot_score_comparison(sklearn_scores, cuda_scores, title="LOF Score Comparison"):
    """Plot comparison between sklearn and CUDA LOF scores."""
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of scores
    plt.subplot(1, 2, 1)
    plt.scatter(sklearn_scores, cuda_scores, alpha=0.5)
    
    # Add perfect correlation line
    min_val = min(np.min(sklearn_scores), np.min(cuda_scores))
    max_val = max(np.max(sklearn_scores), np.max(cuda_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('scikit-learn LOF Scores')
    plt.ylabel('CUDA LOF Scores')
    plt.title('Score Correlation')
    plt.grid(True, alpha=0.3)
    
    # Histogram of differences
    plt.subplot(1, 2, 2)
    differences = sklearn_scores - cuda_scores
    plt.hist(differences, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Score Difference (sklearn - CUDA)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Score Differences')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return plt.gcf()

def test_dataset(dataset_type='blobs', n_samples=500, n_features=2, n_outliers=50, k=20):
    """Test and visualize results on a specific dataset."""
    print(f"\n=== Testing {dataset_type} dataset ===")
    print(f"Samples: {n_samples}, Features: {n_features}, Outliers: {n_outliers}, k: {k}")
    
    # Generate dataset
    X, y = generate_dataset(n_samples, n_features, n_outliers, dataset_type)
    
    # Compare scores
    sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff = compare_lof_scores(X, k)
    
    # If we have 2D data, visualize it
    if n_features == 2:
        plt.figure(figsize=(15, 5))
        
        # Plot ground truth
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title('Ground Truth')
        plt.colorbar(label='Is Outlier')
        plt.grid(True, alpha=0.3)
        
        # Plot sklearn scores
        plt.subplot(1, 3, 2)
        plt.scatter(X[:, 0], X[:, 1], c=sklearn_scores, cmap='plasma', alpha=0.7)
        plt.title('scikit-learn LOF Scores')
        plt.colorbar(label='LOF Score')
        plt.grid(True, alpha=0.3)
        
        # Plot CUDA scores
        plt.subplot(1, 3, 3)
        plt.scatter(X[:, 0], X[:, 1], c=cuda_scores, cmap='plasma', alpha=0.7)
        plt.title('CUDA LOF Scores')
        plt.colorbar(label='LOF Score')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{dataset_type.capitalize()} Dataset Comparison")
        plt.tight_layout()
        plt.savefig(f"lof_{dataset_type}_comparison.png")
    
    # Plot score comparison
    fig = plot_score_comparison(sklearn_scores, cuda_scores, 
                         title=f"LOF Score Comparison - {dataset_type.capitalize()} Dataset")
    fig.savefig(f"lof_{dataset_type}_score_comparison.png")
    
    # Calculate AUC-ROC for outlier detection
    sklearn_auc = roc_auc_score(y, sklearn_scores)
    cuda_auc = roc_auc_score(y, cuda_scores)
    
    print(f"scikit-learn AUC-ROC: {sklearn_auc:.4f}")
    print(f"CUDA LOF AUC-ROC: {cuda_auc:.4f}")
    print(f"AUC-ROC difference: {abs(sklearn_auc - cuda_auc):.4f}")
    
    return sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff

def run_all_tests():
    """Run all numerical accuracy tests."""
    results = []
    
    # Test different dataset types
    for dataset_type in ['blobs', 'moons', 'circles']:
        sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff = test_dataset(
            dataset_type=dataset_type, n_samples=500, n_features=2, n_outliers=50, k=20)
        results.append({
            'dataset': dataset_type,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        })
    
    # Test different k values
    for k in [5, 10, 20, 50]:
        sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff = test_dataset(
            dataset_type='blobs', n_samples=500, n_features=2, n_outliers=50, k=k)
        results.append({
            'dataset': f'blobs_k{k}',
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        })
    
    # Test higher dimensions
    for dim in [3, 5, 10]:
        sklearn_scores, cuda_scores, max_diff, mean_diff, std_diff = test_dataset(
            dataset_type='blobs', n_samples=500, n_features=dim, n_outliers=50, k=20)
        results.append({
            'dataset': f'blobs_{dim}d',
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        })
    
    # Print summary table
    print("\n=== Summary of Numerical Differences ===")
    print("{:<15} {:<15} {:<15} {:<15}".format(
        "Dataset", "Max Diff", "Mean Diff", "Std Diff"))
    print("-" * 60)
    
    for result in results:
        print("{:<15} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            result['dataset'], result['max_diff'], result['mean_diff'], result['std_diff']))

if __name__ == "__main__":
    run_all_tests()
    plt.show()  # Show all figures at the end 