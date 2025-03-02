#!/usr/bin/env python
"""
Benchmark script to compare scikit-learn and CUDA LOF implementations
across various dataset sizes.

This script generates synthetic datasets of increasing size and measures
the execution time of both implementations, creating a performance comparison
visualization.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF
from sklearn.datasets import make_blobs
import os
import pandas as pd
from tqdm import tqdm
import argparse
import sys

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try different import strategies for CUDA LOF
try:
    from python import LOF as CudaLOF
    print("Imported LOF from python package")
except ImportError:
    try:
        import cuda_lof
        CudaLOF = cuda_lof.LOF
        print("Imported LOF from cuda_lof package")
    except ImportError:
        try:
            import _cuda_lof
            CudaLOF = _cuda_lof.LOF
            print("Imported LOF from _cuda_lof package")
        except ImportError:
            print("Error: Could not import CUDA LOF module. Make sure it's built and installed.")
            print("Try running: python setup.py install")
            sys.exit(1)

def generate_dataset(n_samples, n_features=10, centers=3, random_state=42):
    """Generate a synthetic dataset with potential outliers."""
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                     centers=centers, random_state=random_state)
    
    # Add some outliers (about 5% of the data)
    n_outliers = max(1, int(0.05 * n_samples))
    outlier_range = np.max(np.abs(X)) * 2
    
    # Generate outliers in a wider range than the clusters
    outliers = np.random.RandomState(random_state).uniform(
        low=-outlier_range, high=outlier_range, size=(n_outliers, n_features)
    )
    
    # Combine the clusters with outliers
    X = np.vstack([X, outliers])
    
    return X

def benchmark_lof(datasets, k=20, n_runs=3):
    """Benchmark scikit-learn and CUDA LOF implementations."""
    results = []
    
    for X in tqdm(datasets, desc="Benchmarking datasets"):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize results for this dataset
        result = {
            "n_samples": n_samples,
            "n_features": n_features,
            "sklearn_time": [],
            "cuda_time": []
        }
        
        for _ in range(n_runs):
            # Benchmark scikit-learn LOF
            start_time = time.time()
            sklearn_lof = SklearnLOF(n_neighbors=k, algorithm='auto')
            sklearn_lof.fit_predict(X)
            sklearn_time = time.time() - start_time
            result["sklearn_time"].append(sklearn_time)
            
            # Benchmark CUDA LOF
            start_time = time.time()
            cuda_lof = CudaLOF(k=k)
            cuda_lof.fit(X)
            cuda_lof.score_samples(X)
            cuda_time = time.time() - start_time
            result["cuda_time"].append(cuda_time)
        
        # Calculate average times
        result["avg_sklearn_time"] = np.mean(result["sklearn_time"])
        result["avg_cuda_time"] = np.mean(result["cuda_time"])
        result["speedup"] = result["avg_sklearn_time"] / result["avg_cuda_time"]
        
        results.append(result)
    
    return results

def plot_results(results, save_path="img/benchmark_results.png"):
    """Plot benchmark results."""
    results_df = pd.DataFrame(results)
    
    # Extract dataset sizes and times
    dataset_sizes = results_df["n_samples"].values
    sklearn_times = results_df["avg_sklearn_time"].values
    cuda_times = results_df["avg_cuda_time"].values
    speedups = results_df["speedup"].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    ax1.plot(dataset_sizes, sklearn_times, 'o-', label='scikit-learn LOF')
    ax1.plot(dataset_sizes, cuda_times, 's-', label='CUDA LOF')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset Size (number of samples)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('LOF Execution Time Comparison')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend()
    
    # Plot speedup
    ax2.plot(dataset_sizes, speedups, 'D-', color='green')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Dataset Size (number of samples)')
    ax2.set_ylabel('Speedup (sklearn time / CUDA time)')
    ax2.set_title('CUDA LOF Speedup Factor')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")
    
    # Create a summary table
    summary = pd.DataFrame({
        "Dataset Size": dataset_sizes,
        "Sklearn Time (s)": sklearn_times,
        "CUDA Time (s)": cuda_times,
        "Speedup": speedups
    })
    
    print("\nPerformance Summary:")
    print(summary.to_string(index=False))
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Benchmark scikit-learn vs CUDA LOF implementations')
    parser.add_argument('--max-size', type=int, default=10000, 
                        help='Maximum dataset size to test (default: 10000)')
    parser.add_argument('--min-size', type=int, default=100, 
                        help='Minimum dataset size to test (default: 100)')
    parser.add_argument('--n-sizes', type=int, default=6, 
                        help='Number of different dataset sizes to test (default: 6)')
    parser.add_argument('--n-features', type=int, default=10, 
                        help='Number of features in the dataset (default: 10)')
    parser.add_argument('--k', type=int, default=20, 
                        help='Number of neighbors to use for LOF (default: 20)')
    parser.add_argument('--n-runs', type=int, default=3, 
                        help='Number of runs for each configuration (default: 3)')
    parser.add_argument('--output', type=str, default='img/benchmark_results.png',
                        help='Output path for the benchmark plot (default: img/benchmark_results.png)')
    
    args = parser.parse_args()
    
    # Generate dataset sizes (logarithmic scale)
    dataset_sizes = np.logspace(
        np.log10(args.min_size),
        np.log10(args.max_size),
        args.n_sizes
    ).astype(int)
    
    print(f"Benchmarking with dataset sizes: {dataset_sizes}")
    
    # Generate datasets
    datasets = [generate_dataset(n, n_features=args.n_features) for n in dataset_sizes]
    
    # Run benchmarks
    results = benchmark_lof(datasets, k=args.k, n_runs=args.n_runs)
    
    # Plot and save results
    plot_results(results, save_path=args.output)
    
    # Show final result
    plt.show()

if __name__ == "__main__":
    main() 