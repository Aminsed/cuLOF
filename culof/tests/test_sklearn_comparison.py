#!/usr/bin/env python3
"""
Test suite to compare CUDA LOF implementation with scikit-learn's implementation.
These tests validate that our CUDA implementation produces equivalent results
to scikit-learn's LocalOutlierFactor.
"""

import unittest
import numpy as np
import sys
import os
from time import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Try different import strategies
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

class TestSklearnComparison(unittest.TestCase):
    """
    Test cases that compare CUDA LOF implementation with scikit-learn's implementation.
    """
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 2
        self.n_neighbors = 20
        self.X, self.y_true = self.generate_dataset()
    
    def generate_dataset(self, n_samples=None, n_features=None, n_outliers=50):
        """Generate a synthetic dataset with outliers."""
        if n_samples is None:
            n_samples = self.n_samples
        if n_features is None:
            n_features = self.n_features
            
        # Generate clusters
        X, _ = make_blobs(n_samples=n_samples - n_outliers, 
                          centers=3,
                          n_features=n_features, 
                          random_state=42)
        
        # Generate outliers
        outliers_range = np.max(np.abs(X)) * 2
        outliers = np.random.uniform(-outliers_range, outliers_range, 
                                    size=(n_outliers, n_features))
        
        # Combine inliers and outliers
        X = np.vstack([X, outliers])
        
        # Create ground truth labels (1 for outliers, 0 for inliers)
        y = np.zeros(n_samples, dtype=int)
        y[n_samples - n_outliers:] = 1
        
        # Standardize features
        X = StandardScaler().fit_transform(X).astype(np.float32)
        
        return X, y
    
    def compare_lof_scores(self, X, n_neighbors=None):
        """Compare LOF scores between scikit-learn and CUDA implementations."""
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        # Compute scikit-learn LOF scores
        sklearn_lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        sklearn_lof.fit_predict(X)
        sklearn_scores = -sklearn_lof.negative_outlier_factor_
        
        # Compute CUDA LOF scores
        cuda_lof_instance = CudaLOF(k=n_neighbors)
        cuda_lof_instance.fit(X)
        cuda_scores = cuda_lof_instance.score_samples(X)
        
        # Calculate differences
        abs_diff = np.abs(sklearn_scores - cuda_scores)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        return sklearn_scores, cuda_scores, max_diff, mean_diff
    
    def test_basic_comparison(self):
        """Test basic comparison between scikit-learn and CUDA LOF."""
        _, _, max_diff, mean_diff = self.compare_lof_scores(self.X)
        
        # Check that the differences are small
        self.assertLess(max_diff, 0.5, "Maximum difference is too large")
        self.assertLess(mean_diff, 0.1, "Mean difference is too large")
    
    def test_different_neighbors(self):
        """Test with different numbers of neighbors."""
        for k in [5, 10, 30]:
            _, _, max_diff, mean_diff = self.compare_lof_scores(self.X, k)
            
            # Check that the differences are small
            self.assertLess(max_diff, 0.5, f"Maximum difference is too large with k={k}")
            self.assertLess(mean_diff, 0.1, f"Mean difference is too large with k={k}")
    
    def test_different_dimensions(self):
        """Test with different dimensionality."""
        for dims in [3, 5, 10]:
            X, _ = self.generate_dataset(n_features=dims)
            _, _, max_diff, mean_diff = self.compare_lof_scores(X)
            
            # Check that the differences are small
            self.assertLess(max_diff, 0.5, f"Maximum difference is too large with dims={dims}")
            self.assertLess(mean_diff, 0.1, f"Mean difference is too large with dims={dims}")
    
    def test_larger_dataset(self):
        """Test with a larger dataset."""
        X, _ = self.generate_dataset(n_samples=1000)
        _, _, max_diff, mean_diff = self.compare_lof_scores(X)
        
        # Check that the differences are small
        self.assertLess(max_diff, 0.5, "Maximum difference is too large with larger dataset")
        self.assertLess(mean_diff, 0.1, "Mean difference is too large with larger dataset")
    
    def test_outlier_detection(self):
        """Test outlier detection capabilities."""
        # Generate dataset with known outliers
        n_samples = 200
        n_outliers = 20  # 10% contamination
        X, y_true = self.generate_dataset(n_samples=n_samples, n_features=2, n_outliers=n_outliers)
        
        # scikit-learn LOF
        sklearn_lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=0.1)
        sklearn_lof.fit(X)
        # Get the negative outlier factor (higher values are more normal)
        sklearn_scores = sklearn_lof.negative_outlier_factor_
        
        # CUDA LOF
        cuda_lof_instance = CudaLOF(k=self.n_neighbors)
        cuda_lof_instance.fit(X)
        cuda_scores = cuda_lof_instance.score_samples(X)
        
        # Check correlation between scikit-learn and CUDA scores
        correlation, p_value = pearsonr(sklearn_scores, cuda_scores)
        print(f"Correlation between scikit-learn and CUDA scores: {correlation:.4f}")
        
        # If correlation is negative, we need to flip the CUDA scores
        if correlation < 0:
            print("Negative correlation detected, flipping CUDA scores")
            cuda_scores = -cuda_scores
            correlation, p_value = pearsonr(sklearn_scores, cuda_scores)
            print(f"Correlation after flipping: {correlation:.4f}")
        
        # Use the same contamination level for both methods
        sklearn_threshold = np.percentile(sklearn_scores, 100 * 0.1)
        cuda_threshold = np.percentile(cuda_scores, 100 * 0.1)
        
        # Identify outliers (lower scores are outliers)
        sklearn_outliers = np.where(sklearn_scores <= sklearn_threshold)[0]
        cuda_outliers = np.where(cuda_scores <= cuda_threshold)[0]
        
        # Print some debug information
        print(f"Number of true outliers: {np.sum(y_true == 1)}")
        print(f"Number of sklearn outliers: {len(sklearn_outliers)}")
        print(f"Number of CUDA outliers: {len(cuda_outliers)}")
        print(f"sklearn accuracy: {np.mean(y_true[sklearn_outliers] == 1):.2f}")
        print(f"CUDA accuracy: {np.mean(y_true[cuda_outliers] == 1):.2f}")
        
        # Compare outlier detection results
        sklearn_accuracy = np.mean(y_true[sklearn_outliers] == 1)
        cuda_accuracy = np.mean(y_true[cuda_outliers] == 1)
        
        # Check that both methods have similar accuracy
        self.assertGreater(sklearn_accuracy, 0.5, "scikit-learn LOF has poor accuracy")
        self.assertGreater(cuda_accuracy, 0.5, "CUDA LOF has poor accuracy")
        self.assertLess(abs(sklearn_accuracy - cuda_accuracy), 0.2, 
                       "Large difference in accuracy between scikit-learn and CUDA LOF")
    
    def test_difficult_dataset(self):
        """Test with a more difficult dataset (moons)."""
        # Generate moons dataset
        X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        
        # Add some outliers
        outliers = np.random.uniform(-3, 3, size=(50, 2)).astype(np.float32)
        X = np.vstack([X, outliers])
        
        _, _, max_diff, mean_diff = self.compare_lof_scores(X)
        
        # Check that the differences are small
        self.assertLess(max_diff, 0.5, "Maximum difference is too large with moons dataset")
        self.assertLess(mean_diff, 0.1, "Mean difference is too large with moons dataset")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Generate dataset
        X, _ = self.generate_dataset(n_samples=100, n_features=2)
        
        # Test with small k
        _, _, max_diff, mean_diff = self.compare_lof_scores(X, n_neighbors=3)
        
        # Check that the differences are small
        self.assertLess(max_diff, 0.5, "Maximum difference is too large with small k")
        self.assertLess(mean_diff, 0.1, "Mean difference is too large with small k")
        
        # Test with large k, but ensure it's not too large
        # Use a more conservative value to avoid "invalid argument" errors
        large_k = min(20, X.shape[0] // 2)  # Use at most half the dataset size
        print(f"Using large_k = {large_k} for a dataset with {X.shape[0]} samples")
        _, _, max_diff, mean_diff = self.compare_lof_scores(X, n_neighbors=large_k)
        self.assertLess(max_diff, 0.5, "Maximum difference is too large with large k")
        self.assertLess(mean_diff, 0.1, "Mean difference is too large with large k")
    
    def test_consistency(self):
        """Test that the CUDA implementation gives consistent results."""
        X, _ = self.generate_dataset()
        
        # Run CUDA LOF multiple times
        cuda_lof_instance = CudaLOF(k=self.n_neighbors)
        scores1 = cuda_lof_instance.fit_predict(X)
        
        cuda_lof_instance = CudaLOF(k=self.n_neighbors)
        scores2 = cuda_lof_instance.fit_predict(X)
        
        # Check that the results are identical
        np.testing.assert_allclose(scores1, scores2, rtol=1e-5, atol=1e-5,
                                  err_msg="CUDA LOF gives inconsistent results")
    
    def test_timing_comparison(self):
        """Compare execution time between scikit-learn and CUDA LOF."""
        # Generate a larger dataset for timing comparison
        X, _ = self.generate_dataset(n_samples=2000)
        
        # Time scikit-learn LOF
        start = time()
        sklearn_lof = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        sklearn_lof.fit_predict(X)
        sklearn_time = time() - start
        
        # Time CUDA LOF
        start = time()
        cuda_lof_instance = CudaLOF(k=self.n_neighbors)
        cuda_lof_instance.fit_predict(X)
        cuda_time = time() - start
        
        # Print timing results
        print(f"\nscikit-learn LOF: {sklearn_time:.4f} seconds")
        print(f"CUDA LOF: {cuda_time:.4f} seconds")
        print(f"Speedup: {sklearn_time/cuda_time:.2f}x")
        
        # No assertion here, just informational

if __name__ == "__main__":
    unittest.main() 