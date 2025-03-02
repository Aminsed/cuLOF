#!/usr/bin/env python3
"""
Unit tests for CUDA-accelerated LOF Python bindings.
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import cuda_lof
    from cuda_lof import LOF, compute_lof
except ImportError:
    try:
        from _cuda_lof import LOF, compute_lof
    except ImportError:
        print("Error: Could not import CUDA LOF module. Make sure it's built and installed.")
        print("Try running: python setup.py install")
        sys.exit(1)

class TestCUDALOF(unittest.TestCase):
    """Test cases for CUDA LOF Python bindings."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple dataset with obvious outliers
        np.random.seed(42)
        # Create a cluster of points
        self.n_samples = 100
        self.n_features = 3
        self.X_inliers = np.random.randn(self.n_samples, self.n_features)
        
        # Create outliers
        self.n_outliers = 5
        self.X_outliers = np.random.uniform(low=5, high=10, 
                                           size=(self.n_outliers, self.n_features))
        
        # Combine inliers and outliers
        self.X = np.vstack([self.X_inliers, self.X_outliers])
        
        # Keep track of true outlier indices
        self.true_outliers = list(range(self.n_samples, self.n_samples + self.n_outliers))
        
        # Parameters for LOF
        self.k = 10
        self.threshold = 1.5  # Threshold for outlier detection
    
    def test_compute_lof(self):
        """Test compute_lof function."""
        # Compute LOF scores
        lof_scores = compute_lof(self.X, self.k)
        
        # Check shape
        self.assertEqual(len(lof_scores), self.n_samples + self.n_outliers)
        
        # Check that outliers have higher LOF scores
        inlier_scores = lof_scores[:self.n_samples]
        outlier_scores = lof_scores[self.n_samples:]
        
        # Average scores
        avg_inlier_score = np.mean(inlier_scores)
        avg_outlier_score = np.mean(outlier_scores)
        
        # Outliers should have higher scores
        self.assertGreater(avg_outlier_score, avg_inlier_score)
    
    def test_get_outliers(self):
        """Test get_outliers function."""
        # Compute LOF scores
        lof_scores = compute_lof(self.X, self.k)
        
        # Get outliers
        outliers = get_outliers(lof_scores, self.threshold)
        
        # Check that at least some true outliers are detected
        # (not all may be detected depending on the dataset and parameters)
        detected_true_outliers = set(outliers).intersection(set(self.true_outliers))
        self.assertGreater(len(detected_true_outliers), 0)
    
    def test_lof_class(self):
        """Test LOF class."""
        # Create LOF detector
        lof = LOF(k=self.k, threshold=self.threshold)
        
        # Compute LOF scores
        lof_scores = lof.fit_predict(self.X)
        
        # Check shape
        self.assertEqual(len(lof_scores), self.n_samples + self.n_outliers)
        
        # Get outliers
        outliers = lof.get_outliers(lof_scores)
        
        # Check that at least some true outliers are detected
        detected_true_outliers = set(outliers).intersection(set(self.true_outliers))
        self.assertGreater(len(detected_true_outliers), 0)
    
    def test_fortran_order(self):
        """Test with Fortran-ordered arrays."""
        # Create a Fortran-ordered array
        X_fortran = np.asfortranarray(self.X)
        self.assertFalse(X_fortran.flags.c_contiguous)
        self.assertTrue(X_fortran.flags.f_contiguous)
        
        # Compute LOF scores
        lof_scores = compute_lof(X_fortran, self.k)
        
        # Check shape
        self.assertEqual(len(lof_scores), self.n_samples + self.n_outliers)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with k=1
        lof_scores = compute_lof(self.X, 1)
        self.assertEqual(len(lof_scores), self.n_samples + self.n_outliers)
        
        # Test with k=n-1
        lof_scores = compute_lof(self.X, self.n_samples + self.n_outliers - 1)
        self.assertEqual(len(lof_scores), self.n_samples + self.n_outliers)
        
        # Test with small dataset
        X_small = np.random.randn(5, self.n_features)
        lof_scores = compute_lof(X_small, 2)
        self.assertEqual(len(lof_scores), 5)

if __name__ == "__main__":
    unittest.main() 