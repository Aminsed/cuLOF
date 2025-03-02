"""
CUDA-accelerated Local Outlier Factor (LOF) implementation.

This module provides a CUDA-accelerated implementation of the Local Outlier Factor
algorithm for anomaly detection, with a scikit-learn compatible interface.
"""

import numpy as np

try:
    from ._cuda_lof import compute_lof
except ImportError:
    try:
        import _cuda_lof
        compute_lof = _cuda_lof.compute_lof
    except ImportError:
        raise ImportError("Could not import CUDA LOF module. Make sure it's built and installed.")

# Version is now defined only at the end of the file
# to avoid confusion and inconsistencies

class LOF:
    """Local Outlier Factor implementation accelerated with CUDA.
    
    The LOF algorithm is an unsupervised anomaly detection method which computes
    the local density deviation of a given data point with respect to its neighbors.
    
    This implementation is compatible with scikit-learn's LOF interface but uses
    CUDA acceleration for improved performance on larger datasets.
    
    Parameters
    ----------
    k : int, default=20
        Number of neighbors to use for LOF computation.
    normalize : bool, default=False
        Whether to normalize the input data before computation.
    contamination : float, default=0.1
        The proportion of outliers in the data set. Used when calling fit_predict.
        
    Attributes
    ----------
    k : int
        Number of neighbors used for LOF computation.
    normalize : bool
        Whether input data is normalized before computation.
    contamination : float
        The proportion of outliers expected in the data.
    """
    
    def __init__(self, k=20, normalize=False, contamination=0.1):
        self.k = k
        self.normalize = normalize
        self.contamination = contamination
        self._scores = None
    
    def fit(self, X):
        """Fit the LOF model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
            
        # Run LOF algorithm
        self._scores = self._compute_lof_scores(X)
        
        return self
    
    def predict(self, X=None):
        """Predict outliers based on fitted LOF scores.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Input data. If not provided, the training data is used.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Returns -1 for outliers and 1 for inliers.
        """
        if X is None:
            if self._scores is None:
                raise ValueError("Model not fitted, call fit first")
            scores = self._scores
        else:
            X = np.asarray(X, dtype=np.float32, order='C')
            if X.ndim != 2:
                raise ValueError("Input data must be 2-dimensional")
            scores = self._compute_lof_scores(X)
        
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores > threshold, -1, 1)
    
    def fit_predict(self, X):
        """Fit the model and predict outliers in one step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Returns -1 for outliers and 1 for inliers.
        """
        return self.fit(X).predict()
    
    def score_samples(self, X=None):
        """Return the LOF scores for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Input data. If not provided, the training data is used.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Returns LOF scores. Higher scores indicate more abnormal samples.
        """
        if X is None:
            if self._scores is None:
                raise ValueError("Model not fitted, call fit first")
            return self._scores
        else:
            X = np.asarray(X, dtype=np.float32, order='C')
            if X.ndim != 2:
                raise ValueError("Input data must be 2-dimensional")
            return self._compute_lof_scores(X)
    
    def _compute_lof_scores(self, X):
        """Compute LOF scores using the CUDA implementation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            LOF scores.
        """
        n_samples, n_features = X.shape
        
        # Ensure k is valid
        k = min(self.k, n_samples - 1)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
            
        # Call the CUDA implementation
        try:
            # The compute_lof function expects: (points, k, normalize, threshold)
            # Use threshold=1.0 to match scikit-learn's behavior
            scores = compute_lof(X, k, self.normalize, 1.0)
            return scores
        except RuntimeError as e:
            raise RuntimeError(f"Error computing LOF: {e}")

# For backwards compatibility
try:
    from ._cuda_lof import compute_lof
except ImportError:
    pass 

"""
cuLOF: CUDA-accelerated Local Outlier Factor

This package provides a GPU-accelerated implementation of the Local Outlier Factor
algorithm for anomaly detection. It is designed to be compatible with scikit-learn's
LOF interface while providing significant speedups for larger datasets.
"""

# Import the LOF class from the module
try:
    from ._culof import LOF
except ImportError:
    try:
        # Try importing from the old module name for backward compatibility
        from ._cuda_lof import LOF
    except ImportError:
        print("Warning: Failed to import the CUDA-accelerated LOF implementation.")
        print("Make sure the package is properly installed with CUDA support.")
        print("Falling back to a placeholder implementation that will raise an error when used.")
        
        class LOF:
            """Placeholder LOF class when CUDA implementation is not available."""
            
            def __init__(self, *args, **kwargs):
                self._not_available()
            
            def _not_available(self):
                raise ImportError(
                    "CUDA LOF implementation is not available. "
                    "Make sure CUDA is installed and the package is built correctly."
                )
            
            def fit(self, *args, **kwargs):
                self._not_available()
            
            def fit_predict(self, *args, **kwargs):
                self._not_available()
            
            def predict(self, *args, **kwargs):
                self._not_available()
                
            def score_samples(self, *args, **kwargs):
                self._not_available()

# Package metadata
__version__ = '0.1.1'
__author__ = 'Amin Sedaghat'
__email__ = 'amin32846@gmail.com'
__all__ = ['LOF'] 