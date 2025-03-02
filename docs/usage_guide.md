# cuLOF Usage Guide

This guide provides detailed instructions and examples for using the `culof` package for anomaly detection with CUDA-accelerated Local Outlier Factor.

## Basic Usage

The core functionality of the `culof` package revolves around the `LOF` class, which provides methods for anomaly detection using the Local Outlier Factor algorithm.

```python
import numpy as np
from culof import LOF

# Create sample data (must be float32 for optimal performance)
X = np.random.randn(1000, 2).astype(np.float32)

# Create an LOF detector
lof = LOF()

# Set parameters
lof.set_k(20)               # Number of neighbors (default: 20)
lof.set_threshold(1.5)      # Threshold for outlier detection (default: 1.5)
lof.set_normalize(True)     # Whether to normalize the data (default: False)

# Compute LOF scores
scores = lof.fit_predict(X)

# Get outliers
outliers = lof.get_outliers(scores)

# Print results
print(f"Detected {len(outliers)} outliers out of {len(X)} samples")
```

## API Reference

### LOF Class

The `LOF` class is the main class for performing anomaly detection.

#### Initialization

```python
lof = LOF()
```

#### Methods

- **set_k(k: int)**:
  Sets the number of neighbors to use for LOF computation.
  ```python
  lof.set_k(20)  # Use 20 neighbors
  ```

- **set_threshold(threshold: float)**:
  Sets the threshold for outlier detection. Points with LOF scores higher than this threshold are considered outliers.
  ```python
  lof.set_threshold(1.5)  # Set threshold to 1.5
  ```

- **set_normalize(normalize: bool)**:
  Sets whether to normalize the input data before computation.
  ```python
  lof.set_normalize(True)  # Enable normalization
  ```

- **fit_predict(X: np.ndarray) -> np.ndarray**:
  Computes LOF scores for the input data. Returns an array of LOF scores.
  ```python
  scores = lof.fit_predict(X)
  ```

- **get_outliers(scores: np.ndarray) -> np.ndarray**:
  Gets the indices of outliers based on LOF scores. Returns an array of indices.
  ```python
  outliers = lof.get_outliers(scores)
  ```

## Example Use Cases

### 1. Anomaly Detection in a Dataset

```python
import numpy as np
from sklearn.datasets import make_blobs
from culof import LOF
import matplotlib.pyplot as plt

# Generate sample data: 1000 samples, 1 cluster
X, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
X = X.astype(np.float32)

# Add outliers
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
outliers = outliers.astype(np.float32)
X_with_outliers = np.vstack([X, outliers])

# Create LOF detector
lof = LOF()
lof.set_k(20)

# Compute LOF scores
scores = lof.fit_predict(X_with_outliers)

# Get outliers
detected_outliers = lof.get_outliers(scores)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Inliers')
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', label='True Outliers')
plt.scatter(X_with_outliers[detected_outliers, 0], X_with_outliers[detected_outliers, 1], 
           edgecolors='green', facecolors='none', s=100, linewidths=2, label='Detected Outliers')
plt.legend()
plt.title('LOF Anomaly Detection')
plt.show()

print(f"Detected {len(detected_outliers)} outliers out of {len(X_with_outliers)} samples")
```

### 2. Setting Different Thresholds

```python
# Try different thresholds
thresholds = [1.2, 1.5, 2.0, 3.0]
results = []

for threshold in thresholds:
    lof.set_threshold(threshold)
    scores = lof.fit_predict(X_with_outliers)
    outliers = lof.get_outliers(scores)
    results.append(len(outliers))
    
    print(f"Threshold {threshold}: Detected {len(outliers)} outliers")

# Plot number of outliers vs threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, results, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Number of Outliers Detected')
plt.title('Effect of Threshold on Outlier Detection')
plt.grid(True)
plt.show()
```

### 3. Performance Comparison with sklearn

```python
import time
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from culof import LOF

# Generate larger dataset
X = np.random.randn(10000, 2).astype(np.float32)

# scikit-learn LOF
start_time = time.time()
sklearn_lof = LocalOutlierFactor(n_neighbors=20)
sklearn_lof.fit_predict(X)
sklearn_time = time.time() - start_time
print(f"scikit-learn LOF time: {sklearn_time:.4f} seconds")

# CUDA LOF
start_time = time.time()
cuda_lof = LOF()
cuda_lof.set_k(20)
scores = cuda_lof.fit_predict(X)
cuda_time = time.time() - start_time
print(f"CUDA LOF time: {cuda_time:.4f} seconds")

print(f"Speedup: {sklearn_time / cuda_time:.2f}x")
```

## Tips for Optimal Performance

1. **Use `float32` Data Type**:
   - Convert your input data to `np.float32` for optimal performance with CUDA.
   ```python
   X = X.astype(np.float32)
   ```

2. **Batch Processing for Very Large Datasets**:
   - For extremely large datasets, you might need to process in batches if you're running into memory issues.

3. **Choose k Appropriately**:
   - The choice of `k` (number of neighbors) significantly impacts the results.
   - Too small: High variance, sensitive to local noise.
   - Too large: Smooths out local density variations.
   - A common rule of thumb is to use k ≈ sqrt(n) where n is the number of data points.

4. **Experiment with Thresholds**:
   - The threshold determines what points are considered outliers.
   - Start with the default (1.5) and adjust based on your domain knowledge.

5. **Normalize Data When Appropriate**:
   - If features are on different scales, normalization can improve results.
   - Use `lof.set_normalize(True)` to enable normalization.

## Common Issues and Solutions

### ImportError

```
ImportError: Could not import CUDA LOF module. Make sure it's built and installed.
```

**Solution**: Ensure CUDA toolkit is installed and properly configured. Reinstall the package:

```bash
pip install -v culof
```

### CUDA Errors

```
RuntimeError: CUDA error: xxx
```

**Solution**: Check your CUDA installation, GPU drivers, and ensure your GPU is CUDA-capable.

### AttributeError During Installation

```
AttributeError: 'NoneType' object has no attribute 'name'
```

**Solution**: This typically indicates missing or incorrect CUDA configuration. Ensure CUDA toolkit is installed and properly configured. 