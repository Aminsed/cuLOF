#!/usr/bin/env python3
"""
Simple example for using the culof package.

This script demonstrates the basic usage of the CUDA-accelerated
Local Outlier Factor (LOF) implementation for anomaly detection.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

try:
    from culof import LOF
    print("Successfully imported culof!")
except ImportError as e:
    print(f"Error importing culof: {e}")
    print("Make sure the package is installed with proper CUDA support.")
    exit(1)

# Print version information
import culof
print(f"Using culof version: {culof.__version__}")

# Generate sample data
print("Generating sample data...")
n_samples = 1000
n_outliers = 20

# Generate inliers
X, _ = make_blobs(n_samples=n_samples, centers=1, random_state=42)
X = X.astype(np.float32)  # Convert to float32 for optimal performance

# Generate outliers
outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
outliers = outliers.astype(np.float32)

# Combine data
X_with_outliers = np.vstack([X, outliers])

# Create LOF detector
print("Creating LOF detector...")
lof = LOF()
lof.set_k(20)  # Set number of neighbors
lof.set_threshold(1.5)  # Set threshold for outlier detection
lof.set_normalize(True)  # Optional: normalize the data

# Compute LOF scores
print("Computing LOF scores...")
start_time = time.time()
scores = lof.fit_predict(X_with_outliers)
end_time = time.time()

# Get outliers
detected_outliers = lof.get_outliers(scores)

# Print results
print(f"Computation time: {end_time - start_time:.4f} seconds")
print(f"Detected {len(detected_outliers)} outliers out of {len(X_with_outliers)} samples")
print(f"Top 5 highest LOF scores: {sorted(scores, reverse=True)[:5]}")

# Plot results
print("Creating visualization...")
plt.figure(figsize=(10, 6))

# Plot inliers and outliers
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Inliers', alpha=0.5)
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', s=100, label='True Outliers')

# Plot detected outliers
plt.scatter(X_with_outliers[detected_outliers, 0], X_with_outliers[detected_outliers, 1], 
           edgecolors='green', facecolors='none', s=100, linewidths=2, label='Detected Outliers')

plt.legend()
plt.title('CUDA-accelerated LOF Anomaly Detection')
plt.savefig('lof_results.png')
print("Visualization saved as 'lof_results.png'")

print("Example completed successfully!") 