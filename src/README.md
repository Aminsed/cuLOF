# Source Code 

This directory contains the core implementation of the CUDA-accelerated LOF algorithm.

## Directory Structure

- `cuda/` - Contains all CUDA kernel implementations:
  - `distances.cu` - Implementation of distance computation kernels
  - `knn.cu` - Implementation of K-nearest neighbors search
  - `lof.cu` - Implementation of the LOF algorithm steps
  - `lof_cpp.cu` - C++ API implementation for the LOF algorithm
  - `utils.cu` - Utility functions for data processing

## Implementation Details

### Distances Computation

The `distances.cu` file contains the CUDA kernel for computing pairwise Euclidean distances between all points. This operation has a time complexity of O(n²d) (n = number of points, d = dimensions) and is significantly accelerated through GPU parallelization.

### K-Nearest Neighbors Search

The `knn.cu` file implements the k-nearest neighbors search algorithm on the GPU. This operation efficiently sorts the distance matrix to find the k closest points for each point in the dataset.

### LOF Score Computation

The `lof.cu` file contains kernels for:
1. Computing k-distances (distance to the k-th nearest neighbor)
2. Computing reachability distances
3. Computing local reachability density (LRD)
4. Computing the final LOF scores

### C++ API

The `lof_cpp.cu` file provides a C++ API to use the CUDA kernels through a simple interface.

### Utilities

The `utils.cu` file contains utility functions such as data normalization, which can be important for the LOF algorithm's accuracy. 