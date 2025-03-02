# Include Directory

This directory contains the header files for the CUDA-accelerated LOF algorithm.

## Files

### cuda_lof.cuh

This header file contains CUDA-specific declarations, including:
- CUDA kernel function declarations
- Device function prototypes
- CUDA-specific structures and utilities

This file is only needed for internal CUDA kernel implementations and should not be included directly by client code.

### cuda_lof.h

This is the main C++ API header that should be included by client applications. It provides:
- The LOF class interface
- Configuration structures
- Error handling utilities
- Function prototypes for the public API

## Usage

For C++ applications:

```cpp
#include "cuda_lof.h"

// Create LOF configuration
LOFConfig config;
config.k = 20;
config.normalize = true;

// Allocate memory for scores
float* scores = new float[n_points];

// Compute LOF scores
cudaError_t err = compute_lof(points, n_points, n_dims, &config, scores);
if (err != cudaSuccess) {
    // Handle error
}

// Use LOF scores
// ...

delete[] scores;
``` 