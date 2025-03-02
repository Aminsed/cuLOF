#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_lof.cuh"

/**
 * CUDA kernel to normalize data by z-score normalization (in-place)
 * Each thread processes one dimension
 */
__global__ void normalize_kernel(float* points, int n_points, int n_dims, 
                                float* means, float* stds) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim_idx < n_dims) {
        // Step 1: Compute mean
        float sum = 0.0f;
        for (int i = 0; i < n_points; i++) {
            sum += points[i * n_dims + dim_idx];
        }
        float mean = sum / n_points;
        means[dim_idx] = mean;
        
        // Step 2: Compute standard deviation
        float var_sum = 0.0f;
        for (int i = 0; i < n_points; i++) {
            float diff = points[i * n_dims + dim_idx] - mean;
            var_sum += diff * diff;
        }
        float std_dev = sqrtf(var_sum / n_points);
        stds[dim_idx] = std_dev;
        
        // Avoid division by zero
        if (std_dev < 1e-8f) {
            std_dev = 1.0f;
        }
        
        // Step 3: Normalize data
        for (int i = 0; i < n_points; i++) {
            points[i * n_dims + dim_idx] = (points[i * n_dims + dim_idx] - mean) / std_dev;
        }
    }
}

/**
 * Normalize input data (in-place using z-score normalization)
 */
cudaError_t normalize_data(float* points, int n_points, int n_dims) {
    cudaError_t err;
    
    // Allocate device memory for means and standard deviations
    float* d_means = nullptr;
    float* d_stds = nullptr;
    
    err = cudaMalloc(&d_means, n_dims * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_stds, n_dims * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_means);
        return err;
    }
    
    // Configure kernel execution
    const int block_size = 256;
    const int grid_size = (n_dims + block_size - 1) / block_size;
    
    // Launch normalization kernel
    normalize_kernel<<<grid_size, block_size>>>(points, n_points, n_dims, d_means, d_stds);
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_means);
        cudaFree(d_stds);
        return err;
    }
    
    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_means);
        cudaFree(d_stds);
        return err;
    }
    
    // Free temporary memory
    cudaFree(d_means);
    cudaFree(d_stds);
    
    return cudaSuccess;
} 