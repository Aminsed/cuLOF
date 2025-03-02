#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_lof.cuh"

/**
 * CUDA kernel to compute pairwise Euclidean distances between points
 * Each thread computes the distance between one pair of points
 */
__global__ void compute_distances_kernel(const float* points, int n_points, int n_dims, float* dist_matrix) {
    // Get global thread ID (which represents a pair of points)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (row < n_points && col < n_points) {
        // Same point has zero distance
        if (row == col) {
            dist_matrix[row * n_points + col] = 0.0f;
            return;
        }
        
        // Compute squared Euclidean distance
        // Use double precision for accumulation to match numpy's behavior
        double sum = 0.0;
        bool has_nan = false;
        
        // Process dimensions in blocks of 4 for better memory coalescing
        int d = 0;
        for (; d < n_dims - 3; d += 4) {
            // Load point values for 4 dimensions at once
            float val1_0 = points[row * n_dims + d];
            float val1_1 = points[row * n_dims + d + 1];
            float val1_2 = points[row * n_dims + d + 2];
            float val1_3 = points[row * n_dims + d + 3];
            
            float val2_0 = points[col * n_dims + d];
            float val2_1 = points[col * n_dims + d + 1];
            float val2_2 = points[col * n_dims + d + 2];
            float val2_3 = points[col * n_dims + d + 3];
            
            // Check for NaN values - scikit-learn propagates NaNs
            if (isnan(val1_0) || isnan(val2_0) || 
                isnan(val1_1) || isnan(val2_1) ||
                isnan(val1_2) || isnan(val2_2) ||
                isnan(val1_3) || isnan(val2_3)) {
                has_nan = true;
                break;
            }
            
            // Compute differences and add to sum
            double diff0 = (double)val1_0 - (double)val2_0;
            double diff1 = (double)val1_1 - (double)val2_1;
            double diff2 = (double)val1_2 - (double)val2_2;
            double diff3 = (double)val1_3 - (double)val2_3;
            
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        
        // Handle remaining dimensions
        for (; d < n_dims && !has_nan; d++) {
            float val1 = points[row * n_dims + d];
            float val2 = points[col * n_dims + d];
            
            // Check for NaN values
            if (isnan(val1) || isnan(val2)) {
                has_nan = true;
                break;
            }
            
            double diff = (double)val1 - (double)val2;
            sum += diff * diff;
        }
        
        if (has_nan) {
            // scikit-learn/numpy propagates NaNs in distance calculations
            dist_matrix[row * n_points + col] = NAN;
        } else {
            // scikit-learn/numpy uses sqrt without any epsilon or flooring
            dist_matrix[row * n_points + col] = sqrt(sum);
        }
    }
}

/**
 * Computes pairwise Euclidean distances between all points
 */
cudaError_t compute_distances(const float* points, int n_points, int n_dims, float* dist_matrix) {
    cudaError_t err;
    
    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) return err;
    
    // Set seed for CUDA random number generator to ensure deterministic behavior
    err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Determine block size based on device capability
    int minGridSize, blockSize;
    err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, compute_distances_kernel, 0, 0);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Adjust block size to 2D grid for better performance
    // Use power-of-2 dimensions for better memory coalescing
    int block_dim_x = 16;
    int block_dim_y = 16;
    
    // Calculate grid dimensions to cover all points
    dim3 blockDim(block_dim_x, block_dim_y);
    dim3 gridDim((n_points + block_dim_x - 1) / block_dim_x, 
                 (n_points + block_dim_y - 1) / block_dim_y);
    
    // Zero out the distance matrix before computation
    err = cudaMemsetAsync(dist_matrix, 0, n_points * n_points * sizeof(float), stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Launch kernel - just use the simple kernel for now to ensure correctness
    compute_distances_kernel<<<gridDim, blockDim, 0, stream>>>(
        points, n_points, n_dims, dist_matrix);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Synchronize and clean up
    err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return err;
} 