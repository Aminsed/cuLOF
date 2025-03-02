#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cuda_lof.cuh"

/**
 * CUDA kernel to find k-nearest neighbors for each point
 * Optimized version with improved memory access patterns
 */
__global__ void find_knn_kernel(const float* dist_matrix, int n_points, int k, 
                               int* indices, float* distances) {
    // Each thread processes one point
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (point_idx < n_points) {
        // Point to the start of distances for this point
        const float* point_distances = &dist_matrix[point_idx * n_points];
        
        // Initialize indices and distances for this point
        int* point_indices = &indices[point_idx * k];
        float* point_dists = &distances[point_idx * k];
        
        // Initialize with positive infinity for scikit-learn compatibility
        for (int i = 0; i < k; i++) {
            point_indices[i] = -1;
            point_dists[i] = INFINITY;  // Use C99 INFINITY constant
        }
        
        // Use registers to track indices and distances for best performance
        // This avoids repeated global memory access during sorting
        float topk_distances[32];  // Assumes k <= 32, adjust if needed
        int topk_indices[32];
        
        // Initialize local arrays
        const int local_k = min(k, 32);  // Safety check
        for (int i = 0; i < local_k; i++) {
            topk_distances[i] = INFINITY;
            topk_indices[i] = -1;
        }
        
        // Process points in chunks to better utilize cache
        const int chunk_size = 64;  // Adjust based on cache size
        for (int chunk_start = 0; chunk_start < n_points; chunk_start += chunk_size) {
            int chunk_end = min(chunk_start + chunk_size, n_points);
            
            // Find nearest neighbors in this chunk
            for (int j = chunk_start; j < chunk_end; j++) {
                // Skip self
                if (j == point_idx) continue;
                
                float dist = point_distances[j];
                
                // Skip NaN distances (scikit-learn behavior)
                if (isnan(dist)) continue;
                
                // Check if this distance should be in the k nearest
                // Find position to insert (if any)
                int insert_pos = local_k;
                for (int i = 0; i < local_k; i++) {
                    if (dist < topk_distances[i]) {
                        insert_pos = i;
                        break;
                    }
                }
                
                // If we found a position to insert
                if (insert_pos < local_k) {
                    // Shift elements to make room
                    for (int i = local_k - 1; i > insert_pos; i--) {
                        topk_indices[i] = topk_indices[i - 1];
                        topk_distances[i] = topk_distances[i - 1];
                    }
                    
                    // Insert the new neighbor
                    topk_indices[insert_pos] = j;
                    topk_distances[insert_pos] = dist;
                }
            }
        }
        
        // Copy the results back to global memory
        for (int i = 0; i < local_k; i++) {
            point_indices[i] = topk_indices[i];
            point_dists[i] = topk_distances[i];
        }
        
        // Count valid neighbors (those with finite distances)
        int valid_count = 0;
        for (int i = 0; i < k; i++) {
            if (point_indices[i] >= 0 && isfinite(point_dists[i])) {
                valid_count++;
            } else {
                // Exit at first invalid entry
                break;
            }
        }
        
        // Mark any remaining slots as invalid
        for (int i = valid_count; i < k; i++) {
            point_indices[i] = -1;
            point_dists[i] = 0.0f;  // This value tells LOF algorithm to ignore
        }
    }
}

/**
 * Finds k-nearest neighbors for each point
 */
cudaError_t find_knn(const float* dist_matrix, int n_points, int k, int* indices, float* distances) {
    cudaError_t err;
    
    // Ensure k is valid and not too large for our kernel
    if (k <= 0 || k >= n_points || k > 32) {
        return cudaErrorInvalidValue;
    }
    
    // Create a CUDA stream for asynchronous execution
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) return err;
    
    // Determine optimal block size for better occupancy
    int minGridSize, blockSize;
    err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, find_knn_kernel, 0, 0);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Configure kernel execution for better performance
    int gridSize = (n_points + blockSize - 1) / blockSize;
    
    // Launch kernel asynchronously
    find_knn_kernel<<<gridSize, blockSize, 0, stream>>>(dist_matrix, n_points, k, indices, distances);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }
    
    // Wait for kernel to finish
    err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return err;
} 