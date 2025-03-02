#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cuda_lof.cuh"

/**
 * CUDA kernel to compute k-distances (distance to the k-th nearest neighbor)
 */
__global__ void compute_kdistances_kernel(int* knn_indices, float* knn_distances, 
                                          int n_points, int k, float* kdistances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points) {
        // k-distance is the distance to the k-th nearest neighbor
        // Note: The k-th nearest neighbor is at index k-1 since indexing starts at 0
        
        // Verify that we have enough valid neighbors
        int valid_neighbors = 0;
        for (int i = 0; i < k; i++) {
            if (knn_indices[idx * k + i] >= 0 && isfinite(knn_distances[idx * k + i])) {
                valid_neighbors++;
            } else {
                break;  // Stop counting at first invalid neighbor
            }
        }
        
        if (valid_neighbors >= k) {
            kdistances[idx] = knn_distances[idx * k + (k - 1)];
        } else if (valid_neighbors > 0) {
            // Use the last valid neighbor's distance
            kdistances[idx] = knn_distances[idx * k + (valid_neighbors - 1)];
        } else {
            // No valid neighbors - use a reasonable default
            kdistances[idx] = 1.0f;  // Default to 1.0 if no valid neighbors
        }
        
        // Ensure k-distance is positive to avoid division by zero later
        // Use a slightly higher epsilon to match scikit-learn behavior
        kdistances[idx] = fmaxf(kdistances[idx], 1e-8f);
    }
}

/**
 * CUDA kernel to compute reachability distances
 * Reachability distance between points a and b is max(k-distance(b), dist(a,b))
 */
__global__ void compute_reach_dists_kernel(int* knn_indices, float* knn_distances, 
                                          float* kdistances, int n_points, int k,
                                          float* reach_dists) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points) {
        // For each neighbor of point idx
        for (int i = 0; i < k; i++) {
            int neighbor_idx = knn_indices[idx * k + i];
            
            // Skip invalid neighbor indices (-1)
            if (neighbor_idx < 0) {
                reach_dists[idx * k + i] = 0.0f;  // Mark as invalid
                continue;
            }
            
            // Get the k-distance of the neighbor
            float kd_neighbor = kdistances[neighbor_idx];
            
            // Get the distance from point to neighbor
            float dist = knn_distances[idx * k + i];
            
            // Reachability distance is the max of k-distance and actual distance
            reach_dists[idx * k + i] = fmaxf(kd_neighbor, dist);
            
            // Ensure reachability distance is positive for numerical stability
            // Use a slightly higher epsilon to match scikit-learn behavior
            reach_dists[idx * k + i] = fmaxf(reach_dists[idx * k + i], 1e-8f);
        }
    }
}

/**
 * CUDA kernel to compute local reachability density (LRD)
 * LRD of a point is the inverse of the average reachability distance to its k neighbors
 */
__global__ void compute_lrd_kernel(int* knn_indices, float* reach_dists, 
                                  int n_points, int k, float* lrd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points) {
        double sum_reach_dist = 0.0;
        int valid_count = 0;
        
        // Sum the reachability distances to all neighbors using scikit-learn's approach
        for (int i = 0; i < k; i++) {
            int neighbor_idx = knn_indices[idx * k + i];
            
            // Skip invalid neighbors
            if (neighbor_idx < 0) continue;
            
            // Get reachability distance
            float reach_dist = reach_dists[idx * k + i];
            
            // Only count valid reachability distances
            if (isfinite(reach_dist) && reach_dist > 0.0f) {
                sum_reach_dist += (double)reach_dist;
                valid_count++;
            }
        }
        
        // Calculate local reachability density
        if (valid_count > 0) {
            // scikit-learn: 1.0 / (sum(reach_dists) / n_neighbors)
            double avg_reach_dist = sum_reach_dist / valid_count;
            
            // Use exactly the same epsilon as scikit-learn (machine epsilon)
            lrd[idx] = (float)(1.0 / fmax(avg_reach_dist, DBL_EPSILON));
        } else {
            // No valid neighbors - scikit-learn defaults to 1.0
            lrd[idx] = 1.0f;
        }
    }
}

/**
 * CUDA kernel to compute final LOF scores
 * LOF of a point is the average ratio of the LRD of its neighbors to its own LRD
 */
__global__ void compute_lof_kernel(int* knn_indices, float* lrd, 
                                  int n_points, int k, float* lof_scores) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points) {
        double sum_lrd_ratio = 0.0;
        int valid_count = 0;
        
        // Get the LRD of the current point
        float lrd_idx = lrd[idx];
        
        // Sum the LRD ratios for all neighbors
        for (int i = 0; i < k; i++) {
            int neighbor_idx = knn_indices[idx * k + i];
            
            // Skip invalid neighbors
            if (neighbor_idx < 0) continue;
            
            // Get the LRD of the neighbor
            float lrd_neighbor = lrd[neighbor_idx];
            
            // scikit-learn uses neighbor_lrd / point_lrd
            // Use DBL_EPSILON to match scikit-learn's numeric stability
            double ratio = (double)lrd_neighbor / fmax((double)lrd_idx, DBL_EPSILON);
            
            // Add the ratio to the sum
            sum_lrd_ratio += ratio;
            valid_count++;
        }
        
        // Calculate the LOF score as the average LRD ratio
        if (valid_count > 0) {
            // scikit-learn divides by valid_count, not k
            double lof_score = sum_lrd_ratio / valid_count;
            
            // Handle potential numerical issues the same way scikit-learn does
            if (!isfinite(lof_score) || lof_score < 0.0) {
                lof_scores[idx] = 1.0f;  // Set to neutral score if invalid
            } else {
                lof_scores[idx] = (float)lof_score;
            }
        } else {
            // If no valid neighbors, set LOF to 1.0 (neutral score)
            lof_scores[idx] = 1.0f;
        }
    }
}

/**
 * Main function to compute LOF scores
 */
cudaError_t compute_lof(const float* points, int n_points, int n_dims, 
                       const LOFConfig* config, float* scores) {
    // Add input validation
    if (points == NULL || n_points <= 0 || n_dims <= 0 || config == NULL || scores == NULL) {
        return cudaErrorInvalidValue;
    }
    
    // Get k value from config
    int k = config->k;
    if (k <= 0 || k >= n_points || k > 32) {  // Match the KNN kernel constraint
        return cudaErrorInvalidValue;
    }
    
    // Initialize all variables first to satisfy compiler
    cudaError_t err = cudaSuccess;
    cudaStream_t stream = NULL;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t graphExec = NULL;
    int minGridSize = 0;
    int blockSize = 256; // Default in case occupancy calculator fails
    int gridSize = (n_points + blockSize - 1) / blockSize; // Default grid size
    float* d_dist_matrix = NULL;
    int* d_indices = NULL;
    float* d_distances = NULL;
    float* d_kdistances = NULL;
    float* d_reach_dists = NULL;
    float* d_lrd = NULL;
    float* d_points = const_cast<float*>(points);
    bool local_points_copy = false;
    bool use_pinned = false;
    bool use_graph = (n_points >= 5000);  // Only use graphs for larger datasets
    
    // Set device
    cudaSetDevice(0);  // Always use the same device

    // Set up CUDA streams for deterministic execution
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) goto cleanup;
    
    // Set the CUDA device to synchronize on errors (more stable)
    err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 0);
    if (err != cudaSuccess) goto cleanup;
    
    // Determine if we should use pinned memory based on data size
    use_pinned = (n_points * n_dims * sizeof(float) > 1048576);
    
    if (config->normalize) {
        // Allocate memory for a copy of the points
        err = cudaMalloc(&d_points, n_points * n_dims * sizeof(float));
        if (err != cudaSuccess) goto cleanup;
        
        // Copy the original points - use asynchronous copy for better performance
        err = cudaMemcpyAsync(d_points, points, n_points * n_dims * sizeof(float), 
                             cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        local_points_copy = true;
        
        // Normalize the points
        err = normalize_data(d_points, n_points, n_dims);
        if (err != cudaSuccess) goto cleanup;
    } else if (use_pinned) {
        // Register host memory as pinned for faster transfers
        err = cudaHostRegister((void*)points, n_points * n_dims * sizeof(float), 
                              cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            // If pinning fails, continue without it (not a critical error)
            err = cudaSuccess;
        }
    }
    
    // Allocate memory for distance matrix
    err = cudaMalloc(&d_dist_matrix, n_points * n_points * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate memory for nearest neighbors
    err = cudaMalloc(&d_indices, n_points * k * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_distances, n_points * k * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate memory for k-distances
    err = cudaMalloc(&d_kdistances, n_points * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate memory for reachability distances
    err = cudaMalloc(&d_reach_dists, n_points * k * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate memory for local reachability density
    err = cudaMalloc(&d_lrd, n_points * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Initialize all buffers
    err = cudaMemsetAsync(d_dist_matrix, 0, n_points * n_points * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_indices, 0xFF, n_points * k * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_distances, 0, n_points * k * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_kdistances, 0, n_points * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_reach_dists, 0, n_points * k * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_lrd, 0, n_points * sizeof(float), stream);
    if (err != cudaSuccess) goto cleanup;
    
    // Make sure initialization is complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto cleanup;
    
    // Compute pairwise distances
    err = compute_distances(d_points, n_points, n_dims, d_dist_matrix);
    if (err != cudaSuccess) goto cleanup;
    
    // Find k-nearest neighbors for each point
    err = find_knn(d_dist_matrix, n_points, k, d_indices, d_distances);
    if (err != cudaSuccess) goto cleanup;
    
    if (use_graph) {
        // Create a CUDA graph to capture the kernel execution pipeline
        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            // If graph capture fails, fall back to normal execution
            use_graph = false;
        } else {
            // Use occupancy calculator to determine optimal block size for each kernel
            err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                                 compute_kdistances_kernel, 0, 0);
            if (err != cudaSuccess) {
                cudaStreamEndCapture(stream, &graph);
                goto cleanup;
            }
            
            gridSize = (n_points + blockSize - 1) / blockSize;
            
            // Compute k-distances for each point
            compute_kdistances_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_indices, d_distances, n_points, k, d_kdistances);
            
            // Compute reachability distances
            compute_reach_dists_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_indices, d_distances, d_kdistances, n_points, k, d_reach_dists);
            
            // Compute local reachability density
            compute_lrd_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_indices, d_reach_dists, n_points, k, d_lrd);
            
            // Compute final LOF scores
            compute_lof_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_indices, d_lrd, n_points, k, scores);
            
            // End capture and create executable graph
            err = cudaStreamEndCapture(stream, &graph);
            if (err != cudaSuccess) {
                use_graph = false;
            } else {
                // Create an executable graph from the captured graph
                err = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
                if (err != cudaSuccess) {
                    use_graph = false;
                }
            }
        }
    }
    
    if (use_graph) {
        // Launch the entire pipeline using the graph
        err = cudaGraphLaunch(graphExec, stream);
        if (err != cudaSuccess) goto cleanup;
    } else {
        // Fall back to sequential kernel execution
        // Use occupancy calculator to determine optimal block size for each kernel
        err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                               compute_kdistances_kernel, 0, 0);
        if (err != cudaSuccess) goto cleanup;
        
        gridSize = (n_points + blockSize - 1) / blockSize;
        
        // Compute k-distances for each point
        compute_kdistances_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_indices, d_distances, n_points, k, d_kdistances);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // Compute reachability distances
        compute_reach_dists_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_indices, d_distances, d_kdistances, n_points, k, d_reach_dists);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // Compute local reachability density
        compute_lrd_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_indices, d_reach_dists, n_points, k, d_lrd);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // Compute final LOF scores
        compute_lof_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_indices, d_lrd, n_points, k, scores);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Single synchronization at the end
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Clean up CUDA graph resources
    if (graphExec) cudaGraphExecDestroy(graphExec);
    if (graph) cudaGraphDestroy(graph);
    
    // Clean up CUDA stream
    if (stream) cudaStreamDestroy(stream);
    
    // Free allocated device memory
    if (d_dist_matrix) cudaFree(d_dist_matrix);
    if (d_indices) cudaFree(d_indices);
    if (d_distances) cudaFree(d_distances);
    if (d_kdistances) cudaFree(d_kdistances);
    if (d_reach_dists) cudaFree(d_reach_dists);
    if (d_lrd) cudaFree(d_lrd);
    
    // Free local copy of points if created
    if (local_points_copy && d_points) cudaFree(d_points);
    
    // Unregister pinned memory if used
    if (!local_points_copy && use_pinned) {
        cudaHostUnregister((void*)points);
    }
    
    return err;
}

/**
 * Compute local reachability density for each point
 */
cudaError_t compute_lrd(const float* dist_matrix, const int* indices, 
                       const float* distances, int n_points, int k, float* lrd) {
    cudaError_t err;
    
    // Initialize all variables at the beginning
    int blockSize = 256;
    int gridSize = (n_points + blockSize - 1) / blockSize;
    float* d_kdistances = nullptr;
    float* d_reach_dists = nullptr;
    
    // Allocate memory for k-distances
    err = cudaMalloc(&d_kdistances, n_points * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate memory for reachability distances
    err = cudaMalloc(&d_reach_dists, n_points * k * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    
    // Compute k-distances for each point
    compute_kdistances_kernel<<<gridSize, blockSize>>>(
        const_cast<int*>(indices), const_cast<float*>(distances), n_points, k, d_kdistances);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;
    
    // Compute reachability distances
    compute_reach_dists_kernel<<<gridSize, blockSize>>>(
        const_cast<int*>(indices), const_cast<float*>(distances), d_kdistances, n_points, k, d_reach_dists);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;
    
    // Compute local reachability density
    compute_lrd_kernel<<<gridSize, blockSize>>>(
        const_cast<int*>(indices), d_reach_dists, n_points, k, lrd);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Free allocated device memory
    if (d_kdistances) cudaFree(d_kdistances);
    if (d_reach_dists) cudaFree(d_reach_dists);
    
    return err;
}

/**
 * Compute LOF scores using local reachability density
 */
cudaError_t compute_lof_scores(const float* dist_matrix, const int* indices, 
                              const float* lrd, int n_points, int k, float* scores) {
    cudaError_t err;
    
    // Determine block size for kernel execution
    int blockSize = 256;
    int gridSize = (n_points + blockSize - 1) / blockSize;
    
    // Compute final LOF scores
    compute_lof_kernel<<<gridSize, blockSize>>>(
        const_cast<int*>(indices), const_cast<float*>(lrd), n_points, k, scores);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    return cudaDeviceSynchronize();
} 