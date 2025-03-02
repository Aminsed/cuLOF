/**
 * CUDA-accelerated Local Outlier Factor (LOF) algorithm
 *
 * This header defines the interface for the CUDA-accelerated LOF implementation.
 * The LOF algorithm identifies outliers by comparing the local density of a
 * point with the local densities of its neighbors.
 */

#ifndef CUDA_LOF_CUH
#define CUDA_LOF_CUH

#include <cuda_runtime.h>

/**
 * Error handling macro for CUDA API calls
 */
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration for LOF computation
 */
typedef struct {
    int k;               // Number of nearest neighbors
    float threshold;     // Threshold for outlier detection (typically > 1.0)
    bool normalize;      // Whether to normalize the input data
    int min_points;      // Minimum points required (defaults to k)
} LOFConfig;

/**
 * Compute Local Outlier Factor scores for each point in the dataset
 *
 * @param points Device pointer to input data (row-major, n_points × n_dims)
 * @param n_points Number of data points
 * @param n_dims Number of dimensions per point
 * @param config Configuration parameters for LOF
 * @param scores Device pointer to output scores (n_points)
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t compute_lof(
    const float* points,
    int n_points,
    int n_dims,
    const LOFConfig* config,
    float* scores
);

/**
 * Compute pairwise distances between points
 *
 * @param points Device pointer to input data (row-major, n_points × n_dims)
 * @param n_points Number of data points
 * @param n_dims Number of dimensions per point
 * @param dist_matrix Device pointer to output distance matrix (n_points × n_points)
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t compute_distances(
    const float* points,
    int n_points,
    int n_dims,
    float* dist_matrix
);

/**
 * Find k-nearest neighbors for each point
 *
 * @param dist_matrix Device pointer to distance matrix (n_points × n_points)
 * @param n_points Number of data points
 * @param k Number of neighbors to find
 * @param indices Device pointer to indices of k-nearest neighbors (n_points × k)
 * @param distances Device pointer to distances of k-nearest neighbors (n_points × k)
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t find_knn(
    const float* dist_matrix,
    int n_points,
    int k,
    int* indices,
    float* distances
);

/**
 * Compute local reachability density for each point
 *
 * @param dist_matrix Device pointer to distance matrix (n_points × n_points)
 * @param indices Device pointer to indices of k-nearest neighbors (n_points × k)
 * @param distances Device pointer to distances of k-nearest neighbors (n_points × k)
 * @param n_points Number of data points
 * @param k Number of neighbors used
 * @param lrd Device pointer to output local reachability density (n_points)
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t compute_lrd(
    const float* dist_matrix,
    const int* indices,
    const float* distances,
    int n_points,
    int k,
    float* lrd
);

/**
 * Compute LOF scores using local reachability density
 *
 * @param dist_matrix Device pointer to distance matrix (n_points × n_points)
 * @param indices Device pointer to indices of k-nearest neighbors (n_points × k)
 * @param lrd Device pointer to local reachability density (n_points)
 * @param n_points Number of data points
 * @param k Number of neighbors used
 * @param scores Device pointer to output LOF scores (n_points)
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t compute_lof_scores(
    const float* dist_matrix,
    const int* indices,
    const float* lrd,
    int n_points,
    int k,
    float* scores
);

/**
 * Normalize input data (in-place)
 *
 * @param points Device pointer to input data to normalize (row-major, n_points × n_dims)
 * @param n_points Number of data points
 * @param n_dims Number of dimensions per point
 * @return cudaSuccess if successful, an error code otherwise
 */
cudaError_t normalize_data(
    float* points,
    int n_points,
    int n_dims
);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_LOF_CUH */ 