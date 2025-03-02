#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_lof.h"
#include "cuda_lof.cuh"

// Constructor implementation
LOF::LOF(int k, bool normalize, float threshold, int min_points)
    : k_(k), normalize_(normalize), threshold_(threshold), 
      min_points_(min_points < 0 ? k : min_points) {
    
    // Initialize CUDA
    initialize_cuda_();
}

// Destructor implementation
LOF::~LOF() {
    // No specific cleanup needed
}

// Initialize CUDA device
void LOF::initialize_cuda_() {
    cudaError_t err = cudaFree(0);  // Simple call to initialize CUDA context
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to initialize CUDA: " + 
                             std::string(cudaGetErrorString(err)));
    }
}

// Allocate device memory and copy data
float* LOF::allocate_and_copy_(const float* host_points, int n_points, int n_dims) {
    float* device_points;
    size_t size = n_points * n_dims * sizeof(float);
    
    cudaError_t err = cudaMalloc(&device_points, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory: " + 
                             std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMemcpy(device_points, host_points, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_points);
        throw std::runtime_error("Failed to copy data to device: " + 
                             std::string(cudaGetErrorString(err)));
    }
    
    return device_points;
}

// Main LOF execution method
std::vector<float> LOF::execute_lof_(const float* points, int n_points, int n_dims) {
    // Validate input parameters
    if (points == nullptr || n_points <= 0 || n_dims <= 0) {
        throw std::invalid_argument("Invalid input data");
    }
    
    if (k_ <= 0 || k_ >= n_points) {
        throw std::invalid_argument("Invalid k value: must be between 1 and n_points-1");
    }
    
    // Allocate device memory for input points
    float* d_points = allocate_and_copy_(points, n_points, n_dims);
    
    // Allocate device memory for output scores
    float* d_scores;
    cudaError_t err = cudaMalloc(&d_scores, n_points * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_points);
        throw std::runtime_error("Failed to allocate device memory for scores: " + 
                             std::string(cudaGetErrorString(err)));
    }
    
    // Prepare configuration
    LOFConfig config;
    config.k = k_;
    config.normalize = normalize_;
    config.threshold = threshold_;
    config.min_points = min_points_;
    
    // Compute LOF scores
    err = compute_lof(d_points, n_points, n_dims, &config, d_scores);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scores);
        throw std::runtime_error("Failed to compute LOF: " + 
                             std::string(cudaGetErrorString(err)));
    }
    
    // Copy results back to host
    std::vector<float> host_scores(n_points);
    err = cudaMemcpy(host_scores.data(), d_scores, n_points * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_points);
        cudaFree(d_scores);
        throw std::runtime_error("Failed to copy results from device: " + 
                             std::string(cudaGetErrorString(err)));
    }
    
    // Free device memory
    cudaFree(d_points);
    cudaFree(d_scores);
    
    return host_scores;
}

// Compute LOF scores (raw pointer version)
std::vector<float> LOF::fit_predict(const float* points, int n_points, int n_dims) {
    return execute_lof_(points, n_points, n_dims);
}

// Compute LOF scores (vector version)
std::vector<float> LOF::fit_predict(const std::vector<std::vector<float>>& points) {
    if (points.empty()) {
        throw std::invalid_argument("Empty input data");
    }
    
    // Get dimensions
    int n_points = static_cast<int>(points.size());
    int n_dims = static_cast<int>(points[0].size());
    
    // Check that all points have the same dimensions
    for (const auto& point : points) {
        if (static_cast<int>(point.size()) != n_dims) {
            throw std::invalid_argument("Input points have inconsistent dimensions");
        }
    }
    
    // Flatten the data
    std::vector<float> flattened_points(n_points * n_dims);
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < n_dims; j++) {
            flattened_points[i * n_dims + j] = points[i][j];
        }
    }
    
    // Execute LOF
    return execute_lof_(flattened_points.data(), n_points, n_dims);
}

// Set number of nearest neighbors
void LOF::set_k(int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be greater than 0");
    }
    k_ = k;
    if (min_points_ < k) {
        min_points_ = k;
    }
}

// Set normalization flag
void LOF::set_normalize(bool normalize) {
    normalize_ = normalize;
}

// Set outlier threshold
void LOF::set_threshold(float threshold) {
    threshold_ = threshold;
}

// Identify outliers based on threshold
std::vector<int> LOF::get_outliers(const std::vector<float>& scores) const {
    std::vector<int> outliers;
    
    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] > threshold_) {
            outliers.push_back(static_cast<int>(i));
        }
    }
    
    return outliers;
} 