#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_lof.cuh"

// Test fixture for distance computation tests
class DistanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
    
    // Allocate device memory
    float* allocate_device_memory(size_t size) {
        float* ptr;
        ASSERT_EQ(cudaMalloc(&ptr, size), cudaSuccess);
        allocations.push_back(ptr);
        return ptr;
    }
    
    // Copy data to device
    void copy_to_device(float* d_ptr, const float* h_ptr, size_t size) {
        ASSERT_EQ(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice), cudaSuccess);
    }
    
    // Copy data from device
    void copy_from_device(float* h_ptr, const float* d_ptr, size_t size) {
        ASSERT_EQ(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    void TearDown() override {
        // Free allocated memory
        for (auto ptr : allocations) {
            cudaFree(ptr);
        }
        allocations.clear();
    }
    
    // Compute Euclidean distance on CPU for reference
    float compute_euclidean_distance(const float* a, const float* b, int dims) {
        float sum = 0.0f;
        for (int i = 0; i < dims; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    std::vector<float*> allocations;
};

// Test with simple 2D points
TEST_F(DistanceTest, Simple2DPoints) {
    // Define test points
    std::vector<float> points = {
        0.0f, 0.0f,  // Point 0: origin
        1.0f, 0.0f,  // Point 1: (1,0)
        0.0f, 1.0f,  // Point 2: (0,1)
        1.0f, 1.0f   // Point 3: (1,1)
    };
    
    int n_points = 4;
    int n_dims = 2;
    
    // Allocate device memory
    float* d_points = allocate_device_memory(n_points * n_dims * sizeof(float));
    float* d_dist_matrix = allocate_device_memory(n_points * n_points * sizeof(float));
    
    // Copy points to device
    copy_to_device(d_points, points.data(), n_points * n_dims * sizeof(float));
    
    // Compute distances
    ASSERT_EQ(compute_distances(d_points, n_points, n_dims, d_dist_matrix), cudaSuccess);
    
    // Copy distance matrix back to host
    std::vector<float> h_dist_matrix(n_points * n_points);
    copy_from_device(h_dist_matrix.data(), d_dist_matrix, n_points * n_points * sizeof(float));
    
    // Expected distances:
    // Point 0 to 0: 0.0
    // Point 0 to 1: 1.0
    // Point 0 to 2: 1.0
    // Point 0 to 3: sqrt(2) ≈ 1.414
    // Point 1 to 1: 0.0
    // Point 1 to 2: sqrt(2) ≈ 1.414
    // Point 1 to 3: 1.0
    // Point 2 to 2: 0.0
    // Point 2 to 3: 1.0
    // Point 3 to 3: 0.0
    
    // Check distances against expected values
    ASSERT_FLOAT_EQ(h_dist_matrix[0 * n_points + 0], 0.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[0 * n_points + 1], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[0 * n_points + 2], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[0 * n_points + 3], std::sqrt(2.0f));
    
    ASSERT_FLOAT_EQ(h_dist_matrix[1 * n_points + 0], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[1 * n_points + 1], 0.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[1 * n_points + 2], std::sqrt(2.0f));
    ASSERT_FLOAT_EQ(h_dist_matrix[1 * n_points + 3], 1.0f);
    
    ASSERT_FLOAT_EQ(h_dist_matrix[2 * n_points + 0], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[2 * n_points + 1], std::sqrt(2.0f));
    ASSERT_FLOAT_EQ(h_dist_matrix[2 * n_points + 2], 0.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[2 * n_points + 3], 1.0f);
    
    ASSERT_FLOAT_EQ(h_dist_matrix[3 * n_points + 0], std::sqrt(2.0f));
    ASSERT_FLOAT_EQ(h_dist_matrix[3 * n_points + 1], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[3 * n_points + 2], 1.0f);
    ASSERT_FLOAT_EQ(h_dist_matrix[3 * n_points + 3], 0.0f);
    
    // Check matrix symmetry
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < n_points; j++) {
            ASSERT_FLOAT_EQ(h_dist_matrix[i * n_points + j], h_dist_matrix[j * n_points + i]);
        }
    }
}

// Test with higher-dimensional points
TEST_F(DistanceTest, HigherDimensionalPoints) {
    // Create test data with different dimensionality
    std::vector<int> test_dims = {3, 5, 10, 32};
    
    for (int n_dims : test_dims) {
        // Create random points
        std::vector<float> points(10 * n_dims);
        for (size_t i = 0; i < points.size(); i++) {
            points[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        int n_points = 10;
        
        // Allocate device memory
        float* d_points = allocate_device_memory(n_points * n_dims * sizeof(float));
        float* d_dist_matrix = allocate_device_memory(n_points * n_points * sizeof(float));
        
        // Copy points to device
        copy_to_device(d_points, points.data(), n_points * n_dims * sizeof(float));
        
        // Compute distances
        ASSERT_EQ(compute_distances(d_points, n_points, n_dims, d_dist_matrix), cudaSuccess);
        
        // Copy distance matrix back to host
        std::vector<float> h_dist_matrix(n_points * n_points);
        copy_from_device(h_dist_matrix.data(), d_dist_matrix, n_points * n_points * sizeof(float));
        
        // Compute distances on CPU and compare
        for (int i = 0; i < n_points; i++) {
            for (int j = i; j < n_points; j++) {
                float expected = compute_euclidean_distance(
                    &points[i * n_dims], &points[j * n_dims], n_dims);
                float actual = h_dist_matrix[i * n_points + j];
                
                // Use a small epsilon for floating-point comparison
                ASSERT_NEAR(actual, expected, 1e-5f);
                
                // Check symmetry
                ASSERT_FLOAT_EQ(h_dist_matrix[i * n_points + j], h_dist_matrix[j * n_points + i]);
            }
        }
    }
}

// Test with large number of points
TEST_F(DistanceTest, LargeNumberOfPoints) {
    int n_points = 100;  // Can be increased for stress testing
    int n_dims = 3;
    
    // Create random points
    std::vector<float> points(n_points * n_dims);
    for (size_t i = 0; i < points.size(); i++) {
        points[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float* d_points = allocate_device_memory(n_points * n_dims * sizeof(float));
    float* d_dist_matrix = allocate_device_memory(n_points * n_points * sizeof(float));
    
    // Copy points to device
    copy_to_device(d_points, points.data(), n_points * n_dims * sizeof(float));
    
    // Compute distances
    ASSERT_EQ(compute_distances(d_points, n_points, n_dims, d_dist_matrix), cudaSuccess);
    
    // Copy distance matrix back to host
    std::vector<float> h_dist_matrix(n_points * n_points);
    copy_from_device(h_dist_matrix.data(), d_dist_matrix, n_points * n_points * sizeof(float));
    
    // Check a subset of distances
    for (int i = 0; i < 10; i++) {
        int idx1 = rand() % n_points;
        int idx2 = rand() % n_points;
        
        float expected = compute_euclidean_distance(
            &points[idx1 * n_dims], &points[idx2 * n_dims], n_dims);
        float actual = h_dist_matrix[idx1 * n_points + idx2];
        
        ASSERT_NEAR(actual, expected, 1e-5f);
    }
    
    // Check diagonal is all zeros
    for (int i = 0; i < n_points; i++) {
        ASSERT_FLOAT_EQ(h_dist_matrix[i * n_points + i], 0.0f);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 