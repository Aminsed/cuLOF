#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "cuda_lof.cuh"

// Test fixture for KNN tests
class KNNTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
    
    // Allocate device memory
    template<typename T>
    T* allocate_device_memory(size_t size) {
        T* ptr;
        ASSERT_EQ(cudaMalloc(&ptr, size), cudaSuccess);
        
        // Cast to void* for storage
        void* void_ptr = static_cast<void*>(ptr);
        allocations.push_back(void_ptr);
        
        return ptr;
    }
    
    // Copy data to device
    template<typename T>
    void copy_to_device(T* d_ptr, const T* h_ptr, size_t size) {
        ASSERT_EQ(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice), cudaSuccess);
    }
    
    // Copy data from device
    template<typename T>
    void copy_from_device(T* h_ptr, const T* d_ptr, size_t size) {
        ASSERT_EQ(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    void TearDown() override {
        // Free allocated memory
        for (auto ptr : allocations) {
            cudaFree(ptr);
        }
        allocations.clear();
    }
    
    // Create a distance matrix from a set of points
    void create_distance_matrix(const std::vector<std::vector<float>>& points, std::vector<float>& dist_matrix) {
        int n_points = static_cast<int>(points.size());
        dist_matrix.resize(n_points * n_points);
        
        for (int i = 0; i < n_points; i++) {
            for (int j = 0; j < n_points; j++) {
                if (i == j) {
                    dist_matrix[i * n_points + j] = 0.0f;
                } else {
                    dist_matrix[i * n_points + j] = compute_euclidean_distance(points[i], points[j]);
                }
            }
        }
    }
    
    // Compute Euclidean distance between two vectors
    float compute_euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    // Find k nearest neighbors on CPU for reference
    void find_k_nearest_neighbors_cpu(const std::vector<float>& dist_matrix, int n_points, int k,
                                    std::vector<int>& indices, std::vector<float>& distances) {
        indices.resize(n_points * k);
        distances.resize(n_points * k);
        
        // For each point
        for (int i = 0; i < n_points; i++) {
            // Get distances from this point to all others
            std::vector<std::pair<float, int>> dist_idx_pairs;
            for (int j = 0; j < n_points; j++) {
                if (i != j) {  // Skip self
                    dist_idx_pairs.push_back(std::make_pair(dist_matrix[i * n_points + j], j));
                }
            }
            
            // Sort by distance
            std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());
            
            // Take k nearest
            for (int j = 0; j < k; j++) {
                if (j < dist_idx_pairs.size()) {
                    distances[i * k + j] = dist_idx_pairs[j].first;
                    indices[i * k + j] = dist_idx_pairs[j].second;
                } else {
                    distances[i * k + j] = FLT_MAX;
                    indices[i * k + j] = -1;
                }
            }
        }
    }
    
    std::vector<void*> allocations;
};

// Test with a simple 2D grid of points
TEST_F(KNNTest, Simple2DGrid) {
    // Create a 4x4 grid of points
    std::vector<std::vector<float>> points;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            points.push_back({static_cast<float>(i), static_cast<float>(j)});
        }
    }
    
    int n_points = static_cast<int>(points.size());
    
    // Create distance matrix
    std::vector<float> dist_matrix;
    create_distance_matrix(points, dist_matrix);
    
    // Allocate device memory
    float* d_dist_matrix = allocate_device_memory<float>(n_points * n_points * sizeof(float));
    copy_to_device(d_dist_matrix, dist_matrix.data(), n_points * n_points * sizeof(float));
    
    // Test different k values
    std::vector<int> k_values = {1, 3, 5, 8};
    
    for (int k : k_values) {
        // Allocate device memory for results
        int* d_indices = allocate_device_memory<int>(n_points * k * sizeof(int));
        float* d_distances = allocate_device_memory<float>(n_points * k * sizeof(float));
        
        // Find k-nearest neighbors
        ASSERT_EQ(find_knn(d_dist_matrix, n_points, k, d_indices, d_distances), cudaSuccess);
        
        // Copy results back to host
        std::vector<int> h_indices(n_points * k);
        std::vector<float> h_distances(n_points * k);
        copy_from_device(h_indices.data(), d_indices, n_points * k * sizeof(int));
        copy_from_device(h_distances.data(), d_distances, n_points * k * sizeof(float));
        
        // Find k nearest neighbors on CPU for comparison
        std::vector<int> cpu_indices;
        std::vector<float> cpu_distances;
        find_k_nearest_neighbors_cpu(dist_matrix, n_points, k, cpu_indices, cpu_distances);
        
        // For each point, check that the neighbors match
        for (int i = 0; i < n_points; i++) {
            // Check that distances are sorted
            for (int j = 1; j < k; j++) {
                ASSERT_LE(h_distances[i * k + j - 1], h_distances[i * k + j]);
            }
            
            // Check that every CPU neighbor is in the GPU results
            // (We may have ties, so the exact order can differ)
            for (int j = 0; j < k; j++) {
                int cpu_idx = cpu_indices[i * k + j];
                float cpu_dist = cpu_distances[i * k + j];
                
                if (cpu_idx == -1) continue;  // Skip invalid neighbors
                
                // Find this neighbor in the GPU results
                bool found = false;
                for (int l = 0; l < k; l++) {
                    if (h_indices[i * k + l] == cpu_idx) {
                        found = true;
                        ASSERT_FLOAT_EQ(h_distances[i * k + l], cpu_dist);
                        break;
                    }
                }
                
                ASSERT_TRUE(found) << "Neighbor " << cpu_idx << " of point " << i << " not found in GPU results";
            }
        }
    }
}

// Test with random points
TEST_F(KNNTest, RandomPoints) {
    // Create random points
    int n_points = 50;
    int n_dims = 3;
    
    std::vector<std::vector<float>> points;
    for (int i = 0; i < n_points; i++) {
        std::vector<float> point;
        for (int j = 0; j < n_dims; j++) {
            point.push_back(static_cast<float>(rand()) / RAND_MAX);
        }
        points.push_back(point);
    }
    
    // Create distance matrix
    std::vector<float> dist_matrix;
    create_distance_matrix(points, dist_matrix);
    
    // Allocate device memory
    float* d_dist_matrix = allocate_device_memory<float>(n_points * n_points * sizeof(float));
    copy_to_device(d_dist_matrix, dist_matrix.data(), n_points * n_points * sizeof(float));
    
    // Test with k = 10
    int k = 10;
    
    // Allocate device memory for results
    int* d_indices = allocate_device_memory<int>(n_points * k * sizeof(int));
    float* d_distances = allocate_device_memory<float>(n_points * k * sizeof(float));
    
    // Find k-nearest neighbors
    ASSERT_EQ(find_knn(d_dist_matrix, n_points, k, d_indices, d_distances), cudaSuccess);
    
    // Copy results back to host
    std::vector<int> h_indices(n_points * k);
    std::vector<float> h_distances(n_points * k);
    copy_from_device(h_indices.data(), d_indices, n_points * k * sizeof(int));
    copy_from_device(h_distances.data(), d_distances, n_points * k * sizeof(float));
    
    // Find k nearest neighbors on CPU for comparison
    std::vector<int> cpu_indices;
    std::vector<float> cpu_distances;
    find_k_nearest_neighbors_cpu(dist_matrix, n_points, k, cpu_indices, cpu_distances);
    
    // Verify a subset of results
    for (int i = 0; i < 10; i++) {
        int point_idx = rand() % n_points;
        
        // Check that the closest neighbor is the same
        ASSERT_EQ(h_indices[point_idx * k], cpu_indices[point_idx * k]);
        ASSERT_FLOAT_EQ(h_distances[point_idx * k], cpu_distances[point_idx * k]);
        
        // Check that distances are sorted
        for (int j = 1; j < k; j++) {
            ASSERT_LE(h_distances[point_idx * k + j - 1], h_distances[point_idx * k + j]);
        }
    }
}

// Test edge cases
TEST_F(KNNTest, EdgeCases) {
    // Create a small set of points
    std::vector<std::vector<float>> points = {
        {0.0f, 0.0f},  // Point 0
        {1.0f, 0.0f},  // Point 1
        {0.0f, 1.0f},  // Point 2
        {1.0f, 1.0f}   // Point 3
    };
    
    int n_points = static_cast<int>(points.size());
    
    // Create distance matrix
    std::vector<float> dist_matrix;
    create_distance_matrix(points, dist_matrix);
    
    // Allocate device memory
    float* d_dist_matrix = allocate_device_memory<float>(n_points * n_points * sizeof(float));
    copy_to_device(d_dist_matrix, dist_matrix.data(), n_points * n_points * sizeof(float));
    
    // Test with k = 1
    {
        int k = 1;
        int* d_indices = allocate_device_memory<int>(n_points * k * sizeof(int));
        float* d_distances = allocate_device_memory<float>(n_points * k * sizeof(float));
        
        ASSERT_EQ(find_knn(d_dist_matrix, n_points, k, d_indices, d_distances), cudaSuccess);
        
        std::vector<int> h_indices(n_points * k);
        std::vector<float> h_distances(n_points * k);
        copy_from_device(h_indices.data(), d_indices, n_points * k * sizeof(int));
        copy_from_device(h_distances.data(), d_distances, n_points * k * sizeof(float));
        
        // Point 0's nearest neighbor should be 1 or 2 (both at distance 1.0)
        int nearest_to_0 = h_indices[0 * k + 0];
        ASSERT_TRUE(nearest_to_0 == 1 || nearest_to_0 == 2);
        ASSERT_FLOAT_EQ(h_distances[0 * k + 0], 1.0f);
        
        // Point 3's nearest neighbor should be 1 or 2 (both at distance 1.0)
        int nearest_to_3 = h_indices[3 * k + 0];
        ASSERT_TRUE(nearest_to_3 == 1 || nearest_to_3 == 2);
        ASSERT_FLOAT_EQ(h_distances[3 * k + 0], 1.0f);
    }
    
    // Test with k = n_points - 1 (maximum valid k)
    {
        int k = n_points - 1;
        int* d_indices = allocate_device_memory<int>(n_points * k * sizeof(int));
        float* d_distances = allocate_device_memory<float>(n_points * k * sizeof(float));
        
        ASSERT_EQ(find_knn(d_dist_matrix, n_points, k, d_indices, d_distances), cudaSuccess);
        
        std::vector<int> h_indices(n_points * k);
        std::vector<float> h_distances(n_points * k);
        copy_from_device(h_indices.data(), d_indices, n_points * k * sizeof(int));
        copy_from_device(h_distances.data(), d_distances, n_points * k * sizeof(float));
        
        // Each point should have all other points as neighbors
        for (int i = 0; i < n_points; i++) {
            std::vector<int> neighbors;
            for (int j = 0; j < k; j++) {
                neighbors.push_back(h_indices[i * k + j]);
            }
            
            std::sort(neighbors.begin(), neighbors.end());
            
            int expected_idx = 0;
            for (int j = 0; j < n_points; j++) {
                if (j != i) {
                    ASSERT_EQ(neighbors[expected_idx], j);
                    expected_idx++;
                }
            }
        }
    }
    
    // Test with invalid k (should return error)
    {
        int k = n_points;  // k should be < n_points
        int* d_indices = allocate_device_memory<int>(n_points * k * sizeof(int));
        float* d_distances = allocate_device_memory<float>(n_points * k * sizeof(float));
        
        ASSERT_EQ(find_knn(d_dist_matrix, n_points, k, d_indices, d_distances), cudaErrorInvalidValue);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 