/**
 * CUDA-accelerated Local Outlier Factor (LOF) algorithm
 *
 * This header defines the C++ interface for the CUDA-accelerated LOF implementation.
 * It provides a clean C++ API for the CUDA functionality that can be used by the
 * Python bindings.
 */

#ifndef CUDA_LOF_H
#define CUDA_LOF_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

/**
 * C++ wrapper class for CUDA LOF implementation
 */
class LOF {
public:
    /**
     * Constructor
     *
     * @param k Number of nearest neighbors to consider
     * @param normalize Whether to normalize input data
     * @param threshold Threshold for outlier detection (typically > 1.0)
     * @param min_points Minimum points required (defaults to k)
     */
    LOF(int k = 20, bool normalize = true, float threshold = 1.5f, int min_points = -1);
    
    /**
     * Destructor
     */
    ~LOF();
    
    /**
     * Compute LOF scores for each point in the dataset
     *
     * @param points Input data points (row-major, n_points × n_dims)
     * @param n_points Number of data points
     * @param n_dims Number of dimensions per point
     * @return Vector of LOF scores for each point
     */
    std::vector<float> fit_predict(const float* points, int n_points, int n_dims);
    
    /**
     * Compute LOF scores (StdVector overload)
     *
     * @param points Input data points as vector of vectors
     * @return Vector of LOF scores for each point
     */
    std::vector<float> fit_predict(const std::vector<std::vector<float>>& points);
    
    /**
     * Set number of nearest neighbors
     *
     * @param k Number of nearest neighbors
     */
    void set_k(int k);
    
    /**
     * Set normalization flag
     *
     * @param normalize Whether to normalize data
     */
    void set_normalize(bool normalize);
    
    /**
     * Set outlier threshold
     *
     * @param threshold Threshold for outlier detection
     */
    void set_threshold(float threshold);
    
    /**
     * Identify outliers based on threshold
     *
     * @param scores LOF scores
     * @return Vector of indices of outliers
     */
    std::vector<int> get_outliers(const std::vector<float>& scores) const;
    
private:
    int k_;
    bool normalize_;
    float threshold_;
    int min_points_;
    
    /**
     * Execute LOF algorithm using CUDA
     */
    std::vector<float> execute_lof_(const float* points, int n_points, int n_dims);
    
    /**
     * Allocate device memory and copy data
     */
    float* allocate_and_copy_(const float* host_points, int n_points, int n_dims);
    
    /**
     * Initialize CUDA device
     */
    void initialize_cuda_();
};

#endif /* CUDA_LOF_H */ 