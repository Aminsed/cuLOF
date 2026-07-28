/**
 * cuLOF - GPU-accelerated Local Outlier Factor.
 *
 * Scores match scikit-learn's
 * `-LocalOutlierFactor(...).negative_outlier_factor_`: about 1.0 for a point as
 * densely surrounded as its neighbours, larger for one in a sparser region.
 *
 * The stateful, scikit-learn-shaped API lives in the Python layer. This header
 * is deliberately the whole C++ surface.
 */

#ifndef CULOF_H
#define CULOF_H

#include <stdexcept>
#include <string>
#include <vector>

namespace culof {

/** Any CUDA or cuBLAS failure. */
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

/** True if a usable CUDA device is present. */
bool cuda_available();

/** Name, compute capability, SM count and memory of the active device. */
std::string device_info();

/**
 * Compute Local Outlier Factor scores.
 *
 * @param points    host pointer to row-major (n_points x n_dims) float32 data
 * @param n_points  number of samples, >= 2
 * @param n_dims    number of features, >= 1
 * @param k         neighbours per point, 1 <= k <= n_points - 1. Selection is
 *                  independent of k; storage and the density/score passes are
 *                  O(n_points * k), so large k does cost time and memory.
 * @param normalize z-score each feature before computing distances
 *
 * @throws std::invalid_argument for bad arguments, CudaError for device failure
 */
std::vector<float> lof(const float* points, int n_points, int n_dims, int k = 20,
                       bool normalize = false);

}  // namespace culof

#endif  // CULOF_H
