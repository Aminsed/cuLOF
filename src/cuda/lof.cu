/**
 * Local reachability density and the final LOF scores: the last two stages.
 *
 * The formulation follows Breunig et al. (2000) and matches scikit-learn's
 * LocalOutlierFactor term for term, including the 1e-10 guard scikit-learn
 * adds before inverting the mean reachability distance.
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "culof.h"
#include "culof_internal.cuh"

namespace culof {
namespace detail {

namespace {

constexpr int kBlock = 256;

/** Matches scikit-learn's `1.0 / (mean(reach_dist) + 1e-10)`. */
constexpr double kLrdEpsilon = 1e-10;

/**
 * lrd[i] = 1 / (mean_j max(k-dist(j), d(i,j)) + eps)
 *
 * Reachability is computed on squared distances and square-rooted once:
 * sqrt is monotonic, so max(sqrt(a), sqrt(b)) == sqrt(max(a, b)).
 */
__global__ void lrd_kernel(const int* __restrict__ nbr_idx,
                           const float* __restrict__ nbr_d2,
                           const float* __restrict__ kdist2, int n_points, int k,
                           float* __restrict__ lrd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    double sum = 0.0;
    for (int t = 0; t < k; ++t) {
        const size_t o = static_cast<size_t>(t) * n_points + i;
        const int j = nbr_idx[o];
        sum += sqrt(static_cast<double>(fmaxf(kdist2[j], nbr_d2[o])));
    }
    lrd[i] = static_cast<float>(1.0 / (sum / k + kLrdEpsilon));
}

/** scores[i] = mean_j (lrd[j] / lrd[i]) over i's k neighbours. */
__global__ void score_kernel(const int* __restrict__ nbr_idx,
                             const float* __restrict__ lrd, int n_points, int k,
                             float* __restrict__ scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    const double self_lrd = lrd[i];
    double sum = 0.0;
    for (int t = 0; t < k; ++t) {
        sum += static_cast<double>(lrd[nbr_idx[static_cast<size_t>(t) * n_points + i]]) /
               self_lrd;
    }
    scores[i] = static_cast<float>(sum / k);
}

}  // namespace

void compute_lrd(const int* nbr_idx, const float* nbr_d2, const float* kdist2,
                 int n_points, int k, float* lrd, cudaStream_t stream) {
    const int grid = (n_points + kBlock - 1) / kBlock;
    lrd_kernel<<<grid, kBlock, 0, stream>>>(nbr_idx, nbr_d2, kdist2, n_points, k, lrd);
    CULOF_CHECK(cudaGetLastError());
}

void compute_scores(const int* nbr_idx, const float* lrd, int n_points, int k,
                    float* scores, cudaStream_t stream) {
    const int grid = (n_points + kBlock - 1) / kBlock;
    score_kernel<<<grid, kBlock, 0, stream>>>(nbr_idx, lrd, n_points, k, scores);
    CULOF_CHECK(cudaGetLastError());
}

}  // namespace detail
}  // namespace culof
