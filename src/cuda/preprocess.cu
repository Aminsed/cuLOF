/**
 * Preprocessing: centring / standardising, and row norms for the GEMM identity.
 *
 * Both kernels are O(n_points * n_dims), against O(n_points^2 * n_dims) for the
 * distance stage. At n = 200k, d = 8 that is 1.6M elements versus 4e10, so these
 * run in tens of microseconds out of a ~1.5 s call. They are written for clarity
 * rather than occupancy on purpose.
 */

#include <cmath>

#include "culof.h"
#include "culof_internal.cuh"

namespace culof {
namespace detail {

namespace {

constexpr int kBlock = 256;

/**
 * One block per feature: reduce the column, then rescale it in place.
 *
 * A block per feature leaves the GPU idle when n_dims is small, which for a
 * kernel on the critical path would be indefensible. Here it costs microseconds
 * and buys a deterministic tree reduction - no atomics, so the mean and variance
 * are bit-identical run to run, which is what keeps the whole pipeline
 * reproducible.
 */
__global__ void standardize_kernel(float* data, int n_points, int n_dims, bool scale) {
    const int f = blockIdx.x;
    __shared__ double partial[kBlock];

    double sum = 0.0;
    for (int i = threadIdx.x; i < n_points; i += kBlock) {
        sum += static_cast<double>(data[static_cast<size_t>(i) * n_dims + f]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int off = kBlock / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) partial[threadIdx.x] += partial[threadIdx.x + off];
        __syncthreads();
    }
    const double mean = partial[0] / n_points;
    __syncthreads();

    double inv_std = 1.0;
    if (scale) {
        double var = 0.0;
        for (int i = threadIdx.x; i < n_points; i += kBlock) {
            const double d =
                static_cast<double>(data[static_cast<size_t>(i) * n_dims + f]) - mean;
            var += d * d;
        }
        partial[threadIdx.x] = var;
        __syncthreads();
        for (int off = kBlock / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) partial[threadIdx.x] += partial[threadIdx.x + off];
            __syncthreads();
        }
        const double sd = sqrt(partial[0] / n_points);
        // A constant feature carries no information. Leave it centred at zero
        // rather than dividing by noise.
        inv_std = (sd > 1e-12) ? 1.0 / sd : 1.0;
    }

    for (int i = threadIdx.x; i < n_points; i += kBlock) {
        const size_t o = static_cast<size_t>(i) * n_dims + f;
        data[o] = static_cast<float>((static_cast<double>(data[o]) - mean) * inv_std);
    }
}

/** One thread per point. Each reads its own contiguous row of n_dims floats. */
__global__ void squared_norms_kernel(const float* __restrict__ data, int n_points,
                                     int n_dims, float* __restrict__ norms) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_points;
         i += gridDim.x * blockDim.x) {
        const float* row = data + static_cast<size_t>(i) * n_dims;
        double sum = 0.0;
        for (int j = 0; j < n_dims; ++j) sum += static_cast<double>(row[j]) * row[j];
        norms[i] = static_cast<float>(sum);
    }
}

}  // namespace

void standardize(float* data, int n_points, int n_dims, bool scale, cudaStream_t stream) {
    standardize_kernel<<<n_dims, kBlock, 0, stream>>>(data, n_points, n_dims, scale);
    CULOF_CHECK(cudaGetLastError());
}

void squared_norms(const float* data, int n_points, int n_dims, float* norms,
                   cudaStream_t stream) {
    const int grid = (n_points + kBlock - 1) / kBlock;
    squared_norms_kernel<<<grid, kBlock, 0, stream>>>(data, n_points, n_dims, norms);
    CULOF_CHECK(cudaGetLastError());
}

}  // namespace detail
}  // namespace culof
