/**
 * cuLOF internals. Device-side stage interfaces, shared between translation
 * units and exposed to the tests. Not part of the public API.
 */

#ifndef CULOF_INTERNAL_CUH
#define CULOF_INTERNAL_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace culof {
namespace detail {

void check_cuda(cudaError_t err, const char* file, int line, const char* expr);
void check_cublas(cublasStatus_t status, const char* file, int line, const char* expr);

#define CULOF_CHECK(expr) ::culof::detail::check_cuda((expr), __FILE__, __LINE__, #expr)
#define CULOF_CHECK_CUBLAS(expr) \
    ::culof::detail::check_cublas((expr), __FILE__, __LINE__, #expr)

/**
 * Subtract each feature's mean in place, and divide by its standard deviation
 * when `scale`.
 *
 * Centring is unconditional. It leaves every pairwise distance unchanged, but
 * the GEMM identity |a-b|^2 = |a|^2 + |b|^2 - 2a.b loses significant digits to
 * cancellation when |a|^2 dwarfs |a-b|^2, and centring makes |a|^2 as small as
 * it can be.
 */
void standardize(float* data, int n_points, int n_dims, bool scale, cudaStream_t stream);

/** Row-wise squared L2 norms of a row-major (n_points x n_dims) matrix. */
void squared_norms(const float* data, int n_points, int n_dims, float* norms,
                   cudaStream_t stream);

/**
 * Exact k-nearest-neighbour selection over one tile of the distance matrix.
 *
 * On entry `tile[r * n_points + c]` holds the dot product x_(row_offset+r).x_c
 * from the GEMM. The first radix pass converts it in place to
 * max(0, |x_r|^2 + |x_c|^2 - 2 x_r.x_c) and sets the diagonal to +inf, so the
 * conversion costs no extra sweep and a point can never select itself.
 *
 * Emits the k neighbours (unordered) and the k-th smallest squared distance.
 * LOF needs only the neighbour set and the k-distance, never a sorted list.
 *
 * Neighbour arrays are transposed - entry t of point i is at
 * `nbr[t * n_points + i]` - which keeps the per-point kernels downstream
 * coalesced.
 */
void select_knn(float* tile, const float* norms, int rows, int row_offset, int n_points,
                int k, int* nbr_idx, float* nbr_d2, float* kdist2, cudaStream_t stream);

/** lrd[i] = 1 / (mean_j max(k-dist(j), d(i,j)) + 1e-10), as scikit-learn does. */
void compute_lrd(const int* nbr_idx, const float* nbr_d2, const float* kdist2,
                 int n_points, int k, float* lrd, cudaStream_t stream);

/** scores[i] = mean_j (lrd[j] / lrd[i]) over i's k neighbours. */
void compute_scores(const int* nbr_idx, const float* lrd, int n_points, int k,
                    float* scores, cudaStream_t stream);

/** Rows per distance tile that keep the scratch buffer within `budget_bytes`. */
int choose_tile_rows(int n_points, size_t budget_bytes);

}  // namespace detail
}  // namespace culof

#endif  // CULOF_INTERNAL_CUH
