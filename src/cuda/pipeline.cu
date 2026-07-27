/**
 * The public entry point, and the tiling that keeps memory linear in n.
 */

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "culof.h"
#include "culof_internal.cuh"

namespace culof {
namespace detail {

void check_cuda(cudaError_t err, const char* file, int line, const char* expr) {
    if (err == cudaSuccess) return;
    std::ostringstream os;
    os << "CUDA error at " << file << ":" << line << " (" << expr
       << "): " << cudaGetErrorString(err);
    throw CudaError(os.str());
}

void check_cublas(cublasStatus_t status, const char* file, int line, const char* expr) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    std::ostringstream os;
    os << "cuBLAS error at " << file << ":" << line << " (" << expr
       << "): " << cublasGetStatusString(status);
    throw CudaError(os.str());
}

int choose_tile_rows(int n_points, size_t budget_bytes) {
    const size_t row_bytes = static_cast<size_t>(n_points) * sizeof(float);
    size_t rows = budget_bytes / std::max<size_t>(row_bytes, 1);
    rows = std::clamp<size_t>(rows, 1, static_cast<size_t>(n_points));
    return static_cast<int>(rows);
}

namespace {

/** Owning device allocation. */
class Buffer {
public:
    explicit Buffer(size_t bytes) { CULOF_CHECK(cudaMalloc(&ptr_, bytes)); }
    ~Buffer() {
        if (ptr_) cudaFree(ptr_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <typename T>
    T* as() const {
        return static_cast<T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
};

/** Owning stream. */
class Stream {
public:
    Stream() { CULOF_CHECK(cudaStreamCreate(&s_)); }
    ~Stream() {
        if (s_) cudaStreamDestroy(s_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return s_; }

private:
    cudaStream_t s_ = nullptr;
};

/** cuBLAS handles are costly to create and not shareable across threads. */
cublasHandle_t cublas_handle() {
    struct Holder {
        cublasHandle_t h = nullptr;
        Holder() { CULOF_CHECK_CUBLAS(cublasCreate(&h)); }
        ~Holder() {
            if (h) cublasDestroy(h);
        }
    };
    static thread_local Holder holder;
    return holder.h;
}

/**
 * Rows per distance tile.
 *
 * The full n x n distance matrix is never materialised: at n = 200,000 it would
 * be 149 GiB. Instead rows are processed in tiles sized to a slice of free
 * device memory, so peak usage grows linearly in n.
 *
 * CULOF_TILE_ROWS overrides the choice. The tests use it to prove tiling does
 * not change the result; users can use it to share a GPU.
 */
int tile_rows_for(int n_points) {
    if (const char* env = std::getenv("CULOF_TILE_ROWS")) {
        const int forced = std::atoi(env);
        if (forced > 0) return std::min(forced, n_points);
    }
    size_t free_bytes = 0, total_bytes = 0;
    CULOF_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    constexpr size_t kCap = size_t(2) << 30;      // 2 GiB
    constexpr size_t kMargin = size_t(128) << 20; // leave headroom
    const size_t usable = free_bytes > kMargin ? free_bytes - kMargin : 0;
    return choose_tile_rows(n_points, std::min(kCap, usable));
}

}  // namespace
}  // namespace detail

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_info() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return "no CUDA device";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "unknown device";
    std::ostringstream os;
    os << prop.name << " (sm_" << prop.major << prop.minor << ", "
       << prop.multiProcessorCount << " SMs, " << (prop.totalGlobalMem >> 20) << " MiB)";
    return os.str();
}

std::vector<float> lof(const float* points, int n_points, int n_dims, int k,
                       bool normalize) {
    if (points == nullptr) throw std::invalid_argument("points must not be null");
    if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
    if (n_dims < 1) throw std::invalid_argument("n_dims must be >= 1");
    if (k < 1 || k >= n_points) {
        std::ostringstream os;
        os << "k must satisfy 1 <= k <= n_points - 1 (got k=" << k << ", n_points="
           << n_points << ")";
        throw std::invalid_argument(os.str());
    }
    if (!cuda_available()) throw CudaError("no CUDA device available");

    using detail::Buffer;
    const size_t n = static_cast<size_t>(n_points);
    const size_t nk = n * static_cast<size_t>(k);

    detail::Stream stream;
    const cudaStream_t s = stream.get();

    Buffer d_points(n * n_dims * sizeof(float));
    Buffer d_norms(n * sizeof(float));
    Buffer d_nbr_idx(nk * sizeof(int));
    Buffer d_nbr_d2(nk * sizeof(float));
    Buffer d_kdist2(n * sizeof(float));
    Buffer d_lrd(n * sizeof(float));
    Buffer d_scores(n * sizeof(float));

    CULOF_CHECK(cudaMemcpyAsync(d_points.as<float>(), points, n * n_dims * sizeof(float),
                                cudaMemcpyHostToDevice, s));

    // Centre always; scale only when asked. See culof_internal.cuh for why
    // centring is unconditional.
    detail::standardize(d_points.as<float>(), n_points, n_dims, normalize, s);
    detail::squared_norms(d_points.as<float>(), n_points, n_dims, d_norms.as<float>(), s);

    const int tile = detail::tile_rows_for(n_points);
    Buffer d_tile(static_cast<size_t>(tile) * n * sizeof(float));

    cublasHandle_t handle = detail::cublas_handle();
    CULOF_CHECK_CUBLAS(cublasSetStream(handle, s));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int row0 = 0; row0 < n_points; row0 += tile) {
        const int rows = std::min(tile, n_points - row0);

        // A row-major (rows x n_points) result is a column-major
        // (n_points x rows) one. Read column-major, the row-major (n x d) point
        // array is already (d x n), so op(A) = A^T supplies the (n x d) left
        // operand and no transpose or copy is needed anywhere.
        CULOF_CHECK_CUBLAS(cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            /*m=*/n_points, /*n=*/rows, /*k=*/n_dims, &alpha,
            /*A=*/d_points.as<float>(), /*lda=*/n_dims,
            /*B=*/d_points.as<float>() + static_cast<size_t>(row0) * n_dims,
            /*ldb=*/n_dims, &beta,
            /*C=*/d_tile.as<float>(), /*ldc=*/n_points));

        // select_knn converts the Gram tile to squared distances in place during
        // its first radix pass, so no separate elementwise kernel is needed.
        detail::select_knn(d_tile.as<float>(), d_norms.as<float>(), rows, row0, n_points,
                           k, d_nbr_idx.as<int>(), d_nbr_d2.as<float>(),
                           d_kdist2.as<float>(), s);
    }

    detail::compute_lrd(d_nbr_idx.as<int>(), d_nbr_d2.as<float>(), d_kdist2.as<float>(),
                        n_points, k, d_lrd.as<float>(), s);
    detail::compute_scores(d_nbr_idx.as<int>(), d_lrd.as<float>(), n_points, k,
                           d_scores.as<float>(), s);

    std::vector<float> out(n);
    CULOF_CHECK(cudaMemcpyAsync(out.data(), d_scores.as<float>(), n * sizeof(float),
                                cudaMemcpyDeviceToHost, s));
    CULOF_CHECK(cudaStreamSynchronize(s));
    return out;
}

}  // namespace culof
