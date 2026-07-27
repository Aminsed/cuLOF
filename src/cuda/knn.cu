/**
 * Exact k-nearest-neighbour selection: the heart of the library.
 *
 * Why selection and not sorting
 * -----------------------------
 * LOF consumes exactly two things per point: the *set* of its k nearest
 * neighbours, and its k-distance. Every downstream quantity is a mean over that
 * set, so their order is irrelevant. Sorting would be strictly more work than
 * the problem requires.
 *
 * Why radix select
 * ----------------
 * For non-negative IEEE-754 floats, the unsigned integer ordering of the bit
 * pattern is identical to the numeric ordering. Squared distances are
 * non-negative by construction, so an integer radix select on
 * `__float_as_uint(d)` is exact - no tolerance, no comparator, no sort.
 *
 * Three passes of 11, 11 and 10 bits pin the k-th smallest key down exactly,
 * and that key *is* the k-distance, so nothing extra is needed to obtain it. A
 * fourth sweep gathers the neighbours.
 *
 * Against the per-thread insertion sort this replaces:
 *
 *                    insertion sort          radix select
 *   max k            32 (register array)     unbounded
 *   work per row     O(n*k)                  O(n) per pass
 *   memory access    stride-n across lanes   fully coalesced
 *
 * The kernel is memory-bound, so sweeps of the tile are the currency. Two
 * choices spend it well: the Gram-to-distance conversion rides along inside the
 * first pass rather than needing its own kernel, and 11-bit digits cover 32 bits
 * in three passes instead of the four an 8-bit digit needs. Four sweeps total,
 * where the obvious arrangement takes six.
 *
 * Determinism
 * -----------
 * The gather assigns output slots with a block-wide prefix sum, not atomics.
 * Atomics would order the neighbour list differently on every launch, and since
 * float addition is not associative that alone would jitter the low bits of
 * every score. With the scan, identical input gives bit-identical output.
 */

#include <cub/block/block_scan.cuh>
#include <math_constants.h>

#include "culof.h"
#include "culof_internal.cuh"

namespace culof {
namespace detail {

namespace {

constexpr int kBlock = 256;
constexpr int kPasses = 3;  // 11 + 11 + 10 bits
constexpr int kMaxBins = 1 << 11;

__device__ __forceinline__ int pass_shift(int p) {
    return (p == 0) ? 21 : (p == 1) ? 10 : 0;
}
__device__ __forceinline__ int pass_width(int p) { return (p == 2) ? 10 : 11; }

using BlockScan = cub::BlockScan<int, kBlock>;

struct Shared {
    typename BlockScan::TempStorage scan;
    int hist[kMaxBins];
    unsigned int prefix;  // bits of the k-th key fixed so far
    int below;            // entries strictly smaller than that prefix
    int chunk;            // bin search: winning slice of the histogram
    int chunk_base;       // running count at the start of that slice
    int emitted_lt;       // gather: neighbours emitted strictly below the k-th
    int emitted_eq;       // gather: neighbours emitted tied with the k-th
};

/**
 * Locate the histogram bin holding the `need`-th smallest entry, and fold it
 * into the running prefix.
 *
 * Each thread reduces a contiguous slice of bins, the block scans those partial
 * sums, and only the winning slice is walked serially. A 2048-bin histogram
 * therefore costs one block scan plus eight serial steps rather than 2048.
 */
__device__ void refine_prefix(Shared& sh, int bins, int shift, int need) {
    const int chunk = bins / kBlock;
    int local = 0;
    for (int b = threadIdx.x * chunk; b < (threadIdx.x + 1) * chunk; ++b) {
        local += sh.hist[b];
    }

    int excl = 0;
    BlockScan(sh.scan).ExclusiveSum(local, excl);

    if (threadIdx.x == 0) sh.chunk = -1;
    __syncthreads();
    // Exactly one thread matches, since 1 <= need <= entries matching the prefix.
    if (excl < need && excl + local >= need) {
        sh.chunk = static_cast<int>(threadIdx.x);
        sh.chunk_base = excl;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int acc = sh.chunk_base;
        int bin = sh.chunk * chunk;
        for (int b = bin; b < (sh.chunk + 1) * chunk; ++b) {
            if (acc + sh.hist[b] >= need) {
                bin = b;
                break;
            }
            acc += sh.hist[b];
        }
        sh.prefix |= static_cast<unsigned int>(bin) << shift;
        sh.below += acc;
    }
    __syncthreads();
}

__global__ void knn_select_kernel(float* __restrict__ tile, const float* __restrict__ norms,
                                  int rows, int row_offset, int n_points, int k,
                                  int* __restrict__ nbr_idx, float* __restrict__ nbr_d2,
                                  float* __restrict__ kdist2) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    float* row = tile + static_cast<size_t>(r) * n_points;
    const int self = row_offset + r;
    const float self_norm = norms[self];

    __shared__ Shared sh;
    if (threadIdx.x == 0) {
        sh.prefix = 0u;
        sh.below = 0;
    }

    // ---- pass 0: turn the Gram row into squared distances, and histogram it ----
    for (int b = threadIdx.x; b < kMaxBins; b += kBlock) sh.hist[b] = 0;
    __syncthreads();

    for (int j = threadIdx.x; j < n_points; j += kBlock) {
        // The expansion can dip below zero for near-identical points, hence the
        // clamp. The diagonal becomes +inf so a point never selects itself.
        //
        // The NaN test is not redundant: if a coordinate is large enough that its
        // square overflows float32, both norms become +inf and the expansion is
        // inf - inf. fmaxf would then return 0.0f -- IEEE says fmax yields the
        // non-NaN operand -- and the pair would look like the *nearest* possible
        // neighbour. Mapping it to +inf instead makes it the furthest, which is
        // the only defensible reading of an unrepresentable distance.
        const float raw = self_norm + norms[j] - 2.0f * row[j];
        const float d2 = (j == self || isnan(raw)) ? CUDART_INF_F : fmaxf(0.0f, raw);
        row[j] = d2;
        atomicAdd(&sh.hist[__float_as_uint(d2) >> pass_shift(0)], 1);
    }
    __syncthreads();
    refine_prefix(sh, 1 << pass_width(0), pass_shift(0), k);

    // ---- passes 1..2: narrow the key, re-reading the distances written above ----
    for (int p = 1; p < kPasses; ++p) {
        const int shift = pass_shift(p);
        const int width = pass_width(p);
        const int bins = 1 << width;
        const unsigned int digit_mask = static_cast<unsigned int>(bins - 1);
        const unsigned int prefix_mask = 0xFFFFFFFFu << (shift + width);
        const unsigned int prefix = sh.prefix;

        for (int b = threadIdx.x; b < bins; b += kBlock) sh.hist[b] = 0;
        __syncthreads();

        for (int j = threadIdx.x; j < n_points; j += kBlock) {
            const unsigned int u = __float_as_uint(row[j]);
            if ((u & prefix_mask) == (prefix & prefix_mask)) {
                atomicAdd(&sh.hist[(u >> shift) & digit_mask], 1);
            }
        }
        __syncthreads();
        refine_prefix(sh, bins, shift, k - sh.below);
    }

    const unsigned int kth = sh.prefix;  // exactly the k-th smallest key
    const int below = sh.below;

    if (threadIdx.x == 0) {
        kdist2[self] = __uint_as_float(kth);
        sh.emitted_lt = 0;
        sh.emitted_eq = 0;
    }
    __syncthreads();

    // ---- gather ----
    // Everything strictly below the k-th key is a neighbour. The remaining
    // k - below slots go to entries equal to it, in ascending index order.
    for (int base = 0; base < n_points; base += kBlock) {
        const int j = base + static_cast<int>(threadIdx.x);
        float v = 0.0f;
        unsigned int u = 0xFFFFFFFFu;
        if (j < n_points) {
            v = row[j];
            u = __float_as_uint(v);
        }
        const int lt = (j < n_points && u < kth) ? 1 : 0;
        const int eq = (j < n_points && u == kth) ? 1 : 0;

        // At most kBlock of each per chunk, so one packed scan serves both.
        int offset = 0;
        int total = 0;
        BlockScan(sh.scan).ExclusiveSum(lt | (eq << 16), offset, total);

        if (lt) {
            const int slot = sh.emitted_lt + (offset & 0xFFFF);
            if (slot < k) {
                const size_t o = static_cast<size_t>(slot) * n_points + self;
                nbr_idx[o] = j;
                nbr_d2[o] = v;
            }
        }
        if (eq) {
            const int slot = below + sh.emitted_eq + ((offset >> 16) & 0xFFFF);
            if (slot < k) {
                const size_t o = static_cast<size_t>(slot) * n_points + self;
                nbr_idx[o] = j;
                nbr_d2[o] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sh.emitted_lt += total & 0xFFFF;
            sh.emitted_eq += (total >> 16) & 0xFFFF;
        }
        __syncthreads();
    }
}

}  // namespace

void select_knn(float* tile, const float* norms, int rows, int row_offset, int n_points,
                int k, int* nbr_idx, float* nbr_d2, float* kdist2, cudaStream_t stream) {
    knn_select_kernel<<<rows, kBlock, 0, stream>>>(tile, norms, rows, row_offset, n_points,
                                                   k, nbr_idx, nbr_d2, kdist2);
    CULOF_CHECK(cudaGetLastError());
}

}  // namespace detail
}  // namespace culof
