/**
 * C++/CUDA tests.
 *
 * Every case is checked against a double-precision CPU implementation of LOF
 * written independently below, so a wrong kernel fails the test rather than
 * merely a crashing one.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "culof.h"
#include "culof_internal.cuh"

namespace {

/** Gaussian points, with the first `n_outliers` scattered over a wide box. */
std::vector<float> make_data(int n, int d, unsigned seed, int n_outliers = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> wide(-12.0f, 12.0f);
    std::vector<float> x(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            x[static_cast<size_t>(i) * d + j] = (i < n_outliers) ? wide(rng) : normal(rng);
        }
    }
    return x;
}

/** Reference LOF, straight from Breunig et al., in double precision. */
std::vector<double> cpu_lof(const std::vector<float>& x, int n, int d, int k) {
    std::vector<std::vector<double>> dist(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int t = 0; t < d; ++t) {
                const double diff = static_cast<double>(x[static_cast<size_t>(i) * d + t]) -
                                    static_cast<double>(x[static_cast<size_t>(j) * d + t]);
                s += diff * diff;
            }
            dist[i][j] = std::sqrt(s);
        }
        dist[i][i] = std::numeric_limits<double>::infinity();
    }

    std::vector<std::vector<int>> nbr(n);
    std::vector<double> kdist(n);
    for (int i = 0; i < n; ++i) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
            return dist[i][a] != dist[i][b] ? dist[i][a] < dist[i][b] : a < b;
        });
        nbr[i].assign(order.begin(), order.begin() + k);
        kdist[i] = dist[i][nbr[i][k - 1]];
    }

    std::vector<double> lrd(n);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j : nbr[i]) s += std::max(kdist[j], dist[i][j]);
        lrd[i] = 1.0 / (s / k + 1e-10);  // the 1e-10 is scikit-learn's
    }

    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j : nbr[i]) s += lrd[j] / lrd[i];
        out[i] = s / k;
    }
    return out;
}

double max_rel_err(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]) /
                                    std::max(std::abs(want[i]), 1e-9));
    }
    return worst;
}

class Culof : public ::testing::Test {
protected:
    void SetUp() override {
        if (!culof::cuda_available()) GTEST_SKIP() << "no CUDA device";
    }
};

}  // namespace

TEST_F(Culof, ReportsDevice) { EXPECT_FALSE(culof::device_info().empty()); }

TEST_F(Culof, MatchesReferenceLowDim) {
    const int n = 400, d = 3, k = 20;
    const auto x = make_data(n, d, 11, 8);
    EXPECT_LT(max_rel_err(culof::lof(x.data(), n, d, k), cpu_lof(x, n, d, k)), 1e-4);
}

TEST_F(Culof, MatchesReferenceHighDim) {
    const int n = 512, d = 32, k = 15;
    const auto x = make_data(n, d, 3, 10);
    EXPECT_LT(max_rel_err(culof::lof(x.data(), n, d, k), cpu_lof(x, n, d, k)), 1e-4);
}

/** The previous implementation rejected any k above 32: its selection kernel
 *  held candidates in a fixed 32-entry register array. */
TEST_F(Culof, SupportsLargeK) {
    const int n = 600, d = 4;
    const auto x = make_data(n, d, 5, 12);
    for (int k : {33, 64, 128, 257}) {
        EXPECT_LT(max_rel_err(culof::lof(x.data(), n, d, k), cpu_lof(x, n, d, k)), 1e-4)
            << "k=" << k;
    }
}

TEST_F(Culof, HandlesExtremeK) {
    const int n = 128, d = 2;
    const auto x = make_data(n, d, 7, 4);
    for (int k : {1, 2, n - 1}) {
        EXPECT_LT(max_rel_err(culof::lof(x.data(), n, d, k), cpu_lof(x, n, d, k)), 1e-4)
            << "k=" << k;
    }
}

/** Heavy duplication drives the tie-handling path of the radix select: most
 *  distances are exactly equal, so the k-th key has many holders. */
TEST_F(Culof, HandlesDuplicatePoints) {
    const int n = 300, d = 2, k = 10;
    std::vector<float> x(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; ++i) {
        const float base = static_cast<float>(i % 15);  // only 15 distinct positions
        x[static_cast<size_t>(i) * d + 0] = base;
        x[static_cast<size_t>(i) * d + 1] = base * 2.0f;
    }
    for (int i = 0; i < 5; ++i) x[static_cast<size_t>(i) * d] += 40.0f * (i + 1);

    const auto got = culof::lof(x.data(), n, d, k);
    for (float v : got) EXPECT_TRUE(std::isfinite(v));
    EXPECT_LT(max_rel_err(got, cpu_lof(x, n, d, k)), 1e-3);
}

TEST_F(Culof, ConstantFeatureSurvivesNormalization) {
    const int n = 200, d = 3, k = 10;
    auto x = make_data(n, d, 9, 5);
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i) * d + 1] = 4.0f;  // zero variance
    for (float v : culof::lof(x.data(), n, d, k, /*normalize=*/true)) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

/**
 * Tile height must not change the answer. CULOF_TILE_ROWS forces it, so the same
 * data goes through one pass and through many small tiles.
 *
 * Compared to a few ULP, not exactly: cuBLAS picks a different SGEMM kernel per
 * tile shape, which reorders the dot-product accumulation. Run-to-run
 * determinism at a fixed shape is asserted by IsDeterministic below.
 */
TEST_F(Culof, TilingDoesNotChangeResults) {
    const int n = 700, d = 5, k = 25;
    const auto x = make_data(n, d, 13, 9);

    unsetenv("CULOF_TILE_ROWS");
    const auto whole = culof::lof(x.data(), n, d, k);

    for (const char* rows : {"1", "7", "64", "699"}) {
        setenv("CULOF_TILE_ROWS", rows, 1);
        const auto tiled = culof::lof(x.data(), n, d, k);
        for (size_t i = 0; i < whole.size(); ++i) {
            EXPECT_FLOAT_EQ(whole[i], tiled[i]) << "tile_rows=" << rows << " i=" << i;
        }
    }
    unsetenv("CULOF_TILE_ROWS");
}

/** Guards the prefix-sum gather: atomics here would make results wobble. */
TEST_F(Culof, IsDeterministic) {
    const int n = 512, d = 6, k = 20;
    const auto x = make_data(n, d, 17, 7);
    const auto first = culof::lof(x.data(), n, d, k);
    for (int rep = 0; rep < 3; ++rep) {
        const auto again = culof::lof(x.data(), n, d, k);
        for (size_t i = 0; i < first.size(); ++i) {
            EXPECT_EQ(first[i], again[i]) << "i=" << i;
        }
    }
}

TEST_F(Culof, RejectsBadArguments) {
    const auto x = make_data(50, 2, 1);
    EXPECT_THROW(culof::lof(nullptr, 50, 2, 5), std::invalid_argument);
    EXPECT_THROW(culof::lof(x.data(), 1, 2, 1), std::invalid_argument);
    EXPECT_THROW(culof::lof(x.data(), 50, 0, 5), std::invalid_argument);
    EXPECT_THROW(culof::lof(x.data(), 50, 2, 0), std::invalid_argument);
    EXPECT_THROW(culof::lof(x.data(), 50, 2, 50), std::invalid_argument);
}

TEST_F(Culof, TileRowsRespectBudget) {
    // 1000 points is 4000 bytes per row.
    EXPECT_EQ(culof::detail::choose_tile_rows(1000, 40000), 10);
    EXPECT_EQ(culof::detail::choose_tile_rows(1000, 0), 1);              // never zero
    EXPECT_EQ(culof::detail::choose_tile_rows(1000, 1ull << 40), 1000);  // never above n
}
