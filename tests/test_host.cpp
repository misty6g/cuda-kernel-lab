#include "cklab/bench.hpp"
#include "cklab/config.hpp"
#include "cklab/cpu.hpp"
#include "cklab/cuda_backend.hpp"
#include "cklab/verify.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;
int g_case_fail = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (cond) {                                                                                \
            ++g_pass;                                                                              \
        } else {                                                                                   \
            ++g_fail;                                                                              \
            ++g_case_fail;                                                                         \
            std::cerr << "  CHECK failed: " #cond "  (" << __FILE__ << ":" << __LINE__ << ")\n";   \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                      \
    do {                                                                                           \
        const double _da = static_cast<double>(a);                                                 \
        const double _db = static_cast<double>(b);                                                 \
        if (std::fabs(_da - _db) <= static_cast<double>(tol)) {                                    \
            ++g_pass;                                                                              \
        } else {                                                                                   \
            ++g_fail;                                                                              \
            ++g_case_fail;                                                                         \
            std::cerr << "  CHECK_NEAR failed: " << _da << " vs " << _db << " tol=" << (tol)       \
                      << "  (" << __FILE__ << ":" << __LINE__ << ")\n";                            \
        }                                                                                          \
    } while (0)

void section(const char* name) {
    g_case_fail = 0;
    std::cout << "[test] " << name << "\n";
}

void section_done() {
    if (g_case_fail == 0) {
        std::cout << "  ok\n";
    }
}

void test_fill_deterministic() {
    section("fill_uniform is deterministic");
    std::vector<float> a(8), b(8);
    cklab::fill_uniform(a, 42u);
    cklab::fill_uniform(b, 42u);
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i] == b[i]);
    }
    cklab::fill_constant(a, 3.5f);
    CHECK(a[0] == 3.5f && a[7] == 3.5f);
    section_done();
}

void test_saxpy() {
    section("saxpy");
    {
        std::vector<float> x = {1, 2, 3};
        std::vector<float> y = {10, 20, 30};
        cklab::saxpy_host(3, 0.0f, x.data(), y.data());
        CHECK(y[0] == 10 && y[1] == 20 && y[2] == 30);
    }
    {
        std::vector<float> x = {1, 2, 3};
        std::vector<float> y = {10, 20, 30};
        cklab::saxpy_host(3, 2.0f, x.data(), y.data());
        CHECK(y[0] == 12 && y[1] == 24 && y[2] == 36);
    }
    {
        std::vector<float> x, y;
        cklab::saxpy_host(0, 2.0f, x.data(), y.data());
        CHECK(y.empty());
    }
    section_done();
}

void test_copy() {
    section("copy");
    std::vector<float> src = {1, -2, 3.5f};
    std::vector<float> dst(3, 0.0f);
    cklab::copy_host(3, src.data(), dst.data());
    CHECK(dst[0] == 1 && dst[1] == -2 && dst[2] == 3.5f);
    cklab::copy_host(0, src.data(), dst.data());
    section_done();
}

void test_reduce() {
    section("reduce sequential + tree model");
    CHECK_NEAR(cklab::reduce_sum_sequential(0, nullptr), 0.0f, 0.0f);

    std::vector<float> ones(17, 1.0f);
    CHECK_NEAR(cklab::reduce_sum_sequential(17, ones.data()), 17.0f, 0.0f);
    CHECK_NEAR(cklab::reduce_sum_tree(17, ones.data(), 8, 4), 17.0f, 0.0f);

    std::vector<float> empty;
    CHECK_NEAR(cklab::reduce_sum_tree(0, empty.data()), 0.0f, 0.0f);

    std::vector<float> single = {3.25f};
    CHECK_NEAR(cklab::reduce_sum_tree(1, single.data()), 3.25f, 0.0f);

    // Integers-as-floats stay exact for modest n.
    std::vector<float> seq(1024);
    for (int i = 0; i < 1024; ++i) {
        seq[static_cast<std::size_t>(i)] = 1.0f;
    }
    CHECK_NEAR(cklab::reduce_sum_tree(1024, seq.data()), 1024.0f, 0.0f);

    bool threw = false;
    try {
        cklab::reduce_sum_tree(4, seq.data(), 12);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    section_done();
}

void test_gemm_identity() {
    section("gemm identity");
    const int n = 5;
    std::vector<float> I(static_cast<std::size_t>(n * n), 0.0f);
    std::vector<float> A(static_cast<std::size_t>(n * n));
    std::vector<float> C(static_cast<std::size_t>(n * n), -1.0f);
    cklab::fill_uniform(A, 9u);
    for (int i = 0; i < n; ++i) {
        I[static_cast<std::size_t>(i * n + i)] = 1.0f;
    }
    cklab::gemm_naive_host(n, n, n, A.data(), I.data(), C.data());
    auto cmp = cklab::allclose(n * n, C.data(), A.data(), 1e-6f, 1e-6f);
    CHECK(cmp.ok);

    cklab::fill_constant(C, 0.0f);
    cklab::gemm_tiled_host(n, n, n, I.data(), A.data(), C.data(), 4);
    cmp = cklab::allclose(n * n, C.data(), A.data(), 1e-6f, 1e-6f);
    CHECK(cmp.ok);
    section_done();
}

void test_gemm_known() {
    section("gemm 2x2 known values");
    const float A[] = {1, 2, 3, 4};
    const float B[] = {5, 6, 7, 8};
    float C[4] = {};
    cklab::gemm_naive_host(2, 2, 2, A, B, C);
    CHECK_NEAR(C[0], 19.0f, 1e-6f);
    CHECK_NEAR(C[1], 22.0f, 1e-6f);
    CHECK_NEAR(C[2], 43.0f, 1e-6f);
    CHECK_NEAR(C[3], 50.0f, 1e-6f);
    section_done();
}

void test_gemm_tiled_matches_naive() {
    section("tiled GEMM matches naive (including remainders)");
    struct Case {
        int M, N, K, tile;
    };
    const Case cases[] = {
        {1, 1, 1, 16},
        {16, 16, 16, 16},
        {17, 13, 19, 16},
        {32, 16, 48, 16},
        {7, 9, 5, 8},
        {3, 20, 3, 4},
        {0, 4, 4, 16},
        {8, 0, 8, 8},
    };
    for (const auto& c : cases) {
        std::vector<float> A(static_cast<std::size_t>(c.M) * static_cast<std::size_t>(c.K));
        std::vector<float> B(static_cast<std::size_t>(c.K) * static_cast<std::size_t>(c.N));
        std::vector<float> naive(static_cast<std::size_t>(c.M) * static_cast<std::size_t>(c.N), 0);
        std::vector<float> tiled(naive.size(), 0);
        cklab::fill_uniform(A, 11u + static_cast<std::uint32_t>(c.M));
        cklab::fill_uniform(B, 22u + static_cast<std::uint32_t>(c.N));
        cklab::gemm_naive_host(c.M, c.N, c.K, A.data(), B.data(), naive.data());
        cklab::gemm_tiled_host(c.M, c.N, c.K, A.data(), B.data(), tiled.data(), c.tile);
        const int elems = c.M * c.N;
        if (elems == 0) {
            CHECK(true);
            continue;
        }
        const auto cmp = cklab::allclose(elems, tiled.data(), naive.data(), 1e-4f, 1e-4f);
        if (!cmp.ok) {
            std::cerr << "  mismatch M=" << c.M << " N=" << c.N << " K=" << c.K
                      << " tile=" << c.tile << " " << cklab::format_compare(cmp) << "\n";
        }
        CHECK(cmp.ok);
    }
    section_done();
}

void test_allclose() {
    section("allclose detects mismatch");
    const float a[] = {1, 2, 3};
    const float b[] = {1, 2, 3};
    const float c[] = {1, 2, 9};
    CHECK(cklab::allclose(3, a, b, 1e-6f, 1e-6f).ok);
    const auto bad = cklab::allclose(3, a, c, 1e-6f, 1e-6f);
    CHECK(!bad.ok);
    CHECK(bad.first_index == 2);
    CHECK(bad.mismatches == 1);
    section_done();
}

void test_host_bench_smoke() {
    section("host bench smoke (tiny GEMM)");
    cklab::BenchConfig cfg;
    cfg.backend = cklab::Backend::Host;
    cfg.n = 32;
    cfg.reps = 2;
    cfg.verify = true;
    const auto report = cklab::run_bench(cfg);
    CHECK(report.resolved == cklab::Backend::Host);
    CHECK(report.all_passed);
    CHECK(!report.rows.empty());
    section_done();
}

void test_cuda_query_host_binary() {
    section("CUDA query on this binary");
    // Host-only CI builds should report not compiled. A local nvcc build
    // reports compiled=true; available() still depends on a device.
    (void)cklab::cuda::compiled();
    (void)cklab::cuda::available();
    const std::string info = cklab::cuda::device_info();
    CHECK(!info.empty());
    if (!cklab::cuda::compiled()) {
        bool threw = false;
        try {
            float y = 0;
            cklab::cuda::saxpy(1, 1.0f, &y, &y);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }
    section_done();
}

void test_cuda_path_if_present() {
    if (!cklab::cuda::compiled() || !cklab::cuda::available()) {
        std::cout << "[test] CUDA kernels skipped (no device or host-only build)\n";
        return;
    }
    section("CUDA kernels vs CPU gold");
    {
        std::vector<float> x = {1, 2, 3, 4};
        std::vector<float> y = {10, 20, 30, 40};
        std::vector<float> gold = y;
        cklab::saxpy_host(4, 2.0f, x.data(), gold.data());
        cklab::cuda::saxpy(4, 2.0f, x.data(), y.data());
        CHECK(cklab::allclose(4, y.data(), gold.data(), 1e-6f, 1e-6f).ok);
    }
    {
        std::vector<float> x(257, 1.0f);
        const float got = cklab::cuda::reduce_sum(257, x.data());
        CHECK_NEAR(got, 257.0f, 1e-3f);
    }
    {
        const int M = 17, N = 13, K = 19;
        std::vector<float> A(M * K), B(K * N), naive(M * N), tiled(M * N);
        cklab::fill_uniform(A, 3u);
        cklab::fill_uniform(B, 4u);
        cklab::gemm_naive_host(M, N, K, A.data(), B.data(), naive.data());
        cklab::cuda::gemm_naive(M, N, K, A.data(), B.data(), tiled.data());
        CHECK(cklab::allclose(M * N, tiled.data(), naive.data(), 2e-3f, 1e-3f).ok);
        cklab::cuda::gemm_tiled(M, N, K, A.data(), B.data(), tiled.data());
        CHECK(cklab::allclose(M * N, tiled.data(), naive.data(), 2e-3f, 1e-3f).ok);
    }
    section_done();
}

} // namespace

int main() {
    test_fill_deterministic();
    test_saxpy();
    test_copy();
    test_reduce();
    test_gemm_identity();
    test_gemm_known();
    test_gemm_tiled_matches_naive();
    test_allclose();
    test_host_bench_smoke();
    test_cuda_query_host_binary();
    test_cuda_path_if_present();

    std::cout << "\n" << g_pass << " checks passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
