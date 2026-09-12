#include "cklab/bench.hpp"

#include "cklab/config.hpp"
#include "cklab/cpu.hpp"
#include "cklab/cuda_backend.hpp"
#include "cklab/timing.hpp"
#include "cklab/verify.hpp"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cklab {
namespace {

int dim_or(int explicit_dim, int fallback) {
    return explicit_dim > 0 ? explicit_dim : fallback;
}

int vec_n(const BenchConfig& cfg) { return dim_or(cfg.vec, cfg.n); }

float gemm_atol(int K) {
    return kAtol + 1.0e-4f * static_cast<float>(std::max(K, 1));
}

void add_row(BenchReport& report, BenchRow row) {
    if (row.verify_ran && !row.verified) {
        report.all_passed = false;
    }
    report.rows.push_back(std::move(row));
}

void run_copy(const BenchConfig& cfg, Backend backend, BenchReport& report) {
    const int n = vec_n(cfg);
    std::vector<float> src(static_cast<std::size_t>(n));
    std::vector<float> dst(static_cast<std::size_t>(n), 0.0f);
    fill_uniform(src, 1u);

    double ms = 0.0;
    if (backend == Backend::Host) {
        ms = time_ms([&] { copy_host(n, src.data(), dst.data()); }, cfg.reps);
    } else {
        ms = cuda::bench_copy(n, src.data(), dst.data(), cfg.reps);
    }

    BenchRow row;
    row.op = "copy";
    row.variant = backend_name(backend);
    row.ms = ms;
    const double bytes = 2.0 * static_cast<double>(bytes_of(n));
    char metric[64];
    std::snprintf(metric, sizeof(metric), "%.2f GB/s", gbps(bytes, ms));
    row.metric = metric;
    if (cfg.verify) {
        const auto cmp = allclose(n, dst.data(), src.data(), kAtol, kRtol);
        row.verify_ran = true;
        row.verified = cmp.ok;
        row.note = format_compare(cmp);
    }
    add_row(report, std::move(row));
}

void run_saxpy(const BenchConfig& cfg, Backend backend, BenchReport& report) {
    constexpr float a = 2.0f;
    const int n = vec_n(cfg);
    std::vector<float> x(static_cast<std::size_t>(n));
    std::vector<float> y(static_cast<std::size_t>(n));
    std::vector<float> y_gold(static_cast<std::size_t>(n));
    fill_uniform(x, 2u);
    fill_uniform(y, 3u);
    y_gold = y;
    saxpy_host(n, a, x.data(), y_gold.data());

    std::vector<float> y_work = y;
    double ms = 0.0;
    if (backend == Backend::Host) {
        saxpy_host(n, a, x.data(), y_work.data()); // warmup
        double total = 0.0;
        for (int i = 0; i < cfg.reps; ++i) {
            y_work = y;
            Stopwatch sw;
            sw.start();
            saxpy_host(n, a, x.data(), y_work.data());
            total += sw.elapsed_ms();
        }
        ms = total / static_cast<double>(cfg.reps);
    } else {
        ms = cuda::bench_saxpy(n, a, x.data(), y.data(), y_work.data(), cfg.reps);
    }

    BenchRow row;
    row.op = "saxpy";
    row.variant = backend_name(backend);
    row.ms = ms;
    const double bytes = 3.0 * static_cast<double>(bytes_of(n));
    char metric[64];
    std::snprintf(metric, sizeof(metric), "%.2f GB/s", gbps(bytes, ms));
    row.metric = metric;
    if (cfg.verify) {
        const auto cmp = allclose(n, y_work.data(), y_gold.data(), kAtol, kRtol);
        row.verify_ran = true;
        row.verified = cmp.ok;
        row.note = format_compare(cmp);
    }
    add_row(report, std::move(row));
}

void run_reduce(const BenchConfig& cfg, Backend backend, BenchReport& report) {
    const int n = vec_n(cfg);
    std::vector<float> x(static_cast<std::size_t>(n));
    fill_uniform(x, 4u);
    const float gold = reduce_sum_sequential(n, x.data());

    auto check_sum = [&](float got, BenchRow& row) {
        if (!cfg.verify) {
            return;
        }
        const float atol = kAtol + 1.0e-5f * static_cast<float>(std::max(n, 1));
        const bool ok = std::fabs(got - gold) <= atol + kRtol * std::fabs(gold);
        row.verify_ran = true;
        row.verified = ok;
        char note[96];
        std::snprintf(note, sizeof(note), "%s got=%.6g gold=%.6g", ok ? "PASS" : "FAIL", got, gold);
        row.note = note;
    };

    if (backend == Backend::Host) {
        float tree = 0.0f;
        const double ms_seq =
            time_ms([&] { (void)reduce_sum_sequential(n, x.data()); }, cfg.reps);
        const double ms_tree = time_ms([&] { tree = reduce_sum_tree(n, x.data()); }, cfg.reps);
        const double bytes = static_cast<double>(bytes_of(n));

        BenchRow seq;
        seq.op = "reduce";
        seq.variant = "sequential";
        seq.ms = ms_seq;
        char m1[64];
        std::snprintf(m1, sizeof(m1), "%.2f GB/s", gbps(bytes, ms_seq));
        seq.metric = m1;
        check_sum(gold, seq);
        add_row(report, std::move(seq));

        BenchRow tr;
        tr.op = "reduce";
        tr.variant = "tree-model";
        tr.ms = ms_tree;
        char m2[64];
        std::snprintf(m2, sizeof(m2), "%.2f GB/s", gbps(bytes, ms_tree));
        tr.metric = m2;
        check_sum(tree, tr);
        add_row(report, std::move(tr));
        return;
    }

    float got = 0.0f;
    const double ms = cuda::bench_reduce(n, x.data(), &got, cfg.reps);
    BenchRow row;
    row.op = "reduce";
    row.variant = "shared-mem";
    row.ms = ms;
    char metric[64];
    std::snprintf(metric, sizeof(metric), "%.2f GB/s",
                  gbps(static_cast<double>(bytes_of(n)), ms));
    row.metric = metric;
    check_sum(got, row);
    add_row(report, std::move(row));
}

void run_gemm(const BenchConfig& cfg, Backend backend, BenchReport& report) {
    const int M = dim_or(cfg.m, cfg.n);
    const int N = cfg.n;
    const int K = dim_or(cfg.k, cfg.n);

    std::vector<float> A(static_cast<std::size_t>(M) * static_cast<std::size_t>(K));
    std::vector<float> B(static_cast<std::size_t>(K) * static_cast<std::size_t>(N));
    std::vector<float> C(static_cast<std::size_t>(M) * static_cast<std::size_t>(N), 0.0f);
    std::vector<float> gold(C.size(), 0.0f);
    fill_uniform(A, 5u);
    fill_uniform(B, 6u);
    gemm_naive_host(M, N, K, A.data(), B.data(), gold.data());

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const float atol = gemm_atol(K);

    auto finish = [&](const char* variant, double ms, const std::vector<float>& got) {
        BenchRow row;
        row.op = "gemm";
        row.variant = variant;
        row.ms = ms;
        char metric[64];
        std::snprintf(metric, sizeof(metric), "%.2f GFLOPS", gflops(flops, ms));
        row.metric = metric;
        if (cfg.verify) {
            const auto cmp =
                allclose(static_cast<int>(got.size()), got.data(), gold.data(), atol, kRtol);
            row.verify_ran = true;
            row.verified = cmp.ok;
            row.note = format_compare(cmp);
        }
        add_row(report, std::move(row));
    };

    if (backend == Backend::Host) {
        const double ms_naive =
            time_ms([&] { gemm_naive_host(M, N, K, A.data(), B.data(), C.data()); }, cfg.reps);
        finish("naive", ms_naive, C);
        const double ms_tiled =
            time_ms([&] { gemm_tiled_host(M, N, K, A.data(), B.data(), C.data(), kTile); },
                    cfg.reps);
        finish("tiled", ms_tiled, C);
        return;
    }

    const double ms_naive =
        cuda::bench_gemm_naive(M, N, K, A.data(), B.data(), C.data(), cfg.reps);
    finish("naive", ms_naive, C);
    const double ms_tiled =
        cuda::bench_gemm_tiled(M, N, K, A.data(), B.data(), C.data(), cfg.reps);
    finish("tiled", ms_tiled, C);
}

} // namespace

Backend resolve_backend(Backend requested) {
    if (requested == Backend::Host) {
        return Backend::Host;
    }
    if (requested == Backend::Cuda) {
        if (!cuda::compiled()) {
            throw std::runtime_error(
                "Requested --mode cuda, but this binary was built without CUDA. "
                "Use --mode host or rebuild with nvcc.");
        }
        if (!cuda::available()) {
            throw std::runtime_error(
                "Requested --mode cuda, but no CUDA device is visible.");
        }
        return Backend::Cuda;
    }
    if (cuda::compiled() && cuda::available()) {
        return Backend::Cuda;
    }
    return Backend::Host;
}

std::string backend_name(Backend b) {
    return b == Backend::Cuda ? "cuda" : "host";
}

BenchReport run_bench(const BenchConfig& cfg) {
    if (cfg.n < 0 || cfg.m < 0 || cfg.k < 0 || cfg.vec < 0) {
        throw std::invalid_argument("dimensions must be >= 0");
    }
    if (cfg.reps <= 0) {
        throw std::invalid_argument("--reps must be > 0");
    }

    BenchReport report;
    report.resolved = resolve_backend(cfg.backend);
    report.device = (report.resolved == Backend::Cuda) ? cuda::device_info()
                                                      : "CPU (gold reference + tiled algorithm model)";

    const bool all = cfg.op == Op::All;
    if (all || cfg.op == Op::Copy) {
        run_copy(cfg, report.resolved, report);
    }
    if (all || cfg.op == Op::Saxpy) {
        run_saxpy(cfg, report.resolved, report);
    }
    if (all || cfg.op == Op::Reduce) {
        run_reduce(cfg, report.resolved, report);
    }
    if (all || cfg.op == Op::Gemm) {
        run_gemm(cfg, report.resolved, report);
    }
    return report;
}

void print_report(const BenchConfig& cfg, const BenchReport& report) {
    const int M = dim_or(cfg.m, cfg.n);
    const int N = cfg.n;
    const int K = dim_or(cfg.k, cfg.n);

    std::cout << "CUDA Kernel Lab\n";
    std::cout << "  backend : " << backend_name(report.resolved) << "\n";
    std::cout << "  device  : " << report.device << "\n";
    std::cout << "  n       : " << cfg.n << "   GEMM " << M << "x" << K << " * " << K << "x" << N
              << "   vec " << vec_n(cfg) << "\n";
    std::cout << "  reps    : " << cfg.reps << " (mean; 1 warmup excluded)\n";
    if (report.resolved == Backend::Cuda) {
        std::cout << "  timers  : CUDA events, kernel only (H2D/D2H excluded)\n";
    } else {
        std::cout << "  timers  : std::chrono::steady_clock (host algorithms)\n";
    }
    std::cout << "\n";
    std::cout << std::left << std::setw(10) << "op" << std::setw(14) << "variant" << std::right
              << std::setw(12) << "time(ms)" << "  " << std::left << std::setw(16) << "metric"
              << "  verify\n";
    std::cout << std::string(9, '-') << "  " << std::string(13, '-') << " " << std::string(12, '-')
              << "  " << std::string(16, '-') << "  ------\n";

    std::cout << std::fixed << std::setprecision(4);
    for (const auto& row : report.rows) {
        std::cout << std::left << std::setw(10) << row.op << std::setw(14) << row.variant
                  << std::right << std::setw(12) << row.ms << "  " << std::left << std::setw(16)
                  << row.metric << "  ";
        if (!row.verify_ran) {
            std::cout << "skip\n";
        } else if (row.verified) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL  " << row.note << "\n";
        }
    }
    std::cout << "\n";
    if (cfg.verify) {
        std::cout << (report.all_passed ? "All checks passed.\n" : "One or more checks FAILED.\n");
    } else {
        std::cout << "Verification disabled (--no-verify).\n";
    }
}

} // namespace cklab
