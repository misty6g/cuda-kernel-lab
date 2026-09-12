#pragma once

#include <string>
#include <vector>

namespace cklab {

enum class Backend { Auto, Host, Cuda };
enum class Op { All, Copy, Saxpy, Reduce, Gemm };

struct BenchConfig {
    Backend backend = Backend::Auto;
    Op op = Op::All;
    int n = 1024;
    int m = 0;     // 0 => use n
    int k = 0;     // 0 => use n
    int vec = 0;   // 0 => use n; vector length for copy / saxpy / reduce
    int reps = 10;
    bool verify = true;
};

struct BenchRow {
    std::string op;
    std::string variant;
    double ms = 0.0;
    std::string metric;
    bool verified = false;
    bool verify_ran = false;
    std::string note;
};

struct BenchReport {
    Backend resolved = Backend::Host;
    std::string device;
    std::vector<BenchRow> rows;
    bool all_passed = true;
};

Backend resolve_backend(Backend requested);
std::string backend_name(Backend b);
BenchReport run_bench(const BenchConfig& cfg);
void print_report(const BenchConfig& cfg, const BenchReport& report);

} // namespace cklab
