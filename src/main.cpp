#include "cklab/bench.hpp"
#include "cklab/cuda_backend.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(std::ostream& os) {
    os << "CUDA Kernel Lab — SAXPY, reduction, and naive vs tiled GEMM\n\n"
       << "Usage:\n"
       << "  cklab [options]\n\n"
       << "Options:\n"
       << "  --mode host|cuda|auto   Backend (default: auto)\n"
       << "  --op all|copy|saxpy|reduce|gemm\n"
       << "  --n N                   GEMM N; also default vector length (default: 256)\n"
       << "  --vec N                 copy/saxpy/reduce length (default: --n)\n"
       << "  --m M --k K             Optional rectangular GEMM (default: n)\n"
       << "  --reps R                Timed repetitions (default: 8)\n"
       << "  --verify / --no-verify  Compare against CPU gold (default: on)\n"
       << "  --help                  This message\n\n"
       << "Build notes:\n"
       << "  Host path works without a GPU and is what CI runs.\n"
       << "  CUDA path requires nvcc at build time and a device at run time.\n";
}

cklab::Backend parse_backend(const std::string& s) {
    if (s == "host") {
        return cklab::Backend::Host;
    }
    if (s == "cuda") {
        return cklab::Backend::Cuda;
    }
    if (s == "auto") {
        return cklab::Backend::Auto;
    }
    throw std::invalid_argument("unknown --mode '" + s + "'");
}

cklab::Op parse_op(const std::string& s) {
    if (s == "all") {
        return cklab::Op::All;
    }
    if (s == "copy") {
        return cklab::Op::Copy;
    }
    if (s == "saxpy") {
        return cklab::Op::Saxpy;
    }
    if (s == "reduce") {
        return cklab::Op::Reduce;
    }
    if (s == "gemm") {
        return cklab::Op::Gemm;
    }
    throw std::invalid_argument("unknown --op '" + s + "'");
}

int parse_int(const char* s, const char* flag) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!s || end == s || *end != '\0') {
        throw std::invalid_argument(std::string("invalid integer for ") + flag);
    }
    return static_cast<int>(v);
}

} // namespace

int main(int argc, char** argv) {
    cklab::BenchConfig cfg;
    cfg.n = 256;
    cfg.reps = 8;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need = [&](const char* flag) -> const char* {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(flag) + " requires a value");
                }
                return argv[++i];
            };
            if (arg == "--help" || arg == "-h") {
                usage(std::cout);
                return 0;
            }
            if (arg == "--version") {
                std::cout << "cklab 1.0.0 cuda_compiled="
                          << (cklab::cuda::compiled() ? "yes" : "no") << "\n";
                return 0;
            }
            if (arg == "--mode") {
                cfg.backend = parse_backend(need("--mode"));
            } else if (arg == "--op") {
                cfg.op = parse_op(need("--op"));
            } else if (arg == "--n") {
                cfg.n = parse_int(need("--n"), "--n");
            } else if (arg == "--vec") {
                cfg.vec = parse_int(need("--vec"), "--vec");
            } else if (arg == "--m") {
                cfg.m = parse_int(need("--m"), "--m");
            } else if (arg == "--k") {
                cfg.k = parse_int(need("--k"), "--k");
            } else if (arg == "--reps") {
                cfg.reps = parse_int(need("--reps"), "--reps");
            } else if (arg == "--verify") {
                cfg.verify = true;
            } else if (arg == "--no-verify") {
                cfg.verify = false;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        const auto report = cklab::run_bench(cfg);
        cklab::print_report(cfg, report);
        return report.all_passed ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 2;
    }
}
