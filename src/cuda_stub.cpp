#include "cklab/cuda_backend.hpp"

#include <stdexcept>

namespace cklab::cuda {
namespace {

[[noreturn]] void no_cuda() {
    throw std::runtime_error(
        "CUDA backend was not compiled into this binary. "
        "Rebuild with a CUDA toolkit (nvcc) or use --mode host.");
}

} // namespace

bool compiled() { return false; }

bool available() { return false; }

std::string device_info() { return "CUDA not compiled (host-only binary)"; }

void copy(int, const float*, float*) { no_cuda(); }
void saxpy(int, float, const float*, float*) { no_cuda(); }
float reduce_sum(int, const float*) { no_cuda(); }
void gemm_naive(int, int, int, const float*, const float*, float*) { no_cuda(); }
void gemm_tiled(int, int, int, const float*, const float*, float*) { no_cuda(); }

double bench_copy(int, const float*, float*, int) { no_cuda(); }
double bench_saxpy(int, float, const float*, const float*, float*, int) { no_cuda(); }
double bench_reduce(int, const float*, float*, int) { no_cuda(); }
double bench_gemm_naive(int, int, int, const float*, const float*, float*, int) { no_cuda(); }
double bench_gemm_tiled(int, int, int, const float*, const float*, float*, int) { no_cuda(); }

} // namespace cklab::cuda
