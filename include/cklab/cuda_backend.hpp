#pragma once

#include <string>

namespace cklab::cuda {

// True when this binary was built with nvcc / CMAKE_CUDA_COMPILER.
bool compiled();

// True when compiled() and cudaGetDeviceCount() > 0.
bool available();

std::string device_info();

// Host pointers in / out. Throws std::runtime_error if CUDA is unavailable.
void copy(int n, const float* src, float* dst);
void saxpy(int n, float a, const float* x, float* y);
float reduce_sum(int n, const float* x);
void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C);
void gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C);

// Kernel-only timers (cudaEvent). Upload once, warmup, time `reps` launches,
// download once into the output buffers. H2D/D2H are not included in the ms.
double bench_copy(int n, const float* src, float* dst, int reps);
double bench_saxpy(int n, float a, const float* x, const float* y_in, float* y_out, int reps);
double bench_reduce(int n, const float* x, float* sum, int reps);
double bench_gemm_naive(int M, int N, int K, const float* A, const float* B, float* C, int reps);
double bench_gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C, int reps);

} // namespace cklab::cuda
