#pragma once

#include <cstdint>
#include <vector>

namespace cklab {

// Deterministic fill for benches and tests (LCG, not a cryptographic RNG).
void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo = -1.0f, float hi = 1.0f);
void fill_constant(std::vector<float>& v, float value);

// y[i] = a * x[i] + y[i]
void saxpy_host(int n, float a, const float* x, float* y);

// Sequential gold sum. Prefer this for correctness checks.
float reduce_sum_sequential(int n, const float* x);

// Block-strided tree reduction that models the CUDA kernel (float associativity
// means this will not be bit-identical to the sequential gold).
float reduce_sum_tree(int n, const float* x, int block = 256, int max_grid = 2048);

// C[M,N] = A[M,K] * B[K,N], row-major.
void gemm_naive_host(int M, int N, int K, const float* A, const float* B, float* C);

// Same math as the CUDA tiled kernel: load TILE x TILE panels of A/B, accumulate
// a TILE x TILE panel of C. `tile` is runtime so tests can sweep remainders.
void gemm_tiled_host(int M, int N, int K, const float* A, const float* B, float* C, int tile = 16);

// In-place STREAM-style copy: dst[i] = src[i]
void copy_host(int n, const float* src, float* dst);

} // namespace cklab
