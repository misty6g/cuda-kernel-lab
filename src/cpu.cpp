#include "cklab/cpu.hpp"

#include "cklab/config.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cklab {
namespace {

void require_nonneg(int n, const char* what) {
    if (n < 0) {
        throw std::invalid_argument(std::string(what) + " must be >= 0");
    }
}

} // namespace

void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo, float hi) {
    std::uint32_t s = seed ? seed : 1u;
    const float span = hi - lo;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;
        const float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        x = lo + span * u;
    }
}

void fill_constant(std::vector<float>& v, float value) {
    std::fill(v.begin(), v.end(), value);
}

void saxpy_host(int n, float a, const float* x, float* y) {
    require_nonneg(n, "n");
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

float reduce_sum_sequential(int n, const float* x) {
    require_nonneg(n, "n");
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

float reduce_sum_tree(int n, const float* x, int block, int max_grid) {
    require_nonneg(n, "n");
    if (n == 0) {
        return 0.0f;
    }
    if (block <= 0 || (block & (block - 1)) != 0) {
        throw std::invalid_argument("reduce block size must be a positive power of two");
    }
    if (max_grid <= 0) {
        throw std::invalid_argument("max_grid must be > 0");
    }

    const int blocks = std::min((n + block - 1) / block, max_grid);
    std::vector<float> partial(static_cast<std::size_t>(blocks), 0.0f);

    // Model one CUDA block: each of `block` threads grid-strides, then a shared-memory tree.
    std::vector<float> smem(static_cast<std::size_t>(block));
    for (int b = 0; b < blocks; ++b) {
        std::fill(smem.begin(), smem.end(), 0.0f);
        for (int tid = 0; tid < block; ++tid) {
            float local = 0.0f;
            for (int i = b * block + tid; i < n; i += block * blocks) {
                local += x[i];
            }
            smem[static_cast<std::size_t>(tid)] = local;
        }
        for (int s = block / 2; s > 0; s >>= 1) {
            for (int tid = 0; tid < s; ++tid) {
                smem[static_cast<std::size_t>(tid)] += smem[static_cast<std::size_t>(tid + s)];
            }
        }
        partial[static_cast<std::size_t>(b)] = smem[0];
    }

    return reduce_sum_sequential(blocks, partial.data());
}

void gemm_naive_host(int M, int N, int K, const float* A, const float* B, float* C) {
    require_nonneg(M, "M");
    require_nonneg(N, "N");
    require_nonneg(K, "K");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

void gemm_tiled_host(int M, int N, int K, const float* A, const float* B, float* C, int tile) {
    require_nonneg(M, "M");
    require_nonneg(N, "N");
    require_nonneg(K, "K");
    if (tile <= 0) {
        throw std::invalid_argument("tile must be > 0");
    }

    std::vector<float> As(static_cast<std::size_t>(tile * tile));
    std::vector<float> Bs(static_cast<std::size_t>(tile * tile));
    std::vector<float> acc(static_cast<std::size_t>(tile * tile));

    for (int brow = 0; brow < M; brow += tile) {
        for (int bcol = 0; bcol < N; bcol += tile) {
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int k0 = 0; k0 < K; k0 += tile) {
                for (int i = 0; i < tile; ++i) {
                    for (int j = 0; j < tile; ++j) {
                        const int ar = brow + i;
                        const int ac = k0 + j;
                        As[i * tile + j] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
                        const int br = k0 + i;
                        const int bc = bcol + j;
                        Bs[i * tile + j] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
                    }
                }
                for (int i = 0; i < tile; ++i) {
                    for (int j = 0; j < tile; ++j) {
                        float s = acc[i * tile + j];
                        for (int t = 0; t < tile; ++t) {
                            s += As[i * tile + t] * Bs[t * tile + j];
                        }
                        acc[i * tile + j] = s;
                    }
                }
            }
            for (int i = 0; i < tile; ++i) {
                for (int j = 0; j < tile; ++j) {
                    const int r = brow + i;
                    const int c = bcol + j;
                    if (r < M && c < N) {
                        C[r * N + c] = acc[i * tile + j];
                    }
                }
            }
        }
    }
}

void copy_host(int n, const float* src, float* dst) {
    require_nonneg(n, "n");
    if (n == 0) {
        return;
    }
    std::memcpy(dst, src, bytes_of(n));
}

} // namespace cklab
