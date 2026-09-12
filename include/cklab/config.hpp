#pragma once

#include <cstddef>

namespace cklab {

constexpr int kTile = 16;     // GEMM shared-memory tile (threads per block dim)
constexpr int kBlock = 256;   // 1D kernels: saxpy / reduction / copy
constexpr int kMaxGrid = 2048;

// Default numerical compare. GEMM uses a K-scaled absolute tolerance on top of this.
constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

inline std::size_t bytes_of(int n) {
    return static_cast<std::size_t>(n) * sizeof(float);
}

} // namespace cklab
