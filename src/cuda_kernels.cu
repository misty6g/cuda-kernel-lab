#include "cklab/config.hpp"
#include "cklab/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CKLAB_CUDA_CHECK(expr)                                                                     \
    do {                                                                                           \
        const cudaError_t err_ = (expr);                                                           \
        if (err_ != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) +     \
                                     " (" + #expr + ")");                                          \
        }                                                                                          \
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(int n) : n_(n) {
        if (n < 0) {
            throw std::invalid_argument("DeviceBuffer size must be >= 0");
        }
        if (n > 0) {
            CKLAB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                        static_cast<size_t>(n) * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    void upload(const T* host) {
        if (n_ == 0) {
            return;
        }
        CKLAB_CUDA_CHECK(cudaMemcpy(ptr_, host, static_cast<size_t>(n_) * sizeof(T),
                                    cudaMemcpyHostToDevice));
    }

    void download(T* host) const {
        if (n_ == 0) {
            return;
        }
        CKLAB_CUDA_CHECK(cudaMemcpy(host, ptr_, static_cast<size_t>(n_) * sizeof(T),
                                    cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    int n_ = 0;
};

__global__ void copy_kernel(int n, const float* src, float* dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

// Grid-stride SAXPY: y = a*x + y. Memory-bound; 3 traffic per element.
__global__ void saxpy_kernel(int n, float a, const float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// One partial sum per block. Threads grid-stride, then a shared-memory tree.
// Block dim must be a power of two (we launch kBlock = 256).
__global__ void reduce_sum_kernel(int n, const float* in, float* partial) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}

__global__ void gemm_naive_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// Tiled GEMM. Each block owns a TILE x TILE panel of C.
// A-tile load is coalesced along K; B-tile load is coalesced along N.
// Reuse: each A element is reused TILE times from shared memory (same for B).
template <int TILE>
__global__ void gemm_tiled_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int a_col = k0 + threadIdx.x;
        const int b_row = k0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int grid_1d(int n) {
    const int need = (n + cklab::kBlock - 1) / cklab::kBlock;
    return std::max(1, std::min(need, cklab::kMaxGrid));
}

void require_ready() {
    if (!cklab::cuda::available()) {
        throw std::runtime_error("No CUDA device available");
    }
}

class CudaEvents {
public:
    CudaEvents() {
        CKLAB_CUDA_CHECK(cudaEventCreate(&start_));
        CKLAB_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEvents() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaEvents(const CudaEvents&) = delete;
    CudaEvents& operator=(const CudaEvents&) = delete;

    template <typename Launch>
    double time_launches(Launch&& launch, int reps) {
        launch();
        CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
        CKLAB_CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < reps; ++i) {
            launch();
        }
        CKLAB_CUDA_CHECK(cudaEventRecord(stop_));
        CKLAB_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CKLAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) / static_cast<double>(reps);
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace

namespace cklab::cuda {

bool compiled() { return true; }

bool available() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

std::string device_info() {
    if (!available()) {
        return compiled() ? "CUDA compiled, no device" : "CUDA not compiled";
    }
    cudaDeviceProp prop{};
    CKLAB_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return std::string(prop.name) + " (SM " + std::to_string(prop.major) + "." +
           std::to_string(prop.minor) + ", " +
           std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MiB)";
}

void copy(int n, const float* src, float* dst) {
    require_ready();
    DeviceBuffer<float> d_src(n);
    DeviceBuffer<float> d_dst(n);
    d_src.upload(src);
    copy_kernel<<<grid_1d(n), kBlock>>>(n, d_src.get(), d_dst.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    d_dst.download(dst);
}

void saxpy(int n, float a, const float* x, float* y) {
    require_ready();
    DeviceBuffer<float> d_x(n);
    DeviceBuffer<float> d_y(n);
    d_x.upload(x);
    d_y.upload(y);
    saxpy_kernel<<<grid_1d(n), kBlock>>>(n, a, d_x.get(), d_y.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.download(y);
}

float reduce_sum(int n, const float* x) {
    require_ready();
    if (n == 0) {
        return 0.0f;
    }
    DeviceBuffer<float> d_x(n);
    d_x.upload(x);
    const int blocks = grid_1d(n);
    DeviceBuffer<float> d_partial(blocks);
    reduce_sum_kernel<<<blocks, kBlock, static_cast<size_t>(kBlock) * sizeof(float)>>>(
        n, d_x.get(), d_partial.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> partial(static_cast<size_t>(blocks));
    d_partial.download(partial.data());
    float sum = 0.0f;
    for (float v : partial) {
        sum += v;
    }
    return sum;
}

void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    require_ready();
    if (M == 0 || N == 0) {
        return;
    }
    DeviceBuffer<float> d_A(M * K);
    DeviceBuffer<float> d_B(K * N);
    DeviceBuffer<float> d_C(M * N);
    d_A.upload(A);
    d_B.upload(B);
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    gemm_naive_kernel<<<blocks, threads>>>(M, N, K, d_A.get(), d_B.get(), d_C.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    d_C.download(C);
}

void gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
    require_ready();
    if (M == 0 || N == 0) {
        return;
    }
    DeviceBuffer<float> d_A(M * K);
    DeviceBuffer<float> d_B(K * N);
    DeviceBuffer<float> d_C(M * N);
    d_A.upload(A);
    d_B.upload(B);
    dim3 threads(kTile, kTile);
    dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    gemm_tiled_kernel<kTile><<<blocks, threads>>>(M, N, K, d_A.get(), d_B.get(), d_C.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    d_C.download(C);
}

double bench_copy(int n, const float* src, float* dst, int reps) {
    require_ready();
    DeviceBuffer<float> d_src(n);
    DeviceBuffer<float> d_dst(n);
    d_src.upload(src);
    CudaEvents ev;
    const double ms = ev.time_launches(
        [&] { copy_kernel<<<grid_1d(n), kBlock>>>(n, d_src.get(), d_dst.get()); }, reps);
    CKLAB_CUDA_CHECK(cudaGetLastError());
    d_dst.download(dst);
    return ms;
}

double bench_saxpy(int n, float a, const float* x, const float* y_in, float* y_out, int reps) {
    require_ready();
    DeviceBuffer<float> d_x(n);
    DeviceBuffer<float> d_y(n);
    d_x.upload(x);
    d_y.upload(y_in);
    CudaEvents ev;
    const double ms = ev.time_launches(
        [&] { saxpy_kernel<<<grid_1d(n), kBlock>>>(n, a, d_x.get(), d_y.get()); }, reps);
    CKLAB_CUDA_CHECK(cudaGetLastError());
    // Timing may have applied SAXPY `reps` times in place. Reset and run once for gold compare.
    d_y.upload(y_in);
    saxpy_kernel<<<grid_1d(n), kBlock>>>(n, a, d_x.get(), d_y.get());
    CKLAB_CUDA_CHECK(cudaGetLastError());
    CKLAB_CUDA_CHECK(cudaDeviceSynchronize());
    d_y.download(y_out);
    return ms;
}

double bench_reduce(int n, const float* x, float* sum, int reps) {
    require_ready();
    if (n == 0) {
        *sum = 0.0f;
        return 0.0;
    }
    DeviceBuffer<float> d_x(n);
    d_x.upload(x);
    const int blocks = grid_1d(n);
    DeviceBuffer<float> d_partial(blocks);
    const size_t smem = static_cast<size_t>(kBlock) * sizeof(float);
    CudaEvents ev;
    const double ms = ev.time_launches(
        [&] {
            reduce_sum_kernel<<<blocks, kBlock, smem>>>(n, d_x.get(), d_partial.get());
        },
        reps);
    CKLAB_CUDA_CHECK(cudaGetLastError());
    std::vector<float> partial(static_cast<size_t>(blocks));
    d_partial.download(partial.data());
    float acc = 0.0f;
    for (float v : partial) {
        acc += v;
    }
    *sum = acc;
    return ms;
}

double bench_gemm_naive(int M, int N, int K, const float* A, const float* B, float* C, int reps) {
    require_ready();
    if (M == 0 || N == 0) {
        return 0.0;
    }
    DeviceBuffer<float> d_A(M * K);
    DeviceBuffer<float> d_B(K * N);
    DeviceBuffer<float> d_C(M * N);
    d_A.upload(A);
    d_B.upload(B);
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    CudaEvents ev;
    const double ms = ev.time_launches(
        [&] { gemm_naive_kernel<<<blocks, threads>>>(M, N, K, d_A.get(), d_B.get(), d_C.get()); },
        reps);
    CKLAB_CUDA_CHECK(cudaGetLastError());
    d_C.download(C);
    return ms;
}

double bench_gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C, int reps) {
    require_ready();
    if (M == 0 || N == 0) {
        return 0.0;
    }
    DeviceBuffer<float> d_A(M * K);
    DeviceBuffer<float> d_B(K * N);
    DeviceBuffer<float> d_C(M * N);
    d_A.upload(A);
    d_B.upload(B);
    dim3 threads(kTile, kTile);
    dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    CudaEvents ev;
    const double ms = ev.time_launches(
        [&] { gemm_tiled_kernel<kTile><<<blocks, threads>>>(M, N, K, d_A.get(), d_B.get(), d_C.get()); },
        reps);
    CKLAB_CUDA_CHECK(cudaGetLastError());
    d_C.download(C);
    return ms;
}

} // namespace cklab::cuda
