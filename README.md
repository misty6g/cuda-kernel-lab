# CUDA Kernel Lab

[![CI](https://github.com/misty6g/cuda-kernel-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/misty6g/cuda-kernel-lab/actions/workflows/ci.yml)

A small, runnable **CUDA / systems microbenchmark lab**: SAXPY, STREAM-style copy,
shared-memory reduction, and **naive vs tiled GEMM**, each checked against a CPU
gold reference.

This is not a cuBLAS wrapper and not a README-only sketch. The kernels, the host
algorithm model, the timers, and the tests are real code. GitHub Actions has no
GPU, so CI builds the **host path** (`-DCKLAB_ENABLE_CUDA=OFF`) and runs unit
tests plus a tiny verified bench. On a machine with the CUDA toolkit, the same
CLI drives the device kernels.

Author: **Gyan Mistry** ([misty6g](https://github.com/misty6g)). Written as a
portfolio piece for NVIDIA software / systems / AV internship resumes.

## What it measures

| Op | What runs | Metric | Why it is in a systems lab |
| --- | --- | --- | --- |
| `copy` | Device/host memcpy-style loop | GB/s | Peak-ish traffic; baseline for other kernels |
| `saxpy` | `y = a*x + y`, grid-stride | GB/s | Classic memory-bound kernel (3 traffic / 2 FLOPs) |
| `reduce` | Shared-memory tree + host finish of partials | GB/s | Divergence, `__syncthreads`, float associativity |
| `gemm` | Naive global-only vs 16×16 tiled shared-memory | GFLOPS | Arithmetic intensity, coalescing, tile remainders |

Naive GEMM is global-memory bound: each multiply-add reloads `A` and `B`. Tiled
GEMM stages `TILE×TILE` panels in shared memory so each loaded value is reused
`TILE` times. That is the same memory-hierarchy idea behind production GEMM
(just not Tensor Cores, swizzle, or `ldmatrix`).

Arithmetic intensity for `C[M,N] = A[M,K] B[K,N]`:

```
FLOPs  = 2 M N K
Bytes  ≈ 4 (M K + K N + M N)
```

For square `N`, intensity grows as `O(N)`. Tiling is how you actually feed the
ALUs instead of the memory pipe.

## Repository layout

```
include/cklab/     public headers (CPU refs, verify, bench, CUDA API)
src/cpu.cpp        host SAXPY / reduce / naive + tiled GEMM
src/verify.cpp     allclose
src/bench.cpp      timers, GB/s + GFLOPS, verify wiring
src/main.cpp       CLI
src/cuda_kernels.cu
src/cuda_stub.cpp  linked when nvcc is not used
tests/test_host.cpp
.github/workflows/ci.yml
```

```
                 +------------------+
  CLI (cklab) -> | bench harness    | --verify--> CPU gold
                 +--------+---------+
                          |
              +-----------+-----------+
              |                       |
        host algorithms         CUDA kernels
        (CI + no-GPU laptops)   (nvcc + device)
              |                       |
              +-----------+-----------+
                          |
                    same numeric ops
                    (tile size, grid-stride, tree reduce)
```

The host tiled GEMM is the **same algorithm** as the CUDA kernel (panel loads,
zero-pad remainders, accumulate a tile of `C`). Tests compare it to a naive
triple loop, including shapes that are not multiples of 16.

## Build

Needs CMake ≥ 3.18 and a C++17 compiler. CUDA is optional.

### CPU-only (CI, no GPU, no toolkit)

```bash
cmake -S . -B build -DCKLAB_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or: `make cpu && make test`

This is the path GitHub Actions runs. It does **not** compile `.cu` files.

### CUDA (toolkit on PATH)

```bash
cmake -S . -B build -DCKLAB_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cklab --mode cuda --n 1024 --reps 20
```

Override arch if needed:

```bash
cmake -S . -B build -DCKLAB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
```

Default `CMAKE_CUDA_ARCHITECTURES` is `75` (Turing). `auto` mode uses CUDA when
the binary was built with nvcc **and** `cudaGetDeviceCount() > 0`; otherwise it
falls back to host.

`--mode cuda` on a host-only binary exits with an error. That is intentional.

## Run

```bash
./build/cklab --help
./build/cklab --mode host --n 256 --vec 1048576 --reps 8 --op all
./build/cklab --mode host --op gemm --n 128 --m 96 --k 160
./build/test_cklab
```

If you have a GPU and a CUDA build:

```bash
./build/cklab --mode cuda --n 2048 --reps 20 --op all
```

Optional Nsight (not required, not faked here):

```bash
ncu --set full ./build/cklab --mode cuda --op gemm --n 1024 --reps 5
```

## Sample output

Captured from the **host path** in this repo (no GPU in the build environment).
Absolute GB/s and GFLOPS are machine-dependent; the table shape and `PASS`
column are what you should see.

```
CUDA Kernel Lab
  backend : host
  device  : CPU (gold reference + tiled algorithm model)
  n       : 256   GEMM 256x256 * 256x256   vec 1048576
  reps    : 8 (mean; 1 warmup excluded)
  timers  : std::chrono::steady_clock (host algorithms)

op         variant          time(ms)  metric            verify
---------  -------------  ----------  ----------------  ------
copy       host               0.2459  34.11 GB/s        PASS
saxpy      host               0.2547  49.41 GB/s        PASS
reduce     sequential         0.5267  7.96 GB/s         PASS
reduce     tree-model         0.5090  8.24 GB/s         PASS
gemm       naive             14.8688  2.26 GFLOPS       PASS
gemm       tiled              7.0741  4.74 GFLOPS       PASS

All checks passed.
```

Numbers above are from one Linux CPU run in this repo (no GPU). Re-run locally;
GB/s and GFLOPS move with the machine. On CUDA builds, the same table is filled
from `cudaEvent` kernel time — host-to-device copies are not counted.

## Tests

`tests/test_host.cpp` is a dependency-free C++17 runner:

- SAXPY identities (`a=0`, `a=2`) and `n=0`
- Sequential vs tree-model reduction (exact on integer-valued floats)
- GEMM identity and a 2×2 closed form
- Tiled vs naive on remainder shapes: `17×13×19`, `7×9×5`, empty dims
- `allclose` mismatch detection
- Full host bench smoke
- CUDA query: stub throws if the binary is host-only; if a device exists, the
  same tests launch the real kernels

CI command: `ctest --test-dir build --output-on-failure`.

## Kernel notes (what a reviewer should look at)

- **Grid-stride loops** on 1D kernels so launch config is not tied to `n`.
- **RAII `DeviceBuffer`** in `cuda_kernels.cu`: `cudaMalloc` / `cudaFree`,
  explicit H2D and D2H, `cudaGetLastError` after every launch. Bench path
  times kernels with `cudaEvent` (copies excluded).
- **Reduction**: per-block shared-memory tree, power-of-two block size (256),
  partials finished on the host. Production code would call `cub::DeviceReduce`;
  this lab shows the teaching kernel, not a fake CUB clone.
- **Tiled GEMM**: bounds-checked panel loads (zero fill), `__syncthreads`
  before and after the k-inner product, `#pragma unroll` on the tile MAC.
- **Correctness first**: every bench row can `--verify` against the CPU gold
  with `atol + rtol * |ref|`. GEMM atol scales with `K`.

## What this is not

- Not a substitute for cuBLAS / CUTLASS / Tensor Core pipelines.
- Not claiming occupancy, roofline, or AV-stack numbers we did not measure.
- CI does not compile CUDA and does not run kernels. A green check means the
  host gold, the tiled algorithm model, and the CLI are correct.

## License

MIT. See [LICENSE](LICENSE).
