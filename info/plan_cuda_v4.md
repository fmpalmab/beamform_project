# Implementation Plan: CUDA Beam Tracker V4 (Tensor Core Architecture)

## 1. Architectural Objectives
The primary goal of the V4 tracker is to break through the performance ceiling of the V3 scalar batched kernel by leveraging **Tensor Cores**, **mixed-precision**, and **asynchronous memory** pipelines. 

**Target Hardware:** NVIDIA Ampere (A100) / Hopper (H100) architectures and newer. (Grace Hopper Superchip compatibility).
**Fallback:** Turing (T4) and Volta (V100) for legacy Tensor Core support.

---

## 2. Kernel Design & Data Pipeline

### 2.1 Mixed-Precision Data Formats
- **Input Voltages:** Convert from FP32 to `__half2` (FP16 complex) or `int8_t`.
- **Beam Weights:** Store as `__half2` or `int8_t`.
- **Accumulation (Output):** Maintain `float` (FP32) accumulation to prevent loss of dynamic range during the reduction across antennas and time.

### 2.2 Tensor Core Mapping (WMMA)
The beamforming equation $Y(t, b) = \sum_{a=1}^{N_{\text{ant}}} W(b, a) \cdot X(a, t)$ will be structured as a matrix-matrix multiply.
- **Complex Decomposition:** Since WMMA natively supports real-valued math, the V4 kernel will execute the four block-level matrix multiplications:
  1. $Z_r = X_r W_r$
  2. $Z_r = Z_r - X_i W_i$
  3. $Z_i = X_r W_i$
  4. $Z_i = Z_i + X_i W_r$
- **WMMA PTX Instructions:** Utilize `nvcuda::wmma::mma_sync` to perform warp-level accumulation on $16 \times 16 \times 16$ or $32 \times 8 \times 16$ tile fragments.

### 2.3 Asynchronous Memory Subsystem
- Implement `cp.async.ca` (Asynchronous Copy with Cache-Allocate) to stream input voltage chunks directly from Global Memory into Shared Memory.
- Use `cp.async.commit_group` and `cp.async.wait_group` to establish a double-buffered pipeline: while the Tensor Cores compute Tile $N$, the memory controllers fetch Tile $N+1$.

### 2.4 Persistent Thread Spin-Waiting
- Port the `v3_device_resident` methodology to a permanent persistent-thread paradigm.
- Eliminate host-side PCIe launch latencies by having warps spin-wait on atomic flags updated by RDMA network packets (GPUDirect RDMA).

### 2.5 Block-Level Cooperative Reduction for Large Arrays ($N_{\text{ant}} \ge 128$)
Currently, the V3 and early V4 fallback kernels map $N_{\text{ant}}$ antennas to a single 32-thread warp. As the telescope expands to 128 or 256 antennas, this single-warp mapping creates severe register pressure (forcing threads to process 4-8 antennas each) and causes a collapse in SM occupancy.
- **V4 Design Pivot:** V4 will implement **Block-Level Reductions** using `__shared__` memory or Cooperative Groups for large antenna configurations.
- By mapping a block of 128 or 256 threads to a single time sample (e.g., 1 thread = 1 antenna), register pressure remains bounded at a constant low baseline (~20 registers/thread), ensuring 100% SM occupancy even at $N_{\text{ant}} = 256$.

---

## 3. Development Phases

### Phase 1: Precision Migration & Baseline
1. **Refactor Data Structures:** Create a mixed-precision conversion layer in the C++ host code to marshal data from FP32 sensors into `__half2`.
2. **Scalar FP16 Kernel:** Implement a baseline scalar V4 kernel that uses `__hadd2` and `__hfma2` instructions. Verify CHIME FRB validation parity with reduced precision.

### Phase 2: WMMA Tensor Core Integration
1. **cuBLAS Prototype:** Implement the complex decomposition using `cublasGemmEx` (with `CUBLAS_COMPUTE_32F` accumulation) as a correctness benchmark.
2. **Custom WMMA Kernel:** Write the custom `nvcuda::wmma` device kernel. Profile register pressure and shared memory bank conflicts.

### Phase 3: Asynchronous Pipeline & Tuning
1. **`cp.async` Integration:** Introduce the double-buffered shared memory pipeline using PTX asynchronous copy.
2. **Auto-Tuning Framework:** Integrate Kernel Tuner (or a custom CMake-time JIT sweep script) to automatically discover the optimal WMMA block sizing ($M, N, K$) and warp configurations for the host GPU.

---

## 4. Risks and Mitigation

- **Dynamic Range Loss:** Astronomical signals can have high dynamic range (e.g., strong RFI mixed with faint thermal noise). FP16 provides only ~11 bits of mantissa. 
  - *Mitigation:* Conduct rigorous astronomical validation (FRB injection sweeps) early in Phase 1 to ensure DM recovery and SNR are not degraded. Implement block-floating-point (BFP) scaling if necessary.
- **Hardware Compatibility:** Older GPUs (like the Quadro P1000 Pascal) do not have Tensor Cores.
  - *Mitigation:* V4 will compile conditionally via CMake (`#if __CUDA_ARCH__ >= 700`). The runtime dispatcher will automatically fall back to the V4 Block-Level Reductions or V3 Batched kernel on unsupported hardware.
