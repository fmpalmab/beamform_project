# Execution & Benchmark Memory Log: Beam Tracker Suite

**Date**: 2026-08-20  
**Machine Specs**:
- **OS**: Linux (x86_64)
- **CPU**: Intel Core i7-9700 @ 3.00GHz (8 physical cores)
- **GPU**: NVIDIA Quadro P1000 (Pascal `sm_61`, 4096 MiB VRAM, Driver 580.173.02, CUDA 13.0 / NVCC 12.4)
- **Compiler**: GCC 15.2.0, CMake 4.2.3
- **Python**: Python 3.14.4 virtual environment (`.venv`) with `numpy 2.5.2`, `scipy 1.18.0`, `matplotlib 3.11.1`

---

## 1. Environment & Build Setup

1. **Python Virtual Environment**:
   - `python3 -m venv .venv` failed due to missing Debian `ensurepip` on Python 3.14.
   - Resolved by using `/home/fernando/.local/bin/uv venv .venv` and installing `requirements.txt` (`numpy`, `scipy`, `matplotlib`).
2. **Build**:
   - CMake configuration: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBEAMFORMER_ENABLE_CUDA=ON`.
   - Compilation: Built all targets in Release mode with CUDA 12.4 support (`libbeamformer_core`, `libbeamformer_cuda_core`, and all test/benchmark binaries).

---

## 2. Test Execution & Verification

### 2.1 Python Unit Tests
- **Command**: `python3 -m unittest discover -s tests/python -p "test_*.py" -v`
- **Result**: **37 / 37 passed** in 5.895s.
- **Coverage**: Dashboard plotting, frequency mapping, trajectory kinematics, temporal integration.

### 2.2 CTest Suite (Full C++ / CUDA Suite)
- **Command**: `ctest --test-dir build --output-on-failure`
- **Result**: **21 / 21 passed** (100% pass rate).
- **Tests Evaluated**:
  1. `contract`: Passed
  2. `geometry`: Passed
  3. `synthetic_data`: Passed
  4. `cpu_beamformer`: Passed
  5. `beam_tracker`: Passed
  6. `cpu_opt_beam_tracker`: Passed
  7. `beam_tracker_opt`: Passed
  8. `beam_tracker_opt_v2`: Passed
  9. `temporal_integration`: Passed
  10. `quantization`: Passed
  11. `cuda_frame_contract`: Passed
  12. `cuda_point_source`: Passed
  13. `cuda_packed`: Passed
  14. `cuda_temporal_integration`: Passed
  15. `cuda_quantization`: Passed
  16. `cuda_frame_pipeline`: Passed
  17. `cuda_offline_runner`: Passed
  18. `cuda_tracker_v2`: Passed
  19. `cuda_beam_tracker_fused_warp_shuffle`: Passed
  20. `cuda_beam_tracker_v3`: Passed
  21. `benchmark_config`: Passed

### 2.3 Standalone Naive CPU Test Suite
- **Command**: `./build/beam_tracker_naive_cpu_test_suite --skip-benchmark`
- **Result**: **30 / 30 passed** across trivial, base, and complex test fixtures.

### 2.4 Astronomical Validation Suite (CHIME Catalog 2 Burst Injections)
- **Command**: `python3 tools/run_astronomical_validation.py --engine <engine> --burst all`
- **Engines Tested**:
  1. `cuda_v3`: **100% PASSED** (4/4 bursts: Canonical FRB20180916B, High DM 1205.4, Scattering-dominated, Faint narrow pulse).
  2. `cuda_fws` (Phase 4 Fused Warp-Shuffle): **100% PASSED** (4/4 bursts).
  3. `cpu_opt_v2`: **100% PASSED** (4/4 bursts).
  4. `cpu_naive`: **100% PASSED** (Canonical burst).
- **Validation Metrics**:
  - Dispersion Measure (DM) recovery error < 0.05 pc cm⁻³.
  - Spectro-temporal pulse fitting alignment.
  - Monotonic off-boresight beam pattern attenuation.
  - Coherent array scaling slope ~ 0.50.

---

## 3. Benchmarks & Performance Analysis

### 3.1 CUDA Tracker V3 Master Benchmark (`benchmark_cuda_tracker_v3`)
Comprehensive sweep across antenna sizes ($N_{\text{ant}} \in \{32, 64\}$) and OpenMP thread configurations ($T \in \{1, 2, 4, 8, 16, 24\}$), evaluating all 13 tracking engines:

| Antenna Count | Engine / Kernel | Latency (ms) | Speedup vs Phase 4 (FWS) | Speedup vs CPU Baseline (Single-Thread) |
| :---: | :--- | :---: | :---: | :---: |
| **64 ant** | **CUDA V3 Batched Kernel Only** | **17.31 ms** | **5.19x** | **21.03x** |
| 64 ant | **CUDA V3 Device Resident** | **19.83 ms** | **4.53x** | **18.36x** |
| 64 ant | **CUDA V3 Batched Stream (PCIe xfer)** | **57.17 ms** | **1.57x** | **6.37x** |
| 64 ant | **CUDA V3 Batched Graph (CUDA Graph)** | **56.98 ms** | **1.58x** | **6.39x** |
| 64 ant | **CUDA V3 Direct (PTX bfe / ILP)** | **67.58 ms** | **1.33x** | **5.39x** |
| 64 ant | CUDA Phase 4 FWS | 89.82 ms | 1.00x | 4.05x |
| 64 ant | CUDA V3 Stream (per-window) | 90.37 ms | 0.99x | 4.03x |
| 64 ant | CUDA Legacy TwoPass | 133.33 ms | 0.67x | 2.73x |
| 64 ant | CUDA Legacy Fused | 544.50 ms | 0.17x | 0.67x |
| 64 ant | CUDA Legacy WarpReduction | 587.44 ms | 0.15x | 0.62x |
| 64 ant | CPU Opt v2 (8 threads) | 51.04 ms | 1.76x | 7.01x |
| 64 ant | CPU Opt v1 (8 threads) | 53.02 ms | 1.69x | 6.75x |
| 64 ant | CPU Naive (Single-thread reference) | 351.84 ms | 0.26x | 1.00x |

*Key Takeaway*: In the previous version without V3, the batched CUDA kernel and Phase 4 FWS were top performers. With CUDA V3, **`v3_batched_kernel_only` achieves 17.31 ms** (down from 89.82 ms on Phase 4 FWS and 133.33 ms on TwoPass), delivering a **5.19x speedup** over Phase 4 FWS and **21.03x speedup** over CPU single-thread. Even with end-to-end host-to-device PCIe stream transfers included, **`v3_batched_stream` reaches 57.17 ms**.

### 3.2 CUDA Tracker V2 Benchmark (`benchmark_cuda_tracker_v2`)
- Generated full summary CSV (`benchmark_cuda_tracker_v2_summary.csv`), frame latency time-series, and numerical validation records.

### 3.3 CPU Optimized Tracker Benchmark (`benchmark_cpu_opt_beam_tracker`)
- Verified Bartlett DOA estimator: Initial open-loop angular error of $0.105$ was actively recovered down to $0.005$ across integration windows.

### 3.4 CPU Tracker Variants (`benchmark_beam_tracker_opt` & `benchmark_beam_tracker_opt_v2`)
- Verified bit-exact output equivalence across all 5,160,960 output cells: `assert(naive == v1 == v2)`.
- CPU Opt v2 delivered 2.43x speedup over CPU Naive on 64 antennas at 8 threads.

### 3.5 Direct Beamformer Offline Benchmark (`benchmark_cpu_cuda`)
- **Memory Limit Encountered**: Running standard 128-beam $\times$ 30,720-sample sweep exceeded 4GB GPU VRAM with `cudaMalloc: out of memory`.
- **Resolution**: Scaled sweep to $15,360$ samples and $32$ beams (0.93 GiB workspace), which completed successfully on the Quadro P1000 GPU and outputted performance/validation heatmaps.

---

---

## 5. CUDA Beam Tracker V4: Implementation, Validation & Benchmarks

### 5.1 Architecture & Design
- **Header**: [`include/beamformer/cuda_beam_tracker_v4.hpp`](file:///home/fernando/charts/beamform_project/include/beamformer/cuda_beam_tracker_v4.hpp)
- **Source**: [`src/cuda_beam_tracker_v4.cu`](file:///home/fernando/charts/beamform_project/src/cuda_beam_tracker_v4.cu)
- **Innovations Implemented**:
  1. **Warp Matrix Multiply-Accumulate (WMMA)** complex decomposition pipeline on Tensor Core architectures (`sm_70+`).
  2. **Mixed-Precision `__half2` Vector SIMD** arithmetic path (`__hfma2`, `__hadd2`, `__floats2half2_rn`).
  3. **8-way & 4-way Deep ILP Fused Warp-Shuffle** kernels with hardware PTX bit-field extraction (`bfe.s32`).
  4. **Asynchronous Multi-Stream Pipeline** (`cuda_beam_tracker_v4_stream`) with configurable stream slots and non-blocking scheduling.
  5. **`BatchedTrackerStreamV4`** supporting persistent device-resident execution and CUDA Graph acceleration.
  6. **Auto-tuning Runtime Dispatcher** mapping array geometries and compute capabilities.

### 5.2 Unit Test Verification
- **Binary**: [`build/test_cuda_beam_tracker_v4`](file:///home/fernando/charts/beamform_project/build/test_cuda_beam_tracker_v4)
- **Result**: **100% Passed (10/10 test blocks)**
  - V4 Deep ILP (Unroll 8, Unroll 4, Chunk 80/320): Passed
  - V4 Half2 Vector SIMD: Passed (FP16 numerical tolerance verified)
  - V4 Multi-Stream Pipeline: Passed
  - BatchedTrackerStreamV4 & CUDA Graph execution: Passed
  - Device-Resident Zero-Copy execution: Passed
  - Pinned memory allocation helpers: Passed
- **CTest**: 22/22 tests passed (including `test_cuda_beam_tracker_v4`).

### 5.3 Astronomical Validation (CHIME Catalog 2 Injections)
- **Report**: [`results/astronomical_validation/v4/astronomical_validation_report.json`](file:///home/fernando/charts/beamform_project/results/astronomical_validation/v4/astronomical_validation_report.json)
- **Plot**: [`results/astronomical_validation/v4/astronomical_validation_dashboard.png`](file:///home/fernando/charts/beamform_project/results/astronomical_validation/v4/astronomical_validation_dashboard.png)
- **Results Across All 4 Benchmarks**:
  - `FRB20180916B_canonical` (DM=348.82): Recovered DM=348.8, SNR=3498.3, Slope=0.51 -> **`PASS`**
  - `High_DM_Burst` (DM=1205.4): Recovered DM=1205.4, SNR=1458.6, Slope=0.50 -> **`PASS`**
  - `Scattering_Dominated` (DM=574.1): Recovered DM=574.1, SNR=897.7, Slope=0.49 -> **`PASS`**
  - `Faint_Narrow_Pulse` (DM=412.3): Recovered DM=412.3, SNR=550.5, Slope=0.52 -> **`PASS`**

### 5.4 Multi-Generation Benchmark Comparison (`benchmark_cuda_tracker_v4`)
Execution time on NVIDIA Quadro P1000 ($N_{\text{time}}=15360$, $N_{\text{freq}}=336$, spectra=$320$):

| Antenna Count | Engine / Kernel | Latency (ms) | Speedup vs Phase 4 (FWS) | Speedup vs CPU Naive |
| :---: | :--- | :---: | :---: | :---: |
| **64 ant** | **CUDA V4 Device Resident** | **14.54 ms** | **4.83x** | **24.33x** |
| **64 ant** | **CUDA V4 Batched Kernel Only** | **15.53 ms** | **4.52x** | **22.77x** |
| **64 ant** | **CUDA V4 Batched Stream** | **54.24 ms** | **1.29x** | **6.52x** |
| **64 ant** | **CUDA V4 Batched Graph** | **54.55 ms** | **1.28x** | **6.48x** |
| **64 ant** | **CUDA V4 Multi-Stream** | **59.07 ms** | **1.18x** | **5.98x** |
| **64 ant** | **CUDA V4 Deep ILP (Unroll 8)** | **63.33 ms** | **1.10x** | **5.58x** |
| **64 ant** | **CUDA V4 Deep ILP (Unroll 4)** | **63.53 ms** | **1.10x** | **5.56x** |
| 64 ant | CUDA V3 Device Resident | 14.49 ms | 4.85x | 24.42x |
| 64 ant | CUDA V3 Batched Kernel Only | 16.03 ms | 4.38x | 22.07x |
| 64 ant | CUDA V3 Batched Stream | 55.16 ms | 1.27x | 6.41x |
| 64 ant | CUDA Phase 4 FWS | 70.26 ms | 1.00x | 5.03x |
| 64 ant | CUDA TwoPass (Legacy) | 136.12 ms | 0.51x | 2.59x |
| 64 ant | CPU Opt v1 (8 threads) | 75.53 ms | 0.93x | 4.68x |
| 64 ant | CPU Opt v2 (8 threads) | 81.86 ms | 0.85x | 4.32x |
| 64 ant | CPU Naive (Single-thread reference) | 353.76 ms | 0.19x | 1.00x |
| **32 ant** | **CUDA V4 Device Resident** | **11.61 ms** | **4.08x** | **15.46x** |
| **32 ant** | **CUDA V4 Batched Kernel Only** | **12.44 ms** | **3.81x** | **14.43x** |
| **32 ant** | **CUDA V4 Batched Stream** | **38.25 ms** | **1.24x** | **4.69x** |
| **32 ant** | **CUDA V4 Batched Graph** | **38.68 ms** | **1.22x** | **4.63x** |
| **32 ant** | **CUDA V4 Deep ILP (Unroll 8)** | **40.64 ms** | **1.16x** | **4.41x** |
| 32 ant | CUDA V3 Device Resident | 12.76 ms | 3.71x | 14.06x |
| 32 ant | CUDA V3 Batched Kernel Only | 13.26 ms | 3.58x | 13.53x |
| 32 ant | CUDA Phase 4 FWS | 47.47 ms | 1.00x | 3.78x |
| 32 ant | CPU Opt v2 (8 threads) | 24.83 ms | 1.91x | 7.22x |
| 32 ant | CPU Naive (Single-thread reference) | 179.47 ms | 0.26x | 1.00x |
| **128 ant** | **CUDA V4 Device Resident** | **55.08 ms** | — | **12.72x** |
| **128 ant** | **CUDA V4 Batched Kernel Only** | **62.90 ms** | — | **11.14x** |
| **128 ant** | **CUDA V4 Multi-Stream** | **138.24 ms** | — | **5.07x** |
| **128 ant** | **CUDA V4 Batched Graph** | **138.50 ms** | — | **5.06x** |
| **128 ant** | **CUDA V4 Batched Stream** | **140.15 ms** | — | **5.00x** |
| **128 ant** | **CUDA V4 Block Reduction (Unroll 4)** | **150.69 ms** | — | **4.65x** |
| 128 ant | CPU Opt v2 (8 threads) | 140.00 ms | — | 5.00x |
| 128 ant | CPU Opt v1 (8 threads) | 146.47 ms | — | 4.78x |
| 128 ant | CPU Naive (Single-thread reference) | 700.86 ms | — | 1.00x |
| **256 ant** | **CUDA V4 Device Resident** | **115.29 ms** | — | **11.78x** |
| **256 ant** | **CUDA V4 Batched Kernel Only** | **134.23 ms** | — | **10.12x** |
| **256 ant** | **CUDA V4 Multi-Stream** | **277.32 ms** | — | **4.89x** |
| **256 ant** | **CUDA V4 Batched Stream** | **284.67 ms** | — | **4.77x** |
| **256 ant** | **CUDA V4 Batched Graph** | **287.91 ms** | — | **4.71x** |
| **256 ant** | **CUDA V4 Block Reduction (Unroll 8)** | **304.30 ms** | — | **4.46x** |
| **256 ant** | **CUDA V4 Block Reduction (Unroll 4)** | **307.38 ms** | — | **4.41x** |
| 256 ant | CPU Opt v2 (8 threads) | 190.37 ms | — | 7.13x |
| 256 ant | CPU Opt v1 (8 threads) | 191.53 ms | — | 7.09x |
| 256 ant | CPU Naive (Single-thread reference) | 1358.24 ms | — | 1.00x |

