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

## 4. Artifact & Visualization Manifest

All benchmark outputs, JSON reports, CSVs, and visualization dashboards are organized in:
`results/everything_latest/` (`results/everything_20260820_154418Z/`)

- **Reports**:
  - `SUMMARY.md`: Unified markdown summary report.
  - `SUMMARY.txt`: Unified terminal summary report.
  - `env_info.txt`: System topology, CPU, and GPU diagnostic log.
- **Dashboards & Visualizations** (`plots/`):
  - `tracker_cpu_vs_gpu_dashboard.png`: Multi-engine dynamic tracker latency & speedup comparison.
  - `cpu_opt_tracker_dashboard.png`: CPU adaptive tracker DOA recovery & per-frame latency.
  - `direct_beamformer_benchmark_performance.png`: Fixed-grid beamformer throughput curves.
  - `direct_beamformer_benchmark_gpu_time_heatmaps.png`: Execution time heatmaps.
  - `direct_beamformer_benchmark_validation.png`: Fixed-grid CPU vs CUDA numerical parity.
  - `astronomical_validation_v3.png`: CHIME FRB injection, DM recovery, and array scaling for V3.
  - `astronomical_validation_phase4_fws.png`: Astronomical validation dashboard for Phase 4 FWS.
  - `astronomical_validation_cpu_opt_v2.png`: Astronomical validation dashboard for CPU Opt v2.
  - `astronomical_validation_cpu_naive.png`: Astronomical validation dashboard for CPU Naive.
- **Archive**:
  - `results/everything_20260820_154418Z.tar.gz` (compressed archive for single-command retrieval).
