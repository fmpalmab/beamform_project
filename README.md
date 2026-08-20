# CHARTS Voltage Beamformer & Dynamic Beam Tracker

High-performance radio astronomy voltage beamformer and dynamic beam tracking engine for the Canadian Hydrogen Observatory and Radio-transient Detector System (CHARTS / CHIME).

Provides direct fixed-grid beamforming, real-time dynamic celestial source tracking, adaptive direction-of-arrival (DOA) estimation, and an end-to-end astronomical validation pipeline with CPU (OpenMP / SIMD) and GPU (CUDA) acceleration.

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Beamforming & Tracking Engines](#beamforming--tracking-engines)
   - [Direct Fixed-Grid Beamformer](#1-direct-fixed-grid-beamformer)
   - [Dynamic Beam Tracker](#2-dynamic-beam-tracker)
   - [Adaptive Direction-of-Arrival (DOA) Estimator](#3-adaptive-direction-of-arrival-doa-estimator)
   - [CUDA High-Throughput Tracker Suite (V1–V3)](#4-cuda-high-throughput-tracker-suite-v1v3)
3. [Repository Layout](#repository-layout)
4. [Data Contracts & Layouts](#data-contracts--layouts)
5. [Build Instructions](#build-instructions)
6. [Testing & Verification](#testing--verification)
7. [Benchmarks & Performance Sweeps](#benchmarks--performance-sweeps)
8. [Astronomical Validation Suite](#astronomical-validation-suite)
9. [CLI Tools & Standalone Utilities](#cli-tools--standalone-utilities)
10. [Visualization & Plotting Utilities](#visualization--plotting-utilities)

---

## Overview & Architecture

The CHARTS Voltage Beamformer processes channelized raw baseband voltage streams from RFSoC digitizers and synthesizes high-gain directional beams across antenna arrays ($N_{\text{ant}} \in \{32, 64\}$) and frequency bands ($N_{\text{freq}} = 336$ channels/shard, $672$ channels full-band).

### Key Capabilities

- **Inline Signed $\text{int4}$ Complex Unpacking**: Direct real-time arithmetic on packed nibbles ($[-8, 7]$) without intermediate full-tensor expansions.
- **Fixed-Grid & Dynamic Tracking**: Supports static pre-computed beam grids (FFT-bin centers, line grids) and continuous dynamic tracking of moving celestial sources (e.g. Fast Radio Bursts, pulsars).
- **Sub-Millisecond Real-Time Target**: Achieves the hard real-time latency objective ($\le 0.5\text{ ms}$ per integration window) across modern multi-core CPUs and NVIDIA GPUs.
- **Astronomical Physical Parity**: Validated against physical radio propagation models, point-source response geometry, dispersion measures ($\text{DM}$ sweeps up to $1000\text{ pc cm}^{-3}$), and real CHIME Catalog 2 benchmarks.

---

## Beamforming & Tracking Engines

```
                           Raw Packed Voltage [T][F][E]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
   Direct Fixed-Grid Beamformer                          Dynamic Beam Tracker
   (Multi-Beam Fixed Directions)                 (Dynamic Direction of Moving Source)
             │                                                     │
   ┌─────────┴─────────┐                         ┌─────────────────┴─────────────────┐
   ▼                   ▼                         ▼                                   ▼
CPU Direct         CUDA Direct             CPU Trackers                        CUDA Trackers
(OpenMP Loop)    (Tiled / Int8)          - Naive Baseline                   - Legacy TwoPass / Fused
                                         - Optimized v1 / v2                - Phase 4 Fused Warp-Shuffle (FWS)
                                         - Adaptive Covariance (DOA)        - V3 Direct / Streaming / Graph
```

### 1. Direct Fixed-Grid Beamformer
- **CPU Reference**: Serial and OpenMP-parallelized baseline calculating $I[t][f][b] = \left| \sum_a x[t][f][a] w^*[b][f][a] \right|^2$.
- **CUDA Direct / Tiled**: GPU acceleration synthesizing up to 128 simultaneous beams with optional fused temporal integration ($10$ or $320$ spectra) and 8-bit quantization (`int8` output).

### 2. Dynamic Beam Tracker
- **Naive CPU Tracker**: Baseline dynamic beamformer computing updated geometric phase weights per integration cadence ($W = \text{ceil}(T / \Delta T_{\text{int}})$).
- **Optimized CPU Tracker (v1 & v2)**: Vectorized OpenMP implementation with NUMA first-touch page placement, zero-redundancy direction projection, and pointer aliasing optimizations delivering bit-exact parity with the naive baseline.

### 3. Adaptive Direction-of-Arrival (DOA) Estimator
- **Covariance-Based Search**: Computes sample spatial covariance matrix $R[f] = \frac{1}{K} \sum_{t} x[t][f] x[t][f]^H$ with exponential forgetting factors ($\alpha$).
- **Bartlett & Capon (MVDR)**: Spatial spectrum scanning with spatial smoothing for coherent multipath suppression and sub-beamwidth quadratic interpolation.

### 4. CUDA High-Throughput Tracker Suite (V1–V3)
- **Legacy V2 Suite**: `TwoPass`, `Fused`, and `WarpReduction` kernels.
- **Phase 4 Fused Warp-Shuffle (FWS)**: Fused tile reduction via intra-warp register exchange (`__shfl_down_sync`) and shared-memory inter-warp accumulation.
- **CUDA V3 Suite**:
  - *Direct ILP*: Instruction-level parallelism ($T_{\text{unroll}} \in \{2, 4\}$) with PTX bit-field extraction (`bfe.s32`).
  - *Multi-Stream Pipelining*: Double-buffered asynchronous compute/transfer overlaps.
  - *CUDA Graph Batched Streaming*: Zero-driver-overhead GPU execution for latency-critical streaming.
  - *Device-Resident Mode*: In-place execution for pipelines where voltages remain in GPU VRAM.

---

## Repository Layout

The codebase is organized into modular directories:

```
beamform_project/
├── CMakeLists.txt                    # CMake build configuration (CPU & CUDA)
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
│
├── include/                          # Public C++ Header API
│   └── beamformer/                   #   Header files for formats, config, trackers, CUDA
│
├── src/                              # Core C++ & CUDA Implementations
│   ├── *.cpp                         #   CPU beamformers, geometry, IO, weights
│   ├── *.cu                          #   CUDA kernels, offline runner, quantized int8
│   └── *.hpp                         #   Internal headers
│
├── tools/                            # Production Utilities & CLI Tools
│   ├── beam_tracker_cpu.cpp          #   CLI: Standalone CPU beam tracker
│   ├── beamformer_cpu.cpp            #   CLI: Standalone CPU direct beamformer
│   ├── beamformer_cuda.cpp           #   CLI: Standalone CUDA direct beamformer
│   ├── generate_fake_data.cpp        #   CLI: Synthetic voltage stream generator
│   ├── generate_weights.cpp          #   CLI: Geometric complex weights generator
│   ├── run_tracker_stream.cpp        #   CLI: Offline tracker streaming bridge
│   ├── plot_results.py               #   Python: Beam patterns & full-sky response
│   ├── plot_tracker_results.py       #   Python: Dynamic tracker trajectory visualizer
│   ├── plot_tracker_comparison.py    #   Python: CPU vs CUDA comparison dashboard
│   ├── plot_cpu_opt_beam_tracker.py  #   Python: Adaptive tracker DOA diagnostics
│   ├── plot_benchmark.py             #   Python: Throughput heatmap visualizer
│   ├── run_astronomical_validation.py#   Python: Master astronomical validation CLI
│   ├── run_temporal_integration_test.py # Python: Temporal integration harness
│   └── astronomical_validation/      #   Python Package: FRB catalog, dispersion, fitter
│
├── benchmarks/                       # Dedicated C++ Benchmark Drivers
│   ├── benchmark_cpu_opt_beam_tracker.cpp
│   ├── benchmark_beam_tracker_opt.cpp
│   ├── benchmark_beam_tracker_opt_v2.cpp
│   ├── benchmark_cpu_cuda.cpp
│   ├── benchmark_cuda_quantized.cpp
│   ├── benchmark_cuda_tracker_v2.cpp
│   └── benchmark_cuda_tracker_v3.cpp
│
├── tests/                            # Structured Test Suite
│   ├── cpu/                          #   CPU unit, regression & integration tests
│   │   ├── test_contract.cpp
│   │   ├── test_geometry.cpp
│   │   ├── test_synthetic_data.cpp
│   │   ├── test_cpu_beamformer.cpp
│   │   ├── test_beam_tracker.cpp
│   │   ├── test_cpu_opt_beam_tracker.cpp
│   │   ├── test_beam_tracker_opt.cpp
│   │   ├── test_beam_tracker_opt_v2.cpp
│   │   ├── test_temporal_integration.cpp
│   │   ├── test_quantization.cpp
│   │   ├── test_cuda_frame_contract.cpp
│   │   └── beam_tracker_naive_cpu_test_suite.cpp
│   ├── cuda/                         #   CUDA kernel, pipeline & numerical tests
│   │   ├── test_cuda_point_source.cpp
│   │   ├── test_cuda_packed.cu
│   │   ├── test_cuda_temporal_integration.cpp
│   │   ├── test_cuda_quantization.cpp
│   │   ├── test_cuda_frame_pipeline.cpp
│   │   ├── test_cuda_offline_runner.cpp
│   │   ├── test_cuda_tracker_v2.cpp
│   │   ├── test_cuda_beam_tracker_fused_warp_shuffle.cpp
│   │   └── test_cuda_beam_tracker_v3.cpp
│   └── python/                       #   Python unit tests
│       ├── test_plot_benchmark.py
│       ├── test_plot_results.py
│       ├── test_plot_tracker_results.py
│       └── test_temporal_integration_script.py
│
├── scripts/                          # Execution Runners & Cluster Batch Scripts
│   ├── run_all_v3_validation.sh      #   Master end-to-end build, test & sweep script
│   ├── run_beam_tracker_naive_cpu_tests.sh
│   ├── run_cuda_tracker_benchmarks.sh
│   ├── run_tracker_comparison_benchmarks.sh
│   └── slurm/                        #   HPC SLURM submission scripts
│       ├── submit_astronomical_validation.sh
│       ├── submit_benchmark.sh
│       ├── submit_benchmark_v2.sh
│       ├── submit_gpu_benchmark.sh
│       ├── submit_tracker_comparision.sh
│       └── submit_v3_complete_suite.sh
│
├── info/                             # Architectural Specifications & Engineering Runbooks
└── results/                          # Output binaries, validation JSONs, and plot figures
```

---

## Data Contracts & Layouts

### Binary Shards & Memory Dimensions
- **Voltage Shard**: Packed signed `int4` byte stream of size $[T][F_{\text{local}}=336][N_{\text{ant}}]$ bytes.
  - Low nibble (bits 0–3): Signed Real component $\in [-8, 7]$
  - High nibble (bits 4–7): Signed Imaginary component $\in [-8, 7]$
- **Weights**: Complex single-precision floats $\{ \text{real}, \text{imag} \}$ stored as $[N_{\text{beams}}][F][N_{\text{ant}}]$ or $[F][B_{\text{tile}}][N_{\text{ant}}][B_{\text{local}}]$.
- **Intensity Output**: Single-precision `float32` power values $[T][F][N_{\text{beams}}]$ or $[W][F][N_{\text{beams}}]$ where $W = \text{ceil}(T / \Delta T_{\text{int}})$.
- **Quantized `int8`**: $[W][F][N_{\text{beams}}]$ bytes with companion per-channel `(offset, scale)` float sidecars.

---

## Build Instructions

### Prerequisites
- C++17 compliant compiler (`g++` $\ge 11$, `clang++` $\ge 14$, or `MSVC` $\ge 2019$)
- CMake $\ge 3.24$
- OpenMP runtime
- NVIDIA CUDA Toolkit $\ge 12.0$ (optional, automatically detected)
- Python $\ge 3.9$ with `numpy`, `scipy`, `matplotlib`

### Building with CMake

```bash
# 1. Configure release build with CUDA enabled
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEAMFORMER_ENABLE_CUDA=ON

# 2. Compile all targets
cmake --build build --config Release -j
```

To build a CPU-only target (e.g. for development or CI without GPUs):
```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DBEAMFORMER_ENABLE_CUDA=OFF
cmake --build build -j
```

---

## Testing & Verification

### Running CTest (C++ Unit Tests)
```bash
# Run full CTest suite
ctest --test-dir build --output-on-failure

# Run only CPU tracker tests
ctest --test-dir build -R "cpu_opt_beam_tracker|beam_tracker" --output-on-failure

# Run only CUDA kernel tests
ctest --test-dir build -R "cuda_" --output-on-failure
```

### Running Python Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Benchmarks & Performance Sweeps

### Running Standalone C++ Benchmarks
```bash
# Run CUDA Tracker V3 multi-engine benchmark
./build/benchmark_cuda_tracker_v3 \
    --n-time 15360 --n-ant 64 --integration-spectra 320 \
    --threads 24 --repeat 5 --window-repeats 3 --outdir results/benchmarks

# Run CPU tracker three-way comparison (Naive vs Opt v1 vs Opt v2)
./build/benchmark_beam_tracker_opt_v2 \
    --n-time 15360 --n-ant 64 --integration-spectra 320 --threads 24
```

### HPC / SLURM Batch Execution

```bash
# ==============================================================================
# MASTER RUNNER: ALL Tests, Astronomical Validations & Benchmarks in 1 Job
# ==============================================================================
# Submit master Slurm job on Trillium (runs everything under identical runtime conditions)
sbatch submit_everything.sh

# Or run interactively (in a debugjob or compute node session):
bash scripts/run_everything.sh

# Quick smoke test run (reduced thread counts & repetitions):
sbatch submit_everything.sh --quick

# All results are aggregated into a single directory (e.g. results/everything_<TIMESTAMP>)
# and packaged as a .tar.gz archive. Retrieve in a single command:
#   scp -r user@trillium.scinet.utoronto.ca:/path/to/results/everything_latest ./
# ==============================================================================

# Submit full V3 validation & benchmark suite on Trillium / HPC
sbatch scripts/slurm/submit_v3_complete_suite.sh

# Submit GPU tracker comparison sweep
sbatch scripts/slurm/submit_tracker_comparision.sh

# Submit CPU thread-scaling benchmark
sbatch scripts/slurm/submit_benchmark_v2.sh
```

---

## Astronomical Validation Suite

The project includes an astronomical validation framework to verify tracker outputs against physical transients and FRB signatures.

```bash
# Run astronomical validation across all bursts for CUDA V3
python tools/run_astronomical_validation.py \
    --engine cuda_v3 \
    --burst all \
    --outdir results/astronomical_validation/v3

# Run astronomical validation for Phase 4 Fused Warp-Shuffle
python tools/run_astronomical_validation.py \
    --engine cuda_fws \
    --burst all \
    --outdir results/astronomical_validation/phase4_fws
```

### Validation Checks
1. **Dispersion Measure ($\text{DM}$) Recovery**: Injects dispersion delay $\Delta t = k_{\text{DM}} \cdot \text{DM} \cdot (\nu^{-2} - \nu_{\text{top}}^{-2})$ and verifies peak $\text{SNR}$ recovery after dedispersion.
2. **Spectro-Temporal Feature Preservation**: Gaussian pulse width ($\sigma_t$), spectral index ($\gamma$), and center frequency alignment.
3. **Array Scaling Gain**: Coherent gain scaling verification across array transitions ($N=32 \to 64$).

---

## CLI Tools & Standalone Utilities

### 1. Synthetic Voltage Generator (`generate_fake_data`)
```bash
# Stationary point source
./build/generate_fake_data \
    --type point-source --n-time 15360 --n-ant 64 \
    --source-l 0.04 --source-m 0.0 --amplitude 4 \
    --output point_source.bin

# Moving celestial source for beam tracker
./build/generate_fake_data \
    --type moving-point-source --n-time 15360 --n-ant 64 \
    --track-l0 0.0 --track-m0 0.0 --dl-per-sample 1e-4 --dm-per-sample 0.0 \
    --output moving_source.bin
```

### 2. Weight Generator (`generate_weights`)
```bash
# Line grid weights for 64 antennas and 5 beams
./build/generate_weights \
    --n-ant 64 --n-beams 5 --beam-grid line \
    --output weights.bin
```

### 3. Standalone CPU & CUDA Beamformers
```bash
# Execute CPU beamformer
./build/beamformer_cpu \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 --output cpu_intensity.bin

# Execute CUDA beamformer
./build/beamformer_cuda \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 --output cuda_intensity.bin
```

### 4. Standalone CPU Beam Tracker
```bash
./build/beam_tracker_cpu \
    --input moving_source.bin \
    --n-time 15360 --n-ant 64 --integration-spectra 320 \
    --track-l0 0.0 --track-m0 0.0 --dl-per-sample 1e-4 \
    --output tracker_intensity.bin --metrics metrics.csv
```

---

## Visualization & Plotting Utilities

### Beamformer Response & Sky Maps (`plot_results.py`)
```bash
# Generate full-sky array beam coverage
python tools/plot_results.py \
    --n-ant 32 --n-beams 32 \
    --beam-grid legacy-rectangular \
    --sky-output results/beam_grid_32_full_sky.png

# Compare CPU vs CUDA numerical outputs
python tools/plot_results.py \
    --input results/cpu_intensity.bin --label CPU \
    --compare results/cuda_intensity.bin --compare-label CUDA \
    --n-time 32 --n-freq 672 --n-ant 64 --n-beams 5 \
    --beam-grid line --synthetic-type point-source \
    --output results/cpu_cuda_validation.png
```

### Dynamic Tracker Dashboard (`plot_tracker_results.py`)
```bash
python tools/plot_tracker_results.py \
    --input results/tracker_intensity.bin \
    --output results/tracker_dashboard.png \
    --n-time 15360 --n-ant 64 --integration-spectra 320 \
    --track-l0 0.0 --track-m0 0.0 --dl-per-sample 1e-4
```

### Engine Comparison Visualizer (`plot_tracker_comparison.py`)
```bash
python tools/plot_tracker_comparison.py \
    --input-prefix results/benchmarks/benchmark_cuda_tracker_v2 \
    --output results/plots/tracker_comparison_dashboard.png \
    --budget-ms 0.5
```
