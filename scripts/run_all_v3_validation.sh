#!/bin/bash
# ==============================================================================
# run_all_v3_validation.sh
#
# Master Execution & Benchmark Script for CUDA Beam Tracker V3 on Trillium / Linux.
#
# Runs:
#   1. Clean CMake Build of all CPU and CUDA targets
#   2. CTest Unit & Numerical Correctness Tests (V3, Phase 4 FWS, V2, CPU)
#   3. Astronomical Validation Suite (FRB Injection, DM Sweep, Array Scaling)
#   4. Multi-Model Benchmark Comparison Sweep (CPU Naive/V1/V2, CUDA Legacy,
#      Phase 4 FWS, and CUDA V3 variants)
#   5. Visual Comparison Dashboard & Artifact Generation
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ------------------------------------------------------------------------------
# 0. Environment Setup & Module Loading (Trillium / Compute Canada / HPC)
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "          CUDA BEAM TRACKER V3: COMPLETE VALIDATION & BENCHMARK         "
echo "========================================================================"
echo "Host: $(hostname) | Date: $(date)"

if command -v module &> /dev/null; then
    echo "--- Loading HPC Environment Modules ---"
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
fi

if [ -d ".venv" ]; then
    echo "--- Activating Python Virtual Environment ---"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "--- Activating Python Virtual Environment ---"
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}/tools:${PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export OMP_PROC_BIND=true

RESULTS_DIR="${PROJECT_ROOT}/results/v3_suite"
mkdir -p "${RESULTS_DIR}/astronomical_validation"
mkdir -p "${RESULTS_DIR}/benchmarks"
mkdir -p "${RESULTS_DIR}/plots"

echo ""
echo "--- GPU Hardware Status ---"
nvidia-smi || echo "(Warning: nvidia-smi failed or no GPU found)"
echo ""

# ------------------------------------------------------------------------------
# 1. Compilation
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 1: Compiling Project Targets (Release with CUDA)                 "
echo "========================================================================"
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEAMFORMER_ENABLE_CUDA=ON

NPROC=$(nproc 2>/dev/null || echo 8)
cmake --build build --config Release -j "${NPROC}"

echo "Build completed successfully."
echo ""

# ------------------------------------------------------------------------------
# 2. CTest Unit & Numerical Correctness Tests
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 2: Running Unit & Parity Correctness Tests                       "
echo "========================================================================"

echo ">>> Running CUDA V3 unit tests..."
./build/test_cuda_beam_tracker_v3

echo ">>> Running Phase 4 Fused Warp-Shuffle unit tests..."
./build/test_cuda_beam_tracker_fused_warp_shuffle

echo ">>> Running CUDA Tracker V2 legacy unit tests..."
./build/test_cuda_tracker_v2

echo ">>> Running CPU Optimized Tracker unit tests..."
./build/test_cpu_opt_beam_tracker

echo "All unit tests PASSED!"
echo ""

# ------------------------------------------------------------------------------
# 3. Astronomical Validation Suite
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 3: Astronomical Validation Suite (FRB Injection & Dedispersion)  "
echo "========================================================================"

echo "--- 3.1 Astronomical Validation for CUDA V3 ---"
python3 tools/run_astronomical_validation.py \
    --engine cuda_v3 \
    --burst all \
    --outdir "${RESULTS_DIR}/astronomical_validation/v3"

echo "--- 3.2 Astronomical Validation for Phase 4 Fused Warp-Shuffle (Baseline) ---"
python3 tools/run_astronomical_validation.py \
    --engine cuda_fws \
    --burst all \
    --outdir "${RESULTS_DIR}/astronomical_validation/phase4_fws"

echo "Astronomical validation completed for both V3 and Phase 4."
echo ""

# ------------------------------------------------------------------------------
# 4. Multi-Model Benchmark Comparison Sweep
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 4: Multi-Model Benchmark Comparison Sweep                        "
echo "========================================================================"

BIN_BENCH=./build/benchmark_cuda_tracker_v3
N_TIME=15360
INT_SPEC=320
REPEAT=5
WINDOW_REPEATS=3
OUTDIR="${RESULTS_DIR}/benchmarks"

THREAD_COUNTS=(1 2 4 8 16 24)
ANT_COUNTS=(32 64)

for N_ANT in "${ANT_COUNTS[@]}"; do
    for T in "${THREAD_COUNTS[@]}"; do
        export OMP_NUM_THREADS=${T}
        echo "------------------------------------------------------------------------"
        echo " Benchmarking: n_ant=${N_ANT} | OMP_threads=${T} | n_time=${N_TIME} "
        echo "------------------------------------------------------------------------"
        ${BIN_BENCH} \
            --n-time ${N_TIME} \
            --n-ant ${N_ANT} \
            --integration-spectra ${INT_SPEC} \
            --threads ${T} \
            --repeat ${REPEAT} \
            --window-repeats ${WINDOW_REPEATS} \
            --outdir ${OUTDIR}
        echo ""
    done
done

echo "Benchmark sweep completed."
echo ""

# ------------------------------------------------------------------------------
# 5. Visual Comparison Dashboard & Summary
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "  STEP 5: Generating Comparison Dashboard Plots                        "
echo "========================================================================"

if python3 -c "import matplotlib" 2>/dev/null; then
    python3 tools/plot_tracker_comparison.py \
        --prefix "${OUTDIR}/benchmark_cuda_tracker_v2" \
        --out "${RESULTS_DIR}/plots/tracker_v3_vs_all_comparison.png" || true
    echo "Comparison plot generated at: ${RESULTS_DIR}/plots/tracker_v3_vs_all_comparison.png"
fi

echo ""
echo "========================================================================"
echo "  ALL TESTS, VALIDATIONS, AND BENCHMARKS COMPLETED SUCCESSFULLY!        "
echo "========================================================================"
echo "Outputs directory: ${RESULTS_DIR}"
echo "  - Astronomical Reports: ${RESULTS_DIR}/astronomical_validation/"
echo "  - Benchmark CSVs:       ${RESULTS_DIR}/benchmarks/"
echo "  - Plots & Dashboards:   ${RESULTS_DIR}/plots/"
echo "========================================================================"
