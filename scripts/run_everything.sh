#!/usr/bin/env bash
# ==============================================================================
# run_everything.sh
#
# Master Test, Astronomical Validation, and Benchmark Runner for CHARTS Voltage
# Beamformer & Dynamic Beam Tracker.
#
# Runs:
#   1. Environment & Hardware Diagnostics (GPU, CPU topology, OS, modules)
#   2. Clean CMake Build of all CPU and CUDA targets in Release mode
#   3. Full Unit & Correctness Test Suite (CTest, Naive CPU suite, Python tests)
#   4. Astronomical Validation Suite (CHIME FRB injection, DM sweep, array scaling)
#   5. Comprehensive Benchmark Sweeps:
#        - CUDA Tracker V3 multi-model sweep (11+ engines)
#        - CUDA Tracker V2 comparison sweep (CSV summaries & frame latencies)
#        - CPU Opt tracker sweep (per-window latency vs 0.5 ms target)
#        - CPU Naive vs Opt v1 vs Opt v2 sweeps
#        - Naive CPU matrix sweep
#        - CPU vs CUDA offline direct beamformer sweep
#   6. Multi-Panel Visualization Dashboards & Plots
#   7. Automated Unified Summary Generation (SUMMARY.md & SUMMARY.txt)
#   8. Archive Packaging & One-Command SCP Retrieval
#
# Usage:
#   bash scripts/run_everything.sh                   # Full standard sweep
#   bash scripts/run_everything.sh --quick           # Faster sweep (reduced threads/repeats)
#   bash scripts/run_everything.sh --skip-bench      # Build, correctness & astro validation only
#   bash scripts/run_everything.sh --skip-tests      # Build & benchmarks only
#   bash scripts/run_everything.sh --outdir DIR      # Custom output directory
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ------------------------------------------------------------------------------
# 0. Configuration & CLI Arguments
# ------------------------------------------------------------------------------
QUICK=0
REBUILD=0
SKIP_BENCH=0
SKIP_TESTS=0
CUSTOM_OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK=1; shift ;;
        --rebuild)      REBUILD=1; shift ;;
        --skip-bench)   SKIP_BENCH=1; shift ;;
        --skip-tests)   SKIP_TESTS=1; shift ;;
        --outdir)       CUSTOM_OUTDIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--rebuild] [--skip-bench] [--skip-tests] [--outdir DIR]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
if [[ -n "${CUSTOM_OUTDIR}" ]]; then
    RESULTS_DIR="${CUSTOM_OUTDIR}"
else
    RESULTS_DIR="${PROJECT_ROOT}/results/everything_${TIMESTAMP}"
fi

# Directory hierarchy
mkdir -p "${RESULTS_DIR}/tests"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v3"
mkdir -p "${RESULTS_DIR}/astronomical_validation/phase4_fws"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_naive"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v3"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v2"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_opt_tracker"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_naive_opt_v1"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_naive_opt_v2"
mkdir -p "${RESULTS_DIR}/benchmarks/naive_cpu_matrix"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_cuda_offline"
mkdir -p "${RESULTS_DIR}/plots"

echo "========================================================================"
echo "      CHARTS VOLTAGE BEAMFORMER & TRACKER: COMPLETE SUITE RUNNER        "
echo "========================================================================"
echo "Project Root : ${PROJECT_ROOT}"
echo "Results Dir  : ${RESULTS_DIR}"
echo "Start Time   : $(date -u)"
echo "Host Node    : $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Slurm Job ID : ${SLURM_JOB_ID}"
fi
echo "========================================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. Environment & Hardware Diagnostics
# ------------------------------------------------------------------------------
echo "--- [1/7] Capturing System Diagnostics & Hardware Environment ---"

if command -v module &> /dev/null; then
    echo "Loading HPC modules (StdEnv/2023, gcc/12.3, cuda/12.6, python/3.11)..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
fi

if [[ -d ".venv" ]]; then
    echo "Activating Python venv (.venv)..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "Activating Python venv (venv)..."
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export OMP_PROC_BIND=true

ENV_LOG="${RESULTS_DIR}/env_info.txt"
{
    echo "Host: $(hostname)"
    echo "Date: $(date -u)"
    echo "Kernel: $(uname -a)"
    echo "Working Directory: $(pwd)"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
        echo "SLURM_NODELIST: ${SLURM_NODELIST:-}"
        echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
    fi
    echo ""
    echo "=== CPU Topology (lscpu) ==="
    lscpu 2>/dev/null || echo "(lscpu unavailable)"
    echo ""
    echo "=== Memory (free -h) ==="
    free -h 2>/dev/null || echo "(free unavailable)"
    echo ""
    echo "=== GPU Status (nvidia-smi) ==="
    nvidia-smi 2>/dev/null || echo "(nvidia-smi unavailable or no GPU present)"
    echo ""
    echo "=== Compilers & Tooling ==="
    cmake --version 2>/dev/null | head -n 1 || true
    g++ --version 2>/dev/null | head -n 1 || true
    nvcc --version 2>/dev/null | tail -n 2 || true
    python3 --version 2>/dev/null || true
    if command -v module &> /dev/null; then
        echo ""
        echo "=== Loaded Modules ==="
        module list 2>&1 || true
    fi
} > "${ENV_LOG}"

cat "${ENV_LOG}"
echo ""

# ------------------------------------------------------------------------------
# 2. Compilation (Clean Release Build with CUDA)
# ------------------------------------------------------------------------------
echo "--- [2/7] Compiling Project Targets (Release with CUDA) ---"
BUILD_LOG="${RESULTS_DIR}/build.log"

if [[ "${REBUILD}" -eq 1 && -d "build" ]]; then
    echo "Cleaning previous build directory..."
    rm -rf build
fi

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEAMFORMER_ENABLE_CUDA=ON \
    >"${BUILD_LOG}" 2>&1

NPROC=$(nproc 2>/dev/null || echo 8)
cmake --build build --config Release -j "${NPROC}" >>"${BUILD_LOG}" 2>&1

echo "Build complete. (See ${BUILD_LOG})"
echo ""

# ------------------------------------------------------------------------------
# 3. Unit & Correctness Tests
# ------------------------------------------------------------------------------
if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo "--- [3/7] Running Complete Unit & Parity Correctness Tests ---"

    # 3.1 CTest Engine Suite
    echo ">>> Running CTest Suite (All C++ and CUDA unit tests)..."
    ctest --test-dir build --output-on-failure >"${RESULTS_DIR}/tests/ctest.log" 2>&1 || {
        echo "(Warning: Some CTest tests reported failures; see ${RESULTS_DIR}/tests/ctest.log)"
    }
    grep -E "tests passed|Test project" "${RESULTS_DIR}/tests/ctest.log" || true

    # 3.2 Naive CPU Test Suite (Trivial + Base + Complex)
    if [[ -x "./build/beam_tracker_naive_cpu_test_suite" ]]; then
        echo ">>> Running Naive CPU Test Suite (trivial, base, complex tests)..."
        ./build/beam_tracker_naive_cpu_test_suite --skip-benchmark >"${RESULTS_DIR}/tests/naive_cpu_suite.log" 2>&1 || true
        grep -E "tests ran|tests passed|tests failed" "${RESULTS_DIR}/tests/naive_cpu_suite.log" || true
    fi

    # 3.3 Python Unit Tests
    echo ">>> Running Python unit tests..."
    python3 -m unittest discover -s tests/python -p "test_*.py" -v >"${RESULTS_DIR}/tests/python_tests.log" 2>&1 || true
    tail -n 2 "${RESULTS_DIR}/tests/python_tests.log" || true

    echo "Unit test suite completed."
    echo ""
else
    echo "--- [3/7] Skipping Unit Tests (--skip-tests) ---"
    echo ""
fi

# ------------------------------------------------------------------------------
# 4. Astronomical Validation Suite (FRB Injection & Sensitivity Scaling)
# ------------------------------------------------------------------------------
echo "--- [4/7] Running Astronomical Validation Suite ---"

ASTRO_SCRIPT="tools/run_astronomical_validation.py"
if [[ -f "${ASTRO_SCRIPT}" && -x "./build/run_tracker_stream" ]]; then
    echo ">>> Running Astronomical Validation: CUDA V3 Engine (all bursts)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v3 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v3" || true

    echo ">>> Running Astronomical Validation: CUDA Phase 4 FWS Engine (all bursts)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_fws \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/phase4_fws" || true

    echo ">>> Running Astronomical Validation: CPU Opt v2 Engine (all bursts)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_v2 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2" || true

    echo ">>> Running Astronomical Validation: CPU Naive Engine (canonical burst)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_naive \
        --burst FRB20180916B_canonical \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_naive" || true

    # Copy dashboard PNGs to plots directory
    cp "${RESULTS_DIR}/astronomical_validation/v3/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v3.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/phase4_fws/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_phase4_fws.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_opt_v2.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_naive/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_naive.png" 2>/dev/null || true

    echo "Astronomical validation completed."
else
    echo "(Warning: run_tracker_stream or ${ASTRO_SCRIPT} not ready; skipping astronomical validation)"
fi
echo ""

# ------------------------------------------------------------------------------
# 5. Multi-Engine Benchmark Sweeps
# ------------------------------------------------------------------------------
if [[ "${SKIP_BENCH}" -eq 0 ]]; then
    echo "--- [5/7] Executing Comprehensive Benchmark Matrix & Performance Sweeps ---"

    if [[ "${QUICK}" -eq 1 ]]; then
        THREAD_COUNTS=(1 24)
        ANT_COUNTS=(64)
        REPEAT=2
        WINDOW_REPEATS=2
    else
        THREAD_COUNTS=(1 2 4 8 16 24)
        ANT_COUNTS=(32 64)
        REPEAT=5
        WINDOW_REPEATS=3
    fi

    N_TIME=15360
    INT_SPEC=320

    # 5.1 CUDA Tracker V3 Master Benchmark (Evaluates all 11+ engines)
    if [[ -x "./build/benchmark_cuda_tracker_v3" ]]; then
        echo ">>> [5.1] Sweeping benchmark_cuda_tracker_v3 across thread counts & antenna sizes..."
        V3_LOG="${RESULTS_DIR}/benchmarks/tracker_v3/tracker_v3_sweep.log"
        for N_ANT in "${ANT_COUNTS[@]}"; do
            for T in "${THREAD_COUNTS[@]}"; do
                export OMP_NUM_THREADS=${T}
                echo "--- benchmark_cuda_tracker_v3: n_ant=${N_ANT} | threads=${T} ---" | tee -a "${V3_LOG}"
                ./build/benchmark_cuda_tracker_v3 \
                    --n-time ${N_TIME} \
                    --n-ant ${N_ANT} \
                    --integration-spectra ${INT_SPEC} \
                    --threads ${T} \
                    --repeat ${REPEAT} \
                    --window-repeats ${WINDOW_REPEATS} \
                    --outdir "${RESULTS_DIR}/benchmarks/tracker_v3" >>"${V3_LOG}" 2>&1 || true
                echo "" >>"${V3_LOG}"
            done
        done
        echo "CUDA Tracker V3 sweep finished."
    fi

    # 5.2 CUDA Tracker V2 Comparison Sweep (Emits summary, frame latencies, validation CSVs)
    if [[ -x "./build/benchmark_cuda_tracker_v2" ]]; then
        echo ">>> [5.2] Running benchmark_cuda_tracker_v2 sweep..."
        V2_OUTDIR="${RESULTS_DIR}/benchmarks/tracker_v2"
        for N_ANT in "${ANT_COUNTS[@]}"; do
            for T in "${THREAD_COUNTS[@]}"; do
                export OMP_NUM_THREADS=${T}
                ./build/benchmark_cuda_tracker_v2 \
                    --n-time ${N_TIME} \
                    --n-ant ${N_ANT} \
                    --integration-spectra ${INT_SPEC} \
                    --threads ${T} \
                    --repeat ${REPEAT} \
                    --window-repeats ${WINDOW_REPEATS} \
                    --outdir "${V2_OUTDIR}" || true
            done
        done
        echo "CUDA Tracker V2 sweep finished. (CSVs written to ${V2_OUTDIR})"
    fi

    # 5.3 CPU Optimized Tracker Benchmark (0.5 ms/frame target)
    if [[ -x "./build/benchmark_cpu_opt_beam_tracker" ]]; then
        echo ">>> [5.3] Running benchmark_cpu_opt_beam_tracker..."
        CPU_OPT_DIR="${RESULTS_DIR}/benchmarks/cpu_opt_tracker"
        for N_ANT in "${ANT_COUNTS[@]}"; do
            for T in "${THREAD_COUNTS[@]}"; do
                export OMP_NUM_THREADS=${T}
                ./build/benchmark_cpu_opt_beam_tracker \
                    --n-time ${N_TIME} \
                    --n-ant ${N_ANT} \
                    --integration-spectra ${INT_SPEC} \
                    --repeat ${REPEAT} \
                    --metrics "${CPU_OPT_DIR}/cpu_opt_metrics_sweep.csv" \
                    --frames "${CPU_OPT_DIR}/frames_${N_ANT}ant_${T}t.csv" || true
            done
        done
        echo "CPU Opt Tracker benchmark finished."
    fi

    # 5.4 CPU Naive vs Opt v1 Sweep
    if [[ -x "./build/benchmark_beam_tracker_opt" ]]; then
        echo ">>> [5.4] Running benchmark_beam_tracker_opt..."
        OPT1_DIR="${RESULTS_DIR}/benchmarks/cpu_naive_opt_v1"
        for T in "${THREAD_COUNTS[@]}"; do
            ./build/benchmark_beam_tracker_opt \
                --n-time ${N_TIME} \
                --n-ant 64 \
                --integration-spectra ${INT_SPEC} \
                --threads ${T} \
                --repeat ${REPEAT} \
                --outdir "${OPT1_DIR}" || true
        done
    fi

    # 5.5 CPU Naive vs Opt v1 vs Opt v2 Sweep (+ NUMA check)
    if [[ -x "./build/benchmark_beam_tracker_opt_v2" ]]; then
        echo ">>> [5.5] Running benchmark_beam_tracker_opt_v2..."
        OPT2_DIR="${RESULTS_DIR}/benchmarks/cpu_naive_opt_v2"
        for T in "${THREAD_COUNTS[@]}"; do
            ./build/benchmark_beam_tracker_opt_v2 \
                --n-time ${N_TIME} \
                --n-ant 64 \
                --integration-spectra ${INT_SPEC} \
                --threads ${T} \
                --repeat ${REPEAT} \
                --outdir "${OPT2_DIR}" || true
        done
        if command -v numactl &>/dev/null; then
            echo "Running NUMA diagnostic under numactl --interleave=all..."
            numactl --interleave=all ./build/benchmark_beam_tracker_opt_v2 \
                --n-time ${N_TIME} \
                --n-ant 64 \
                --integration-spectra ${INT_SPEC} \
                --threads 24 \
                --repeat ${REPEAT} \
                --outdir "${OPT2_DIR}" || true
        fi
    fi

    # 5.6 Naive CPU Matrix Sweep
    if [[ -x "./build/beam_tracker_naive_cpu_test_suite" ]]; then
        echo ">>> [5.6] Running beam_tracker_naive_cpu_test_suite matrix sweep..."
        ./build/beam_tracker_naive_cpu_test_suite \
            --skip-trivial --skip-base --skip-complex \
            --n-ant 32,64 \
            --times 15360,30720 \
            --beams 1,16,64 \
            --metrics "${RESULTS_DIR}/benchmarks/naive_cpu_matrix/benchmark_naive_cpu_matrix.csv" || true
    fi

    # 5.7 Direct Offline CPU vs CUDA Beamformer Benchmark
    if [[ -x "./build/benchmark_cpu_cuda" ]]; then
        echo ">>> [5.7] Running direct offline beamformer benchmark (benchmark_cpu_cuda)..."
        ./build/benchmark_cpu_cuda \
            --output-prefix "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct" \
            --n-ant 32,64 \
            --times 15360,24576,30720 \
            --beams 16,32,64,128 \
            --validation-time 16 \
            --warmup 2 \
            --repetitions 5 || true
    fi

    echo "All benchmarks completed."
    echo ""
else
    echo "--- [5/7] Skipping Benchmarks (--skip-bench) ---"
    echo ""
fi

# ------------------------------------------------------------------------------
# 6. Visualization Dashboards & Plot Generation
# ------------------------------------------------------------------------------
echo "--- [6/7] Rendering Multi-Panel Comparison Dashboards & Visualizations ---"

if python3 -c "import matplotlib, numpy" 2>/dev/null; then
    # 6.1 CPU vs GPU Comparison Dashboard
    if [[ -f "${RESULTS_DIR}/benchmarks/tracker_v2/benchmark_cuda_tracker_v2_summary.csv" ]]; then
        echo ">>> Plotting tracker comparison dashboard..."
        python3 tools/plot_tracker_comparison.py \
            --input-prefix "${RESULTS_DIR}/benchmarks/tracker_v2/benchmark_cuda_tracker_v2" \
            --output "${RESULTS_DIR}/plots/tracker_cpu_vs_gpu_dashboard.png" \
            --budget-ms 0.5 || true
    fi

    # 6.2 CPU Opt Tracker Dashboard
    if [[ -f "${RESULTS_DIR}/benchmarks/cpu_opt_tracker/cpu_opt_metrics_sweep.csv" ]]; then
        echo ">>> Plotting CPU opt tracker dashboard..."
        LATEST_FRAMES=$(find "${RESULTS_DIR}/benchmarks/cpu_opt_tracker" -name "frames_*.csv" | head -n 1)
        if [[ -n "${LATEST_FRAMES}" ]]; then
            python3 tools/plot_cpu_opt_beam_tracker.py \
                --metrics "${RESULTS_DIR}/benchmarks/cpu_opt_tracker/cpu_opt_metrics_sweep.csv" \
                --frames "${LATEST_FRAMES}" \
                --output "${RESULTS_DIR}/plots/cpu_opt_tracker_dashboard.png" || true
        fi
    fi

    # 6.3 Direct Offline Beamformer Plots
    if [[ -f "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct_timings.csv" ]]; then
        echo ">>> Plotting direct beamformer benchmark..."
        python3 tools/plot_benchmark.py \
            --input-prefix "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct" \
            --output-prefix "${RESULTS_DIR}/plots/direct_beamformer_benchmark" || true
    fi
else
    echo "(Warning: Python matplotlib/numpy not available in current environment; skipping plotting)"
fi
echo ""

# ------------------------------------------------------------------------------
# 7. Summary Generation, Packaging & SCP Instructions
# ------------------------------------------------------------------------------
echo "--- [7/7] Compiling Unified Summary & Packaging All Results ---"

if [[ -f "tools/generate_run_summary.py" ]]; then
    python3 tools/generate_run_summary.py \
        --results-dir "${RESULTS_DIR}" \
        --slurm-job-id "${SLURM_JOB_ID:-}" || true
fi

# Print text summary inline
if [[ -f "${RESULTS_DIR}/SUMMARY.txt" ]]; then
    echo ""
    cat "${RESULTS_DIR}/SUMMARY.txt"
    echo ""
fi

# Update symlink to latest run
LATEST_LINK="${PROJECT_ROOT}/results/everything_latest"
rm -f "${LATEST_LINK}"
ln -s "${RESULTS_DIR}" "${LATEST_LINK}" 2>/dev/null || true

# Generate compressed tarball
TAR_FILE="${RESULTS_DIR}.tar.gz"
echo "Creating compressed archive: ${TAR_FILE}..."
tar -czf "${TAR_FILE}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")" 2>/dev/null || true

echo ""
echo "========================================================================"
echo "          ALL TESTS, VALIDATIONS, AND BENCHMARKS COMPLETE!              "
echo "========================================================================"
echo "Results Directory: ${RESULTS_DIR}"
echo "Summary Report:   ${RESULTS_DIR}/SUMMARY.md"
echo "Tarball Archive:  ${TAR_FILE}"
echo ""
echo "========================================================================"
echo "                 ONE-COMMAND SCP RETRIEVAL INSTRUCTIONS                 "
echo "========================================================================"
echo "To transfer all results to your local computer in one command, run:"
echo ""
echo "  # Download full results folder:"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR} ./"
echo ""
echo "  # Or download the single compressed tarball:"
echo "  scp <user>@trillium.scinet.utoronto.ca:${TAR_FILE} ./"
echo ""
echo "========================================================================"
