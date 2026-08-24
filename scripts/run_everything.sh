#!/usr/bin/env bash
# ==============================================================================
# run_everything.sh
#
# Master Test, Astronomical Validation, and Benchmark Runner for CHARTS Voltage
# Beamformer & Dynamic Beam Tracker on Trillium / Linux.
#
# Runs:
#   1. Environment & Hardware Diagnostics (GPU, CPU topology, OS, modules)
#   2. Clean CMake Build of all CPU and CUDA targets in Release mode
#   3. Full Unit & Correctness Test Suite (CTest, Naive CPU suite, Python tests)
#   4. Astronomical Validation Suite (CHIME FRB injection across V5, V4, V3, CPU)
#   5. Comprehensive Multi-Generation Benchmark Sweeps:
#        - CUDA Tracker V5 master sweep (Unified Engine, 32-256 ant)
#        - CUDA Tracker V4 master sweep (Tensor Core, Half2, Deep ILP)
#        - CUDA Tracker V3 multi-model sweep (11+ engines)
#        - CUDA Tracker V2 comparison sweep (CSV summaries & frame latencies)
#        - CPU Opt tracker sweep (per-window latency vs 0.5 ms target)
#        - CPU Naive vs Opt v1 vs Opt v2 sweeps
#        - Naive CPU matrix sweep
#        - CPU vs CUDA offline direct beamformer sweep
#   6. Publication & Presentation Visual Suite (generate_presentation_suite.py)
#   7. Multi-Panel Visualization Dashboards & Plots
#   8. Automated Unified Summary Generation (SUMMARY.md & SUMMARY.txt)
#   9. Archive Packaging & One-Command SCP Retrieval
#
# Usage:
#   bash scripts/run_everything.sh                   # Full standard sweep
#   bash scripts/run_everything.sh --quick           # Faster sweep (reduced threads/repeats)
#   bash scripts/run_everything.sh --skip-bench      # Build, correctness & astro validation only
#   bash scripts/run_everything.sh --skip-tests      # Build & benchmarks only
#   bash scripts/run_everything.sh --outdir DIR      # Custom output directory
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
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
mkdir -p "${RESULTS_DIR}/astronomical_validation/v5"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v4"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v3"
mkdir -p "${RESULTS_DIR}/astronomical_validation/phase4_fws"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_naive"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v5"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v4"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v3"
mkdir -p "${RESULTS_DIR}/benchmarks/tracker_v2"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_opt_tracker"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_naive_opt_v1"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_naive_opt_v2"
mkdir -p "${RESULTS_DIR}/benchmarks/naive_cpu_matrix"
mkdir -p "${RESULTS_DIR}/benchmarks/cpu_cuda_offline"
mkdir -p "${RESULTS_DIR}/presentation_assets"
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
echo "--- [1/8] Environment Setup & Hardware Diagnostics ---"
if command -v module &> /dev/null; then
    echo "Loading HPC modules (StdEnv/2023, gcc/12.3, cuda/12.6, python/3.11, scipy-stack)..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
fi

if [[ -d ".venv" ]]; then
    echo "Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "Activating Python virtual environment (venv)..."
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}
export OMP_PROC_BIND=true
export MPLCONFIGDIR="/tmp/matplotlib_${USER:-charts}_${TIMESTAMP}"
mkdir -p "${MPLCONFIGDIR}"

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
# 2. Compilation
# ------------------------------------------------------------------------------
echo "--- [2/8] Compiling Project Targets (Release with CUDA) ---"
BUILD_LOG="${RESULTS_DIR}/build.log"

if [[ "${REBUILD}" -eq 1 || ! -f "build/CMakeCache.txt" || ! -f "build/benchmark_cuda_tracker_v5" || ! -f "build/run_tracker_stream" ]]; then
    echo "Configuring clean build with CUDA support..."
    rm -rf build
fi

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEAMFORMER_ENABLE_CUDA=ON \
    -DBUILD_TESTING=ON \
    2>&1 | tee "${BUILD_LOG}"

NPROC=$(nproc 2>/dev/null || echo 24)
cmake --build build --config Release -j "${NPROC}" 2>&1 | tee -a "${BUILD_LOG}"

echo "Build complete. (Log: ${BUILD_LOG})"
echo ""

# ------------------------------------------------------------------------------
# 3. Unit & Parity Correctness Tests
# ------------------------------------------------------------------------------
if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo "--- [3/8] Running Unit & Parity Correctness Tests ---"

    # CTest Suite
    echo ">>> Running CTest Suite (All C++ and CUDA unit tests)..."
    ctest --test-dir build --output-on-failure 2>&1 | tee "${RESULTS_DIR}/tests/ctest.log" || {
        echo "(Warning: Some CTest tests reported failures; see ${RESULTS_DIR}/tests/ctest.log)"
    }

    # Naive CPU Test Suite
    if [[ -x "./build/beam_tracker_naive_cpu_test_suite" ]]; then
        echo ">>> Running Naive CPU Test Suite..."
        ./build/beam_tracker_naive_cpu_test_suite --skip-benchmark 2>&1 | tee "${RESULTS_DIR}/tests/naive_cpu_suite.log" || true
    fi

    # Python Unit Tests
    echo ">>> Running Python unit test suite..."
    python3 -m unittest discover -s tests/python -p "test_*.py" -v 2>&1 | tee "${RESULTS_DIR}/tests/python_tests.log" || true

    if [[ -f "tools/run_temporal_integration_test.py" ]]; then
        echo ">>> Running temporal integration verification test..."
        python3 tools/run_temporal_integration_test.py 2>&1 | tee -a "${RESULTS_DIR}/tests/python_tests.log" || true
    fi

    echo "Unit test suite completed."
    echo ""
else
    echo "--- [3/8] Skipping Unit Tests (--skip-tests) ---"
    echo ""
fi

# ------------------------------------------------------------------------------
# 4. Astronomical Validation Suite
# ------------------------------------------------------------------------------
echo "--- [4/8] Running Astronomical Validation Suite ---"
ASTRO_SCRIPT="tools/run_astronomical_validation.py"
if [[ -f "${ASTRO_SCRIPT}" && -x "./build/run_tracker_stream" ]]; then
    echo ">>> Astronomical Validation: CUDA V5 Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v5 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v5" || true

    echo ">>> Astronomical Validation: CUDA V4 Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v4 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v4" || true

    echo ">>> Astronomical Validation: CUDA V3 Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v3 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v3" || true

    echo ">>> Astronomical Validation: CUDA Phase 4 FWS Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_fws \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/phase4_fws" || true

    echo ">>> Astronomical Validation: CPU Opt v2 Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_v2 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2" || true

    echo ">>> Astronomical Validation: CPU Naive Engine..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_naive \
        --burst FRB20180916B_canonical \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_naive" || true

    # Copy dashboard PNGs to plots directory
    cp "${RESULTS_DIR}/astronomical_validation/v5/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v5.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/v4/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v4.png" 2>/dev/null || true
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
# 5. Comprehensive Multi-Engine Benchmark Sweeps
# ------------------------------------------------------------------------------
if [[ "${SKIP_BENCH}" -eq 0 ]]; then
    echo "--- [5/8] Executing Comprehensive Benchmark Sweeps ---"

    if [[ "${QUICK}" -eq 1 ]]; then
        THREAD_COUNTS=(1 24)
        ANT_COUNTS=(64)
        REPEAT=2
        WINDOW_REPEATS=2
    else
        THREAD_COUNTS=(1 2 4 8 16 24)
        ANT_COUNTS=(32 64 128 256)
        REPEAT=5
        WINDOW_REPEATS=3
    fi

    N_TIME=15360
    INT_SPEC=320

    # 5.1 CUDA Tracker V5 Master Benchmark
    if [[ -x "./build/benchmark_cuda_tracker_v5" ]]; then
        echo ">>> [5.1] benchmark_cuda_tracker_v5 sweep..."
        V5_LOG="${RESULTS_DIR}/benchmarks/tracker_v5/tracker_v5_sweep.log"
        for N_ANT in "${ANT_COUNTS[@]}"; do
            echo "--- benchmark_cuda_tracker_v5: n_ant=${N_ANT} ---" | tee -a "${V5_LOG}"
            ./build/benchmark_cuda_tracker_v5 \
                --n-time ${N_TIME} \
                --n-ant ${N_ANT} \
                --integration-spectra ${INT_SPEC} \
                --repeat ${REPEAT} \
                --window-repeats ${WINDOW_REPEATS} \
                --outdir "${RESULTS_DIR}/benchmarks/tracker_v5" >>"${V5_LOG}" 2>&1 || true
            echo "" >>"${V5_LOG}"
        done
        echo "CUDA Tracker V5 sweep finished."
    fi

    # 5.2 CUDA Tracker V4 Master Benchmark
    if [[ -x "./build/benchmark_cuda_tracker_v4" ]]; then
        echo ">>> [5.2] benchmark_cuda_tracker_v4 sweep..."
        V4_LOG="${RESULTS_DIR}/benchmarks/tracker_v4/tracker_v4_sweep.log"
        for N_ANT in 32 64; do
            echo "--- benchmark_cuda_tracker_v4: n_ant=${N_ANT} ---" | tee -a "${V4_LOG}"
            ./build/benchmark_cuda_tracker_v4 \
                --n-time ${N_TIME} \
                --n-ant ${N_ANT} \
                --integration-spectra ${INT_SPEC} \
                --repeat ${REPEAT} \
                --window-repeats ${WINDOW_REPEATS} \
                --outdir "${RESULTS_DIR}/benchmarks/tracker_v4" >>"${V4_LOG}" 2>&1 || true
            echo "" >>"${V4_LOG}"
        done
        echo "CUDA Tracker V4 sweep finished."
    fi

    # 5.3 CUDA Tracker V3 Master Benchmark
    if [[ -x "./build/benchmark_cuda_tracker_v3" ]]; then
        echo ">>> [5.3] benchmark_cuda_tracker_v3 sweep..."
        V3_LOG="${RESULTS_DIR}/benchmarks/tracker_v3/tracker_v3_sweep.log"
        for N_ANT in 32 64; do
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

    # 5.4 CUDA Tracker V2 Comparison Sweep
    if [[ -x "./build/benchmark_cuda_tracker_v2" ]]; then
        echo ">>> [5.4] benchmark_cuda_tracker_v2 sweep..."
        V2_OUTDIR="${RESULTS_DIR}/benchmarks/tracker_v2"
        for N_ANT in 32 64; do
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
        echo "CUDA Tracker V2 sweep finished. (CSVs in ${V2_OUTDIR})"
    fi

    # 5.5 CPU Optimized Tracker Benchmark
    if [[ -x "./build/benchmark_cpu_opt_beam_tracker" ]]; then
        echo ">>> [5.5] benchmark_cpu_opt_beam_tracker..."
        CPU_OPT_DIR="${RESULTS_DIR}/benchmarks/cpu_opt_tracker"
        for N_ANT in 32 64; do
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

    # 5.6 CPU Naive vs Opt v1 Sweep
    if [[ -x "./build/benchmark_beam_tracker_opt" ]]; then
        echo ">>> [5.6] benchmark_beam_tracker_opt..."
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

    # 5.7 CPU Naive vs Opt v1 vs Opt v2 Sweep
    if [[ -x "./build/benchmark_beam_tracker_opt_v2" ]]; then
        echo ">>> [5.7] benchmark_beam_tracker_opt_v2..."
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
    fi

    # 5.8 Naive CPU Matrix Sweep
    if [[ -x "./build/beam_tracker_naive_cpu_test_suite" ]]; then
        echo ">>> [5.8] beam_tracker_naive_cpu_test_suite matrix sweep..."
        ./build/beam_tracker_naive_cpu_test_suite \
            --skip-trivial --skip-base --skip-complex \
            --n-ant 32,64 \
            --times 15360,30720 \
            --beams 1,16,64 \
            --metrics "${RESULTS_DIR}/benchmarks/naive_cpu_matrix/benchmark_naive_cpu_matrix.csv" || true
    fi

    # 5.9 Direct Offline CPU vs CUDA Beamformer Benchmark
    if [[ -x "./build/benchmark_cpu_cuda" ]]; then
        echo ">>> [5.9] direct offline beamformer benchmark (benchmark_cpu_cuda)..."
        ./build/benchmark_cpu_cuda \
            --output-prefix "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct" \
            --n-ant 32,64 \
            --times 15360,24576,30720 \
            --beams 16,32,64,128 \
            --validation-time 16 \
            --warmup 2 \
            --repetitions 5 || true
    fi

    # 5.10 Quantized CUDA Beamformer Benchmark
    if [[ -x "./build/benchmark_cuda_quantized" && -x "./build/generate_fake_data" && -x "./build/generate_weights" ]]; then
        echo ">>> [5.10] quantized CUDA beamformer benchmark (benchmark_cuda_quantized)..."
        mkdir -p "${RESULTS_DIR}/benchmarks/quantized"
        TMP_DATA="${RESULTS_DIR}/benchmarks/quantized/fake_data.bin"
        TMP_WEIGHTS="${RESULTS_DIR}/benchmarks/quantized/fake_weights.bin"
        ./build/generate_fake_data --output "${TMP_DATA}" --n-time 15360 --n-ant 64 >/dev/null 2>&1 || true
        ./build/generate_weights --output "${TMP_WEIGHTS}" --n-ant 64 --beams 16 >/dev/null 2>&1 || true
        if [[ -f "${TMP_DATA}" && -f "${TMP_WEIGHTS}" ]]; then
            ./build/benchmark_cuda_quantized \
                --input "${TMP_DATA}" \
                --weights "${TMP_WEIGHTS}" \
                --log "${RESULTS_DIR}/benchmarks/quantized/quantized_log.txt" \
                --summary "${RESULTS_DIR}/benchmarks/quantized/quantized_summary.txt" \
                --n-time 15360 \
                --integration-spectra 320 \
                --kernel tiled \
                --warmups 2 \
                --repetitions 5 || true
            rm -f "${TMP_DATA}" "${TMP_WEIGHTS}"
        fi
    fi

    echo "All benchmarks completed."
    echo ""
else
    echo "--- [5/8] Skipping Benchmarks (--skip-bench) ---"
    echo ""
fi

# ------------------------------------------------------------------------------
# 6. Publication & Presentation Visual Material Generation
# ------------------------------------------------------------------------------
echo "--- [6/8] Generating Presentation Visual Suite ---"
if [[ -f "tools/generate_presentation_suite.py" ]]; then
    python3 tools/generate_presentation_suite.py \
        --outdir "${RESULTS_DIR}/presentation_assets" \
        --engine cuda_v5 || true

    # Copy presentation figures to plots directory
    cp "${RESULTS_DIR}/presentation_assets/"*.png "${RESULTS_DIR}/plots/" 2>/dev/null || true
    echo "Presentation suite generated in ${RESULTS_DIR}/presentation_assets/"
fi
echo ""

# ------------------------------------------------------------------------------
# 7. Visualization Dashboards & Plot Generation
# ------------------------------------------------------------------------------
echo "--- [7/8] Rendering Comparison Dashboards & Timings Visualizations ---"
if python3 -c "import matplotlib, numpy" 2>/dev/null; then
    # Tracker Comparison Dashboard
    if [[ -f "${RESULTS_DIR}/benchmarks/tracker_v2/benchmark_cuda_tracker_v2_summary.csv" ]]; then
        echo ">>> Plotting tracker comparison dashboard..."
        python3 tools/plot_tracker_comparison.py \
            --input-prefix "${RESULTS_DIR}/benchmarks/tracker_v2/benchmark_cuda_tracker_v2" \
            --output "${RESULTS_DIR}/plots/tracker_cpu_vs_gpu_dashboard.png" \
            --budget-ms 0.5 || true
    fi

    # CPU Opt Tracker Dashboard
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

    # Direct Offline Beamformer Plots
    if [[ -f "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct_timings.csv" ]]; then
        echo ">>> Plotting direct beamformer benchmark..."
        python3 tools/plot_benchmark.py \
            --input-prefix "${RESULTS_DIR}/benchmarks/cpu_cuda_offline/gpu_benchmark_direct" \
            --output-prefix "${RESULTS_DIR}/plots/direct_beamformer_benchmark" || true
    fi
fi
echo ""

# ------------------------------------------------------------------------------
# 8. Summary Generation, Packaging & SCP Instructions
# ------------------------------------------------------------------------------
echo "--- [8/8] Compiling Unified Summary & Packaging All Results ---"
if [[ -f "tools/generate_run_summary.py" ]]; then
    python3 tools/generate_run_summary.py \
        --results-dir "${RESULTS_DIR}" \
        --slurm-job-id "${SLURM_JOB_ID:-}" || true
fi

# Print text summary inline to job output
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

# Copy Slurm logs into the results directory if running under batch
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${PROJECT_ROOT}/results/slurm_everything_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/results/slurm_everything_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/" 2>/dev/null || true
fi

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
echo "  # Download only the presentation figures & plots:"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR}/plots ./"
echo "========================================================================"
