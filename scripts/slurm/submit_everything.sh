#!/bin/bash
#SBATCH --job-name=beamform_everything
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm_everything_%j.out
#SBATCH --error=results/slurm_everything_%j.err

# ==============================================================================
# submit_everything.sh
#
# Standalone, self-contained master runner for Trillium GPU nodes.
# Runs ALL unit tests, astronomical validations, and benchmark sweeps in one job
# under identical runtime conditions, packaging all outputs, CSVs, logs,
# reports, and plots into a single directory for 1-command SCP retrieval.
#
# Usage:
#   sbatch scripts/slurm/submit_everything.sh     # Full production sweep
#   sbatch scripts/slurm/submit_everything.sh --quick # Smoke test (reduced threads/reps)
#   bash scripts/slurm/submit_everything.sh       # Interactive / debugjob
# ==============================================================================

set -uo pipefail

# 1. Set working directory to project root
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
PROJECT_ROOT="$(pwd)"

# 2. Parse optional command line flags
QUICK=0
REBUILD=0
SKIP_BENCH=0
SKIP_TESTS=0

for arg in "$@"; do
    case "${arg}" in
        --quick)        QUICK=1 ;;
        --rebuild)      REBUILD=1 ;;
        --skip-bench)   SKIP_BENCH=1 ;;
        --skip-tests)   SKIP_TESTS=1 ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--rebuild] [--skip-bench] [--skip-tests]"
            exit 0
            ;;
    esac
done

# 3. Create timestamped results directory
TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
RESULTS_DIR="${PROJECT_ROOT}/results/everything_${TIMESTAMP}"

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

# 4. Load HPC Environment Modules & Python Environment
echo "--- [1/7] Environment Setup & Hardware Diagnostics ---"
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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OMP_PROC_BIND=true
export MPLCONFIGDIR="/tmp/matplotlib_${USER:-charts}"
mkdir -p "${MPLCONFIGDIR}"

# Capture hardware & system diagnostics
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

# 5. Clean Release Build with CUDA
echo "--- [2/7] Compiling Project Targets (Release with CUDA) ---"
BUILD_LOG="${RESULTS_DIR}/build.log"

if [[ "${REBUILD}" -eq 1 || ! -f "build/CMakeCache.txt" || ! -f "build/benchmark_cuda_tracker_v3" || ! -f "build/run_tracker_stream" ]]; then
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

# 6. Unit & Parity Correctness Tests
if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo "--- [3/7] Running Unit & Parity Correctness Tests ---"

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
    echo ">>> Running Python unit tests..."
    python3 -m unittest discover -s tests/python -p "test_*.py" -v 2>&1 | tee "${RESULTS_DIR}/tests/python_tests.log" || true

    echo "Unit test suite completed."
    echo ""
else
    echo "--- [3/7] Skipping Unit Tests (--skip-tests) ---"
    echo ""
fi

# 7. Astronomical Validation Suite
echo "--- [4/7] Running Astronomical Validation Suite ---"
ASTRO_SCRIPT="tools/run_astronomical_validation.py"
if [[ -f "${ASTRO_SCRIPT}" && -x "./build/run_tracker_stream" ]]; then
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
    cp "${RESULTS_DIR}/astronomical_validation/v3/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v3.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/phase4_fws/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_phase4_fws.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_opt_v2/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_opt_v2.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_naive/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_naive.png" 2>/dev/null || true

    echo "Astronomical validation completed."
else
    echo "(Warning: run_tracker_stream or ${ASTRO_SCRIPT} not ready; skipping astronomical validation)"
fi
echo ""

# 8. Comprehensive Multi-Engine Benchmark Sweeps
if [[ "${SKIP_BENCH}" -eq 0 ]]; then
    echo "--- [5/7] Executing Comprehensive Benchmark Sweeps ---"

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

    # 8.1 CUDA Tracker V3 Master Benchmark (11+ engines)
    if [[ -x "./build/benchmark_cuda_tracker_v3" ]]; then
        echo ">>> [8.1] benchmark_cuda_tracker_v3 sweep..."
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

    # 8.2 CUDA Tracker V2 Comparison Sweep (Emits CSVs)
    if [[ -x "./build/benchmark_cuda_tracker_v2" ]]; then
        echo ">>> [8.2] benchmark_cuda_tracker_v2 sweep..."
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
        echo "CUDA Tracker V2 sweep finished. (CSVs in ${V2_OUTDIR})"
    fi

    # 8.3 CPU Optimized Tracker Benchmark (0.5 ms target)
    if [[ -x "./build/benchmark_cpu_opt_beam_tracker" ]]; then
        echo ">>> [8.3] benchmark_cpu_opt_beam_tracker..."
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

    # 8.4 CPU Naive vs Opt v1 Sweep
    if [[ -x "./build/benchmark_beam_tracker_opt" ]]; then
        echo ">>> [8.4] benchmark_beam_tracker_opt..."
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

    # 8.5 CPU Naive vs Opt v1 vs Opt v2 Sweep (+ NUMA check)
    if [[ -x "./build/benchmark_beam_tracker_opt_v2" ]]; then
        echo ">>> [8.5] benchmark_beam_tracker_opt_v2..."
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

    # 8.6 Naive CPU Matrix Sweep
    if [[ -x "./build/beam_tracker_naive_cpu_test_suite" ]]; then
        echo ">>> [8.6] beam_tracker_naive_cpu_test_suite matrix sweep..."
        ./build/beam_tracker_naive_cpu_test_suite \
            --skip-trivial --skip-base --skip-complex \
            --n-ant 32,64 \
            --times 15360,30720 \
            --beams 1,16,64 \
            --metrics "${RESULTS_DIR}/benchmarks/naive_cpu_matrix/benchmark_naive_cpu_matrix.csv" || true
    fi

    # 8.7 Direct Offline CPU vs CUDA Beamformer Benchmark
    if [[ -x "./build/benchmark_cpu_cuda" ]]; then
        echo ">>> [8.7] direct offline beamformer benchmark (benchmark_cpu_cuda)..."
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

# 9. Visualization Dashboards & Plot Generation
echo "--- [6/7] Rendering Multi-Panel Dashboards & Visualizations ---"
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

# 10. Summary Generation, Packaging & SCP Instructions
echo "--- [7/7] Compiling Unified Summary & Packaging All Results ---"
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
echo "========================================================================"
