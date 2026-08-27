#!/bin/bash
#SBATCH --job-name=beamform_stress_test
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:45:00
#SBATCH --output=results/slurm_stress_%j.out
#SBATCH --error=results/slurm_stress_%j.err

# ==============================================================================
# submit_stress_test.sh
#
# Dedicated Slurm Batch Continuous Stress Test for SciNet Trillium GPU Nodes.
#
# Objectives:
# 1. Continuous high-throughput line-rate ingestion (3.33 us ADC cadence) for 2 hours.
# 2. Combined CUDA V5 Beam Tracker + Warp-level Upchannelizer (M=32 fine channels).
# 3. Detect and capture:
#    - Real-time deadline overruns (exceeding 1.066 ms / 320-spectra window).
#    - Numerical anomalies (NaNs, Infs, negative power, divergence).
#    - Memory leaks over time (cudaMemGetInfo VRAM tracking).
#    - GPU hardware throttling, power draw, and thermals.
# 4. Generate 300 DPI multi-panel reliability and jitter analysis dashboard.
#
# Usage:
#   sbatch submit_stress_test.sh
#   sbatch submit_stress_test.sh --duration 7200 --antennas 128
#   bash submit_stress_test.sh --duration 60             # 1-minute quick test
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_ROOT}"

# Default parameters
DURATION=7200 # 2 hours = 7200 seconds
ENGINE="cuda_v5"
ANTENNAS=64
UPCHAN=32
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)     DURATION="$2"; shift 2 ;;
        --engine)       ENGINE="$2"; shift 2 ;;
        --antennas)     ANTENNAS="$2"; shift 2 ;;
        --upchan)       UPCHAN="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--duration SEC (default 7200)] [--engine cuda_v5|cuda_v4] [--antennas 32|64|128|256] [--upchan 32] [--skip-build]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
OUTDIR="${PROJECT_ROOT}/results/stress_test_${TIMESTAMP}"
mkdir -p "${OUTDIR}"
mkdir -p "${PROJECT_ROOT}/results"

echo "========================================================================"
echo "    BEAM TRACKER + UPCHANNELIZER: 2-HOUR CONTINUOUS STRESS TEST         "
echo "========================================================================"
echo "Project Root : ${PROJECT_ROOT}"
echo "Output Dir   : ${OUTDIR}"
echo "Duration     : ${DURATION} seconds ($(( DURATION / 3600 )) hours $(( (DURATION % 3600) / 60 )) mins)"
echo "Engine       : ${ENGINE}"
echo "Antennas     : ${ANTENNAS}"
echo "Upchannel M  : ${UPCHAN}"
echo "Start Time   : $(date -u)"
echo "Host Node    : $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Slurm Job ID : ${SLURM_JOB_ID}"
fi
echo "========================================================================"
echo ""

# 1. Environment & HPC Modules
echo "--- [1/5] Loading HPC Modules & Python Environment ---"
if command -v module &> /dev/null; then
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
fi

if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MPLCONFIGDIR="/tmp/matplotlib_${USER:-charts}_${TIMESTAMP}"
mkdir -p "${MPLCONFIGDIR}"

# 2. Build Release Binaries
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "--- [2/5] Building Release CUDA Binaries ---"
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DBEAMFORMER_ENABLE_CUDA=ON \
        -DBUILD_TESTING=ON

    NPROC=$(nproc 2>/dev/null || echo 24)
    cmake --build build --config Release --target bench_continuous_stress_test -j "${NPROC}"
    echo "Compilation complete."
else
    echo "--- [2/5] Skipping Compilation (--skip-build) ---"
fi
echo ""

# 3. Start GPU Hardware Telemetry Logging (nvidia-smi in background)
echo "--- [3/5] Starting Background GPU Telemetry Monitor ---"
NVIDIA_SMI_PID=""
if command -v nvidia-smi &> /dev/null; then
    TELEMETRY_CSV="${OUTDIR}/gpu_telemetry.csv"
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
               --format=csv -l 5 > "${TELEMETRY_CSV}" 2>&1 &
    NVIDIA_SMI_PID=$!
    echo "GPU telemetry logging active (PID: ${NVIDIA_SMI_PID})."
fi
echo ""

# 4. Run Continuous Stress Test Loop
echo "--- [4/5] Launching Continuous Streaming Ingestion Engine (${DURATION}s) ---"
STRESS_EXE="./build/bench_continuous_stress_test"
if [[ ! -x "${STRESS_EXE}" ]]; then
    echo "Error: ${STRESS_EXE} not found or not executable." >&2
    exit 1
fi

"${STRESS_EXE}" \
    --duration "${DURATION}" \
    --engine "${ENGINE}" \
    --antennas "${ANTENNAS}" \
    --upchan "${UPCHAN}" \
    --outdir "${OUTDIR}"

STRESS_EXIT_CODE=$?

# Stop telemetry logger
if [[ -n "${NVIDIA_SMI_PID}" ]]; then
    kill "${NVIDIA_SMI_PID}" 2>/dev/null || true
fi
echo ""

# 5. Render Diagnostic Plots & Package Assets
echo "--- [5/5] Analyzing Timeline & Rendering Reliability Dashboard ---"
if [[ -f "${OUTDIR}/stress_test_timeline.csv" ]]; then
    python3 tools/plot_stress_test_results.py --dir "${OUTDIR}" || true
fi

LATEST_LINK="${PROJECT_ROOT}/results/stress_test_latest"
rm -f "${LATEST_LINK}"
ln -s "${OUTDIR}" "${LATEST_LINK}" 2>/dev/null || true

TAR_FILE="${OUTDIR}.tar.gz"
tar -czf "${TAR_FILE}" -C "$(dirname "${OUTDIR}")" "$(basename "${OUTDIR}")" 2>/dev/null || true

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${PROJECT_ROOT}/results/slurm_stress_${SLURM_JOB_ID}.out" "${OUTDIR}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/results/slurm_stress_${SLURM_JOB_ID}.err" "${OUTDIR}/" 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "    CONTINUOUS STRESS TEST SUITE COMPLETE!                              "
echo "========================================================================"
echo "Results Directory: ${OUTDIR}"
echo "Summary Metrics  : ${OUTDIR}/stress_test_summary.json"
echo "Timeline Data    : ${OUTDIR}/stress_test_timeline.csv"
echo "Dashboard Plot   : ${OUTDIR}/stress_test_dashboard.png"
echo "Tarball Archive  : ${TAR_FILE}"
echo "Exit Status      : ${STRESS_EXIT_CODE}"
echo ""
echo "========================================================================"
echo "                 ONE-COMMAND SCP RETRIEVAL INSTRUCTIONS                 "
echo "========================================================================"
echo "To download all stress test logs, CSV timelines, and PNG dashboards:"
echo ""
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${OUTDIR} ./"
echo "  # or download the single compressed archive:"
echo "  scp <user>@trillium.scinet.utoronto.ca:${TAR_FILE} ./"
echo "========================================================================"

exit "${STRESS_EXIT_CODE}"
