#!/bin/bash
#SBATCH --job-name=beamform_sim_targets
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm_sim_%j.out
#SBATCH --error=results/slurm_sim_%j.err

# ==============================================================================
# submit_target_simulations.sh
#
# Dedicated Slurm Batch Submission Script for SciNet Trillium GPU Nodes.
#
# Compiles Release C++/CUDA binaries and executes the live GPU Beam Tracker kernel
# on physical simulations of:
# 1. Vela Pulsar (PSR B0833-45)
# 2. The Sun (Solar Radio Burst / Flare)
# 3. Canonical Extragalactic Fast Radio Burst (FRB20180916B)
#
# Generates 300 DPI multi-panel dashboards showing:
# - 2D Synthesized Beam Directivity Footprint on Sky (l, m)
# - Raw Dynamic Spectrum Waterfall (Time x Frequency)
# - Coherently Dedispersed Waterfall & Integrated Pulse Profile
# - Power Dynamics & Transit Profiles (Tracked vs Untracked Drift Scan in dB)
#
# Usage:
#   sbatch submit_target_simulations.sh
#   sbatch submit_target_simulations.sh --engine cuda_v5 --antennas 128
#   bash submit_target_simulations.sh                   # Interactive node debug
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${PROJECT_ROOT}"

# Parse optional arguments
ENGINE="cuda_v5"
ANTENNAS=64
TARGET="all"
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)       ENGINE="$2"; shift 2 ;;
        --antennas)     ANTENNAS="$2"; shift 2 ;;
        --target)       TARGET="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--engine cuda_v5|cuda_v4|cpu_v2] [--antennas 64|128|256] [--target vela|sun|frb|all] [--skip-build]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
OUTDIR="${PROJECT_ROOT}/results/simulations_${TIMESTAMP}"
mkdir -p "${OUTDIR}"
mkdir -p "${PROJECT_ROOT}/results"

echo "========================================================================"
echo "    BEAM TRACKER: LIVE GPU TARGET SIMULATION & VALIDATION RUNNER       "
echo "========================================================================"
echo "Project Root : ${PROJECT_ROOT}"
echo "Output Dir   : ${OUTDIR}"
echo "Engine       : ${ENGINE}"
echo "Antennas     : ${ANTENNAS}"
echo "Target(s)    : ${TARGET}"
echo "Start Time   : $(date -u)"
echo "Host Node    : $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Slurm Job ID : ${SLURM_JOB_ID}"
fi
echo "========================================================================"
echo ""

# 1. Load HPC Modules & Environment
echo "--- [1/4] Loading Trillium HPC Modules & Environment ---"
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

# 2. Build C++/CUDA Binaries
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "--- [2/4] Compiling C++/CUDA Release Binaries ---"
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DBEAMFORMER_ENABLE_CUDA=ON \
        -DBUILD_TESTING=ON

    NPROC=$(nproc 2>/dev/null || echo 24)
    cmake --build build --config Release -j "${NPROC}"
    echo "Compilation complete."
else
    echo "--- [2/4] Skipping Compilation (--skip-build) ---"
fi
echo ""

# 3. Execute Target Simulations on GPU Kernel
echo "--- [3/4] Running Astronomical Simulations on Engine: ${ENGINE} ---"
python3 tools/simulate_astronomical_targets.py \
    --engine "${ENGINE}" \
    --antennas "${ANTENNAS}" \
    --target "${TARGET}" \
    --outdir "${OUTDIR}"
echo ""

# 4. Packaging & Archive Creation
echo "--- [4/4] Packaging Simulation Assets ---"
LATEST_LINK="${PROJECT_ROOT}/results/simulations_latest"
rm -f "${LATEST_LINK}"
ln -s "${OUTDIR}" "${LATEST_LINK}" 2>/dev/null || true

TAR_FILE="${OUTDIR}.tar.gz"
tar -czf "${TAR_FILE}" -C "$(dirname "${OUTDIR}")" "$(basename "${OUTDIR}")" 2>/dev/null || true

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${PROJECT_ROOT}/results/slurm_sim_${SLURM_JOB_ID}.out" "${OUTDIR}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/results/slurm_sim_${SLURM_JOB_ID}.err" "${OUTDIR}/" 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "    SIMULATION RUN COMPLETE!                                            "
echo "========================================================================"
echo "Outputs Saved To: ${OUTDIR}"
echo "Tarball Archive : ${TAR_FILE}"
echo ""
echo "========================================================================"
echo "                 ONE-COMMAND SCP RETRIEVAL INSTRUCTIONS                 "
echo "========================================================================"
echo "To copy the generated 300 DPI simulation dashboards to your local PC:"
echo ""
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${OUTDIR} ./"
echo "  # or download the single compressed archive:"
echo "  scp <user>@trillium.scinet.utoronto.ca:${TAR_FILE} ./"
echo "========================================================================"
