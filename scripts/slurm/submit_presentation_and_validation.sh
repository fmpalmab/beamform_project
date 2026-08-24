#!/bin/bash
#SBATCH --job-name=beamform_pres_val
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm_pres_val_%j.out
#SBATCH --error=results/slurm_pres_val_%j.err

# ==============================================================================
# scripts/slurm/submit_presentation_and_validation.sh
#
# Dedicated Slurm batch runner for Trillium GPU nodes:
# Runs Python unit testing ("Verython"), astronomical FRB validation suite,
# and generates the complete presentation material & visualization suite
# (generate_presentation_suite.py) in a single timestamped folder with 1-command
# SCP retrieval.
#
# Usage:
#   sbatch scripts/slurm/submit_presentation_and_validation.sh
#   bash scripts/slurm/submit_presentation_and_validation.sh     # Interactive / debug
# ==============================================================================

set -uo pipefail

# 1. Set working directory to project root
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
PROJECT_ROOT="$(pwd)"

# 2. Create timestamped results directory
TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
RESULTS_DIR="${PROJECT_ROOT}/results/presentation_validation_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}/tests"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v5"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v4"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v3"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_v2"
mkdir -p "${RESULTS_DIR}/presentation_assets"
mkdir -p "${RESULTS_DIR}/plots"

echo "========================================================================"
echo "    CHARTS BEAMFORMER: PYTHON VALIDATION & PRESENTATION SUITE RUNNER    "
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

# 3. Load HPC Environment Modules & Python Environment
echo "--- [1/5] Loading HPC Modules & Virtual Environment ---"
if command -v module &> /dev/null; then
    echo "Loading StdEnv/2023, gcc/12.3, cuda/12.6, python/3.11, scipy-stack..."
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
export MPLCONFIGDIR="/tmp/matplotlib_${USER:-charts}_${TIMESTAMP}"
mkdir -p "${MPLCONFIGDIR}"

# Capture hardware & system diagnostics
ENV_LOG="${RESULTS_DIR}/env_info.txt"
{
    echo "Host: $(hostname)"
    echo "Date: $(date -u)"
    echo "Kernel: $(uname -a)"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
        echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
    fi
    echo ""
    echo "=== GPU Status (nvidia-smi) ==="
    nvidia-smi 2>/dev/null || echo "(nvidia-smi unavailable or no GPU present)"
    echo ""
    echo "=== Python & Compilers ==="
    python3 --version 2>/dev/null || true
    gcc --version 2>/dev/null | head -n 1 || true
    nvcc --version 2>/dev/null | tail -n 2 || true
} > "${ENV_LOG}"
cat "${ENV_LOG}"
echo ""

# 4. Build C++/CUDA Stream Runner CLI
echo "--- [2/5] Compiling Stream Runner & C++ Libraries ---"
BUILD_LOG="${RESULTS_DIR}/build.log"

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DBEAMFORMER_ENABLE_CUDA=ON \
    -DBUILD_TESTING=ON \
    2>&1 | tee "${BUILD_LOG}"

NPROC=$(nproc 2>/dev/null || echo 24)
cmake --build build --config Release -j "${NPROC}" 2>&1 | tee -a "${BUILD_LOG}"
echo "Compilation complete."
echo ""

# 5. Run Python Unit & Integration Tests ("Verython")
echo "--- [3/5] Executing Python Test Suite (Verython) ---"
PYTHON_LOG="${RESULTS_DIR}/tests/python_test_suite.log"

python3 -m unittest discover -s tests/python -p "test_*.py" -v 2>&1 | tee "${PYTHON_LOG}"

if [[ -f "tools/run_temporal_integration_test.py" ]]; then
    echo ">>> Running temporal integration verification script..."
    python3 tools/run_temporal_integration_test.py 2>&1 | tee -a "${PYTHON_LOG}" || true
fi
echo "Python tests completed."
echo ""

# 6. Run Astronomical Validation Suite
echo "--- [4/5] Running Astronomical Validation Suite ---"
ASTRO_SCRIPT="tools/run_astronomical_validation.py"
if [[ -f "${ASTRO_SCRIPT}" && -x "./build/run_tracker_stream" ]]; then
    echo ">>> Astronomical Validation on CUDA V5 (Unified Engine)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v5 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v5" || true

    echo ">>> Astronomical Validation on CUDA V4 (Tensor Core / Deep ILP)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v4 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v4" || true

    echo ">>> Astronomical Validation on CUDA V3 (Batched Streaming)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v3 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v3" || true

    echo ">>> Astronomical Validation on CPU Opt v2 (OpenMP)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_v2 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_v2" || true

    # Copy astro dashboard PNGs to plots directory
    cp "${RESULTS_DIR}/astronomical_validation/v5/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v5.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/v4/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v4.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/v3/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v3.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_v2/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_v2.png" 2>/dev/null || true

    echo "Astronomical validation completed."
else
    echo "(Warning: run_tracker_stream or ${ASTRO_SCRIPT} not ready; skipping astronomical validation)"
fi
echo ""

# 7. Generate Full Presentation Material & Visualization Suite
echo "--- [5/5] Generating Presentation Suite & Visual Assets ---"
if [[ -f "tools/generate_presentation_suite.py" ]]; then
    python3 tools/generate_presentation_suite.py \
        --outdir "${RESULTS_DIR}/presentation_assets" \
        --engine cuda_v5 || true

    # Copy all generated presentation figures to plots directory
    cp "${RESULTS_DIR}/presentation_assets/"*.png "${RESULTS_DIR}/plots/" 2>/dev/null || true
    echo "Presentation materials generated in ${RESULTS_DIR}/presentation_assets/"
fi
echo ""

# 8. Packaging & Compression
LATEST_LINK="${PROJECT_ROOT}/results/presentation_validation_latest"
rm -f "${LATEST_LINK}"
ln -s "${RESULTS_DIR}" "${LATEST_LINK}" 2>/dev/null || true

TAR_FILE="${RESULTS_DIR}.tar.gz"
echo "Creating compressed archive: ${TAR_FILE}..."
tar -czf "${TAR_FILE}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")" 2>/dev/null || true

# Copy Slurm logs into the results directory if running under batch
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${PROJECT_ROOT}/results/slurm_pres_val_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/results/slurm_pres_val_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/" 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "    PYTHON VALIDATION & PRESENTATION SUITE GENERATION COMPLETE!         "
echo "========================================================================"
echo "Results Directory: ${RESULTS_DIR}"
echo "Plots & Dashboards: ${RESULTS_DIR}/plots"
echo "Presentation PNGs:  ${RESULTS_DIR}/presentation_assets"
echo "Tarball Archive:   ${TAR_FILE}"
echo ""
echo "========================================================================"
echo "                 ONE-COMMAND SCP RETRIEVAL INSTRUCTIONS                 "
echo "========================================================================"
echo "To transfer all generated presentation assets and validation results to"
echo "your local computer in one command, run:"
echo ""
echo "  # Download entire results directory (all PNGs, JSON reports, logs):"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR} ./"
echo ""
echo "  # Or download the single compressed tarball archive:"
echo "  scp <user>@trillium.scinet.utoronto.ca:${TAR_FILE} ./"
echo ""
echo "  # Or download only the presentation images folder:"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR}/presentation_assets ./"
echo "========================================================================"
