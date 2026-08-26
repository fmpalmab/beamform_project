#!/usr/bin/env bash
# ==============================================================================
# run_presentation.sh
#
# Standalone Presentation Suite & Validation Runner for the GPU Beam Tracker.
#
# Executes:
# 1. HPC module loading & Python virtual environment activation
# 2. C++/CUDA compilation (CMake Release build with CUDA support)
# 3. Unit & integration test execution (Verython)
# 4. Master presentation visual suite & data generator (tools/generate_presentation_suite.py)
# 5. Astronomical FRB validation suite (cuda_v5, cuda_v4, cpu_v2)
# 6. Packaging, tarball creation, and 1-command SCP retrieval instructions
#
# Usage:
#   bash run_presentation.sh
#   bash run_presentation.sh --outdir results/my_presentation
#   bash run_presentation.sh --skip-build
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${SCRIPT_DIR}/CMakeLists.txt" ]]; then
        PROJECT_ROOT="${SCRIPT_DIR}"
    else
        PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi
cd "${PROJECT_ROOT}"

# Parse optional arguments
SKIP_BUILD=0
QUICK=0
CUSTOM_OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)       CUSTOM_OUTDIR="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --quick)        QUICK=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--outdir DIR] [--skip-build] [--quick] [--help]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Output directories
TIMESTAMP="$(date -u +"%Y%m%d_%H%M%SZ")"
if [[ -n "${CUSTOM_OUTDIR}" ]]; then
    RESULTS_DIR="${CUSTOM_OUTDIR}"
else
    RESULTS_DIR="${PROJECT_ROOT}/results/presentation_${TIMESTAMP}"
fi

mkdir -p "${RESULTS_DIR}/plots"
mkdir -p "${RESULTS_DIR}/presentation_assets"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v5"
mkdir -p "${RESULTS_DIR}/astronomical_validation/v4"
mkdir -p "${RESULTS_DIR}/astronomical_validation/cpu_v2"
mkdir -p "${RESULTS_DIR}/tests"
mkdir -p "${RESULTS_DIR}/logs"

echo "========================================================================"
echo "    STANDALONE GPU BEAM TRACKER: PRESENTATION & VALIDATION SUITE        "
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

# 1. Environment & HPC Modules Setup
echo "--- [1/6] Environment Setup & HPC Modules ---"
if command -v module &> /dev/null; then
    echo "Loading Trillium modules: StdEnv/2023, gcc/12.3, cuda/12.6, python/3.11, scipy-stack..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || true
    module load python/3.11 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
fi

if [[ -d ".venv" ]]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OMP_PROC_BIND=true
export MPLCONFIGDIR="/tmp/matplotlib_${USER:-charts}_${TIMESTAMP}"
mkdir -p "${MPLCONFIGDIR}"

# Record hardware environment diagnostics
ENV_LOG="${RESULTS_DIR}/logs/env_info.txt"
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
    echo "=== Tooling & Compilers ==="
    python3 --version 2>/dev/null || true
    gcc --version 2>/dev/null | head -n 1 || true
    nvcc --version 2>/dev/null | tail -n 2 || true
} > "${ENV_LOG}"
cat "${ENV_LOG}"
echo ""

# 2. Build Targets
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "--- [2/6] Compiling C++/CUDA Targets (Release) ---"
    BUILD_LOG="${RESULTS_DIR}/logs/build.log"

    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DBEAMFORMER_ENABLE_CUDA=ON \
        -DBUILD_TESTING=ON \
        2>&1 | tee "${BUILD_LOG}"

    NPROC=$(nproc 2>/dev/null || echo 24)
    cmake --build build --config Release -j "${NPROC}" 2>&1 | tee -a "${BUILD_LOG}"
    echo "Compilation complete."
else
    echo "--- [2/6] Skipping Compilation (--skip-build) ---"
fi
echo ""

# 3. Execute Unit Tests
echo "--- [3/6] Running Unit & Integration Tests ---"
TEST_LOG="${RESULTS_DIR}/tests/unit_tests.log"
python3 -m unittest discover -s tests/python -p "test_*.py" -v 2>&1 | tee "${TEST_LOG}"
echo "Unit tests finished."
echo ""

# 4. Master Presentation Suite Generation (Early Generation ensures all presentation assets are immediately ready)
echo "--- [4/6] Generating Master Presentation Assets & Data ---"
python3 tools/generate_presentation_suite.py \
    --outdir "${RESULTS_DIR}/presentation_assets" \
    --engine cuda_v5

# Copy presentation figures and data to plots and root directory
cp "${RESULTS_DIR}/presentation_assets/"*.png "${RESULTS_DIR}/plots/" 2>/dev/null || true
cp "${RESULTS_DIR}/presentation_assets/"*.csv "${RESULTS_DIR}/" 2>/dev/null || true
cp "${RESULTS_DIR}/presentation_assets/"*.json "${RESULTS_DIR}/" 2>/dev/null || true
cp "${RESULTS_DIR}/presentation_assets/PRESENTATION_DECK.md" "${RESULTS_DIR}/" 2>/dev/null || true

echo "Presentation visual assets generated in ${RESULTS_DIR}/presentation_assets/"
echo ""

# 5. Astronomical Validation Suite
echo "--- [5/6] Executing Live Astronomical FRB Validation Suite ---"
ASTRO_SCRIPT="tools/run_astronomical_validation.py"
if [[ -f "${ASTRO_SCRIPT}" && -x "./build/run_tracker_stream" ]]; then
    echo ">>> Running Live Astronomical Validation on CUDA V5 (Full 4-Burst Benchmark Suite)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v5 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v5" || true

    echo ">>> Running Live Astronomical Validation on CUDA V4 (Full 4-Burst Benchmark Suite)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cuda_v4 \
        --burst all \
        --outdir "${RESULTS_DIR}/astronomical_validation/v4" || true

    echo ">>> Running Live Astronomical Validation on CPU Opt v2 (Canonical Benchmark)..."
    python3 "${ASTRO_SCRIPT}" \
        --engine cpu_v2 \
        --burst FRB20180916B_canonical \
        --outdir "${RESULTS_DIR}/astronomical_validation/cpu_v2" || true

    # Copy astro dashboards to plots directory
    cp "${RESULTS_DIR}/astronomical_validation/v5/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v5.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/v4/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_v4.png" 2>/dev/null || true
    cp "${RESULTS_DIR}/astronomical_validation/cpu_v2/astronomical_validation_dashboard.png" "${RESULTS_DIR}/plots/astronomical_validation_cpu_v2.png" 2>/dev/null || true
else
    echo "(Note: Live stream runner binary not found; astronomical validation will run in exact synthetic mode)"
fi
echo ""

# 6. Packaging & SCP Instructions
echo "--- [6/6] Packaging Artifacts & Finalizing ---"
LATEST_LINK="${PROJECT_ROOT}/results/presentation_latest"
rm -f "${LATEST_LINK}"
ln -s "${RESULTS_DIR}" "${LATEST_LINK}" 2>/dev/null || true

TAR_FILE="${RESULTS_DIR}.tar.gz"
echo "Creating compressed archive: ${TAR_FILE}..."
tar -czf "${TAR_FILE}" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")" 2>/dev/null || true

# Copy Slurm logs into results directory if running under batch
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${PROJECT_ROOT}/results/slurm_pres_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/results/slurm_pres_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true
fi

echo ""
echo "========================================================================"
echo "    PRESENTATION MATERIAL & VALIDATION SUITE GENERATION COMPLETE!       "
echo "========================================================================"
echo "Results Directory: ${RESULTS_DIR}"
echo "Presentation PNGs: ${RESULTS_DIR}/plots"
echo "Slide Deck Guide:  ${RESULTS_DIR}/PRESENTATION_DECK.md"
echo "Tarball Archive:   ${TAR_FILE}"
echo ""
echo "========================================================================"
echo "                 ONE-COMMAND SCP RETRIEVAL INSTRUCTIONS                 "
echo "========================================================================"
echo "To transfer all generated presentation assets and validation results to"
echo "your local computer in one command, run:"
echo ""
echo "  # Download entire results directory (all 300 DPI PNGs, CSVs, JSONs, slides):"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR} ./"
echo ""
echo "  # Or download the single compressed tarball archive:"
echo "  scp <user>@trillium.scinet.utoronto.ca:${TAR_FILE} ./"
echo ""
echo "  # Or download only the presentation images folder:"
echo "  scp -r <user>@trillium.scinet.utoronto.ca:${RESULTS_DIR}/plots ./"
echo "========================================================================"
