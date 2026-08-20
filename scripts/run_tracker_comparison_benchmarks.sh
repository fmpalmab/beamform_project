#!/bin/bash
# scripts/run_tracker_comparison_benchmarks.sh
#
# End-to-end CPU-vs-GPU beam-tracker benchmark + visualization suite.
#
# Per the task spec (section 4 "Runner Shell Script") this script:
#   1. Builds the benchmark binary via:
#         cmake --build build --target benchmark_cuda_tracker_v2
#   2. Sweeps OpenMP thread counts (1, 4, 8, 16, 32, 64) and antenna sizes
#      (n_ant = 32, 64), invoking benchmarks/benchmark_cuda_tracker_v2 for every
#      (threads, n_ant) pair. Each run writes the *_summary.csv,
#      *_frame_latencies.csv, *_validation.csv, *_window_validation.csv, and
#      *_metadata.json files under a per-sweep output directory.
#   3. Executes tools/plot_tracker_comparison.py on the most-recent run's
#      CSVs and writes the final dashboard PNG to:
#         results/tracker_cpu_vs_gpu_dashboard.png
#
# Reuses the physical helpers, CSV column conventions, and Agg-backend plotting
# expectations already established by tools/plot_tracker_results.py,
# tools/plot_results.py, and tools/plot_tracker_comparison.py so the suite stays
# portable across Trillium / Compute Canada (plain python3 + numpy + matplotlib,
# matplotlib.use("Agg") when non-interactive -- the plotter sets that itself).
#
# The frame-latency / window-validation CSVs are TRUNCATED per run (see the
# benchmark tool's file header), so the dashboard reflects whichever run
# completed last. The summary CSV is APPENDED across runs, so the full sweep
# accumulates into a single comparison table at:
#   ${OUTDIR}/benchmark_cuda_tracker_v2_summary.csv

set -euo pipefail

# Resolve the project root from the script location so the suite works whether
# it is invoked as `./scripts/run_tracker_comparison_benchmarks.sh` from the
# repo root or `bash scripts/...` from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUILD_DIR="build"
BENCHMARK_TARGET="benchmark_cuda_tracker_v2"
BENCHMARK_BIN="${BUILD_DIR}/${BENCHMARK_TARGET}"
PLOTTER="tools/plot_tracker_comparison.py"

# Sweep parameters (spec section 4).
THREAD_COUNTS=(1 4 8 16 32 64)
ANT_COUNTS=(32 64)

# Benchmark-size knobs -- kept at the benchmark tool's defaults unless
# overridden via environment, so the suite is easy to retune without editing.
N_TIME="${N_TIME:-15360}"
INTEGRATION_SPECTRA="${INTEGRATION_SPECTRA:-320}"
REPEAT="${REPEAT:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
WINDOW_REPEATS="${WINDOW_REPEATS:-3}"

# Output layout: a fresh per-invocation sweep directory keeps the TRUNCATED
# per-run CSVs (frame_latencies / window_validation / metadata) isolated, while
# the APPENDED summary CSV accumulates across the whole sweep. The dashboard
# PNG is written to the standard results/ location.
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
OUTDIR="${OUTDIR:-results/tracker_sweep_${TIMESTAMP}}"
FINAL_PNG="${FINAL_PNG:-results/tracker_cpu_vs_gpu_dashboard.png}"

mkdir -p "${OUTDIR}"
mkdir -p "$(dirname "${FINAL_PNG}")"

echo "=== CPU vs GPU Beam Tracker Comparison Suite ==="
echo "project root : ${PROJECT_ROOT}"
echo "build dir    : ${BUILD_DIR}"
echo "output dir   : ${OUTDIR}"
echo "dashboard    : ${FINAL_PNG}"
echo "sweep        : threads=${THREAD_COUNTS[*]}  n_ant=${ANT_COUNTS[*]}"
echo

# ---------------------------------------------------------------------------
# Step 1 -- Build the benchmark binary.
# ---------------------------------------------------------------------------
echo "=== [1/3] Building ${BENCHMARK_TARGET} ==="
if [ ! -d "${BUILD_DIR}" ]; then
    echo "(configure) cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release"
    cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
fi
cmake --build "${BUILD_DIR}" --target "${BENCHMARK_TARGET}" --config Release -j"$(nproc 2>/dev/null || echo 2)"

if [ ! -x "${BENCHMARK_BIN}" ]; then
    echo "FATAL: ${BENCHMARK_BIN} not found after build (CUDA likely unavailable)." >&2
    echo "       benchmark_cuda_tracker_v2 requires a CUDA-enabled build (see CMakeLists.txt)." >&2
    exit 1
fi
echo

# ---------------------------------------------------------------------------
# Step 2 -- Sweep (threads x n_ant).
# ---------------------------------------------------------------------------
echo "=== [2/3] Running benchmark sweep ==="
LAST_PREFIX=""
for N_ANT in "${ANT_COUNTS[@]}"; do
    for THREADS in "${THREAD_COUNTS[@]}"; do
        echo "-> n_ant=${N_ANT}  OMP_NUM_THREADS=${THREADS}  n_time=${N_TIME}  spectra=${INTEGRATION_SPECTRA}"
        OMP_NUM_THREADS="${THREADS}" "${BENCHMARK_BIN}" \
            --n-time "${N_TIME}" \
            --n-ant "${N_ANT}" \
            --integration-spectra "${INTEGRATION_SPECTRA}" \
            --threads "${THREADS}" \
            --repeat "${REPEAT}" \
            --warmup-runs "${WARMUP_RUNS}" \
            --window-repeats "${WINDOW_REPEATS}" \
            --outdir "${OUTDIR}"
        # Benchmark writes files with prefix "${OUTDIR}/benchmark_cuda_tracker_v2".
        # The per-run (TRUNCATED) CSVs reflect the most recent invocation, so the
        # dashboard will visualize this final run of the sweep.
        LAST_PREFIX="${OUTDIR}/benchmark_cuda_tracker_v2"
    done
done
echo

echo "Sweep summary CSV accumulated at: ${LAST_PREFIX}_summary.csv"
echo "Per-run CSVs (last run): ${LAST_PREFIX}_{frame_latencies,validation,window_validation}.csv"
echo "Run metadata JSON      : ${LAST_PREFIX}_metadata.json"
echo

# ---------------------------------------------------------------------------
# Step 3 -- Render the dashboard PNG.
# ---------------------------------------------------------------------------
echo "=== [3/3] Rendering dashboard ==="
# Find a python3 that has numpy + matplotlib. Prefer the project's own python3,
# then fall back to the module form `python3 -m ...` which is what the
# benchmark/CI nodes (no conda, just a system python3) expose.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c "import numpy, matplotlib" >/dev/null 2>&1; then
    echo "(warn) ${PYTHON_BIN} missing numpy/matplotlib; dashboard plotting will fail." >&2
fi

"${PYTHON_BIN}" "${PLOTTER}" \
    --input-prefix "${LAST_PREFIX}" \
    --output "${FINAL_PNG}" \
    --budget-ms 0.5

echo
echo "=== Suite complete ==="
echo "Dashboard PNG : ${FINAL_PNG}"
echo "Output dir    : ${OUTDIR}"
