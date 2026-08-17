#!/usr/bin/env bash
#
# run_naive_cpu_tests.sh
# -----------------------------------------------------------------------------
# Reproducible test + benchmark runner for the *naive* CPU beamformer
# (src/cpu_beamformer.cpp). It drives the `beam_tracker_naive_cpu_test_suite` executable
# over a wide matrix of configurations:
#
#   * Trivial  correctness tests  (degenerate sizes, exact closed-form answers)
#   * Base     correctness tests  (core math, unitarity, packed/unpacked equiv.)
#   * Complex  correctness tests  (physical localization, multi-time, contracts)
#   * Benchmark runtime sweep     (per-frame time vs the 0.5 ms/frame target)
#
# All artifacts (build log, run log, benchmark CSV, stdout/exit codes) are
# written under a timestamped results directory so every run is reproducible
# and comparable against historical runs.
#
# Usage:
#   scripts/run_naive_cpu_tests.sh                   # full reproducible run
#   scripts/run_naive_cpu_tests.sh --quick           # small matrix, fast bench
#   scripts/run_naive_cpu_tests.sh --skip-bench      # correctness only
#   scripts/run_naive_cpu_tests.sh --rebuild         # force a clean rebuild
#
# Environment overrides:
#   BUILD_DIR   cmake build directory       (default: build_beam_tracker_naive_cpu_tests)
#   RESULT_DIR  results root                (default: results_beam_tracker_naive_cpu_tests)
#   CMAKE_GEN   cmake generator             (default: empty -> cmake default)
#   N_ANT       comma list of n_ant         (default: 32,64)
#   TIMES       comma list of n_time        (default: 15360,24576,30720)
#   BEAMS       comma list of n_beams       (default: 1,5,16,32,64,128)
#   WARMUP      benchmark warmup reps       (default: 2)
#   REPETITIONS benchmark measured reps     (default: 5)
#   SEED        benchmark noise seed        (default: 1)
# -----------------------------------------------------------------------------

set -euo pipefail

# Resolve workspace directory from the script location so it works regardless of
# the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build_beam_tracker_naive_cpu_tests}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results_beam_tracker_naive_cpu_tests}"
CMAKE_GEN="${CMAKE_GEN:-}"

N_ANT="${N_ANT:-32,64}"
TIMES="${TIMES:-15360,24576,30720}"
BEAMS="${BEAMS:-1,5,16,32,64,128}"
WARMUP="${WARMUP:-2}"
REPETITIONS="${REPETITIONS:-5}"
SEED="${SEED:-1}"

QUICK=0
REBUILD=0
SKIP_BENCH=0
SKIP_CORRECT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)        QUICK=1; shift ;;
    --rebuild)      REBUILD=1; shift ;;
    --skip-bench)   SKIP_BENCH=1; shift ;;
    --skip-correct) SKIP_CORRECT=1; shift ;;
    --help|-h)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# A quick matrix keeps the run time short while still exercising the extremes
# of the configuration space (smallest and largest sizes).
if [[ "${QUICK}" -eq 1 ]]; then
  N_ANT="32,64"
  TIMES="15360,30720"
  BEAMS="1,64"
  REPETITIONS=2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULT_DIR}/${STAMP}"
mkdir -p "${RUN_DIR}"

BUILD_LOG="${RUN_DIR}/build.log"
CORRECT_LOG="${RUN_DIR}/correctness.log"
BENCH_LOG="${RUN_DIR}/benchmark.log"
BENCH_CSV="${RUN_DIR}/benchmark.csv"
SUMMARY_LOG="${RUN_DIR}/summary.txt"

echo "naive CPU test+benchmark runner"
echo "  project root : ${PROJECT_ROOT}"
echo "  build dir    : ${BUILD_DIR}"
echo "  run results  : ${RUN_DIR}"
echo "  matrix       : n_ant={${N_ANT}} n_time={${TIMES}} n_beams={${BEAMS}}"
echo "  benchmark    : warmup=${WARMUP} reps=${REPETITIONS} seed=${SEED}"
echo

# -----------------------------------------------------------------------------
# 1. Configure + build the beam_tracker_naive_cpu_test_suite target.
# -----------------------------------------------------------------------------
if [[ "${REBUILD}" -eq 1 && -d "${BUILD_DIR}" ]]; then
  echo "  [build] cleaning previous build directory"
  rm -rf "${BUILD_DIR}"
fi

if [[ ! -d "${BUILD_DIR}" || ! -x "${BUILD_DIR}/beam_tracker_naive_cpu_test_suite" ]]; then
  echo "  [build] configuring with cmake -> ${BUILD_LOG}"
  CMAKE_ARGS=(-S "${PROJECT_ROOT}" -B "${BUILD_DIR}"
              -DCMAKE_BUILD_TYPE=Release
              -DBEAMFORMER_ENABLE_CUDA=OFF)
  if [[ -n "${CMAKE_GEN}" ]]; then
    CMAKE_ARGS+=( -G "${CMAKE_GEN}" )
  fi
  cmake "${CMAKE_ARGS[@]}" >"${BUILD_LOG}" 2>&1
fi

echo "  [build] compiling beam_tracker_naive_cpu_test_suite (Release -O2)"
cmake --build "${BUILD_DIR}" --target beam_tracker_naive_cpu_test_suite --parallel >>"${BUILD_LOG}" 2>&1

SUITE="${BUILD_DIR}/beam_tracker_naive_cpu_test_suite"
if [[ ! -x "${SUITE}" ]]; then
  echo "  [build] FAILED: executable not found at ${SUITE}" >&2
  echo "  see ${BUILD_LOG}" >&2
  exit 1
fi
echo "  [build] done"
echo

# -----------------------------------------------------------------------------
# 2. Correctness tests (trivial + base + complex).
# -----------------------------------------------------------------------------
CORRECT_RC=unknown
if [[ "${SKIP_CORRECT}" -eq 0 ]]; then
  echo "  [correctness] running trivial + base + complex tests -> ${CORRECT_LOG}"
  if "${SUITE}" --skip-benchmark >"${CORRECT_LOG}" 2>&1; then
    CORRECT_RC=0
    echo "  [correctness] PASS (exit 0)"
  else
    CORRECT_RC=$?
    echo "  [correctness] FAIL (exit ${CORRECT_RC}) -- see ${CORRECT_LOG}"
  fi
  # Echo the summary line so it is visible inline.
  grep -E "tests (ran|passed|failed)" "${CORRECT_LOG}" || true
  echo
fi

# -----------------------------------------------------------------------------
# 3. Benchmark sweep across the configuration matrix.
# -----------------------------------------------------------------------------
BENCH_RC=unknown
if [[ "${SKIP_BENCH}" -eq 0 ]]; then
  echo "  [benchmark] sweeping matrix -> ${BENCH_LOG} (csv: ${BENCH_CSV})"
  BENCH_ARGS=(--skip-trivial --skip-base --skip-complex
              --warmup "${WARMUP}"
              --repetitions "${REPETITIONS}"
              --seed "${SEED}"
              --n-ant "${N_ANT}"
              --times "${TIMES}"
              --beams "${BEAMS}"
              --metrics "${BENCH_CSV}")
  if "${SUITE}" "${BENCH_ARGS[@]}" >"${BENCH_LOG}" 2>&1; then
    BENCH_RC=0
  else
    BENCH_RC=$?
  fi
  echo "  [benchmark] exit ${BENCH_RC} -- see ${BENCH_LOG}"
  echo
fi

# -----------------------------------------------------------------------------
# 4. Summarize the benchmark CSV: how many configs met the 0.5 ms/frame target.
# -----------------------------------------------------------------------------
OVER_COUNT=0
MEET_COUNT=0
WORST_PER_FRAME_MS=0
WORST_CFG=""
if [[ -f "${BENCH_CSV}" ]]; then
  # CSV columns: n_time,n_freq,n_ant,n_beams,compute_ms,per_frame_ms,
  #              outputs_per_second,complex_gmac_per_second,meets_target_0p5ms
  while IFS=, read -r nt nf na nb cms pfms ops gmacs meets; do
    [[ "${nt}" == "n_time" ]] && continue
    if [[ "${meets}" == "1" ]]; then
      MEET_COUNT=$((MEET_COUNT+1))
    else
      OVER_COUNT=$((OVER_COUNT+1))
    fi
    # Track the worst per-frame time.
    awk_check="$(awk -v a="${pfms}" -v b="${WORST_PER_FRAME_MS}" 'BEGIN{print (a>b)?1:0}')"
    if [[ "${awk_check}" == "1" ]]; then
      WORST_PER_FRAME_MS="${pfms}"
      WORST_CFG="n_time=${nt} n_ant=${na} n_beams=${nb}"
    fi
  done < "${BENCH_CSV}"
fi

# -----------------------------------------------------------------------------
# 5. Write a machine-readable + human-readable summary.
# -----------------------------------------------------------------------------
{
  echo "Naive CPU test+benchmark run: ${STAMP}"
  echo "  build_dir       : ${BUILD_DIR}"
  echo "  run_dir         : ${RUN_DIR}"
  echo "  matrix n_ant    : ${N_ANT}"
  echo "  matrix n_time   : ${TIMES}"
  echo "  matrix n_beams  : ${BEAMS}"
  echo "  warmup/reps/seed: ${WARMUP}/${REPETITIONS}/${SEED}"
  echo
  if [[ "${SKIP_CORRECT}" -eq 0 ]]; then
    echo "  correctness exit: ${CORRECT_RC}"
    grep -E "tests (ran|passed|failed)" "${CORRECT_LOG}" 2>/dev/null || true
  else
    echo "  correctness exit: skipped"
  fi
  echo
  if [[ "${SKIP_BENCH}" -eq 0 ]]; then
    echo "  benchmark exit  : ${BENCH_RC}"
    echo "  total configs   : $((MEET_COUNT+OVER_COUNT))"
    echo "  meets 0.5ms/frame: ${MEET_COUNT}"
    echo "  over  0.5ms/frame: ${OVER_COUNT}"
    echo "  worst per-frame : ${WORST_PER_FRAME_MS} ms  [${WORST_CFG}]"
  else
    echo "  benchmark exit  : skipped"
  fi
} | tee "${SUMMARY_LOG}"

echo
echo "  summary written to ${SUMMARY_LOG}"

# Non-zero exit code if correctness failed or (for bench runs) any config
# exceeded the target. A correct-but-slow result still returns 0 so CI can
# distinguish "logic broke" (which we must fix) from "too slow" (known).
FINAL_RC=0
if [[ "${SKIP_CORRECT}" -eq 0 && "${CORRECT_RC}" != "0" ]]; then
  FINAL_RC=1
fi
echo "  final exit code: ${FINAL_RC}"
exit "${FINAL_RC}"
