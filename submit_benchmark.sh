#!/bin/bash
#SBATCH --job-name=beam_tracker_bench
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --output=build/bench/slurm_%j.out
#SBATCH --error=build/bench/slurm_%j.err

# Sweep benchmark_beam_tracker_opt (the optimized NAIVE tracker,
# src/beam_tracker_opt.cpp) against the reference naive tracker
# (src/beam_tracker.cpp) across a range of OpenMP thread counts so we get
# honest before/after latency numbers on a dedicated node instead of the
# oversubscribed login node (where 192 threads on a ~107k-cell-per-window
# kernel fork/join made the opt path SLOWER than serial; see the 0.71x
# result on tri-login04).
#
# The benchmark re-asserts naive == opt byte-equality in-process before timing,
# so a fast-but-wrong kernel is caught here too. All output goes to
# build/bench/, including a CSV summary with one row per (config) run.
#
# Repeat each config 3 times and take the median (the benchmark reports median
# + max itself). The naive path is serial (thread-count independent) so we run
# it once at the start as the reference baseline.

set -e

module load gcc 2>/dev/null || true   # best-effort; override locally if needed

mkdir -p build/bench

echo "=== Job started on $(hostname) at $(date) ==="
echo "Working directory: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-192}"
echo ""

BIN=./build/benchmark_beam_tracker_opt
N_TIME=15360
N_ANT=64
INT_SPEC=320
REPEAT=3
OUTDIR=build/bench

# Reference baseline: naive tracker at the default thread count (the naive
# path is serial, so this number is independent of --threads). Run it once
# explicitly with --threads 1 for a stable record.
echo "### baseline: naive (opt --threads 1, since naive is serial) ###"
${BIN} \
    --n-time ${N_TIME} \
    --n-ant ${N_ANT} \
    --integration-spectra ${INT_SPEC} \
    --threads 1 \
    --repeat ${REPEAT} \
    --outdir ${OUTDIR}

echo "### thread sweep ###"
for T in 1 2 4 8 16 32 64 128 192; do
    echo "----- OMP threads = ${T} -----"
    ${BIN} \
        --n-time ${N_TIME} \
        --n-ant ${N_ANT} \
        --integration-spectra ${INT_SPEC} \
        --threads ${T} \
        --repeat ${REPEAT} \
        --outdir ${OUTDIR}
    echo ""
done

echo "=== Job completed successfully at $(date) ==="
echo "CSV summary: ${OUTDIR}/benchmark_beam_tracker_opt.csv"
