#!/bin/bash
#SBATCH --job-name=beam_tracker_bench_v2
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:40:00
#SBATCH --output=build/bench/slurm_v2_%j.out
#SBATCH --error=build/bench/slurm_v2_%j.err

# Sweep benchmark_beam_tracker_opt_v2 (the v2 optimized NAIVE tracker,
# src/beam_tracker_opt_v2.cpp) against v1 (src/beam_tracker_opt.cpp) and the
# reference naive tracker (src/beam_tracker.cpp) across the same OpenMP thread
# sweep as submit_benchmark.sh, at the same config (n_time=15360, n_freq=336,
# n_ant=64, integration_spectra=320) and same 3-repeat methodology.
#
# This is deliberately a SEPARATE script so the v1 submit_benchmark.sh stays
# bit-for-bit untouched (v1 files are subject to the zero-diff hard constraint).
#
# ADDITIONALLY runs ONE diagnostic pair with no code involved: v1 and v2 each
# under `numactl --interleave=all` at 192 threads, to sanity-check the NUMA
# first-touch hypothesis behind FIX 1 independent of whether the code change
# itself moved the needle.
#
# The benchmark re-asserts naive==v1==v2 byte-equality in-process before timing.
# All output goes to build/bench/, including a CSV summary with one row per
# config run.

set -e

module load gcc 2>/dev/null || true   # best-effort; override locally if needed

mkdir -p build/bench

echo "=== Job started on $(hostname) at $(date) ==="
echo "Working directory: $(pwd)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-192}"
echo ""

BIN=./build/benchmark_beam_tracker_opt_v2
N_TIME=15360
N_ANT=64
INT_SPEC=320
REPEAT=3
OUTDIR=build/bench

# Reference baseline: naive tracker (serial, thread-count independent). Run
# once explicitly with --threads 1 for a stable record alongside the opt/v2
# sweep rows.
echo "### baseline: naive (opt + v2 at --threads 1; naive is serial) ###"
${BIN} \
    --n-time ${N_TIME} \
    --n-ant ${N_ANT} \
    --integration-spectra ${INT_SPEC} \
    --threads 1 \
    --repeat ${REPEAT} \
    --outdir ${OUTDIR}

# Full thread sweep: v1 vs. v2 vs. naive under the same config, at each thread
# count from the v1 record (1/2/4/8/16/32/64/128/192).
echo "### thread sweep (naive vs v1 vs v2) ###"
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

# --- NUMA diagnostic pair (FIX 1 hypothesis sanity-check, no code involved) ---
# Run v1 and v2 each at 192 threads under `numactl --interleave=all`, which
# defeats first-touch NUMA placement by interleaving every page round-robin
# across all nodes. If FIX 1's first-touch fix actually helped in the default
# run, then a separate interleaved run of v1 should close part of the gap to
# the default run of v2 (because interleaving also removes the one-socket
# first-touch penalty on a serial memset). If interleaving does NOT change
# things much, the NUMA hypothesis is weak and FIX 1 did not matter at this
# scale. We do NOT require numactl to be installed — skip the diagnostic pair
# gracefully if it is missing so the main sweep still completes on any node.
if command -v numactl >/dev/null 2>&1; then
    echo "### NUMA diagnostic: numactl --interleave=all at 192 threads ###"
    echo "----- v1 under numactl --interleave=all, 192 threads -----"
    numactl --interleave=all ${BIN} \
        --n-time ${N_TIME} \
        --n-ant ${N_ANT} \
        --integration-spectra ${INT_SPEC} \
        --threads 192 \
        --repeat ${REPEAT} \
        --outdir ${OUTDIR}
    echo ""
    echo "----- v2 under numactl --interleave=all, 192 threads -----"
    numactl --interleave=all ${BIN} \
        --n-time ${N_TIME} \
        --n-ant ${N_ANT} \
        --integration-spectra ${INT_SPEC} \
        --threads 192 \
        --repeat ${REPEAT} \
        --outdir ${OUTDIR}
else
    echo "### NUMA diagnostic SKIPPED: numactl not installed on this node ###"
fi

echo "=== Job completed successfully at $(date) ==="
echo "CSV summary: ${OUTDIR}/benchmark_beam_tracker_opt_v2.csv"
