#!/bin/bash
#SBATCH --job-name=tracker_cpu_gpu_comparison
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --output=build/bench/slurm_comparison_%j.out
#SBATCH --error=build/bench/slurm_comparison_%j.err

# Full CPU-vs-GPU comparison sweep, superseding the CPU-only version of this
# script (which used benchmark_beam_tracker_opt_v2 and swept only naive/v1/v2
# on a 192-core CPU node -- see submit_benchmark_v2.sh for that CPU-only
# path, left untouched).
#
# This version switches to tools/benchmark_cuda_tracker_v2 (see
# benchmark_cuda_tracker_v2.cpp), which times ALL SIX engines together in a
# single invocation -- CPU Naive, CPU Opt v1, CPU Opt v2, and the three CUDA
# v2 kernels (TwoPass, Fused, WarpReduction) -- and re-verifies tolerance
# equality of every non-naive engine against the CPU naive reference before
# timing. Each invocation appends one row per engine-pair to
# *_summary.csv, so the (n_ant x threads) sweep below accumulates into a
# single comparison table across CPU thread counts AND GPU kernels.
#
# Requesting --gpus-per-node=1 gets a quarter node on Trillium's GPU
# subcluster: 24 cores / ~188GiB RAM (not the 96/192 you'd get from a
# whole/CPU-only node), so the OpenMP thread sweep below is capped at 24
# instead of the CPU-only script's 1..192. If you need the 128/192-thread
# points too, run submit_benchmark_v2.sh (CPU-only, no GPU request) alongside
# this and merge the CSVs -- a single job can't hold both a 192-core CPU
# allocation and a GPU allocation on Trillium.
#
# Per Trillium's docs, job output must land on the scratch file system, and
# jobs should be submitted from $SCRATCH -- submit from there so the relative
# build/bench path resolves correctly:
#   $ sbatch submit_tracker_comparision.sh

set -e

cd "${SLURM_SUBMIT_DIR}"

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6

mkdir -p build/bench

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi || echo "(warn) nvidia-smi failed -- GPU may not be allocated correctly"
echo ""

BIN=./build/benchmark_cuda_tracker_v2
N_TIME=15360
INT_SPEC=320
REPEAT=3
WINDOW_REPEATS=3
OUTDIR=build/bench

# Thread sweep is capped at the 24 cores available to a single-GPU
# (quarter-node) allocation -- see note above.
THREAD_COUNTS=(1 2 4 8 16 24)
ANT_COUNTS=(32 64)

for N_ANT in "${ANT_COUNTS[@]}"; do
    for T in "${THREAD_COUNTS[@]}"; do
        export OMP_NUM_THREADS=${T}
        echo "----- n_ant=${N_ANT}  OMP threads=${T} -----"
        ${BIN} \
            --n-time ${N_TIME} \
            --n-ant ${N_ANT} \
            --integration-spectra ${INT_SPEC} \
            --threads ${T} \
            --repeat ${REPEAT} \
            --window-repeats ${WINDOW_REPEATS} \
            --outdir ${OUTDIR}
        echo ""
    done
done

echo "=== Job completed successfully at $(date) ==="
echo "Summary CSV (CPU naive/v1/v2 + CUDA TwoPass/Fused/WarpReduction, all sweep points):"
echo "  ${OUTDIR}/benchmark_cuda_tracker_v2_summary.csv"
echo "Per-window latency / validation CSVs (reflect the LAST sweep point only):"
echo "  ${OUTDIR}/benchmark_cuda_tracker_v2_frame_latencies.csv"
echo "  ${OUTDIR}/benchmark_cuda_tracker_v2_validation.csv"
echo "  ${OUTDIR}/benchmark_cuda_tracker_v2_window_validation.csv"
