#!/bin/bash
#SBATCH --job-name=beam_tracker_gpu_bench
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:40:00
#SBATCH --output=build/bench/slurm_gpu_%j.out
#SBATCH --error=build/bench/slurm_gpu_%j.err

# Runs benchmarks/benchmark_cuda_tracker_v2 (see benchmark_cuda_tracker_v2.cpp) on
# Trillium's GPU subcluster. That binary times ALL SIX engines in one
# invocation -- CPU Naive, CPU Opt v1, CPU Opt v2, and the three CUDA v2
# kernels (TwoPass, Fused, WarpReduction) -- and re-verifies tolerance
# equality of every non-naive engine against the CPU naive reference before
# timing, so a fast-but-wrong kernel gets caught instead of benchmarked.
#
# This script is the single-GPU counterpart to submit_benchmark_v2.sh (the
# CPU-only, 192-core sweep): here we request exactly one GPU, which on
# Trillium's GPU subcluster is a quarter node -- 24 cores and ~188GiB RAM,
# not the full 192 cores the CPU-only script gets. Do NOT request
# --gpus-per-node=2 or 3 (Trillium only allows exactly 1 GPU or a multiple
# of 4); a second GPU means moving to --gpus-per-node=4 (a whole node).
#
# Per Trillium's docs, job output must land on the scratch file system, and
# jobs should be *submitted* from $SCRATCH (home/project are read-only on
# compute nodes) -- submit this script from your $SCRATCH directory so the
# relative build/bench path below resolves there:
#   $ sbatch scripts/slurm/submit_gpu_benchmark.sh
#
# For a quick sanity check before committing to a full compute-partition
# job, you can instead run this same script via debugjob, or submit with
# `-p debug` (GPU debug partition allows 1 GPU jobs up to 2 hours).

set -e

cd "${SLURM_SUBMIT_DIR}"

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6

mkdir -p build/bench

echo "=== Job started on $(hostname) at $(date) ==="
echo "Working directory: $(pwd)"
nvidia-smi || echo "(warn) nvidia-smi failed -- GPU may not be allocated correctly"
echo ""

BIN=./build/benchmark_cuda_tracker_v2
N_TIME=15360
N_ANT=64
INT_SPEC=320
REPEAT=3
WINDOW_REPEATS=3
THREADS=${SLURM_CPUS_PER_TASK:-24}
OUTDIR=build/bench

export OMP_NUM_THREADS=${THREADS}

echo "### GPU benchmark: CPU naive/v1/v2 + CUDA TwoPass/Fused/WarpReduction ###"
echo "n_ant=${N_ANT}  n_time=${N_TIME}  integration_spectra=${INT_SPEC}  OMP_threads=${THREADS}"
${BIN} \
    --n-time ${N_TIME} \
    --n-ant ${N_ANT} \
    --integration-spectra ${INT_SPEC} \
    --threads ${THREADS} \
    --repeat ${REPEAT} \
    --window-repeats ${WINDOW_REPEATS} \
    --outdir ${OUTDIR}

echo ""
echo "=== Job completed successfully at $(date) ==="
echo "Summary CSV: ${OUTDIR}/benchmark_cuda_tracker_v2_summary.csv"
echo "Per-window latency CSV: ${OUTDIR}/benchmark_cuda_tracker_v2_frame_latencies.csv"
echo "Validation CSV: ${OUTDIR}/benchmark_cuda_tracker_v2_validation.csv"
echo "Metadata JSON: ${OUTDIR}/benchmark_cuda_tracker_v2_metadata.json"
