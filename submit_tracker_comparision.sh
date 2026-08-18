#!/bin/bash
#SBATCH --job-name=beam_tracker_bench_v2
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:40:00
#SBATCH --output=build/bench/slurm_v2_%j.out
#SBATCH --error=build/bench/slurm_v2_%j.err

cd $SLURM_SUBMIT_DIR

module load StdEnv/2023
module load gcc/12.3

mkdir -p build/bench

BIN=./build/benchmark_beam_tracker_opt_v2
N_TIME=15360
N_ANT=64
INT_SPEC=320
REPEAT=3
OUTDIR=build/bench

# Baseline naive tracker
${BIN} --n-time ${N_TIME} --n-ant ${N_ANT} --integration-spectra ${INT_SPEC} --threads 1 --repeat ${REPEAT} --outdir ${OUTDIR}

# OpenMP thread sweep up to 192 cores
for T in 1 2 4 8 16 32 64 128 192; do
    export OMP_NUM_THREADS=${T}
    ${BIN} --n-time ${N_TIME} --n-ant ${N_ANT} --integration-spectra ${INT_SPEC} --threads ${T} --repeat ${REPEAT} --outdir ${OUTDIR}
done

# NUMA interleave diagnostics
if command -v numactl >/dev/null 2>&1; then
    numactl --interleave=all ${BIN} --n-time ${N_TIME} --n-ant ${N_ANT} --integration-spectra ${INT_SPEC} --threads 192 --repeat ${REPEAT} --outdir ${OUTDIR}
fi