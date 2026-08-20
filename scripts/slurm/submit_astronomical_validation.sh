#!/bin/bash
#SBATCH --job-name=astronomical_validation
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=results/astronomical_validation/slurm_astronomical_%j.out
#SBATCH --error=results/astronomical_validation/slurm_astronomical_%j.err

set -e

cd "${SLURM_SUBMIT_DIR}"

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load python/3.11

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}/tools:${PYTHONPATH}"
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=true

mkdir -p results/astronomical_validation

echo "=== Astronomical Validation Job started on $(hostname) at $(date) ==="
nvidia-smi || echo "(warn) nvidia-smi failed -- GPU may not be allocated correctly"
echo ""

# 1. Build C++ run_tracker_stream CLI executable
echo "--- Building run_tracker_stream CLI Bridge ---"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target run_tracker_stream -j 8
echo ""

# 2. Run Astronomical Validation Suite for CUDA Fused WarpShuffle (Best GPU Kernel)
echo "--- Running Astronomical Validation: CUDA FusedWarpShuffle (cuda_fws) ---"
python tools/run_astronomical_validation.py \
    --engine cuda_fws \
    --burst all \
    --outdir results/astronomical_validation

echo "=== Astronomical Validation Job completed successfully at $(date) ==="
echo "Report JSON written to:"
echo "  results/astronomical_validation/astronomical_validation_report.json"
echo "Dashboard Plot written to:"
echo "  results/astronomical_validation/astronomical_validation_dashboard.png"
