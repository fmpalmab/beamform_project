#!/bin/bash
#SBATCH --job-name=v3_complete_suite
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=results/v3_suite/slurm_v3_suite_%j.out
#SBATCH --error=results/v3_suite/slurm_v3_suite_%j.err

# ==============================================================================
# submit_v3_complete_suite.sh
#
# Slurm batch submission script for Trillium GPU nodes.
# Runs the full V3 pipeline (Build, Unit Tests, Astro Validation, Benchmark Sweep).
#
# Usage:
#   sbatch scripts/slurm/submit_v3_complete_suite.sh
# ==============================================================================

set -e

cd "${SLURM_SUBMIT_DIR}"

mkdir -p results/v3_suite

# Execute master validation script
bash scripts/run_all_v3_validation.sh
