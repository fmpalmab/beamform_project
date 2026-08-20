#!/bin/bash
#SBATCH --job-name=beamform_everything
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm_everything_%j.out
#SBATCH --error=results/slurm_everything_%j.err

# ==============================================================================
# submit_everything.sh
#
# Master Slurm batch submission script for Trillium GPU nodes.
# Runs ALL unit tests, astronomical validations, and benchmark sweeps under the
# exact same runtime and hardware conditions, packaging all outputs, CSVs,
# logs, reports, and plots into a single directory for 1-command SCP retrieval.
#
# Hardware Request on Trillium GPU Subcluster:
#   - 1 GPU (Quarter Node): 24 CPU cores, ~188 GiB RAM, 1 NVIDIA GPU
#
# Usage:
#   sbatch scripts/slurm/submit_everything.sh
#   sbatch scripts/slurm/submit_everything.sh --quick
# ==============================================================================

set -uo pipefail

# Ensure working directory is the submit directory ($SCRATCH per Trillium guidelines)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

mkdir -p results

# Execute the master runner script
bash scripts/run_everything.sh "$@"

# Copy Slurm logs into the generated results directory if available
if [[ -n "${SLURM_JOB_ID:-}" && -d "results/everything_latest" ]]; then
    cp "results/slurm_everything_${SLURM_JOB_ID}.out" "results/everything_latest/" 2>/dev/null || true
    cp "results/slurm_everything_${SLURM_JOB_ID}.err" "results/everything_latest/" 2>/dev/null || true
fi
