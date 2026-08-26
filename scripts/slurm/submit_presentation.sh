#!/bin/bash
#SBATCH --job-name=beamform_presentation
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=results/slurm_pres_%j.out
#SBATCH --error=results/slurm_pres_%j.err

# ==============================================================================
# scripts/slurm/submit_presentation.sh
#
# Dedicated Slurm Batch Submission Script for SciNet Trillium GPU Nodes.
# (Delegates to project root scripts/run_presentation.sh)
# ==============================================================================

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${SCRIPT_DIR}/../.."
fi

exec bash ./scripts/run_presentation.sh "$@"
