#!/bin/bash
#SBATCH --job-name=beamform_pres_val
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm_pres_%j.out
#SBATCH --error=results/slurm_pres_%j.err

# ==============================================================================
# submit_presentation_and_validation.sh
#
# Dedicated Slurm Batch Submission Script for SciNet Trillium GPU Nodes.
# (Delegates to project root scripts/run_presentation.sh)
# ==============================================================================

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

exec bash ./scripts/run_presentation.sh "$@"
