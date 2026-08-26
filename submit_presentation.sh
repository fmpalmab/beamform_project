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
# submit_presentation.sh
#
# Dedicated Slurm Batch Submission Script for SciNet Trillium GPU Nodes.
#
# Compiles Release targets with CUDA, runs Python unit tests ("Verython"),
# generates presentation visual suite & data tables (tools/generate_presentation_suite.py),
# executes astronomical validation, and packages all assets in a single timestamped
# directory with 1-command SCP retrieval.
#
# Usage:
#   sbatch submit_presentation.sh
#   sbatch submit_presentation.sh --skip-build
#   bash submit_presentation.sh                # Interactive debug
# ==============================================================================

set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

exec bash ./scripts/run_presentation.sh "$@"
