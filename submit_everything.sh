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
# submit_everything.sh (Root Entry Point)
#
# Master Slurm batch submission script for Trillium GPU nodes.
# Runs the full test suite, astronomical validation, and multi-engine benchmarks
# in one single job so all measurements share identical node conditions.
#
# Usage:
#   sbatch submit_everything.sh
#   sbatch submit_everything.sh --quick
#   bash submit_everything.sh (interactive / debugjob session)
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

bash scripts/slurm/submit_everything.sh "$@"
