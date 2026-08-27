#!/bin/bash
# ==============================================================================
# scripts/smoke_test_all_slurms.sh
#
# Quick Smoke Test Suite for all new Slurm workflows.
# Runs fast, lightweight iterations (10-30 seconds) to verify:
# 1. Compilation & linking
# 2. CUDA kernel execution
# 3. Output file generation & formatting
# 4. Plot generation & directory packaging
#
# Usage:
#   bash scripts/smoke_test_all_slurms.sh
#   sbatch --time=00:10:00 --gpus=1 --cpus-per-task=8 scripts/smoke_test_all_slurms.sh
# ==============================================================================

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "          RUNNING RAPID SMOKE TESTS FOR ALL SLURM SCRIPTS               "
echo "========================================================================"
echo "Host Node   : $(hostname)"
echo "Start Time  : $(date -u)"
echo "========================================================================"
echo ""

# 1. Smoke Test: Target Simulations (Vela Pulsar single run, 32 antennas)
echo ">>> [1/2] Smoke Testing Target Simulations (submit_target_simulations.sh)..."
bash submit_target_simulations.sh \
    --engine cuda_v5 \
    --antennas 32 \
    --target vela

if [[ $? -eq 0 ]]; then
    echo "✅ Target Simulations Smoke Test: PASSED"
else
    echo "❌ Target Simulations Smoke Test: FAILED"
    exit 1
fi
echo ""

# 2. Smoke Test: Continuous Stress Test (15-second quick endurance check)
echo ">>> [2/2] Smoke Testing Continuous Stress Test (submit_stress_test.sh)..."
bash submit_stress_test.sh \
    --duration 15 \
    --engine cuda_v5 \
    --antennas 32 \
    --upchan 32 \
    --skip-build

if [[ $? -eq 0 ]]; then
    echo "✅ Continuous Stress Test Smoke Test: PASSED"
else
    echo "❌ Continuous Stress Test Smoke Test: FAILED"
    exit 1
fi
echo ""

echo "========================================================================"
echo "       🎉 ALL SMOKE TESTS COMPLETED SUCCESSFULLY IN RECORD TIME!        "
echo "========================================================================"
