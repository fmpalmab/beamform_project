#!/bin/bash
# ==============================================================================
# submit_overnight_campaign.sh
#
# Master 10-Hour Automated Overnight Slurm Campaign Orchestrator.
#
# Sequentially schedules and chains the complete suite of:
# 1. Target Simulations across all arrays (64, 128, 256 Antennas: Vela, Sun, FRB)
# 2. Continuous 2-Hour Line-Rate Stress Tests (64, 128, 256 Antennas + M=16/32)
#
# Uses Slurm job dependencies (--dependency=afterany) so:
# - Only 1 GPU is active at any time (never exceeds account concurrency limits).
# - Jobs start automatically one after another across the 10-hour window.
# - Even if any job fails or finishes early, the next configuration executes.
#
# Usage:
#   bash submit_overnight_campaign.sh
# ==============================================================================

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

CAMPAIGN_ID="$(date -u +"%Y%m%d_%H%M%SZ")"
CAMPAIGN_LOG="${PROJECT_ROOT}/results/campaign_${CAMPAIGN_ID}.log"
mkdir -p "${PROJECT_ROOT}/results"

echo "========================================================================"
echo "    LAUNCHING 10-HOUR OVERNIGHT BENCHMARK & STRESS TEST CAMPAIGN       "
echo "========================================================================"
echo "Campaign ID  : ${CAMPAIGN_ID}"
echo "Start Time   : $(date -u)"
echo "Host Node    : $(hostname)"
echo "Log File     : ${CAMPAIGN_LOG}"
echo "========================================================================"
echo ""

JOB_IDS=()

submit_chained_job() {
    local job_name="$1"
    local script_cmd="$2"
    local prev_job="${3:-}"

    local sbatch_opts=()
    if [[ -n "${prev_job}" ]]; then
        sbatch_opts+=("--dependency=afterany:${prev_job}")
    fi

    # Submit job
    local submit_out
    submit_out=$(sbatch "${sbatch_opts[@]}" ${script_cmd})
    local job_id
    job_id=$(echo "${submit_out}" | awk '{print $NF}')

    echo "  [Scheduled] ${job_name}"
    echo "              Job ID     : ${job_id}"
    if [[ -n "${prev_job}" ]]; then
        echo "              Dependency : Runs after Job ${prev_job}"
    else
        echo "              Dependency : Starts immediately"
    fi
    echo "              Command    : ${script_cmd}"
    echo ""

    echo "${job_id}|${job_name}|${script_cmd}" >> "${CAMPAIGN_LOG}"
    JOB_IDS+=("${job_id}")
    LAST_JOB_ID="${job_id}"
}

LAST_JOB_ID=""

echo "--- [PHASE 1] Target Simulations (Vela, Sun, FRB) across Antenna Arrays ---"

# Job 1: 64 Antennas Target Simulations (All targets: Vela, Sun, FRB)
submit_chained_job "Target Sims (64 Antennas, All Targets)" \
    "submit_target_simulations.sh --target all --antennas 64 --engine cuda_v5" \
    "${LAST_JOB_ID}"

# Job 2: 128 Antennas Target Simulations (All targets)
submit_chained_job "Target Sims (128 Antennas, All Targets)" \
    "submit_target_simulations.sh --target all --antennas 128 --engine cuda_v5" \
    "${LAST_JOB_ID}"

# Job 3: 256 Antennas Target Simulations (All targets)
submit_chained_job "Target Sims (256 Antennas, All Targets)" \
    "submit_target_simulations.sh --target all --antennas 256 --engine cuda_v5" \
    "${LAST_JOB_ID}"

echo "--- [PHASE 2] Continuous Line-Rate Stress Tests (2-Hour Runs) ---"

# Job 4: 2-Hour Continuous Stress Test (64 Antennas, M=32 Upchannelization)
submit_chained_job "Stress Test (64 Antennas, M=32, 2 Hours)" \
    "submit_stress_test.sh --duration 7200 --antennas 64 --upchan 32 --engine cuda_v5" \
    "${LAST_JOB_ID}"

# Job 5: 2-Hour Continuous Stress Test (128 Antennas, M=32 Upchannelization)
submit_chained_job "Stress Test (128 Antennas, M=32, 2 Hours)" \
    "submit_stress_test.sh --duration 7200 --antennas 128 --upchan 32 --engine cuda_v5" \
    "${LAST_JOB_ID}"

# Job 6: 2-Hour Continuous Stress Test (256 Antennas, M=32 Upchannelization)
submit_chained_job "Stress Test (256 Antennas, M=32, 2 Hours)" \
    "submit_stress_test.sh --duration 7200 --antennas 256 --upchan 32 --engine cuda_v5" \
    "${LAST_JOB_ID}"

# Job 7: 1.5-Hour Continuous Stress Test (64 Antennas, M=16 Fast Upchannelization)
submit_chained_job "Stress Test (64 Antennas, M=16, 1.5 Hours)" \
    "submit_stress_test.sh --duration 5400 --antennas 64 --upchan 16 --engine cuda_v5" \
    "${LAST_JOB_ID}"

echo "========================================================================"
echo "  🎉 ALL 7 JOBS SUCCESSFULLY CHAINED FOR THE NEXT ~10 HOURS!            "
echo "========================================================================"
echo "Total Jobs Queued: ${#JOB_IDS[@]}"
echo "Job Chain IDs    : ${JOB_IDS[*]}"
echo ""
echo "To monitor the status anytime, run:"
echo "  squeue -u \$USER"
echo "  bash check_campaign_status.sh"
echo "========================================================================"
