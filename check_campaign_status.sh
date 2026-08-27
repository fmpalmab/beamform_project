#!/bin/bash
# ==============================================================================
# check_campaign_status.sh
#
# Check the execution status, metrics, and outputs of the overnight campaign.
#
# Usage:
#   bash check_campaign_status.sh
# ==============================================================================

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "          OVERNIGHT 10-HOUR CAMPAIGN STATUS & REPORT CHECKER            "
echo "========================================================================"
echo "Checked Time : $(date -u)"
echo "Host Node    : $(hostname)"
echo "========================================================================"
echo ""

# Find active campaign log (matching running/pending jobs in squeue)
ACTIVE_JOBS=$(squeue -u "${USER:-$USER}" -h -o "%i" 2>/dev/null || true)
LATEST_CAMP_LOG=""
for log in $(ls -t "${PROJECT_ROOT}/results"/campaign_*.log 2>/dev/null); do
    for j in ${ACTIVE_JOBS}; do
        if grep -q "^${j}|" "${log}" 2>/dev/null; then
            LATEST_CAMP_LOG="${log}"
            break 2
        fi
    done
done

if [[ -z "${LATEST_CAMP_LOG}" ]]; then
    LATEST_CAMP_LOG=$(ls -t "${PROJECT_ROOT}/results"/campaign_*.log 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${LATEST_CAMP_LOG}" || ! -f "${LATEST_CAMP_LOG}" ]]; then
    echo "No active campaign log found. Showing general Slurm queue status:"
    squeue -u "${USER:-$USER}"
    exit 0
fi

echo "Active Campaign Log: ${LATEST_CAMP_LOG}"
echo ""

# Check Slurm Queue for active jobs
echo "--- Current Slurm Queue Status ---"
squeue -u "${USER:-$USER}"
echo ""

# Summary Table
echo "--- Campaign Job Execution History (sacct) ---"
JOB_LIST=$(awk -F'|' '{print $1}' "${LATEST_CAMP_LOG}" | tr '\n' ',' | sed 's/,$//')

if [[ -n "${JOB_LIST}" ]]; then
    sacct -j "${JOB_LIST}" --format=JobID%14,JobName%20,State%12,ExitCode%10,Elapsed%12,Start%20,End%20
fi
echo ""

# Check Results Directories
echo "--- Generated Output Directories & Artifacts ---"
echo "Simulation Runs:"
ls -dt "${PROJECT_ROOT}/results"/simulations_* 2>/dev/null | head -n 5 || echo "  (None found yet)"

echo ""
echo "Stress Test Runs:"
ls -dt "${PROJECT_ROOT}/results"/stress_test_* 2>/dev/null | head -n 5 || echo "  (None found yet)"

echo ""
echo "Tarball Archives Ready for Local Download:"
ls -lh "${PROJECT_ROOT}/results"/*.tar.gz 2>/dev/null | head -n 5 || echo "  (None found yet)"

echo ""
echo "========================================================================"
