#!/usr/bin/env bash
# ==============================================================================
# run_presentation.sh
#
# Standalone Presentation Suite & Validation Runner for the GPU Beam Tracker.
# (Delegates to scripts/run_presentation.sh)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/scripts/run_presentation.sh" "$@"
