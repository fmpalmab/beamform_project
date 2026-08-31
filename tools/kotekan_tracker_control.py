#!/usr/bin/env python3
"""Convenience entry point for Kotekan Tracker Control from beamform_project."""

import os
import sys
from pathlib import Path

_charts_root = Path(__file__).resolve().parent.parent.parent
_kotekan_control = _charts_root / "kotekan" / "test_charts" / "kotekan_tracker_control.py"

if _kotekan_control.exists():
    os.execv(sys.executable, [sys.executable, str(_kotekan_control)] + sys.argv[1:])
else:
    print(f"Error: Could not locate {_kotekan_control}")
    sys.exit(1)
