#!/usr/bin/env python3
"""Convenience proxy for plot_beam_patterns_and_outputs.py from beamform_project."""

import os
import sys
from pathlib import Path

_charts_root = Path(__file__).resolve().parent.parent.parent
_kotekan_script = _charts_root / "kotekan" / "test_charts" / "plot_beam_patterns_and_outputs.py"

if _kotekan_script.exists():
    os.execv(sys.executable, [sys.executable, str(_kotekan_script)] + sys.argv[1:])
else:
    print(f"Error: Could not locate {_kotekan_script}")
    sys.exit(1)
