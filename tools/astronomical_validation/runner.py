"""Beam Tracker Execution Harness for C++/CUDA Implementations.

Interfaces Python test scripts to compiled beam tracker engines via run_tracker_stream.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np


def find_tracker_executable() -> Path:
    """Locate the run_tracker_stream executable in build directories or PATH."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "run_tracker_stream.exe",
        root / "build" / "run_tracker_stream",
        root / "build" / "Release" / "run_tracker_stream.exe",
        root / "build" / "Debug" / "run_tracker_stream.exe",
    ]

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate

    # Fallback to PATH search
    import shutil
    path_exe = shutil.which("run_tracker_stream")
    if path_exe:
        return Path(path_exe)

    raise FileNotFoundError(
        "Executable 'run_tracker_stream' not found. "
        "Please build the project first (cmake -B build && cmake --build build)."
    )


def run_beam_tracker(
    packed_bytes: bytes,
    n_time: int,
    n_ant: int,
    n_freq: int = 336,
    integration_spectra: int = 320,
    engine: str = "cpu_v2",
    source_l0: float = 0.0,
    source_m0: float = 0.0,
    source_dl: float = 0.0,
    source_dm: float = 0.0,
    executable: Path | None = None,
) -> np.ndarray:
    """Run beam tracker implementation on packed voltage binary stream.

    Returns float32 intensity waterfall array of shape (n_time, n_freq).
    """
    if executable is None:
        executable = find_tracker_executable()

    with tempfile.TemporaryDirectory(prefix="tracker_test_") as tmp_dir:
        input_path = Path(tmp_dir) / "packed_input.bin"
        output_path = Path(tmp_dir) / "intensity_output.bin"

        with open(input_path, "wb") as f:
            f.write(packed_bytes)

        cmd = [
            str(executable),
            "--engine", engine,
            "--n-time", str(n_time),
            "--n-freq", str(n_freq),
            "--n-ant", str(n_ant),
            "--spectra", str(integration_spectra),
            "--source-l0", str(source_l0),
            "--source-m0", str(source_m0),
            "--source-dl", str(source_dl),
            "--source-dm", str(source_dm),
            "--input", str(input_path),
            "--output", str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"run_tracker_stream failed (exit code {result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        # Read float32 binary output
        out_floats = n_time * n_freq
        expected_bytes = out_floats * 4

        if not output_path.exists() or output_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Output file missing or wrong size: expected {expected_bytes} bytes, "
                f"got {output_path.stat().st_size if output_path.exists() else 0}"
            )

        raw_floats = np.fromfile(output_path, dtype=np.float32)
        waterfall = raw_floats.reshape((n_time, n_freq))

        return waterfall
