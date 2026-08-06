#!/usr/bin/env python3
"""Run the 15360-spectrum CPU temporal-integration validation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


N_TIME = 15360
POST_UPCHAN_TIME = 480
N_FREQ = 336
N_ANT = 64
N_BEAMS = 64
UPCHAN_FACTOR = 32  # fixture construction only; this is not the upchan implementation
INTEGRATION_AFTER_UPCHAN = 10
INTEGRATION_DIRECT = 320


def run(command: list[str], repo: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def make_post_upchan_fixture(raw: np.ndarray) -> np.ndarray:
    """Create a shape-compatible fixture, not a numerical upchannelizer."""
    if raw.ndim != 3 or raw.shape[0] % UPCHAN_FACTOR != 0:
        raise ValueError("raw intensity must be [T][F][B] and divisible by 32")
    return np.add.reduce(
        raw.reshape(raw.shape[0] // UPCHAN_FACTOR,
                   UPCHAN_FACTOR, raw.shape[1], raw.shape[2]),
        axis=1,
        dtype=np.float32,
    )


def integrate_array(intensity: np.ndarray, integration_spectra: int) -> np.ndarray:
    """Sum consecutive [T][F][B] windows while preserving float32 accumulation."""
    if intensity.ndim != 3 or intensity.shape[0] % integration_spectra != 0:
        raise ValueError("intensity shape must be divisible by integration_spectra")
    return np.add.reduce(
        intensity.reshape(intensity.shape[0] // integration_spectra,
                          integration_spectra, intensity.shape[1], intensity.shape[2]),
        axis=1,
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, beamform, integrate, and compare the two temporal modes."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("results/weights_hex64.bin"),
        help="64-beam weight file, relative to the repository by default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/temporal_15360"),
    )
    parser.add_argument(
        "--directions",
        type=Path,
        default=Path("results/hex64_directions.txt"),
        help="directions used to generate the supplied weights",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-l", type=float, default=0.04)
    parser.add_argument("--source-m", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=4.0)
    parser.add_argument(
        "--type",
        choices=("noise", "point-source"),
        default="noise",
        help="synthetic input type; noise exercises temporal accumulation differences",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also create the comparison dashboard with tools/plot_results.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    weights = args.weights if args.weights.is_absolute() else repo / args.weights
    directions = (args.directions if args.directions.is_absolute()
                  else repo / args.directions)
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir

    if not weights.is_file():
        raise FileNotFoundError(f"weight file not found: {weights}")
    if not directions.is_file():
        raise FileNotFoundError(f"direction file not found: {directions}")
    direction_values = np.loadtxt(directions)
    if direction_values.shape != (N_BEAMS, 3):
        raise ValueError(
            f"direction file has shape {direction_values.shape}; "
            f"expected {(N_BEAMS, 3)} for the supplied weights"
        )
    expected_weights = N_BEAMS * N_FREQ * N_ANT * 8
    if weights.stat().st_size != expected_weights:
        raise ValueError(
            f"weight file has {weights.stat().st_size} bytes; "
            f"expected {expected_weights} for [B=64][F=336][E=64]"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    voltage = output_dir / "voltage.bin"
    intensity = output_dir / "intensity.bin"
    metrics = output_dir / "cpu_metrics.csv"
    generated = (
        voltage,
        intensity,
        metrics,
        output_dir / "post_upchan_intensity_480.bin",
        output_dir / "integrated_10_after_upchan.bin",
        output_dir / "integrated_320_direct.bin",
        output_dir / "temporal_diff.json",
        output_dir / "geometry.png",
    )
    if not args.force:
        existing = [path for path in generated if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"output exists: {names}; use --force or another --output-dir"
            )

    generate = repo / "build/generate_fake_data"
    beamformer = repo / "build/beamformer_cpu"
    if not generate.is_file() or not beamformer.is_file():
        raise FileNotFoundError(
            "missing build/generate_fake_data or build/beamformer_cpu; "
            "run cmake --build build --target generate_fake_data beamformer_cpu -j2"
        )

    run(
        [
            str(generate),
            "--type",
            args.type,
            "--n-time",
            str(N_TIME),
            "--n-ant",
            str(N_ANT),
            "--seed",
            str(args.seed),
            "--source-l",
            str(args.source_l),
            "--source-m",
            str(args.source_m),
            "--amplitude",
            str(args.amplitude),
            "--output",
            str(voltage),
        ],
        repo,
    )
    run(
        [
            str(beamformer),
            "--input",
            str(voltage),
            "--weights",
            str(weights),
            "--n-time",
            str(N_TIME),
            "--n-ant",
            str(N_ANT),
            "--n-beams",
            str(N_BEAMS),
            "--output",
            str(intensity),
            "--metrics",
            str(metrics),
        ],
        repo,
    )

    raw = np.memmap(intensity, dtype="<f4", mode="r", shape=(N_TIME, N_FREQ, N_BEAMS))
    effective_count = N_TIME // INTEGRATION_DIRECT
    if N_TIME % UPCHAN_FACTOR != 0:
        raise ValueError("N_TIME must be divisible by the fake upchan factor")
    if POST_UPCHAN_TIME != N_TIME // UPCHAN_FACTOR:
        raise ValueError("post-upchan fixture dimensions are inconsistent")

    # This creates only a contract fixture for the separate upchan stage. It is
    # deliberately not presented as a numerical upchannelizer implementation.
    post_upchan_data = make_post_upchan_fixture(raw)
    post_upchan = np.memmap(
        output_dir / "post_upchan_intensity_480.bin",
        dtype="<f4",
        mode="w+",
        shape=post_upchan_data.shape,
    )
    post_upchan[:] = post_upchan_data
    post_upchan.flush()

    integrated_10_data = integrate_array(post_upchan_data, INTEGRATION_AFTER_UPCHAN)
    direct_320_data = integrate_array(raw, INTEGRATION_DIRECT)
    integrated_10 = np.memmap(
        output_dir / "integrated_10_after_upchan.bin",
        dtype="<f4",
        mode="w+",
        shape=integrated_10_data.shape,
    )
    direct_320 = np.memmap(
        output_dir / "integrated_320_direct.bin",
        dtype="<f4",
        mode="w+",
        shape=direct_320_data.shape,
    )
    integrated_10[:] = integrated_10_data
    direct_320[:] = direct_320_data
    integrated_10.flush()
    direct_320.flush()

    difference = np.asarray(integrated_10, dtype=np.float64) - np.asarray(
        direct_320, dtype=np.float64
    )
    absolute = np.abs(difference)
    relative = absolute / np.maximum(np.abs(np.asarray(direct_320)), 1.0e-12)
    max_abs = float(absolute.max())
    max_rel = float(relative.max())
    sum_squared_error = float(np.sum(difference * difference))
    compared_values = difference.size

    beam_power_10 = np.asarray(integrated_10, dtype=np.float64).sum(axis=(0, 1))
    beam_power_320 = np.asarray(direct_320, dtype=np.float64).sum(axis=(0, 1))
    summary = {
        "raw_n_time": N_TIME,
        "post_upchan_n_time": POST_UPCHAN_TIME,
        "n_freq": N_FREQ,
        "n_ant": N_ANT,
        "n_beams": N_BEAMS,
        "spectrum_period_us": 10.0 / 3.0,
        "post_upchan_period_us": 320.0 / 3.0,
        "integration_after_upchan": INTEGRATION_AFTER_UPCHAN,
        "integration_direct": INTEGRATION_DIRECT,
        "effective_windows": effective_count,
        "effective_period_us": 3200.0 / 3.0,
        "post_upchan_fixture": "sum of 32 raw intensity samples; not a real upchannelizer",
        "max_absolute_difference": max_abs,
        "rms_difference": float(np.sqrt(sum_squared_error / compared_values)),
        "max_relative_difference": max_rel,
        "max_beam_total_difference": float(np.max(np.abs(beam_power_10 - beam_power_320))),
        "peak_beam_10_after_upchan": int(np.argmax(beam_power_10)),
        "peak_beam_320_direct": int(np.argmax(beam_power_320)),
    }
    (output_dir / "temporal_diff.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if args.plot:
        run(
            [
                "python",
                "tools/plot_results.py",
                "--input",
                str(output_dir / "integrated_10_after_upchan.bin"),
                "--compare",
                str(output_dir / "integrated_320_direct.bin"),
                "--label",
                "10 post-upchan spectra",
                "--compare-label",
                "320 direct spectra",
                "--n-time",
                str(effective_count),
                "--n-freq",
                str(N_FREQ),
                "--n-ant",
                str(N_ANT),
                "--n-beams",
                str(N_BEAMS),
                "--beam-grid",
                "fft",
                "--output",
                str(output_dir / "temporal_dashboard.png"),
                "--geometry-output",
                str(output_dir / "geometry.png"),
                "--directions",
                str(directions),
                "--synthetic-type",
                "point-source" if args.type == "point-source" else "noise",
                "--source-l",
                str(args.source_l),
                "--source-m",
                str(args.source_m),
                "--amplitude",
                str(args.amplitude),
                "--seed",
                str(args.seed),
                "--comparison-output",
                str(output_dir / "temporal_diff.png"),
                "--summary-json",
                str(output_dir / "plot_diff.json"),
            ],
            repo,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
