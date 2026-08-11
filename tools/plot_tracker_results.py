#!/usr/bin/env python3
"""Tracker-beam visualization dashboard.

Reads a headerless [time, frequency, beam] float32 intensity file produced by
beam_tracker_cpu (n_beams == 1) and renders a tracker-specific dashboard:

  1. Sky map: tracker steering trajectory (l(t), m(t)) plus the per-integration
     window steering markers, and (optionally) a parametric moving point source.
  2. Frequency-integrated tracker-beam power versus time, with an optional
     comparison intensity file overlaid (e.g. constant-direction baseline).
  3. Time-integrated single-beam spectrum.
  4. Single-beam intensity heatmap [time, frequency] (n_beams == 1 drops the
     beam axis).

This tool deliberately reuses the geometry / frequency / response helpers from
plot_results.py so physical conventions (positions, channelized frequencies,
direction cosines) stay identical to the fixed-grid pipeline.

The tracker trajectory model mirrors src/beam_tracker.cpp exactly: (l, m) is
advanced linearly per time sample and re-projected to the unit disk, and the
direction used for an integration window is the one at the first sample of that
window.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

# Reuse shared physical helpers instead of re-implementing them.
from plot_results import (
    SPEED_OF_LIGHT_M_PER_S,
    DEFAULT_SPACING_M,
    DEFAULT_FREQUENCY_START_HZ,
    DEFAULT_CHANNEL_WIDTH_HZ,
    BEAM_GRID_DESIGN_FREQUENCY_HZ,
    LOCAL_FREQUENCY_CHANNELS,
    FULL_BAND_FREQUENCY_CHANNELS,
    DEFAULT_N_FREQ,
    array_shape,
    baseline_uv,
    beam_power_cube,
    default_frequencies,
    default_positions,
    load_intensity,
    normalized_db,
)


TRACKER_BEAM_COUNT = 1


def tracker_directions(
    l0: float, m0: float, dl_per_sample: float, dm_per_sample: float,
    n_time: int,
) -> np.ndarray:
    """Per-sample tracker directions (n, m, l) for the linear model.

    Matches tracker_direction() in src/beam_tracker.cpp: l(t) = l0 + t*dl,
    m(t) = m0 + t*dm, then n = sqrt(1 - l*l - m*m). Raises if any sample
    leaves the unit disk (mirrors direction_from_lm's rejection).
    """
    t = np.arange(n_time, dtype=np.float64)
    l = l0 + t * dl_per_sample
    m = m0 + t * dm_per_sample
    transverse = l * l + m * m
    if np.any(transverse > 1.0):
        worst = int(np.argmax(transverse))
        raise ValueError(
            f"trajectory leaves the unit disk at t={worst}: "
            f"l={l[worst]:.4f}, m={m[worst]:.4f}, l*l+m*m={transverse[worst]:.4f}"
        )
    n = np.sqrt(np.maximum(1.0 - transverse, 0.0))
    return np.stack([l, m, n], axis=1)


def window_steering_directions(
    l0: float, m0: float, dl_per_sample: float, dm_per_sample: float,
    n_time: int, integration_spectra: int,
) -> np.ndarray:
    """One steering direction per integration window (first sample of window)."""
    window_count = (n_time + integration_spectra - 1) // integration_spectra
    starts = np.arange(window_count, dtype=np.float64) * integration_spectra
    l = l0 + starts * dl_per_sample
    m = m0 + starts * dm_per_sample
    transverse = l * l + m * m
    if np.any(transverse > 1.0):
        worst = int(np.argmax(transverse))
        raise ValueError(
            f"window steering leaves the unit disk at window={worst}: "
            f"l={l[worst]:.4f}, m={m[worst]:.4f}"
        )
    n = np.sqrt(np.maximum(1.0 - transverse, 0.0))
    return np.stack([l, m, n], axis=1)


def moving_point_source_directions(
    source_l0: float, source_m0: float,
    source_dl_per_sample: float, source_dm_per_sample: float,
    n_time: int,
) -> np.ndarray:
    """Optional moving point-source track to overlay on the sky map."""
    return tracker_directions(
        source_l0, source_m0, source_dl_per_sample, source_dm_per_sample, n_time)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a single-beam tracker intensity file [T][F][B=1]."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="tracker intensity file [T][F][B=1] float32")
    parser.add_argument("--compare", type=Path,
                        help="optional second tracker intensity file to overlay")
    parser.add_argument("--output", type=Path, required=True,
                        help="dashboard PNG path")
    parser.add_argument("--n-time", type=int, required=True)
    parser.add_argument("--n-ant", type=int, default=32)
    parser.add_argument("--spacing-m", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--frequency-start-hz", type=float,
                        default=DEFAULT_FREQUENCY_START_HZ)
    parser.add_argument("--channel-width-hz", type=float,
                        default=DEFAULT_CHANNEL_WIDTH_HZ)
    parser.add_argument("--buffer", choices=("0", "1", "both"), default="0",
                        help="local buffer width to use")
    parser.add_argument("--n-freq", type=int,
                        help="override n_freq (336 for one buffer, 672 for both)")
    # Tracker trajectory params (must match the run that produced --input).
    parser.add_argument("--track-l0", type=float, default=0.0)
    parser.add_argument("--track-m0", type=float, default=0.0)
    parser.add_argument("--dl-per-sample", type=float, default=0.0)
    parser.add_argument("--dm-per-sample", type=float, default=0.0)
    parser.add_argument("--integration-spectra", type=int, default=320)
    # Optional moving point-source overlay.
    parser.add_argument("--source-l0", type=float, default=None)
    parser.add_argument("--source-m0", type=float, default=None)
    parser.add_argument("--source-dl-per-sample", type=float, default=0.0)
    parser.add_argument("--source-dm-per-sample", type=float, default=0.0)
    # Display.
    parser.add_argument("--sky-resolution", type=int, default=121)
    parser.add_argument("--design-frequency-hz", type=float,
                        default=BEAM_GRID_DESIGN_FREQUENCY_HZ)
    parser.add_argument("--label", default="tracker_cpu")
    parser.add_argument("--compare-label", default="tracker_baseline")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser


def resolve_frequency_channels(buffer: str, n_freq: int | None) -> int:
    expected = (FULL_BAND_FREQUENCY_CHANNELS if buffer == "both"
                else LOCAL_FREQUENCY_CHANNELS)
    if n_freq is not None and n_freq != expected:
        raise ValueError(f"buffer={buffer} requires n_freq={expected}, got {n_freq}")
    return expected


def _sky_axes(ax, title: str) -> None:
    ax.set(title=title, xlabel="l", ylabel="m", aspect="equal",
           adjustable="box", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    ax.grid(alpha=0.25)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="black",
            linewidth=1.0, alpha=0.6, label="horizon (l^2+m^2=1)")


def plot_sky(ax, args, frequencies, perch) -> None:
    """Sky map with the tracker trajectory and per-window steering markers."""
    directions = tracker_directions(
        args.track_l0, args.track_m0,
        args.dl_per_sample, args.dm_per_sample, args.n_time)
    # Background: array response at the design frequency for an on-axis steering
    # of the very first tracker direction. This gives a visual sense of the
    # beam footprint that the tracker carries along the trajectory.
    l_axis = np.linspace(-1.0, 1.0, args.sky_resolution)
    m_axis = np.linspace(-1.0, 1.0, args.sky_resolution)
    design_freq = args.design_frequency_hz
    power = beam_power_cube(l_axis, m_axis, directions[:1], args.n_ant,
                            args.spacing_m, design_freq)[0]
    band_max = np.nan_to_num(np.nanmax(power), nan=0.0)
    if band_max > 0.0:
        ax.imshow(np.where(np.isnan(power), np.nan, 10.0 * np.log10(power / band_max)),
                  origin="lower", extent=(-1, 1, -1, 1), cmap="viridis",
                  vmin=-40.0, vmax=0.0)
    # Trajectory path.
    ax.plot(directions[:, 0], directions[:, 1], color="cyan", linewidth=1.6,
            label="tracker trajectory")
    ax.scatter(directions[0, 0], directions[0, 1], s=60, marker="o",
               facecolor="lime", edgecolor="black", zorder=5, label="t=0 (start)")
    ax.scatter(directions[-1, 0], directions[-1, 1], s=60, marker="s",
               facecolor="yellow", edgecolor="black", zorder=5, label="t=N-1 (end)")
    # Per-window steering markers.
    windows = window_steering_directions(
        args.track_l0, args.track_m0, args.dl_per_sample, args.dm_per_sample,
        args.n_time, args.integration_spectra)
    ax.scatter(windows[:, 0], windows[:, 1], s=30, facecolor="white",
               edgecolor="black", linewidth=0.5, zorder=4,
               label=f"window steering ({len(windows)} windows)")
    # Optional moving source track.
    if args.source_l0 is not None and args.source_m0 is not None:
        source = moving_point_source_directions(
            args.source_l0, args.source_m0,
            args.source_dl_per_sample, args.source_dm_per_sample,
            args.n_time)
        ax.plot(source[:, 0], source[:, 1], color="red", linewidth=1.6,
                linestyle="--", label="moving source")
        ax.scatter(source[0, 0], source[0, 1], s=50, marker="^",
                   facecolor="red", edgecolor="black", zorder=5)
    _sky_axes(ax, f"Tracker sky map | {args.label}\n"
                  f"start=({args.track_l0:g},{args.track_m0:g}) "
                  f"rate=({args.dl_per_sample:g},{args.dm_per_sample:g})/sample; "
                  f"{len(windows)} windows of {args.integration_spectra} spectra")
    ax.legend(fontsize=7, loc="upper right")


def plot_power_vs_time(ax, intensity, frequencies, args, compare) -> None:
    """Frequency-integrated single-beam power vs time, optional overlay."""
    power = intensity.sum(axis=1, dtype=np.float64).ravel()
    time_axis = np.arange(args.n_time)
    ax.plot(time_axis, power, color="tab:blue", label=args.label)
    if compare is not None:
        compare_power = compare.sum(axis=1, dtype=np.float64).ravel()
        ax.plot(time_axis, compare_power, color="tab:orange", linestyle="--",
                alpha=0.85, label=args.compare_label)
    # Mark integration-window boundaries.
    window_count = (args.n_time + args.integration_spectra - 1) // args.integration_spectra
    for window in range(1, window_count):
        boundary = window * args.integration_spectra
        if boundary < args.n_time:
            ax.axvline(boundary, color="grey", linewidth=0.6, alpha=0.5)
    ax.set(title="Frequency-integrated tracker power vs time",
           xlabel="time sample", ylabel="integrated power")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_spectrum(ax, intensity, frequencies, args) -> None:
    """Time-integrated single-beam spectrum."""
    spectrum = intensity.sum(axis=0, dtype=np.float64).ravel()
    frequencies_vary = not np.allclose(frequencies, frequencies[0])
    if frequencies_vary:
        x_axis = frequencies / 1e6
        xlabel = "frequency [MHz]"
    else:
        x_axis = np.arange(args.n_freq)
        xlabel = "frequency-channel index"
    ax.plot(x_axis, spectrum, color="tab:green")
    ax.set(title="Time-integrated tracker-beam spectrum", xlabel=xlabel,
           ylabel="integrated power")
    ax.grid(alpha=0.25)


def plot_intensity_heatmap(ax, intensity, args) -> None:
    """Single-beam [time, frequency] intensity heatmap in dB."""
    power = intensity[:, :, 0]
    image = ax.imshow(normalized_db(power), origin="lower", aspect="auto",
                      cmap="magma",
                      extent=(-0.5, args.n_freq - 0.5, -0.5, args.n_time - 0.5))
    ax.set(title="Tracker-beam intensity [time, frequency]",
           xlabel="frequency channel", ylabel="time sample")
    fig = ax.figure
    fig.colorbar(image, ax=ax, label="power relative to maximum [dB]")


def plot_array_uv(ax, positions, frequencies, args) -> None:
    """Array geometry + u-v coverage at the design frequency (single panel)."""
    freq_hz = args.design_frequency_hz
    uv = baseline_uv(positions, float(freq_hz))
    first_block = min(32, args.n_ant)
    ax.scatter(positions[:first_block, 0], positions[:first_block, 1],
               color="royalblue", s=40, edgecolor="black", linewidth=0.4,
               label=f"E[0..{first_block - 1}]")
    if args.n_ant > first_block:
        ax.scatter(positions[first_block:, 0], positions[first_block:, 1],
                   color="crimson", s=40, edgecolor="black", linewidth=0.4,
                   label=f"E[{first_block}..{args.n_ant - 1}]")
    ax.set(title=f"Array geometry | {args.n_ant} elements, d={args.spacing_m:g} m",
           xlabel="x [m]", ylabel="y [m]", aspect="equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def validate_args(args: argparse.Namespace) -> None:
    if args.n_time <= 0:
        raise ValueError("n_time must be positive")
    if args.n_ant not in (32, 64):
        raise ValueError("n_ant must be 32 or 64")
    if args.integration_spectra <= 0:
        raise ValueError("integration_spectra must be positive")
    if args.sky_resolution <= 0:
        raise ValueError("sky_resolution must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    validate_args(args)

    args.n_freq = resolve_frequency_channels(args.buffer, args.n_freq)

    intensity = load_intensity(args.input, args.n_time, args.n_freq,
                               TRACKER_BEAM_COUNT)
    compare = None
    if args.compare is not None:
        compare = load_intensity(args.compare, args.n_time, args.n_freq,
                                 TRACKER_BEAM_COUNT)

    positions = default_positions(args.n_ant, args.spacing_m)
    freq_start = (DEFAULT_FREQUENCY_START_HZ if args.buffer in ("0", "both")
                  else DEFAULT_FREQUENCY_START_HZ
                  + LOCAL_FREQUENCY_CHANNELS * DEFAULT_CHANNEL_WIDTH_HZ)
    frequencies = default_frequencies(args.n_freq, freq_start,
                                      args.channel_width_hz)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(
        f"Tracker beam dashboard | {args.label} | layout "
        f"[T={args.n_time}][F={args.n_freq}][B=1]\n"
        f"trajectory start=({args.track_l0:g},{args.track_m0:g}) "
        f"rate=({args.dl_per_sample:g},{args.dm_per_sample:g})/sample | "
        f"{args.integration_spectra} spectra/window",
        fontsize=13,
    )

    plot_sky(axes[0, 0], args, frequencies, positions)
    plot_power_vs_time(axes[0, 1], intensity, frequencies, args, compare)
    plot_spectrum(axes[1, 0], intensity, frequencies, args)
    plot_intensity_heatmap(axes[1, 1], intensity, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Wrote tracker dashboard to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
