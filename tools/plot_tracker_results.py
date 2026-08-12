#!/usr/bin/env python3
"""Tracker-beam visualization.

Two modes:

  * Default dashboard (one PNG): sky map, freq-integrated power vs time,
    single-beam spectrum, and a [time, frequency] dB heatmap.
  * Per-window frames mode (--frames-dir DIR): one PNG per integration window,
    each showing the steered beam footprint, the moving source location, and the
    instantaneous recorded power, so the tracker "following" the source is
    visible. Files are zero-padded inside DIR so you can scp -r the whole folder.

Reuses physical helpers from plot_results.py so conventions (positions,
channelized frequencies, direction cosines, element factor) stay identical to
the fixed-grid pipeline. The trajectory model mirrors src/beam_tracker.cpp:
l(t) = l0 + t*dl, m(t) = m0 + t*dm, reprojected to the unit disk; the direction
used for a window is the one at the first sample of that window.

Designed to run with plain ``python`` onCompute-Canada / trillium (no conda
required) as long as numpy + matplotlib are available.
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
    array_shape,
    baseline_uv,
    beam_power_cube,
    default_frequencies,
    default_positions,
    interpolated_antenna_specs,
    load_intensity,
    normalized_db,
)


TRACKER_BEAM_COUNT = 1


# --------------------------------------------------------------------------
# Trajectory model (mirrors src/beam_tracker.cpp)
# --------------------------------------------------------------------------

def tracker_directions(
    l0: float, m0: float, dl_per_sample: float, dm_per_sample: float,
    n_time: int,
) -> np.ndarray:
    """Per-sample tracker directions as columns (l, m, n) for the linear model.

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


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------

def array_metadata_text(args, frequencies, positions) -> str:
    """Compact array/run metadata block, shown as a legend on every panel."""
    rows, columns = array_shape(args.n_ant)
    aperture_x = columns * args.spacing_m
    aperture_y = rows * args.spacing_m
    window_count = (args.n_time + args.integration_spectra - 1) // args.integration_spectra
    centre_freq = float(frequencies[len(frequencies) // 2])
    return (
        f"array: {rows}x{columns}  n_ant={args.n_ant}  spacing={args.spacing_m:g} m  "
        f"aperture={aperture_x:g}x{aperture_y:g} m\n"
        f"channels: n_freq={args.n_freq}  "
        f"centre~{centre_freq / 1e6:.2f} MHz  "
        f"sync 300/400/500 MHz element lookup\n"
        f"run: n_time={args.n_time}  integration_spectra={args.integration_spectra}  "
        f"windows={window_count}\n"
        f"tracker: start=({args.track_l0:g},{args.track_m0:g}) "
        f"rate=({args.dl_per_sample:g},{args.dm_per_sample:g})/sample"
    )


def add_metadata_legend(ax, text: str, loc: str = "upper left") -> None:
    """Attach a small text box with array/run metadata to an axes."""
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="grey", alpha=0.85),
        zorder=10,
    )


# --------------------------------------------------------------------------
# Sky rendering (desktop dashboard + per-window frames share this)
# --------------------------------------------------------------------------

def _sky_axes(ax, title: str) -> None:
    ax.set(title=title, xlabel="l (H / x direction)", ylabel="m (E / y direction)",
           aspect="equal", adjustable="box", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    ax.grid(alpha=0.25)
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="black",
            linewidth=1.0, alpha=0.6, label="horizon (l^2+m^2=1)")


def _steered_beam_image(ax, n_ant, spacing_m, frequency_hz, direction,
                        n_pixels=121):
    """Background image of a single steered beam's array response in dB."""
    l_axis = np.linspace(-1.0, 1.0, n_pixels)
    m_axis = np.linspace(-1.0, 1.0, n_pixels)
    power = beam_power_cube(l_axis, m_axis, direction[np.newaxis],
                            n_ant, spacing_m, frequency_hz)[0]
    peak = np.nan_to_num(np.nanmax(power), nan=0.0)
    if peak > 0.0:
        image = ax.imshow(
            np.where(np.isnan(power), np.nan, 10.0 * np.log10(power / peak)),
            origin="lower", extent=(-1, 1, -1, 1), cmap="viridis",
            vmin=-40.0, vmax=0.0,
        )
        return image, (l_axis, m_axis, power)
    return None, (l_axis, m_axis, power)


def _antenna_fov_ellipse(ax, frequency_hz) -> None:
    """White dashed antenna-element 3 dB FoV ellipse, like plot_results.py."""
    from matplotlib.patches import Ellipse
    bw_e, bw_h, _ = interpolated_antenna_specs(frequency_hz)
    fov_u = min(1.0, math.sin(math.radians(float(bw_h) / 2.0)))
    fov_v = min(1.0, math.sin(math.radians(float(bw_e) / 2.0)))
    ax.add_patch(Ellipse(
        (0.0, 0.0), width=2.0 * fov_u, height=2.0 * fov_v,
        fill=False, edgecolor="white", linestyle="--", linewidth=1.0,
        label="antenna 3 dB FoV", zorder=5))


def draw_sky_panel(ax, args, frequencies, positions, metadata_text,
                   *, frame_label=None, window_dirs=None, window_index=None,
                   source_dirs=None):
    """Draw a sky map. Used by both the dashboard and per-window frames.

    - frame_label: if given, override the title (used for frame PNGs).
    - window_dirs: the per-window steering directions; if both window_dirs and
      window_index are given, the current window's steering direction is drawn
      as a contour footprint and a marker, so the per-frame PNG shows the beam
      being recorded.
    - source_dirs: optional moving source track to overlay.
    """
    design_freq = args.design_frequency_hz
    base_dir = tracker_directions(
        args.track_l0, args.track_m0, args.dl_per_sample, args.dm_per_sample,
        args.n_time)[0]
    image, _ = _steered_beam_image(ax, args.n_ant, args.spacing_m,
                                   design_freq, base_dir)
    _antenna_fov_ellipse(ax, design_freq)

    # Full tracker trajectory path (light).
    dirs = tracker_directions(
        args.track_l0, args.track_m0, args.dl_per_sample, args.dm_per_sample,
        args.n_time)
    ax.plot(dirs[:, 0], dirs[:, 1], color="cyan", linewidth=1.2,
            alpha=0.55, label="tracker trajectory")
    ax.scatter(dirs[0, 0], dirs[0, 1], s=60, marker="o",
               facecolor="lime", edgecolor="black", zorder=6, label="t=0 (start)")
    ax.scatter(dirs[-1, 0], dirs[-1, 1], s=60, marker="s",
               facecolor="yellow", edgecolor="black", zorder=6, label="t=N-1 (end)")

    # Per-window steering markers (white dots).
    if window_dirs is None:
        window_dirs = window_steering_directions(
            args.track_l0, args.track_m0, args.dl_per_sample, args.dm_per_sample,
            args.n_time, args.integration_spectra)
    ax.scatter(window_dirs[:, 0], window_dirs[:, 1], s=28,
               facecolor="white", edgecolor="black", linewidth=0.4, zorder=5,
               label=f"window steering ({len(window_dirs)})")

    # Current-window footprint + marker (frame mode).
    if window_dirs is not None and window_index is not None:
        current_dir = window_dirs[window_index]
        l_axis = np.linspace(-1.0, 1.0, 121)
        m_axis = np.linspace(-1.0, 1.0, 121)
        power = beam_power_cube(l_axis, m_axis, current_dir[np.newaxis],
                                args.n_ant, args.spacing_m, design_freq)[0]
        peak = np.nan_to_num(np.nanmax(power), nan=0.0)
        if peak > 0.0:
            ax.contour(l_axis, m_axis, power / peak,
                       levels=[10.0 ** (-3.0 / 10.0)], colors=["magenta"],
                       linewidths=1.4)
        ax.scatter(current_dir[0], current_dir[1], s=80, marker="*",
                   facecolor="magenta", edgecolor="black", linewidth=0.5,
                   zorder=8, label=f"recording window {window_index}")

    # Moving source track.
    if source_dirs is not None:
        ax.plot(source_dirs[:, 0], source_dirs[:, 1], color="red",
                linewidth=1.4, linestyle="--", label="moving source", zorder=7)
        if window_index is not None:
            # Source position at the start time of the current window.
            t0 = window_index * args.integration_spectra
            if 0 <= t0 < len(source_dirs):
                ax.scatter(source_dirs[t0, 0], source_dirs[t0, 1], s=110,
                           marker="^", facecolor="red", edgecolor="black",
                           linewidth=0.5, zorder=9,
                           label="source(this window)")
        else:
            ax.scatter(source_dirs[0, 0], source_dirs[0, 1], s=70, marker="^",
                       facecolor="red", edgecolor="black", zorder=8)

    _sky_axes(ax, frame_label or "Tracker sky map")
    if image is not None:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03,
                           label="beam response [dB]")
    add_metadata_legend(ax, metadata_text)
    ax.legend(fontsize=7, loc="upper right")


def draw_array_inset(ax, positions, frequencies, args) -> None:
    """Array geometry inset pin (this echoes the fixed-grid ``plot_results``
    antenna-block colour convention)."""
    first_block = min(32, args.n_ant)
    ax.scatter(positions[:first_block, 0], positions[:first_block, 1],
               s=45, color="royalblue", edgecolor="black", linewidth=0.4,
               label=f"E[0..{first_block - 1}]")
    if args.n_ant > first_block:
        ax.scatter(positions[first_block:, 0], positions[first_block:, 1],
                   s=45, color="crimson", edgecolor="black", linewidth=0.4,
                   label=f"E[{first_block}..{args.n_ant - 1}]")
    rows, columns = array_shape(args.n_ant)
    ax.set(title=f"Array pins ({rows}x{columns}, {args.n_ant} elements)",
           xlabel="x [m]", ylabel="y [m]", aspect="equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


# --------------------------------------------------------------------------
# Dashboard panels (unchanged behaviour, now with metadata legend)
# --------------------------------------------------------------------------

def plot_sky(ax, args, frequencies, positions, metadata_text,
             source_dirs=None) -> None:
    draw_sky_panel(
        ax, args, frequencies, positions, metadata_text,
        frame_label=f"Tracker sky map | {args.label}",
        source_dirs=source_dirs,
    )


def plot_power_vs_time(ax, intensity, frequencies, args, compare, metadata_text) -> None:
    power = intensity.sum(axis=1, dtype=np.float64).ravel()
    time_axis = np.arange(args.n_time)
    ax.plot(time_axis, power, color="tab:blue", label=args.label)
    if compare is not None:
        compare_power = compare.sum(axis=1, dtype=np.float64).ravel()
        ax.plot(time_axis, compare_power, color="tab:orange", linestyle="--",
                alpha=0.85, label=args.compare_label)
    window_count = (args.n_time + args.integration_spectra - 1) // args.integration_spectra
    for window in range(1, window_count):
        boundary = window * args.integration_spectra
        if boundary < args.n_time:
            ax.axvline(boundary, color="grey", linewidth=0.6, alpha=0.5)
    ax.set(title="Frequency-integrated tracker power vs time",
           xlabel="time sample", ylabel="integrated power")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    add_metadata_legend(ax, metadata_text)


def plot_spectrum(ax, intensity, frequencies, args, metadata_text) -> None:
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
    add_metadata_legend(ax, metadata_text)


def plot_intensity_heatmap(ax, intensity, args, metadata_text) -> None:
    power = intensity[:, :, 0]
    image = ax.imshow(normalized_db(power), origin="lower", aspect="auto",
                      cmap="magma",
                      extent=(-0.5, args.n_freq - 0.5, -0.5, args.n_time - 0.5))
    ax.set(title="Tracker-beam intensity [time, frequency]",
           xlabel="frequency channel", ylabel="time sample")
    ax.figure.colorbar(image, ax=ax, label="power relative to maximum [dB]")
    add_metadata_legend(ax, metadata_text)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a single-beam tracker intensity file [T][F][B=1]."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="tracker intensity file [T][F][B=1] float32")
    parser.add_argument("--compare", type=Path,
                        help="optional second tracker intensity file to overlay")
    parser.add_argument("--output", type=Path,
                        help="dashboard PNG path (default mode)")
    parser.add_argument("--frames-dir", type=Path,
                        help="per-integration-window frames folder (scp -r friendly)")
    parser.add_argument("--frames-stride", type=int, default=1,
                        help="emit every Nth window as a frame; default: 1 (all)")
    parser.add_argument("--frames-max", type=int, default=256,
                        help="safety cap on the number of frame PNGs emitted")
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
    parser.add_argument("--track-l0", type=float, default=0.0)
    parser.add_argument("--track-m0", type=float, default=0.0)
    parser.add_argument("--dl-per-sample", type=float, default=0.0)
    parser.add_argument("--dm-per-sample", type=float, default=0.0)
    parser.add_argument("--integration-spectra", type=int, default=320)
    parser.add_argument("--source-l0", type=float, default=None)
    parser.add_argument("--source-m0", type=float, default=None)
    parser.add_argument("--source-dl-per-sample", type=float, default=0.0)
    parser.add_argument("--source-dm-per-sample", type=float, default=0.0)
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


def validate_args(args: argparse.Namespace) -> None:
    if args.n_time <= 0:
        raise ValueError("n_time must be positive")
    if args.n_ant not in (32, 64):
        raise ValueError("n_ant must be 32 or 64")
    if args.integration_spectra <= 0:
        raise ValueError("integration_spectra must be positive")
    if args.sky_resolution <= 0:
        raise ValueError("sky_resolution must be positive")
    if not args.output and not args.frames_dir:
        raise ValueError("either --output or --frames-dir is required")
    if args.frames_stride <= 0:
        raise ValueError("frames-stride must be positive")
    if args.frames_max <= 0:
        raise ValueError("frames-max must be positive")


def build_source_track(args):
    if args.source_l0 is None or args.source_m0 is None:
        return None
    return moving_point_source_directions(
        args.source_l0, args.source_m0,
        args.source_dl_per_sample, args.source_dm_per_sample,
        args.n_time)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def render_dashboard(args, intensity, compare, frequencies, positions,
                     source_dirs, metadata_text) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"Tracker beam dashboard | {args.label} | layout "
        f"[T={args.n_time}][F={args.n_freq}][B=1]\n"
        f"trajectory start=({args.track_l0:g},{args.track_m0:g}) "
        f"rate=({args.dl_per_sample:g},{args.dm_per_sample:g})/sample | "
        f"{args.integration_spectra} spectra/window",
        fontsize=13,
    )
    plot_sky(axes[0, 0], args, frequencies, positions, metadata_text, source_dirs)
    draw_array_inset(axes[0, 1], positions, frequencies, args)
    plot_power_vs_time(axes[1, 0], intensity, frequencies, args, compare, metadata_text)
    plot_intensity_heatmap(axes[1, 1], intensity, args, metadata_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    return args.output


def render_frames(args, intensity, frequencies, positions,
                  source_dirs, metadata_text) -> list[Path]:
    """Emit one PNG per integration window (downsampled by frames-stride).

    Each frame shows the array inset, the antenna FoV ellipse, the tracker
    trajectory, the moving-source track, the *current* window's steered-beam
    footprint (magenta contour) and marker, plus the source position at that
    window's start, so you can watch the tracker "follow" the source and the
    beam being recorded. Output files are zero-padded so scp -r preserves
    order.
    """
    import matplotlib.pyplot as plt

    window_dirs = window_steering_directions(
        args.track_l0, args.track_m0, args.dl_per_sample, args.dm_per_sample,
        args.n_time, args.integration_spectra)
    total_windows = len(window_dirs)
    selected = list(range(0, total_windows, max(1, args.frames_stride)))
    if len(selected) > args.frames_max:
        raise ValueError(
            f"{len(selected)} frames exceeds --frames-max={args.frames_max}; "
            f"increase --frames-max or use a larger --frames-stride"
        )

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(selected[-1])) if selected else 1)
    written: list[Path] = []
    per_window_power = (
        intensity.reshape(total_windows, args.integration_spectra,
                          args.n_freq, TRACKER_BEAM_COUNT)
        .sum(axis=(1, 3)) if total_windows * args.integration_spectra == args.n_time
        else None
    )
    for idx, window_index in enumerate(selected):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(
            f"Tracker frame {idx + 1}/{len(selected)}  "
            f"(window {window_index}/{total_windows - 1}) | {args.label}\n"
            f"[T={args.n_time}][F={args.n_freq}][B=1]  "
            f"integration_spectra={args.integration_spectra}",
            fontsize=12,
        )
        draw_sky_panel(
            axes[0], args, frequencies, positions, metadata_text,
            frame_label=(
                f"Window {window_index} steering + source\n"
                f"recording beam footprint (magenta -3 dB)"),
            window_dirs=window_dirs, window_index=window_index,
            source_dirs=source_dirs,
        )
        # Right panel: accumulated power-vs-window bar for context.
        ax = axes[1]
        if per_window_power is not None:
            x = np.arange(total_windows)
            ax.bar(x, per_window_power, color="lightgrey", width=0.9)
            ax.bar([window_index], [per_window_power[window_index]],
                   color="magenta", width=0.9, label="this window")
            ax.set(title="Per-window recorded power",
                   xlabel="window index", ylabel="integrated power")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5,
                    "n_time not a multiple of integration_spectra:\n"
                    "per-window power histogram skipped.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        add_metadata_legend(ax, metadata_text)
        name = args.frames_dir / f"frame_{str(idx).zfill(width)}.png"
        fig.savefig(name, dpi=args.dpi)
        plt.close(fig)
        written.append(name)
    return written


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
    metadata_text = array_metadata_text(args, frequencies, positions)
    source_dirs = build_source_track(args)

    if args.output is not None:
        render_dashboard(args, intensity, compare, frequencies, positions,
                         source_dirs, metadata_text)
        print(f"Wrote tracker dashboard to {args.output}")

    if args.frames_dir is not None:
        written = render_frames(args, intensity, frequencies, positions,
                                source_dirs, metadata_text)
        print(f"Wrote {len(written)} tracker frames to {args.frames_dir}")
        print("  first:", written[0])
        if len(written) > 1:
            print("  last :", written[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
