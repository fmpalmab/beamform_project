#!/usr/bin/env python3
"""Visualization for the optimized CPU beam-tracker benchmark + DOA tracking.

Three input modes (one CLI produces a single multi-panel dashboard PNG):

  * --frames FILE   : per-integration-window latency CSV emitted by
    `benchmark_cpu_opt_beam_tracker --frames ...` (columns: window, ms, ...).
    Plotted against the 0.5 ms/frame target (the kernel performance objective).

  * --metrics FILE  : summary CSV emitted by the benchmark's `--metrics ...`
    (one row per config run). Plotted as a summary bar chart of min/mean/p95/max
    per-frame latency, coloured PASS/FAIL against the target.

  * --intensity FILE + trajectory flags : mirrors `tools/plot_tracker_results.py`
    but overlays the *estimated* DOA track read from an optional `--doa FILE`
    (one (l, m) row per window, whitespace or comma separated) on top of the
    true source track, so the closing of the open-loop gap is visible.

Design goal: one `python tools/plot_cpu_opt_beam_tracker.py ...` call per
benchmark sweep produces a self-contained PNG suitable for a lab book on
trillium (plain ``python``, numpy + matplotlib only; no conda).

The benchmark frame/metrics CSVs are deliberately simple text so they can be
`scp`'d from trillium and re-plotted on a laptop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

# Reuse the shared physical / IO helpers from plot_results.py and the trajectory
# model from plot_tracker_results.py so the conventions (positions, channelized
# frequencies, direction cosines, the linear-track model) stay identical to the
# fixed-grid and naive-tracker dashboards.
from plot_results import (  # noqa: E402
    DEFAULT_FREQUENCY_START_HZ,
    DEFAULT_CHANNEL_WIDTH_HZ,
    DEFAULT_SPACING_M,
    LOCAL_FREQUENCY_CHANNELS,
    array_shape,
    default_frequencies,
    default_positions,
    load_intensity,
    normalized_db,
)
from plot_tracker_results import (  # noqa: E402
    tracker_directions,
    window_steering_directions,
    moving_point_source_directions,
    add_metadata_legend,
)


TARGET_MS_PER_FRAME = 0.5
TRACKER_BEAM_COUNT = 1


# --------------------------------------------------------------------------
# CSV loaders (deliberately tolerant of the benchmark's simple text format)
# --------------------------------------------------------------------------

def load_frames_csv(path: Path) -> dict:
    """Read the per-frame latency CSV emitted by the benchmark.

    Returns a dict with keys: window (np.ndarray), ms (np.ndarray),
    and the run metadata columns if present.
    """
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"frames CSV {path} is empty")
    window = np.array([int(r["window"]) for r in rows], dtype=np.int64)
    ms = np.array([float(r["ms"]) for r in rows], dtype=np.float64)
    meta = {k: rows[0][k] for k in rows[0] if k not in {"window", "ms"}}
    return {"window": window, "ms": ms, "meta": meta}


def load_metrics_csv(path: Path) -> list[dict]:
    """Read the summary metrics CSV (possibly multiple rows)."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_doa_track(path: Path) -> np.ndarray:
    """Read (l, m) per-window estimated directions. Whitespace or comma."""
    data = np.loadtxt(path, delimiter=",") if path.suffix == ".csv" \
        else np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"DOA track {path} needs at least (l, m) columns")
    return data[:, :2]


# --------------------------------------------------------------------------
# Panel drawers
# --------------------------------------------------------------------------

def plot_frame_latencies(ax, frames: dict, target: float) -> None:
    window = frames["window"]
    ms = frames["ms"]
    meta = frames["meta"]
    ax.plot(window, ms, color="tab:blue", marker=".", linestyle="-",
            linewidth=1.0, markersize=3, label="per-frame kernel time")
    ax.axhline(target, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"target {target:g} ms")
    if ms.size:
        ax.axhline(np.mean(ms), color="tab:green", linestyle=":", linewidth=1.0,
                   label=f"mean {np.mean(ms):.3f} ms")
        ax.axhline(np.percentile(ms, 95), color="tab:orange", linestyle=":",
                   linewidth=1.0, label=f"p95 {np.percentile(ms, 95):.3f} ms")
    ax.set(title="Per-frame (per-window) kernel latency",
           xlabel="integration window index",
           ylabel="latency [ms]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    label = (f"{meta.get('estimator', '?')}  coarse={meta.get('coarse_grid_resolution', '?')} "
             f"refine={meta.get('refinement_levels', '?')}  "
             f"n_ant={meta.get('n_ant', '?')}  win={window.size}")
    ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=7, va="top",
            ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor="grey", alpha=0.85))


def plot_metrics_bars(ax, metrics: list[dict], target: float) -> None:
    if not metrics:
        ax.text(0.5, 0.5, "no metrics rows", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    n = len(metrics)
    labels = []
    mins, means, medians, p95s, maxs = [], [], [], [], []
    passes = []
    for r in metrics:
        labels.append(f"{r.get('estimator', '?')[:3]} c{r.get('coarse_grid_resolution', '?')}"
                      f"r{r.get('refinement_levels', '?')}")
        mins.append(float(r.get("frame_min_ms", 0.0) or 0.0))
        means.append(float(r.get("frame_mean_ms", 0.0) or 0.0))
        medians.append(float(r.get("frame_median_ms", 0.0) or 0.0))
        p95s.append(float(r.get("frame_p95_ms", 0.0) or 0.0))
        maxs.append(float(r.get("frame_max_ms", 0.0) or 0.0))
        passes.append(int(r.get("target_pass", 0)))
    x = np.arange(n)
    width = 0.16
    ax.bar(x - 1.5 * width, mins, width, label="min", color="tab:blue")
    ax.bar(x - 0.5 * width, means, width, label="mean", color="tab:green")
    ax.bar(x + 0.5 * width, p95s, width, label="p95", color="tab:orange")
    ax.bar(x + 1.5 * width, maxs, width, label="max", color="tab:red")
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0,
               label=f"target {target:g} ms")
    for i, p in enumerate(passes):
        ax.text(i, maxs[i] * 1.02, "PASS" if p else "FAIL",
                ha="center", va="bottom", fontsize=7,
                color="green" if p else "red")
    ax.set(title="Per-frame latency summary across configs",
           xlabel="config run", ylabel="latency [ms]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="upper right")


def plot_sky_track(ax, args, true_dirs, prior_dirs, est_dirs) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.0,
            alpha=0.6, label="horizon")
    if true_dirs is not None:
        ax.plot(true_dirs[:, 0], true_dirs[:, 1], color="red", linewidth=1.4,
                linestyle="--", label="true source", zorder=7)
        ax.scatter(true_dirs[0, 0], true_dirs[0, 1], color="red", marker="^",
                   s=60, zorder=8, label="source t=0")
    if prior_dirs is not None:
        ax.plot(prior_dirs[:, 0], prior_dirs[:, 1], color="grey", linewidth=1.0,
                linestyle=":", label="open-loop prior", zorder=4)
        ax.scatter(prior_dirs[:, 0], prior_dirs[:, 1], color="grey", s=18,
                   zorder=5)
    if est_dirs is not None:
        ax.plot(est_dirs[:, 0], est_dirs[:, 1], color="magenta", linewidth=1.4,
                marker="*", markersize=5, label="optimized estimate", zorder=9)
    ax.set(title="DOA tracking: true vs prior vs estimate",
           xlabel="l (H / x)", ylabel="m (E / y)", aspect="equal",
           adjustable="box", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def plot_doa_error(ax, true_dirs, prior_dirs, est_dirs) -> None:
    if true_dirs is None:
        ax.text(0.5, 0.5, "no DOA track", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    windows = np.arange(true_dirs.shape[0])
    err_true = np.linalg.norm(true_dirs[:, :2] - true_dirs[0:1, :2], axis=1)
    err_prior = (np.linalg.norm(prior_dirs[:, :2] - true_dirs[:, :2], axis=1)
                 if prior_dirs is not None else None)
    err_est = (np.linalg.norm(est_dirs[:, :2] - true_dirs[:, :2], axis=1)
               if est_dirs is not None else None)
    ax.plot(windows, err_true, color="red", linestyle="--",
            label="source drift from t=0")
    if err_prior is not None:
        ax.plot(windows, err_prior, color="grey", linestyle=":",
                label="prior error vs true")
    if err_est is not None:
        ax.plot(windows, err_est, color="magenta", linewidth=1.4,
                label="estimate error vs true")
    ax.set(title="Direction-error vs true source (lower is better)",
           xlabel="integration window", ylabel="||(l,m) error||")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot optimized CPU tracker benchmark + DOA tracking.")
    p.add_argument("--frames", type=Path,
                   help="per-frame latency CSV from the benchmark")
    p.add_argument("--metrics", type=Path,
                   help="summary metrics CSV from the benchmark (multi-run)")
    p.add_argument("--intensity", type=Path,
                   help="optional optimized intensity cube [T][F][B=1] float32")
    p.add_argument("--doa", type=Path,
                   help="optional per-window (l,m) estimated DOA track")
    p.add_argument("--output", type=Path, required=True, help="dashboard PNG")
    # Trajectory / geometry for the sky + DOA panels.
    p.add_argument("--n-time", type=int, default=15360)
    p.add_argument("--n-ant", type=int, default=64)
    p.add_argument("--integration-spectra", type=int, default=320)
    p.add_argument("--source-l0", type=float, default=0.03)
    p.add_argument("--source-m0", type=float, default=0.0)
    p.add_argument("--source-dl", type=float, default=1.0e-5)
    p.add_argument("--source-dm", type=float, default=0.0)
    p.add_argument("--prior-l0", type=float, default=0.0)
    p.add_argument("--prior-m0", type=float, default=0.0)
    p.add_argument("--target-ms-per-frame", type=float, default=TARGET_MS_PER_FRAME)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = load_frames_csv(args.frames) if args.frames else None
    metrics = load_metrics_csv(args.metrics) if args.metrics else None

    # Trajectory model true / prior window directions (mirrors the C++ models).
    true_dirs = moving_point_source_directions(
        args.source_l0, args.source_m0, args.source_dl, args.source_dm,
        args.n_time)
    prior_dirs = window_steering_directions(
        args.prior_l0, args.prior_m0, 0.0, 0.0,
        args.n_time, args.integration_spectra)
    est_dirs = load_doa_track(args.doa) if args.doa else None

    # Decide the panel layout: always include latency + DOA-error; add metrics
    # bars only if multiple runs; add sky + intensity heatmap as available.
    has_intensity = args.intensity is not None
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9),
                              constrained_layout=True)
    fig.suptitle("Optimized CPU beam tracker — benchmark + DOA dashboard",
                 fontsize=13)

    # Top-left: per-frame latency.
    ax = axes[0, 0]
    if frames is not None:
        plot_frame_latencies(ax, frames, args.target_ms_per_frame)
    else:
        ax.text(0.5, 0.5, "supply --frames FILE for per-frame latency",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Top-right: metrics bars (multi-run) or sky track.
    ax = axes[0, 1]
    if metrics is not None:
        plot_metrics_bars(ax, metrics, args.target_ms_per_frame)
    else:
        plot_sky_track(ax, args, true_dirs, prior_dirs, est_dirs)

    # Bottom-left: DOA error vs true (one true direction per window,
    # evaluated at the window's first sample — mirrors the C++ model).
    window_starts = (np.arange(prior_dirs.shape[0], dtype=np.int64)
                     * args.integration_spectra)
    window_starts = np.clip(window_starts, 0, args.n_time - 1)
    true_dirs_per_window = true_dirs[window_starts]
    plot_doa_error(axes[1, 0], true_dirs_per_window, prior_dirs, est_dirs)

    # Bottom-right: intensity heatmap (if provided) or sky track fallback.
    ax = axes[1, 1]
    if has_intensity:
        intensity = load_intensity(args.intensity, args.n_time,
                                   LOCAL_FREQUENCY_CHANNELS, TRACKER_BEAM_COUNT)
        power = intensity[:, :, 0]
        ax.imshow(normalized_db(power), origin="lower", aspect="auto",
                  cmap="magma",
                  extent=(-0.5, LOCAL_FREQUENCY_CHANNELS - 0.5,
                          -0.5, args.n_time - 0.5))
        ax.set(title="Optimized intensity [time, frequency]",
               xlabel="frequency channel", ylabel="time sample")
    else:
        plot_sky_track(ax, args, true_dirs, prior_dirs, est_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote optimized tracker dashboard to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
