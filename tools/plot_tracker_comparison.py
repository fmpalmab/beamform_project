#!/usr/bin/env python3
"""Render the CPU-vs-GPU beam-tracker comparison dashboard.

Reads the four CSV/JSON files a `tools/benchmark_cuda_tracker_v2` run writes
under --outdir (prefix "benchmark_cuda_tracker_v2"):

  <prefix>_summary.csv            sweep-friendly, one row per (n_ant, threads)
  <prefix>_frame_latencies.csv    per-window latency, one run's worth
  <prefix>_validation.csv         aggregate numerical parity, one run's worth
  <prefix>_window_validation.csv  per-window error + power, one run's worth
  <prefix>_metadata.json          array layout, trajectory, GPU info

and renders a single multi-panel dashboard PNG:
  1. Per-window latency curves, one line per engine, with a real-time budget
     threshold line.
  2. Performance summary: median/p95/min/max per-window latency, grouped by
     engine, annotated with speedup vs CPU Naive and vs CPU Opt v2.
  3. Numerical drift vs the CPU Naive reference, per window.
  4. Frequency-integrated power profile, overlaid across every engine, to
     confirm all six implementations are tracking the same source.
  5. A metadata legend (array layout, trajectory, hardware).

*_summary.csv is not itself plotted here (it is a sweep table across many
configurations, one row per (n_ant, threads) combination -- see
scripts/run_tracker_comparison_benchmarks.sh); this dashboard is a per-run
detail view. Extending it into a sweep-summary panel (e.g. best latency vs
thread count) is a natural follow-up once a sweep has been run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# (engine, kernel, display label, plot color) -- shared across every panel so
# a given implementation always gets the same color/label, and the legend
# order matches the CPU-then-GPU, slow-to-fast order used in the console
# output of tools/benchmark_cuda_tracker_v2.
ENGINES: list[tuple[str, str, str, str]] = [
    ("cpu", "naive", "CPU Naive", "#7f7f7f"),
    ("cpu", "opt_v1", "CPU Opt v1", "#1f77b4"),
    ("cpu", "opt_v2", "CPU Opt v2", "#17becf"),
    ("gpu", "twopass", "CUDA TwoPass", "#ff7f0e"),
    ("gpu", "fused", "CUDA Fused", "#d62728"),
    ("gpu", "warp_reduction", "CUDA WarpReduction", "#9467bd"),
]


def with_suffix(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def _rows_for(records: list[dict], engine: str, kernel: str) -> list[dict]:
    return [r for r in records if r["engine"] == engine and r["kernel"] == kernel]


def load_frame_latencies(path: Path) -> list[dict]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = [
            {
                "window_index": int(row["window_index"]),
                "time_start": int(row["time_start"]),
                "engine": row["engine"],
                "kernel": row["kernel"],
                "latency_ms": float(row["latency_ms"]),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def load_window_validation(path: Path) -> list[dict]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        return [
            {
                "window_index": int(row["window_index"]),
                "time_start": int(row["time_start"]),
                "engine": row["engine"],
                "kernel": row["kernel"],
                "mean_power": float(row["mean_power"]),
                "max_abs_diff": float(row["max_abs_diff"]),
                "rms_diff": float(row["rms_diff"]),
            }
            for row in reader
        ]


def load_validation(path: Path) -> list[dict]:
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        return [
            {
                "engine_name": row["engine_name"],
                "max_abs_diff": float(row["max_abs_diff"]),
                "rms_diff": float(row["rms_diff"]),
                "rel_error": float(row["rel_error"]),
                "max_rel_error": float(row["max_rel_error"]),
                "passed_tolerance": row["passed_tolerance"] not in ("0", "0.0", "False", "false"),
            }
            for row in reader
        ]


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _plot_latency_curves(ax, frames: list[dict], budget_ms: float) -> None:
    for engine, kernel, label, color in ENGINES:
        rows = sorted(_rows_for(frames, engine, kernel), key=lambda r: r["window_index"])
        if not rows:
            continue
        x = np.asarray([r["window_index"] for r in rows])
        y = np.asarray([r["latency_ms"] for r in rows], dtype=np.float64)
        ax.plot(x, y, marker=".", markersize=3, linewidth=1.2, label=label, color=color)
    ax.axhline(budget_ms, color="black", linestyle="--", linewidth=1.2,
              label=f"Real-time budget ({budget_ms:g} ms/frame)")
    ax.set_yscale("log")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Per-window latency [ms] (log)")
    ax.set_title("Latencia por ventana de integración")
    ax.legend(fontsize=7, ncols=2)
    ax.grid(True, which="both", alpha=0.25)


def _plot_performance_summary(ax, frames: list[dict]) -> None:
    labels: list[str] = []
    medians: list[float] = []
    p95s: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    colors: list[str] = []
    for engine, kernel, label, color in ENGINES:
        rows = _rows_for(frames, engine, kernel)
        if not rows:
            continue
        values = np.asarray([r["latency_ms"] for r in rows], dtype=np.float64)
        labels.append(label)
        medians.append(float(np.median(values)))
        p95s.append(float(np.percentile(values, 95.0)))
        mins.append(float(np.min(values)))
        maxs.append(float(np.max(values)))
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.2
    ax.bar(x - 1.5 * width, mins, width, label="Min", color="#c7c7c7")
    ax.bar(x - 0.5 * width, medians, width, label="Median", color="#1f77b4")
    ax.bar(x + 0.5 * width, p95s, width, label="P95", color="#ff7f0e")
    ax.bar(x + 1.5 * width, maxs, width, label="Max", color="#d62728")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Per-window latency [ms] (log)")
    ax.set_title("Resumen de rendimiento por ventana")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.25)

    naive_median = medians[labels.index("CPU Naive")] if "CPU Naive" in labels else None
    v2_median = medians[labels.index("CPU Opt v2")] if "CPU Opt v2" in labels else None
    for index, (label, median) in enumerate(zip(labels, medians)):
        if median <= 0.0:
            continue
        annotations = []
        if naive_median is not None and label != "CPU Naive":
            annotations.append(f"{naive_median / median:.2f}x Naive")
        if v2_median is not None and label not in ("CPU Naive", "CPU Opt v2"):
            annotations.append(f"{v2_median / median:.2f}x Opt v2")
        if annotations:
            ax.annotate("\n".join(annotations), (x[index], max(mins[index], medians[index])),
                       textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.5)


def _plot_error_drift(ax, window_validation: list[dict]) -> None:
    any_positive = False
    for engine, kernel, label, color in ENGINES:
        if kernel == "naive":
            continue  # trivially zero against itself; the flat line adds no signal
        rows = sorted(_rows_for(window_validation, engine, kernel), key=lambda r: r["window_index"])
        if not rows:
            continue
        x = np.asarray([r["window_index"] for r in rows])
        max_abs = np.asarray([r["max_abs_diff"] for r in rows], dtype=np.float64)
        rms = np.asarray([r["rms_diff"] for r in rows], dtype=np.float64)
        any_positive = any_positive or bool(np.any(max_abs > 0.0)) or bool(np.any(rms > 0.0))
        ax.plot(x, np.where(max_abs > 0.0, max_abs, np.nan), linewidth=1.1, color=color,
               label=f"{label} max|\u0394|")
        ax.plot(x, np.where(rms > 0.0, rms, np.nan), linestyle="--", linewidth=1.1,
               color=color, alpha=0.7, label=f"{label} RMS \u0394")
    ax.set_xlabel("Window index")
    if any_positive:
        ax.set_yscale("log")
        ax.set_ylabel("|\u0394 intensidad| vs CPU Naive (log)")
        ax.legend(fontsize=6, ncols=2)
    else:
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel("|\u0394 intensidad| vs CPU Naive")
        ax.text(0.5, 0.5, "Sin desviaci\u00f3n medible (bit-exact) en todas las ventanas",
               transform=ax.transAxes, ha="center", va="center", fontsize=9, color="green")
    ax.set_title("Deriva num\u00e9rica por ventana")
    ax.grid(True, which="both", alpha=0.25)


def _plot_power_profile(ax, window_validation: list[dict]) -> None:
    for engine, kernel, label, color in ENGINES:
        rows = sorted(_rows_for(window_validation, engine, kernel), key=lambda r: r["window_index"])
        if not rows:
            continue
        x = np.asarray([r["window_index"] for r in rows])
        power = np.asarray([r["mean_power"] for r in rows], dtype=np.float64)
        ax.plot(x, power, linewidth=1.3, color=color, label=label, alpha=0.85)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Potencia integrada en frecuencia [u.a.]")
    ax.set_title("Perfil de potencia \u2014 paridad entre implementaciones")
    ax.legend(fontsize=7, ncols=2)
    ax.grid(True, alpha=0.25)


def _format_metadata(metadata: dict) -> str:
    if not metadata:
        return "Sin metadata.json disponible para esta corrida."
    lines = []
    n_ant, n_freq = metadata.get("n_ant"), metadata.get("n_freq")
    if n_ant is not None and n_freq is not None:
        lines.append(
            f"Array: {n_ant} ant x {n_freq} freq   |   n_time={metadata.get('n_time')}   "
            f"|   integration_spectra={metadata.get('integration_spectra')}"
        )
    lines.append(f"OMP threads: {metadata.get('omp_threads')}")
    source = metadata.get("source_trajectory") or {}
    prior = metadata.get("prior_trajectory") or {}
    if source:
        lines.append(
            f"Source traj: l0={source.get('l0')}, m0={source.get('m0')}, "
            f"dl/sample={source.get('dl_per_sample')}, dm/sample={source.get('dm_per_sample')}"
        )
    if prior:
        lines.append(
            f"Prior traj:  l0={prior.get('l0')}, m0={prior.get('m0')}, "
            f"dl/sample={prior.get('dl_per_sample')}, dm/sample={prior.get('dm_per_sample')}"
        )
    gpu_name = metadata.get("gpu_name")
    if gpu_name:
        lines.append(
            f"GPU: {gpu_name}  (compute {metadata.get('compute_capability')}, "
            f"driver {metadata.get('cuda_driver_version')}, "
            f"runtime {metadata.get('cuda_runtime_version')})"
        )
    return "\n".join(lines)


def _plot_metadata_panel(ax, metadata: dict) -> None:
    ax.axis("off")
    ax.text(0.0, 0.95, "Configuraci\u00f3n de la corrida", fontsize=11, fontweight="bold",
           transform=ax.transAxes, va="top")
    ax.text(0.0, 0.72, _format_metadata(metadata), fontsize=9, family="monospace",
           transform=ax.transAxes, va="top")


def plot_dashboard(frames: list[dict], window_validation: list[dict], validation: list[dict],
                   metadata: dict, budget_ms: float, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 15), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)

    _plot_latency_curves(figure.add_subplot(grid[0, 0]), frames, budget_ms)
    _plot_performance_summary(figure.add_subplot(grid[0, 1]), frames)
    _plot_error_drift(figure.add_subplot(grid[1, 0]), window_validation)
    _plot_power_profile(figure.add_subplot(grid[1, 1]), window_validation)
    _plot_metadata_panel(figure.add_subplot(grid[2, :]), metadata)

    n_failed = sum(1 for row in validation if not row["passed_tolerance"])
    status = ("todas las implementaciones dentro de tolerancia" if n_failed == 0
             else f"{n_failed} implementaci\u00f3n(es) FUERA de tolerancia")
    figure.suptitle(f"CPU vs GPU Beam Tracker \u2014 {status}", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-prefix", type=Path,
                        default=Path("results/tracker_sweep/benchmark_cuda_tracker_v2"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/tracker_cpu_vs_gpu_dashboard.png"))
    parser.add_argument("--budget-ms", type=float, default=0.5,
                        help="Real-time per-window latency budget line (default: 0.5 ms/frame)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix: Path = args.input_prefix

    frames = load_frame_latencies(with_suffix(prefix, "_frame_latencies.csv"))
    validation_path = with_suffix(prefix, "_validation.csv")
    validation = load_validation(validation_path) if validation_path.exists() else []
    window_validation_path = with_suffix(prefix, "_window_validation.csv")
    window_validation = (
        load_window_validation(window_validation_path) if window_validation_path.exists() else []
    )
    metadata = load_metadata(with_suffix(prefix, "_metadata.json"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_dashboard(frames, window_validation, validation, metadata, args.budget_ms, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
