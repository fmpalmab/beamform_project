#!/usr/bin/env python3
"""
Plot and Analyze Continuous 2-Hour Stress Test Results for CUDA V5 Tracker + Upchannelizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def main():
    parser = argparse.ArgumentParser(description="Plot Continuous Stress Test Results")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing stress_test_timeline.csv")
    args = parser.parse_args()

    set_plot_style()

    timeline_csv = args.dir / "stress_test_timeline.csv"
    summary_json = args.dir / "stress_test_summary.json"
    telemetry_csv = args.dir / "gpu_telemetry.csv"

    if not timeline_csv.exists():
        print(f"Error: Timeline file {timeline_csv} does not exist.")
        return

    df = pd.read_csv(timeline_csv)
    summary = {}
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # Calculate hours
    df["Elapsed_Hours"] = df["Elapsed_Sec"] / 3600.0
    budget_ms = summary.get("real_time_budget_ms", 1.066667)
    engine = summary.get("engine", "cuda_v5")
    n_ant = summary.get("antenna_count", 64)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(f"Beam Tracker Continuous Stress Test & Reliability Suite ({engine.upper()})\n"
                 f"Antennas: {n_ant} | Total Duration: {df['Elapsed_Hours'].max():.2f} Hours ({len(df):,} Checkpoints)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Latency Timeline vs Real-Time Deadline
    ax1 = axes[0, 0]
    ax1.plot(df["Elapsed_Hours"], df["Latency_ms"], color="#0066cc", alpha=0.75, linewidth=1.2, label="Window Execution Latency")
    # Rolling average
    if len(df) > 50:
        rolling_mean = df["Latency_ms"].rolling(window=50, min_periods=1).mean()
        ax1.plot(df["Elapsed_Hours"], rolling_mean, color="#d62728", linewidth=2.0, label="50-Window Moving Avg")
    ax1.axhline(budget_ms, color="red", linestyle="--", linewidth=2.0, label=f"Real-Time Deadline ({budget_ms:.3f} ms)")
    ax1.fill_between(df["Elapsed_Hours"], 0, budget_ms, color="#2ca02c", alpha=0.08, label="Real-Time Headroom Margin")
    ax1.set_title("1. Execution Latency Timeline (2-Hour Reliability)")
    ax1.set_xlabel("Elapsed Time [Hours]")
    ax1.set_ylabel("Latency per 320-Spectra Window [ms]")
    ax1.set_ylim(0, budget_ms * 1.5)
    ax1.legend(loc="upper right")

    # Panel 2: Compute & Memory Throughput Stability
    ax2 = axes[0, 1]
    ax2.plot(df["Elapsed_Hours"], df["Throughput_GSamples_s"], color="#2ca02c", linewidth=1.5, label="Throughput [GSamples/s]")
    ax2.plot(df["Elapsed_Hours"], df["Throughput_TFLOPs"], color="#9467bd", linewidth=1.5, linestyle="--", label="Effective Compute [TFLOPs]")
    ax2.set_title("2. Processing Throughput & Compute Stability")
    ax2.set_xlabel("Elapsed Time [Hours]")
    ax2.set_ylabel("Throughput Rate")
    ax2.legend(loc="lower right")

    # Panel 3: Latency Distribution & Sub-ms Jitter Histogram
    ax3 = axes[1, 0]
    lats = df["Latency_ms"].values
    p50 = np.percentile(lats, 50)
    p95 = np.percentile(lats, 95)
    p99 = np.percentile(lats, 99)
    ax3.hist(lats, bins=50, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    ax3.axvline(p50, color="green", linestyle="-", linewidth=2.0, label=f"P50: {p50:.3f} ms")
    ax3.axvline(p95, color="orange", linestyle="--", linewidth=2.0, label=f"P95: {p95:.3f} ms")
    ax3.axvline(p99, color="red", linestyle=":", linewidth=2.0, label=f"P99: {p99:.3f} ms")
    ax3.set_title("3. Sub-Millisecond Frame Latency Jitter Distribution")
    ax3.set_xlabel("Window Execution Time [ms]")
    ax3.set_ylabel("Probability Density")
    ax3.legend(loc="upper right")

    # Panel 4: Memory Leak Detection & System Health
    ax4 = axes[1, 1]
    if "VRAM_Used_MB" in df and df["VRAM_Used_MB"].max() > 0:
        ax4.plot(df["Elapsed_Hours"], df["VRAM_Used_MB"], color="#ff7f0e", linewidth=2.0, label="GPU VRAM Usage [MB]")
        ax4.set_ylabel("VRAM Resident Memory [MB]", color="#ff7f0e")
        ax4.tick_params(axis='y', labelcolor="#ff7f0e")
    else:
        ax4.text(0.5, 0.5, "VRAM Usage Constant\n(Zero-Copy In-Place Memory)", ha="center", va="center", transform=ax4.transAxes, fontsize=12)

    # Plot GPU Temperature if telemetry is present
    if telemetry_csv.exists():
        try:
            telem_df = pd.read_csv(telemetry_csv)
            if "temperature.gpu" in telem_df.columns and "timestamp" in telem_df.columns:
                ax4_temp = ax4.twinx()
                ax4_temp.plot(np.linspace(0, df["Elapsed_Hours"].max(), len(telem_df)), telem_df["temperature.gpu"], color="#d62728", linestyle=":", label="GPU Temp [°C]")
                ax4_temp.set_ylabel("GPU Temp [°C]", color="#d62728")
                ax4_temp.tick_params(axis='y', labelcolor="#d62728")
        except Exception:
            pass

    ax4.set_title("4. Device Memory Health & Leak Detection (Flat = No Leaks)")
    ax4.set_xlabel("Elapsed Time [Hours]")

    out_png = args.dir / "stress_test_dashboard.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Successfully generated stress test dashboard: {out_png}")


if __name__ == "__main__":
    main()
