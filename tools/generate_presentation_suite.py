#!/usr/bin/env python3
"""Comprehensive Presentation Material & Astronomical Visualization Suite for CUDA Beam Tracker V5.

Generates complete publication- and presentation-grade visualizations covering:
1. Multi-Antenna Astronomical Dashboards (64, 128, 256 antennas)
2. Tracker Motion, Moving Source Kinematics & Angular Pointing Error
3. Approach & Departure Intensity Profiles (Synthesized Beam Power vs Proximity)
4. Dispersion Physics & Dedispersed Pulse Profile Reconstruction
5. Synthesized Beamfootprint & Array Spatial Directivity Comparison (64 vs 128 vs 256)
6. Multi-Generation Latency & Speedup Benchmarks (CPU Naive -> CPU Opt -> V3 -> V4 -> V5)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(tools_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astronomical_validation.chime_catalog import get_frb_benchmark
from astronomical_validation.dedispersion import (
    compute_profile_snr,
    dedisperse_waterfall,
    run_dispersion_sweep,
)
from astronomical_validation.injector import (
    default_frequencies_hz,
    generate_frb_packed_voltage_stream,
    synthesize_frb_intensity_waterfall,
)
from astronomical_validation.runner import find_tracker_executable, run_beam_tracker


def set_presentation_style():
    """Configure modern, publication-quality presentation plot styles."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.autolayout": False,
    })


# ---------------------------------------------------------------------------
# 1. Multi-Antenna Astronomical Dashboards (64, 128, 256 Antennas)
# ---------------------------------------------------------------------------
def generate_astronomical_dashboards(outdir: Path, engine: str = "cuda_v5"):
    """Run astronomical test on V5 for 64, 128, and 256 antennas and plot dashboards."""
    print("\n[1/6] Generating Multi-Antenna Astronomical Dashboards on V5...")
    params = get_frb_benchmark("FRB20180916B_canonical")
    n_time = 15360
    n_freq = 336
    freqs_hz = default_frequencies_hz(n_freq)
    dt_ms = 1.05
    time_ms = np.arange(n_time) * dt_ms

    antenna_configs = [64, 128, 256]

    for n_ant in antenna_configs:
        print(f"  -> Running V5 astronomical test for N_ant={n_ant}...")
        packed, _ = generate_frb_packed_voltage_stream(params, n_time=n_time, n_ant=n_ant, n_freq=n_freq)
        waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)

        dedispersed, profile = dedisperse_waterfall(waterfall, params.dm, freqs_hz=freqs_hz)
        sweep = run_dispersion_sweep(waterfall, params.dm, freqs_hz=freqs_hz)

        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        fig.suptitle(f"CUDA Beam Tracker V5: Astronomical FRB Validation Dashboard (N_ant = {n_ant})\n"
                     f"Source: {params.name} | DM = {params.dm:.2f} pc cm⁻³ | Width = {params.width_s * 1000.0:.1f} ms",
                     fontsize=15, fontweight="bold")

        # 1. Dedispersed Pulse Profile
        ax1 = axes[0, 0]
        ax1.plot(time_ms, profile, color="#0066cc", linewidth=1.5, label="Beamformed Coherent Profile")
        snr, peak_val, _, _ = compute_profile_snr(profile)
        ax1.set_title(f"Dedispersed Pulse Profile (Peak S/N = {snr:.1f})")
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Integrated Intensity [a.u.]")
        ax1.legend(loc="upper right")

        # Zoom into the pulse burst
        burst_center_ms = params.arrival_time_s * 1000.0
        ax1.set_xlim(max(0, burst_center_ms - 600), min(n_time * dt_ms, burst_center_ms + 600))

        # 2. Butterfly DM Sweep
        ax2 = axes[0, 1]
        ax2.plot(sweep["trial_dms"], sweep["snrs"], marker="o", markersize=4, color="#e65c00", linewidth=1.8, label="S/N vs Trial DM")
        ax2.axvline(params.dm, color="black", linestyle="--", linewidth=1.5, label=f"Injected DM ({params.dm:.1f})")
        ax2.axvline(sweep["best_dm"], color="red", linestyle=":", linewidth=2, label=f"Recovered Peak DM ({sweep['best_dm']:.1f})")
        ax2.set_title(f"Blind Dispersion Sweep & Butterfly Curve (Error = {sweep['dm_error']:.2f} pc cm⁻³)")
        ax2.set_xlabel("Trial DM [pc cm⁻³]")
        ax2.set_ylabel("Profile S/N")
        ax2.legend(loc="upper right")

        # 3. Dynamic Spectrum (Dispersed Waterfall vs Dedispersed Waterfall)
        ax3 = axes[1, 0]
        # Downsample time for clean visualization
        t_sub = 8
        wf_sub = waterfall.reshape(n_time // t_sub, t_sub, n_freq).mean(axis=1)
        extent = [0, n_time * dt_ms, freqs_hz[-1] / 1e6, freqs_hz[0] / 1e6]
        im3 = ax3.imshow(wf_sub.T, aspect="auto", origin="upper", extent=extent, cmap="viridis")
        ax3.set_title(f"Dynamic Spectrum Waterfall [Time × Frequency] (N_ant={n_ant})")
        ax3.set_xlabel("Time [ms]")
        ax3.set_ylabel("Frequency [MHz]")
        fig.colorbar(im3, ax=ax3, label="Intensity [dB / linear]")

        # 4. Dedispersed Waterfall
        ax4 = axes[1, 1]
        dedisp_sub = dedispersed.reshape(n_time // t_sub, t_sub, n_freq).mean(axis=1)
        im4 = ax4.imshow(dedisp_sub.T, aspect="auto", origin="upper", extent=extent, cmap="magma")
        ax4.set_title("Coherently Dedispersed Waterfall (Pulse Time-Aligned)")
        ax4.set_xlabel("Time [ms]")
        ax4.set_ylabel("Frequency [MHz]")
        ax4.set_xlim(max(0, burst_center_ms - 600), min(n_time * dt_ms, burst_center_ms + 600))
        fig.colorbar(im4, ax=ax4, label="Intensity [a.u.]")

        dashboard_png = outdir / f"presentation_1_astronomical_dashboard_{n_ant}_ant.png"
        fig.savefig(dashboard_png, dpi=180)
        plt.close(fig)
        print(f"  -> Saved: {dashboard_png}")


# ---------------------------------------------------------------------------
# 2. Moving Target Kinematics & Angular Pointing Error
# ---------------------------------------------------------------------------
def generate_tracker_kinematics(outdir: Path):
    """Plot how tracker trajectory follows moving object and track angular error."""
    print("\n[2/6] Generating Tracker Motion & Pointing Error Analysis...")
    n_time = 15360
    spectra = 320
    dt_ms = 0.2  # 200 us sample rate -> 3.072s total
    time_s = np.arange(n_time) * (dt_ms / 1000.0)

    # Moving source trajectory
    l0, m0 = -0.15, -0.10
    dl, dm = 2.0e-5, 1.3e-5

    source_l = l0 + np.arange(n_time) * dl
    source_m = m0 + np.arange(n_time) * dm

    # Tracker discrete integration window steering
    window_count = (n_time + spectra - 1) // spectra
    window_starts = np.arange(window_count) * spectra
    win_l = l0 + window_starts * dl
    win_m = m0 + window_starts * dm

    # Per-sample tracker steering (stepped per window)
    steer_l = np.repeat(win_l, spectra)[:n_time]
    steer_m = np.repeat(win_m, spectra)[:n_time]

    # Angular error in direction cosines
    error_l = source_l - steer_l
    error_m = source_m - steer_m
    angular_error_arcmin = np.sqrt(error_l**2 + error_m**2) * (180.0 / np.pi) * 60.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle("CUDA Beam Tracker V5: Dynamic Tracking Kinematics & Angular Pointing Error",
                 fontsize=15, fontweight="bold")

    # 1. 2D Sky Plane Trajectory
    ax1 = axes[0, 0]
    ax1.plot(source_l, source_m, color="red", linewidth=2.0, label="Continuous Source Trajectory (Sky Path)")
    ax1.step(steer_l, steer_m, color="#0066cc", where="post", linewidth=1.5, linestyle="--", label="Tracker Window Steering Center")
    ax1.scatter([source_l[0]], [source_m[0]], color="green", s=100, zorder=5, label="Start Position")
    ax1.scatter([source_l[-1]], [source_m[-1]], color="darkred", s=100, marker="X", zorder=5, label="End Position")
    ax1.set_title("2D Sky Direction Cosine Plane (l, m)")
    ax1.set_xlabel("Direction Cosine l (East-West)")
    ax1.set_ylabel("Direction Cosine m (North-South)")
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal", adjustable="datalim")

    # 2. Coordinate Tracking vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_s, source_l, color="red", label="Source l(t)")
    ax2.plot(time_s, steer_l, color="#0066cc", linestyle="--", label="Tracker l(t)")
    ax2.plot(time_s, source_m, color="orange", label="Source m(t)")
    ax2.plot(time_s, steer_m, color="purple", linestyle="--", label="Tracker m(t)")
    ax2.set_title("Coordinate Propagation vs Time")
    ax2.set_xlabel("Observation Time [s]")
    ax2.set_ylabel("Direction Cosine Coordinates")
    ax2.legend(loc="upper left")

    # 3. Sawtooth Pointing Error
    ax3 = axes[1, 0]
    ax3.plot(time_s[:1600], angular_error_arcmin[:1600], color="#d62728", linewidth=1.5)
    ax3.set_title("Discrete Window Sawtooth Pointing Error (First 5 Windows)")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Pointing Offset Δθ [arcminutes]")
    for w in range(1, 6):
        ax3.axvline(w * spectra * (dt_ms / 1000.0), color="grey", linestyle=":", alpha=0.7)

    # 4. Error Histogram & Max Loss Bound
    ax4 = axes[1, 1]
    ax4.hist(angular_error_arcmin, bins=30, color="#17becf", edgecolor="black", alpha=0.75)
    mean_err = np.mean(angular_error_arcmin)
    max_err = np.max(angular_error_arcmin)
    ax4.axvline(mean_err, color="blue", linestyle="--", linewidth=1.8, label=f"Mean Error: {mean_err:.2f}'")
    ax4.axvline(max_err, color="red", linestyle="-", linewidth=1.8, label=f"Max Window Error: {max_err:.2f}'")
    ax4.set_title("Pointing Offset Distribution across Full Observation")
    ax4.set_xlabel("Angular Error [arcminutes]")
    ax4.set_ylabel("Sample Count")
    ax4.legend(loc="upper right")

    png_path = outdir / "presentation_2_tracker_kinematics_and_error.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {png_path}")


# ---------------------------------------------------------------------------
# 3. Proximity & Beam Crossing Dynamics (Approach, Transit, Departure)
# ---------------------------------------------------------------------------
def generate_proximity_dynamics(outdir: Path):
    """Plot beam intensity changes as target approaches boresight and departs."""
    print("\n[3/6] Generating Target Approach / Transit / Departure Dynamics...")
    # Simulate a target crossing a beam with closest approach at center
    angles_deg = np.linspace(-3.0, 3.0, 500)
    theta_rad = np.radians(angles_deg)

    # Synthesized beam response for 64, 128, and 256 antennas
    # D_eff = columns * spacing -> 64: 8x8 (4.2m), 128: 16x8 (9.0m), 256: 16x16 (9.0m)
    wavelength = 0.5  # 600 MHz
    
    # Array factor sinc-like profile
    patterns = {}
    for n_ant, d_eff, color, label in [
        (64, 4.2, "#1f77b4", "64 Antennas (D = 4.2 m, FWHM ≈ 6.8°)"),
        (128, 6.0, "#ff7f0e", "128 Antennas (D = 6.0 m, FWHM ≈ 4.8°)"),
        (256, 9.0, "#2ca02c", "256 Antennas (D = 9.0 m, FWHM ≈ 3.2°)"),
    ]:
        u = (math.pi * d_eff / wavelength) * np.sin(theta_rad)
        power = (np.sinc(u / math.pi))**2
        patterns[n_ant] = (power, color, label)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle("CUDA Beam Tracker V5: Source Approach, Transit (Boresight), and Departure Dynamics",
                 fontsize=15, fontweight="bold")

    # 1. Linear Normalized Intensity vs Angular Distance
    ax1 = axes[0]
    for n_ant, (power, color, label) in patterns.items():
        ax1.plot(angles_deg, power, color=color, linewidth=2.0, label=label)
    ax1.axvline(0, color="black", linestyle="--", alpha=0.7, label="Closest Approach (Boresight Alignment)")
    ax1.axhline(0.5, color="red", linestyle=":", label="Half-Power (-3 dB) Beam Width")
    ax1.set_title("Normalized Coherent Intensity Profile vs Source Offset")
    ax1.set_xlabel("Angular Offset from Beam Center [degrees]")
    ax1.set_ylabel("Normalized Beamformed Power [linear]")
    ax1.legend(loc="upper right")
    ax1.set_ylim(-0.05, 1.05)

    # 2. Logarithmic (dB) Beam Attenuation Curve
    ax2 = axes[1]
    for n_ant, (power, color, label) in patterns.items():
        power_db = 10.0 * np.log10(np.maximum(power, 1e-4))
        ax2.plot(angles_deg, power_db, color=color, linewidth=2.0, label=label)
    ax2.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax2.axhline(-3.0, color="red", linestyle=":", label="-3 dB Cutoff")
    ax2.axhline(-13.2, color="purple", linestyle="--", label="First Sidelobe Level (-13.2 dB)")
    ax2.set_title("Synthesized Beam Power Response [dB]")
    ax2.set_xlabel("Angular Offset from Beam Center [degrees]")
    ax2.set_ylabel("Power Response [dB]")
    ax2.legend(loc="upper right")
    ax2.set_ylim(-35, 2)

    png_path = outdir / "presentation_3_proximity_approach_departure.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {png_path}")


# ---------------------------------------------------------------------------
# 4. Dispersion Physics & Reconstruction
# ---------------------------------------------------------------------------
def generate_dispersion_physics(outdir: Path):
    """Plot cold plasma dispersion curve, dedispersion realignment and SNR gain."""
    print("\n[4/6] Generating Dispersion Physics & Signal Reconstruction...")
    n_freq = 336
    freqs_mhz = np.linspace(300, 400.8, n_freq)
    dm = 348.82
    k_dm = 4.148808e3  # MHz^2 pc^-1 cm^3 s

    # Cold plasma delay: t(f) = k_dm * DM * (f^-2 - f_top^-2)
    f_top = freqs_mhz[-1]
    delays_ms = k_dm * dm * (freqs_mhz**(-2) - f_top**(-2)) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle("Cold Plasma Dispersion Mechanics & Coherent Reconstruction",
                 fontsize=15, fontweight="bold")

    # 1. Quadratic Delay Law
    ax1 = axes[0]
    ax1.plot(delays_ms, freqs_mhz, color="#d62728", linewidth=2.5, label=f"Dispersion Curve: Δt ∝ DM·f⁻² (DM={dm:.1f})")
    ax1.set_title("Interstellar Plasma Time-Delay vs Frequency")
    ax1.set_xlabel("Relative Arrival Delay Δt [ms]")
    ax1.set_ylabel("Observing Frequency [MHz]")
    ax1.legend(loc="lower left")
    ax1.invert_yaxis()

    # 2. Coherent Pulse SNR Enhancement Before vs After Dedispersion
    ax2 = axes[1]
    t = np.linspace(-50, 50, 500)
    raw_smeared = np.exp(-(t / 25.0)**2) * 0.15 + np.random.normal(0, 0.05, len(t))
    dedispersed = np.exp(-(t / 3.0)**2) * 1.0 + np.random.normal(0, 0.05, len(t))

    ax2.plot(t, raw_smeared, color="grey", linestyle="--", linewidth=1.5, label="Dispersed (Uncompensated) Signal: S/N ≈ 3.0")
    ax2.plot(t, dedispersed, color="#0066cc", linewidth=2.0, label="Coherently Dedispersed Pulse: S/N ≈ 20.0 (6.7x Gain)")
    ax2.set_title("Pulse Compression & Coherent Signal Reconstruction")
    ax2.set_xlabel("Time around Burst Center [ms]")
    ax2.set_ylabel("Integrated Intensity [a.u.]")
    ax2.legend(loc="upper right")

    png_path = outdir / "presentation_4_dispersion_physics_reconstruction.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {png_path}")


# ---------------------------------------------------------------------------
# 5. Spatial Beam Footprint Comparison (64 vs 128 vs 256 Antennas)
# ---------------------------------------------------------------------------
def generate_beam_footprints(outdir: Path):
    """Plot 2D beam pattern contours on the sky for 64, 128, and 256 antenna arrays."""
    print("\n[5/6] Generating 2D Synthesized Beam Footprints on the Sky...")
    n_pix = 201
    l_axis = np.linspace(-0.3, 0.3, n_pix)
    m_axis = np.linspace(-0.3, 0.3, n_pix)
    L, M = np.meshgrid(l_axis, m_axis)
    R = np.sqrt(L**2 + M**2)

    wavelength = 0.5  # 600 MHz

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("Synthesized Beam Footprint & Spatial Resolving Power (600 MHz Center Frequency)",
                 fontsize=15, fontweight="bold")

    configs = [
        (64, 4.2, 4.2, axes[0], "64 Antennas (8×8 Array, D=4.2m)"),
        (128, 9.0, 4.2, axes[1], "128 Antennas (16×8 Array, D=9.0×4.2m)"),
        (256, 9.0, 9.0, axes[2], "256 Antennas (16×16 Array, D=9.0m)"),
    ]

    for n_ant, dx, dy, ax, title in configs:
        ux = (math.pi * dx / wavelength) * L
        uy = (math.pi * dy / wavelength) * M
        power = (np.sinc(ux / math.pi) * np.sinc(uy / math.pi))**2
        power_db = 10.0 * np.log10(np.maximum(power, 1e-3))

        im = ax.imshow(power_db, origin="lower", extent=[l_axis[0], l_axis[-1], m_axis[0], m_axis[-1]],
                       cmap="magma", vmin=-25, vmax=0)
        ax.contour(L, M, power_db, levels=[-10, -3], colors=["cyan", "white"], linewidths=[1.0, 1.5])
        ax.set_title(title)
        ax.set_xlabel("Direction Cosine l")
        ax.set_ylabel("Direction Cosine m")
        ax.scatter([0], [0], color="lime", marker="+", s=100, label="Boresight")
        ax.legend(loc="upper right", fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Beam Power Response [dB]", shrink=0.85)

    png_path = outdir / "presentation_5_spatial_beam_footprints_64_128_256.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {png_path}")


# ---------------------------------------------------------------------------
# 6. Multi-Generation Benchmark & Speedup Comparison
# ---------------------------------------------------------------------------
def generate_benchmark_comparison(outdir: Path):
    """Plot multi-generation benchmark numbers comparing CPU, V3, V4, and V5 across 64/128/256 ant."""
    print("\n[6/6] Generating Multi-Generation Benchmark Charts...")

    # Data from measured benchmarks
    categories = ["64 Antennas", "128 Antennas", "256 Antennas"]
    cpu_naive = [352.93, 697.09, 1379.85]
    cpu_opt_v2 = [47.95, 91.75, 192.54]
    v3_device = [15.18, 0, 0]  # V3 didn't support 128/256
    v4_device = [14.59, 57.54, 119.79]
    v5_device = [14.38, 18.28, 34.85]
    v5_kernel = [13.33, 17.22, 33.24]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("CUDA Beam Tracker Architectural Progression: Multi-Generation Benchmark Suite\n"
                 "Array: N_freq = 336, N_time = 15360 (NVIDIA Quadro P1000 GPU / sm_61)",
                 fontsize=15, fontweight="bold")

    # 1. Latency Bar Chart (Log scale)
    ax1 = axes[0]
    x = np.arange(len(categories))
    w = 0.15

    ax1.bar(x - 2 * w, cpu_naive, w, label="CPU Naive (Single-Thread)", color="#7f7f7f")
    ax1.bar(x - 1 * w, cpu_opt_v2, w, label="CPU Opt v2 (8 OpenMP Threads)", color="#17becf")
    ax1.bar(x, v4_device, w, label="CUDA V4 Device Resident", color="#ff7f0e")
    ax1.bar(x + 1 * w, v5_device, w, label="CUDA V5 Device Resident (Unified Engine)", color="#2ca02c")
    ax1.bar(x + 2 * w, v5_kernel, w, label="CUDA V5 Batched Kernel Only", color="#0066cc")

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Execution Latency [ms] (Log Scale)")
    ax1.set_title("Execution Time Comparison (Lower is Faster)")
    ax1.legend(loc="upper left", fontsize=9.5)
    ax1.grid(True, which="both", axis="y", alpha=0.3)

    # Annotate V5 numbers
    for i in range(len(categories)):
        ax1.text(x[i] + w, v5_device[i] * 1.15, f"{v5_device[i]:.1f} ms", ha="center", fontsize=9, fontweight="bold", color="#2ca02c")
        if v4_device[i] > 0:
            speedup_vs_v4 = v4_device[i] / v5_device[i]
            ax1.text(x[i] + w, v5_device[i] * 0.5, f"{speedup_vs_v4:.1f}x vs V4", ha="center", fontsize=8.5, fontweight="bold", color="#004d00")

    # 2. Speedup vs CPU Naive Reference
    ax2 = axes[1]
    speedup_v4 = [cpu_naive[i] / max(v4_device[i], 1e-4) for i in range(3)]
    speedup_v5 = [cpu_naive[i] / v5_device[i] for i in range(3)]
    speedup_v5_kernel = [cpu_naive[i] / v5_kernel[i] for i in range(3)]

    ax2.plot(categories, speedup_v5_kernel, marker="D", markersize=9, linewidth=2.5, color="#0066cc", label="CUDA V5 Batched Kernel (Peak Compute)")
    ax2.plot(categories, speedup_v5, marker="s", markersize=8, linewidth=2.2, color="#2ca02c", label="CUDA V5 Device Resident (Unified)")
    ax2.plot(categories, speedup_v4, marker="o", markersize=7, linewidth=1.8, color="#ff7f0e", linestyle="--", label="CUDA V4 Device Resident (V4 Bottleneck)")

    ax2.set_title("GPU Speedup vs CPU Naive Baseline (Higher is Better)")
    ax2.set_ylabel("Speedup Multiplier (x Baseline)")
    ax2.legend(loc="center left", fontsize=10)
    ax2.set_ylim(0, 50)

    # Annotate speedup numbers
    for i in range(len(categories)):
        ax2.annotate(f"{speedup_v5[i]:.1f}x", (categories[i], speedup_v5[i] + 1.5), ha="center", fontweight="bold", color="#2ca02c")
        ax2.annotate(f"{speedup_v4[i]:.1f}x", (categories[i], speedup_v4[i] - 3.0), ha="center", color="#ff7f0e")

    png_path = outdir / "presentation_6_multi_generation_benchmark_speedup.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"  -> Saved: {png_path}")


def main():
    set_presentation_style()
    outdir = Path("results/presentation_assets")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  CUDA BEAM TRACKER V5: PRESENTATION MATERIAL & VISUALIZATION GENERATOR")
    print("=" * 75)

    generate_astronomical_dashboards(outdir, engine="cuda_v5")
    generate_tracker_kinematics(outdir)
    generate_proximity_dynamics(outdir)
    generate_dispersion_physics(outdir)
    generate_beam_footprints(outdir)
    generate_benchmark_comparison(outdir)

    print("\n" + "=" * 75)
    print(f"  SUCCESS: All presentation visual materials generated in: {outdir.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
