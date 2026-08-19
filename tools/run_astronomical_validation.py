#!/usr/bin/env python3
"""Run Astronomical Validation Suite for Beam Trackers.

Executes digital stream FRB injection tests, incoherent dedispersion sweeps,
spectro-temporal parameter fitting, beam pattern attenuation checks, and array
sensitivity scaling. Writes summary JSON report and renders multi-panel dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from astronomical_validation.chime_catalog import CHIME_CATALOG2_BENCHMARKS, get_frb_benchmark
from astronomical_validation.dedispersion import dedisperse_waterfall, run_dispersion_sweep
from astronomical_validation.injector import default_frequencies_hz, generate_frb_packed_voltage_stream
from astronomical_validation.runner import find_tracker_executable, run_beam_tracker
from astronomical_validation.validator import (
    test_dispersion_sweep_recovery,
    test_off_boresight_beam_response,
    test_radiometer_array_scaling,
    test_spectro_temporal_refit,
)


def render_astronomical_dashboard(
    params,
    engine: str,
    output_png: Path,
) -> None:
    """Render 4-panel astronomical validation dashboard PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_time = 15360
    n_ant = 64
    n_freq = 336
    freqs_hz = default_frequencies_hz(n_freq)

    # 1. Generate & run baseline pulse
    packed, _ = generate_frb_packed_voltage_stream(params, n_time=n_time, n_ant=n_ant, n_freq=n_freq)
    waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)

    # 2. Dedisperse & DM sweep
    dedispersed, profile = dedisperse_waterfall(waterfall, params.dm, freqs_hz=freqs_hz)
    sweep = run_dispersion_sweep(waterfall, params.dm, freqs_hz=freqs_hz)

    # 3. Off-boresight data
    off_res = test_off_boresight_beam_response(params, engine=engine, n_time=n_time, n_ant=n_ant, n_freq=n_freq)

    # 4. Scaling data
    scale_res = test_radiometer_array_scaling(params, engine=engine, n_time=n_time, n_freq=n_freq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel 1: Dynamic Spectrum & Dedispersed Profile
    ax1 = axes[0, 0]
    dt_ms = 1.05
    time_ms = np.arange(n_time) * dt_ms
    ax1.plot(time_ms, profile, color="#1f77b4", linewidth=1.2, label="Beamformed Profile")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Integrated Intensity [a.u.]")
    ax1.set_title(f"Dedispersed Pulse Profile ({params.name}, DM={params.dm})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: DM Sweep Butterfly Curve
    ax2 = axes[0, 1]
    ax2.plot(sweep["trial_dms"], sweep["snrs"], marker="o", markersize=3, color="#ff7f0e", label="S/N vs Trial DM")
    ax2.axvline(params.dm, color="black", linestyle="--", label=f"Injected DM ({params.dm})")
    ax2.axvline(sweep["best_dm"], color="red", linestyle=":", label=f"Peak DM ({sweep['best_dm']:.1f})")
    ax2.set_xlabel("Trial DM [pc cm^-3]")
    ax2.set_ylabel("Profile S/N")
    ax2.set_title("DM Sweep Butterfly Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Panel 3: Off-Boresight Beam Pattern Response
    ax3 = axes[1, 0]
    ax3.plot(off_res["off_axis_l"], off_res["norm_powers"], marker="s", color="#2ca02c", label="Synthesized Beam Response")
    ax3.set_xlabel("Off-Axis Offset [l-coordinate]")
    ax3.set_ylabel("Normalized Beam Power")
    ax3.set_title("Off-Boresight Beam Response Pattern")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # Panel 4: Radiometer Coherent Scaling SNR ~ sqrt(N_ant)
    ax4 = axes[1, 1]
    ants = scale_res["ant_counts"]
    snrs = scale_res["snrs"]
    ax4.loglog(ants, snrs, marker="D", color="#d62728", label=f"Measured (slope={scale_res['scaling_slope']:.2f})")
    # Ideal sqrt line
    ideal_snrs = snrs[0] * np.sqrt(np.array(ants) / ants[0])
    ax4.loglog(ants, ideal_snrs, linestyle="--", color="black", label="Theoretical SNR ~ sqrt(N_ant)")
    ax4.set_xlabel("Antenna Element Count (N_ant)")
    ax4.set_ylabel("Beamformed S/N")
    ax4.set_title("Coherent Array Sensitivity Scaling")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(loc="upper left")

    fig.suptitle(f"CHIME-Style Astronomical Validation Dashboard — Engine: {engine}", fontsize=14)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    print(f"Wrote dashboard plot to {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=str, default="cpu_v2",
                        help="Tracker engine to test (cpu_naive, cpu_v1, cpu_v2, cuda_fws)")
    parser.add_argument("--burst", type=str, default="FRB20180916B_canonical",
                        help="FRB benchmark name or 'all'")
    parser.add_argument("--outdir", type=Path, default=Path("results/astronomical_validation"),
                        help="Output directory for reports and plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        exe = find_tracker_executable()
        print(f"[astronomical_validation] Found tracker executable: {exe}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    burst_names = list(CHIME_CATALOG2_BENCHMARKS.keys()) if args.burst == "all" else [args.burst]
    results_summary = []

    print("\n" + "=" * 70)
    print(f"  CHIME-Style Astronomical Validation Engine: {args.engine}")
    print("=" * 70)

    all_passed = True
    for name in burst_names:
        params = get_frb_benchmark(name)
        print(f"\n--- Testing Burst Benchmark: {name} (DM={params.dm}) ---")

        # Test 1: Dispersion Sweep
        t1 = test_dispersion_sweep_recovery(params, engine=args.engine)
        print(f"  [Test 1] Dispersion Sweep DM Recovery: Injected={params.dm:.1f}, "
              f"Recovered={t1['recovered_dm']:.1f} pc cm^-3 -> {'PASS' if t1['passed'] else 'FAIL'}")

        # Test 2: Spectro-Temporal Refit
        t2 = test_spectro_temporal_refit(params, engine=args.engine)
        print(f"  [Test 2] Spectro-Temporal Refitting: SNR={t2['recovered_snr']:.1f}, "
              f"t0_err={t2['err_t0_ms']:.2f}ms -> {'PASS' if t2['passed'] else 'FAIL'}")

        # Test 3: Off-Boresight Beam Pattern
        t3 = test_off_boresight_beam_response(params, engine=args.engine)
        print(f"  [Test 3] Off-Boresight Beam Attenuation: Monotonic Falloff -> {'PASS' if t3['passed'] else 'FAIL'}")

        # Test 4: Array Sensitivity Scaling
        t4 = test_radiometer_array_scaling(params, engine=args.engine)
        print(f"  [Test 4] Radiometer Array Scaling: SNR Slope={t4['scaling_slope']:.2f} "
              f"(Expected ~ 0.50) -> {'PASS' if t4['passed'] else 'FAIL'}")

        burst_passed = t1['passed'] and t2['passed'] and t3['passed'] and t4['passed']
        all_passed = all_passed and burst_passed

        results_summary.append({
            "burst": name,
            "test_1_dispersion_sweep": t1,
            "test_2_spectro_temporal": t2,
            "test_3_off_boresight": t3,
            "test_4_array_scaling": t4,
            "burst_passed": burst_passed,
        })

    # Save JSON report
    report_file = args.outdir / "astronomical_validation_report.json"
    with open(report_file, "w") as f:
        json.dump({
            "engine": args.engine,
            "all_passed": all_passed,
            "results": results_summary,
        }, f, indent=2)
    print(f"\nWrote JSON report to {report_file}")

    # Render dashboard for primary benchmark
    primary_params = get_frb_benchmark(burst_names[0])
    dashboard_file = args.outdir / "astronomical_validation_dashboard.png"
    render_astronomical_dashboard(primary_params, args.engine, dashboard_file)

    print("=" * 70)
    if all_passed:
        print("  VERDICT: ASTRONOMICAL VALIDATION PASSED SUCCESSFULLY")
    else:
        print("  VERDICT: ASTRONOMICAL VALIDATION FAILED — CHECK REPORT")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
