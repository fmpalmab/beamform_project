"""Astronomical Validation Suite Coordinator.

Executes the four core CHIME-style astronomical validation tests:
  1. Blind Dispersion Sweep & DM Recovery
  2. Spectro-Temporal Parameter Refitting
  3. Off-Boresight Pointing & Synthesized Beam Pattern Response
  4. Radiometer Coherent Sensitivity Array Scaling (SNR ~ sqrt(N_ant))
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .chime_catalog import CHIME_CATALOG2_BENCHMARKS, FRBParameters
from .dedispersion import compute_profile_snr, dedisperse_waterfall, run_dispersion_sweep
from .fitter import refit_spectro_temporal_parameters
from .injector import default_frequencies_hz, generate_frb_packed_voltage_stream
from .runner import run_beam_tracker


def test_dispersion_sweep_recovery(
    params: FRBParameters,
    engine: str = "cpu_v2",
    n_time: int = 15360,
    n_ant: int = 64,
    n_freq: int = 336,
) -> Dict[str, float | bool | str]:
    """Test 1: Blind Dispersion Sweep & DM Recovery."""
    packed, ref_wf = generate_frb_packed_voltage_stream(params, n_time=n_time, n_ant=n_ant, n_freq=n_freq)
    waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)

    freqs_hz = default_frequencies_hz(n_freq)
    sweep_results = run_dispersion_sweep(waterfall, params.dm, freqs_hz=freqs_hz)

    dm_err = float(sweep_results["dm_error"])
    passed = dm_err <= 5.0  # DM recovered within 5.0 pc cm^-3

    return {
        "test_name": "Dispersion_Sweep_Recovery",
        "burst_name": params.name,
        "engine": engine,
        "injected_dm": float(params.dm),
        "recovered_dm": float(sweep_results["best_dm"]),
        "dm_error": dm_err,
        "recovered_snr": float(sweep_results["recovered_snr"]),
        "passed": passed,
    }


def test_spectro_temporal_refit(
    params: FRBParameters,
    engine: str = "cpu_v2",
    n_time: int = 15360,
    n_ant: int = 64,
    n_freq: int = 336,
) -> Dict[str, float | bool | str]:
    """Test 2: Spectro-Temporal Refitting vs Ground Truth."""
    packed, _ = generate_frb_packed_voltage_stream(params, n_time=n_time, n_ant=n_ant, n_freq=n_freq)
    waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)

    freqs_hz = default_frequencies_hz(n_freq)
    fit_results = refit_spectro_temporal_parameters(waterfall, params, freqs_hz=freqs_hz)
    fit_results["test_name"] = "Spectro_Temporal_Refitting"
    fit_results["engine"] = engine
    fit_results["passed"] = fit_results["passed_overall"]
    return fit_results


def test_off_boresight_beam_response(
    params: FRBParameters,
    engine: str = "cpu_v2",
    n_time: int = 15360,
    n_ant: int = 64,
    n_freq: int = 336,
) -> Dict[str, float | bool | str | List[float]]:
    """Test 3: Off-Boresight Pointing & Synthesized Beam Pattern Response."""
    freqs_hz = default_frequencies_hz(n_freq)
    # Off-axis angles in l-coordinate (sine of angle off boresight)
    off_axis_l = [0.0, 0.005, 0.010, 0.015, 0.020, 0.025]
    peak_powers = []

    for l_val in off_axis_l:
        packed, _ = generate_frb_packed_voltage_stream(
            params, n_time=n_time, n_ant=n_ant, n_freq=n_freq,
            source_dir_lm=(l_val, 0.0), steer_dir_lm=(0.0, 0.0)
        )
        waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)
        _, profile = dedisperse_waterfall(waterfall, params.dm, freqs_hz=freqs_hz)
        snr, _, _, _ = compute_profile_snr(profile)
        peak_powers.append(snr)

    peak_powers = np.array(peak_powers)
    norm_powers = peak_powers / max(peak_powers[0], 1e-6)

    # Theoretical array factor FWHM ~ lambda / D_eff
    # For n_ant=64 (8x8 grid, 0.6m spacing -> D = 4.2m), at f_center=600MHz (lambda=0.5m),
    # FWHM ~ 0.5 / 4.2 ~ 0.12 rad ~ 0.12 l-units.
    # Check that power decreases monotonically with off-axis offset
    is_monotonic = bool(np.all(np.diff(norm_powers[:4]) <= 0.05))
    boresight_peak = float(norm_powers[0]) > 0.90
    passed = bool(is_monotonic and boresight_peak)

    return {
        "test_name": "Off_Boresight_Beam_Response",
        "burst_name": params.name,
        "engine": engine,
        "off_axis_l": off_axis_l,
        "norm_powers": norm_powers.tolist(),
        "passed": passed,
    }


def test_radiometer_array_scaling(
    params: FRBParameters,
    engine: str = "cpu_v2",
    n_time: int = 15360,
    n_freq: int = 336,
) -> Dict[str, float | bool | str | List[int] | List[float]]:
    """Test 4: Radiometer Coherent Sensitivity Array Scaling (SNR ~ sqrt(N_ant))."""
    freqs_hz = default_frequencies_hz(n_freq)
    ant_counts = [32, 64]
    snrs = []

    for n_ant in ant_counts:
        packed, _ = generate_frb_packed_voltage_stream(params, n_time=n_time, n_ant=n_ant, n_freq=n_freq)
        waterfall = run_beam_tracker(packed, n_time=n_time, n_ant=n_ant, n_freq=n_freq, engine=engine)
        _, profile = dedisperse_waterfall(waterfall, params.dm, freqs_hz=freqs_hz)
        snr, _, _, _ = compute_profile_snr(profile)
        snrs.append(snr)

    # Fit log(SNR) vs log(N_ant) -> slope should be close to 0.50
    log_nant = np.log(ant_counts)
    log_snr = np.log(snrs)
    poly = np.polyfit(log_nant, log_snr, 1)
    scaling_slope = float(poly[0])

    # Pass condition: slope between 0.40 and 0.60 (theoretical = 0.50)
    passed = 0.40 <= scaling_slope <= 0.60

    return {
        "test_name": "Radiometer_Array_Scaling",
        "burst_name": params.name,
        "engine": engine,
        "ant_counts": ant_counts,
        "snrs": snrs,
        "scaling_slope": scaling_slope,
        "expected_slope": 0.50,
        "passed": passed,
    }
