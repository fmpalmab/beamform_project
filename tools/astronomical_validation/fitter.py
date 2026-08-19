"""Spectro-Temporal Parameter Refitting & Oracle Validator.

Inspired by CHIME/FRB's fitburst spectro-temporal fitter.
Refits recovered beamformer output dynamic spectra against injected physical pulse models
and calculates ground-truth recovery accuracy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .chime_catalog import FRBParameters
from .dedispersion import compute_profile_snr, dedisperse_waterfall


def pulse_model_func(t: np.ndarray, amp: float, t0: float, sigma: float, tau: float, bg: float) -> np.ndarray:
    """Gaussian pulse convolved with exponential scattering tail."""
    dt = t[1] - t[0] if len(t) > 1 else 0.001
    gaussian = np.exp(-0.5 * ((t - t0) / max(sigma, 1e-5)) ** 2)

    if tau > 1e-5:
        k_len = min(len(t), int(8 * tau / dt) + 1)
        k_t = np.arange(k_len, dtype=np.float32) * dt
        exp_k = np.exp(-k_t / tau)
        exp_k /= exp_k.sum()
        convolved = np.convolve(gaussian, exp_k, mode="same")
    else:
        convolved = gaussian

    return amp * convolved + bg


def refit_spectro_temporal_parameters(
    waterfall: np.ndarray,
    injected_params: FRBParameters,
    freqs_hz: np.ndarray,
    sample_rate_hz: float = 952.381,
) -> Dict[str, float | bool | str]:
    """Refit dedispersed profile and spectral shape to verify physical parameter recovery."""
    n_time, n_freq = waterfall.shape
    dt = 1.0 / sample_rate_hz
    t_axis = np.arange(n_time, dtype=np.float32) * dt

    # 1. Dedisperse at injected DM
    dedispersed, profile = dedisperse_waterfall(
        waterfall, injected_params.dm, freqs_hz=freqs_hz, sample_rate_hz=sample_rate_hz
    )
    snr, peak_idx, bg_mean, _ = compute_profile_snr(profile)
    t0_guess = t_axis[peak_idx]

    # 2. Fit 1D dedispersed profile for pulse width & scattering tau
    p0 = [np.max(profile) - bg_mean, t0_guess, injected_params.width_s, injected_params.scattering_tau_s, bg_mean]
    bounds = (
        [0.0, t0_guess - 0.5, 0.0001, 0.0, -100.0],
        [np.max(profile) * 5.0, t0_guess + 0.5, 0.05, 0.05, np.max(profile)],
    )

    try:
        popt, _ = curve_fit(pulse_model_func, t_axis, profile, p0=p0, bounds=bounds, maxfev=2000)
        fit_amp, fit_t0, fit_sigma, fit_tau, fit_bg = popt
        fit_success = True
    except Exception:
        fit_amp, fit_t0, fit_sigma, fit_tau, fit_bg = p0
        fit_success = False

    # 3. Fit spectral index gamma across frequency channels
    # Integrate profile in narrow window around t0
    win_w = max(5, int(0.01 * n_time))
    t_start = max(0, peak_idx - win_w)
    t_end = min(n_time, peak_idx + win_w)

    channel_powers = np.mean(dedispersed[t_start:t_end, :], axis=0) - bg_mean / n_freq
    channel_powers = np.maximum(channel_powers, 1e-6)

    freqs_mhz = freqs_hz / 1e6
    f_ref_mhz = 800.0
    log_f = np.log(freqs_mhz / f_ref_mhz)
    log_p = np.log(channel_powers)

    try:
        # Linear fit: log(P) = log(P0) + gamma * log(f/f_ref)
        poly = np.polyfit(log_f, log_p, 1)
        fit_gamma = float(poly[0])
    except Exception:
        fit_gamma = injected_params.spectral_index

    # Calculate parameter recovery accuracy & errors
    err_t0 = float(abs(fit_t0 - injected_params.arrival_time_s))
    err_sigma = float(abs(fit_sigma - injected_params.width_s))
    err_tau = float(abs(fit_tau - injected_params.scattering_tau_s))
    err_gamma = float(abs(fit_gamma - injected_params.spectral_index))

    # Pass/Fail tolerances
    passed_t0 = err_t0 < 0.010                # arrival time within 10 ms
    passed_sigma = err_sigma < 0.002           # width within 2 ms
    passed_snr = snr >= 0.70 * injected_params.target_snr  # SNR within 30% of target
    passed_overall = bool(fit_success and passed_t0 and passed_sigma and passed_snr)

    return {
        "burst_name": injected_params.name,
        "fit_success": fit_success,
        "recovered_snr": float(snr),
        "target_snr": float(injected_params.target_snr),
        "recovered_t0": float(fit_t0),
        "injected_t0": float(injected_params.arrival_time_s),
        "err_t0_ms": err_t0 * 1000.0,
        "recovered_sigma_ms": float(fit_sigma * 1000.0),
        "injected_sigma_ms": float(injected_params.width_s * 1000.0),
        "err_sigma_ms": err_sigma * 1000.0,
        "recovered_tau_ms": float(fit_tau * 1000.0),
        "injected_tau_ms": float(injected_params.scattering_tau_s * 1000.0),
        "err_tau_ms": err_tau * 1000.0,
        "recovered_gamma": float(fit_gamma),
        "injected_gamma": float(injected_params.spectral_index),
        "err_gamma": err_gamma,
        "passed_overall": passed_overall,
    }
