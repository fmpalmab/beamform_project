"""Incoherent Dedispersion & Dispersion Sweep Analysis.

Implements DM trial shifting, frequency integration, time profile construction,
SNR computation, and DM butterfly curve generation for beam tracker dynamic spectra.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from .injector import K_DM, default_frequencies_hz


def dedisperse_waterfall(
    waterfall: np.ndarray,
    dm_trial: float,
    freqs_hz: np.ndarray | None = None,
    sample_rate_hz: float = 952.381,
    f_ref_hz: float = 800e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dedisperse 2D intensity waterfall (n_time, n_freq) at trial DM.

    Returns (dedispersed_waterfall, time_profile_1d).
    """
    n_time, n_freq = waterfall.shape
    if freqs_hz is None:
        freqs_hz = default_frequencies_hz(n_freq)

    dt = 1.0 / sample_rate_hz
    freqs_mhz = freqs_hz / 1e6
    f_ref_mhz = f_ref_hz / 1e6

    # Delays per frequency channel in seconds
    delays_s = K_DM * dm_trial * ((freqs_mhz ** -2.0) - (f_ref_mhz ** -2.0))
    shift_samples = np.round(delays_s / dt).astype(int)

    dedispersed = np.zeros_like(waterfall)
    for f_idx in range(n_freq):
        shift = shift_samples[f_idx]
        dedispersed[:, f_idx] = np.roll(waterfall[:, f_idx], -shift)

    profile = np.sum(dedispersed, axis=1)
    return dedispersed, profile


def compute_profile_snr(profile: np.ndarray, peak_idx: int | None = None) -> Tuple[float, int, float, float]:
    """Compute S/N ratio of a 1D time profile.

    Returns (snr, peak_index, mean_baseline, std_baseline).
    """
    n_time = len(profile)
    if peak_idx is None:
        peak_idx = int(np.argmax(profile))

    # Mask region around peak (+- 5% of window) for baseline statistics
    mask_width = max(10, int(0.05 * n_time))
    mask_start = max(0, peak_idx - mask_width)
    mask_end = min(n_time, peak_idx + mask_width)

    baseline_mask = np.ones(n_time, dtype=bool)
    baseline_mask[mask_start:mask_end] = False

    if not np.any(baseline_mask):
        baseline_mask[:] = True

    baseline_mean = float(np.mean(profile[baseline_mask]))
    baseline_std = float(np.std(profile[baseline_mask]))

    if baseline_std <= 1e-12:
        snr = 0.0
    else:
        snr = float((profile[peak_idx] - baseline_mean) / baseline_std)

    return snr, peak_idx, baseline_mean, baseline_std


def run_dispersion_sweep(
    waterfall: np.ndarray,
    dm_injected: float,
    freqs_hz: np.ndarray | None = None,
    dm_range_pct: float = 0.4,
    n_steps: int = 51,
    sample_rate_hz: float = 952.381,
) -> Dict[str, np.ndarray | float]:
    """Perform a trial DM sweep around the injected DM to generate butterfly curve.

    Returns dictionary with 'trial_dms', 'snrs', 'best_dm', 'recovered_snr'.
    """
    dm_min = max(0.0, dm_injected * (1.0 - dm_range_pct))
    dm_max = dm_injected * (1.0 + dm_range_pct)
    trial_dms = np.linspace(dm_min, dm_max, n_steps)

    snrs = []
    for dm in trial_dms:
        _, profile = dedisperse_waterfall(waterfall, dm, freqs_hz=freqs_hz, sample_rate_hz=sample_rate_hz)
        snr, _, _, _ = compute_profile_snr(profile)
        snrs.append(snr)

    snrs = np.array(snrs, dtype=np.float32)
    best_idx = int(np.argmax(snrs))
    best_dm = float(trial_dms[best_idx])
    recovered_snr = float(snrs[best_idx])

    return {
        "trial_dms": trial_dms,
        "snrs": snrs,
        "best_dm": best_dm,
        "recovered_snr": recovered_snr,
        "injected_dm": dm_injected,
        "dm_error": abs(best_dm - dm_injected),
    }
