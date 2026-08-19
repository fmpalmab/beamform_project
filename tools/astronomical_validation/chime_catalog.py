"""CHIME/FRB Catalog 2 physical parameter definitions and curated benchmark sample.

Ref: arXiv:2601.09399 (The Second CHIME/FRB Catalog)
Model: fitburst spectro-temporal functional form (github.com/CHIMEFRB/fitburst)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FRBParameters:
    name: str
    dm: float                  # Dispersion Measure [pc cm^-3]
    arrival_time_s: float      # Arrival time t0 [seconds]
    width_s: float             # Intrinsic Gaussian width sigma_t [seconds]
    scattering_tau_s: float    # Scattering time tau at 400 MHz [seconds]
    spectral_index: float      # Spectral index gamma
    spectral_running: float    # Spectral running r
    target_snr: float          # Desired post-beamforming S/N ratio


# Curated FRB parameter benchmarks representing diverse astrophysical profiles
CHIME_CATALOG2_BENCHMARKS: Dict[str, FRBParameters] = {
    "FRB20180916B_canonical": FRBParameters(
        name="FRB20180916B_canonical",
        dm=348.82,
        arrival_time_s=8.064,
        width_s=0.0015,
        scattering_tau_s=0.0020,
        spectral_index=-1.2,
        spectral_running=-0.4,
        target_snr=25.0,
    ),
    "High_DM_Burst": FRBParameters(
        name="High_DM_Burst",
        dm=1205.40,
        arrival_time_s=8.064,
        width_s=0.0020,
        scattering_tau_s=0.0040,
        spectral_index=-2.0,
        spectral_running=0.0,
        target_snr=20.0,
    ),
    "Scattering_Dominated": FRBParameters(
        name="Scattering_Dominated",
        dm=574.10,
        arrival_time_s=8.064,
        width_s=0.0010,
        scattering_tau_s=0.0120,
        spectral_index=-1.5,
        spectral_running=-0.8,
        target_snr=18.0,
    ),
    "Faint_Narrow_Pulse": FRBParameters(
        name="Faint_Narrow_Pulse",
        dm=412.30,
        arrival_time_s=8.064,
        width_s=0.0008,
        scattering_tau_s=0.0005,
        spectral_index=0.5,
        spectral_running=-1.0,
        target_snr=9.5,
    ),
}


def get_frb_benchmark(name: str) -> FRBParameters:
    if name not in CHIME_CATALOG2_BENCHMARKS:
        raise KeyError(f"Unknown benchmark FRB '{name}'. Available: {list(CHIME_CATALOG2_BENCHMARKS.keys())}")
    return CHIME_CATALOG2_BENCHMARKS[name]
