"""Digital Signal Injector for Beam Trackers.

Modeled after CHIME/FRB's software injection framework (mimic, Merryfield et al.).
Generates per-antenna packed int4 complex voltage streams with cold-plasma dispersion,
exponential scattering tails, spectro-temporal modulation, array phase delays,
and calibrated background thermal noise.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from .chime_catalog import FRBParameters

# Dispersion constant K_DM in s * MHz^2 / (pc * cm^-3)
K_DM = 4.148808e3
SPEED_OF_LIGHT = 2.99792458e8  # m/s


def default_antenna_positions(n_ant: int, spacing_m: float = 0.6) -> np.ndarray:
    """Generate (n_ant, 3) antenna coordinates in meters for a regular planar grid."""
    if n_ant not in (32, 64):
        grid_size = int(math.sqrt(n_ant))
        nx = grid_size
        ny = n_ant // grid_size
    elif n_ant == 32:
        nx, ny = 8, 4
    else:  # 64
        nx, ny = 8, 8

    positions = []
    for y in range(ny):
        for x in range(nx):
            if len(positions) < n_ant:
                positions.append([(x - (nx - 1) / 2.0) * spacing_m,
                                  (y - (ny - 1) / 2.0) * spacing_m,
                                  0.0])
    return np.array(positions, dtype=np.float32)


def default_frequencies_hz(n_freq: int = 336, f_min_mhz: float = 400.0, f_max_mhz: float = 800.0) -> np.ndarray:
    """Generate array of channel center frequencies in Hz across 400-800 MHz CHIME band."""
    f_mhz = np.linspace(f_min_mhz, f_max_mhz, n_freq, dtype=np.float32)
    return f_mhz * 1e6


def compute_dispersion_delays(dm: float, freqs_hz: np.ndarray, f_ref_hz: float = 800e6) -> np.ndarray:
    """Compute dispersion delay Delta t(f) in seconds relative to reference frequency."""
    freqs_mhz = freqs_hz / 1e6
    f_ref_mhz = f_ref_hz / 1e6
    delays_s = K_DM * dm * ((freqs_mhz ** -2.0) - (f_ref_mhz ** -2.0))
    return delays_s.astype(np.float32)


def synthesize_frb_intensity_waterfall(
    params: FRBParameters,
    n_time: int,
    freqs_hz: np.ndarray,
    sample_rate_hz: float = 952.381,  # ~1.05 ms sample cadence
    f_ref_hz: float = 800e6,
) -> np.ndarray:
    """Synthesize 2D spectro-temporal intensity waterfall I(t, f) for an FRB burst.

    Returns array of shape (n_time, n_freq).
    """
    dt = 1.0 / sample_rate_hz
    t_axis = np.arange(n_time, dtype=np.float32) * dt
    n_freq = len(freqs_hz)
    waterfall = np.zeros((n_time, n_freq), dtype=np.float32)

    dispersion_delays = compute_dispersion_delays(params.dm, freqs_hz, f_ref_hz=f_ref_hz)
    freqs_mhz = freqs_hz / 1e6
    f_ref_mhz = f_ref_hz / 1e6

    for f_idx in range(n_freq):
        f_mhz = freqs_mhz[f_idx]
        t_center = params.arrival_time_s + dispersion_delays[f_idx]

        # Intrinsic Gaussian envelope
        sigma = params.width_s
        gaussian = np.exp(-0.5 * ((t_axis - t_center) / sigma) ** 2)

        # Exponential scattering tail: tau(f) = tau_400 * (f / 400 MHz)^-4
        tau = params.scattering_tau_s * ((f_mhz / 400.0) ** -4.0)
        if tau > 1e-6:
            # Discrete exponential kernel
            kernel_len = min(n_time, int(8 * tau / dt) + 1)
            k_t = np.arange(kernel_len, dtype=np.float32) * dt
            exp_kernel = np.exp(-k_t / tau)
            exp_kernel /= exp_kernel.sum()
            profile = np.convolve(gaussian, exp_kernel, mode="same")
        else:
            profile = gaussian

        # Spectral index / running modulation: A(f) = (f / f_ref)^(gamma + r * ln(f / f_ref))
        log_f_ratio = np.log(f_mhz / f_ref_mhz)
        spectral_factor = (f_mhz / f_ref_mhz) ** (params.spectral_index + params.spectral_running * log_f_ratio)

        waterfall[:, f_idx] = profile * spectral_factor

    return waterfall


def pack_complex_int4(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """Pack real and imaginary arrays of signed 4-bit integers into int4 byte array.

    Low 4 bits = real, high 4 bits = imaginary.
    """
    real_clamped = np.clip(np.round(real), -7, 7).astype(np.int8)
    imag_clamped = np.clip(np.round(imag), -7, 7).astype(np.int8)

    real_uint4 = (real_clamped & 0x0F).view(np.uint8)
    imag_uint4 = ((imag_clamped & 0x0F) << 4).view(np.uint8)

    packed = real_uint4 | imag_uint4
    return packed


def generate_frb_packed_voltage_stream(
    params: FRBParameters,
    n_time: int = 15360,
    n_ant: int = 64,
    n_freq: int = 336,
    source_dir_lm: Tuple[float, float] = (0.0, 0.0),
    steer_dir_lm: Tuple[float, float] = (0.0, 0.0),
    seed: int = 42,
    ref_n_ant: int = 64,
    waterfall: np.ndarray | None = None,
) -> Tuple[bytes, np.ndarray]:
    """Generate per-antenna complex int4 packed voltage binary stream.

    Returns (packed_bytes, reference_waterfall).
    Layout: [n_time, n_freq, n_ant] byte array.
    """
    rng = np.random.default_rng(seed)
    ant_pos = default_antenna_positions(n_ant)
    freqs_hz = default_frequencies_hz(n_freq)

    # Calculate spatial beam factor / off-boresight angle
    sl, sm = source_dir_lm
    tl, tm = steer_dir_lm
    sn = math.sqrt(max(0.0, 1.0 - sl*sl - sm*sm))
    source_vec = np.array([sl, sm, sn], dtype=np.float32)

    # Phase delays per antenna: phi_n(f) = 2*pi * f/c * (pos_n . source_vec)
    # pos_n shape: (n_ant, 3), source_vec shape: (3,) -> (n_ant,)
    pos_dot_s = ant_pos @ source_vec  # (n_ant,)
    # Phase shape: (n_freq, n_ant)
    phases = (2.0 * np.pi / SPEED_OF_LIGHT) * np.outer(freqs_hz, pos_dot_s)

    # Dynamic spectrum pulse intensity (n_time, n_freq)
    if waterfall is None:
        waterfall = synthesize_frb_intensity_waterfall(params, n_time, freqs_hz)

    # Scale signal amplitude so beamformed coherent sum achieves target SNR for reference array
    sigma_noise = 1.5
    peak_intensity = np.max(waterfall) if np.max(waterfall) > 0 else 1.0
    amp_scale = np.float32((params.target_snr * sigma_noise) / (math.sqrt(ref_n_ant) * math.sqrt(peak_intensity)))

    # Baseband complex signal per (time, freq, ant)
    # Signal voltage = amp_scale * sqrt(I(t, f)) * exp(i * phase)
    signal_volts = (np.sqrt(waterfall[:, :, np.newaxis]) * amp_scale).astype(np.float32)
    complex_phases = np.exp(1j * phases, dtype=np.complex64)[np.newaxis, :, :]

    signal_complex = signal_volts * complex_phases                     # (n_time, n_freq, n_ant)

    # Additive background Gaussian noise per antenna in float32
    noise_r = rng.standard_normal(size=(n_time, n_freq, n_ant), dtype=np.float32)
    noise_r *= sigma_noise
    noise_r += signal_complex.real
    real_clamped = np.clip(np.round(noise_r), -7, 7).astype(np.int8)
    del noise_r

    noise_i = rng.standard_normal(size=(n_time, n_freq, n_ant), dtype=np.float32)
    noise_i *= sigma_noise
    noise_i += signal_complex.imag
    imag_clamped = np.clip(np.round(noise_i), -7, 7).astype(np.int8)
    del noise_i

    # Quantize to int4 packed bytes directly
    real_uint4 = (real_clamped & 0x0F).view(np.uint8)
    imag_uint4 = ((imag_clamped & 0x0F) << 4).view(np.uint8)
    packed_array = real_uint4 | imag_uint4

    return packed_array.tobytes(), waterfall
