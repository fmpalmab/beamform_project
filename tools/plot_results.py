#!/usr/bin/env python3
"""Plot and compare headerless [time, frequency, beam] intensity products."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


import sys
_tools_dir = Path(__file__).resolve().parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from constants import (
    SPEED_OF_LIGHT_M_PER_S,
    LOCAL_FREQUENCY_CHANNELS,
    FULL_BAND_FREQUENCY_CHANNELS,
    DEFAULT_N_FREQ,
    DEFAULT_FREQUENCY_START_HZ,
    DEFAULT_CHANNEL_WIDTH_HZ,
    BEAM_GRID_DESIGN_FREQUENCY_HZ,
    DEFAULT_SPACING_M,
    ETA_HEX,
    ANTENNA_SPECS,
)

def resolve_frequency_configuration(
    buffer: str | None, requested_n_freq: int | None,
) -> tuple[str, int]:
    """Resolve a local buffer or full-band plotting configuration.

    ``buffer=None`` keeps compatibility with the old ``--n-freq`` interface:
    336 selects buffer 0 and 672 selects the full band. The explicit buffer
    option is preferred because it also documents which local shard is being
    plotted.
    """
    if buffer is None:
        if requested_n_freq in (None, LOCAL_FREQUENCY_CHANNELS):
            return "0", LOCAL_FREQUENCY_CHANNELS
        if requested_n_freq == FULL_BAND_FREQUENCY_CHANNELS:
            return "both", FULL_BAND_FREQUENCY_CHANNELS
        raise ValueError("n_freq must be 336 or 672")

    expected = (FULL_BAND_FREQUENCY_CHANNELS if buffer == "both"
                else LOCAL_FREQUENCY_CHANNELS)
    if requested_n_freq is not None and requested_n_freq != expected:
        raise ValueError(
            f"buffer={buffer} requires n_freq={expected}, got {requested_n_freq}"
        )
    return buffer, expected


def buffer_label(buffer: str, n_freq: int) -> str:
    if buffer == "both":
        return f"full band ({n_freq} channels)"
    return f"buffer {buffer} ({n_freq} local channels)"


def combine_intensity_shards(shard0: np.ndarray, shard1: np.ndarray) -> np.ndarray:
    """Combine two local output shards for plotting along frequency only."""
    if shard0.ndim != 3 or shard1.ndim != 3:
        raise ValueError("shard intensity arrays must have shape [T][F][B]")
    if shard0.shape[0] != shard1.shape[0] or shard0.shape[2] != shard1.shape[2]:
        raise ValueError("shard intensity dimensions differ in time or beams")
    if shard0.shape[1] != LOCAL_FREQUENCY_CHANNELS \
            or shard1.shape[1] != LOCAL_FREQUENCY_CHANNELS:
        raise ValueError("both shard intensity arrays must contain 336 channels")
    return np.concatenate((shard0, shard1), axis=1)


def _text_rows(path: Path, columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        data = raw_line.split("#", maxsplit=1)[0].replace(",", " ").split()
        if not data:
            continue
        if len(data) != columns:
            raise ValueError(f"{path}:{line_number}: expected {columns} values")
        rows.append([float(value) for value in data])
    return np.asarray(rows, dtype=np.float64)


def default_positions(n_ant: int, spacing_m: float = DEFAULT_SPACING_M) -> np.ndarray:
    if n_ant == 32:
        rows, columns = 4, 8
    elif n_ant == 64:
        rows, columns = 8, 8
    else:
        raise ValueError("n_ant must be 32 or 64")
    return np.asarray(
        [(column * spacing_m, row * spacing_m, 0.0)
         for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )


def default_beam_directions(n_beams: int, l_step: float = 0.02,
                            m: float = 0.0) -> np.ndarray:
    if not 1 <= n_beams <= 128:
        raise ValueError("n_beams must be between 1 and 128")
    center = n_beams // 2
    directions = []
    for beam in range(n_beams):
        l_value = (beam - center) * l_step
        transverse_squared = l_value * l_value + m * m
        if transverse_squared > 1.0:
            raise ValueError("beam directions must satisfy l*l + m*m <= 1")
        directions.append((l_value, m, math.sqrt(1.0 - transverse_squared)))
    return np.asarray(directions, dtype=np.float64)


def default_frequencies(n_freq: int = DEFAULT_N_FREQ,
                        start_hz: float = DEFAULT_FREQUENCY_START_HZ,
                        channel_width_hz: float = DEFAULT_CHANNEL_WIDTH_HZ) -> np.ndarray:
    if n_freq <= 0 or start_hz <= 0.0 or channel_width_hz <= 0.0:
        raise ValueError("frequency count, start, and channel width must be positive")
    return start_hz + np.arange(n_freq, dtype=np.float64) * channel_width_hz


def array_shape(n_ant: int) -> tuple[int, int]:
    if n_ant == 32:
        return 4, 8
    if n_ant == 64:
        return 8, 8
    raise ValueError("n_ant must be 32 or 64")


def centered_integer_range(n: int) -> np.ndarray:
    """Centered integer indices of length n (e.g. n=4 -> [-2, -1, 0, 1])."""
    if n <= 0:
        raise ValueError("centered integer range length must be positive")
    start = -int(np.floor(n / 2))
    return np.arange(start, start + n)


def select_fft_beam_centers_rectangular(
    du_fft_u: float,
    dv_fft_v: float,
    n_beams: int,
    m_side: int,
    n_side: int,
) -> tuple[list[tuple[float, float]], int, int, int, int]:
    """Select a centered block from the (2M)x(2N) zero-padded FFT bin bank."""
    if du_fft_u <= 0.0 or dv_fft_v <= 0.0 or not np.isfinite(du_fft_u + dv_fft_v):
        raise ValueError("FFT beam spacing must be positive and finite")
    if n_beams <= 0 or m_side <= 0 or n_side <= 0:
        raise ValueError("beam count and array sides must be positive")

    n_bank_u = 2 * m_side
    n_bank_v = 2 * n_side
    n_bank_total = n_bank_u * n_bank_v
    if n_beams > n_bank_total:
        raise ValueError(
            f"n-beams must be <= (2M)*(2N) = {n_bank_total} "
            f"for M={m_side}, N={n_side}"
        )

    n_u = int(np.ceil(np.sqrt(n_beams)))
    n_v = int(np.ceil(n_beams / n_u))
    if n_u > n_bank_u:
        n_u = n_bank_u
        n_v = int(np.ceil(n_beams / n_u))
    if n_v > n_bank_v:
        n_v = n_bank_v
        # Recompute the other dimension after clamping. This is required for
        # A=32, B=128, whose complete bank is 16x8.
        n_u = int(np.ceil(n_beams / n_v))
    if n_u > n_bank_u or n_u * n_v < n_beams:
        raise RuntimeError("failed to select a sufficient FFT beam window")

    iu = centered_integer_range(n_u)
    iv = centered_integer_range(n_v)
    centers_ranked: list[tuple[float, float, float]] = []
    for i in iu:
        for j in iv:
            u0 = float(i * du_fft_u)
            v0 = float(j * dv_fft_v)
            centers_ranked.append((u0 * u0 + v0 * v0, u0, v0))
    centers_ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    centers = [(u0, v0) for _, u0, v0 in centers_ranked[:n_beams]]
    return centers, n_u, n_v, n_bank_u, n_bank_v


def fft_beam_directions(
    n_ant: int,
    n_beams: int,
    spacing_m: float = DEFAULT_SPACING_M,
    design_frequency_hz: float = BEAM_GRID_DESIGN_FREQUENCY_HZ,
) -> np.ndarray:
    """Directions selected from FFT-bin geometry for the direct beamformer."""
    rows, columns = array_shape(n_ant)
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / design_frequency_hz
    du_fft_u = wavelength_m / (2 * columns * spacing_m)
    dv_fft_v = wavelength_m / (2 * rows * spacing_m)
    centers, _, _, _, _ = select_fft_beam_centers_rectangular(
        du_fft_u, dv_fft_v, n_beams, columns, rows)
    directions = []
    for u_value, v_value in centers:
        transverse_squared = u_value * u_value + v_value * v_value
        if transverse_squared > 1.0:
            raise ValueError("the selected FFT beam grid extends outside the visible sky")
        directions.append(
            (u_value, v_value, math.sqrt(1.0 - transverse_squared)))
    return np.asarray(directions, dtype=np.float64)


def fov_axis(n_points: int, u_fov: float) -> np.ndarray:
    if n_points <= 1:
        return np.array([0.0])
    return np.linspace(-u_fov, u_fov, n_points)


def estimate_nbeams_hex_formula(
    d_m: float,
    m_side: int,
    n_side: int,
    bw_e_deg: float,
    bw_h_deg: float,
    wavelength_m: float,
) -> int:
    """Estimate hex beam count over the requested E/H field of view."""
    d_u = d_m * (m_side - 1)
    d_v = d_m * (n_side - 1)
    bw_e_rad = np.radians(bw_e_deg)
    bw_h_rad = np.radians(bw_h_deg)
    n_est = (
        ETA_HEX
        * 4.0
        * d_u
        * d_v
        * np.sin(bw_e_rad / 2.0)
        * np.sin(bw_h_rad / 2.0)
        * (wavelength_m ** -2)
    )
    return max(1, int(np.ceil(n_est)))


def generate_hex_targets_cropped_fov(
    n_beams: int,
    u_max: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a hexagonal target lattice cropped by an FoV ellipse."""
    if n_beams <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    s_unit = np.sqrt((2.0 * np.pi) / (np.sqrt(3.0) * n_beams))
    points_u: list[float] = []
    points_v: list[float] = []
    for _ in range(10):
        points_u.clear()
        points_v.clear()
        dy = np.sqrt(3.0) * 0.5 * s_unit
        j_max = int(np.ceil(1.0 / max(dy, 1e-12))) + 2
        i_max = int(np.ceil(1.0 / max(s_unit, 1e-12))) + 2
        for j in range(-j_max, j_max + 1):
            y = j * dy
            x_shift = 0.5 * s_unit if (j % 2 != 0) else 0.0
            for i in range(-i_max, i_max + 1):
                x = i * s_unit + x_shift
                if x * x + y * y <= 1.0:
                    points_u.append(float(x * u_max))
                    points_v.append(float(y * v_max))
        if len(points_u) >= n_beams:
            break
        s_unit *= 0.9

    if not points_u:
        return np.array([0.0]), np.array([0.0])
    uu = np.array(points_u, dtype=float)
    vv = np.array(points_v, dtype=float)
    order = np.argsort(uu**2 + vv**2)[:n_beams]
    return uu[order], vv[order]


def rectangular_beam_directions(
    n_ant: int,
    spacing_m: float = DEFAULT_SPACING_M,
    design_frequency_hz: float = BEAM_GRID_DESIGN_FREQUENCY_HZ,
) -> np.ndarray:
    rows, columns = array_shape(n_ant)
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / design_frequency_hz
    delta_l = wavelength_m / (columns * spacing_m)
    delta_m = wavelength_m / (rows * spacing_m)
    l_centers = (np.arange(columns) - (columns - 1) / 2.0) * delta_l
    m_centers = (np.arange(rows) - (rows - 1) / 2.0) * delta_m
    directions = []
    for m_value in m_centers:
        for l_value in l_centers:
            transverse_squared = l_value * l_value + m_value * m_value
            if transverse_squared > 1.0:
                raise ValueError("the confirmed beam grid extends outside the visible sky")
            directions.append((l_value, m_value, math.sqrt(1.0 - transverse_squared)))
    return np.asarray(directions, dtype=np.float64)


def _linear_interpolate_extrapolate(frequency_hz: np.ndarray | float,
                                    values: Sequence[float]) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    anchors = np.asarray(sorted(ANTENNA_SPECS), dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    result = np.interp(frequency, anchors, samples)
    below = frequency < anchors[0]
    above = frequency > anchors[-1]
    result = np.asarray(result)
    result = np.where(
        below,
        samples[0] + (frequency - anchors[0])
        * (samples[1] - samples[0]) / (anchors[1] - anchors[0]),
        result,
    )
    result = np.where(
        above,
        samples[-1] + (frequency - anchors[-1])
        * (samples[-1] - samples[-2]) / (anchors[-1] - anchors[-2]),
        result,
    )
    return result


def interpolated_antenna_specs(
    frequency_hz: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchors = sorted(ANTENNA_SPECS)
    bw_e = _linear_interpolate_extrapolate(
        frequency_hz, [ANTENNA_SPECS[value]["BW_E"] for value in anchors])
    bw_h = _linear_interpolate_extrapolate(
        frequency_hz, [ANTENNA_SPECS[value]["BW_H"] for value in anchors])
    gain_dbi = _linear_interpolate_extrapolate(
        frequency_hz, [ANTENNA_SPECS[value]["gain_dBi"] for value in anchors])
    return bw_e, bw_h, gain_dbi


def element_factor_2d(l: np.ndarray, m: np.ndarray,
                      frequency_hz: float) -> np.ndarray:
    bw_e_deg, bw_h_deg, gain_dbi = interpolated_antenna_specs(frequency_hz)
    sigma_e = np.radians(float(bw_e_deg)) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_h = np.radians(float(bw_h_deg)) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gain_linear = 10.0 ** (float(gain_dbi) / 10.0)
    return gain_linear * np.exp(
        -((l * l) / (2.0 * sigma_h * sigma_h)
          + (m * m) / (2.0 * sigma_e * sigma_e))
    )


def _ula_power(coordinates: np.ndarray, steering: float, elements: int,
               spacing_m: float, frequency_hz: float) -> np.ndarray:
    phase = (2.0 * np.pi * spacing_m * frequency_hz / SPEED_OF_LIGHT_M_PER_S
             * (coordinates - steering))
    element_index = np.arange(elements, dtype=np.float64)
    voltage = np.exp(-1j * phase[:, np.newaxis] * element_index).mean(axis=1)
    return np.abs(voltage) ** 2


def beam_power_cube(l_axis: np.ndarray, m_axis: np.ndarray,
                    directions: np.ndarray, n_ant: int, spacing_m: float,
                    frequency_hz: float) -> np.ndarray:
    rows, columns = array_shape(n_ant)
    l_grid, m_grid = np.meshgrid(l_axis, m_axis)
    element_power = element_factor_2d(l_grid, m_grid, frequency_hz)
    power = np.empty((len(directions), len(m_axis), len(l_axis)), dtype=np.float64)
    for beam, direction in enumerate(directions):
        power_l = _ula_power(l_axis, direction[0], columns, spacing_m, frequency_hz)
        power_m = _ula_power(m_axis, direction[1], rows, spacing_m, frequency_hz)
        # Uniform weights with fixed total array power give N times the element gain on axis.
        power[beam] = n_ant * element_power * power_m[:, np.newaxis] * power_l[np.newaxis, :]
    visible = l_grid * l_grid + m_grid * m_grid <= 1.0
    power[:, ~visible] = np.nan
    return power


def load_intensity(path: Path, n_time: int, n_freq: int,
                   n_beams: int) -> np.ndarray:
    expected_values = n_time * n_freq * n_beams
    expected_bytes = expected_values * np.dtype("<f4").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} "
            f"for [{n_time}][{n_freq}][{n_beams}]"
        )
    return np.fromfile(path, dtype="<f4").reshape(n_time, n_freq, n_beams)


def load_selected_intensity(primary: Path, secondary: Path | None,
                            n_time: int, n_freq: int, n_beams: int,
                            buffer: str) -> np.ndarray:
    """Load one buffer, a precombined band, or two local output buffers."""
    if buffer == "both" and secondary is not None:
        shard0 = load_intensity(primary, n_time, LOCAL_FREQUENCY_CHANNELS, n_beams)
        shard1 = load_intensity(secondary, n_time, LOCAL_FREQUENCY_CHANNELS, n_beams)
        return combine_intensity_shards(shard0, shard1)
    return load_intensity(primary, n_time, n_freq, n_beams)


def baseline_uv(positions_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / frequency_hz
    baselines = []
    for first in range(len(positions_m)):
        for second in range(first + 1, len(positions_m)):
            delta = (positions_m[second] - positions_m[first]) / wavelength_m
            baselines.append((delta[0], delta[1]))
            baselines.append((-delta[0], -delta[1]))
    return np.asarray(baselines, dtype=np.float64)


def normalized_db(values: np.ndarray, reference: float | None = None,
                  floor_db: float = -80.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if reference is None:
        reference = float(np.max(values)) if values.size else 0.0
    if reference <= 0.0:
        return np.full_like(values, floor_db)
    ratio = np.maximum(values / reference, 10.0 ** (floor_db / 10.0))
    return 10.0 * np.log10(ratio)


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray,
                       abs_tolerance: float, rel_tolerance: float) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes do not match")
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    absolute_error = np.abs(candidate64 - reference64)
    valid_relative = np.abs(reference64) > abs_tolerance
    relative_error = np.zeros_like(absolute_error)
    np.divide(absolute_error, np.abs(reference64), out=relative_error,
              where=valid_relative)
    relative_values = relative_error[valid_relative]
    flat_reference = reference64.ravel()
    flat_candidate = candidate64.ravel()
    if np.std(flat_reference) == 0.0 or np.std(flat_candidate) == 0.0:
        correlation = 1.0 if np.array_equal(flat_reference, flat_candidate) else math.nan
    else:
        correlation = float(np.corrcoef(flat_reference, flat_candidate)[0, 1])
    outside = (absolute_error > abs_tolerance) & (
        (~valid_relative) | (relative_error > rel_tolerance)
    )
    return {
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_relative_error": float(np.mean(relative_values)) if relative_values.size else 0.0,
        "p99_relative_error": float(np.percentile(relative_values, 99.0))
        if relative_values.size else 0.0,
        "max_relative_error": float(np.max(relative_values)) if relative_values.size else 0.0,
        "correlation": correlation,
        "outside_tolerance": int(np.count_nonzero(outside)),
        "total_values": int(reference.size),
    }


def synthetic_description(args: argparse.Namespace) -> str:
    if args.synthetic_type == "point-source":
        return (f"Synthetic point source: l={args.source_l:g}, m={args.source_m:g}, "
                f"amplitude={args.amplitude:g} int4-quantized")
    if args.synthetic_type == "one-hot":
        return ("Synthetic one-hot: "
                f"t={args.active_time}, f={args.active_frequency}, "
                f"element={args.active_element}")
    if args.synthetic_type == "constant":
        return (f"Synthetic constant: value={args.value_real:g}"
                f"{args.value_imag:+g}j")
    if args.synthetic_type == "noise":
        return f"Synthetic independent complex int4 noise: seed={args.seed}"
    return "Input data: user-provided/real (no synthetic source marker)"


def _load_geometry(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = (_text_rows(args.positions, 3) if args.positions
                 else default_positions(args.n_ant, args.spacing_m))
    if positions.shape != (args.n_ant, 3):
        raise ValueError(f"positions must contain exactly {args.n_ant} x,y,z rows")

    if args.frequencies:
        frequencies = _text_rows(args.frequencies, 1).reshape(-1)
    elif args.frequency_hz is not None:
        frequencies = np.full(args.n_freq, args.frequency_hz, dtype=np.float64)
    else:
        frequency_start_hz = args.frequency_start_hz
        if args.buffer == "1":
            frequency_start_hz += LOCAL_FREQUENCY_CHANNELS * args.channel_width_hz
        frequencies = default_frequencies(
            args.n_freq, frequency_start_hz, args.channel_width_hz)
    if frequencies.shape != (args.n_freq,) or np.any(frequencies <= 0.0):
        raise ValueError(f"frequencies must contain exactly {args.n_freq} positive values")

    directions = (_text_rows(args.directions, 3) if args.directions
                  else fft_beam_directions(
                      args.n_ant, args.n_beams, args.spacing_m,
                      args.design_frequency_hz)
                  if args.beam_grid == "fft"
                  else rectangular_beam_directions(
                      args.n_beams, args.spacing_m,
                      args.design_frequency_hz)
                  if args.beam_grid == "legacy-rectangular"
                  else default_beam_directions(
                      args.n_beams, args.beam_l_step, args.beam_m))
    if directions.shape != (args.n_beams, 3):
        raise ValueError(f"directions must contain exactly {args.n_beams} x,y,z rows")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, atol=1.0e-3):
        raise ValueError("beam directions must be unit vectors")
    return positions, frequencies, directions


def _output_path(input_path: Path, requested: Path | None, suffix: str) -> Path:
    return requested if requested else input_path.with_name(f"{input_path.stem}{suffix}.png")


def _sky_axes(ax: object, title: str) -> None:
    import matplotlib.pyplot as plt

    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="black", linewidth=1.0))
    ax.set(title=title, xlabel="l (H / x direction)", ylabel="m (E / y direction)",
           xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), aspect="equal")


def _beam_contours(ax: object, l_axis: np.ndarray, m_axis: np.ndarray,
                   power: np.ndarray, directions: np.ndarray,
                   threshold_linear: float) -> None:
    colors = __import__("matplotlib").colormaps["turbo"](
        np.linspace(0.0, 1.0, len(directions)))
    for beam, color in enumerate(colors):
        peak = np.nanmax(power[beam])
        if peak > 0.0:
            ax.contour(l_axis, m_axis, power[beam] / peak,
                       levels=[threshold_linear], colors=[color], linewidths=0.65)


def _plot_beam_directions_and_power(
    fig: object,
    ax: object,
    args: argparse.Namespace,
    directions: np.ndarray,
    normalized_power: np.ndarray,
) -> None:
    """Plot beam centers over a local l/m response map."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    centers = directions[:, :2]
    source = np.asarray([args.source_l, args.source_m], dtype=np.float64)
    # Keep the complete visible-sky context in this plot.
    l_axis = np.linspace(-1.0, 1.0, 241)
    m_axis = np.linspace(-1.0, 1.0, 241)

    response = beam_power_cube(
        l_axis, m_axis, directions, args.n_ant, args.spacing_m,
        args.design_frequency_hz)
    beam_peaks = np.nanmax(response, axis=(1, 2), keepdims=True)
    response = response / np.maximum(beam_peaks, np.finfo(float).tiny)
    finite_response = np.nan_to_num(response, nan=0.0)
    maximum_response = np.max(finite_response, axis=0)
    response_db = np.full_like(maximum_response, -24.0)
    valid = np.any(np.isfinite(response), axis=0)
    response_db[valid] = 10.0 * np.log10(
        np.maximum(maximum_response[valid], 10.0 ** (-24.0 / 10.0)))
    response_image = ax.contourf(
        l_axis, m_axis, response_db,
        levels=np.linspace(-24.0, 0.0, 13), cmap="inferno",
        vmin=-24.0, vmax=0.0, alpha=0.88)

    contour_colors = plt.colormaps["viridis"](
        np.linspace(0.0, 1.0, len(directions)))
    for beam, color in enumerate(contour_colors):
        ax.contour(
            l_axis, m_axis, response[beam],
            levels=[10.0 ** (-3.0 / 10.0)], colors=[color],
            linewidths=0.75, alpha=0.9)

    bw_e, bw_h, _ = interpolated_antenna_specs(args.design_frequency_hz)
    fov_u = min(1.0, np.sin(np.radians(float(bw_h) / 2.0)))
    fov_v = min(1.0, np.sin(np.radians(float(bw_e) / 2.0)))
    ax.add_patch(Ellipse(
        (0.0, 0.0), width=2.0 * fov_u, height=2.0 * fov_v,
        fill=False, edgecolor="white", linestyle="--", linewidth=1.0,
        label="antenna 3 dB FoV", zorder=5))
    ax.add_patch(plt.Circle(
        (0.0, 0.0), 1.0, fill=False, color="white", linestyle=":",
        linewidth=0.9, label="visible sky", zorder=5))

    points = ax.scatter(
        centers[:, 0], centers[:, 1], c=normalized_power,
        s=45.0 + 240.0 * normalized_power, cmap="viridis", vmin=0.0,
        vmax=1.0, edgecolor="black", linewidth=0.5, zorder=7)
    label_count = min(len(centers), 12)
    label_beams = np.argsort(normalized_power)[-label_count:]
    for beam in label_beams:
        l_value, m_value = centers[beam]
        ax.annotate(
            f"B{beam}", (l_value, m_value), xytext=(3, 3),
            textcoords="offset points", fontsize=6, zorder=8)
    if args.synthetic_type == "point-source":
        ax.scatter(
            [args.source_l], [args.source_m], marker="*", s=220,
            color="red", edgecolor="black", label="injected source", zorder=9)

    fig.colorbar(response_image, ax=ax, fraction=0.046, pad=0.03,
                 label="maximum beam response [dB]")
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.10,
                 label="recovered power / maximum")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set(
        title=("Beam directions and recovered power\n"
               f"response contours at {args.design_frequency_hz / 1e6:g} MHz"),
        xlabel="l (H / x direction)", ylabel="m (E / y direction)")
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(fontsize=7, loc="best")


def plot_sky_coverage(args: argparse.Namespace, frequencies: np.ndarray,
                      directions: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    output = args.sky_output
    output.parent.mkdir(parents=True, exist_ok=True)
    l_axis = np.linspace(-1.0, 1.0, args.sky_resolution)
    m_axis = np.linspace(-1.0, 1.0, args.sky_resolution)
    l_grid, m_grid = np.meshgrid(l_axis, m_axis)
    visible = l_grid * l_grid + m_grid * m_grid <= 1.0
    threshold_linear = 10.0 ** (args.overlap_db / 10.0)

    display_frequencies = tuple(float(value) for value in np.linspace(
        frequencies[0], frequencies[-1], 3))
    display_power = {
        frequency: beam_power_cube(
            l_axis, m_axis, directions, args.n_ant, args.spacing_m, frequency)
        for frequency in display_frequencies
    }

    band_average = np.zeros_like(display_power[display_frequencies[1]])
    for frequency in frequencies:
        band_average += np.nan_to_num(
            beam_power_cube(l_axis, m_axis, directions, args.n_ant,
                            args.spacing_m, float(frequency)),
            nan=0.0,
        )
    band_average /= len(frequencies)
    band_average[:, ~visible] = np.nan

    reference_power = display_power[display_frequencies[1]]
    normalized_reference = reference_power / np.nanmax(
        reference_power, axis=(1, 2), keepdims=True)
    overlap_count = np.sum(normalized_reference >= threshold_linear, axis=0).astype(float)
    overlap_count[~visible] = np.nan
    dominant_beam = np.argmax(np.nan_to_num(reference_power, nan=-1.0), axis=0).astype(float)
    dominant_beam[~visible] = np.nan

    rows, columns = array_shape(args.n_ant)
    wavelength_design = SPEED_OF_LIGHT_M_PER_S / args.design_frequency_hz
    delta_l = wavelength_design / (columns * args.spacing_m)
    delta_m = wavelength_design / (rows * args.spacing_m)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle(
        f"Full-sky rectangular beam grid | {rows}x{columns} array, "
        f"{args.n_ant} elements/beams, d={args.spacing_m:g} m\n"
        f"grid designed at {args.design_frequency_hz / 1e6:g} MHz: "
        f"delta_l={delta_l:.4f}, delta_m={delta_m:.4f}; "
        f"element gain and beamwidth linearly interpolated/extrapolated",
        fontsize=13,
    )

    ax = axes[0, 0]
    dominant_image = ax.imshow(
        dominant_beam, origin="lower", extent=(-1, 1, -1, 1),
        cmap="turbo", interpolation="nearest", vmin=0, vmax=args.n_beams - 1,
    )
    ax.scatter(directions[:, 0], directions[:, 1], s=12, color="black")
    _sky_axes(ax, f"Dominant beam at {display_frequencies[1] / 1e6:.3f} MHz")
    fig.colorbar(dominant_image, ax=ax, label="beam index")

    for ax, frequency in zip(axes.flat[1:4], display_frequencies):
        power = display_power[frequency]
        maximum_power = np.max(np.nan_to_num(power, nan=0.0), axis=0)
        maximum_gain_dbi = np.full_like(maximum_power, np.nan)
        maximum_gain_dbi[visible] = 10.0 * np.log10(maximum_power[visible])
        gain_image = ax.imshow(maximum_gain_dbi, origin="lower", extent=(-1, 1, -1, 1),
                               cmap="viridis", vmin=0.0,
                               vmax=10.0 * np.log10(args.n_ant) + 9.0)
        _beam_contours(ax, l_axis, m_axis, power, directions, threshold_linear)
        ax.scatter(directions[:, 0], directions[:, 1], s=6, color="white", alpha=0.8)
        bw_e, bw_h, gain_dbi = interpolated_antenna_specs(frequency)
        _sky_axes(
            ax,
            f"{frequency / 1e6:.0f} MHz | max absolute gain [dBi]\n"
            f"element: BW_H={float(bw_h):.0f} deg, BW_E={float(bw_e):.0f} deg, "
            f"gain={float(gain_dbi):.2f} dBi",
        )
        fig.colorbar(gain_image, ax=ax, label="absolute array gain [dBi]")

    ax = axes[1, 1]
    band_maximum = np.max(np.nan_to_num(band_average, nan=0.0), axis=0)
    band_gain_dbi = np.full_like(band_maximum, np.nan)
    band_gain_dbi[visible] = 10.0 * np.log10(band_maximum[visible])
    band_image = ax.imshow(band_gain_dbi, origin="lower", extent=(-1, 1, -1, 1),
                           cmap="viridis", vmin=0.0,
                           vmax=10.0 * np.log10(args.n_ant) + 9.0)
    _sky_axes(ax, f"Exact average over {len(frequencies)} channels | {args.buffer_label} | maximum absolute gain")
    fig.colorbar(band_image, ax=ax, label="band-averaged gain [dBi]")

    ax = axes[1, 2]
    overlap_image = ax.imshow(overlap_count, origin="lower", extent=(-1, 1, -1, 1),
                              cmap="magma", interpolation="nearest", vmin=0)
    ax.scatter(directions[:, 0], directions[:, 1], s=8, color="cyan")
    _sky_axes(ax, f"Beam overlap at {display_frequencies[1] / 1e6:.3f} MHz | responses >= {args.overlap_db:g} dB")
    fig.colorbar(overlap_image, ax=ax, label="number of overlapping beams")

    fig.savefig(output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def plot_geometry_and_uv(args: argparse.Namespace,
                         positions: np.ndarray,
                         frequencies: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    output = args.geometry_output
    output.parent.mkdir(parents=True, exist_ok=True)
    uv_channel = args.uv_channel if args.uv_channel is not None else args.n_freq // 2
    if not 0 <= uv_channel < args.n_freq:
        raise ValueError("uv-channel is outside the frequency range")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(
        f"Array configuration | {args.n_ant} elements | d={args.spacing_m:g} m\n"
        f"{args.buffer_label}", fontsize=13)

    first_block = min(32, args.n_ant)
    axes[0].scatter(
        positions[:first_block, 0], positions[:first_block, 1],
        color="royalblue", s=45, edgecolor="black", linewidth=0.4,
        label=f"E[0..{first_block - 1}]",
    )
    if args.n_ant > first_block:
        axes[0].scatter(
            positions[first_block:, 0], positions[first_block:, 1],
            color="crimson", s=45, edgecolor="black", linewidth=0.4,
            label=f"E[{first_block}..{args.n_ant - 1}]",
        )
    axes[0].set(title=f"Array geometry ({args.n_ant} elements)",
                xlabel="x [m]", ylabel="y [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    uv = baseline_uv(positions, float(frequencies[uv_channel]))
    axes[1].scatter(uv[:, 0], uv[:, 1], s=8, alpha=0.55)
    axes[1].scatter([0.0], [0.0], marker="+", color="black",
                    label="autocorrelation")
    axes[1].set(
        title=(f"u-v baseline coverage | channel {uv_channel}, "
               f"{frequencies[uv_channel] / 1e6:.3f} MHz"),
        xlabel="u [wavelengths]", ylabel="v [wavelengths]",
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.savefig(output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def plot_dashboard(args: argparse.Namespace, intensity: np.ndarray,
                   comparison: np.ndarray | None, positions: np.ndarray,
                   frequencies: np.ndarray, directions: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    output = _output_path(args.input, args.output, "_validation")
    output.parent.mkdir(parents=True, exist_ok=True)
    integrated = intensity.sum(axis=(0, 1), dtype=np.float64)
    peak_beam = int(np.argmax(integrated))
    comparison_integrated = (comparison.sum(axis=(0, 1), dtype=np.float64)
                             if comparison is not None else None)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(
        f"{args.label} beamforming validation | layout "
        f"[T={args.n_time}][F={args.n_freq}][B={args.n_beams}]\n"
        f"{args.buffer_label}\n"
        f"{synthetic_description(args)}",
        fontsize=14,
    )

    ax = axes[0, 0]
    normalized_power = integrated / max(float(np.max(integrated)), np.finfo(float).tiny)
    _plot_beam_directions_and_power(
        fig, ax, args, directions, normalized_power)

    ax = axes[0, 1]
    top_count = min(8, args.n_beams)
    top_beams = np.argsort(integrated)[-top_count:][::-1]
    top_labels = [f"B{beam}" for beam in top_beams]
    top_values = integrated[top_beams]
    ax.bar(np.arange(top_count) - 0.2,
           top_values / max(top_values[0], 1.0e-30), width=0.4, label=args.label)
    if comparison_integrated is not None:
        comparison_top = comparison_integrated[top_beams]
        ax.bar(np.arange(top_count) + 0.2,
               comparison_top / max(comparison_top[0], 1.0e-30),
               width=0.4, label=args.compare_label)
    ax.set(title="Top recovered beams", xlabel="beam",
           ylabel="relative integrated power")
    ax.set_xticks(np.arange(top_count), top_labels)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    spectrum = intensity.sum(axis=0, dtype=np.float64)
    comparison_spectrum = (comparison.sum(axis=0, dtype=np.float64)
                           if comparison is not None else None)
    selected = list(top_beams[:min(3, top_count)])
    frequencies_vary = not np.allclose(frequencies, frequencies[0])
    if frequencies_vary:
        spectrum_axis = frequencies / 1e6
        spectrum_xlabel = "frequency [MHz]"
        spectrum_title = "Selected-beam spectra"
    else:
        spectrum_axis = np.arange(args.n_freq)
        spectrum_xlabel = "frequency-channel index"
        spectrum_title = ("Selected-beam spectra | all synthetic channels at "
                          f"{frequencies[0] / 1e6:g} MHz")
    spectrum_reference = max(float(np.max(spectrum[:, selected])), np.finfo(float).tiny)
    for beam in selected:
        ax.plot(spectrum_axis, normalized_db(spectrum[:, beam], spectrum_reference),
                label=f"{args.label} B{beam} (l={directions[beam, 0]:g})")
        if comparison_spectrum is not None:
            ax.plot(spectrum_axis,
                    normalized_db(comparison_spectrum[:, beam], spectrum_reference),
                    linestyle="--", alpha=0.8,
                    label=f"{args.compare_label} B{beam}")
    ax.set(title=spectrum_title, xlabel=spectrum_xlabel,
           ylabel="power relative to displayed maximum [dB]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    time_beam = intensity.sum(axis=1, dtype=np.float64)
    image = ax.imshow(normalized_db(time_beam), origin="lower", aspect="auto",
                      cmap="magma",
                      extent=(-0.5, args.n_beams - 0.5, -0.5, args.n_time - 0.5))
    ax.set(title="Intensity versus time and beam", xlabel="beam index", ylabel="time index")
    if args.n_beams <= 16:
        ax.set_xticks(np.arange(args.n_beams))
    else:
        ax.set_xticks([])
    fig.colorbar(image, ax=ax, label="power relative to maximum [dB]")

    fig.savefig(output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def plot_comparison(args: argparse.Namespace, reference: np.ndarray,
                    candidate: np.ndarray, directions: np.ndarray,
                    metrics: dict[str, float | int]) -> Path:
    import matplotlib.pyplot as plt

    output = _output_path(args.input, args.comparison_output, "_comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    absolute_error = np.abs(candidate64 - reference64)
    valid = np.abs(reference64) > args.abs_tolerance
    relative_error = np.zeros_like(absolute_error)
    np.divide(absolute_error, np.abs(reference64), out=relative_error, where=valid)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(
        f"{args.label} versus {args.compare_label} | "
        f"max abs={metrics['max_absolute_error']:.3e}, "
        f"p99 rel={metrics['p99_relative_error']:.3e}, "
        f"outside={metrics['outside_tolerance']}/{metrics['total_values']}",
        fontsize=13,
    )

    ax = axes[0, 0]
    flat_reference = reference64.ravel()
    flat_candidate = candidate64.ravel()
    stride = max(1, flat_reference.size // 100_000)
    ax.scatter(flat_reference[::stride], flat_candidate[::stride], s=5, alpha=0.25)
    limits = [min(float(np.min(flat_reference)), float(np.min(flat_candidate))),
              max(float(np.max(flat_reference)), float(np.max(flat_candidate)))]
    ax.plot(limits, limits, color="red", linestyle="--", label="identity")
    ax.set(title=f"Intensity agreement | correlation={metrics['correlation']:.9g}",
           xlabel=args.label, ylabel=args.compare_label)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    relative_values = relative_error[valid]
    positive = relative_values[relative_values > 0.0]
    if positive.size:
        ax.hist(np.log10(positive), bins=60, color="tab:orange", alpha=0.8)
        ax.axvline(math.log10(args.rel_tolerance), color="red", linestyle="--",
                   label=f"relative tolerance={args.rel_tolerance:g}")
        ax.set_xlabel("log10(relative error)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No nonzero relative errors", ha="center", va="center",
                transform=ax.transAxes)
    ax.set(title="Relative-error distribution", ylabel="count")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    error_time_beam = absolute_error.max(axis=1)
    image = ax.imshow(error_time_beam, origin="lower", aspect="auto", cmap="inferno",
                      extent=(-0.5, args.n_beams - 0.5, -0.5, args.n_time - 0.5))
    ax.set(title="Maximum absolute error over frequency", xlabel="beam index",
           ylabel="time index")
    ax.set_xticks(np.arange(args.n_beams))
    fig.colorbar(image, ax=ax, label="absolute intensity error")

    ax = axes[1, 1]
    integrated_reference = reference64.sum(axis=(0, 1))
    integrated_candidate = candidate64.sum(axis=(0, 1))
    denominator = np.maximum(np.abs(integrated_reference), args.abs_tolerance)
    percent_difference = 100.0 * (integrated_candidate - integrated_reference) / denominator
    ax.bar(np.arange(args.n_beams), percent_difference, color="tab:purple")
    for beam, direction in enumerate(directions):
        ax.annotate(f"l={direction[0]:g}", (beam, percent_difference[beam]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
    ax.set(title="Integrated-power difference", xlabel="beam index",
           ylabel=f"({args.compare_label} - {args.label}) / {args.label} [%]")
    ax.set_xticks(np.arange(args.n_beams))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)

    fig.savefig(output, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize and optionally compare [T][F][B] beamformer intensity files."
    )
    parser.add_argument("--input", type=Path, help="reference file; shard 0 when --buffer both")
    parser.add_argument("--input-shard1", type=Path, help="optional shard 1 file when --buffer both")
    parser.add_argument("--compare", type=Path, help="optional second CPU/CUDA intensity file")
    parser.add_argument("--compare-shard1", type=Path, help="optional shard 1 comparison file when --buffer both")
    parser.add_argument("--output", type=Path, help="validation dashboard PNG")
    parser.add_argument("--comparison-output", type=Path, help="CPU/GPU comparison PNG")
    parser.add_argument("--summary-json", type=Path, help="optional comparison metrics JSON")
    parser.add_argument("--sky-output", type=Path,
                        help="optional full-sky beam-grid coverage PNG")
    parser.add_argument("--geometry-output", type=Path,
                        help="optional array geometry and u-v coverage PNG")
    parser.add_argument("--label", default="CPU", help="reference dataset label")
    parser.add_argument("--compare-label", default="CUDA", help="second dataset label")
    parser.add_argument("--n-time", type=int, default=15360)
    parser.add_argument("--buffer", choices=("0", "1", "both"),
                        help="plot local buffer 0, local buffer 1, or both")
    parser.add_argument("--n-freq", type=int,
                        help="compatibility override: 336 for one buffer or 672 for both")
    parser.add_argument("--n-ant", type=int, default=32)
    parser.add_argument("--n-beams", type=int, default=5)
    parser.add_argument("--spacing-m", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--positions", type=Path, help="optional antenna x,y,z text file")
    parser.add_argument("--frequency-hz", type=float,
                        help="optional constant-frequency override")
    parser.add_argument("--frequency-start-hz", type=float,
                        default=DEFAULT_FREQUENCY_START_HZ)
    parser.add_argument("--channel-width-hz", type=float,
                        default=DEFAULT_CHANNEL_WIDTH_HZ)
    parser.add_argument("--design-frequency-hz", type=float,
                        default=BEAM_GRID_DESIGN_FREQUENCY_HZ)
    parser.add_argument("--frequencies", type=Path, help="optional frequency-Hz text file")
    parser.add_argument("--directions", type=Path, help="optional beam x,y,z text file")
    parser.add_argument(
        "--beam-grid", choices=("fft", "line", "legacy-rectangular"),
        default="fft")
    parser.add_argument("--beam-l-step", type=float, default=0.02)
    parser.add_argument("--beam-m", type=float, default=0.0)
    parser.add_argument("--uv-channel", type=int, help="frequency channel used for u-v coverage")
    parser.add_argument("--synthetic-type", choices=("point-source", "one-hot", "constant",
                                                      "noise", "real"),
                        default="point-source")
    parser.add_argument("--source-l", type=float, default=0.04)
    parser.add_argument("--source-m", type=float, default=0.0)
    parser.add_argument("--amplitude", type=float, default=4.0)
    parser.add_argument("--active-time", type=int, default=0)
    parser.add_argument("--active-frequency", type=int, default=0)
    parser.add_argument("--active-element", type=int, default=0)
    parser.add_argument("--value-real", type=float, default=1.0)
    parser.add_argument("--value-imag", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--abs-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--rel-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--sky-resolution", type=int, default=121,
                        help="pixels per l/m axis for sky coverage; default: 121")
    parser.add_argument("--overlap-db", type=float, default=-3.0,
                        help="per-beam relative threshold for overlap contours")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="also open an interactive window")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_time <= 0:
        raise ValueError("n_time must be positive")
    args.buffer, args.n_freq = resolve_frequency_configuration(args.buffer, args.n_freq)
    args.buffer_label = buffer_label(args.buffer, args.n_freq)
    valid_beam_count = 1 <= args.n_beams <= 128
    if args.n_ant not in (32, 64) or not valid_beam_count:
        raise ValueError("n_ant must be 32 or 64; n_beams must be 1 to 128")
    if args.beam_grid == "legacy-rectangular" and args.n_beams != args.n_ant:
        raise ValueError("legacy-rectangular requires n_beams == n_ant")
    if args.spacing_m <= 0.0 or args.frequency_start_hz <= 0.0 \
            or args.channel_width_hz <= 0.0 or args.design_frequency_hz <= 0.0:
        raise ValueError("spacing and frequency parameters must be positive")
    if args.frequency_hz is not None and args.frequency_hz <= 0.0:
        raise ValueError("constant frequency override must be positive")
    if args.abs_tolerance <= 0.0 or args.rel_tolerance <= 0.0:
        raise ValueError("comparison tolerances must be positive")
    if args.sky_resolution < 51 or args.overlap_db >= 0.0:
        raise ValueError("sky resolution must be at least 51 and overlap-db must be negative")
    if args.input is None and args.sky_output is None and args.geometry_output is None:
        raise ValueError("provide --input, --geometry-output, --sky-output, or a combination")
    if args.compare is not None and args.input is None:
        raise ValueError("--compare requires --input")
    if args.input_shard1 is not None and args.buffer != "both":
        raise ValueError("--input-shard1 requires --buffer both")
    if args.compare_shard1 is not None and args.buffer != "both":
        raise ValueError("--compare-shard1 requires --buffer both")
    if args.input_shard1 is not None and args.input is None:
        raise ValueError("--input-shard1 requires --input as shard 0")
    if args.compare_shard1 is not None and args.compare is None:
        raise ValueError("--compare-shard1 requires --compare as shard 0")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    positions, frequencies, directions = _load_geometry(args)
    intensity = (load_selected_intensity(
        args.input, args.input_shard1, args.n_time, args.n_freq, args.n_beams, args.buffer)
                 if args.input else None)
    comparison = (load_selected_intensity(
        args.compare, args.compare_shard1, args.n_time, args.n_freq, args.n_beams, args.buffer)
                  if args.compare else None)
    if intensity is not None:
        dashboard = plot_dashboard(
            args, intensity, comparison, positions, frequencies, directions)
        print(f"Wrote validation dashboard: {dashboard}")
        print(f"Peak integrated beam: {int(np.argmax(intensity.sum(axis=(0, 1))))}")

    if intensity is not None and comparison is not None:
        metrics = comparison_metrics(intensity, comparison, args.abs_tolerance,
                                     args.rel_tolerance)
        comparison_plot = plot_comparison(args, intensity, comparison, directions, metrics)
        print(f"Wrote comparison dashboard: {comparison_plot}")
        print(json.dumps(metrics, indent=2, allow_nan=True))
        if args.summary_json:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
            print(f"Wrote comparison metrics: {args.summary_json}")
    elif args.summary_json:
        raise ValueError("--summary-json requires --compare")
    if args.geometry_output:
        geometry_plot = plot_geometry_and_uv(args, positions, frequencies)
        print(f"Wrote geometry validation: {geometry_plot}")
    if args.sky_output:
        sky_plot = plot_sky_coverage(args, frequencies, directions)
        print(f"Wrote full-sky beam coverage: {sky_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
