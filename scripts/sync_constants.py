#!/usr/bin/env python3
"""Synchronize and verify CHARTS constants across C++ headers and Python tooling.

Reads from `tools/constants.py` (which loads from `charts_constants` if installed/available
or uses the built-in CHARTS fallback definitions) and generates or validates
`include/beamformer/constants.hpp`.

Usage:
    python scripts/sync_constants.py         # Regenerate header
    python scripts/sync_constants.py --check # Verify parity without modifying
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
INCLUDE_DIR = PROJECT_ROOT / "include" / "beamformer"
HEADER_PATH = INCLUDE_DIR / "constants.hpp"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import constants as c


def generate_header_content() -> str:
    """Generate the contents of include/beamformer/constants.hpp."""
    return f"""#pragma once

#include <cstddef>

namespace beamformer {{
namespace constants {{

// Physical and Astrophysical Constants
inline constexpr double speed_of_light_m_per_s = {c.C_LIGHT:.1f};
inline constexpr double k_dm = {c.K_DM:.6f}; // MHz^2 s / (pc cm^-3)
inline constexpr double two_pi = 6.283185307179586476925286766559;

// Instrumental & Sampling Constants
inline constexpr double adc_sampling_freq_hz = {c.ADC_SAMPLING_FREQ_HZ:.1e};
inline constexpr double adc_sampling_freq_mhz = {c.ADC_SAMPLING_FREQ_MHZ:.1f};
inline constexpr std::size_t fpga_num_samp_fft = {c.FPGA_NUM_SAMP_FFT};
inline constexpr double fpga_time_resolution_s = {c.FPGA_TIME_RESOLUTION_S:.17e}; // ~3.333 us

// Band & Channel Specifications
inline constexpr float charts_channel_width_hz = {c.CHARTS_CHANNEL_WIDTH_HZ:.1f}F;
inline constexpr float charts_channel_width_mhz = {c.CHARTS_CHANNEL_WIDTH_MHZ:.1f}F;
inline constexpr float charts_frequency_start_hz = {c.DEFAULT_FREQUENCY_START_HZ:.1f}F;
inline constexpr float charts_design_frequency_hz = {c.BEAM_GRID_DESIGN_FREQUENCY_HZ:.1f}F;
inline constexpr std::size_t charts_full_band_channels = {c.FULL_BAND_FREQUENCY_CHANNELS};
inline constexpr std::size_t charts_shard_count = {c.FREQUENCY_SHARD_COUNT};
inline constexpr std::size_t charts_local_channels = {c.LOCAL_FREQUENCY_CHANNELS}; // 336
inline constexpr std::size_t charts_upchannelized_factor = {c.UPCHANNELIZED_FACTOR};
inline constexpr std::size_t charts_upchannelized_channels = {c.UPCHANNELIZED_N_FREQ}; // 21504

// Telescope Geometry & Array Defaults
inline constexpr float charts_default_spacing_m = {c.DEFAULT_SPACING_M:.1f}F;
inline constexpr std::size_t charts_total_antennas = {c.CHARTS_N_ANTENNAS};

// Site Coordinates (Carén Observatory Site)
inline constexpr double charts_caren_lat_deg = {c.CHARTS_LATITUDE_DEG:.7f};
inline constexpr double charts_caren_lon_deg = {c.CHARTS_LONGITUDE_DEG:.7f};
inline constexpr double charts_caren_alt_m = {c.CHARTS_ALTITUDE_M:.1f};

// CPT Dual-Band Networking Constants
inline constexpr double cpt_sample_rate_mhz = {c.CPT_SAMPLE_RATE:.1f};
inline constexpr double cpt_delta_time_s = {c.CPT_DELTA_TIME:.17e};
inline constexpr std::size_t cpt_spectra_per_packet = {c.CPT_SPECTRA_PER_PACKET};
inline constexpr std::size_t cpt_udp_payload_bytes = {c.CPT_UDP_PAYLOAD_BYTES};
inline constexpr std::size_t cpt_total_raw_packet_bytes = {c.CPT_TOTAL_RAW_PACKET_BYTES};

}} // namespace constants
}} // namespace beamformer
"""


def main():
    parser = argparse.ArgumentParser(description="Synchronize CHARTS constants")
    parser.add_argument("--check", action="store_true", help="Check parity without writing")
    args = parser.parse_args()

    content = generate_header_content()

    print(f"Loaded constants from source: {c.CONSTANTS_SOURCE}")
    print(f"  K_DM    = {c.K_DM}")
    print(f"  C_LIGHT = {c.C_LIGHT}")
    print(f"  N_FREQ  = {c.CHARTS_N_FREQ} (sharded: {c.LOCAL_FREQUENCY_CHANNELS})")

    if args.check:
        if not HEADER_PATH.exists():
            print(f"ERROR: {HEADER_PATH} does not exist!")
            sys.exit(1)
        existing = HEADER_PATH.read_text(encoding="utf-8")
        # Normalize whitespace
        if existing.strip() != content.strip():
            print("ERROR: C++ constants header is out of sync with Python constants!")
            sys.exit(1)
        print("OK: C++ constants header is in sync.")
        sys.exit(0)

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {HEADER_PATH}")


if __name__ == "__main__":
    main()
