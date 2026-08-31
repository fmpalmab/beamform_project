#!/usr/bin/env python3
"""CHARTS Constants Compatibility Layer for beamform_project.

Loads official constants from the `charts_constants` package if installed or
present in the filesystem. If `charts_constants` is not available (e.g. initial
clone or standalone environment), gracefully falls back to embedded exact standard
CHARTS constants so that all beamformer tools, simulations, and tests run out of the box.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("beamformer.constants")

# ---------------------------------------------------------------------------
# Attempt to load from external `charts_constants` package or adjacent repository
# ---------------------------------------------------------------------------
_external_module = None
_source_description = "embedded_fallback"

# 1. Check if charts_constants is already in sys.modules or installed in environment
try:
    _external_module = importlib.import_module("charts_constants")
    _source_description = "installed_charts_constants"
except Exception:
    # 2. Check adjacent directories (e.g., ../charts-constants)
    _project_root = Path(__file__).resolve().parent.parent
    _adjacent_paths = [
        _project_root.parent / "charts-constants",
        _project_root / "charts-constants",
    ]
    for _path in _adjacent_paths:
        if _path.exists() and (_path / "charts_constants").is_dir():
            if str(_path) not in sys.path:
                sys.path.insert(0, str(_path))
            try:
                _external_module = importlib.import_module("charts_constants")
                _source_description = f"path_charts_constants ({_path})"
                break
            except Exception:
                pass


def _extract_val(val: Any) -> Any:
    """Extract raw float/int/array from Astropy Quantity or return as-is."""
    if hasattr(val, "value"):
        return val.value
    return val


def _has_external(attr_name: str) -> bool:
    return _external_module is not None and hasattr(_external_module, attr_name)


def _get_external(attr_name: str, default: Any) -> Any:
    if _has_external(attr_name):
        raw = getattr(_external_module, attr_name)
        return _extract_val(raw)
    return default


# ---------------------------------------------------------------------------
# Metadata Flags
# ---------------------------------------------------------------------------
USING_CHARTS_CONSTANTS_PACKAGE: bool = _external_module is not None
USING_FALLBACK_CONSTANTS: bool = not USING_CHARTS_CONSTANTS_PACKAGE
CONSTANTS_SOURCE: str = _source_description


# ---------------------------------------------------------------------------
# 1. Dispersion Measure & Physical Constants
# ---------------------------------------------------------------------------
# Dispersion constant in MHz^2 s / (pc cm^-3)
# Exact CHARTS definition: 4148.741601
K_DM: float = float(_get_external("K_DM", 4148.741601))

# Speed of light in m/s
C_LIGHT: float = float(_get_external("C_LIGHT", 299_792_458.0))
SPEED_OF_LIGHT: float = C_LIGHT
SPEED_OF_LIGHT_M_PER_S: float = C_LIGHT


# ---------------------------------------------------------------------------
# 2. CHARTS Instrumental & FPGA Constants
# ---------------------------------------------------------------------------
# Sampling frequency
ADC_SAMPLING_FREQ_HZ: float = float(_get_external("ADC_SAMPLING_FREQ", 2457.6e6))
ADC_SAMPLING_FREQ_MHZ: float = float(_get_external("ADC_SAMPLING_FREQ_MHZ", 2457.6))
DEFAULT_SAMPLE_RATE: float = ADC_SAMPLING_FREQ_MHZ
FPGA_FREQ0_MHZ: float = ADC_SAMPLING_FREQ_MHZ

# FPGA FFT parameters
FPGA_NUM_SAMP_FFT: int = int(_get_external("FPGA_NUM_SAMP_FFT", 8192))
DEFAULT_NFFT: int = FPGA_NUM_SAMP_FFT
NFFT: int = FPGA_NUM_SAMP_FFT

# Channel width
CHARTS_CHANNEL_WIDTH_MHZ: float = float(_get_external("CHARTS_CHANNEL_WIDTH_MHZ", 0.3))
CHARTS_CHANNEL_WIDTH_HZ: float = CHARTS_CHANNEL_WIDTH_MHZ * 1e6
DEFAULT_CHANNEL_WIDTH_HZ: float = CHARTS_CHANNEL_WIDTH_HZ

# FPGA frame cadence / time resolution
FPGA_TIME_RESOLUTION_S: float = float(_get_external("FPGA_TIME_RESOLUTION_S", 8192 / 2457.6e6))
FPGA_TIME_RESOLUTION_MS: float = FPGA_TIME_RESOLUTION_S * 1e3

# ADC labels
ADC_LABELS: List[str] = list(_get_external("ADC_LABELS", ["D", "C", "B", "A"]))


# ---------------------------------------------------------------------------
# 3. Frequency Multiplexing & Band Constants
# ---------------------------------------------------------------------------
# Coarse channels inside chosen band (Full Band)
CHARTS_N_FREQ: int = int(_get_external("CHARTS_N_FREQ", 672))
NUM_CHANNELS: int = CHARTS_N_FREQ
ANTENNA_CHANNELS: int = CHARTS_N_FREQ
FULL_BAND_FREQUENCY_CHANNELS: int = CHARTS_N_FREQ

# Sharded processing parameters (Beamformer local contract)
FREQUENCY_SHARD_COUNT: int = 2
LOCAL_FREQUENCY_CHANNELS: int = CHARTS_N_FREQ // FREQUENCY_SHARD_COUNT  # 336
DEFAULT_N_FREQ: int = LOCAL_FREQUENCY_CHANNELS

# 8 Analog Chains / Frequency Bands: (freq_min, freq_max) in MHz
CHAIN_BANDS: List[Tuple[float, float]] = list(_get_external("CHAIN_BANDS", [
    (300.0, 501.6),     # Chain 0 (bins 1000:1672)
    (564.0, 765.6),     # Chain 1 (bins 1880:2552)
    (830.4, 1032.0),    # Chain 2 (bins 2768:3440)
    (1099.2, 1300.8),   # Chain 3 (bins 3664:4336)
    (1365.6, 1567.2),   # Chain 4 (bins 4552:5224)
    (1632.0, 1833.6),   # Chain 5 (bins 5440:6112)
    (1900.8, 2102.4),   # Chain 6 (bins 6336:7008)
    (2164.8, 2366.4),   # Chain 7 (bins 7216:7888)
]))

BAND_EDGES: List[float] = [edge for band in CHAIN_BANDS for edge in band]
BAND_EDGES_SPECTROMETER: List[float] = BAND_EDGES

# Default band 0 start frequency
DEFAULT_FREQUENCY_START_HZ: float = CHAIN_BANDS[0][0] * 1e6  # 300_000_000.0
BEAM_GRID_DESIGN_FREQUENCY_HZ: float = 400_000_000.0
DEFAULT_SPACING_M: float = 0.6

# Upchannelized parameters
UPCHANNELIZED_FACTOR: int = int(_get_external("UPCHANNELIZED_FACTOR", 32))
UPCHANNELIZED_N_FREQ: int = CHARTS_N_FREQ * UPCHANNELIZED_FACTOR  # 21504


# ---------------------------------------------------------------------------
# 4. Telescope Physical & Antenna Specs
# ---------------------------------------------------------------------------
ANTENNA_SPECS: Dict[float, Dict[str, float]] = {
    300e6: {"BW_E": 92.0, "BW_H": 66.0, "gain_dBi": 8.7},
    400e6: {"BW_E": 108.0, "BW_H": 74.0, "gain_dBi": 7.75},
    500e6: {"BW_E": 120.0, "BW_H": 87.0, "gain_dBi": 7.0},
}
if _has_external("ANTENNA_SPECS"):
    ANTENNA_SPECS = _get_external("ANTENNA_SPECS", ANTENNA_SPECS)

# Hexagonal lattice scaling factor
ETA_HEX: float = float(_get_external("ETA_HEX", np.pi / (2.0 * np.sqrt(3.0))))


# ---------------------------------------------------------------------------
# 5. Site Location & Telescope Layout
# ---------------------------------------------------------------------------
PRESET_LOCATIONS: Dict[str, Tuple[float, float, float]] = {
    "Calan": (-33.39628, -70.536695, 860.0),
    "Caren": (-33.4211146, -70.8634710, 458.0),
}
if _has_external("PRESET_LOCATIONS"):
    PRESET_LOCATIONS = _get_external("PRESET_LOCATIONS", PRESET_LOCATIONS)

LOCATIONS: Dict[str, Tuple[float, float, float]] = PRESET_LOCATIONS

CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG, CHARTS_ALTITUDE_M = PRESET_LOCATIONS["Caren"]
CHARTS_N_ANTENNAS: int = int(_get_external("CHARTS_N_ANTENNAS", 256))
CHARTS_N_POL: int = int(_get_external("CHARTS_N_POL", 1))


# ---------------------------------------------------------------------------
# 6. CPT Dual-Band Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: float = float(_get_external("SAMPLE_RATE", 4915.2))
DELTA_TIME: float = float(_get_external("DELTA_TIME", (10.0 / 3.0) * 1e-6))
FREQ_0_BAND1: float = float(_get_external("FREQ_0_BAND1", 300.0))
FREQ_0_BAND2: float = float(_get_external("FREQ_0_BAND2", 1365.6))
DELTA_FREQ: float = float(_get_external("DELTA_FREQ", 0.3))
NUM_FREQ: int = int(_get_external("NUM_FREQ", 672))
NUM_BANDS: int = int(_get_external("NUM_BANDS", 2))
SPECTRA_PER_PACKET: int = int(_get_external("SPECTRA_PER_PACKET", 4))

NET_HEADER_BYTES: int = int(_get_external("NET_HEADER_BYTES", 42))
HEADER_BYTES: int = int(_get_external("HEADER_BYTES", 64))
DATA_WORDS: int = int(_get_external("DATA_WORDS", 84))
WORD_BYTES: int = int(_get_external("WORD_BYTES", 64))
EXPECTED_PAYLOAD_BYTES: int = int(_get_external("EXPECTED_PAYLOAD_BYTES", 5376))
UDP_PAYLOAD_BYTES: int = int(_get_external("UDP_PAYLOAD_BYTES", 5440))
TOTAL_RAW_PACKET_BYTES: int = int(_get_external("TOTAL_RAW_PACKET_BYTES", 5482))

# CPT-prefixed aliases
CPT_SAMPLE_RATE = SAMPLE_RATE
CPT_DELTA_TIME = DELTA_TIME
CPT_FREQ_0_BAND1 = FREQ_0_BAND1
CPT_FREQ_0_BAND2 = FREQ_0_BAND2
CPT_DELTA_FREQ = DELTA_FREQ
CPT_NUM_FREQ = NUM_FREQ
CPT_NUM_BANDS = NUM_BANDS
CPT_SPECTRA_PER_PACKET = SPECTRA_PER_PACKET
CPT_NET_HEADER_BYTES = NET_HEADER_BYTES
CPT_HEADER_BYTES = HEADER_BYTES
CPT_DATA_WORDS = DATA_WORDS
CPT_WORD_BYTES = WORD_BYTES
CPT_EXPECTED_PAYLOAD_BYTES = EXPECTED_PAYLOAD_BYTES
CPT_UDP_PAYLOAD_BYTES = UDP_PAYLOAD_BYTES
CPT_TOTAL_RAW_PACKET_BYTES = TOTAL_RAW_PACKET_BYTES


def get_constants_summary() -> Dict[str, Any]:
    """Return a dictionary summarizing current constants and source."""
    return {
        "source": CONSTANTS_SOURCE,
        "using_external_package": USING_CHARTS_CONSTANTS_PACKAGE,
        "K_DM": K_DM,
        "C_LIGHT": C_LIGHT,
        "ADC_SAMPLING_FREQ_MHZ": ADC_SAMPLING_FREQ_MHZ,
        "CHARTS_CHANNEL_WIDTH_MHZ": CHARTS_CHANNEL_WIDTH_MHZ,
        "CHARTS_N_FREQ": CHARTS_N_FREQ,
        "LOCAL_FREQUENCY_CHANNELS": LOCAL_FREQUENCY_CHANNELS,
        "DEFAULT_SPACING_M": DEFAULT_SPACING_M,
        "CHARTS_SITE": "Caren",
        "LATITUDE_DEG": CHARTS_LATITUDE_DEG,
        "LONGITUDE_DEG": CHARTS_LONGITUDE_DEG,
    }
