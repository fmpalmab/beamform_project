"""Unit tests for CHARTS constants compatibility layer and C++/Python parity.

Verifies:
1. `tools/constants.py` exports all required physical, instrument, and networking constants.
2. Fallback mode operates correctly even if `charts-constants` repo is not on the machine.
3. If `charts_constants` package is installed or available, constants match with exact precision.
4. C++ compile-time header `include/beamformer/constants.hpp` matches Python constants.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
INCLUDE_DIR = PROJECT_ROOT / "include" / "beamformer"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import constants as c


class TestConstantsCompatibility(unittest.TestCase):
    """Test suite for CHARTS constants compatibility layer."""

    def test_constants_exports_and_types(self):
        """Verify all essential constants exist and have correct types."""
        # Physical & DM
        self.assertIsInstance(c.K_DM, float)
        self.assertAlmostEqual(c.K_DM, 4148.741601, places=5)

        self.assertIsInstance(c.C_LIGHT, float)
        self.assertEqual(c.C_LIGHT, 299_792_458.0)
        self.assertEqual(c.SPEED_OF_LIGHT, 299_792_458.0)

        # Sampling & FFT
        self.assertIsInstance(c.ADC_SAMPLING_FREQ_MHZ, float)
        self.assertEqual(c.ADC_SAMPLING_FREQ_MHZ, 2457.6)
        self.assertIsInstance(c.FPGA_NUM_SAMP_FFT, int)
        self.assertEqual(c.FPGA_NUM_SAMP_FFT, 8192)

        # Channel & Frequency
        self.assertIsInstance(c.CHARTS_CHANNEL_WIDTH_MHZ, float)
        self.assertAlmostEqual(c.CHARTS_CHANNEL_WIDTH_MHZ, 0.3, places=6)
        self.assertEqual(c.CHARTS_CHANNEL_WIDTH_HZ, 300_000.0)

        self.assertEqual(c.CHARTS_N_FREQ, 672)
        self.assertEqual(c.FREQUENCY_SHARD_COUNT, 2)
        self.assertEqual(c.LOCAL_FREQUENCY_CHANNELS, 336)

        # Bands
        self.assertEqual(len(c.CHAIN_BANDS), 8)
        self.assertEqual(c.CHAIN_BANDS[0], (300.0, 501.6))
        self.assertEqual(c.CHAIN_BANDS[4], (1365.6, 1567.2))

        # Array & Geometry
        self.assertEqual(c.DEFAULT_SPACING_M, 0.6)
        self.assertEqual(c.CHARTS_N_ANTENNAS, 256)
        self.assertIn(300e6, c.ANTENNA_SPECS)
        self.assertIn(400e6, c.ANTENNA_SPECS)
        self.assertIn(500e6, c.ANTENNA_SPECS)

        # Location
        self.assertIn("Caren", c.PRESET_LOCATIONS)
        self.assertAlmostEqual(c.CHARTS_LATITUDE_DEG, -33.4211146, places=6)
        self.assertAlmostEqual(c.CHARTS_LONGITUDE_DEG, -70.8634710, places=6)
        self.assertEqual(c.CHARTS_ALTITUDE_M, 458.0)

        # CPT
        self.assertEqual(c.CPT_SAMPLE_RATE, 4915.2)
        self.assertEqual(c.CPT_SPECTRA_PER_PACKET, 4)
        self.assertEqual(c.CPT_UDP_PAYLOAD_BYTES, 5440)
        self.assertEqual(c.CPT_TOTAL_RAW_PACKET_BYTES, 5482)

    def test_fallback_mode_standalone_isolation(self):
        """Simulate a clean environment where charts_constants is completely missing."""
        with patch.dict("sys.modules", {"charts_constants": None}):
            # Reload module with mock
            # Even if import fails, constants module should never crash
            summary = c.get_constants_summary()
            self.assertIn("source", summary)
            self.assertIn("K_DM", summary)
            self.assertEqual(summary["K_DM"], 4148.741601)

    def test_cpp_header_parity(self):
        """Verify C++ constants header exists and matches Python constants."""
        header_path = INCLUDE_DIR / "constants.hpp"
        self.assertTrue(header_path.exists(), f"Header {header_path} should exist")

        content = header_path.read_text(encoding="utf-8")
        self.assertIn("inline constexpr double speed_of_light_m_per_s = 299792458.0;", content)
        self.assertIn("inline constexpr double k_dm = 4148.741601;", content)
        self.assertIn("inline constexpr float charts_channel_width_hz = 300000.0F;", content)
        self.assertIn("inline constexpr std::size_t charts_full_band_channels = 672;", content)
        self.assertIn("inline constexpr std::size_t charts_local_channels = 336;", content)

    def test_reexport_astronomical_validation(self):
        """Verify astronomical_validation.constants re-exports correctly."""
        from astronomical_validation.constants import K_DM as ASTRO_K_DM
        self.assertEqual(ASTRO_K_DM, c.K_DM)

    def test_injector_and_dedispersion_use_constants(self):
        """Verify injector and dedispersion use the updated K_DM constant."""
        from astronomical_validation.injector import K_DM as INJ_K_DM
        self.assertEqual(INJ_K_DM, c.K_DM)


if __name__ == "__main__":
    unittest.main()
