#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT)
plot_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_results)


class PlotResultsTest(unittest.TestCase):
    def test_default_geometry_matches_cpp_order(self):
        positions = plot_results.default_positions(32)
        self.assertEqual(positions.shape, (32, 3))
        np.testing.assert_array_equal(positions[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(positions[7], [4.2, 0.0, 0.0])
        np.testing.assert_allclose(positions[8], [0.0, 0.6, 0.0])
        np.testing.assert_allclose(positions[-1], [4.2, 1.8, 0.0])

    def test_default_frequency_centers(self):
        frequencies = plot_results.default_frequencies()
        self.assertEqual(frequencies.shape, (672,))
        self.assertEqual(frequencies[0], 300_000_000.0)
        self.assertEqual(frequencies[1], 300_300_000.0)
        self.assertEqual(frequencies[-1], 501_300_000.0)

    def test_default_beams_match_cpp_grid(self):
        directions = plot_results.default_beam_directions(5)
        np.testing.assert_allclose(directions[:, 0], [-0.04, -0.02, 0.0, 0.02, 0.04])
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        self.assertEqual(plot_results.default_beam_directions(16).shape, (16, 3))
        self.assertEqual(
            plot_results.default_beam_directions(128, l_step=0.005).shape,
            (128, 3),
        )

    def test_uv_coverage_contains_conjugate_pairs(self):
        positions = plot_results.default_positions(32)
        uv = plot_results.baseline_uv(positions, 400_000_000.0)
        self.assertEqual(uv.shape, (32 * 31, 2))
        np.testing.assert_allclose(uv[0], -uv[1])
        np.testing.assert_allclose(uv.sum(axis=0), [0.0, 0.0], atol=1.0e-12)

    def test_rectangular_final_grid(self):
        directions = plot_results.rectangular_beam_directions(32)
        self.assertEqual(directions.shape, (32, 3))
        wavelength_m = plot_results.SPEED_OF_LIGHT_M_PER_S / 400_000_000.0
        expected_delta_l = wavelength_m / (8 * 0.6)
        expected_delta_m = wavelength_m / (4 * 0.6)
        np.testing.assert_allclose(np.diff(directions[:8, 0]), expected_delta_l)
        self.assertAlmostEqual(directions[8, 1] - directions[0, 1], expected_delta_m)
        np.testing.assert_allclose(
            directions[12, :2], [0.5 * expected_delta_l, -0.5 * expected_delta_m]
        )
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        self.assertLess(directions[0, 0], 0.0)
        self.assertLess(directions[0, 1], 0.0)
        self.assertGreater(directions[-1, 0], 0.0)
        self.assertGreater(directions[-1, 1], 0.0)

    def test_fft_grid_fills_rectangular_32_antenna_bank(self):
        wavelength_m = plot_results.SPEED_OF_LIGHT_M_PER_S / 400_000_000.0
        du = wavelength_m / (2 * 8 * 0.6)
        dv = wavelength_m / (2 * 4 * 0.6)
        centers, n_u, n_v, bank_u, bank_v = (
            plot_results.select_fft_beam_centers_rectangular(
                du, dv, 128, 8, 4)
        )
        self.assertEqual(len(centers), 128)
        self.assertEqual((n_u, n_v), (16, 8))
        self.assertEqual((bank_u, bank_v), (16, 8))
        self.assertEqual(centers[0], (0.0, 0.0))

        directions = plot_results.fft_beam_directions(32, 128)
        self.assertEqual(directions.shape, (128, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_fft_grid_64_antenna_window_and_hex_helpers(self):
        directions = plot_results.fft_beam_directions(64, 128)
        self.assertEqual(directions.shape, (128, 3))
        wavelength_m = plot_results.SPEED_OF_LIGHT_M_PER_S / 400_000_000.0
        du = wavelength_m / (2 * 8 * 0.6)
        _, n_u, n_v, bank_u, bank_v = (
            plot_results.select_fft_beam_centers_rectangular(
                du, du, 128, 8, 8)
        )
        self.assertEqual((n_u, n_v), (12, 11))
        self.assertEqual((bank_u, bank_v), (16, 16))

        estimate = plot_results.estimate_nbeams_hex_formula(
            0.6, 8, 8, 108.0, 74.0, wavelength_m)
        self.assertGreater(estimate, 0)
        uu, vv = plot_results.generate_hex_targets_cropped_fov(128, 0.8, 0.6)
        self.assertEqual(uu.shape, (128,))
        self.assertEqual(vv.shape, (128,))
        self.assertTrue(np.all((uu / 0.8) ** 2 + (vv / 0.6) ** 2 <= 1.0 + 1e-12))

    def test_antenna_specs_interpolate_and_extrapolate(self):
        bw_e, bw_h, gain = plot_results.interpolated_antenna_specs(400e6)
        self.assertEqual(float(bw_e), 108.0)
        self.assertEqual(float(bw_h), 74.0)
        self.assertEqual(float(gain), 7.75)
        bw_e_high, bw_h_high, gain_high = plot_results.interpolated_antenna_specs(501.3e6)
        self.assertAlmostEqual(float(bw_e_high), 120.156)
        self.assertAlmostEqual(float(bw_h_high), 87.169)
        self.assertAlmostEqual(float(gain_high), 6.99025)

    def test_absolute_array_gain_at_beam_center(self):
        directions = plot_results.rectangular_beam_directions(32)
        center = directions[12]
        power = plot_results.beam_power_cube(
            np.asarray([center[0]]), np.asarray([center[1]]), directions[12:13],
            32, 0.6, 400e6,
        )
        element_gain = plot_results.element_factor_2d(
            np.asarray([[center[0]]]), np.asarray([[center[1]]]), 400e6,
        )[0, 0]
        self.assertAlmostEqual(power[0, 0, 0], 32.0 * element_gain, places=9)

    def test_intensity_loader_validates_size_and_layout(self):
        values = np.arange(2 * 672 * 3, dtype="<f4").reshape(2, 672, 3)
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.bin"
            invalid = Path(directory) / "invalid.bin"
            values.tofile(valid)
            invalid.write_bytes(b"\0")
            loaded = plot_results.load_intensity(valid, 2, 672, 3)
            np.testing.assert_array_equal(loaded, values)
            with self.assertRaisesRegex(ValueError, "expected"):
                plot_results.load_intensity(invalid, 2, 672, 3)

    def test_comparison_metrics(self):
        reference = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
        identical = plot_results.comparison_metrics(reference, reference, 1.0e-3, 1.0e-3)
        self.assertEqual(identical["max_absolute_error"], 0.0)
        self.assertEqual(identical["outside_tolerance"], 0)
        self.assertEqual(identical["correlation"], 1.0)

        candidate = np.asarray([1.0, 2.2, 4.0], dtype=np.float32)
        changed = plot_results.comparison_metrics(reference, candidate, 1.0e-3, 1.0e-3)
        self.assertAlmostEqual(changed["max_absolute_error"], 0.2, places=5)
        self.assertEqual(changed["outside_tolerance"], 1)

if __name__ == "__main__":
    unittest.main()
