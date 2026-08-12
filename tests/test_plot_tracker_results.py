#!/usr/bin/env python3

"""Unit tests for the tracker-beam plotting helper (tools/plot_tracker_results.py)."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# plot_tracker_results.py imports shared helpers from plot_results via a
# bare module import, so the tools/ directory must be importable before we
# exec the script under test.
TOOLS_DIR = Path(__file__).parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

SCRIPT = TOOLS_DIR / "plot_tracker_results.py"
SPEC = importlib.util.spec_from_file_location("plot_tracker_results", SCRIPT)
plot_tracker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["plot_tracker_results"] = plot_tracker
SPEC.loader.exec_module(plot_tracker)

class TrackerTrajectoryTest(unittest.TestCase):
    def test_stationary_trajectory_repeats_start(self):
        directions = plot_tracker.tracker_directions(0.0, 0.0, 0.0, 0.0, 10)
        self.assertEqual(directions.shape, (10, 3))
        np.testing.assert_allclose(directions[:, 0], 0.0)
        np.testing.assert_allclose(directions[:, 1], 0.0)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_linear_drift_matches_closed_form(self):
        n_time = 5
        directions = plot_tracker.tracker_directions(
            0.02, -0.03, 1.0e-4, -2.0e-4, n_time)
        t = np.arange(n_time, dtype=np.float64)
        expected_l = 0.02 + t * 1.0e-4
        expected_m = -0.03 + t * -2.0e-4
        np.testing.assert_allclose(directions[:, 0], expected_l)
        np.testing.assert_allclose(directions[:, 1], expected_m)
        np.testing.assert_allclose(
            directions[:, 2], np.sqrt(1.0 - expected_l**2 - expected_m**2))

    def test_off_disk_trajectory_raises(self):
        with self.assertRaises(ValueError):
            plot_tracker.tracker_directions(0.9, 0.0, 0.01, 0.0, 20)

    def test_window_steering_uses_first_sample(self):
        n_time = 8
        window_dirs = plot_tracker.window_steering_directions(
            0.0, 0.0, 0.001, 0.0, n_time, integration_spectra=2)
        self.assertEqual(window_dirs.shape, (4, 3))
        # Window w starts at sample w * integration_spectra.
        np.testing.assert_allclose(window_dirs[:, 0], [0.0, 0.002, 0.004, 0.006])
        # All on the unit disk.
        np.testing.assert_allclose(np.linalg.norm(window_dirs, axis=1), 1.0)

    def test_moving_point_source_overlay_matches_trajectory(self):
        n_time = 3
        source = plot_tracker.moving_point_source_directions(
            0.0, 0.0, 0.005, -0.01, n_time)
        traj = plot_tracker.tracker_directions(0.0, 0.0, 0.005, -0.01, n_time)
        np.testing.assert_allclose(source, traj)

    def test_resolution_and_buffer_validation(self):
        args = plot_tracker.build_parser().parse_args([
            "--input", "x.bin", "--output", "out.png", "--n-time", "4",
        ])
        plot_tracker.validate_args(args)
        self.assertEqual(plot_tracker.resolve_frequency_channels("0", None), 336)
        self.assertEqual(plot_tracker.resolve_frequency_channels("both", None), 672)
        with self.assertRaises(ValueError):
            plot_tracker.resolve_frequency_channels("0", 672)

    def test_load_intensity_round_trip(self):
        # A tiny [2][2][1] float32 file should load unchanged.
        expected = np.arange(4, dtype="<f4").reshape(2, 2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracker.bin"
            expected.tofile(path)
            loaded = plot_tracker.load_intensity(path, 2, 2, 1)
        np.testing.assert_array_equal(loaded, expected)


class TrackerDashboardRenderTest(unittest.TestCase):
    def test_metadata_block_contains_run_geometry(self):
        args = plot_tracker.build_parser().parse_args([
            "--input", "x.bin", "--output", "out.png", "--n-time", "320",
            "--n-ant", "64", "--integration-spectra", "320",
            "--track-l0", "0.0", "--dl-per-sample", "1.0e-4",
        ])
        args.n_freq = plot_tracker.resolve_frequency_channels(args.buffer,
                                                              args.n_freq)
        positions = plot_tracker.default_positions(args.n_ant, args.spacing_m)
        frequencies = plot_tracker.default_frequencies(
            args.n_freq, plot_tracker.DEFAULT_FREQUENCY_START_HZ,
            args.channel_width_hz)
        text = plot_tracker.array_metadata_text(args, frequencies, positions)
        self.assertIn("8x8", text)
        self.assertIn("n_ant=64", text)
        self.assertIn("windows=1", text)
        self.assertIn("rate=(0.0001,0.0)/sample", text)

    def test_dashboard_writes_png(self):
        import matplotlib
        matplotlib.use("Agg")
        # The buffer contract fixes n_freq at 336; the intensity file must match.
        n_time, n_freq, n_ant = 4, 336, 32
        intensity = np.arange(n_time * n_freq, dtype="<f4").reshape(
            n_time, n_freq, 1) + 1.0
        with tempfile.TemporaryDirectory() as tmp:
            intensity_path = Path(tmp) / "tracker_intensity.bin"
            output_path = Path(tmp) / "tracker_dashboard.png"
            intensity.tofile(intensity_path)
            rc = plot_tracker.main([
                "--input", str(intensity_path),
                "--output", str(output_path),
                "--n-time", str(n_time),
                "--n-ant", str(n_ant),
                "--track-l0", "0.0",
                "--track-m0", "0.0",
                "--dl-per-sample", "0.001",
                "--dm-per-sample", "0.0",
                "--integration-spectra", "1",
                "--source-l0", "0.0",
                "--source-m0", "0.0",
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_frames_dir_emits_zero_padded_pngs(self):
        import matplotlib
        matplotlib.use("Agg")
        # 4 windows (n_time=320, integration_spectra=80); emit every window.
        n_time, n_freq, n_ant = 320, 336, 32
        intensity = (np.arange(n_time * n_freq, dtype="<f4")
                     .reshape(n_time, n_freq, 1) + 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            intensity_path = Path(tmp) / "tracker_intensity.bin"
            frames_dir = Path(tmp) / "frames"
            intensity.tofile(intensity_path)
            rc = plot_tracker.main([
                "--input", str(intensity_path),
                "--frames-dir", str(frames_dir),
                "--n-time", str(n_time),
                "--n-ant", str(n_ant),
                "--track-l0", "0.0",
                "--track-m0", "0.0",
                "--dl-per-sample", "1.0e-4",
                "--dm-per-sample", "0.0",
                "--integration-spectra", "80",
                "--frames-stride", "1",
                "--source-l0", "0.0",
                "--source-m0", "0.0",
                "--source-dl-per-sample", "1.0e-4",
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(frames_dir.is_dir())
        written = sorted(frames_dir.glob("frame_*.png"))
        self.assertEqual(len(written), 4)
        # Zero-padded sequential names, all non-empty.
        names = [p.name for p in written]
        self.assertEqual(names, ["frame_0.png", "frame_1.png",
                                 "frame_2.png", "frame_3.png"])
        for p in written:
            self.assertGreater(p.stat().st_size, 0)

    def test_frames_stride_downsamples(self):
        import matplotlib
        matplotlib.use("Agg")
        n_time, n_freq, n_ant = 320, 336, 32
        intensity = (np.arange(n_time * n_freq, dtype="<f4")
                     .reshape(n_time, n_freq, 1) + 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            intensity_path = Path(tmp) / "tracker_intensity.bin"
            frames_dir = Path(tmp) / "frames_stride"
            intensity.tofile(intensity_path)
            rc = plot_tracker.main([
                "--input", str(intensity_path),
                "--frames-dir", str(frames_dir),
                "--n-time", str(n_time),
                "--n-ant", str(n_ant),
                "--integration-spectra", "80",
                "--frames-stride", "2",
            ])
        self.assertEqual(rc, 0)
        written = sorted(frames_dir.glob("frame_*.png"))
        self.assertEqual(len(written), 2)  # windows 0 and 2

    def test_frames_max_caps_emitted_count(self):
        import matplotlib
        matplotlib.use("Agg")
        n_time, n_freq, n_ant = 320, 336, 32
        intensity = (np.arange(n_time * n_freq, dtype="<f4")
                     .reshape(n_time, n_freq, 1) + 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            intensity_path = Path(tmp) / "tracker_intensity.bin"
            frames_dir = Path(tmp) / "frames_cap"
            intensity.tofile(intensity_path)
            with self.assertRaises(ValueError):
                plot_tracker.main([
                    "--input", str(intensity_path),
                    "--frames-dir", str(frames_dir),
                    "--n-time", str(n_time),
                    "--n-ant", str(n_ant),
                    "--integration-spectra", "80",
                    "--frames-max", "2",
                ])


if __name__ == "__main__":
    unittest.main()
