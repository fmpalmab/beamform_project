#!/usr/bin/env python3
"""
Unit and Integration Tests for tools/generate_presentation_suite.py.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

# Add project root and tools to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"

import sys
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_presentation_suite as gps


class TestGeneratePresentationSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)
        gps.set_presentation_style()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_evolution_dataset(self):
        dataset = gps.get_default_evolution_dataset()
        self.assertGreaterEqual(len(dataset), 8)
        v5_entries = [d for d in dataset if "V5" in d.version]
        self.assertTrue(len(v5_entries) >= 2)
        # Check speedup values are positive and reasonable
        for d in dataset:
            self.assertGreater(d.lat_64_ms, 0.0)
            self.assertGreater(d.speedup_64_vs_naive, 0.0)

    def test_generate_evolution_plots_and_data(self):
        png_path = gps.generate_evolution_plots_and_data(self.outdir)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

        csv_path = self.outdir / "evolution_comparison.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("Version", header)
            self.assertIn("Speedup_64ant_vs_CPUNaive", header)
            rows = list(reader)
            self.assertGreaterEqual(len(rows), 8)

    def test_generate_cuda_v5_benchmark_deepdive(self):
        png_path = gps.generate_cuda_v5_benchmark_deepdive(self.outdir)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

        csv_path = self.outdir / "cuda_v5_benchmark_results.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("Antenna_Count", header)
            self.assertIn("Compute_Throughput_TFLOPs", header)

    def test_generate_astronomical_validation_dashboard(self):
        png_path = gps.generate_astronomical_validation_dashboard(self.outdir, engine="cuda_v5")
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

        json_path = self.outdir / "astronomical_validation_metrics.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("benchmark_source", data)
            self.assertIn("injected_dm_pc_cm3", data)
            self.assertIn("recovered_dm_pc_cm3", data)
            self.assertIn("radiometer_scaling_slope", data)
            self.assertLess(data["dm_recovery_error"], 0.5)

    def test_generate_beam_footprints_plot(self):
        png_path = gps.generate_beam_footprints_plot(self.outdir)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

    def test_generate_tracker_motion_and_power_dynamics(self):
        png_path = gps.generate_tracker_motion_and_power_dynamics(self.outdir)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

        csv_path = self.outdir / "tracker_power_dynamics.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("Pointing_Error_arcmin", header)
            self.assertIn("Tracked_Power_dB", header)
            self.assertIn("Untracked_Drift_Power_dB", header)

    def test_generate_classification_signals_demonstration(self):
        png_path = gps.generate_classification_signals_demonstration(self.outdir)
        self.assertTrue(png_path.exists())
        self.assertGreater(png_path.stat().st_size, 1000)

        json_path = self.outdir / "classification_signals_data.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("classes", data)
            self.assertIn("FRB_Astrophysical", data["classes"])
            self.assertIn("RFI_Broadband_ZeroDM", data["classes"])
            self.assertIn("Background_Noise_Nothing", data["classes"])

    def test_generate_presentation_deck_markdown_and_manifest(self):
        md_path = gps.generate_presentation_deck_markdown(self.outdir)
        self.assertTrue(md_path.exists())
        self.assertGreater(md_path.stat().st_size, 500)

        manifest_path = gps.generate_manifest_json(self.outdir)
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            self.assertIn("figures", manifest)
            self.assertEqual(len(manifest["figures"]), 6)


if __name__ == "__main__":
    unittest.main()
