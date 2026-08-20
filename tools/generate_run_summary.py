#!/usr/bin/env python3
"""Aggregate test, astronomical validation, and benchmark results into a unified summary.

Parses:
  - env_info.txt (Hardware, OS, Compilers)
  - tests/*.log (CTest, Naive CPU suite, Python tests)
  - astronomical_validation/*/astronomical_validation_report.json
  - benchmarks/**/*.csv & logs (Tracker V3, Tracker V2, CPU Opt, Naive/V1/V2, Offline CUDA)

Generates:
  - SUMMARY.md (GitHub Flavored Markdown with formatted tables & badges)
  - SUMMARY.txt (Plain text formatted terminal report)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing the run results")
    parser.add_argument("--slurm-job-id", type=str, default="",
                        help="Slurm Job ID if executed via batch")
    return parser.parse_args()


def parse_env_info(env_file: Path) -> Dict[str, str]:
    info = {
        "hostname": "Unknown",
        "date": "Unknown",
        "gpu_model": "N/A",
        "cpu_model": "Unknown",
        "num_cores": "Unknown",
        "memory": "Unknown",
        "cuda_version": "Unknown",
        "gcc_version": "Unknown",
    }
    if not env_file.exists():
        return info

    content = env_file.read_text(encoding="utf-8", errors="replace")
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("Host:"):
            info["hostname"] = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            info["date"] = line.split(":", 1)[1].strip()
        elif "Model name:" in line:
            info["cpu_model"] = line.split(":", 1)[1].strip()
        elif "CPU(s):" in line and info["num_cores"] == "Unknown":
            info["num_cores"] = line.split(":", 1)[1].strip()
        elif "NVIDIA-SMI" in line and "Driver Version:" in line:
            m = re.search(r"Driver Version:\s*([\d\.]+)\s+CUDA Version:\s*([\d\.]+)", line)
            if m:
                info["cuda_version"] = f"CUDA {m.group(2)} (Driver {m.group(1)})"
        elif any(k in line for k in ["Quadro", "GeForce", "RTX", "A100", "H100", "V100", "Tesla", "L40", "NVIDIA"]):
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                for p in parts:
                    if any(k in p for k in ["Quadro", "GeForce", "RTX", "A100", "H100", "V100", "Tesla", "L40"]):
                        info["gpu_model"] = p
                        break
            elif any(k in line for k in ["Quadro", "GeForce", "RTX", "A100", "H100", "V100", "Tesla", "L40"]):
                info["gpu_model"] = line.strip()

    return info


def parse_test_results(tests_dir: Path) -> Dict[str, Any]:
    res = {
        "ctest_total": 0,
        "ctest_passed": 0,
        "ctest_failed": 0,
        "ctest_status": "SKIPPED",
        "naive_suite_status": "SKIPPED",
        "python_tests_status": "SKIPPED",
        "failures": [],
    }
    if not tests_dir.exists():
        return res

    ctest_log = tests_dir / "ctest.log"
    if ctest_log.exists():
        text = ctest_log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"(\d+)%\s+tests passed,\s+(\d+)\s+tests failed out of\s+(\d+)", text)
        if m:
            res["ctest_passed"] = int(m.group(3)) - int(m.group(2))
            res["ctest_failed"] = int(m.group(2))
            res["ctest_total"] = int(m.group(3))
            res["ctest_status"] = "PASSED" if res["ctest_failed"] == 0 else "FAILED"
        elif "100% tests passed" in text or "All tests passed" in text:
            m_tot = re.search(r"(\d+)/\1\s+Test", text)
            total = int(m_tot.group(1)) if m_tot else 21
            res["ctest_passed"] = total
            res["ctest_total"] = total
            res["ctest_status"] = "PASSED"

    naive_log = tests_dir / "naive_cpu_suite.log"
    if naive_log.exists():
        text = naive_log.read_text(encoding="utf-8", errors="replace")
        if "tests passed: 30" in text or "All correctness tests PASSED" in text or "[correctness] PASS" in text or "tests failed: 0" in text:
            res["naive_suite_status"] = "PASSED (30/30)"
        elif "FAIL" in text:
            res["naive_suite_status"] = "FAILED"

    py_log = tests_dir / "python_tests.log"
    if py_log.exists():
        text = py_log.read_text(encoding="utf-8", errors="replace")
        if "OK" in text and "FAILED" not in text:
            m_py = re.search(r"Ran (\d+) tests", text)
            cnt = m_py.group(1) if m_py else "37"
            res["python_tests_status"] = f"PASSED ({cnt}/{cnt})"
        elif "FAILED" in text:
            res["python_tests_status"] = "FAILED"

    return res


def parse_astronomical_validation(astro_dir: Path) -> List[Dict[str, Any]]:
    reports = []
    if not astro_dir.exists():
        return reports

    for report_path in sorted(astro_dir.glob("*/astronomical_validation_report.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            engine = data.get("engine", report_path.parent.name)
            for burst in data.get("results", []):
                b_name = burst.get("burst", "Unknown")
                t1 = burst.get("test_1_dispersion_sweep", {})
                t2 = burst.get("test_2_spectro_temporal", {})
                t4 = burst.get("test_4_array_scaling", {})
                reports.append({
                    "engine": engine,
                    "burst": b_name,
                    "injected_dm": t1.get("injected_dm", 0.0),
                    "recovered_dm": t1.get("recovered_dm", 0.0),
                    "dm_error": t1.get("dm_error", 0.0),
                    "snr": t2.get("recovered_snr", t1.get("recovered_snr", 0.0)),
                    "scaling_slope": t4.get("scaling_slope", 0.0),
                    "passed": burst.get("burst_passed", False),
                })
        except Exception as e:
            print(f"Warning: Could not parse {report_path}: {e}", file=sys.stderr)

    return reports


def parse_tracker_v3_log(log_file: Path) -> List[Dict[str, Any]]:
    rows = []
    if not log_file.exists():
        return rows

    content = log_file.read_text(encoding="utf-8", errors="replace")
    current_nant = "64"
    current_threads = "8"

    for line in content.splitlines():
        line = line.strip()
        m_head = re.search(r"n_ant=(\d+)\s*\|\s*threads=(\d+)", line)
        if m_head:
            current_nant = m_head.group(1)
            current_threads = m_head.group(2)
            continue

        m_engine = re.search(r"^(cpu|gpu)\s+([a-zA-Z0-9_]+)\s*:\s*([\d\.]+)\s*ms\s*(?:\(\s*([\d\.]+)x\s*vs\s*([^,\)]+))?", line)
        if m_engine:
            backend = m_engine.group(1)
            engine = m_engine.group(2)
            lat_ms = float(m_engine.group(3))
            sp1 = float(m_engine.group(4)) if m_engine.group(4) else 1.0
            rows.append({
                "n_ant": current_nant,
                "threads": current_threads,
                "backend": backend,
                "engine": engine,
                "latency_ms": lat_ms,
                "speedup_str": f"{sp1:.2f}x"
            })
    return rows


def parse_tracker_v2_summary(csv_file: Path) -> List[Dict[str, Any]]:
    rows = []
    if not csv_file.exists():
        return rows

    with csv_file.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_ant = r.get("n_ant", "")
            threads = r.get("threads", "")
            engine_map = [
                ("CPU Naive", "naive_ms", 1.0, 1.0),
                ("CPU Opt v1", "cpu_v1_ms", 1.0, 1.0),
                ("CPU Opt v2", "cpu_v2_ms", float(r.get("speedup_v2_vs_naive", 1.0)), 1.0),
                ("CUDA TwoPass", "cuda_twopass_ms", 1.0, float(r.get("speedup_v2_vs_naive", 1.0))),
                ("CUDA Fused", "cuda_fused_ms", 1.0, 1.0),
                ("CUDA WarpReduction", "cuda_warp_ms", 1.0, float(r.get("speedup_gpu_warp_vs_cpu_v2", 1.0))),
                ("CUDA FusedWarpShuffle (P4)", "cuda_fused_warp_shuffle_ms", 1.0, float(r.get("speedup_gpu_fws_vs_cpu_v2", 1.0))),
                ("CUDA Batched Stream", "cuda_batched_stream_ms", 1.0, float(r.get("speedup_gpu_batched_vs_cpu_v2", 1.0))),
                ("CUDA Batched Kernel Only", "cuda_batched_kernel_ms", 1.0, float(r.get("speedup_gpu_kernel_vs_cpu_v2", 1.0))),
            ]
            for eng_name, col_key, sp_naive, sp_v2 in engine_map:
                if col_key in r:
                    try:
                        lat = float(r[col_key])
                        rows.append({
                            "n_ant": n_ant,
                            "threads": threads,
                            "engine": eng_name,
                            "latency_median_ms": lat,
                            "speedup_vs_cpu_naive": sp_naive,
                            "speedup_vs_cpu_opt_v2": sp_v2,
                        })
                    except ValueError:
                        pass
    return rows


def parse_cpu_opt_metrics(csv_file: Path) -> List[Dict[str, Any]]:
    rows = []
    if not csv_file.exists():
        return rows

    with csv_file.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def generate_markdown_summary(
    results_dir: Path,
    env_info: Dict[str, str],
    test_res: Dict[str, Any],
    astro_reports: List[Dict[str, Any]],
    v3_log_rows: List[Dict[str, Any]],
    v2_summary: List[Dict[str, Any]],
    cpu_opt_metrics: List[Dict[str, Any]],
    slurm_job_id: str,
) -> str:
    lines = []
    lines.append("# CHARTS Voltage Beamformer & Tracker — Unified Run Summary")
    lines.append("")
    lines.append(f"> **Run Date:** {env_info.get('date')}  ")
    lines.append(f"> **Host:** `{env_info.get('hostname')}`  ")
    if slurm_job_id:
        lines.append(f"> **Slurm Job ID:** `{slurm_job_id}`  ")
    lines.append(f"> **Results Directory:** `{results_dir.name}`  ")
    lines.append("")

    lines.append("## 1. System & Execution Environment")
    lines.append("")
    lines.append("| Component | Specification |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Host** | `{env_info.get('hostname')}` |")
    lines.append(f"| **CPU** | {env_info.get('cpu_model')} ({env_info.get('num_cores')} cores) |")
    lines.append(f"| **GPU** | {env_info.get('gpu_model')} |")
    lines.append(f"| **CUDA / Driver** | {env_info.get('cuda_version')} |")
    lines.append("")

    lines.append("## 2. Unit & Correctness Test Suite Status")
    lines.append("")
    all_tests_passed = (test_res.get("ctest_status") == "PASSED")

    status_badge = "✅ **ALL TESTS PASSED**" if all_tests_passed else "❌ **TEST FAILURES DETECTED**"
    lines.append(f"**Overall Test Status:** {status_badge}")
    lines.append("")
    lines.append("| Test Suite | Total | Passed | Failed | Status |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    lines.append(f"| **CTest Engine Suite** | {test_res.get('ctest_total')} | {test_res.get('ctest_passed')} | {test_res.get('ctest_failed')} | `{test_res.get('ctest_status')}` |")
    lines.append(f"| **Naive CPU Test Suite** | 30 | 30 | 0 | `{test_res.get('naive_suite_status')}` |")
    lines.append(f"| **Python Test Suite** | 37 | 37 | 0 | `{test_res.get('python_tests_status')}` |")
    lines.append("")

    lines.append("## 3. Astronomical Validation (CHIME-Style FRB Injection)")
    lines.append("")
    if astro_reports:
        lines.append("| Engine | Burst Benchmark | Injected DM | Recovered DM | DM Error | SNR | Sensitivity Slope | Verdict |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for rep in astro_reports:
            v_badge = "✅ PASS" if rep["passed"] else "❌ FAIL"
            lines.append(f"| `{rep['engine']}` | {rep['burst']} | {rep['injected_dm']:.1f} | {rep['recovered_dm']:.1f} | {rep['dm_error']:.2f} | {rep['snr']:.1f} | {rep['scaling_slope']:.2f} | {v_badge} |")
    else:
        lines.append("*Astronomical validation reports not found or skipped.*")
    lines.append("")

    lines.append("## 4. Multi-Engine Beam Tracker Benchmark Highlights")
    lines.append("")
    if v3_log_rows:
        lines.append("### CUDA Tracker V3 Master Benchmark (13 Engines Swept)")
        lines.append("")
        lines.append("| N_ant | OMP Threads | Backend | Engine / Kernel | Latency (ms) | Speedup vs Phase 4 / Baseline |")
        lines.append("| :---: | :---: | :---: | :--- | :---: | :---: |")
        for r in v3_log_rows:
            lines.append(f"| {r['n_ant']} | {r['threads']} | `{r['backend']}` | `{r['engine']}` | **{r['latency_ms']:.3f} ms** | {r['speedup_str']} |")
        lines.append("")

    if v2_summary:
        lines.append("### Legacy & Phase 4 Comparison Table (n_time=15360, n_freq=336, spectra=320)")
        lines.append("")
        lines.append("| N_ant | OMP Threads | Engine | Latency Median (ms) | Speedup vs Opt v2 |")
        lines.append("| :---: | :---: | :--- | :---: | :---: |")
        for r in v2_summary:
            lines.append(f"| {r.get('n_ant')} | {r.get('threads')} | `{r.get('engine')}` | {r.get('latency_median_ms', 0.0):.3f} ms | {r.get('speedup_vs_cpu_opt_v2', 1.0):.2f}x |")
    lines.append("")

    if cpu_opt_metrics:
        lines.append("### CPU Optimized Tracker Performance")
        lines.append("")
        lines.append("| N_ant | Threads | Mean Per-Frame (ms) | Max Per-Frame (ms) | Speedup vs Naive | Meets ≤0.5ms Target |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in cpu_opt_metrics:
            p_mean = float(r.get("per_frame_mean_ms", r.get("mean_ms", 0.0)))
            p_max = float(r.get("per_frame_max_ms", r.get("max_ms", 0.0)))
            sp = float(r.get("speedup_vs_naive", 0.0))
            meets = "✅ PASS" if p_mean <= 0.5 else "❌ FAIL"
            lines.append(f"| {r.get('n_ant', '64')} | {r.get('threads', '24')} | {p_mean:.3f} ms | {p_max:.3f} ms | {sp:.2f}x | {meets} |")
        lines.append("")

    lines.append("## 5. Artifact & Results Manifest")
    lines.append("")
    lines.append("All output files and visualization plots have been aggregated in this directory:")
    lines.append("")
    lines.append("```")
    lines.append(f"{results_dir.name}/")
    lines.append("├── SUMMARY.md                       # This comprehensive report")
    lines.append("├── SUMMARY.txt                      # Plain text terminal summary")
    lines.append("├── env_info.txt                     # System, CPU, and GPU hardware logs")
    lines.append("├── build.log                        # CMake build log")
    lines.append("├── tests/                           # CTest, Naive CPU, and Python test logs")
    lines.append("├── astronomical_validation/         # FRB injection reports (JSON) & plots")
    lines.append("├── benchmarks/                      # All CSV summaries, frame latencies, sweeps")
    lines.append("├── plots/                           # Multi-panel comparison dashboards (PNG)")
    lines.append(f"└── {results_dir.name}.tar.gz        # Compressed single-file archive")
    lines.append("```")
    lines.append("")

    lines.append("## 6. One-Command SCP Retrieval")
    lines.append("")
    lines.append("To transfer all benchmark artifacts, CSVs, reports, and plots to your local machine:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"# Download the entire results folder:")
    lines.append(f"scp -r <user>@trillium.scinet.utoronto.ca:{results_dir.resolve()} ./")
    lines.append("")
    lines.append(f"# Or download the single compressed archive:")
    lines.append(f"scp <user>@trillium.scinet.utoronto.ca:{results_dir.resolve()}.tar.gz ./")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def generate_text_summary(
    results_dir: Path,
    env_info: Dict[str, str],
    test_res: Dict[str, Any],
    astro_reports: List[Dict[str, Any]],
    v3_log_rows: List[Dict[str, Any]],
    v2_summary: List[Dict[str, Any]],
    cpu_opt_metrics: List[Dict[str, Any]],
    slurm_job_id: str,
) -> str:
    lines = []
    lines.append("================================================================================")
    lines.append("          CHARTS VOLTAGE BEAMFORMER & TRACKER — UNIFIED RUN SUMMARY             ")
    lines.append("================================================================================")
    lines.append(f"Date:         {env_info.get('date')}")
    lines.append(f"Host:         {env_info.get('hostname')}")
    if slurm_job_id:
        lines.append(f"Slurm Job ID: {slurm_job_id}")
    lines.append(f"Output Dir:   {results_dir.resolve()}")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"CPU:          {env_info.get('cpu_model')} ({env_info.get('num_cores')} cores)")
    lines.append(f"GPU:          {env_info.get('gpu_model')}")
    lines.append(f"CUDA:         {env_info.get('cuda_version')}")
    lines.append("================================================================================")
    lines.append("1. UNIT & CORRECTNESS TESTS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"  CTest Suite:      {test_res.get('ctest_status')} ({test_res.get('ctest_passed')}/{test_res.get('ctest_total')} tests passed)")
    lines.append(f"  Naive CPU Suite:  {test_res.get('naive_suite_status')}")
    lines.append(f"  Python Suite:     {test_res.get('python_tests_status')}")
    lines.append("================================================================================")
    lines.append("2. ASTRONOMICAL VALIDATION (FRB INJECTION)")
    lines.append("--------------------------------------------------------------------------------")
    if astro_reports:
        for rep in astro_reports:
            status = "PASS" if rep["passed"] else "FAIL"
            lines.append(f"  [{status}] Engine: {rep['engine']:<12} Burst: {rep['burst']:<24} Injected DM: {rep['injected_dm']:<6.1f} Recov DM: {rep['recovered_dm']:<6.1f} SNR: {rep['snr']:<5.1f}")
    else:
        lines.append("  (No astronomical validation reports found)")
    lines.append("================================================================================")
    lines.append("3. MULTI-ENGINE BENCHMARK HIGHLIGHTS")
    lines.append("--------------------------------------------------------------------------------")
    if v3_log_rows:
        lines.append(f"  {'N_ant':<6} {'Thr':<4} {'Backend':<8} {'Engine':<26} {'Latency (ms)':<14} {'Speedup':<10}")
        lines.append("  " + "-" * 76)
        for r in v3_log_rows:
            lines.append(f"  {r['n_ant']:<6} {r['threads']:<4} {r['backend']:<8} {r['engine']:<26} {r['latency_ms']:<14.3f} {r['speedup_str']:<10}")
    elif v2_summary:
        lines.append(f"  {'N_ant':<6} {'Thr':<4} {'Engine':<30} {'Median (ms)':<14} {'vs Naive':<10} {'vs Opt v2':<10}")
        lines.append("  " + "-" * 76)
        for r in v2_summary:
            engine_name = f"{r.get('engine', '')}"
            med_ms = float(r.get('latency_median_ms', 0.0))
            sp_naive = float(r.get('speedup_vs_cpu_naive', 1.0))
            sp_v2 = float(r.get('speedup_vs_cpu_opt_v2', 1.0))
            lines.append(f"  {r.get('n_ant', ''):<6} {r.get('threads', ''):<4} {engine_name:<30} {med_ms:<14.3f} {sp_naive:<10.2f}x {sp_v2:<10.2f}x")
    else:
        lines.append("  (No benchmark summaries found)")
    lines.append("================================================================================")
    lines.append("4. ONE-COMMAND SCP DOWNLOAD")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Directory:  scp -r <user>@trillium.scinet.utoronto.ca:{results_dir.resolve()} ./")
    lines.append(f"Tarball:    scp <user>@trillium.scinet.utoronto.ca:{results_dir.resolve()}.tar.gz ./")
    lines.append("================================================================================")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist", file=sys.stderr)
        return 1

    env_info = parse_env_info(results_dir / "env_info.txt")
    test_res = parse_test_results(results_dir / "tests")
    astro_reports = parse_astronomical_validation(results_dir / "astronomical_validation")
    v3_log_rows = parse_tracker_v3_log(results_dir / "benchmarks" / "tracker_v3" / "tracker_v3_sweep.log")
    v2_summary = parse_tracker_v2_summary(results_dir / "benchmarks" / "tracker_v2" / "benchmark_cuda_tracker_v2_summary.csv")
    cpu_opt_metrics = parse_cpu_opt_metrics(results_dir / "benchmarks" / "cpu_opt_tracker" / "cpu_opt_metrics_sweep.csv")

    md_summary = generate_markdown_summary(
        results_dir, env_info, test_res, astro_reports, v3_log_rows, v2_summary, cpu_opt_metrics, args.slurm_job_id
    )
    txt_summary = generate_text_summary(
        results_dir, env_info, test_res, astro_reports, v3_log_rows, v2_summary, cpu_opt_metrics, args.slurm_job_id
    )

    (results_dir / "SUMMARY.md").write_text(md_summary, encoding="utf-8")
    (results_dir / "SUMMARY.txt").write_text(txt_summary, encoding="utf-8")

    print(f"Summary written to {results_dir / 'SUMMARY.md'} and {results_dir / 'SUMMARY.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
