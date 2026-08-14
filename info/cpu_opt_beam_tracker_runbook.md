# Optimized CPU Beam Tracker — Tests, Benchmark & Plots Runbook

> Companion to [`info/cpu_opt_beam_tracker.md`](cpu_opt_beam_tracker.md) (the
> algorithmic spec) and [`src/cpu_opt_beam_tracker.cpp`](../src/cpu_opt_beam_tracker.cpp)
> (the implementation). This file is the **operator runbook**: build, run the
> extensive unit tests, run the per-frame benchmark against the
> **< 0.5 ms / frame** kernel objective, and produce the plots — all on the
> trillium login/compute nodes. **Do not run anything from this repo on the
> development machine**; everything below is meant to be executed on trillium.

---

## 1. What this stage adds

| Artifact | Purpose |
|---|---|
| [`tests/test_cpu_opt_beam_tracker.cpp`](../tests/test_cpu_opt_beam_tracker.cpp) | Extensive unit tests (13 blocks): construction/validation, back-compat equivalence to the naive tracker, free-function + stateful mirrors, Bartlett DOA recovery, Capon estimator, spatial smoothing, forgetting factor, quadratic interpolation, multi-window adaptive tracking, n_ant ∈ {32, 64}, coarse-grid argmax sanity, byte-layout + aligned-vs-misaligned energy. |
| [`tools/benchmark_cpu_opt_beam_tracker.cpp`](../tools/benchmark_cpu_opt_beam_tracker.cpp) | Benchmark: naive vs optimized end-to-end wall time, **per-frame kernel latencies** (min/mean/median/p95/max) read from the `BEAMFORMER_TRACKER_PERF`-guarded accumulator, DOA-recovery check, PASS/FAIL against the 0.5 ms/frame target. Emits a summary CSV (`--metrics`) and a per-frame CSV (`--frames`). |
| [`tools/plot_cpu_opt_beam_tracker.py`](../tools/plot_cpu_opt_beam_tracker.py) | Dashboard PNG: per-frame latency vs the 0.5 ms target, multi-config summary bars (PASS/FAIL), DOA tracking sky panel (true source / open-loop prior / optimized estimate), direction-error curve, optional optimized intensity heatmap. |
| [`src/cpu_opt_beam_tracker.cpp`](../src/cpu_opt_beam_tracker.cpp) (edited) | Removed the dead/placeholder `capon_power` (which returned `bartlett_power(R, w)` with the wrong vector); kept the correct `capon_power_correct`. Added a zero-overhead per-frame timing hook behind `BEAMFORMER_TRACKER_PERF`. |
| [`CMakeLists.txt`](../CMakeLists.txt) (edited) | Added `src/cpu_opt_beam_tracker.cpp` to `beamformer_core`; registered the test and a separate `beamformer_core_perf` static lib (perf macro + `-O3`) used only by the benchmark so production builds keep zero timing overhead. |

---

## 2. Build (trillium)

Plain CMake / Make, no CUDA required for this stage (the tracker is CPU-only):

```bash
# from the repo root on trillium
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEAMFORMER_ENABLE_CUDA=OFF
cmake --build build -j
```

> The `beamformer_core_perf` target (used by the benchmark) forces `-O3` and
> `-DBEAMFORMER_TRACKER_PERF` regardless of `CMAKE_BUILD_TYPE`, so the
> per-frame numbers are measured at release optimisation. The default
> `beamformer_core` keeps the perf macro **off** → zero timing overhead for
> the CLI / tests / production.

---

## 3. Run the extensive tests

```bash
ctest --test-dir build -R cpu_opt_beam_tracker --output-on-failure
# or directly:
./build/test_cpu_opt_beam_tracker
```

The test binary is assertion-based (no external framework), mirroring
[`tests/test_beam_tracker.cpp`](../tests/test_beam_tracker.cpp). It exits `0`
on success and aborts on the first failing assert with a core file. The 13
blocks run on synthetic point sources (deterministic, no fixtures on disk).

### What each block asserts

1. **Construction / validation** — all `CpuOptTrackerConfig` rejections
   (n_beams != 1, wrong position/frequency counts, integration_spectra == 0,
   forgetting_factor ∉ (0,1], smoothing > n_ant) throw `std::invalid_argument`.
2. **Back-compat** — default config (`coarse_grid_resolution = 1`) reproduces
   the naive `beam_tracker_cpu_packed_intensity` output **byte-equal**
   (`naive == opt`). This is the spec's regression anchor.
3. **Free-function mirrors** — byte-equal to naive for a drifting trajectory
   across two windows (per-window directions exercised).
4. **Stateful class** — persists direction estimates across `run_into` calls;
   `window_direction(0)` equals the trajectory prior when scanning is disabled.
5. **Bartlett DOA recovery** — estimated direction closer to the true source
   than the open-loop prior, and within one coarse cell (O1+O3+O4+O6).
6. **Capon estimator** — MVDR branch runs end-to-end and recovers the DOA.
7. **Spatial smoothing (O2)** — coherent source (the rank-1 case the spec calls
   out) does not crash and still recovers the DOA.
8. **Forgetting factor (O5)** — λ = 1.0 (block estimate) vs λ = 0.9 (adaptive);
   both recover the DOA; output sizes are byte-compatible regardless of λ.
9. **Quadratic interpolation (O4)** — interp-on vs interp-off for an off-grid
   source; interp is no worse than the raw argmax.
10. **Multi-window tracking** — a slowly moving source recovered across
    windows; the estimate at window 1 beats the zero-drift prior.
11. **n_ant == 64** — the 8×8 geometry runs the optimized path and recovers DOA.
12. **Coarse-grid argmax sanity** — a zenith source peaks in the central 3×3.
13. **Byte-layout + aligned-vs-misaligned energy** — output size ==
    `n_time*n_freq*1`, all intensities ≥ 0, aligned energy > misaligned energy
    (the carried-over `aligned_total > misaligned_total` invariant).

### 3.1. Dumping search state to debug a failing DOA-recovery assertion

If a `err_est < err_prior` assert fires (the tracker's estimated window
direction is *not* closer to the true source than the open-loop prior —
the symptom of a broken Bartlett/Capon peak, a steering/sign error, a
smoothing bug, or a quadratic-interpolation inversion), you can ask the
tracker to capture the entire coarse-to-fine search and write it to disk.

1. Configure with the capture enabled (adds
   `-DBEAMFORMER_TRACKER_DEBUG` to the default `beamformer_core` TU and
   the test target; the perf-only `beamformer_core_perf` used by the
   benchmark is left untouched, so its per-frame timings stay clean):

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
         -DBEAMFORMER_ENABLE_CUDA=OFF \
         -DBEAMFORMER_TRACKER_DEBUG=ON
   cmake --build build -j
   ```

2. (Re)run the test. Each DOA-recovery block (Bartlett / Capon / smoothing)
   calls [`CpuOptBeamTracker::debug_search_dump`](../src/cpu_opt_beam_tracker.cpp)
   *before* its assertion, so you get the dump even if the assert then
   aborts:

   ```bash
   ./build/test_cpu_opt_beam_tracker
   # stdout: [block5 bartlett] true=(...) est=(...) err_prior=... err_est=... est<=NO
   #         [tracker] debug dump written to ./tracker_dumps (block5_bartlett)
   # then aborts at the assert, leaving the on-disk dump behind.
   ```

   Point the dumps elsewhere with the env var:

   ```bash
   BEAMFORMER_TRACKER_DUMP_DIR=build/debug_dumps ./build/test_cpu_opt_beam_tracker
   ```

3. Inspect the dump. Each run writes a timestamped subdirectory
   `<dir>/<label>_<n>/` containing:

   - `README.txt` / `config.txt` — config knobs, `M_eff`, seeded prior, file
     layout reference.
   - `positions.csv`, `frequencies.csv` — array geometry / channel plan.
   - `estimates.csv` — per-window prior centre vs final estimate (l,m,n).
   - `coarse_<w>.csv` — per-window coarse grid cell directions + integrated
     Bartlett/Capon power (row-major `G×G`), so the argmax cell and the
     peak shape relative to the true source are immediately visible.
   - `refine_<w>_<lvl>.csv` — per-window per refinement level 3×3 candidate
     dirs + integrated power (the level argmax + O4 input).
   - `R_<w>_<f>.bin` — per-window per-frequency `M_eff×M_eff` Hermitian
     covariance (post O2 smoothing + O5 fold + Capon diagonal load), raw
     `std::complex<float>` row-major.
   - `snaps_<w>_<f>.bin` — per-window per-frequency `K` decoded length-`n_ant`
     snapshots (pre-smoothing), so the source phase / amplitude can be
     re-derived independently.
   - `shapes_<w>.txt` — `M_eff`, freq count, snapshot counts so the binary
     files are interpretable.
   - `DUMP_COMPLETE` — sentinel present when the dump finished.

   Typical diagnosis flow: load `coarse_0.csv` in a spreadsheet/Python and
   confirm the power peak lands near the true `(l,m)`; if it doesn't, the
   bug is in the covariance (`R_*.bin`) or the steering convention —
   reconstruct the expected peak from `snaps_*.bin` + `positions.csv` and
   compare. If the coarse peak is correct but `estimates.csv` is wrong,
   the bug is in refinement (`refine_*.csv`) or the O4 quadratic
   interpolation denominator.

> When `BEAMFORMER_TRACKER_DEBUG` is **off** (the default), the capture
> members and the dump method are compiled out — `debug_search_dump` is a
> no-op stub and the test prints a one-line "built WITHOUT debug capture"
> reminder, so production / benchmark builds carry zero overhead.

---

## 4. Run the benchmark (the < 0.5 ms/frame objective)

```bash
# Default production-sized grid: n_time=15360, n_freq=336, n_ant=64.
./build/benchmark_cpu_opt_beam_tracker \
    --n-time 15360 --n-ant 64 --integration-spectra 320 \
    --coarse-grid-resolution 12 --refinement-levels 2 \
    --source-l0 0.03 --source-dl 1e-5 --prior-l0 0.0 \
    --metrics build/bench/metrics.csv \
    --frames  build/bench/frames.csv \
    --intensity-out build/bench/opt_intensity.bin
```

Stdout reports (example shape — real numbers from trillium):

```
=== CPU optimized beam tracker benchmark ===
grid: n_time=15360 n_freq=336 n_ant=64 integration_spectra=320  windows=48
search: coarse=12 refine=2 fov=(0.2,0.2) estimator=bartlett lambda=1 smoothing=0
naive  end-to-end: <X> ms
opt    end-to-end: <Y> ms  (speedup <Z>x vs naive)
per-frame kernel latencies (ms):
  min    = ...
  mean   = ...
  median = ...
  p95    = ...
  max    = ...
  frames measured = 48
TARGET per-frame <= 0.5 ms : PASS  (worst of p95/max = ... ms)
DOA recovery (mean (l,m) error over windows):
  prior (open-loop) = ...
  optimized estimate= ...
  improvement       = yes
```

The headline number is **`worst of (p95, max)` ≤ 0.5 ms** → `PASS`. Tune the
search knobs to explore the speed/accuracy frontier:

| Knob | Cheaper | More accurate |
|---|---|---|
| `--coarse-grid-resolution` | ↓ | ↑ |
| `--refinement-levels` | ↓ (0 = coarse interp only) | ↑ |
| `--capon` | (off, Bartlett) | on (MVDR, + a Cholesky solve per cell) |
| `--forgetting-factor` | — | ↓ tracks faster (noisier) |
| `--smoothing-subarray` | (0 = no smoothing) | >0 robust to coherent sources |

### Useful sweeps (append rows to the same metrics CSV)

```bash
# Cheap baseline: coarse grid, no refinement.
./build/benchmark_cpu_opt_beam_tracker --coarse-grid-resolution 8 --refinement-levels 0 \
    --metrics build/bench/metrics.csv --frames build/bench/frames_coarse8_r0.csv

# High-resolution: 16-cell coarse, 3 refinement levels.
./build/benchmark_cpu_opt_beam_tracker --coarse-grid-resolution 16 --refinement-levels 3 \
    --metrics build/bench/metrics.csv --frames build/bench/frames_fine16_r3.csv

# Capon (MVDR) for higher angular resolution.
./build/benchmark_cpu_opt_beam_tracker --capon --diagonal-load 1e-2 \
    --metrics build/bench/metrics.csv --frames build/bench/frames_capon.csv

# 32-ant array.
./build/benchmark_cpu_opt_beam_tracker --n-ant 32 \
    --metrics build/bench/metrics.csv --frames build/bench/frames_nant32.csv
```

---

## 5. Plots

```bash
# Single-run dashboard (per-frame latency + DOA).
python tools/plot_cpu_opt_beam_tracker.py \
    --frames  build/bench/frames.csv \
    --metrics build/bench/metrics.csv \
    --intensity build/bench/opt_intensity.bin \
    --n-time 15360 --n-ant 64 --integration-spectra 320 \
    --source-l0 0.03 --source-dl 1e-5 --prior-l0 0.0 \
    --output build/bench/dashboard.png
```

Panels:
1. **Per-frame kernel latency** — blue line vs the red 0.5 ms target, with
   mean / p95 reference lines.
2. **Multi-config summary bars** (when `--metrics` has multiple rows) —
   min/mean/p95/max per config, coloured PASS/FAIL.
3. **Direction-error vs true source** — true drift, prior error, estimate
   error (lower is better).
4. **Optimized intensity [time, frequency] heatmap** (dB) when `--intensity`
   is supplied, else the DOA sky panel (horizon + true source + prior + estimate)

> The plotter imports the shared physical helpers from
> [`tools/plot_results.py`](../tools/plot_results.py) and the trajectory model
> from [`tools/plot_tracker_results.py`](../tools/plot_tracker_results.py) so
> conventions stay identical to the fixed-grid and naive-tracker dashboards.
> Requires only `numpy` + `matplotlib` (the trillium `python` already has them).

---

## 6. If a frame exceeds 0.5 ms

The algorithmic spec's deferred items (SIMD/AVX, SoA packing,
frequency-axis threading, FFT coarse scan) are the documented next levers.
Before going there, the cheapest algorithmic moves from the runbook above are:

1. Drop `--refinement-levels` first (3→3×3→constant per level cut).
2. Drop `--coarse-grid-resolution` (scan cost is `Θ(n_freq * G^2)`).
3. Keep `--capon` off unless you need the resolution (Capon adds a Cholesky
   solve per scanned cell per frequency).
4. Enable `--forgetting-factor 1.0` for a stationary source (matches the
   block estimate — no extra blending math).

The `BEAMFORMER_TRACKER_PERF` per-frame vector will pinpoint whether the
covariance update (`O5`) or the scan (`O3+O6`) dominates.
