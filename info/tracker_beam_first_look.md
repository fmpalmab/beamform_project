# CHARTS Tracker Beam — First Look, Plan, and v0 Realization

> This note has two parts:
>
> 1. The original **first-look and plan** (sections 1–8) — kept verbatim for history.
> 2. The **realized v0** addendum below (section 9 onward), describing the components that
>    were actually implemented, the resolved design decisions, the build/test status, the CLIs,
>    the visualization workflow, and an end-to-end example.
>
> The v0 implementation follows the plan's **Strategy A** (blockwise re-launch over the single
> tracked beam) and adds `beam_tracker_*` files without modifying any existing tested component.
>
> ---
>
> **Original exploratory note (kept verbatim).** This section records what the cloned
> "normal" beamforming repository gives us as a base, and the path it proposed to add a
> **tracker beam** (a single beam whose pointing direction follows a tracked source over time)
> on top of it.

---

## 1. What a "tracker beam" means here

The current repository is a **fixed-grid voltage beamformer**: it precomputes a static set
of beam directions (a `Vec3` per beam), generates weights for all of them up front, and
every time sample is beamformed against the *whole* grid. The output is per-beam intensity.

A **tracker beam** is conceptually different:

- There is **one** beam (or a small number) whose direction `(l, m)` changes as a function
  of time `t`.
- The weights are therefore **time-dependent**: `w[b][f][a]` becomes
  `w[beam][f][a][t]` (explicit) or recomputed on the fly from a `direction(t)` function.
- The output is a single `intensity[t][f]` stream (or `[t][f]` per tracked beam), not a
  `[t][f][b]` grid.

So the two key deltas from the base are:

1. **Time-varying steer direction** → time-varying weights (or per-time recompute).
2. **Single-beam (or few-beam) output** instead of a `[t][f][B]` cube.

This maps cleanly onto the existing geometry/weight-generation machinery; the new work is
mostly in **pipeline shape** (one beam, per-time weights) and in **steering logic** (the
tracker `direction(t)` and weight-update cadence).

---

## 2. What the cloned repository gives us

### 2.1 Build system and layout

[`CMakeLists.txt`](CMakeLists.txt:1) defines:

- `beamformer_core` — host library (CPU reference, geometry, weights, quantization, IO,
  synthetic data, temporal integration).
- `beamformer_cuda_core` — CUDA kernels (only built when `nvcc` is detected).
- Tools: `generate_fake_data`, `generate_weights`, `beamformer_cpu`, `beamformer_cuda`,
  `benchmark_cpu_cuda`, `benchmark_cuda_quantized`.
- CTest suite covering contracts, geometry, point-source physics, packed int4, temporal
  integration, int8 quantization, and the offline frame runner.

`CMAKE_CXX_STANDARD 17`, `CMAKE_CUDA_STANDARD 17`, hard warnings
(`-Wall -Wextra -Wpedantic`). CUDA is optional and gracefully degrades to CPU-only.

### 2.2 Directory map

| Area | Files | Role |
|------|-------|------|
| Config/contract | [`include/beamformer/config.hpp`](include/beamformer/config.hpp:1) | `Dimensions`, `ShardDescriptor`, 336×2 channel layout, antenna 32/64, ≤128 beams |
| Indexing | [`include/beamformer/indexing.hpp`](include/beamformer/indexing.hpp:1) | `voltage_index`, `weight_index`, `intensity_index`, RFSoC element reorder |
| Geometry | [`include/beamformer/geometry.hpp`](include/beamformer/geometry.hpp:1), [`src/geometry.cpp`](src/geometry.cpp:1) | array positions, FFT-bin beam grids, hex FoV helpers, file loaders |
| Physics | [`include/beamformer/physics.hpp`](include/beamformer/physics.hpp:1) | `speed_of_light_m_per_s`, `two_pi` |
| Weights | [`include/beamformer/weights.hpp`](include/beamformer/weights.hpp:1), [`src/weights.cpp`](src/weights.cpp:1) | `generate_weights`, `generate_tiled_weights` |
| Formats | [`include/beamformer/formats.hpp`](include/beamformer/formats.hpp:1) | `PackedVoltage`, `Weights`, `Intensities`, byte-size contracts |
| CPU beamformer | [`include/beamformer/cpu_beamformer.hpp`](include/beamformer/cpu_beamformer.hpp:1) | `cpu_beamform_packed_intensity` (int4 decode in-loop) |
| CUDA beamformer | [`include/beamformer/cuda_beamformer.hpp`](include/beamformer/cuda_beamformer.hpp:1), [`src/cuda_beamformer.cu`](src/cuda_beamformer.cu:1) | `Direct` and `Tiled` kernels, workspace, integrated + int8 paths |
| CUDA stage API | [`include/beamformer/cuda_stage_api.hpp`](include/beamformer/cuda_stage_api.hpp:1) | stream-launched kernels on caller-owned device buffers |
| CUDA frame views | [`include/beamformer/cuda_frame.hpp`](include/beamformer/cuda_frame.hpp:1) | non-owning `CudaFrameView`/`CudaBufferView` descriptors |
| Offline runner | [`include/beamformer/cuda_offline_runner.hpp`](include/beamformer/cuda_offline_runner.hpp:1) | one-frame convenience layer over `run_device_frame` |
| Temporal integration | [`include/beamformer/temporal_integration.hpp`](include/beamformer/temporal_integration.hpp:1) | 10/320 spectra configs, CPU reference |
| Quantization | [`include/beamformer/quantization.hpp`](include/beamformer/quantization.hpp:1) | CHIME-style int8, per-chunk `(offset,scale)` |
| int4 packing | `include/beamformer/int4.hpp` | signed two's-complement int4x2 unpack |
| Synthetic data | [`include/beamformer/synthetic_data.hpp`](include/beamformer/synthetic_data.hpp:1), [`src/synthetic_data.cpp`](src/synthetic_data.cpp:1) | one-hot/constant/noise/point-source voltage |

### 2.3 The data contract (most important thing to internalize)

From [`include/beamformer/config.hpp`](include/beamformer/config.hpp:10) and
[`include/beamformer/indexing.hpp`](include/beamformer/indexing.hpp:10):

- Input voltage is **packed signed int4x2**, one byte per complex sample:
  `voltage[time][freq][element]`, `index = (t*n_freq + f)*n_ant + e`.
  Real in low nibble, imag in high nibble.
- Two shards of 336 channels each cover the 672-channel band; shards stay **separate**
  buffers (never concatenated on the wire or in H2D).
- Antennas: **32 or 64**. Default geometry: regular 4×8 or 8×8 grid, 0.6 m spacing.
- Frequency centers: 300 MHz start, 300 kHz/channel, design 400 MHz.
- Element order reproduces the RFSoC handler:
  `element = (1 - rfsoc_id)*32 + packet_element`
  (RFSoC 1 → 0..31, RFSoC 0 → 32..63).

### 2.4 Weight generation (what we will reuse heavily)

[`include/beamformer/weights.hpp`](include/beamformer/weights.hpp:11) exposes:

```cpp
Weights generate_weights(const Dimensions& dims,
                         const std::vector<Vec3>& positions_m,
                         const std::vector<float>& frequencies_hz,
                         const std::vector<Vec3>& beam_directions);
```

This is the geometric phase-steering `w[b][f][a] = exp(-j*2*pi*f*dot(pos[a], dir[b])/c)`.
For the tracker we keep **exactly** this formula, but:

- `beam_directions` becomes a **time sequence** `dir(t)`,
- we generate one beam's weights **per time block** (or per integration window).

### 2.5 The two CUDA kernels

From [`src/cuda_beamformer.cu`](src/cuda_beamformer.cu:108):

- **Direct kernel** (`direct_packed_voltage_beamformer_kernel`): one thread per
  `[time][freq][beam]` output; loops over antennas, decodes int4 in-place, accumulates,
  writes `|sum|^2`. Simple, weight layout `[beam][freq][ant]`.
- **Tiled kernel** (`tiled_packed_voltage_beamformer_kernel`): block = one frequency,
  `32` beams × `8` times; weights + voltage staged into shared memory; weight layout
  `[freq][beam_tile][ant][local_beam]`. Better throughput for many beams.

For a **single tracker beam** the Direct kernel's threading (`thread → (t,f,beam)`) is a
natural fit, with `n_beams = 1`. The Tiled kernel's 32-beam tile is overkill for one beam
but could be reused if we track several beams at once (e.g. a small constellation).

### 2.6 Temporal integration & quantization (already fused)

- CUDA fuses beamforming + temporal integration into one kernel, writing
  `[T_int][F][B]` directly (no `[T][F][B]` intermediate).
- int8 quantization (CHIME-style, 1×16×16 chunks) runs as a separate stage on the
  integrated float32 tensor.

For a tracker pipeline the same fusion idea applies: produce
`[T_int][F]` (single beam) with optional int8.

### 2.7 The device-frame / offline-runner abstraction

[`include/beamformer/cuda_frame.hpp`](include/beamformer/cuda_frame.hpp:40) and
[`include/beamformer/cuda_offline_runner.hpp`](include/beamformer/cuda_offline_runner.hpp:28)
define a **non-owning device buffer** contract (`CudaFrameView`) and a one-frame host
helper. This is the cleanest extension point for streaming tracker frames: a tracker frame
is "voltage in → one beam's intensity out" with a per-frame direction.

---

## 3. What needs to change / be added for a tracker beam

### 3.1 Conceptual split

| Concern | Existing base | Tracker need |
|---------|---------------|--------------|
| Beam directions | static `vector<Vec3>`, fixed at config time | `direction(t)` trajectory; per-time-block |
| Weights | `w[beam][freq][ant]`, uploaded once | `w[freq][ant]` per tracked beam, per time block |
| Output dims | `[T][F][B]` | `[T][F]` per tracked beam (few beams) |
| Kernel launch | once per whole grid | per tracked beam, possibly several times if direction changes mid-block |
| Integration | fused, `[T_int][F][B]` | fused, `[T_int][F]` per beam |
| CPU reference | full grid | single-beam path for validation |

### 3.2 Proposed new/modified components

1. **Tracker trajectory model** (new header, e.g. `include/beamformer/tracker.hpp`):
   - `struct TrackerConfig { Vec3 initial_direction; ... trajectory params ... }`.
   - `Vec3 tracker_direction(const TrackerConfig&, std::uint64_t frame_id, std::size_t t)`,
     or a coarser `tracker_direction_block(...)` returning one direction per time block.
   - Start simple: linear drift in `(l,m)`, or a constant direction (sanity), then add
     e.g. a circular track or a file-loaded ephemeris.

2. **Per-block weight generator** (reuse `generate_weights` with `n_beams=1`):
   - `Weights generate_tracker_weights(dims, positions, freqs, Vec3 dir)` — thin wrapper
     that calls the existing generator with a 1-element `beam_directions` vector.
   - Optionally a **device-side** weight generator so weights never round-trip through
     host for fast-changing directions.

3. **Single-beam CUDA kernel / entry point** (extend
   [`include/beamformer/cuda_beamformer.hpp`](include/beamformer/cuda_beamformer.hpp:98)):
   - `Intensities cuda_beamform_tracker_packed(...)` with `n_beams=1` and time-varying
     weights. Two strategies (see §4).
   - A `CudaBeamformerWorkspace` variant that accepts a **weight schedule** rather than a
     single resident weight tensor.

4. **Output format / indexing**:
   - Reuse `intensity_index` with `n_beams=1`, or add a 2-D `[time][freq]` helper to keep
     intent explicit and avoid accidental misindexing.

5. **CPU reference tracker** (extend
   [`include/beamformer/cpu_beamformer.hpp`](include/beamformer/cpu_beamformer.hpp:14)):
   - `cpu_beamform_tracker_packed_intensity` that loops time, recomputes weights per
     block, and accumulates one beam. Used as the numerical truth for GPU validation.

6. **Synthetic tracker source** (extend
   [`include/beamformer/synthetic_data.hpp`](include/beamformer/synthetic_data.hpp:1)):
   - A point source whose `(l,m)` moves with `t` so the tracker can be validated by
     "does the tracker beam's power stay high as the source moves?"

7. **CLI/tool**: `tools/beamformer_tracker.cpp` (and a `--tracker` mode on existing CLIs)
   plus a `generate_tracker_weights` or a runtime-steering option in `beamformer_cuda`.

8. **Tests**:
   - `tests/test_tracker.cpp` — trajectory, per-block weights, single-beam intensity vs
     CPU.
   - `tests/test_cuda_tracker.cpp` — GPU path matches CPU, and a moving-source recovery
     check (analogous to the existing `test_cuda_point_source`).

9. **CMake**:
   - Add the new headers to `beamformer_core`, new sources, new test/executable targets,
     guarded by CUDA where relevant (mirror the existing pattern in
     [`CMakeLists.txt`](CMakeLists.txt:48)).

### 3.3 What stays the **same** (do not rebuild)

- The packed int4 input contract and
  [`include/beamformer/int4.hpp`](include/beamformer/int4.hpp:1) decode.
- `Dimensions`, `ShardDescriptor`, the two-shard separation, element ordering — a tracker
  processes one shard at a time just like the grid beamformer.
- Array geometry and the phase-steering weight formula — only the *direction input*
  becomes time-dependent.
- Temporal integration fusion concept and int8 quantization chunking.
- The `CudaFrameView` device-buffer contract for the streaming path.

---

## 4. Key design decision: how time-varying weights meet the kernel

This is the central architectural choice. Three viable strategies, in order of complexity:

### 4.1 Strategy A — blockwise constant direction, re-launch per block (simplest)

- Split `n_time` into blocks (e.g. one integration window, or a few).
- For each block: compute `dir(block)`, generate `n_beams=1` weights, upload, launch the
  existing **Direct** kernel for that time block, accumulate output.
- Reuses **everything**; the only new code is the block loop and the direction function.
- Cost: one weight upload + one launch per block. Fine for slowly moving sources and
  coarse blocks (e.g. one direction per integration window → 48 launches for the default
  15360/320 config — cheap relative to one full-grid launch).
- **Recommended starting point.**

### 4.2 Strategy B — device-side weight generation (no host round-trip)

- Pass a compact track description (a few floats) to a small kernel that writes
  `[freq][ant]` weights into shared/global memory per block, then call the beamformer
  kernel on those weights.
- Avoids H2D weight transfers when direction changes every block. Worth it only if block
  granularity becomes small (e.g. per-spectrum steering).
- Add a `launch_tracker_weight_kernel` in
  [`include/beamformer/cuda_stage_api.hpp`](include/beamformer/cuda_stage_api.hpp:21).

### 4.3 Strategy C — fully fused time-varying kernel (most performant, most work)

- One kernel where each thread/block reads `dir(t)` from a precomputed device-side
  `direction[t]` table (or computes it analytically), computes weights on the fly in
  registers/shared, and accumulates. No separate weight tensor at all.
- Best throughput and lowest memory traffic, but it couples trajectory math into the
  beamforming kernel and complicates reuse of the existing tested kernels.
- Reserve for after Strategy A is validated and a bottleneck is measured.

### 4.4 Recommendation

Implement **Strategy A first** as a thin layer over the existing
[`cuda_beamform_packed_intensity`](include/beamformer/cuda_beamformer.hpp:98) /
`CudaBeamformerWorkspace`, validate against a new CPU tracker reference, then decide
whether B or C is needed based on measured launch/transfer overhead.

---

## 5. Suggested first concrete steps (ordered)

1. **Trajectory header** — `include/beamformer/tracker.hpp` with a `TrackerConfig` and a
   `direction(t)` function. Pure host, trivially unit-testable.
2. **Single-beam CPU reference** — `cpu_beamform_tracker_packed_intensity` looping over
   time blocks, reusing `generate_weights` with `n_beams=1`. This is the validation
   anchor and exposes any indexing subtleties early.
3. **Synthetic moving source** — extend `synthetic_data` so we can feed a source whose
   `(l,m)` matches the tracker trajectory, enabling a "power stays on the beam" test.
4. **Strategy-A GPU path** — add `cuda_beamform_tracker_packed_intensity` that loops
   blocks, uploads per-block weights, and calls the existing Direct kernel with
   `n_beams=1`. Keep it host-orchestrated initially.
5. **Numerical validation test** — CPU vs GPU, `atol=1e-3, rtol=1e-5` (matching the
   project's existing point-source tolerances in the
   [`README.md`](README.md:50) point-source check).
6. **Moving-source recovery test** — analogous to `test_cuda_point_source`: confirm the
   tracker beam's integrated maximum stays on the moving source across `n_time=1..4`
   style increments.
7. **CLI** — `tools/beamformer_tracker` with `--trajectory`, `--n-blocks`,
   `--integration-spectra`, reusing the existing `--n-ant/--n-freq/--input` flags.
8. **Benchmark** — extend `benchmark_cpu_cuda` patterns (or a small new tool) to measure
   per-block launch overhead for a single tracker beam vs one full-grid launch, to
   justify or rule out Strategy B/C.

---

## 6. Open questions to resolve before coding

- **Direction update granularity**: per integration window (320 spectra at 3.33 µs each
  ≈ 1.07 ms)? Per spectrum? Per frame? Drives Strategy choice (A vs B/C).
- **Trackable source model**: is there a real ephemeris format, or do we start with a
  parametric trajectory (linear / great-circle / circular)? Affects `tracker.hpp` API.
- **Number of simultaneously tracked beams**: 1 (pure tracker) or a small set (e.g.
  track + guard beams)? Affects whether we reuse Direct (n_beams=1) or Tiled (32-beam
  tile) kernels.
- **Output product**: float32 only, or also the int8 quantized path? The int8 chunk
  layout (1×16×16) is beam-tiled; with very few beams the beam-tile padding may need a
  tracker-specific layout or acceptable waste.
- **Single shard vs full band**: does the tracker run per-NIC (336 ch) per shard, or does
  it combine the two shards' tracker outputs in a downstream step? The base keeps shards
  separate; the tracker should probably honor that and combine in plotting only.

---

## 7. Risks / gotchas inherited from the base

- **`validate_dimensions` pins `n_freq = 336` and `n_ant ∈ {32,64}`** ([`config.hpp`](include/beamformer/config.hpp:81)). Any tracker code must go through the same validator; do not
  invent a parallel dimension set.
- **Element order** is RFSoC-reversed ([`indexing.hpp`](include/beamformer/indexing.hpp:21)); the tracker's
  weights must use the same `positions_m` ordering as the base or steering will point the
  wrong way.
- **No temporal chunking yet** ([`README.md`](README.md:394)): each run allocates the full `n_time` voltage/intensity. A per-block tracker must still respect device-memory limits unless
  we add streaming; reuse `CudaOfflineFrameRunner`'s per-frame pattern to keep allocations
  bounded.
- **Direct kernel supports only 320 spectra** for integration; Tiled supports 10 and 320
  ([`cuda_beamformer.hpp`](include/beamformer/cuda_beamformer.hpp:110)). A tracker using
  Direct + per-window integration is restricted to 320 spectra; if we need finer (10)
  per-block integration, use the Tiled kernel with one beam and accept the pad, or extend
  Direct.

---

## 8. TL;DR

- The repo is a solid, well-tested **fixed-grid packed-int4 voltage beamformer** with CPU
  reference, CUDA Direct/Tiled kernels, fused temporal integration, and int8 quantization.
- A **tracker beam** = time-varying single-beam steering; the weight formula and I/O
  contract are unchanged, only the direction becomes a function of time.
- **Start with Strategy A** (blockwise re-launch of the existing Direct kernel with
  `n_beams=1`, per-block weights from the existing `generate_weights`), validate with a
  new CPU tracker reference and a moving synthetic point source, then measure whether
  launch/transfer overhead justifies a device-side or fully-fused kernel.

---

# 9. Realized v0 — design, components, build, and visualization

> This section documents the implementation that was built on top of the plan above. It is
> kept separate from the original exploratory text (sections 1–8) so the history of the
> analysis is preserved.

## 9.1 Resolved design decisions (from the open questions in §6)

| Question | Decision realized in v0 |
|----------|--------------------------|
| Direction update granularity | **One direction per integration window** (default 320 spectra ≈ 1.07 ms). The direction used for a window is the one at the window's first time sample. |
| Trackable source model | **Linear parametric placeholder**: `dir(t)` advances `(l, m)` by `t * (dl, dm)` and re-projects onto the unit disk via `direction_from_lm`. A zero rate reproduces a stationary beam. |
| Number of tracked beams | **One** (`n_beams == 1`, `tracker_beam_count`). Tiled-kernel padding waste is accepted for later reuse, but v0 uses the direct single-beam loop. |
| Output product | **float32 intensity in the same `[time][freq][beam]` layout** as the fixed-grid beamformer (with `beam` of size 1), so downstream stages and `write_intensities`/`read_weights` IO stay unchanged. |
| Single shard vs full band | The tracker **runs per node on a single local shard** (buffer 0 or 1, 336 channels). Shards are combined downstream at the classification node; the v0 CLI and plotter operate one shard at a time. |

## 9.2 Components added (all `beam_tracker_*` prefixed)

| File | Role |
|------|------|
| [`include/beamformer/beam_tracker.hpp`](include/beamformer/beam_tracker.hpp:1) | `TrackerTrajectoryConfig`, `TrackerConfig`, `tracker_beam_count`, `tracker_direction`, `tracker_window_direction`, `tracker_window_count`, `beam_tracker_cpu_packed_intensity[_into]`, `beam_tracker_make_moving_point_source`. |
| [`src/beam_tracker.cpp`](src/beam_tracker.cpp:1) | Naive CPU tracker (Strategy A): one `n_beams==1` weight set per window via the existing `generate_weights`, inline int4 decode, `[t][f][0]` intensity. Linear trajectory projection. Moving point-source generator (per-`t` re-quantized spectrum). |
| [`tools/beam_tracker_cpu.cpp`](tools/beam_tracker_cpu.cpp:1) | CLI: reads a packed shard, runs `beam_tracker_cpu_packed_intensity`, writes float32 `[T][F][B=1]`, optional metrics CSV. Accepts `--track-l0/m0`, `--dl/dm-per-sample`, `--integration-spectra`. |
| [`tools/plot_tracker_results.py`](tools/plot_tracker_results.py:1) | Tracker dashboard (sky-map trajectory + per-window steering + moving-source overlay, freq-integrated power vs time with optional overlay, single-beam spectrum, `[time,freq]` dB heatmap). Reuses geometry/response helpers from `plot_results.py`. |
| [`tools/generate_fake_data.cpp`](tools/generate_fake_data.cpp:1) | Extended with a single-shard `moving-point-source` synthetic type and tracker trajectory parsing; the two-shard path rejects it (tracker is per shard). |
| [`tests/cpu/test_beam_tracker.cpp`](tests/cpu/test_beam_tracker.cpp:1) | C++ unit tests: trajectory projection, window cadence, stationary==fixed-grid equivalence, per-window decode match, moving-source recovery, validation rejections. |
| [`tests/python/test_plot_tracker_results.py`](tests/python/test_plot_tracker_results.py:1) | Python unit tests: trajectory math, resolution/buffer validation, intensity load round-trip, dashboard PNG render. |
| [`CMakeLists.txt`](CMakeLists.txt:1) | `src/beam_tracker.cpp` added to `beamformer_core`; `beam_tracker_cpu` executable; `test_beam_tracker` CTest target. |

No existing tested file (`cpu_beamformer.*`, `cuda_beamformer.*`, `synthetic_data.*`, `weights.*`, `geometry.*`, `io.*`) was modified to add behavior — the only edited existing files are `tools/generate_fake_data.cpp` (new single-shard `moving-point-source` branch, additive) and `CMakeLists.txt` (new targets, additive).

## 9.3 Build and test status

Built on GCC 12.3.1 with `BEAMFORMER_ENABLE_CUDA=OFF`. The C++ test passes:

```text
Start 5: beam_tracker
1/1 Test #5: beam_tracker .....................   Passed    0.01 sec
100% tests passed, 0 tests failed out of 1
```

The Python plotting helper module is covered by `tests/python/test_plot_tracker_results.py` (run
under the project's conda env, see §9.5).

## 9.4 End-to-end example (single shard)

```bash
# 1. Build (CPU-only is sufficient for the v0 tracker).
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Generate a moving point source:: straight-line drift in l.
./build/generate_fake_data \
    --type moving-point-source \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0  --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --amplitude 4 \
    --output results/moving_source.bin

# 3. Run the CPU tracker aligned with the source trajectory.
./build/beam_tracker_cpu \
    --input results/moving_source.bin \
    --output results/tracker_aligned.bin \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0  --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --metrics results/tracker_metrics.csv

# 4. Render the tracker dashboard.
conda run -n kotekan_test python tools/plot_tracker_results.py \
    --input results/tracker_aligned.bin \
    --output results/tracker_dashboard.png \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --source-l0 0.0 --source-m0 0.0 \
    --source-dl-per-sample 1.0e-4 --source-dm-per-sample 0.0
```

The fourth panel of the dashboard is a `[time, frequency]` dB heatmap of the single tracker
beam; for an aligned source it stays bright across the whole run. A deliberately misaligned
tracker (e.g. `--dl-per-sample 0`) run on the same moving source and overlaid via
`--compare` produces a dimming power-vs-time curve as the source drifts out of the
stationary beam — a visual confirmation that per-window steering tracks the source.

## 9.5 Tests for the plotting helper

```bash
conda run -n kotekan_test python tests/python/test_plot_tracker_results.py
```

These cover trajectory projection, window steering, off-disk rejection, intensity loading,
and an actual dashboard PNG render (Agg backend, no display required).

## 9.6 What is deliberately *not* in v0

- No CUDA tracker kernel yet — the v0 is a naive CPU reference and Strategy-A orchestrator.
- No quantized int8 tracker output, no offline frame-streaming runner variant.
- No two-shard / full-band tracker (per-node single shard only).
- No ephemeris loader — the linear model is the placeholder until the final trajectory format
  is defined.
- No upchannelizer; the `integration_spectra` knob mirrors the existing temporal-integration
  direct path (320 spectra for Direct).

## 9.7 Next steps after v0

- Port the per-window loop to a CUDA tracker stage (Strategy A on the existing Direct kernel
  with `n_beams=1`), then measure per-block launch/transfer overhead to decide between
  host-orchestrated re-launch (A), device-side weight generation (B), or a fully-fused
  time-varying kernel (C).
- Add a moving-source recovery test analogous to `test_cuda_point_source`.
- Replace the linear trajectory with the production ephemeris format once defined.
- Optionally add an int8 tracker output by reusing the existing quantization stage on the
  single-beam `[T_int][F]` tensor.

## 9.8 Visualizations: array metadata and per-window frames

[`tools/plot_tracker_results.py`](tools/plot_tracker_results.py:1) was extended with two
improvements (run on Trillium with plain `python`, no conda required as long as numpy +
matplotlib are available):

1. **Array/run metadata legend** on every panel: array shape (`rows×cols`), `n_ant`, spacing,
   aperture size, channel count, centre frequency, `n_time`, `integration_spectra`, window
   count, and the tracker trajectory parameters — so each PNG is self-describing.
2. **Per-integration-window frame export** via `--frames-dir DIR`: one PNG per window, each
   showing the steered-beam footprint (magenta `-3 dB` contour), the antenna 3 dB FoV
   ellipse, the tracker trajectory, the moving-source track, the current window's source
   position, and a per-window recorded-power histogram with the current frame highlighted.
   Files are zero-padded (`frame_0000.png`, `frame_0001.png`, …) inside a single folder so
   `scp -r` brings the whole sequence over and frame order is preserved.

The frame count is the **number of integration windows**, not the number of time samples
(`(n_time + integration_spectra - 1) // integration_spectra`). For the §9.4 example that is
1 frame; for `n_time=15360, integration_spectra=320` it is 48 frames. `--frames-stride N`
emits every Nth window and `--frames-max` is a safety cap. The default dashboard (`--output`)
and the frames mode (`--frames-dir`) can be used together or independently.

### Per-window frame example (Trillium, plain python)

```bash
# Bigger window count: 16 integration windows of 320 spectra -> 16 frames.
./build/beam_tracker_cpu \
    --input results/moving_source.bin \
    --output results/tracker_aligned.bin \
    --n-time 5120 --n-ant 64 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 --metrics results/tracker_metrics.csv

python tools/plot_tracker_results.py \
    --input results/tracker_aligned.bin \
    --frames-dir results/tracker_frames \
    --n-time 5120 --n-ant 64 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --source-l0 0.0 --source-m0 0.0 \
    --source-dl-per-sample 1.0e-4 --source-dm-per-sample 0.0

# Copy the whole sequence back to this PC:
#   scp -r <trillium_user>@<login>:beamform_project/results/tracker_frames .
# Then page through frame_0000.png .. frame_0015.png to watch the tracker follow the source.
```

To also emit the single-PNG dashboard alongside the frames, pass both `--output` and
`--frames-dir`. To downsample large runs for a quick preview, use `--frames-stride 4` (one
frame every 4 windows) and/or raise `--frames-max`.
