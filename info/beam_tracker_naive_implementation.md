# Naive CPU Beam Tracker for the CHARTS Voltage Beamformer PoC

A **tracker beam** is the natural companion to the fixed-grid beamformer in this repository. The fixed-grid path precomputes a static set of beam directions and beamforms every time sample against the *whole* grid, emitting a `[T][F][B]` intensity cube. The tracker does the opposite: it maintains **one** (or a small number of) beam whose pointing direction `(l, m)` changes with time, emitting a single `[T][F]` stream instead of a `[T][F][B]` grid. The naive implementation documented here is **CPU-only** and runs **per node on a single local frequency shard** (336 channels), exactly as the fixed-grid pipeline does. The two shards are combined downstream at the classification node; the tracker never concatenates them.

This document is the single reference for the v0 tracker: what it does, where each piece lives, how the physics maps to code, how to run it, what every parameter does, and how to interpret the outputs. Astronomer-facing and developer-facing material are interleaved so a reader can follow the physics from sky direction to `float32` on disk in one pass.

---

## 1. The physics in plain language — for astronomers first, then the code hooks

Coherent beamforming is just **aligning the phases** of the voltages from every antenna so a wavefront arriving from a chosen sky direction adds in phase, then taking the squared magnitude of that coherent sum. Everything else here is bookkeeping.

- **Direction on the sky.** A pointing direction is parameterized by the direction cosines `(l, m)` on the unit sphere (the tangential plane projection centered on zenith), with the third coordinate recovered as `n = sqrt(1 - l^2 - m^2)`. The helper that does this projection is [`direction_from_lm`](src/geometry.cpp).
- **Geometric delay.** For antenna `a` at position `pos[a]` (metres, in the array frame) steered to direction `dir`, the path-length difference is the dot product `delay_m = dot(pos[a], dir)`.
- **Wavenumber.** At frequency `f` the wavenumber is `k = 2*pi*f/c` (with `c` the speed of light; see [`speed_of_light_m_per_s`](include/beamformer/physics.hpp)).
- **Steering weight.** The per-antenna phasor that undoes the geometric delay is `w[f][a] = exp(-j * k * delay_m) = exp(-j * 2*pi*f * dot(pos[a], dir) / c)`. This is exactly the formula reused from the fixed-grid path in [`generate_weights`](src/weights.cpp).
- **Intensity.** Each spectrum is the coherent sum over antennas:

  ```
  intensity[t][f] = | Σ_a w[f][a] * x[t][f][a] |^2
  ```

That is the entire astronomy. The tracker's only twist on the fixed grid is that `dir` is a *function of time* `dir(t)`: the steering weights become time-dependent, but the formula is unchanged.

**Where this lives in code.** The direction-from-cosine projection is [`direction_from_lm`](src/geometry.cpp). The weight formula is the existing [`generate_weights`](src/weights.cpp), invoked by the tracker with a **one-element** direction vector (so it produces the `n_beams==1` weight set the tracker needs). The per-window accumulation loop — decode the packed int4 sample, multiply-accumulate the complex weight, write the squared magnitude — is the inner body of [`beam_tracker_cpu_packed_intensity_into`](src/beam_tracker.cpp). The trajectory that turns `t` into a direction is the linear model in [`tracker_direction`](src/beam_tracker.cpp).

---

## 2. What we have and where it lives

Everything was added alongside the existing tested components without modifying them. The tracker reuses the packed-int4 input contract, the geometry/weight machinery, and the `[time][freq][beam]` output layout with `n_beams==1`.

- **[`include/beamformer/beam_tracker.hpp`](include/beamformer/beam_tracker.hpp)** — the public API. Declares the trajectory and run configuration structs ([`TrackerTrajectoryConfig`](include/beamformer/beam_tracker.hpp) and [`TrackerConfig`](include/beamformer/beam_tracker.hpp)), the compile-time [`tracker_beam_count`](include/beamformer/beam_tracker.hpp) (== 1), the helper queries [`tracker_window_count`](include/beamformer/beam_tracker.hpp), [`tracker_direction`](include/beamformer/beam_tracker.hpp), [`tracker_window_direction`](include/beamformer/beam_tracker.hpp), the two beamformer entry points [`beam_tracker_cpu_packed_intensity`](include/beamformer/beam_tracker.hpp) and [`beam_tracker_cpu_packed_intensity_into`](include/beamformer/beam_tracker.hpp), and the synthetic-source generator [`beam_tracker_make_moving_point_source`](include/beamformer/beam_tracker.hpp).
- **[`src/beam_tracker.cpp`](src/beam_tracker.cpp)** — the implementation. [`validate_tracker_config`](src/beam_tracker.cpp) enforces `n_beams==1`, a positive `integration_spectra`, and a finite unit-vector start. The linear trajectory model is [`tracker_direction`](src/beam_tracker.cpp). The per-window loop — generate one `n_beams==1` weight set, decode each packed int4 byte (real in the low nibble, imag in the high nibble, signed -8..7), do the complex MAC, write `sum_real^2 + sum_imag^2` — is the heart of [`beam_tracker_cpu_packed_intensity_into`](src/beam_tracker.cpp). The moving point-source generator is [`beam_tracker_make_moving_point_source`](src/beam_tracker.cpp).
- **[`tools/beam_tracker_cpu.cpp`](tools/beam_tracker_cpu.cpp)** — the CLI tool. Parses flags, builds [`Dimensions`](include/beamformer/config.hpp) with `n_freq=336` and `n_beams=1`, builds a [`TrackerConfig`](include/beamformer/beam_tracker.hpp), reads the packed shard, runs the tracker, writes the float32 intensity, and (optionally) appends a timing row to a metrics CSV.
- **[`tools/generate_fake_data.cpp`](tools/generate_fake_data.cpp)** — the synthetic-data CLI, extended *additively* with `--type moving-point-source` (single shard only). The block in [`generate`](tools/generate_fake_data.cpp) that produces a moving source resolves positions/frequencies exactly like the stationary `point-source` mode, then hands a [`TrackerTrajectoryConfig`](include/beamformer/beam_tracker.hpp) to [`beam_tracker_make_moving_point_source`](src/beam_tracker.cpp).
- **[`tools/plot_tracker_results.py`](tools/plot_tracker_results.py)** — the Python visualizer. Renders a dashboard PNG and (optionally) a per-window frames directory. See [Understanding the outputs](#understanding-the-outputs).
- **[`tests/test_beam_tracker.cpp`](tests/test_beam_tracker.cpp)** — unit tests covering the trajectory model, the constant-trajectory == one-beam-fixed-grid equivalence, per-window weight changes, below-horizon rejection, and the moving-source recovery check. See [Validation / testing](#validation--testing).
- **Reused infrastructure** (do not rebuild — the tracker leans on all of it):
  - [`include/beamformer/config.hpp`](include/beamformer/config.hpp) — [`Dimensions`](include/beamformer/config.hpp), `n_freq=336` (`default_frequency_channels`), `n_ant ∈ {32, 64}`, the two-shard layout.
  - [`include/beamformer/geometry.hpp`](include/beamformer/geometry.hpp) and [`src/geometry.cpp`](src/geometry.cpp) — antenna positions ([`default_positions`](src/geometry.cpp): regular 4×8 for 32 ant, 8×8 for 64 ant, 0.6 m spacing), channelized frequencies ([`channelized_frequencies`](src/geometry.cpp): 300 MHz start, 300 kHz/channel, 336 channels, 400 MHz design frequency), and the [`direction_from_lm`](src/geometry.cpp) projection.
  - [`include/beamformer/complex.hpp`](include/beamformer/complex.hpp) — the [`ComplexFloat`](include/beamformer/complex.hpp) type used by the weight tensor.
  - [`include/beamformer/io.hpp`](include/beamformer/io.hpp) and [`src/io.cpp`](src/io.cpp) — headerless binary readers/writers ([`read_packed_voltage`](src/io.cpp), [`write_intensities`](src/io.cpp)) that *reject* files whose byte count differs from the exact expected size.
  - [`include/beamformer/weights.hpp`](include/beamformer/weights.hpp) and [`src/weights.cpp`](src/weights.cpp) — [`generate_weights`](src/weights.cpp), the geometric phase-steering weight formula shared with the fixed grid.
  - [`include/beamformer/physics.hpp`](include/beamformer/physics.hpp) — `speed_of_light_m_per_s` and `two_pi`.
  - [`include/beamformer/indexing.hpp`](include/beamformer/indexing.hpp) — `voltage_index`, `weight_index`, `intensity_index` (and the RFSoC element reorder noted in the README).

---

## 3. How the physics maps to code — a trace through one integration window

For a single integration window the tracker does, in order:

1. **Compute the direction at the window's first time sample.** [`tracker_window_direction`](src/beam_tracker.cpp) evaluates the trajectory at `t = window * integration_spectra`, returning a unit `Vec3` via [`direction_from_lm`](src/geometry.cpp).
2. **Generate one `n_beams==1` weight set** by calling [`generate_weights`](src/weights.cpp) with a single-element `beam_directions` vector `[direction]`. The result lives in the standard `weight_index(beam=0, freq, element, dims)` layout.
3. **Loop over every spectrum in the window** (`first_time = window*integration_spectra` … `last_time = min(first_time+integration_spectra, n_time)`), then **loop over frequency**, and for each `(t, f)`:
   - decode each packed byte with `unpack_complex_int4` → `(real, imag)` in the signed -8..7 range (real in the low nibble, imag in the high nibble);
   - accumulate `sum += w * sample` as a complex multiply-accumulate;
   - write `intensity[intensity_index(time, frequency, 0, dims)] = sum_real^2 + sum_imag^2`.

The key invariant: **one direction per window ⇒ one weight set per window ⇒ that weight set is reused for every spectrum inside the window.** A zero drift rate collapses this to a single direction for all time, i.e. a one-beam fixed grid.

```cpp
const auto positions    = default_positions(dims.n_ant);
const auto frequencies  = channelized_frequencies(dims.n_freq);
const std::size_t window_count =
    tracker_window_count(dims.n_time, tracker.integration_spectra);

for (std::size_t window = 0; window < window_count; ++window) {
    const Vec3 direction = tracker_window_direction(tracker.trajectory,
                                                    window,
                                                    tracker.integration_spectra);
    const auto window_weights = generate_weights(dims, positions, frequencies,
                                                 std::vector<Vec3>{direction});

    const std::size_t first_time = window * tracker.integration_spectra;
    const std::size_t last_time  = std::min(first_time + tracker.integration_spectra,
                                            dims.n_time);
    for (std::size_t time = first_time; time < last_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            float sum_real = 0.0F, sum_imag = 0.0F;
            for (std::size_t element = 0; element < dims.n_ant; ++element) {
                const auto s = unpack_complex_int4(
                    packed[voltage_index(time, frequency, element, dims)]);
                const auto& w = window_weights[weight_index(0, frequency, element, dims)];
                sum_real += w.real * float(s.real) - w.imag * float(s.imag);
                sum_imag += w.real * float(s.imag) + w.imag * float(s.real);
            }
            intensity[intensity_index(time, frequency, 0, dims)] =
                sum_real * sum_real + sum_imag * sum_imag;
        }
    }
}
```

Mathematically, for one window with direction `d_w`:

```
w[f][a] = exp(-j * 2*pi*f * dot(pos[a], d_w) / c)
intensity[t][f] = | Σ_a w[f][a] * x[t][f][a] |^2 ,  t ∈ [window*integration_spectra, last_time)
```

This is Strategy A from the design note [`info/tracker_beam_first_look.md`](info/tracker_beam_first_look.md): one weight-set generation per integration window, reused for every spectrum in that window — the simplest viable realization that reuses 100% of the fixed-grid machinery.

---

## 4. The trajectory model

The placeholder trajectory is **linear in direction-cosine space**:

```
l(t) = l0 + t * dl_per_sample
m(t) = m0 + t * dm_per_sample
```

The resulting `(l, m)` pair is then **re-projected onto the unit sphere** by [`direction_from_lm`](src/geometry.cpp), which recomputes `n = sqrt(1 - l^2 - m^2)`. Concretely, [`tracker_direction`](src/beam_tracker.cpp) advances the cosines and calls [`direction_from_lm`](src/geometry.cpp); [`tracker_window_direction`](src/beam_tracker.cpp) is just [`tracker_direction`](src/beam_tracker.cpp) evaluated at the window's first sample.

Two behavioural details matter:

- **Below-horizon rejection, not clamping.** [`direction_from_lm`](src/geometry.cpp) throws `std::invalid_argument` when `l*l + m*m > 1` (i.e. the cosines leave the unit disk, which would point the beam below the horizon). The tracker does **not** clamp; bad configurations surface immediately as exceptions. Callers must keep the whole `(start, t*rate)` path inside the unit disk for `0 <= t < n_time`.
- **Zero rate ⇒ a stationary beam.** `dl_per_sample = dm_per_sample = 0` produces a constant direction for all `t`, which is byte-identical to a one-beam fixed grid beamformed at that direction. This is the anchor used by the equivalence test in [`tests/test_beam_tracker.cpp`](tests/test_beam_tracker.cpp) ("constant trajectory equals the one-beam fixed grid").

**Per-window direction is taken at the window's first sample** (`window * integration_spectra`). With `integration_spectra=320` at the ~3.33 µs spectrum period, one window ≈ 1.07 ms, so the tweak-rate (cost) of weight regeneration is one set per ~1 ms. A **final partial window is still emitted with no truncation** — `last_time = min(first_time + integration_spectra, n_time)` handles the remainder, mirroring the temporal-integration reference's prefix-only contract.

---

## 5. The input / output contracts

The tracker shares the fixed-grid byte contracts; only the beam axis collapses to size 1.

### Input

A **headerless packed-int4 voltage shard**, layout `voltage[time][freq][element]` with

```
index = (t * n_freq + f) * n_ant + e
```

One byte per complex sample: **real in the low nibble, imag in the high nibble**, signed int4 in the range `-8..7`. The expected file size is exactly

```
n_time * 336 * n_ant    bytes
```

(Per the project README, the element order reproduces the RFSoC handler: `element = (1 - rfsoc_id)*32 + packet_element`, so RFSoC 1 → elements 0..31 and RFSoC 0 → elements 32..63. The tracker inherits this order unchanged.)

### Output

A **headerless float32 intensity cube**, layout `intensity[time][freq][beam]` with `n_beams == 1`, so beam index 0 is always the only beam. The expected file size is exactly

```
n_time * 336 * 1 * sizeof(float)    bytes
```

On the tested host this is little-endian IEEE-754 float32. The `[time][freq][beam]` layout is reused deliberately so downstream stages (and [`write_intensities`](src/io.cpp) / [`read_weights`](src/io.cpp) style I/O) stay byte-compatible with the fixed-grid path.

### Optional metrics CSV

Produced only when `--metrics FILE` is passed. One row is appended per [`beam_tracker_cpu`](tools/beam_tracker_cpu.cpp) run. The header line (written when the file is new or empty) is:

```
backend,n_time,n_freq,n_ant,n_beams,integration_spectra,track_l0,track_m0,dl_per_sample,dm_per_sample,load_ms,compute_ms,write_ms,total_ms
```

and each data row begins with `tracker_cpu` followed by the run's dimensions, trajectory parameters, and the load/compute/write/total timings in milliseconds.

### Dimensions are supplied on the command line

The CLI hard-codes `n_freq = 336` (`default_frequency_channels`) and `n_beams = 1` ([`tracker_beam_count`](include/beamformer/beam_tracker.hpp)); only `--n-time` and `--n-ant` are user-selectable. [`read_packed_voltage`](src/io.cpp) and [`write_intensities`](src/io.cpp) **reject** any file whose byte count differs from the exact expected size (`n_time*n_freq*n_ant` bytes in, `n_time*n_freq*n_beams*sizeof(float)` bytes out), so a wrong `--n-time` or `--n-ant` fails loudly instead of silently misindexing.

---

## 6. How to build (tracker is CPU-only)

The tracker ships in the CPU core library and needs no CUDA. Disable CUDA and build in Release:

```bash
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -R beam_tracker --output-on-failure   # run just the tracker test
```

[`CMakeLists.txt`](CMakeLists.txt) adds [`src/beam_tracker.cpp`](src/beam_tracker.cpp) to the [`beamformer_core`](CMakeLists.txt) library and defines the [`beam_tracker_cpu`](CMakeLists.txt) executable (linked against `beamformer_core`) and the [`beam_tracker`](CMakeLists.txt) CTest target.

---

## 7. How to run a full moving-beam simulation (end-to-end)

A self-contained worked example: build a synthetic source whose direction matches the tracker's trajectory, run the aligned tracker, and render the dashboard. Each step is one fenced block with a one-line comment.

### (a) Build

```bash
# CPU-only Release build of the tracker + synthetic-data tooling
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### (b) Generate a moving point source

```bash
# Source starts at (l,m)=(0,0) and drifts in +l at 1e-4 per sample; amplitude 4
./build/generate_fake_data --type moving-point-source \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --amplitude 4 \
    --output results/moving_source.bin
```

### (c) Run the aligned CPU tracker

```bash
# Tracker steers along the same trajectory as the source; one window (320 spectra)
./build/beam_tracker_cpu \
    --input  results/moving_source.bin \
    --output results/tracker_aligned.bin \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --metrics results/tracker_metrics.csv
```

### (d) Render the dashboard PNG

```bash
# Plot the aligned run; overlay the red-dashed source track from its own trajectory
python tools/plot_tracker_results.py \
    --input  results/tracker_aligned.bin \
    --output results/tracker_dashboard.png \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --source-l0 0.0 --source-m0 0.0 \
    --source-dl-per-sample 1.0e-4 --source-dm-per-sample 0.0
```

### (e) Misaligned comparison (source drifts out of a stationary beam)

To demonstrate the tracker actually *tracking*, steer a **stationary** beam (`--dl-per-sample 0`) at the source's starting direction and compare it against the aligned run:

```bash
# Stationary beam at (0,0) — source drifts out over the 320-sample run
./build/beam_tracker_cpu \
    --input  results/moving_source.bin \
    --output results/tracker_stationary.bin \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 0 --dm-per-sample 0 \
    --integration-spectra 320
```

```bash
# Overlay the stationary (misaligned) run against the aligned run
python tools/plot_tracker_results.py \
    --input  results/tracker_aligned.bin \
    --compare results/tracker_stationary.bin \
    --output results/tracker_compare.png \
    --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 \
    --source-l0 0.0 --source-m0 0.0 \
    --source-dl-per-sample 1.0e-4 --source-dm-per-sample 0.0
```

In the aligned spectrum the `[time, frequency]` heatmap stays bright across the whole run (the beam follows the source), while the stationary comparison dims in **power vs time** as the source drifts out of the fixed lobe.

---

## 8. Parameters reference (what each knob does)

### Table A — `beam_tracker_cpu` CLI flags

These are parsed in [`parse_options`](tools/beam_tracker_cpu.cpp); the usage text is in [`print_usage`](tools/beam_tracker_cpu.cpp).

| Flag | Meaning | Default | Notes |
|------|---------|---------|-------|
| `--input FILE` | packed-int4 voltage shard to beamform | required | exactly `n_time*336*n_ant` bytes; size checked on read |
| `--output FILE` | float32 intensity output, `[T][F][1]` | required | `n_time*336*sizeof(float)` bytes |
| `--metrics FILE` | append a timing row to this CSV | unset | header written when file is new/empty |
| `--n-time N` | number of time samples | 15360 | sets `Dimensions.n_time` |
| `--n-ant N` | number of antennas | 64 | must be 32 or 64 (default geometry); sets `Dimensions.n_ant` |
| `--track-l0 F` | initial direction cosine `l` at t=0 | 0.0 | fed to [`direction_from_lm`](src/geometry.cpp) to build `direction_start` |
| `--track-m0 F` | initial direction cosine `m` at t=0 | 0.0 | as above |
| `--dl-per-sample F` | linear drift in `l` per time sample | 0.0 | zero rate ⇒ one-beam fixed grid |
| `--dm-per-sample F` | linear drift in `m` per time sample | 0.0 | trajectory stays in the unit disk or it throws |
| `--integration-spectra N` | spectra per window ⇒ one weight set per window | 320 | ≈ 1.07 ms at ~3.33 µs/spectrum; final partial window still emitted |
| `--help` | print usage and exit | — | also exits on an unknown flag |

`n_freq` and `n_beams` are **not** CLI flags: `n_freq` is hard-coded to 336 (`default_frequency_channels`) and `n_beams` is hard-coded to 1 ([`tracker_beam_count`](include/beamformer/beam_tracker.hpp)).

### Table B — `generate_fake_data` `moving-point-source` flags

These are the flags relevant to `--type moving-point-source`. See [`generate`](tools/generate_fake_data.cpp) for the resolution logic (which mirrors the stationary `point-source` mode for positions/frequencies).

| Flag | Meaning | Default | Notes |
|------|---------|---------|-------|
| `--type moving-point-source` | select the moving-source generator | `point-source` | single-shard only; rejects `--shard-output-prefix` |
| `--output FILE` | output packed-int4 shard | required | standard `voltage[time][freq][element]` layout |
| `--n-time N` | number of time samples | 15360 | source direction recomputed per `t` |
| `--n-ant N` | number of antennas | 64 | 32 or 64; selects default geometry |
| `--track-l0 F` | source direction cosine `l` at t=0 | 0.0 | builds `TrackerTrajectoryConfig.direction_start` |
| `--track-m0 F` | source direction cosine `m` at t=0 | 0.0 | as above |
| `--dl-per-sample F` | source drift in `l` per sample | 0.0 | re-projected onto the unit disk per `t` |
| `--dm-per-sample F` | source drift in `m` per sample | 0.0 | as above |
| `--amplitude A` | coherent source amplitude | 4.0 | must be in `(0, 7]` (int4 range); rounded into int4 per sample |
| `--spacing-m M` | default-geometry antenna spacing | 0.6 | ignored if `--positions` is given |
| `--positions FILE` | optional `x,y,z` rows indexed by element | unset | overrides default geometry |
| `--frequencies FILE` | optional one-FHz-per-line override | unset | overrides channelized frequencies |
| `--frequency-hz HZ` | optional constant-frequency override | unset | `300 + 0.3*channel MHz` channelization by default |

### Table C — key `plot_tracker_results.py` flags

From the [`build_parser`](tools/plot_tracker_results.py) function.

| Flag | Meaning | Default | Notes |
|------|---------|---------|-------|
| `--input FILE` | tracker intensity file `[T][F][B=1]` float32 | required | produced by `beam_tracker_cpu` |
| `--compare FILE` | optional second intensity file to overlay | unset | renders the comparison and `--compare-label` series |
| `--output PNG` | dashboard PNG path | unset | default rendering mode |
| `--frames-dir DIR` | per-integration-window frames folder | unset | `scp -r` friendly; zero-padded `frame_NNNN.png` |
| `--frames-stride N` | emit every Nth window as a frame | 1 (all windows) | downsampling |
| `--frames-max N` | hard cap on number of frame PNGs | 256 | safety against huge runs |
| `--n-time N` | number of time samples in the input | required | must match the tracker run |
| `--n-ant N` | antenna count for geometry inset | 32 | promotes first 32 to royal blue, 33+ to crimson in the inset |
| `--integration-spectra N` | window size for direction cadence / frames | 320 | must match the tracker run |
| `--track-l0 F` / `--track-m0 F` | tracker trajectory start cosines | 0.0 / 0.0 | draws the cyan trajectory |
| `--dl-per-sample F` / `--dm-per-sample F` | tracker drift per sample | 0.0 / 0.0 | extends the trajectory |
| `--source-l0 F` / `--source-m0 F` | moving-source start cosines (overlay) | unset / unset | red dashed track when supplied |
| `--source-dl-per-sample F` / `--source-dm-per-sample F` | moving-source drift per sample | 0.0 / 0.0 | extends the source track |
| `--sky-resolution N` | sky-map pixel resolution | 121 | higher ⇒ sharper beam-response image but slower |
| `--design-frequency-hz HZ` | design frequency for the beam footprint | 400 MHz (`beam_grid_design_frequency_hz`) | sets the FoV ellipse scale |
| `--label STR` / `--compare-label STR` | legend labels | `tracker_cpu` / `tracker_baseline` | used on the power-vs-time panel |
| `--dpi N` | output PNG resolution | 150 | — |
| `--show` | open the figure interactively (in addition to saving) | off | — |

---

## 9. Understanding the outputs

### The float32 intensity `[T][F][1]` cube

The product is a stream `intensity[t][f][0]` of squared coherent magnitudes. **Bright means the source was coherently summed in the steered direction.** For an *aligned* source (tracker steering == source motion) the `[time, frequency]` heatmap stays bright across the whole run; for a *misaligned* / drifting-away case it dims — the classic signature of a source moving out of a fixed lobe. Because `n_beams==1` and `n_freq==336`, the file is `n_time*336*4` bytes of raw float32.

### The four dashboard panels

[plot_tracker_results.py](tools/plot_tracker_results.py) composes a single figure with:

1. **Sky map** — the steered-beam response in dB across `(l, m)`, with the **FoV ellipse**, the **trajectory** (cyan line with a lime start marker and a yellow end marker), the **per-window steering markers** (white dots at each `tracker_window_direction`), and the optional **moving-source track** (red dashed) drawn when `--source-*` arguments are supplied.
2. **Array geometry inset** — the antenna positions, with the first 32 elements in royal blue and elements 33+ (the second RFSoC's 32 elements) in crimson, so the 64-antenna layout is legible at a glance.
3. **Frequency-integrated power vs time** — the `Σ_f intensity[t][f]` curve, with vertical lines at integration-window boundaries, plus an optional `--compare` overlay series labelled with `--compare-label`.
4. **`[time, frequency]` dB heatmap** — a magma-coloured view of the whole cube, the cleanest one-glance diagnostic of aligned vs misaligned steering.

### The per-window frames directory (`--frames-dir`)

When `--frames-dir DIR` is set, the script drops one **zero-padded** `frame_0000.png … frame_NNNN.png` per integration window. Each frame shows the array inset, the FoV ellipse, the trajectory, the **current window's steered-beam `-3 dB` footprint** as a magenta contour, a **star marker** at the current window's steering direction, and a **per-window recorded-power histogram** with the current window highlighted. Use `--frames-stride N` to downsample (emit every Nth window) and `--frames-max N` as a hard cap on the number of frame PNGs written (safety against huge runs).

### The metrics CSV

One row per `beam_tracker_cpu` run, with `backend=tracker_cpu`, the dimensions (`n_time, n_freq, n_ant, n_beams`), `integration_spectra`, the trajectory parameters (`track_l0, track_m0, dl_per_sample, dm_per_sample`), and the timings (`load_ms, compute_ms, write_ms, total_ms` in milliseconds). See the appended header in [`append_metrics`](tools/beam_tracker_cpu.cpp).

---

## 10. Validation / testing

**Constant trajectory equals the one-beam fixed grid.** With zero drift and `n_beams==1`, the tracker must produce byte-identical output to the existing [`cpu_beamform_packed_intensity`](src/cpu_beamformer.cpp) for that single beam. This is the numerical anchor and is asserted in [`tests/test_beam_tracker.cpp`](tests/test_beam_tracker.cpp) under the heading "constant trajectory equals the one-beam fixed grid". Also exercised there:

- **Per-window weight changes** — a small drift changes weights window-to-window.
- **Below-horizon rejection** — a trajectory that leaves the unit disk throws via [`direction_from_lm`](src/geometry.cpp) rather than clamping.
- **Moving-source recovery** — with an aligned tracker ([`beam_tracker_make_moving_point_source`](src/beam_tracker.cpp) feeding [`beam_tracker_cpu_packed_intensity`](src/beam_tracker.cpp) using the source's own trajectory), the total integrated power exceeds that of a deliberately misaligned (zero-drift, off-axis) tracker.

[`CMakeLists.txt`](CMakeLists.txt) defines the [`beam_tracker`](CMakeLists.txt) CTest target, and the test executable is built with the `-UNDEBUG` flag so `assert` is active:

```bash
ctest --test-dir build -R beam_tracker --output-on-failure
```

---

## 11. Scope and what is deliberately NOT here (v0)

The v0 tracker is intentionally narrow so the tested fixed-grid path stays untouched. Deliberately absent:

- **CPU-only.** There is no CUDA tracker kernel (no `cuda_beamform_tracker_*`). Strategy A is realized purely in host code.
- **No int8 quantized tracker output.** The tracker emits float32 intensity only; the CHIME-style int8 quantization stage is not wired into the tracker.
- **No offline frame-streaming.** The [`CudaFrameView`](include/beamformer/cuda_frame.hpp) / [`cuda_offline_runner`](include/beamformer/cuda_offline_runner.hpp) streaming abstraction is not used by the tracker; this v0 processes one whole shard at once.
- **No two-shard / full-band tracker.** The tracker runs on a single local frequency shard (336 channels); the two shards are combined downstream at the classification node, not here.
- **No ephemeris loader.** The trajectory is the linear placeholder built from `(l0, m0, dl_per_sample, dm_per_sample)`. A file/ephemeris-driven `direction(t)` is future work.
- **No upchannelizer beyond the existing 320-spectra direct path.** Temporal integration granularity is whatever `--integration-spectra` selects (default 320).

**Future work pointers** (from the design note [`info/tracker_beam_first_look.md`](info/tracker_beam_first_look.md)):

- **Strategy B — device-side weight generation.** Pass a compact track description to a small kernel that writes `[freq][ant]` weights per block, avoiding the host round-trip when the direction changes every block.
- **Strategy C — fully fused time-varying kernel.** One kernel that reads `dir(t)` from a device-side table (or computes it analytically), builds weights in registers/shared memory, and accumulates — no separate weight tensor at all. Reserve for after Strategy A (the v0 here) is validated and a bottleneck is measured.

---

## 12. Quick-start cheatsheet (one fenced block)

Copy-paste a working moving-beam simulation in under a minute. Assumes `n-time 320`, `n-ant 32`, one window (320 spectra), source and tracker both starting at `(l,m) = (0,0)` and drifting in `+l` at `1e-4` per sample.

```bash
# 1) build (CPU-only)
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# 2) generate a moving point source
./build/generate_fake_data --type moving-point-source --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --amplitude 4 --output results/moving_source.bin
# 3) run the aligned CPU tracker
./build/beam_tracker_cpu --input results/moving_source.bin --output results/tracker_aligned.bin \
    --n-time 320 --n-ant 32 --track-l0 0.0 --track-m0 0.0 \
    --dl-per-sample 1.0e-4 --dm-per-sample 0.0 --integration-spectra 320 \
    --metrics results/tracker_metrics.csv
# 4) render the dashboard PNG
python tools/plot_tracker_results.py --input results/tracker_aligned.bin \
    --output results/tracker_dashboard.png --n-time 320 --n-ant 32 \
    --track-l0 0.0 --track-m0 0.0 --dl-per-sample 1.0e-4 --dm-per-sample 0.0 \
    --integration-spectra 320 --source-l0 0.0 --source-m0 0.0 \
    --source-dl-per-sample 1.0e-4 --source-dm-per-sample 0.0
```

Open `results/tracker_dashboard.png` for the four-panel diagnostic described in [Understanding the outputs](#9-understanding-the-outputs).
