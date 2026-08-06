# CHARTS Voltage Beamformer PoC

Small standalone proof of concept for comparing a direct CPU voltage beamformer with an
equivalent CUDA implementation.

## Current implementation

The input foundation, synthetic data, weight generation, serial CPU reference, and direct
CUDA implementation are available:

- CMake/C++17 project with optional CUDA detection;
- the packed output layout produced by `rfsocHandlerShuffle`;
- signed complex `int4` packing and unpacking;
- common voltage, weight, and intensity indexing and size contracts;
- regular 4x8 and 8x8 array geometries with 0.6 m spacing in x and y;
- 336 local-frequency centers per shard; two shards map the full 672-channel band;
- optional position and frequency overrides from text files;
- deterministic zero-padded FFT-bin beam grids in direction cosines;
- 1 to 128 beams for 32- or 64-element arrays;
- one-hot, constant, point-source, and seeded-noise voltage generation;
- a small `generate_fake_data` CLI that writes headerless RFSoC-layout files;
- geometric complex `float32` weights in `[beam][frequency][element]` order;
- a serial CPU beamformer that produces `float32` intensity in
  `[time][frequency][beam]` order;
- a CUDA beamformer with one thread per `[time][frequency][beam]` output and a direct
  complex sum over all elements;
- packed-input CPU and CUDA references that decode signed `int4` samples inline and produce
  `float32` intensity without a full unpacked voltage tensor or expanded H2D transfer;
- strict binary-size validation and optional per-run CSV timing metrics.

Compact CPU/CUDA validation, a repeatable GPU-only timing sweep, summary tables, and
comparison plots are available. CPU point-source validation and full-sky beam-coverage
plots are also included.

## Build and test

```bash
cmake -S . -B build \
    -DBEAMFORMER_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

When CUDA is enabled, CTest also runs a physical point-source check with a static 32-beam
rectangular grid, 32 antennas, all 672 channels, and `n_time=1,2,3,4`. A source is placed
exactly at non-central beam 12. The test requires the frequency- and time-integrated CPU
and GPU maxima, as well as every per-time maximum, to recover beam 12; it also checks all
CPU/GPU intensities with `atol=1e-3`, `rtol=1e-5`.

On the tested RTX 4090, all four time sizes recovered beam 12 on both backends. The
integrated target-to-runner-up power ratio was `28.675` and the maximum absolute CPU/GPU
error was `0.0078125`, with no output outside the combined tolerance.

```bash
ctest --test-dir build -R cuda_point_source --output-on-failure -V
```

The graphical realization uses `T=4` and the exact beam-12 direction
`(l,m)=(0.078070952604,-0.156141905208)`. After generating the common packed voltage and
weight files and running both backends, reproduce the separate CPU and GPU dashboards
with:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --input results/point_source_32beam_nxd_grid_cpu_intensity.bin --label CPU \
    --n-time 4 --n-freq 672 --n-ant 32 --n-beams 32 \
    --beam-grid legacy-rectangular \
    --synthetic-type point-source \
    --source-l 0.078070952604 --source-m -0.156141905208 --amplitude 4 \
    --output results/cpu_32beam_point_source_validation_nxd_grid.png

conda run -n kotekan_test python tools/plot_results.py \
    --input results/point_source_32beam_nxd_grid_gpu_intensity.bin --label GPU \
    --n-time 4 --n-freq 672 --n-ant 32 --n-beams 32 \
    --beam-grid legacy-rectangular \
    --synthetic-type point-source \
    --source-l 0.078070952604 --source-m -0.156141905208 --amplitude 4 \
    --output results/gpu_32beam_point_source_validation_nxd_grid.png
```

When a CUDA compiler is found, CMake builds `beamformer_cuda`; otherwise the CPU-only
targets remain available. `CMAKE_CUDA_ARCHITECTURES=89` can replace `native` for the
tested RTX 4090. The CPU code remains serial as a transparent numerical reference.

## Generate synthetic voltage files

The single-shard commands below write exactly `n_time * 336 * n_ant` payload bytes:

```bash
./build/generate_fake_data \
    --type one-hot --n-time 32 --n-ant 64 \
    --active-time 1 --active-frequency 100 --active-element 63 \
    --output one_hot.bin

./build/generate_fake_data \
    --type constant --n-time 32 --n-ant 64 \
    --value-real 1 --value-imag 0 \
    --output constant.bin

./build/generate_fake_data \
    --type point-source --n-time 32 --n-ant 64 \
    --source-l 0.04 --source-m 0.0 --amplitude 4 \
    --output point_source.bin

./build/generate_fake_data \
    --type noise --n-time 32 --n-ant 64 --seed 1 \
    --output noise.bin
```

For the production-shaped two-shard input, use a prefix. This writes two independent
payloads, two one-byte-per-`[T][F_local]` loss masks, and one metadata file per shard:

```bash
./build/generate_fake_data \
    --type noise --n-time 15360 --n-ant 64 --seed 1 \
    --shard-output-prefix results/voltage
```

The files are `voltage.shard{0,1}.bin`, `.mask`, and `.meta`. Each payload is exactly
`n_time * 336 * n_ant` bytes; the payloads are never concatenated.

The point source uses
`x_a[f] = A * exp(-j * 2*pi*frequency[f]*dot(position[a], direction)/c)`,
quantized to signed `int4`, and repeats that spectrum for every requested time sample.
Default weights use centers selected from the native bin bank of a zero-padded `(2M,2N)`
FFT geometry. This only defines beam directions: the CPU and CUDA implementations remain
direct voltage beamformers. At the 400 MHz design frequency,
`du=lambda/(2*M*d)` and `dv=lambda/(2*N*d)`. A centered candidate window is ranked by
radial distance and truncated to `n_beams`; the selected directions stay fixed while
weights remain frequency-dependent. `--beam-grid line` retains the earlier one-dimensional
grid, and `--beam-grid legacy-rectangular` reproduces existing 32/64-beam artifacts.

For 32 antennas, `(M,N)=(8,4)` gives a `16x8` bank and exactly 128 possible centers. For
64 antennas, `(M,N)=(8,8)` gives a `16x16` bank; 128 beams use a centered `12x11` candidate
window followed by radial truncation. Hexagonal FoV count and target-lattice helpers are
available for design studies, but are not yet mapped to hardware FFT bins.

## Generate weights and run CPU/CUDA

The following commands use five beams at `l = -0.04, -0.02, 0, 0.02, 0.04`, matching
the default synthetic point source with the last beam:

```bash
./build/generate_weights \
    --n-ant 64 --n-beams 5 --beam-grid line \
    --output weights.bin

./build/beamformer_cpu \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 \
    --output cpu_intensity.bin --metrics metrics.csv

./build/beamformer_cuda \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 \
    --output cuda_intensity.bin --metrics metrics.csv
```

Both executables report the peak integrated beam and write the same output layout. CPU and
CUDA decode each packed byte inside their direct accumulation loops, so `unpack_ms` is zero
rather than a separate full-tensor conversion stage. CUDA timing separates device/context
setup, packed-byte host-to-device copies, kernel execution, and device-to-host copy. In the
common CSV, `compute_ms` means the packed serial loop for CPU and the kernel event time for
CUDA; CPU rows store zero for CUDA-only stages. Throughput and complex GMAC/s are derived
from `compute_ms`. Repeated invocations append rows to one table, but a proper benchmark
should include warmup and repeated samples rather than use the first smoke run.

## Binary products

- Voltage: one byte per complex signed `int4` sample, `[T][F][E]`.
- Weights: `{float real, float imag}`, `[B][F][E]`.
- Intensity: one native `float32` (little-endian on the tested x86-64 host), `[T][F][B]`.

All products are headerless. Dimensions are supplied on the command line, and readers
reject files whose byte count differs from the exact expected size.

## Select buffer(s) in plots

The plotting tools follow the local-shard contract. plot_results.py defaults to buffer 0
with 336 channels. Select the other local buffer with --buffer 1; its default frequency
origin is 400.8 MHz. Use --buffer both for the 672-channel band. For that mode, a single
precombined intensity file is accepted, or two independent output files can be combined only
inside the plotting step:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --buffer 0 --input results/shard0_intensity.bin \
    --n-time 32 --n-ant 64 --n-beams 64 \
    --output results/shard0_validation.png

conda run -n kotekan_test python tools/plot_results.py \
    --buffer both \
    --input results/shard0_intensity.bin \
    --input-shard1 results/shard1_intensity.bin \
    --n-time 32 --n-ant 64 --n-beams 64 \
    --output results/full_band_validation.png
```

The same --input-shard1 and --compare-shard1 options apply to CPU/CUDA comparison
plots. plot_benchmark.py --buffer 0 filters local-buffer rows (n_freq=336), while
--buffer both filters full-band rows (n_freq=672). If a benchmark CSV contains only one
frequency width, no filter is required.

## Visualize and compare results

Run the backend-independent visualizer with the same dimensions and synthetic-source
parameters used to generate the input:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --input cpu_intensity.bin \
    --n-time 32 --n-freq 672 --n-ant 64 --n-beams 5 \
    --beam-grid line \
    --synthetic-type point-source \
    --source-l 0.04 --source-m 0.0 --amplitude 4 \
    --label CPU \
    --output results/cpu_point_source_validation.png
```

The validation dashboard shows:

- array geometry and output-element blocks;
- `u-v` baseline coverage in wavelengths for a selected frequency channel;
- beam and injected-source positions in the directional `l-m` plane;
- integrated intensity by beam;
- spectra for the recovered and offset beams;
- frequency-integrated intensity versus time and beam.

The default spectrum uses the physical centers from 300 to 501.3 MHz. `--frequency-hz`
remains available only as an explicit constant-frequency override, and a frequency text
file can replace all default centers.

### Full-sky 32/64-beam coverage

Generate the complete local `l-m` coverage without requiring an intensity file:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --n-ant 32 --n-beams 32 \
    --beam-grid legacy-rectangular \
    --spacing-m 0.6 \
    --frequency-start-hz 300000000 \
    --channel-width-hz 300000 \
    --design-frequency-hz 400000000 \
    --overlap-db -3 \
    --sky-output results/beam_grid_32_full_sky.png
```

The sky dashboard contains dominant-beam regions, individual -3 dB contours and maximum
absolute gain at 300/400/500 MHz, an exact 672-channel average, and the number of
overlapping beams at 400 MHz. Antenna `BW_E`, `BW_H`, and `gain_dBi` are linearly
interpolated and extrapolated from the supplied 300/400/500 MHz values. Absolute array
gain assumes uniform weights with fixed total power, so the coherent array contribution is
`10*log10(n_ant)` above the interpolated element gain.

The current generated artifacts are:

- `results/beam_grid_32_full_sky.png`;
- `results/cpu_32beam_point_source_validation.png`;
- `results/cpu_cuda_point_source_32beam_validation.png`;
- `results/cpu_cuda_point_source_32beam_comparison.png`;
- `results/cpu_cuda_point_source_32beam_metrics.json`;
- `results/cpu_32beam_point_source_validation_nxd_grid.png`;
- `results/gpu_32beam_point_source_validation_nxd_grid.png`.

Use the same script for numerical and graphical CPU/CUDA comparison:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --input cpu_intensity.bin --label CPU \
    --compare cuda_intensity.bin --compare-label CUDA \
    --n-time 32 --n-freq 672 --n-ant 64 --n-beams 5 \
    --beam-grid line \
    --synthetic-type point-source --source-l 0.04 --source-m 0.0 \
    --output results/cpu_cuda_validation.png \
    --comparison-output results/cpu_cuda_comparison.png \
    --summary-json results/cpu_cuda_errors.json
```

This second dashboard contains an intensity scatter plot, relative-error histogram,
maximum absolute error over frequency, and per-beam integrated-power difference. The JSON
reports maximum absolute error, mean/p99/maximum relative error, correlation, and the
number of samples outside the selected tolerances.

The plotting helpers have standalone tests:

```bash
conda run -n kotekan_test python tests/test_plot_results.py
```

## Input contract

A production input is one headerless binary shard with one packed complex byte per sample:

```text
voltage[time][local_frequency][element]
index = (time * 336 + local_frequency) * n_elements + element
```

The real handler creates two independent `[T][336][64]` buffers, one per NIC. The PoC
keeps them as separate allocations. Shard 0 maps local frequencies `0..335` to absolute
frequencies `0..335`; shard 1 maps them to `336..671`. A downstream consumer may combine
outputs using metadata, but no input file or H2D buffer concatenates the shards.

The two-shard CLI writes payload files plus `.meta` and `.mask` sidecars. Metadata records
shard identity, local width, absolute-frequency origin, timestamp start/step, and loss-mask
identity. The loss mask has one byte per `[time][local_frequency]` frame (`1=valid`,
`0=lost`) and is independent for each shard; payload bytes are not rewritten when a frame
is marked lost.

The element order reproduces the handler:

```text
element = (1 - rfsoc_id) * 32 + packet_element
RFSoC 1 -> element 0..31
RFSoC 0 -> element 32..63
```

Each byte uses signed two's-complement `int4`, with real in the low nibble and imaginary
in the high nibble. The one-hot, constant, seeded-noise, and analytical point-source
functions in `beamformer/synthetic_data.hpp` generate both shards without materializing
an unpacked complex-float voltage tensor. The packed CPU and CUDA references use the same
nibble rule at the point of accumulation and emit `float32` intensity in `[T][F][B]`.

## Initial CUDA smoke check

The first manual check used `T=4`, `F=672`, `E=32`, and five beams with the synthetic
point source at `l=0.04`, `m=0`. CPU and CUDA both selected beam 4. Across 13440 output
values, `max_abs_error=0.005859375`, `max_relative_error=4.76e-7`, and NumPy
`allclose(rtol=1e-5, atol=1e-3)` passed. On the tested RTX 4090 the tiny workload took
about 0.070 ms in the kernel, while the first-process CUDA setup took about 91 ms. These
numbers only establish that the implementation runs and is numerically sensible; they are
not the final CPU/GPU performance result.

## Reproducible GPU benchmark with compact CPU validation

The benchmark keeps `F=672` fixed and uses this default matrix:

- `n_ant={32,64}`;
- `n_beams={16,32,64,128}`;
- GPU timing at `n_time={15360,24576,30720}`;
- one CPU/CUDA numerical comparison per antenna/beam pair at `n_time=16`;
- three GPU warmups and ten measured repetitions.

The 24 long configurations never execute the serial CPU implementation. First verify the
matrix and maximum allocations without creating a CUDA context or output files:

```bash
./build/benchmark_cpu_cuda --dry-run
```

The dry run reports the current packed-input host/GPU working-set estimates. Then run and
plot with:

```bash
./build/benchmark_cpu_cuda \
    --output-prefix results/gpu_benchmark_fft

conda run -n kotekan_test python tools/plot_benchmark.py \
    --input-prefix results/gpu_benchmark_fft
```

The process generates one deterministic signed-`int4` noise spectrum per antenna count
and repeats it over time outside timed regions. Compact CPU validation and all GPU runs
consume the same packed prefix and FFT-grid weights. The principal steady-state metric
keeps weights resident on the GPU. It reports resident CUDA kernel time and pipeline wall
time containing voltage H2D, kernel, output D2H, and synchronization; context/buffer setup
and the one-time weight upload are recorded separately.

Work and throughput use these conventions:

```text
Ncmac = n_time * n_freq * n_beams * n_ant
estimated_FLOP = 8 * Ncmac + 3 * (n_time * n_freq * n_beams)
```

`Ncmac/time` is reported as CMAC/s. Estimated FLOP/s counts eight real operations per
complex multiply-accumulate and three for the final squared magnitude. CSV results contain
every GPU repetition; plots use medians and p25/p75 intervals. Speedups are intentionally
not reported because the long CPU cases are not measured.

Generated products are:

- `results/gpu_benchmark_fft_timings.csv`;
- `results/gpu_benchmark_fft_validation.csv`;
- `results/gpu_benchmark_fft_metadata.json`;
- `results/gpu_benchmark_fft_summary.csv`;
- `results/gpu_benchmark_fft_performance.png`;
- `results/gpu_benchmark_fft_gpu_time_heatmaps.png`;
- `results/gpu_benchmark_fft_validation.png`.

Temporal chunking remains intentionally pending. Each current configuration allocates and
transfers its complete `n_time` voltage and intensity products; do not increase the default
maximum beyond the available device memory without implementing chunked execution.

For a short functional check before a full run:

```bash
./build/benchmark_cpu_cuda \
    --output-prefix /tmp/gpu_benchmark_fft_smoke \
    --n-ant 32 --beams 16 --times 16 --validation-time 2 \
    --warmup 1 --repetitions 2
```

The existing `results/cpu_cuda_benchmark_*` files are the archived pre-change CPU/CUDA
sweep. The new default prefix deliberately avoids overwriting them.
