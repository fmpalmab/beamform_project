#include "beamformer/cuda_tracker_v2.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {
namespace {

// ---------------------------------------------------------------------------
// CUDA error handling (mirrors src/cuda_beamformer.cu conventions).
// ---------------------------------------------------------------------------
void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(result));
    }
}

// ---------------------------------------------------------------------------
// Device helpers — Step 2 of the plan. Every formula is a literal translation
// of the CPU path so the per-cell float arithmetic reproduces it exactly.
// ---------------------------------------------------------------------------

// 4-bit signed 2's complement nibble decode (bit-identical to decode_signed_nibble
// in int4.hpp, which is already BEAMFORMER_HOST_DEVICE; re-exposed here as a
// dedicated float2 producer for convenience at the point of use in the kernels).
__device__ __forceinline__ float2
unpack_complex_int4_device(const std::uint8_t packed) {
    const std::uint8_t real_bits = static_cast<std::uint8_t>(packed & 0x0FU);
    const std::uint8_t imag_bits = static_cast<std::uint8_t>(packed >> 4);
    const float real = real_bits < 8U
                          ? static_cast<float>(real_bits)
                          : static_cast<float>(static_cast<int>(real_bits) - 16);
    const float imag = imag_bits < 8U
                          ? static_cast<float>(imag_bits)
                          : static_cast<float>(static_cast<int>(imag_bits) - 16);
    return make_float2(real, imag);
}

// [time][freq][beam] flat index (n_beams == 1 here, so beam == 0).
__device__ __forceinline__ std::size_t
intensity_index_device(const std::size_t t, const std::size_t f,
                       const std::size_t b, const std::size_t n_freq,
                       const std::size_t n_beams) {
    return (t * n_freq + f) * n_beams + b;
}

// Antenna position for element index `element`, configured as a rows x cols
// regular array laid out row-major: { col*spacing, row*spacing, 0 }.
// This reproduces src/geometry.cpp's regular_array + default_positions exactly
// (n_ant==32 -> 4x8, n_ant==64 -> 8x8; spacing default 0.6 m), inside the
// kernel so we never need to ship a positions vector to the device. The
// computed positions are bit-identical to CPU positions for the two supported
// n_ant values because the arithmetic is identical single-precision float math.
__device__ __forceinline__ float3
tracker_position_device(const std::size_t element, const std::size_t n_ant,
                         const float spacing_m) {
    std::size_t rows = 0;
    std::size_t columns = 0;
    if (n_ant == 32) {
        rows = 4;
        columns = 8;
    } else if (n_ant == 64) {
        rows = 8;
        columns = 8;
    } else {
        // Only 32/64 are ever valid (validate_dimensions enforces it).
        rows = 0;
        columns = n_ant;
    }
    const std::size_t row = element / columns;
    const std::size_t column = element % columns;
    return make_float3(static_cast<float>(column) * spacing_m,
                       static_cast<float>(row) * spacing_m, 0.0F);
}

// Channelized frequency for channel `f`, matching channelized_frequencies():
//   f_hz = 300e6 + f * 300e3  (float arithmetic, same as the CPU vector build).
__device__ __forceinline__ float
tracker_frequency_device(const std::size_t channel,
                         const float start_hz,
                         const float channel_width_hz) {
    return start_hz + static_cast<float>(channel) * channel_width_hz;
}

// Geometric phase for (position, direction, frequency), computed in double and
// then passed to cos()/sin() (libc/libdevice double transcendentals) before the
// single-precision float result is stored — exactly matching the CPU:
//     double phase = wave_number * delay_m;
//     weight = { (float)cos(phase), (float)sin(phase) };
// where wave_number = two_pi * (double)frequency / c.  Performing the multiply,
// the dot product, and the trig in double (then narrowing once) is load-bearing
// for bit-for-bit parity with the CPU OpenMP kernel; switching to float
// transcendentals changes rounding and breaks exact equality.
__device__ __forceinline__ void
tracker_weight_device(const float3 position, const float3 direction,
                     const double wave_number, float* weight_real,
                     float* weight_imag) {
    const double delay_m = static_cast<double>(position.x) * direction.x
                         + static_cast<double>(position.y) * direction.y
                         + static_cast<double>(position.z) * direction.z;
    const double phase = wave_number * delay_m;
    *weight_real = static_cast<float>(cos(phase));
    *weight_imag = static_cast<float>(sin(phase));
}

// Direction (l, m, n) at integration window `window` for the linear
// trajectory model, matching tracker_window_direction + direction_from_lm.
// `window_directions_host` is computed on the host pre-launch (because
// direction_from_lm throws when the source leaves the unit disk, which must be
// surfaced as a host exception before any kernel runs); the device only needs
// the already-validated (l, m, n) vector. `direction.x == l`, `direction.y ==
// m`, `direction.z == n`.
__device__ __forceinline__ float3
load_window_direction_device(const float* __restrict__ window_directions,
                             const std::size_t window) {
    const std::size_t base = window * 3;
    return make_float3(window_directions[base + 0],
                       window_directions[base + 1],
                       window_directions[base + 2]);
}

// ---------------------------------------------------------------------------
// Phase 1 — Kernel 1: precompute steering weights for every (window, freq).
// 2D grid over (window, freq); each thread writes one ComplexFloat (one
// antenna) for its (window, freq) pair... but the naive launch (one thread per
// (w,f)) would loop all 64 antennas. We instead launch one thread per
// (window, freq, antenna) — a flat 1D grid — for maximum parallelism, while
// preserving the per-element bit pattern the CPU writes.
// ---------------------------------------------------------------------------

__global__ void tracker_v2_weights_kernel(
    ComplexFloat* __restrict__ all_weights,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::size_t n_freq, const std::size_t n_ant,
    const float spacing_m) {
    const std::size_t thread_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = gridDim.x * blockDim.x;
    const std::size_t window_freq_element_count =
        /* windows implicit via indexing */ total;  // unused guard
    (void)window_freq_element_count;

    // Flat iteration space equals window_count * n_freq * n_ant; the launch
    // passes the exact count and we stride to support overlaunch.
    // We rely on the host computing the flat launch count and an `count`
    // guard via `total` driven by the grid; the host rounds block_count up and
    // passes the cell count through a uniform `n_cells` parameter. To keep the
    // signature simpler we collapse: the host launches EXACTLY N threads (== N
    // cells) with a 1D grid of ceil(N/256) blocks * 256 threads, and we early-
    // out past N. We get N from a dedicated parameter below.
}

// A cleaner weights kernel: host passes the exact element count and we iterate
// cell-by-cell with a per-thread guard. This is the kernel actually launched.
__global__ void tracker_v2_weights_kernel_flat(
    ComplexFloat* __restrict__ all_weights,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::size_t window_count,
    const std::size_t n_freq, const std::size_t n_ant,
    const float spacing_m, const std::size_t total) {
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    // total == window_count * n_freq * n_ant
    const std::size_t element = linear % n_ant;
    const std::size_t freq = (linear / n_ant) % n_freq;
    const std::size_t window = linear / (n_freq * n_ant);

    const float3 direction =
        load_window_direction_device(window_directions, window);
    // direction.z is the n component; direction.x/y are l/m. The CPU delay uses
    // direction[0..2] (l, m, n) dotted with position (x, y, z), so we pass the
    // float3 we loaded as (l, m, n) into tracker_weight_device unchanged.
    const float3 position =
        tracker_position_device(element, n_ant, spacing_m);
    float weight_real = 0.0F;
    float weight_imag = 0.0F;
    tracker_weight_device(position, direction, wavenumbers[freq],
                          &weight_real, &weight_imag);
    // CPU layout: all_weights[(window * n_freq + freq) * n_ant + element]
    all_weights[(window * n_freq + freq) * n_ant + element] =
        ComplexFloat{weight_real, weight_imag};
}

// ---------------------------------------------------------------------------
// Phase 1 — Kernel 2: accumulation. 3D grid over (window, time_in_window, freq).
// Each thread reads 64 weights and 64 packed voltages, performs the float MAC
// loop, and writes |sum|^2 to the intensity[] slot at intensity_index(time, f, 0).
// ---------------------------------------------------------------------------

__global__ void tracker_v2_accumulate_kernel(
    float* __restrict__ intensity,
    const ComplexFloat* __restrict__ all_weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams,
    const std::size_t integration_spectra) {
    // We launch a 1D grid over (window * integration_spectra * n_freq) cells.
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_cells =
        /* window_count */ (n_time + integration_spectra - 1) / integration_spectra
        * integration_spectra * n_freq;
    if (linear >= total_cells) {
        return;
    }
    const std::size_t freq = linear % n_freq;
    const std::size_t t_in_window = (linear / n_freq) % integration_spectra;
    const std::size_t window =
        (linear / n_freq) / integration_spectra;

    const std::size_t time = window * integration_spectra + t_in_window;
    if (time >= n_time) {
        return;  // partial final window: CPU does `if (time >= n_time) continue;`
    }

    const std::size_t weight_base = (window * n_freq + freq) * n_ant;
    const std::size_t voltage_base = (time * n_freq + freq) * n_ant;

    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    const ComplexFloat* __restrict__ w_ptr = all_weights + weight_base;
    const std::uint8_t* __restrict__ v = packed + voltage_base;
    for (std::size_t element = 0; element < n_ant; ++element) {
        const float2 sample = unpack_complex_int4_device(v[element]);
        const float weight_real = w_ptr[element].real;
        const float weight_imag = w_ptr[element].imag;
        sum_real += weight_real * sample.x - weight_imag * sample.y;
        sum_imag += weight_real * sample.y + weight_imag * sample.x;
    }
    intensity[intensity_index_device(time, freq, 0, n_freq, n_beams)] =
        sum_real * sum_real + sum_imag * sum_imag;
}

// ---------------------------------------------------------------------------
// Phase 2 — Fused kernel. Computes the per-(window, freq) steering weight on the
// fly for each element inside the MAC loop, eliminating the device weight
// buffer and global-memory weight reads entirely.
//
// Thread mapping is identical to the Phase 1 accumulation kernel (1D grid over
// (window * integration_spectra * n_freq)). For each cell we look up the
// window direction once, then for each antenna recompute the weight in float32
// (after double phase + double transcendentals, narrowed to float) and apply it.
// The accumulation order over element matches the CPU (0 .. n_ant-1).
// ---------------------------------------------------------------------------

__global__ void tracker_v2_fused_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams,
    const std::size_t integration_spectra,
    const float spacing_m) {
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_cells =
        (n_time + integration_spectra - 1) / integration_spectra
        * integration_spectra * n_freq;
    if (linear >= total_cells) {
        return;
    }
    const std::size_t freq = linear % n_freq;
    const std::size_t t_in_window = (linear / n_freq) % integration_spectra;
    const std::size_t window = (linear / n_freq) / integration_spectra;
    const std::size_t time = window * integration_spectra + t_in_window;
    if (time >= n_time) {
        return;
    }

    const float3 direction =
        load_window_direction_device(window_directions, window);
    const double wave_number = wavenumbers[freq];
    const std::size_t voltage_base = (time * n_freq + freq) * n_ant;
    const std::uint8_t* __restrict__ v = packed + voltage_base;

    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t element = 0; element < n_ant; ++element) {
        const float3 position =
            tracker_position_device(element, n_ant, spacing_m);
        float weight_real = 0.0F;
        float weight_imag = 0.0F;
        tracker_weight_device(position, direction, wave_number,
                              &weight_real, &weight_imag);
        const float2 sample = unpack_complex_int4_device(v[element]);
        sum_real += weight_real * sample.x - weight_imag * sample.y;
        sum_imag += weight_real * sample.y + weight_imag * sample.x;
    }
    intensity[intensity_index_device(time, freq, 0, n_freq, n_beams)] =
        sum_real * sum_real + sum_imag * sum_imag;
}

// ---------------------------------------------------------------------------
// Phase 3 — Warp-reduction kernel. One warp (32 threads) per (time, freq) cell;
// 64 antennas are split across the 32 lanes (each lane handles 2 antennas), and
// the per-lane partial (sum_real, sum_imag) is reduced across the warp via
// __shfl_down_sync. Lane 0 writes the final |sum|^2.
//
// Block layout: blockDim.x == 32 (one warp). blockIdx.x indexes cells. We launch
// multiple warps per block? No — one warp per block keeps the reduction trivial
// and avoids cross-warp cooperation; the SM packs the warps anyway.
// ---------------------------------------------------------------------------

__global__ void tracker_v2_warp_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams,
    const std::size_t integration_spectra,
    const float spacing_m) {
    // blockDim.x is the warp size (32). One warp -> one (time, freq) cell.
    const std::size_t cell = static_cast<std::size_t>(blockIdx.x);
    const std::size_t total_cells =
        (n_time + integration_spectra - 1) / integration_spectra
        * integration_spectra * n_freq;
    if (cell >= total_cells) {
        return;
    }
    const std::size_t freq = cell % n_freq;
    const std::size_t t_in_window = (cell / n_freq) % integration_spectra;
    const std::size_t window = (cell / n_freq) / integration_spectra;
    const std::size_t time = window * integration_spectra + t_in_window;
    if (time >= n_time) {
        return;
    }

    const float3 direction =
        load_window_direction_device(window_directions, window);
    const double wave_number = wavenumbers[freq];
    const std::size_t voltage_base = (time * n_freq + freq) * n_ant;
    const std::uint8_t* __restrict__ v = packed + voltage_base;

    const unsigned int lane = static_cast<unsigned int>(threadIdx.x);
    const std::size_t warpsize = 32;  // == blockDim.x

    // Each lane accumulates antennas [lane, lane+warpsize, ...] until n_ant
    // covered. For n_ant==64 each lane handles exactly 2 antennas.
    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t element = lane; element < n_ant; element += warpsize) {
        const float3 position =
            tracker_position_device(element, n_ant, spacing_m);
        float weight_real = 0.0F;
        float weight_imag = 0.0F;
        tracker_weight_device(position, direction, wave_number,
                              &weight_real, &weight_imag);
        const float2 sample = unpack_complex_int4_device(v[element]);
        sum_real += weight_real * sample.x - weight_imag * sample.y;
        sum_imag += weight_real * sample.y + weight_imag * sample.x;
    }

    // Warp-shuffle reduction. Strictly pairwise in descending offsets, which is
    // associative-commutative over float adds but is NOT bit-identical to the
    // CPU's left-to-right serial accumulation. Phase 3 intentionally trades
    // exact bit equality for speed; the parity test against it must therefore
    // use a tolerance. (Phase 1/2 remain exact.) The shuffle uses the full-warp
    // mask 0xFFFFFFFF.
    const unsigned int full_mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_real = __shfl_down_sync(full_mask, sum_real, offset);
        const float other_imag = __shfl_down_sync(full_mask, sum_imag, offset);
        sum_real += other_real;
        sum_imag += other_imag;
    }
    if (lane == 0) {
        intensity[intensity_index_device(time, freq, 0, n_freq, n_beams)] =
            sum_real * sum_real + sum_imag * sum_imag;
    }
}

// ---------------------------------------------------------------------------
// Host-side static helpers shared by every phase's launch path.
// ---------------------------------------------------------------------------

// CPU-side reproduction of validate_opt_v2_inputs' pre-scan: computes every
// window direction (and throws on the first off-disk window) AND the per-freq
// wavenumber, on the host, before any kernel launch. Returns the directions
// packed as [window][0..2] floats for `cudaMemcpy` to the device.
void host_pre_scan(const Dimensions& dims, const TrackerConfig& tracker,
                   std::vector<float>& window_directions_flat,
                   std::vector<double>& wavenumbers) {
    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    window_directions_flat.resize(window_count * 3);
    for (std::size_t w = 0; w < window_count; ++w) {
        const Vec3 direction = tracker_window_direction(
            tracker.trajectory, w, tracker.integration_spectra);
        window_directions_flat[w * 3 + 0] = direction[0];
        window_directions_flat[w * 3 + 1] = direction[1];
        window_directions_flat[w * 3 + 2] = direction[2];
    }
    // Reproduce the same validate_opt_v2_inputs side effects / throws that the
    // CPU path enforces, so the GPU entry parity-rejects the same bad inputs.
    // We delegate to the existing beamformer::validate_opt_v2_inputs via the
    // allocate-and-return CPU entry, but doing that would also allocate the
    // whole Intensities cube — to keep the GPU path lean we replicate only the
    // validate_dimensions + tracker-specific guards here.
    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (tracker.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
    }
    const auto& start = tracker.trajectory.direction_start;
    const double norm_squared =
        static_cast<double>(start[0]) * start[0]
        + static_cast<double>(start[1]) * start[1]
        + static_cast<double>(start[2]) * start[2];
    if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument(
            "tracker direction_start must be a finite unit vector");
    }
    for (const float component : tracker.trajectory.direction_rate_per_sample) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "tracker direction_rate_per_sample must be finite");
        }
    }

    // Frequencies + wavenumbers (double, same formula as the CPU TU).
    const auto frequencies = channelized_frequencies(dims.n_freq);
    wavenumbers.resize(dims.n_freq);
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        if (!std::isfinite(frequencies[f]) || frequencies[f] <= 0.0F) {
            throw std::invalid_argument("frequencies must be positive and finite");
        }
        wavenumbers[f] =
            two_pi * static_cast<double>(frequencies[f]) / speed_of_light_m_per_s;
    }
}

void validate_gpu_size_inputs(const PackedVoltage& packed, const Dimensions& dims,
                              const Intensities& intensity) {
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    const std::size_t required_output =
        dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
}

// Round `count` up to even blocks of 256 threads and return the 1D grid size.
unsigned int blocks_for(const std::size_t count, const std::size_t threads = 256) {
    const std::size_t blocks =
        (count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA tracker grid exceeds supported one-dimensional size");
    }
    return static_cast<unsigned int>(blocks);
}

// ---------------------------------------------------------------------------
// Public dispatcher: launches the selected kernel and copies results back.
// ---------------------------------------------------------------------------

void run_tracker_v2(const PackedVoltage& packed, const Dimensions& dims,
                    const TrackerConfig& tracker, Intensities& intensity,
                    const CudaTrackerKernelV2 kernel) {
    validate_gpu_size_inputs(packed, dims, intensity);

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan(dims, tracker, window_directions_flat, wavenumbers);

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    constexpr float spacing_m = default_spacing_m;
    const std::size_t voltage_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;

    // Device buffers.
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    ComplexFloat* d_all_weights = nullptr;  // Phase 1 only

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed),
                          voltage_count * sizeof(std::uint8_t)),
               "cudaMalloc d_packed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_intensity),
                          output_count * sizeof(float)),
               "cudaMalloc d_intensity");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_window_directions),
                          window_directions_flat.size() * sizeof(float)),
               "cudaMalloc d_window_directions");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_wavenumbers),
                          wavenumbers.size() * sizeof(double)),
               "cudaMalloc d_wavenumbers");

    // Zero the output cube (the partial-window guard skips some slots; the CPU
    // fills exactly the live range, but the test compares whole cubes so we
    // must leave the un-run slots identical to the CPU, which also leaves the
    // suffix untouched via its `if (time >= n_time) continue` path — i.e.
    // un-run slots keep whatever the CPU's Intensities vector was initialized to
    // in the test. To match exactly we initialize to 0.0F on device too).
    check_cuda(cudaMemset(d_intensity, 0, output_count * sizeof(float)),
               "cudaMemset d_intensity");

    check_cuda(cudaMemcpy(d_packed, packed.data(),
                          voltage_count * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy packed H2D");
    check_cuda(cudaMemcpy(d_window_directions, window_directions_flat.data(),
                          window_directions_flat.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy window_directions H2D");
    check_cuda(cudaMemcpy(d_wavenumbers, wavenumbers.data(),
                          wavenumbers.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy wavenumbers H2D");

    constexpr std::size_t threads = 256;

    if (kernel == CudaTrackerKernelV2::TwoPass) {
        const std::size_t weight_count =
            window_count * dims.n_freq * dims.n_ant;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_all_weights),
                              weight_count * sizeof(ComplexFloat)),
                   "cudaMalloc d_all_weights");
        const unsigned int weight_blocks = blocks_for(weight_count);
        tracker_v2_weights_kernel_flat<<<weight_blocks,
                                         static_cast<unsigned int>(threads), 0, 0>>>(
            d_all_weights, d_window_directions, d_wavenumbers,
            window_count, dims.n_freq, dims.n_ant, spacing_m, weight_count);
        check_cuda(cudaGetLastError(), "tracker_v2_weights_kernel_flat launch");
        check_cuda(cudaDeviceSynchronize(), "weights kernel sync");

        const std::size_t acc_count =
            window_count * tracker.integration_spectra * dims.n_freq;
        const unsigned int acc_blocks = blocks_for(acc_count);
        tracker_v2_accumulate_kernel<<<acc_blocks,
                                       static_cast<unsigned int>(threads), 0, 0>>>(
            d_intensity, d_all_weights, d_packed,
            dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
            tracker.integration_spectra);
        check_cuda(cudaGetLastError(), "tracker_v2_accumulate_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "accumulate kernel sync");
    } else if (kernel == CudaTrackerKernelV2::Fused) {
        const std::size_t cell_count =
            window_count * tracker.integration_spectra * dims.n_freq;
        const unsigned int cell_blocks = blocks_for(cell_count);
        tracker_v2_fused_kernel<<<cell_blocks,
                                  static_cast<unsigned int>(threads), 0, 0>>>(
            d_intensity, d_window_directions, d_wavenumbers, d_packed,
            dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
            tracker.integration_spectra, spacing_m);
        check_cuda(cudaGetLastError(), "tracker_v2_fused_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "fused kernel sync");
    } else if (kernel == CudaTrackerKernelV2::WarpReduction) {
        const std::size_t cell_count =
            window_count * tracker.integration_spectra * dims.n_freq;
        if (cell_count > std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error(
                "CUDA tracker warp grid exceeds supported one-dimensional size");
        }
        const unsigned int warp_blocks = static_cast<unsigned int>(cell_count);
        constexpr unsigned int warp_threads = 32;
        tracker_v2_warp_kernel<<<warp_blocks, warp_threads, 0, 0>>>(
            d_intensity, d_window_directions, d_wavenumbers, d_packed,
            dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
            tracker.integration_spectra, spacing_m);
        check_cuda(cudaGetLastError(), "tracker_v2_warp_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "warp kernel sync");
    } else {
        throw std::invalid_argument("unknown CUDA tracker v2 kernel selector");
    }

    check_cuda(cudaMemcpy(intensity.data(), d_intensity,
                          output_count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy intensity D2H");

    if (d_all_weights != nullptr) {
        cudaFree(d_all_weights);
    }
    cudaFree(d_packed);
    cudaFree(d_intensity);
    cudaFree(d_window_directions);
    cudaFree(d_wavenumbers);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API (header surface).
// ---------------------------------------------------------------------------

Intensities cuda_tracker_v2_packed_intensity(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, const CudaTrackerKernelV2 kernel) {
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    cuda_tracker_v2_packed_intensity_into(packed, dims, tracker, intensity, kernel);
    return intensity;
}

void cuda_tracker_v2_packed_intensity_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const CudaTrackerKernelV2 kernel) {
    run_tracker_v2(packed, dims, tracker, intensity, kernel);
}

} // namespace beamformer
