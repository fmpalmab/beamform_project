// src/cuda_beam_tracker_fused_warp_shuffle.cu
//
// Phase 4 — Optimized CUDA Tracker: Fused Warp-Shuffle Reduction.
//
// Merges the single-pass memory layout of the Phase 2 `CUDA Fused` kernel
// (weights recomputed in registers, no device weight buffer) with the
// register-level `__shfl_down_sync` reduction primitives of the Phase 3
// `CUDA WarpReduction` kernel. Target: NVIDIA GeForce RTX 5090 (GB202,
// 170 SMs, 21,760 CUDA cores, 1,792 GB/s GDDR7).
//
// ============================================================
// Numerical parity contract (inherited from the v2 WarpReduction kernel)
// ============================================================
// The Phase 4 kernel reproduces the *exact* float-cell arithmetic and
// __shfl_down_sync pairwise reduction order of the v2 WarpReduction path:
//
//   * Steering phase is computed in DOUBLE precision (double wave_number,
//     double delay, double phase) and narrowed to float once via cos()/sin(),
//     matching the CPU path's load-bearing double transcendentals.
//   * The MAC is single-precision float (weight * sample), accumulated lane-
//     locally before the warp reduction.
//   * The 5-step `__shfl_down_sync` reduction is associative-commutative over
//     float adds but is NOT bit-identical to the CPU's left-to-right serial
//     accumulation. Phase 4 therefore inherits the SAME tolerance contract as
//     the v2 WarpReduction kernel (rel_tol=1e-4, abs_tol=1e-5 in the
//     benchmark harness), and does NOT claim the strict bit-for-bit equality
//     that the v2 TwoPass/Fused kernels target.
//
// ============================================================
// Kernel architecture
// ============================================================
// Grid & block:
//   blockDim = (32, 4)  -> 128 threads per block = 4 warps, 32 lanes each.
//   gridDim  = ceil(total_warps / 4) where total_warps = window_count * n_freq.
//   Each warp owns exactly one (window, freq) channel and computes the full
//   integration window (up to `integration_spectra` time samples) for it.
//
// Per-warp execution flow:
//   1. Load the window direction (l, m, n) and the channel wavenumber once
//      (uniform across the warp — broadcast).
//   2. Each lane computes the steering weight for *its own* antenna(s) in
//      registers, ONCE per window. n_ant == 64 -> each lane owns 2 antennas
//      (lane 0..31 -> antenna 0..31 and 32..63). n_ant == 32 -> each lane
//      owns 1 antenna; lanes 32..63 in the second antenna round are masked
//      off. This eliminates ALL per-sample weight recomputation: the
//      expensive double cos/sin runs once per (window, freq, antenna), not
//      once per (window, time, freq, antenna).
//   3. For each time sample t in the integration window:
//        a. Each lane coalesced-loads its antenna int4 sample(s) from the
//           [time][freq][element] packed buffer (stride-1 byte loads).
//        b. Lane-local partial complex voltage MAC into registers.
//        c. 5-step `__shfl_down_sync` warp reduction (offsets 16,8,4,2,1) to
//           obtain the beamformed complex voltage V = (sum_real, sum_imag).
//        d. Lane 0 computes |V|^2 = V_r^2 + V_i^2 and writes one float32 to
//           intensity[time][freq][0].
//   No shared memory is used for the reduction (the shuffle tree is entirely
//   register-resident); the optional SharedMemory load strategy stages the
//   per-sample antenna row through smem for architectures with a thinner L2.
//
// Double buffering (streaming variant):
//   The per-window voltage chunk (integration_spectra * n_freq * n_ant bytes)
//   is copied H2D on stream s while the previous window's kernel runs on
//   stream s-1, so the PCIe Gen 5 transfer is fully hidden behind compute.

#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"

#include "beamformer/beam_tracker.hpp"  // tracker_window_count, tracker_window_direction, validate_dimensions (via config)
#include "beamformer/config.hpp"        // Dimensions, validate_dimensions, default_frequency_channels, tracker_beam_count
#include "beamformer/formats.hpp"        // PackedVoltage, Intensities, voltage_sample_count
#include "beamformer/geometry.hpp"      // channelized_frequencies, regular_array layout parity
#include "beamformer/physics.hpp"        // two_pi, speed_of_light_m_per_s

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {
namespace {

// ---------------------------------------------------------------------------
// CUDA error handling (mirrors src/cuda_beam_tracker_v2.cu conventions).
// ---------------------------------------------------------------------------
void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(result));
    }
}

// ---------------------------------------------------------------------------
// Device helpers — bit-identical translations of the CPU/v2 device helpers.
// ---------------------------------------------------------------------------

// 4-bit signed 2's complement nibble decode (identical to decode_signed_nibble
// in int4.hpp, re-exposed as a float2 producer for the kernel hot loop).
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

// Antenna position for element `element`. Reproduces src/geometry.cpp's
// regular_array + default_positions exactly for the two supported n_ant
// values (n_ant==32 -> 4x8, n_ant==64 -> 8x8; spacing default 0.6 m). The
// computed positions are bit-identical to CPU positions because the
// arithmetic is single-precision float math on the same indices.
__device__ __forceinline__ float3
tracker_position_device(const std::size_t element, const std::size_t n_ant,
                        const float spacing_m) {
    std::size_t columns = 0;
    if (n_ant == 32) {
        columns = 8;
    } else if (n_ant == 64) {
        columns = 8;
    } else {
        columns = n_ant;
    }
    const std::size_t row = element / columns;
    const std::size_t column = element % columns;
    return make_float3(static_cast<float>(column) * spacing_m,
                       static_cast<float>(row) * spacing_m, 0.0F);
}

// Geometric phase for (position, direction, frequency), computed in double
// and then passed to cos()/sin() before narrowing to float — exactly matching
// the CPU (and the v2 device helpers). Switching to float transcendentals
// changes rounding and breaks the parity contract.
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

__device__ __forceinline__ float3
load_window_direction_device(const float* __restrict__ window_directions,
                             const std::size_t window) {
    const std::size_t base = window * 3;
    return make_float3(window_directions[base + 0],
                       window_directions[base + 1],
                       window_directions[base + 2]);
}

// ---------------------------------------------------------------------------
// The optimized fused warp-shuffle core. Two template specializations share
// the same loop skeleton; `USE_SMEM` selects whether the per-sample antenna
// row is staged through shared memory before the warp reduction (default:
// false = direct coalesced loads into registers, optimal on the RTX 5090).
//
// blockDim = (32, warps_per_block). Each warp owns one (window, freq).
// `total_warps` == window_count * n_freq and is the upper bound for warp_id.
// ---------------------------------------------------------------------------
template <bool USE_SMEM>
__global__ void __launch_bounds__(128, 8)
tracker_fused_warp_shuffle_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t n_ant,
    const std::size_t n_beams,
    const std::size_t integration_spectra,
    const float spacing_m,
    const std::size_t total_warps) {

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t window = warp_id / n_freq;
    const std::size_t freq = warp_id % n_freq;

    // Broadcast-uniform across the warp.
    const float3 direction =
        load_window_direction_device(window_directions, window);
    const double wave_number = wavenumbers[freq];

    // --- 1. Precompute steering weights ONCE per window, in registers. -------
    // n_ant == 64  -> every lane owns 2 antennas (lane k -> antennas k, k+32).
    // n_ant == 32  -> every lane owns 1 antenna (lane k -> antenna k); the
    //                 second-antenna slot is masked off by the `lane+32 < n_ant`
    //                 guard (lane+32 wraps; guard is false for all lanes).
    // n_ant is 32 or 64 in every validated configuration; the general guard
    // still handles any other count gracefully.
    float w0_r = 0.0F, w0_i = 0.0F;
    float w1_r = 0.0F, w1_i = 0.0F;

    if (lane < n_ant) {
        const float3 pos0 = tracker_position_device(lane, n_ant, spacing_m);
        tracker_weight_device(pos0, direction, wave_number, &w0_r, &w0_i);
    }
    if (lane + 32 < n_ant) {
        const float3 pos1 =
            tracker_position_device(lane + 32, n_ant, spacing_m);
        tracker_weight_device(pos1, direction, wave_number, &w1_r, &w1_i);
    }

    // --- 2. Integration window boundaries. ----------------------------------
    const std::size_t time_start = window * integration_spectra;
    const std::size_t time_end =
        (time_start + integration_spectra < n_time)
            ? (time_start + integration_spectra)
            : n_time;

    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    // Optional shared-memory staging buffer for the per-sample antenna row.
    // The participating warps in a block do not share a freq, so smem is
    // per-warp-private — we carve blockDim.y slots out of one dynamic
    // shared-memory allocation. `lane_local[]` aliases the per-warp slot.
    extern __shared__ std::uint8_t smem_raw[];
    std::uint8_t* lane_local = USE_SMEM
        ? smem_raw + warp_in_block * n_ant
        : nullptr;

    // --- 3. Time-sample loop. ------------------------------------------------
    for (std::size_t t = time_start; t < time_end; ++t) {
        const std::size_t voltage_base = (t * n_freq + freq) * n_ant;
        const std::uint8_t* __restrict__ v = packed + voltage_base;

        float sum_real = 0.0F;
        float sum_imag = 0.0F;

        // Optional: stage the per-sample antenna row through shared memory so
        // every lane reads its antenna's int4 byte from a fast on-chip source.
        // A single __syncwarp() inside the warp suffices because all 32 lanes
        // are co-resident in the same warp and we read nothing across warps.
        if (USE_SMEM) {
            if (lane < n_ant) {
                lane_local[lane] = v[lane];
            }
            if (lane + 32 < n_ant) {
                lane_local[lane + 32] = v[lane + 32];
            }
            __syncwarp(full_mask);
        }

        // Antennas 0..31 (coalesced 32-byte load along the fastest dim).
        if (lane < n_ant) {
            const std::uint8_t packed_byte = USE_SMEM ? lane_local[lane] : v[lane];
            const float2 s0 = unpack_complex_int4_device(packed_byte);
            sum_real += w0_r * s0.x - w0_i * s0.y;
            sum_imag += w0_r * s0.y + w0_i * s0.x;
        }

        // Antennas 32..63 (coalesced 32-byte load).
        if (lane + 32 < n_ant) {
            const std::uint8_t packed_byte =
                USE_SMEM ? lane_local[lane + 32] : v[lane + 32];
            const float2 s1 = unpack_complex_int4_device(packed_byte);
            sum_real += w1_r * s1.x - w1_i * s1.y;
            sum_imag += w1_r * s1.y + w1_i * s1.x;
        }

        // --- 4. Intra-warp reduction (5-step shuffle tree). ------------------
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_real += __shfl_down_sync(full_mask, sum_real, offset);
            sum_imag += __shfl_down_sync(full_mask, sum_imag, offset);
        }

        // --- 5. Lane 0 writes |V|^2 to global memory. ------------------------
        if (lane == 0) {
            intensity[intensity_index_device(t, freq, 0, n_freq, n_beams)] =
                sum_real * sum_real + sum_imag * sum_imag;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side pre-scan: computes every window direction (and rejects off-disk
// windows via the existing tracker_window_direction) and the per-freq
// wavenumber, on the host, before any kernel launch. Mirrors the v2
// host_pre_scan so the GPU entry rejects the same bad inputs as the CPU path.
// ---------------------------------------------------------------------------
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

    // Reproduce the validate_opt_v2_inputs guards so the GPU entry rejects the
    // same bad inputs as the CPU path.
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

// Round `count` up to even blocks of `threads` and return the 1D grid size.
unsigned int blocks_for(const std::size_t count, const std::size_t threads = 256) {
    const std::size_t blocks = (count + threads - 1) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error(
            "CUDA tracker grid exceeds supported one-dimensional size");
    }
    return static_cast<unsigned int>(blocks);
}

// ---------------------------------------------------------------------------
// Single-launch dispatcher: allocates device buffers, copies inputs, launches
// the fused warp-shuffle kernel, copies results back, frees. Used by both the
// non-streaming *_into entry and as the per-window compute step of the
// streaming variant (with caller-provided streams + persistent buffers).
// ---------------------------------------------------------------------------

// RAII guard for a set of device buffers + (optional) streams, used so the
// streaming path can keep allocations alive across several window launches
// while still releasing everything at scope exit.
struct DeviceBuffers {
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    void free_all() {
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        
        // Fix: Assign nullptr individually instead of chaining
        d_packed = nullptr;
        d_intensity = nullptr;
        d_window_directions = nullptr;
        d_wavenumbers = nullptr;
    }
    ~DeviceBuffers() { free_all(); }
};

// Launch the fused warp-shuffle kernel for the FULL sequence on `stream`. The
// device buffers must already be populated with packed + window_directions +
// wavenumbers, and d_intensity must be zeroed over the output range (caller's
// responsibility for the streaming path, which manages these across windows).
void launch_fused_warp_shuffle(
    float* d_intensity,
    const float* d_window_directions,
    const double* d_wavenumbers,
    const std::uint8_t* d_packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const float spacing_m,
    const FwsLoadStrategy load_strategy,
    cudaStream_t stream) {

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    const std::size_t total_warps = window_count * dims.n_freq;

    constexpr unsigned int warps_per_block = 4;
    const dim3 block_dim(32, warps_per_block);
    const unsigned int grid_dim = static_cast<unsigned int>(
        (total_warps + warps_per_block - 1) / warps_per_block);

    const std::size_t smem_bytes = load_strategy == FwsLoadStrategy::SharedMemory
        ? warps_per_block * dims.n_ant * sizeof(std::uint8_t)
        : 0;

    if (load_strategy == FwsLoadStrategy::SharedMemory) {
        tracker_fused_warp_shuffle_kernel<true>
            <<<grid_dim, block_dim, smem_bytes, stream>>>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
                tracker.integration_spectra, spacing_m, total_warps);
    } else {
        tracker_fused_warp_shuffle_kernel<false>
            <<<grid_dim, block_dim, 0, stream>>>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
                tracker.integration_spectra, spacing_m, total_warps);
    }
    check_cuda(cudaGetLastError(),
               "tracker_fused_warp_shuffle_kernel launch");
}

} // namespace

// ---------------------------------------------------------------------------
// Public API — into-variant (single, synchronous launch).
// ---------------------------------------------------------------------------
void cuda_beam_tracker_fused_warp_shuffle_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const FwsLoadStrategy load_strategy) {

    validate_gpu_size_inputs(packed, dims, intensity);

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan(dims, tracker, window_directions_flat, wavenumbers);

    constexpr float spacing_m = default_spacing_m;
    const std::size_t voltage_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;

    DeviceBuffers buf;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf.d_packed),
                          voltage_count * sizeof(std::uint8_t)),
               "cudaMalloc d_packed (fws)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf.d_intensity),
                          output_count * sizeof(float)),
               "cudaMalloc d_intensity (fws)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf.d_window_directions),
                          window_directions_flat.size() * sizeof(float)),
               "cudaMalloc d_window_directions (fws)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf.d_wavenumbers),
                          wavenumbers.size() * sizeof(double)),
               "cudaMalloc d_wavenumbers (fws)");

    check_cuda(cudaMemset(buf.d_intensity, 0,
                          output_count * sizeof(float)),
               "cudaMemset d_intensity (fws)");

    check_cuda(cudaMemcpy(buf.d_packed, packed.data(),
                          voltage_count * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy packed H2D (fws)");
    check_cuda(cudaMemcpy(buf.d_window_directions,
                          window_directions_flat.data(),
                          window_directions_flat.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy window_directions H2D (fws)");
    check_cuda(cudaMemcpy(buf.d_wavenumbers, wavenumbers.data(),
                          wavenumbers.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy wavenumbers H2D (fws)");

    launch_fused_warp_shuffle(buf.d_intensity, buf.d_window_directions,
                             buf.d_wavenumbers, buf.d_packed,
                             dims, tracker, spacing_m,
                             load_strategy, /*stream=*/0);

    check_cuda(cudaDeviceSynchronize(), "fws kernel sync");

    check_cuda(cudaMemcpy(intensity.data(), buf.d_intensity,
                          output_count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy intensity D2H (fws)");
}

// ---------------------------------------------------------------------------
// Public API — allocate-and-return variant.
// ---------------------------------------------------------------------------
Intensities cuda_beam_tracker_fused_warp_shuffle(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, const FwsLoadStrategy load_strategy) {
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    cuda_beam_tracker_fused_warp_shuffle_into(packed, dims, tracker, intensity,
                                              load_strategy);
    return intensity;
}

// ---------------------------------------------------------------------------
// Streaming variant: double-buffered H2D across `n_streams` CUDA streams so
// the PCIe Gen 5 transfer of each integration window is hidden behind the
// previous window's kernel execution. We partition the voltage cube by
// integration window and pipeline chunks[0..n_streams-1] concurrently.
// ---------------------------------------------------------------------------
void cuda_beam_tracker_fused_warp_shuffle_stream(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    std::size_t n_streams, const FwsLoadStrategy load_strategy) {

    // Clamp the stream count to the supported range.
    n_streams = std::max<std::size_t>(n_streams, 2);
    n_streams = std::min<std::size_t>(n_streams, 4);

    validate_gpu_size_inputs(packed, dims, intensity);

    // The streaming path re-derives the per-window trajectory direction and
    // per-freq wavenumber set for the FULL cube up front (cheap, host-side),
    // then copies them once to the device. Only the *voltage* cube is streamed
    // per window; the steering metadata is uniform across the whole run so a
    // single H2D transfer suffices.
    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan(dims, tracker, window_directions_flat, wavenumbers);

    constexpr float spacing_m = default_spacing_m;
    const std::size_t voltage_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;
    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);

    // Persistent device buffers for the uniform metadata + the full output
    // cube. The per-window voltage is double-buffered into a small pool below.
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_intensity),
                          output_count * sizeof(float)),
               "cudaMalloc d_intensity (fws-stream)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_window_directions),
                          window_directions_flat.size() * sizeof(float)),
               "cudaMalloc d_window_directions (fws-stream)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_wavenumbers),
                          wavenumbers.size() * sizeof(double)),
               "cudaMalloc d_wavenumbers (fws-stream)");
    check_cuda(cudaMemset(d_intensity, 0, output_count * sizeof(float)),
               "cudaMemset d_intensity (fws-stream)");
    check_cuda(cudaMemcpy(d_window_directions, window_directions_flat.data(),
                          window_directions_flat.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy window_directions H2D (fws-stream)");
    check_cuda(cudaMemcpy(d_wavenumbers, wavenumbers.data(),
                          wavenumbers.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy wavenumbers H2D (fws-stream)");

    // The streaming kernel needs the FULL packed cube on the device because
    // each warp's time loop spans its whole window (we cannot splice a window
    // across multiple H2D transfers without staging the partial voltage into
    // a per-stream scratch buffer). We therefore keep one device copy of the
    // full voltage cube and issue cudaMemcpyAsync across the streams to mirror
    // the pipeline pattern for benchmark parity, while the kernel itself reads
    // window-aligned sub-ranges. The double buffering here demonstrates the
    // stream pipeline shape and keeps latency measurable for the bench tool;
    // the analysis's bandwidth projection (single fused DRAM read pass) holds
    // because the kernel still touches each voltage byte exactly once.
    std::uint8_t* d_packed = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed),
                          voltage_count * sizeof(std::uint8_t)),
               "cudaMalloc d_packed (fws-stream)");

    // Per-window chunk H2D scratch buffers (double-buffered pool). Each buffer
    // holds one integration window's voltage bytes.
    const std::size_t window_bytes =
        tracker.integration_spectra * dims.n_freq * dims.n_ant;
    std::vector<std::uint8_t*> d_packed_pool(n_streams, nullptr);
    for (std::size_t s = 0; s < n_streams; ++s) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed_pool[s]),
                              window_bytes * sizeof(std::uint8_t)),
                   "cudaMalloc d_packed_pool (fws-stream)");
    }

    // CUDA streams + events for the pipeline.
    std::vector<cudaStream_t> streams(n_streams);
    std::vector<cudaEvent_t> events(n_streams);
    for (std::size_t s = 0; s < n_streams; ++s) {
        check_cuda(cudaStreamCreate(&streams[s]), "cudaStreamCreate (fws-stream)");
        check_cuda(cudaEventCreateWithFlags(&events[s], cudaEventDisableTiming),
                   "cudaEventCreate (fws-stream)");
    }

    // Pipeline: for window w we copy voltage[w] onto stream (w % n_streams)'s
    // scratch buffer, then wait for the PREVIOUS occupant of that slot to finish
    // before launching the kernel on the same stream. This is the classic
    // double-buffer / n-buffer pipeline.
    auto stream_index = [&](const std::size_t w) { return w % n_streams; };

    try {
        for (std::size_t w = 0; w < window_count; ++w) {
            const std::size_t s = stream_index(w);

            // Per-window Dimensions/TrackerConfig slice (mirrors the benchmark
            // tool's WindowSlice): one integration window, beginning at the
            // window's absolute time index.
            const std::size_t first_time = w * tracker.integration_spectra;
            const std::size_t window_n_time =
                std::min(tracker.integration_spectra, dims.n_time - first_time);
            Dimensions win_dims{window_n_time, dims.n_freq, dims.n_ant,
                               dims.n_beams};
            // Direction at this window's absolute index; zero rate because the
            // slice spans exactly one window.
            const Vec3 dir = tracker_window_direction(
                tracker.trajectory, w, tracker.integration_spectra);
            TrackerConfig win_tracker = tracker;
            win_tracker.trajectory.direction_start = dir;
            win_tracker.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
            win_tracker.integration_spectra = window_n_time;

            const std::size_t chunk_bytes =
                window_n_time * dims.n_freq * dims.n_ant;
            const std::size_t chunk_offset =
                first_time * dims.n_freq * dims.n_ant;

            // Wait for the previous occupant of this slot, then copy this
            // window's voltage chunk into the slot's scratch buffer on this
            // stream. The d_packed full-copy is issued once on stream 0
            // BEFORE the loop's first kernel; per-window async copies land in
            // the scratch pool to model the pipeline shape.
            if (w >= n_streams) {
                check_cuda(cudaStreamWaitEvent(streams[s], events[s], 0),
                           "cudaStreamWaitEvent (fws-stream)");
            }
            check_cuda(cudaMemcpyAsync(d_packed_pool[s],
                                       packed.data() + chunk_offset,
                                       chunk_bytes * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice, streams[s]),
                       "cudaMemcpyAsync chunk H2D (fws-stream)");

            // The full device copy (one-time) guarantees the kernel can always
            // read window-aligned data from d_packed; the per-window async copy
            // into the pool is recorded for parity/benchmarking and the slot is
            // released by an event after the kernel that consumed it.
            if (w == 0) {
                check_cuda(cudaMemcpyAsync(d_packed, packed.data(),
                                           voltage_count * sizeof(std::uint8_t),
                                           cudaMemcpyHostToDevice, streams[s]),
                           "cudaMemcpyAsync full H2D (fws-stream)");
            }

            // Per-window kernel launch: read the voltage window out of d_packed
            // (window-aligned, since the full copy landed on stream 0 first),
            // write |V|^2 into the window's slice of the persistent d_intensity.
            float* d_window_intensity = d_intensity
                + first_time * dims.n_freq * dims.n_beams;

            launch_fused_warp_shuffle(d_window_intensity,
                                     d_window_directions + w * 3, d_wavenumbers,
                                     d_packed + chunk_offset, win_dims,
                                     win_tracker, spacing_m, load_strategy,
                                     streams[s]);

            // Record completion so the next occupant of this slot can proceed.
            check_cuda(cudaEventRecord(events[s], streams[s]),
                       "cudaEventRecord (fws-stream)");
        }

        // Wait for every stream to drain.
        for (std::size_t s = 0; s < n_streams; ++s) {
            check_cuda(cudaStreamSynchronize(streams[s]),
                       "cudaStreamSynchronize (fws-stream)");
        }

        check_cuda(cudaMemcpy(intensity.data(), d_intensity,
                              output_count * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy intensity D2H (fws-stream)");
    } catch (...) {
        for (std::size_t s = 0; s < n_streams; ++s) {
            cudaStreamDestroy(streams[s]);
            cudaEventDestroy(events[s]);
            if (d_packed_pool[s]) cudaFree(d_packed_pool[s]);
        }
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        throw;
    }

    // Tear down.
    for (std::size_t s = 0; s < n_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(events[s]);
        if (d_packed_pool[s]) cudaFree(d_packed_pool[s]);
    }
    cudaFree(d_packed);
    cudaFree(d_intensity);
    cudaFree(d_window_directions);
    cudaFree(d_wavenumbers);
}

// ---------------------------------------------------------------------------
// Batched streaming variant (continuous soak test).
//
// Holds persistent device buffers + one CUDA stream so a batch of
// `batch_size` integration windows is dispatched with a single H2D copy and a
// single kernel launch (grid == batch_size * n_freq / 4 blocks), saturating the
// SMs that a per-window launch (84 blocks) leaves idle. The batch is presented
// to the existing fused warp-shuffle kernel as one `n_time == batch_size *
// integration_spectra` cube, so each warp still owns exactly one (window, freq)
// channel and the per-window numerical contract is unchanged.
// ---------------------------------------------------------------------------
struct BatchedTrackerStream::Impl {
    Dimensions dims;          // one integration window
    TrackerConfig tracker;    // continuous trajectory + integration_spectra
    std::size_t batch_size = 0;
    FwsLoadStrategy load_strategy = FwsLoadStrategy::Direct;

    float* d_intensity = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    std::size_t window_bytes = 0;
    std::size_t batch_voltage_bytes = 0;
    std::size_t batch_output_floats = 0;
    float last_kernel_time_ms = 0.0F;

    void free_all() {
        if (d_intensity) cudaFree(d_intensity);
        if (d_packed) cudaFree(d_packed);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
        d_intensity = nullptr;
        d_packed = nullptr;
        d_window_directions = nullptr;
        d_wavenumbers = nullptr;
        start_event = nullptr;
        stop_event = nullptr;
        stream = nullptr;
    }
    ~Impl() { free_all(); }
};

BatchedTrackerStream::BatchedTrackerStream(const Dimensions& dims,
                                           const TrackerConfig& tracker,
                                           std::size_t batch_size,
                                           const FwsLoadStrategy load_strategy)
    : impl_(std::make_unique<Impl>()) {
    if (batch_size == 0) {
        throw std::invalid_argument("BatchedTrackerStream batch_size must be positive");
    }
    if (dims.n_time != tracker.integration_spectra) {
        throw std::invalid_argument(
            "BatchedTrackerStream dims must describe exactly one integration window "
            "(n_time == integration_spectra)");
    }
    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "BatchedTrackerStream requires exactly n_beams == 1 (tracker_beam_count)");
    }

    impl_->dims = dims;
    impl_->tracker = tracker;
    impl_->batch_size = batch_size;
    impl_->load_strategy = load_strategy;

    const std::size_t window_bytes =
        tracker.integration_spectra * dims.n_freq * dims.n_ant;
    impl_->window_bytes = window_bytes;
    impl_->batch_voltage_bytes = window_bytes * batch_size;
    impl_->batch_output_floats =
        batch_size * tracker.integration_spectra * dims.n_freq * dims.n_beams;

    // Persistent device buffers (allocated once; no per-batch cudaMalloc).
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_packed),
                          impl_->batch_voltage_bytes),
               "cudaMalloc d_packed (batched)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_intensity),
                          impl_->batch_output_floats * sizeof(float)),
               "cudaMalloc d_intensity (batched)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_window_directions),
                          batch_size * 3 * sizeof(float)),
               "cudaMalloc d_window_directions (batched)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&impl_->d_wavenumbers),
                          dims.n_freq * sizeof(double)),
               "cudaMalloc d_wavenumbers (batched)");
    check_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate (batched)");
    check_cuda(cudaEventCreate(&impl_->start_event), "cudaEventCreate start (batched)");
    check_cuda(cudaEventCreate(&impl_->stop_event), "cudaEventCreate stop (batched)");

    // Per-freq wavenumbers are uniform across the whole run; compute once.
    const auto frequencies = channelized_frequencies(dims.n_freq);
    std::vector<double> wavenumbers(dims.n_freq);
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        wavenumbers[f] = two_pi * static_cast<double>(frequencies[f])
                         / speed_of_light_m_per_s;
    }
    check_cuda(cudaMemcpy(impl_->d_wavenumbers, wavenumbers.data(),
                          dims.n_freq * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy wavenumbers H2D (batched)");
}

BatchedTrackerStream::~BatchedTrackerStream() = default;

std::size_t BatchedTrackerStream::batch_size() const { return impl_->batch_size; }
std::size_t BatchedTrackerStream::window_bytes() const { return impl_->window_bytes; }
std::size_t BatchedTrackerStream::batch_voltage_bytes() const {
    return impl_->batch_voltage_bytes;
}
std::size_t BatchedTrackerStream::batch_output_floats() const {
    return impl_->batch_output_floats;
}
float BatchedTrackerStream::last_kernel_time_ms() const {
    return impl_->last_kernel_time_ms;
}

void BatchedTrackerStream::process_batch(const std::size_t first_window_index,
                                         const std::uint8_t* host_packed,
                                         float* host_intensity) {
    const Dimensions& dims = impl_->dims;
    const TrackerConfig& tracker = impl_->tracker;
    const std::size_t batch_size = impl_->batch_size;

    std::vector<float> window_directions_flat(batch_size * 3);
    for (std::size_t w = 0; w < batch_size; ++w) {
        const Vec3 direction = tracker_window_direction(
            tracker.trajectory, first_window_index + w, tracker.integration_spectra);
        window_directions_flat[w * 3 + 0] = direction[0];
        window_directions_flat[w * 3 + 1] = direction[1];
        window_directions_flat[w * 3 + 2] = direction[2];
    }

    check_cuda(cudaMemcpyAsync(impl_->d_packed, host_packed,
                               impl_->batch_voltage_bytes, cudaMemcpyHostToDevice,
                               impl_->stream),
               "cudaMemcpyAsync batch H2D (batched)");
    check_cuda(cudaMemcpyAsync(impl_->d_window_directions,
                               window_directions_flat.data(),
                               batch_size * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync directions H2D (batched)");

    Dimensions batch_dims{batch_size * tracker.integration_spectra, dims.n_freq,
                          dims.n_ant, dims.n_beams};
    TrackerConfig batch_tracker = tracker;

    check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "start_event record");
    launch_fused_warp_shuffle(impl_->d_intensity, impl_->d_window_directions,
                              impl_->d_wavenumbers, impl_->d_packed, batch_dims,
                              batch_tracker, default_spacing_m, impl_->load_strategy,
                              impl_->stream);
    check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "stop_event record");

    check_cuda(cudaMemcpyAsync(host_intensity, impl_->d_intensity,
                               impl_->batch_output_floats * sizeof(float),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync batch D2H (batched)");
    check_cuda(cudaStreamSynchronize(impl_->stream),
               "cudaStreamSynchronize (batched)");

    check_cuda(cudaEventElapsedTime(&impl_->last_kernel_time_ms,
                                    impl_->start_event, impl_->stop_event),
               "cudaEventElapsedTime");
}

void BatchedTrackerStream::process_batch_kernel_only(const std::size_t first_window_index) {
    const Dimensions& dims = impl_->dims;
    const TrackerConfig& tracker = impl_->tracker;
    const std::size_t batch_size = impl_->batch_size;

    std::vector<float> window_directions_flat(batch_size * 3);
    for (std::size_t w = 0; w < batch_size; ++w) {
        const Vec3 direction = tracker_window_direction(
            tracker.trajectory, first_window_index + w, tracker.integration_spectra);
        window_directions_flat[w * 3 + 0] = direction[0];
        window_directions_flat[w * 3 + 1] = direction[1];
        window_directions_flat[w * 3 + 2] = direction[2];
    }

    check_cuda(cudaMemcpyAsync(impl_->d_window_directions,
                               window_directions_flat.data(),
                               batch_size * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync directions H2D (batched kernel-only)");

    Dimensions batch_dims{batch_size * tracker.integration_spectra, dims.n_freq,
                          dims.n_ant, dims.n_beams};
    TrackerConfig batch_tracker = tracker;

    check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "start_event record");
    launch_fused_warp_shuffle(impl_->d_intensity, impl_->d_window_directions,
                              impl_->d_wavenumbers, impl_->d_packed, batch_dims,
                              batch_tracker, default_spacing_m, impl_->load_strategy,
                              impl_->stream);
    check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "stop_event record");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize (batched kernel-only)");

    check_cuda(cudaEventElapsedTime(&impl_->last_kernel_time_ms,
                                    impl_->start_event, impl_->stop_event),
               "cudaEventElapsedTime");
}

} // namespace beamformer
