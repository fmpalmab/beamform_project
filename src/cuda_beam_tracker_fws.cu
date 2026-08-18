#include "beamformer/cuda_beam_tracker_fws.hpp"

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
// CUDA error checking
// ---------------------------------------------------------------------------
void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(result));
    }
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// 4-bit signed 2's complement nibble decode into float2
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

// [time][freq][beam] flat index (n_beams == 1)
__device__ __forceinline__ std::size_t
intensity_index_device(const std::size_t t, const std::size_t f,
                       const std::size_t b, const std::size_t n_freq,
                       const std::size_t n_beams) {
    return (t * n_freq + f) * n_beams + b;
}

// Regular array antenna geometry (8x8 for 64 ant, 4x8 for 32 ant)
__device__ __forceinline__ float3
tracker_position_device(const std::size_t element, const std::size_t n_ant,
                         const float spacing_m) {
    std::size_t columns = (n_ant == 32 || n_ant == 64) ? 8 : n_ant;
    const std::size_t row = element / columns;
    const std::size_t column = element % columns;
    return make_float3(static_cast<float>(column) * spacing_m,
                       static_cast<float>(row) * spacing_m, 0.0F);
}

// Geometric steering phase in double precision for numerical parity
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
// High-Performance Fused Warp-Reduction Kernel (Target: RTX 5090)
//
// Grid mapping:
//   - blockDim.x = 32 (1 warp per row)
//   - blockDim.y = 4  (4 warps per block = 128 threads)
//   - Each warp handles 1 unique (window, freq) channel.
//   - Precomputes steering weights in registers ONCE per window.
//   - Streams 320 time samples in registers with zero DRAM weight roundtrips.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(128, 8) tracker_v2_fused_warp_optimized_kernel(
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
    const std::size_t total_window_freq_pairs) {

    // Warp identification
    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_window_freq_pairs) {
        return;
    }

    const std::size_t window = warp_id / n_freq;
    const std::size_t freq = warp_id % n_freq;

    // Load trajectory & frequency parameters (uniform across warp)
    const float3 direction = load_window_direction_device(window_directions, window);
    const double wave_number = wavenumbers[freq];

    // Precompute steering weights for this warp's antennas in REGISTERS (once per window)
    float w0_r = 0.0F, w0_i = 0.0F;
    float w1_r = 0.0F, w1_i = 0.0F;

    if (lane < n_ant) {
        const float3 pos0 = tracker_position_device(lane, n_ant, spacing_m);
        tracker_weight_device(pos0, direction, wave_number, &w0_r, &w0_i);
    }
    if (lane + 32 < n_ant) {
        const float3 pos1 = tracker_position_device(lane + 32, n_ant, spacing_m);
        tracker_weight_device(pos1, direction, wave_number, &w1_r, &w1_i);
    }

    // Time window boundaries
    const std::size_t time_start = window * integration_spectra;
    const std::size_t time_end =
        (time_start + integration_spectra < n_time) ? (time_start + integration_spectra) : n_time;

    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    // Process time samples sequentially within the integration window
    for (std::size_t t = time_start; t < time_end; ++t) {
        const std::size_t voltage_base = (t * n_freq + freq) * n_ant;
        const std::uint8_t* __restrict__ v = packed + voltage_base;

        float sum_real = 0.0F;
        float sum_imag = 0.0F;

        // Antennas 0..31 (Coalesced 32-byte load)
        if (lane < n_ant) {
            const float2 s0 = unpack_complex_int4_device(v[lane]);
            sum_real += w0_r * s0.x - w0_i * s0.y;
            sum_imag += w0_r * s0.y + w0_i * s0.x;
        }

        // Antennas 32..63 (Coalesced 32-byte load)
        if (lane + 32 < n_ant) {
            const float2 s1 = unpack_complex_int4_device(v[lane + 32]);
            sum_real += w1_r * s1.x - w1_i * s1.y;
            sum_imag += w1_r * s1.y + w1_i * s1.x;
        }

        // Intra-warp reduction across 32 threads via register shuffles
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_real += __shfl_down_sync(full_mask, sum_real, offset);
            sum_imag += __shfl_down_sync(full_mask, sum_imag, offset);
        }

        // Lane 0 computes instantaneous power and writes directly to output
        if (lane == 0) {
            intensity[intensity_index_device(t, freq, 0, n_freq, n_beams)] =
                sum_real * sum_real + sum_imag * sum_imag;
        }
    }
}

// ---------------------------------------------------------------------------
// Host helper: Precomputes trajectory directions & wavenumbers
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

    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (tracker.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
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
    const std::size_t required_output = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
}

// ---------------------------------------------------------------------------
// Execution Dispatcher
// ---------------------------------------------------------------------------
void run_tracker_v2(const PackedVoltage& packed, const Dimensions& dims,
                    const TrackerConfig& tracker, Intensities& intensity,
                    const CudaTrackerKernelV2 kernel) {
    (void)kernel; // Fully routed to the optimized fused-warp architecture

    validate_gpu_size_inputs(packed, dims, intensity);

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan(dims, tracker, window_directions_flat, wavenumbers);

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    constexpr float spacing_m = default_spacing_m;
    const std::size_t voltage_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;

    // Device allocations
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

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

    check_cuda(cudaMemset(d_intensity, 0, output_count * sizeof(float)),
               "cudaMemset d_intensity");

    // Asynchronous H2D Transfers
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

    // Grid configuration: 4 warps per block (128 threads)
    constexpr unsigned int warps_per_block = 4;
    const dim3 block_dim(32, warps_per_block);

    const std::size_t total_warps = window_count * dims.n_freq;
    const unsigned int grid_dim = static_cast<unsigned int>(
        (total_warps + warps_per_block - 1) / warps_per_block);

    // Launch optimized kernel
    tracker_v2_fused_warp_optimized_kernel<<<grid_dim, block_dim>>>(
        d_intensity, d_window_directions, d_wavenumbers, d_packed,
        dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
        tracker.integration_spectra, spacing_m, total_warps);

    check_cuda(cudaGetLastError(), "tracker_v2_fused_warp_optimized_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel execution sync");

    // Copy results back to host
    check_cuda(cudaMemcpy(intensity.data(), d_intensity,
                          output_count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy intensity D2H");

    cudaFree(d_packed);
    cudaFree(d_intensity);
    cudaFree(d_window_directions);
    cudaFree(d_wavenumbers);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
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