// src/cuda_upchannelizer.cu
//
// CUDA Upchannelizer for Beam Tracker and Voltage Beamformer Outputs.

#include "beamformer/cuda_upchannelizer.hpp"
#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double PI = 3.14159265358979323846;

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void validate_tracker(const Dimensions& dims, const TrackerConfig& tracker) {
    validate_dimensions(dims);
    if (tracker.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

float compute_window_sample(std::size_t m, std::size_t M, UpchannelizerWindowType type) {
    if (M <= 1) return 1.0F;
    const double x = static_cast<double>(m) / static_cast<double>(M);
    switch (type) {
    case UpchannelizerWindowType::Rectangular:
        return 1.0F;
    case UpchannelizerWindowType::Hann:
        return static_cast<float>(0.5 * (1.0 - std::cos(2.0 * PI * x)));
    case UpchannelizerWindowType::Hamming:
        return static_cast<float>(0.54 - 0.46 * std::cos(2.0 * PI * x));
    case UpchannelizerWindowType::Blackman:
        return static_cast<float>(0.42 - 0.5 * std::cos(2.0 * PI * x) + 0.08 * std::cos(4.0 * PI * x));
    }
    return 1.0F;
}

std::vector<float> generate_window_table(std::size_t M, UpchannelizerWindowType type) {
    std::vector<float> w(M);
    for (std::size_t m = 0; m < M; ++m) {
        w[m] = compute_window_sample(m, M, type);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Fast Device Bitfield Extraction Primitive (PTX bfe.s32)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 unpack_int4_fast(const std::uint32_t byte_val) {
#if defined(__CUDA_ARCH__)
    int r, i;
    asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
    asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
    return make_float2(__int2float_rn(r), __int2float_rn(i));
#else
    const int r = (static_cast<int>(byte_val) << 28) >> 28;
    const int i = ((static_cast<int>(byte_val) << 24) >> 28);
    return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
}

template <int N_ANT>
__device__ __forceinline__ float3 tracker_pos(const unsigned int element, const float spacing_m) {
    if constexpr (N_ANT == 32 || N_ANT == 64) {
        return make_float3(static_cast<float>(element & 7U) * spacing_m,
                           static_cast<float>(element >> 3U) * spacing_m, 0.0F);
    } else {
        return make_float3(static_cast<float>(element & 15U) * spacing_m,
                           static_cast<float>(element >> 4U) * spacing_m, 0.0F);
    }
}

// ---------------------------------------------------------------------------
// CUDA Upchannelizer Kernel (Complex Voltage -> Fine Intensities)
// ---------------------------------------------------------------------------
template <int M>
__global__ void __launch_bounds__(128, 4)
upchannelize_voltage_kernel(
    float* __restrict__ fine_intensity,
    const ComplexFloat* __restrict__ voltage,
    const float* __restrict__ window_lut,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t n_beams,
    const std::size_t n_fine_time,
    const std::size_t total_warps) {

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t beam = warp_id % n_beams;
    const std::size_t rest = warp_id / n_beams;
    const std::size_t freq = rest % n_freq;
    const std::size_t tau = rest / n_freq;

    if (tau >= n_fine_time) {
        return;
    }

    // Load sample m and apply window
    float val_r = 0.0F;
    float val_i = 0.0F;

    if (lane < M) {
        const std::size_t t = tau * M + lane;
        if (t < n_time) {
            const std::size_t v_idx = (t * n_freq + freq) * n_beams + beam;
            const ComplexFloat c = voltage[v_idx];
            const float w = window_lut[lane];
            val_r = c.real * w;
            val_i = c.imag * w;
        }
    }

    // Warp-level DFT across M samples for output channel k = lane
    float xr = 0.0F;
    float xi = 0.0F;

    if (lane < M) {
        const float k_f = static_cast<float>(lane);
        constexpr float two_pi_over_M = static_cast<float>(-2.0 * PI / static_cast<double>(M));

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float src_r = __shfl_sync(0xFFFFFFFFu, val_r, m);
            const float src_i = __shfl_sync(0xFFFFFFFFu, val_i, m);
            const float angle = two_pi_over_M * k_f * static_cast<float>(m);
            float s, c;
            __sincosf(angle, &s, &c);
            xr = fmaf(src_r, c, fmaf(-src_i, s, xr));
            xi = fmaf(src_r, s, fmaf(src_i, c, xi));
        }

        const float intensity = xr * xr + xi * xi;
        const std::size_t out_idx = (tau * (n_freq * M) + freq * M + lane) * n_beams + beam;
        fine_intensity[out_idx] = intensity;
    }
}

// ---------------------------------------------------------------------------
// CUDA Fused Tracker + Upchannelizer Kernel (Packed RFSoC -> Fine Intensities)
// ---------------------------------------------------------------------------
template <int N_ANT, int M>
__global__ void __launch_bounds__(128, 4)
tracker_upchannelize_kernel(
    float* __restrict__ fine_intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const float* __restrict__ window_lut,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t integration_spectra,
    const std::size_t n_fine_time,
    const float spacing_m,
    const std::size_t total_warps) {

    constexpr unsigned int ANT_PER_LANE = static_cast<unsigned int>(N_ANT / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t freq = warp_id % n_freq;
    const std::size_t tau = warp_id / n_freq;

    if (tau >= n_fine_time) {
        return;
    }

    const std::size_t t_start = tau * M;
    const std::size_t window = t_start / integration_spectra;

    const float3 direction = make_float3(
        window_directions[window * 3 + 0],
        window_directions[window * 3 + 1],
        window_directions[window * 3 + 2]);
    const double wave_number = wavenumbers[freq];

    // Preload steering weights per lane
    float w_r[ANT_PER_LANE];
    float w_i[ANT_PER_LANE];
    float nw_i[ANT_PER_LANE];

    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const float3 pos = tracker_pos<N_ANT>(lane + a * 32U, spacing_m);
        const double delay_m = static_cast<double>(pos.x) * direction.x
                             + static_cast<double>(pos.y) * direction.y
                             + static_cast<double>(pos.z) * direction.z;
        const double phase = wave_number * delay_m;
        double s, c;
        sincos(phase, &s, &c);
        w_r[a] = static_cast<float>(c);
        w_i[a] = static_cast<float>(s);
        nw_i[a] = -w_i[a];
    }

    // Shared storage for M complex time samples across the warp
    __shared__ float smem_r[4][M];
    __shared__ float smem_i[4][M];

    // Beamform M time steps
    for (int m = 0; m < M; ++m) {
        const std::size_t t = t_start + m;
        float s_r = 0.0F;
        float s_i = 0.0F;

        if (t < n_time) {
            const std::uint8_t* packed_ptr = packed + (t * n_freq + freq) * N_ANT + lane;
            #pragma unroll
            for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
                const float2 p = unpack_int4_fast(packed_ptr[a * 32U]);
                s_r = fmaf(w_r[a], p.x, fmaf(nw_i[a], p.y, s_r));
                s_i = fmaf(w_r[a], p.y, fmaf(w_i[a], p.x, s_i));
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_r += __shfl_down_sync(full_mask, s_r, offset);
            s_i += __shfl_down_sync(full_mask, s_i, offset);
        }

        if (lane == 0) {
            const float w = window_lut[m];
            smem_r[warp_in_block][m] = s_r * w;
            smem_i[warp_in_block][m] = s_i * w;
        }
    }
    __syncwarp();

    // Warp-level DFT across M samples for output fine channel k = lane
    if (lane < M) {
        float xr = 0.0F;
        float xi = 0.0F;
        const float k_f = static_cast<float>(lane);
        constexpr float two_pi_over_M = static_cast<float>(-2.0 * PI / static_cast<double>(M));

        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float src_r = smem_r[warp_in_block][m];
            const float src_i = smem_i[warp_in_block][m];
            const float angle = two_pi_over_M * k_f * static_cast<float>(m);
            float s, c;
            __sincosf(angle, &s, &c);
            xr = fmaf(src_r, c, fmaf(-src_i, s, xr));
            xi = fmaf(src_r, s, fmaf(src_i, c, xi));
        }

        const float intensity = xr * xr + xi * xi;
        fine_intensity[tau * (n_freq * M) + freq * M + lane] = intensity;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// CPU Reference Implementations
// ---------------------------------------------------------------------------

Intensities cpu_upchannelize_voltage(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config) {

    UpchannelizerDimensions udims{dims.n_time, dims.n_freq, dims.n_beams, config.upchan_factor};
    Intensities fine(udims.fine_intensity_count());
    cpu_upchannelize_voltage_into(voltage, dims, config, fine);
    return fine;
}

void cpu_upchannelize_voltage_into(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config,
    Intensities& fine_intensity) {

    validate_dimensions(dims);
    const std::size_t M = config.upchan_factor;
    if (M == 0 || dims.n_time % M != 0) {
        throw std::invalid_argument("n_time must be divisible by upchan_factor M");
    }

    const auto window_lut = generate_window_table(M, config.window);
    const std::size_t n_fine_time = dims.n_time / M;
    const std::size_t fine_freq = dims.n_freq * M;

    #pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t tau = 0; tau < n_fine_time; ++tau) {
        for (std::size_t f = 0; f < dims.n_freq; ++f) {
            for (std::size_t b = 0; b < dims.n_beams; ++b) {
                // Windowed input block
                std::vector<ComplexFloat> y(M);
                for (std::size_t m = 0; m < M; ++m) {
                    const std::size_t t = tau * M + m;
                    const std::size_t v_idx = (t * dims.n_freq + f) * dims.n_beams + b;
                    const auto c = voltage[v_idx];
                    const float w = window_lut[m];
                    y[m] = {c.real * w, c.imag * w};
                }

                // M-point DFT
                for (std::size_t k = 0; k < M; ++k) {
                    float sum_r = 0.0F;
                    float sum_i = 0.0F;
                    const double angle_base = -2.0 * PI * static_cast<double>(k) / static_cast<double>(M);
                    for (std::size_t m = 0; m < M; ++m) {
                        const double angle = angle_base * static_cast<double>(m);
                        const float c = static_cast<float>(std::cos(angle));
                        const float s = static_cast<float>(std::sin(angle));
                        sum_r += y[m].real * c - y[m].imag * s;
                        sum_i += y[m].real * s + y[m].imag * c;
                    }
                    const std::size_t out_idx = (tau * fine_freq + f * M + k) * dims.n_beams + b;
                    fine_intensity[out_idx] = sum_r * sum_r + sum_i * sum_i;
                }
            }
        }
    }
}

Intensities cpu_tracker_upchannelize(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg) {

    validate_dimensions(dims);
    validate_tracker(dims, tracker);

    const std::size_t M = upchan_cfg.upchan_factor;
    if (M == 0 || dims.n_time % M != 0) {
        throw std::invalid_argument("n_time must be divisible by upchan_factor M");
    }

    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const std::size_t window_count = tracker_window_count(dims.n_time, tracker.integration_spectra);

    // 1. Coherent beam tracking -> Complex Voltage
    ComplexVoltage tracked_voltage(dims.n_time * dims.n_freq);

    for (std::size_t window = 0; window < window_count; ++window) {
        const Vec3 direction = tracker_window_direction(
            tracker.trajectory, window, tracker.integration_spectra);
        const auto weights = generate_weights(dims, positions, frequencies, {direction});

        const std::size_t first_time = window * tracker.integration_spectra;
        const std::size_t last_time = std::min(first_time + tracker.integration_spectra, dims.n_time);

        #pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t t = first_time; t < last_time; ++t) {
            for (std::size_t f = 0; f < dims.n_freq; ++f) {
                float sum_r = 0.0F;
                float sum_i = 0.0F;
                for (std::size_t a = 0; a < dims.n_ant; ++a) {
                    const auto s = unpack_complex_int4(
                        packed[voltage_index(t, f, a, dims)]);
                    const auto& w = weights[weight_index(0, f, a, dims)];
                    sum_r += w.real * static_cast<float>(s.real) - w.imag * static_cast<float>(s.imag);
                    sum_i += w.real * static_cast<float>(s.imag) + w.imag * static_cast<float>(s.real);
                }
                tracked_voltage[t * dims.n_freq + f] = {sum_r, sum_i};
            }
        }
    }

    // 2. Upchannelize
    Dimensions single_beam_dims{dims.n_time, dims.n_freq, dims.n_ant, 1};
    return cpu_upchannelize_voltage(tracked_voltage, single_beam_dims, upchan_cfg);
}

// ---------------------------------------------------------------------------
// CUDA Implementations
// ---------------------------------------------------------------------------

void cuda_upchannelize_voltage_device_resident(
    const ComplexFloat* d_voltage,
    float* d_fine_intensity,
    const Dimensions& dims,
    const UpchannelizerConfig& config,
    void* stream_ptr) {

    validate_dimensions(dims);
    const std::size_t M = config.upchan_factor;
    if (M != 32 && M != 16 && M != 8 && M != 4) {
        throw std::invalid_argument("CUDA Upchannelizer supports M in {4, 8, 16, 32}");
    }
    if (dims.n_time % M != 0) {
        throw std::invalid_argument("n_time must be divisible by M");
    }

    const std::size_t n_fine_time = dims.n_time / M;
    const std::size_t total_warps = n_fine_time * dims.n_freq * dims.n_beams;

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Window LUT on GPU
    const auto h_window = generate_window_table(M, config.window);
    float* d_window = nullptr;
    check_cuda(cudaMallocAsync(&d_window, M * sizeof(float), stream), "cudaMallocAsync window");
    check_cuda(cudaMemcpyAsync(d_window, h_window.data(), M * sizeof(float), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync window");

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    if (M == 32) {
        upchannelize_voltage_kernel<32><<<grid_dim, block_dim, 0, stream>>>(
            d_fine_intensity, d_voltage, d_window, dims.n_time, dims.n_freq, dims.n_beams, n_fine_time, total_warps);
    } else if (M == 16) {
        upchannelize_voltage_kernel<16><<<grid_dim, block_dim, 0, stream>>>(
            d_fine_intensity, d_voltage, d_window, dims.n_time, dims.n_freq, dims.n_beams, n_fine_time, total_warps);
    } else if (M == 8) {
        upchannelize_voltage_kernel<8><<<grid_dim, block_dim, 0, stream>>>(
            d_fine_intensity, d_voltage, d_window, dims.n_time, dims.n_freq, dims.n_beams, n_fine_time, total_warps);
    } else if (M == 4) {
        upchannelize_voltage_kernel<4><<<grid_dim, block_dim, 0, stream>>>(
            d_fine_intensity, d_voltage, d_window, dims.n_time, dims.n_freq, dims.n_beams, n_fine_time, total_warps);
    }
    check_cuda(cudaGetLastError(), "upchannelize_voltage_kernel launch");
    check_cuda(cudaFreeAsync(d_window, stream), "cudaFreeAsync window");
}

void cuda_tracker_upchannelize_device_resident(
    const std::uint8_t* d_packed,
    float* d_fine_intensity,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg,
    void* stream_ptr) {

    validate_dimensions(dims);
    validate_tracker(dims, tracker);
    const std::size_t M = upchan_cfg.upchan_factor;
    if (M != 32 && M != 16 && M != 8 && M != 4) {
        throw std::invalid_argument("CUDA Tracker Upchannelizer supports M in {4, 8, 16, 32}");
    }
    if (dims.n_time % M != 0) {
        throw std::invalid_argument("n_time must be divisible by M");
    }

    const std::size_t n_fine_time = dims.n_time / M;
    const std::size_t total_warps = n_fine_time * dims.n_freq;
    const std::size_t window_count = tracker_window_count(dims.n_time, tracker.integration_spectra);

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Precompute directions & wavenumbers on host and upload
    std::vector<float> h_window_directions(window_count * 3);
    for (std::size_t w = 0; w < window_count; ++w) {
        const Vec3 d = tracker_window_direction(tracker.trajectory, w, tracker.integration_spectra);
        h_window_directions[w * 3 + 0] = d[0];
        h_window_directions[w * 3 + 1] = d[1];
        h_window_directions[w * 3 + 2] = d[2];
    }

    const auto frequencies = channelized_frequencies(dims.n_freq);
    std::vector<double> h_wavenumbers(dims.n_freq);
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        h_wavenumbers[f] = two_pi * frequencies[f] / speed_of_light_m_per_s;
    }

    const auto h_window = generate_window_table(M, upchan_cfg.window);

    float* d_dirs = nullptr;
    double* d_waves = nullptr;
    float* d_win_lut = nullptr;

    check_cuda(cudaMallocAsync(&d_dirs, window_count * 3 * sizeof(float), stream), "cudaMallocAsync dirs");
    check_cuda(cudaMallocAsync(&d_waves, dims.n_freq * sizeof(double), stream), "cudaMallocAsync waves");
    check_cuda(cudaMallocAsync(&d_win_lut, M * sizeof(float), stream), "cudaMallocAsync win_lut");

    check_cuda(cudaMemcpyAsync(d_dirs, h_window_directions.data(), window_count * 3 * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync dirs");
    check_cuda(cudaMemcpyAsync(d_waves, h_wavenumbers.data(), dims.n_freq * sizeof(double), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync waves");
    check_cuda(cudaMemcpyAsync(d_win_lut, h_window.data(), M * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync win_lut");

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const float spacing_m = default_spacing_m;

    auto launch_ant = [&](auto n_ant_tag) {
        constexpr int N_ANT = decltype(n_ant_tag)::value;
        if (M == 32) {
            tracker_upchannelize_kernel<N_ANT, 32><<<grid_dim, block_dim, 0, stream>>>(
                d_fine_intensity, d_dirs, d_waves, d_win_lut, d_packed,
                dims.n_time, dims.n_freq, tracker.integration_spectra, n_fine_time, spacing_m, total_warps);
        } else if (M == 16) {
            tracker_upchannelize_kernel<N_ANT, 16><<<grid_dim, block_dim, 0, stream>>>(
                d_fine_intensity, d_dirs, d_waves, d_win_lut, d_packed,
                dims.n_time, dims.n_freq, tracker.integration_spectra, n_fine_time, spacing_m, total_warps);
        } else if (M == 8) {
            tracker_upchannelize_kernel<N_ANT, 8><<<grid_dim, block_dim, 0, stream>>>(
                d_fine_intensity, d_dirs, d_waves, d_win_lut, d_packed,
                dims.n_time, dims.n_freq, tracker.integration_spectra, n_fine_time, spacing_m, total_warps);
        } else if (M == 4) {
            tracker_upchannelize_kernel<N_ANT, 4><<<grid_dim, block_dim, 0, stream>>>(
                d_fine_intensity, d_dirs, d_waves, d_win_lut, d_packed,
                dims.n_time, dims.n_freq, tracker.integration_spectra, n_fine_time, spacing_m, total_warps);
        }
    };

    if (dims.n_ant == 32) {
        launch_ant(std::integral_constant<int, 32>{});
    } else if (dims.n_ant == 64) {
        launch_ant(std::integral_constant<int, 64>{});
    } else if (dims.n_ant == 128) {
        launch_ant(std::integral_constant<int, 128>{});
    } else if (dims.n_ant == 256) {
        launch_ant(std::integral_constant<int, 256>{});
    } else {
        throw std::invalid_argument("CUDA Tracker Upchannelizer supports n_ant = 32, 64, 128, 256");
    }

    check_cuda(cudaGetLastError(), "tracker_upchannelize_kernel launch");
    check_cuda(cudaFreeAsync(d_dirs, stream), "cudaFreeAsync dirs");
    check_cuda(cudaFreeAsync(d_waves, stream), "cudaFreeAsync waves");
    check_cuda(cudaFreeAsync(d_win_lut, stream), "cudaFreeAsync win_lut");
}

Intensities cuda_upchannelize_voltage(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config) {

    UpchannelizerDimensions udims{dims.n_time, dims.n_freq, dims.n_beams, config.upchan_factor};
    Intensities fine(udims.fine_intensity_count());
    cuda_upchannelize_voltage_into(voltage, dims, config, fine);
    return fine;
}

void cuda_upchannelize_voltage_into(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config,
    Intensities& fine_intensity) {

    UpchannelizerDimensions udims{dims.n_time, dims.n_freq, dims.n_beams, config.upchan_factor};
    const std::size_t v_bytes = voltage.size() * sizeof(ComplexFloat);
    const std::size_t out_bytes = udims.fine_intensity_count() * sizeof(float);

    ComplexFloat* d_v = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_v, v_bytes), "cudaMalloc d_v");
    check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out");

    check_cuda(cudaMemcpy(d_v, voltage.data(), v_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D voltage");
    cuda_upchannelize_voltage_device_resident(d_v, d_out, dims, config, nullptr);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize upchan");
    check_cuda(cudaMemcpy(fine_intensity.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H fine_intensity");

    cudaFree(d_v);
    cudaFree(d_out);
}

Intensities cuda_tracker_upchannelize(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg) {

    UpchannelizerDimensions udims{dims.n_time, dims.n_freq, 1, upchan_cfg.upchan_factor};
    Intensities fine(udims.fine_intensity_count());
    cuda_tracker_upchannelize_into(packed, dims, tracker, upchan_cfg, fine);
    return fine;
}

void cuda_tracker_upchannelize_into(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg,
    Intensities& fine_intensity) {

    UpchannelizerDimensions udims{dims.n_time, dims.n_freq, 1, upchan_cfg.upchan_factor};
    const std::size_t p_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t out_bytes = udims.fine_intensity_count() * sizeof(float);

    std::uint8_t* d_packed = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_packed, p_bytes), "cudaMalloc d_packed");
    check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc d_out");

    check_cuda(cudaMemcpy(d_packed, packed.data(), p_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D packed");
    cuda_tracker_upchannelize_device_resident(d_packed, d_out, dims, tracker, upchan_cfg, nullptr);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize tracker upchan");
    check_cuda(cudaMemcpy(fine_intensity.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H fine_intensity");

    cudaFree(d_packed);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// CudaUpchannelizerWorkspace
// ---------------------------------------------------------------------------
struct CudaUpchannelizerWorkspace::Impl {
    Impl(const Dimensions& cap, const UpchannelizerConfig& cfg)
        : dims(cap), config(cfg),
          udims{dims.n_time, dims.n_freq, dims.n_beams, config.upchan_factor} {

        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        voltage_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(ComplexFloat);
        packed_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
        fine_bytes = udims.fine_intensity_count() * sizeof(float);

        check_cuda(cudaMalloc(&d_voltage, voltage_bytes), "cudaMalloc d_voltage");
        check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc d_packed");
        check_cuda(cudaMalloc(&d_fine, fine_bytes), "cudaMalloc d_fine");
    }

    ~Impl() {
        if (d_voltage) cudaFree(d_voltage);
        if (d_packed) cudaFree(d_packed);
        if (d_fine) cudaFree(d_fine);
        if (stream) cudaStreamDestroy(stream);
    }

    Dimensions dims;
    UpchannelizerConfig config;
    UpchannelizerDimensions udims;
    cudaStream_t stream = nullptr;
    ComplexFloat* d_voltage = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_fine = nullptr;
    std::size_t voltage_bytes = 0;
    std::size_t packed_bytes = 0;
    std::size_t fine_bytes = 0;
    double measured_setup_ms = 0.0;
};

CudaUpchannelizerWorkspace::CudaUpchannelizerWorkspace(
    const Dimensions& capacity, const UpchannelizerConfig& config) {
    const auto t0 = Clock::now();
    impl_ = std::make_unique<Impl>(capacity, config);
    const auto t1 = Clock::now();
    impl_->measured_setup_ms = elapsed_ms(t0, t1);
}

CudaUpchannelizerWorkspace::~CudaUpchannelizerWorkspace() = default;

double CudaUpchannelizerWorkspace::setup_ms() const {
    return impl_->measured_setup_ms;
}

const UpchannelizerConfig& CudaUpchannelizerWorkspace::config() const {
    return impl_->config;
}

const UpchannelizerDimensions& CudaUpchannelizerWorkspace::dimensions() const {
    return impl_->udims;
}

void CudaUpchannelizerWorkspace::process(const ComplexVoltage& voltage, Intensities& fine_intensity) {
    check_cuda(cudaMemcpyAsync(impl_->d_voltage, voltage.data(), impl_->voltage_bytes,
                               cudaMemcpyHostToDevice, impl_->stream), "H2D voltage");
    cuda_upchannelize_voltage_device_resident(impl_->d_voltage, impl_->d_fine, impl_->dims, impl_->config, impl_->stream);
    check_cuda(cudaMemcpyAsync(fine_intensity.data(), impl_->d_fine, impl_->fine_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream), "D2H fine");
    check_cuda(cudaStreamSynchronize(impl_->stream), "sync");
}

void CudaUpchannelizerWorkspace::process_tracker(
    const PackedVoltage& packed, const TrackerConfig& tracker, Intensities& fine_intensity) {
    check_cuda(cudaMemcpyAsync(impl_->d_packed, packed.data(), impl_->packed_bytes,
                               cudaMemcpyHostToDevice, impl_->stream), "H2D packed");
    cuda_tracker_upchannelize_device_resident(impl_->d_packed, impl_->d_fine, impl_->dims, tracker, impl_->config, impl_->stream);
    check_cuda(cudaMemcpyAsync(fine_intensity.data(), impl_->d_fine, impl_->fine_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream), "D2H fine");
    check_cuda(cudaStreamSynchronize(impl_->stream), "sync");
}

ComplexFloat* CudaUpchannelizerWorkspace::device_voltage() {
    return impl_->d_voltage;
}

std::uint8_t* CudaUpchannelizerWorkspace::device_packed() {
    return impl_->d_packed;
}

float* CudaUpchannelizerWorkspace::device_fine_intensity() {
    return impl_->d_fine;
}

void* CudaUpchannelizerWorkspace::device_stream() {
    return static_cast<void*>(impl_->stream);
}

} // namespace beamformer
