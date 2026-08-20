// src/cuda_beam_tracker_v5.cu
//
// CUDA Beam Tracker V5 — Unified Single-Engine Warp-Reduction Architecture.
//
// Target architectures: NVIDIA Blackwell, Hopper, Ada Lovelace, Ampere, Turing,
// Volta, and Pascal (SM_60+).
//
// Key Architectural Highlights:
// 1. Unified Warp-Reduction Strategy:
//    Eliminates block-level shared memory barriers and inter-warp synchronization.
//    Every antenna count (32, 64, 128, 256) is partitioned across the 32 warp lanes
//    (ANT_PER_LANE = N_ANT / 32) in registers, accumulating across unrolled time
//    steps and reducing in a single 5-step __shfl_down_sync tree.
// 2. sincos() fused transcendental evaluation.
// 3. Pre-negated imaginary steering weights (nw_i) in registers.
// 4. Pointer-based stride increments eliminating 64-bit index arithmetic in loops.
// 5. Single dispatcher and streamlined single-engine architecture.

#include "beamformer/cuda_beam_tracker_v5.hpp"

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/physics.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

// ---------------------------------------------------------------------------
// Pinned Host Memory Helpers
// ---------------------------------------------------------------------------

void PinnedDeleterV5::operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

PinnedVectorV5<std::uint8_t> allocate_pinned_voltage_v5(const Dimensions& dims) {
    const std::size_t bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for packed voltage: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV5<std::uint8_t>(static_cast<std::uint8_t*>(ptr));
}

PinnedVectorV5<float> allocate_pinned_intensities_v5(const Dimensions& dims) {
    const std::size_t bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for intensities: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV5<float>(static_cast<float*>(ptr));
}

namespace {

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

// ---------------------------------------------------------------------------
// Fast Device Decoding & Geometry Primitives
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
__device__ __forceinline__ float3 tracker_position_v5(const unsigned int element,
                                                      const float spacing_m) {
    if constexpr (N_ANT == 32 || N_ANT == 64) {
        const unsigned int col = element & 7U;
        const unsigned int row = element >> 3U;
        return make_float3(static_cast<float>(col) * spacing_m,
                           static_cast<float>(row) * spacing_m,
                           0.0F);
    } else { // 128 or 256 antennas (16 columns)
        const unsigned int col = element & 15U;
        const unsigned int row = element >> 4U;
        return make_float3(static_cast<float>(col) * spacing_m,
                           static_cast<float>(row) * spacing_m,
                           0.0F);
    }
}

__device__ __forceinline__ void tracker_weight_v5(const float3 position,
                                                  const float3 direction,
                                                  const double wave_number,
                                                  float* weight_real,
                                                  float* weight_imag) {
    const double delay_m = static_cast<double>(position.x) * direction.x
                         + static_cast<double>(position.y) * direction.y
                         + static_cast<double>(position.z) * direction.z;
    const double phase = wave_number * delay_m;
    double s, c;
    sincos(phase, &s, &c);
    *weight_real = static_cast<float>(c);
    *weight_imag = static_cast<float>(s);
}

// ---------------------------------------------------------------------------
// Unified V5 Tracker Kernel Template
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
tracker_v5_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t n_beams,
    const std::size_t integration_spectra,
    const std::size_t time_chunk_size,
    const std::size_t chunks_per_window,
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
    const std::size_t chunk_global = warp_id / n_freq;
    const std::size_t chunk_in_win = chunk_global % chunks_per_window;
    const std::size_t window = chunk_global / chunks_per_window;

    const float3 direction = make_float3(window_directions[window * 3 + 0],
                                         window_directions[window * 3 + 1],
                                         window_directions[window * 3 + 2]);
    const double wave_number = wavenumbers[freq];

    // Precompute steering weights and pre-negated imaginary components
    float w_r[ANT_PER_LANE];
    float w_i[ANT_PER_LANE];
    float nw_i[ANT_PER_LANE];

    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const float3 pos = tracker_position_v5<N_ANT>(lane + a * 32U, spacing_m);
        tracker_weight_v5(pos, direction, wave_number, &w_r[a], &w_i[a]);
        nw_i[a] = -w_i[a];
    }

    const std::size_t t_win_start = window * integration_spectra;
    const std::size_t t_win_end = (t_win_start + integration_spectra < n_time)
                                     ? (t_win_start + integration_spectra)
                                     : n_time;

    const std::size_t t_chunk_start = t_win_start + chunk_in_win * time_chunk_size;
    if (t_chunk_start >= t_win_end) {
        return;
    }
    const std::size_t t_chunk_end = (t_chunk_start + time_chunk_size < t_win_end)
                                       ? (t_chunk_start + time_chunk_size)
                                       : t_win_end;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::size_t intensity_stride = n_freq * n_beams;

    const std::uint8_t* packed_ptr = packed + (t_chunk_start * n_freq + freq) * N_ANT + lane;
    float* intensity_ptr = intensity + (t_chunk_start * n_freq + freq) * n_beams;

    std::size_t t = t_chunk_start;

    for (; t + (TIME_UNROLL - 1) < t_chunk_end; t += TIME_UNROLL) {
        float s_r[TIME_UNROLL] = {0.0F};
        float s_i[TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float wra = w_r[a];
            const float wia = w_i[a];
            const float nwi = nw_i[a];
            const unsigned int a_offset = a * 32U;

            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                const float2 p = unpack_int4_fast(packed_ptr[k * t_stride + a_offset]);
                s_r[k] = fmaf(wra, p.x, fmaf(nwi, p.y, s_r[k]));
                s_i[k] = fmaf(wra, p.y, fmaf(wia, p.x, s_i[k]));
            }
        }

        // Intra-warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                s_r[k] += __shfl_down_sync(full_mask, s_r[k], offset);
                s_i[k] += __shfl_down_sync(full_mask, s_i[k], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                intensity_ptr[k * intensity_stride] = s_r[k] * s_r[k] + s_i[k] * s_i[k];
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
        intensity_ptr += TIME_UNROLL * intensity_stride;
    }

    // Remainder loop
    for (; t < t_chunk_end; ++t) {
        float s_r = 0.0F;
        float s_i = 0.0F;

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_fast(packed_ptr[a * 32U]);
            s_r = fmaf(w_r[a], p.x, fmaf(nw_i[a], p.y, s_r));
            s_i = fmaf(w_r[a], p.y, fmaf( w_i[a], p.x, s_i));
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_r += __shfl_down_sync(full_mask, s_r, offset);
            s_i += __shfl_down_sync(full_mask, s_i, offset);
        }

        if (lane == 0) {
            *intensity_ptr = s_r * s_r + s_i * s_i;
        }

        packed_ptr += t_stride;
        intensity_ptr += intensity_stride;
    }
}

// ---------------------------------------------------------------------------
// Host Pre-Scan & Geometry Validation
// ---------------------------------------------------------------------------
void host_pre_scan_v5(const Dimensions& dims, const TrackerConfig& tracker,
                      std::vector<float>& window_directions_flat,
                      std::vector<double>& wavenumbers) {
    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument("tracker requires exactly n_beams == 1");
    }
    if (tracker.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
    }
    if (dims.n_ant != 32 && dims.n_ant != 64 && dims.n_ant != 128 && dims.n_ant != 256) {
        throw std::invalid_argument("tracker supports n_ant == 32, 64, 128, or 256");
    }

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

// ---------------------------------------------------------------------------
// Unified Kernel Launch Dispatcher
// ---------------------------------------------------------------------------
void dispatch_v5_kernel(
    float* d_intensity,
    const float* d_window_directions,
    const double* d_wavenumbers,
    const std::uint8_t* d_packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const V5ExecutionConfig& config,
    cudaStream_t stream) {

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    const std::size_t chunk_size = (config.time_chunk_size > 0)
        ? std::min(config.time_chunk_size, tracker.integration_spectra)
        : tracker.integration_spectra;
    const std::size_t chunks_per_window =
        (tracker.integration_spectra + chunk_size - 1) / chunk_size;

    constexpr float spacing_m = default_spacing_m;

    const std::size_t total_warps = window_count * chunks_per_window * dims.n_freq;
    constexpr unsigned int warps_per_block = 4;
    const dim3 block_dim(32, warps_per_block);
    const unsigned int grid_blocks = static_cast<unsigned int>(
        (total_warps + warps_per_block - 1) / warps_per_block);

    auto launch_with_unroll = [&](auto n_ant_ic) {
        constexpr int N_A = decltype(n_ant_ic)::value;
        if (config.time_unroll >= 8) {
            tracker_v5_kernel<N_A, 8><<<grid_blocks, block_dim, 0, stream>>>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                tracker.integration_spectra, chunk_size, chunks_per_window,
                spacing_m, total_warps);
        } else if (config.time_unroll >= 4) {
            tracker_v5_kernel<N_A, 4><<<grid_blocks, block_dim, 0, stream>>>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                tracker.integration_spectra, chunk_size, chunks_per_window,
                spacing_m, total_warps);
        } else {
            tracker_v5_kernel<N_A, 2><<<grid_blocks, block_dim, 0, stream>>>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                tracker.integration_spectra, chunk_size, chunks_per_window,
                spacing_m, total_warps);
        }
    };

    if (dims.n_ant == 256) {
        launch_with_unroll(std::integral_constant<int, 256>{});
    } else if (dims.n_ant == 128) {
        launch_with_unroll(std::integral_constant<int, 128>{});
    } else if (dims.n_ant == 64) {
        launch_with_unroll(std::integral_constant<int, 64>{});
    } else if (dims.n_ant == 32) {
        launch_with_unroll(std::integral_constant<int, 32>{});
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Functional Interface Implementations
// ---------------------------------------------------------------------------

void cuda_beam_tracker_v5_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const V5ExecutionConfig& config,
    void* stream) {

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan_v5(dims, tracker, window_directions_flat, wavenumbers);

    const cudaStream_t cuda_stream = (stream != nullptr)
        ? static_cast<cudaStream_t>(stream)
        : nullptr;

    const std::size_t dir_bytes = window_directions_flat.size() * sizeof(float);
    const std::size_t wave_bytes = wavenumbers.size() * sizeof(double);

    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&d_window_directions), dir_bytes, cuda_stream),
               "cudaMallocAsync window_directions");
    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&d_wavenumbers), wave_bytes, cuda_stream),
               "cudaMallocAsync wavenumbers");

    check_cuda(cudaMemcpyAsync(d_window_directions, window_directions_flat.data(),
                               dir_bytes, cudaMemcpyHostToDevice, cuda_stream),
               "cudaMemcpyAsync window_directions");
    check_cuda(cudaMemcpyAsync(d_wavenumbers, wavenumbers.data(),
                               wave_bytes, cudaMemcpyHostToDevice, cuda_stream),
               "cudaMemcpyAsync wavenumbers");

    dispatch_v5_kernel(d_intensity, d_window_directions, d_wavenumbers,
                       d_packed, dims, tracker, config, cuda_stream);

    check_cuda(cudaFreeAsync(d_window_directions, cuda_stream), "cudaFreeAsync window_directions");
    check_cuda(cudaFreeAsync(d_wavenumbers, cuda_stream), "cudaFreeAsync wavenumbers");
}

void cuda_beam_tracker_v5_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const V5ExecutionConfig& config) {

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;
    host_pre_scan_v5(dims, tracker, window_directions_flat, wavenumbers);

    const std::size_t expected_voltage = voltage_sample_count(dims);
    if (packed.size() != expected_voltage) {
        throw std::invalid_argument("voltage buffer size does not match dimensions");
    }
    const std::size_t expected_output = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() != expected_output) {
        intensity.resize(expected_output);
    }

    const std::size_t voltage_bytes = expected_voltage * sizeof(std::uint8_t);
    const std::size_t intensity_bytes = expected_output * sizeof(float);
    const std::size_t dir_bytes = window_directions_flat.size() * sizeof(float);
    const std::size_t wave_bytes = wavenumbers.size() * sizeof(double);

    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_packed), voltage_bytes), "cudaMalloc d_packed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_intensity), intensity_bytes), "cudaMalloc d_intensity");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_window_directions), dir_bytes), "cudaMalloc d_window_directions");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_wavenumbers), wave_bytes), "cudaMalloc d_wavenumbers");

    check_cuda(cudaMemcpyAsync(d_packed, packed.data(), voltage_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D d_packed");
    check_cuda(cudaMemcpyAsync(d_window_directions, window_directions_flat.data(), dir_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D d_dir");
    check_cuda(cudaMemcpyAsync(d_wavenumbers, wavenumbers.data(), wave_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D d_wave");

    dispatch_v5_kernel(d_intensity, d_window_directions, d_wavenumbers,
                       d_packed, dims, tracker, config, stream);

    check_cuda(cudaMemcpyAsync(intensity.data(), d_intensity, intensity_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H d_intensity");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    cudaFree(d_packed);
    cudaFree(d_intensity);
    cudaFree(d_window_directions);
    cudaFree(d_wavenumbers);
    cudaStreamDestroy(stream);
}

Intensities cuda_beam_tracker_v5(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const V5ExecutionConfig& config) {
    Intensities result(dims.n_time * dims.n_freq * dims.n_beams);
    cuda_beam_tracker_v5_into(packed, dims, tracker, result, config);
    return result;
}

// ---------------------------------------------------------------------------
// BatchedTrackerStreamV5 Implementation
// ---------------------------------------------------------------------------
struct BatchedTrackerStreamV5::Impl {
    Dimensions win_dims;
    Dimensions batch_dims;
    TrackerConfig tracker_cfg;
    std::size_t batch_size;
    V5ExecutionConfig config;

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float last_time_ms = 0.0F;

    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    std::vector<float> window_directions_flat;
    std::vector<double> wavenumbers;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;

    Impl(const Dimensions& single_win_dims,
         const TrackerConfig& tracker,
         std::size_t b_size,
         const V5ExecutionConfig& cfg)
        : win_dims(single_win_dims), tracker_cfg(tracker),
          batch_size(b_size), config(cfg) {

        batch_dims = win_dims;
        batch_dims.n_time = win_dims.n_time * batch_size;

        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");

        const std::size_t v_bytes = voltage_sample_count(batch_dims) * sizeof(std::uint8_t);
        const std::size_t out_bytes = batch_dims.n_time * batch_dims.n_freq * batch_dims.n_beams * sizeof(float);

        check_cuda(cudaMalloc(&d_packed, v_bytes), "cudaMalloc d_packed");
        check_cuda(cudaMalloc(&d_intensity, out_bytes), "cudaMalloc d_intensity");

        host_pre_scan_v5(batch_dims, tracker_cfg, window_directions_flat, wavenumbers);

        const std::size_t dir_bytes = window_directions_flat.size() * sizeof(float);
        const std::size_t wave_bytes = wavenumbers.size() * sizeof(double);

        check_cuda(cudaMalloc(&d_window_directions, dir_bytes), "cudaMalloc d_dir");
        check_cuda(cudaMalloc(&d_wavenumbers, wave_bytes), "cudaMalloc d_wave");

        check_cuda(cudaMemcpyAsync(d_window_directions, window_directions_flat.data(),
                                   dir_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync dir");
        check_cuda(cudaMemcpyAsync(d_wavenumbers, wavenumbers.data(),
                                   wave_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync wave");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize init");
    }

    ~Impl() {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph) {
            cudaGraphDestroy(graph);
        }
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
    }

    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed,
                       float* host_intensity) {
        const std::size_t v_bytes = voltage_sample_count(batch_dims) * sizeof(std::uint8_t);
        const std::size_t out_bytes = batch_dims.n_time * batch_dims.n_freq * batch_dims.n_beams * sizeof(float);

        check_cuda(cudaMemcpyAsync(d_packed, host_packed, v_bytes,
                                   cudaMemcpyHostToDevice, stream), "H2D d_packed");

        process_batch_kernel_only(first_window_index);

        check_cuda(cudaMemcpyAsync(host_intensity, d_intensity, out_bytes,
                                   cudaMemcpyDeviceToHost, stream), "D2H d_intensity");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    void process_batch_kernel_only(std::size_t first_window_index) {
        (void)first_window_index;

        if (config.enable_cuda_graph) {
            if (!graph_captured) {
                check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "BeginCapture");
                dispatch_v5_kernel(d_intensity, d_window_directions, d_wavenumbers,
                                   d_packed, batch_dims, tracker_cfg, config, stream);
                check_cuda(cudaStreamEndCapture(stream, &graph), "EndCapture");
                check_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "GraphInstantiate");
                graph_captured = true;
            }

            check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord start");
            check_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
            check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize");
            check_cuda(cudaEventElapsedTime(&last_time_ms, start_event, stop_event), "cudaEventElapsedTime");
        } else {
            check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord start");
            dispatch_v5_kernel(d_intensity, d_window_directions, d_wavenumbers,
                               d_packed, batch_dims, tracker_cfg, config, stream);
            check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize");
            check_cuda(cudaEventElapsedTime(&last_time_ms, start_event, stop_event), "cudaEventElapsedTime");
        }
    }
};

BatchedTrackerStreamV5::BatchedTrackerStreamV5(const Dimensions& single_window_dims,
                                               const TrackerConfig& tracker,
                                               std::size_t batch_size,
                                               const V5ExecutionConfig& config)
    : impl_(std::make_unique<Impl>(single_window_dims, tracker, batch_size, config)) {}

BatchedTrackerStreamV5::~BatchedTrackerStreamV5() = default;

void BatchedTrackerStreamV5::process_batch(std::size_t first_window_index,
                                           const std::uint8_t* host_packed,
                                           float* host_intensity) {
    impl_->process_batch(first_window_index, host_packed, host_intensity);
}

void BatchedTrackerStreamV5::process_batch_kernel_only(std::size_t first_window_index) {
    impl_->process_batch_kernel_only(first_window_index);
}

float BatchedTrackerStreamV5::last_kernel_time_ms() const {
    return impl_->last_time_ms;
}

std::uint8_t* BatchedTrackerStreamV5::device_packed_buffer() {
    return impl_->d_packed;
}

float* BatchedTrackerStreamV5::device_intensity_buffer() {
    return impl_->d_intensity;
}

void* BatchedTrackerStreamV5::device_stream() {
    return impl_->stream;
}

std::size_t BatchedTrackerStreamV5::batch_size() const {
    return impl_->batch_size;
}

std::size_t BatchedTrackerStreamV5::window_bytes() const {
    return voltage_sample_count(impl_->win_dims) * sizeof(std::uint8_t);
}

std::size_t BatchedTrackerStreamV5::batch_voltage_bytes() const {
    return voltage_sample_count(impl_->batch_dims) * sizeof(std::uint8_t);
}

std::size_t BatchedTrackerStreamV5::batch_output_floats() const {
    return impl_->batch_dims.n_time * impl_->batch_dims.n_freq * impl_->batch_dims.n_beams;
}

} // namespace beamformer
