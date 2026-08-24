// src/cuda_beam_tracker_v5.cu
//
// CUDA Beam Tracker V5 — Unified Single-Engine Warp-Reduction Architecture with Dynamic Multi-Beam Support.
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
// 5. Dynamic multi-beam active counter & instantaneous 0 ms bypass when idle.

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
#include <mutex>

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
// Unified V5 Multi-Beam Tracker Kernel Template
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
tracker_v5_multibeam_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t integration_spectra,
    const std::size_t time_chunk_size,
    const std::size_t chunks_per_window,
    const float spacing_m,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride,
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

    const std::size_t beam_idx = warp_id % num_active_beams;
    const std::size_t rest = warp_id / num_active_beams;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk_global = rest / n_freq;
    const std::size_t chunk_in_win = chunk_global % chunks_per_window;
    const std::size_t window = chunk_global / chunks_per_window;

    const std::size_t dir_idx = (window * num_active_beams + beam_idx) * 3;
    const float3 direction = make_float3(window_directions[dir_idx + 0],
                                         window_directions[dir_idx + 1],
                                         window_directions[dir_idx + 2]);
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
    const std::size_t intensity_stride = n_freq * max_beams_stride;

    const std::uint8_t* packed_ptr = packed + (t_chunk_start * n_freq + freq) * N_ANT + lane;
    float* intensity_ptr = intensity + (t_chunk_start * n_freq + freq) * max_beams_stride + beam_idx;

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
            s_i = fmaf(w_r[a], p.y, fmaf(w_i[a], p.x, s_i));
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
// Unified Multi-Beam Dispatcher
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
void dispatch_kernel_multibeam(
    float* d_intensity,
    const float* d_window_directions,
    const double* d_wavenumbers,
    const std::uint8_t* d_packed,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t integration_spectra,
    std::size_t time_chunk_size,
    std::size_t chunks_per_window,
    float spacing_m,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t total_warps,
    cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    tracker_v5_multibeam_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_intensity,
        d_window_directions,
        d_wavenumbers,
        d_packed,
        n_time,
        n_freq,
        integration_spectra,
        time_chunk_size,
        chunks_per_window,
        spacing_m,
        num_active_beams,
        max_beams_stride,
        total_warps);
}

} // namespace

// ---------------------------------------------------------------------------
// Multi-Beam Device-Resident Implementation
// ---------------------------------------------------------------------------
void cuda_beam_tracker_v5_multibeam_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, std::size_t max_beams_allocated,
    const MultiTrackerConfig& multi_tracker,
    const V5ExecutionConfig& config,
    void* stream_ptr) {

    if (multi_tracker.num_active_beams == 0) {
        return; // Instantaneous bypass
    }

    validate_dimensions(dims);
    const std::size_t num_active = std::min(multi_tracker.num_active_beams, MAX_TRACKER_BEAMS);
    const std::size_t max_beams_stride = std::max(max_beams_allocated, num_active);

    const std::size_t window_count =
        tracker_window_count(dims.n_time, multi_tracker.integration_spectra);
    const std::size_t chunk_size = (config.time_chunk_size > 0)
        ? std::min(config.time_chunk_size, multi_tracker.integration_spectra)
        : multi_tracker.integration_spectra;
    const std::size_t chunks_per_window =
        (multi_tracker.integration_spectra + chunk_size - 1) / chunk_size;

    std::vector<float> h_window_directions(window_count * num_active * 3);
    for (std::size_t w = 0; w < window_count; ++w) {
        for (std::size_t b = 0; b < num_active; ++b) {
            const Vec3 direction = tracker_window_direction(
                multi_tracker.trajectories[b], w, multi_tracker.integration_spectra);
            const std::size_t base = (w * num_active + b) * 3;
            h_window_directions[base + 0] = direction[0];
            h_window_directions[base + 1] = direction[1];
            h_window_directions[base + 2] = direction[2];
        }
    }

    const auto frequencies = channelized_frequencies(dims.n_freq);
    std::vector<double> h_wavenumbers(dims.n_freq);
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        h_wavenumbers[f] =
            two_pi * static_cast<double>(frequencies[f]) / speed_of_light_m_per_s;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    check_cuda(cudaMallocAsync(&d_window_directions, h_window_directions.size() * sizeof(float), stream),
               "cudaMallocAsync d_window_directions");
    check_cuda(cudaMallocAsync(&d_wavenumbers, h_wavenumbers.size() * sizeof(double), stream),
               "cudaMallocAsync d_wavenumbers");

    check_cuda(cudaMemcpyAsync(d_window_directions, h_window_directions.data(),
                               h_window_directions.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync d_window_directions");
    check_cuda(cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(),
                               h_wavenumbers.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync d_wavenumbers");

    const std::size_t total_warps = window_count * chunks_per_window * dims.n_freq * num_active;
    constexpr float spacing_m = default_spacing_m;

    auto dispatch_ant = [&](auto n_ant_ic) {
        constexpr int N_A = decltype(n_ant_ic)::value;
        if (config.time_unroll >= 8) {
            dispatch_kernel_multibeam<N_A, 8>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, multi_tracker.integration_spectra, chunk_size,
                chunks_per_window, spacing_m, num_active, max_beams_stride, total_warps, stream);
        } else if (config.time_unroll >= 4) {
            dispatch_kernel_multibeam<N_A, 4>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, multi_tracker.integration_spectra, chunk_size,
                chunks_per_window, spacing_m, num_active, max_beams_stride, total_warps, stream);
        } else {
            dispatch_kernel_multibeam<N_A, 2>(
                d_intensity, d_window_directions, d_wavenumbers, d_packed,
                dims.n_time, dims.n_freq, multi_tracker.integration_spectra, chunk_size,
                chunks_per_window, spacing_m, num_active, max_beams_stride, total_warps, stream);
        }
    };

    switch (dims.n_ant) {
        case 32:  dispatch_ant(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_ant(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_ant(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_ant(std::integral_constant<int, 256>{}); break;
        default:
            cudaFreeAsync(d_window_directions, stream);
            cudaFreeAsync(d_wavenumbers, stream);
            throw std::invalid_argument("Unsupported n_ant: must be 32, 64, 128, or 256");
    }

    check_cuda(cudaFreeAsync(d_window_directions, stream), "cudaFreeAsync d_window_directions");
    check_cuda(cudaFreeAsync(d_wavenumbers, stream), "cudaFreeAsync d_wavenumbers");
}

// ---------------------------------------------------------------------------
// Single-Beam Backward-Compatible Functional API
// ---------------------------------------------------------------------------
void cuda_beam_tracker_v5_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const V5ExecutionConfig& config,
    void* stream) {

    MultiTrackerConfig multi_cfg;
    multi_cfg.num_active_beams = 1;
    multi_cfg.trajectories[0] = tracker.trajectory;
    multi_cfg.integration_spectra = tracker.integration_spectra;

    cuda_beam_tracker_v5_multibeam_device_resident(
        d_packed, d_intensity, dims, 1, multi_cfg, config, stream);
}

void cuda_beam_tracker_v5_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const V5ExecutionConfig& config) {

    const std::size_t packed_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t intensity_bytes = dims.n_time * dims.n_freq * sizeof(float);

    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc d_packed");
    check_cuda(cudaMalloc(&d_intensity, intensity_bytes), "cudaMalloc d_intensity");

    check_cuda(cudaMemcpy(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    cuda_beam_tracker_v5_device_resident(d_packed, d_intensity, dims, tracker, config, nullptr);

    check_cuda(cudaMemcpy(intensity.data(), d_intensity, intensity_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    cudaFree(d_packed);
    cudaFree(d_intensity);
}

Intensities cuda_beam_tracker_v5(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const V5ExecutionConfig& config) {
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    cuda_beam_tracker_v5_into(packed, dims, tracker, intensity, config);
    return intensity;
}

// ---------------------------------------------------------------------------
// Single-Beam Batched Tracker Stream V5
// ---------------------------------------------------------------------------
struct BatchedTrackerStreamV5::Impl {
    Dimensions single_window_dims;
    Dimensions batch_dims;
    TrackerConfig tracker;
    V5ExecutionConfig config;
    std::size_t batch_size;

    cudaStream_t stream = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float last_time_ms = 0.0F;

    std::size_t chunk_size = 0;
    std::size_t chunks_per_window = 0;
    std::size_t total_warps = 0;
    std::vector<float> h_window_directions;
    std::vector<double> h_wavenumbers;

    Impl(const Dimensions& dims, const TrackerConfig& trk, std::size_t b_size, const V5ExecutionConfig& cfg)
        : single_window_dims(dims), tracker(trk), config(cfg), batch_size(b_size) {

        batch_dims = single_window_dims;
        batch_dims.n_time = tracker.integration_spectra * batch_size;

        chunk_size = (config.time_chunk_size > 0)
            ? std::min(config.time_chunk_size, tracker.integration_spectra)
            : tracker.integration_spectra;
        chunks_per_window = (tracker.integration_spectra + chunk_size - 1) / chunk_size;
        total_warps = batch_size * chunks_per_window * batch_dims.n_freq;

        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");

        check_cuda(cudaMalloc(&d_packed, voltage_sample_count(batch_dims) * sizeof(std::uint8_t)), "cudaMalloc");
        check_cuda(cudaMalloc(&d_intensity, batch_dims.n_time * batch_dims.n_freq * sizeof(float)), "cudaMalloc");

        h_window_directions.resize(batch_size * 3);
        const auto freqs = channelized_frequencies(batch_dims.n_freq);
        h_wavenumbers.resize(batch_dims.n_freq);
        for (std::size_t f = 0; f < batch_dims.n_freq; ++f) {
            h_wavenumbers[f] = two_pi * static_cast<double>(freqs[f]) / speed_of_light_m_per_s;
        }

        check_cuda(cudaMalloc(&d_window_directions, h_window_directions.size() * sizeof(float)), "cudaMalloc");
        check_cuda(cudaMalloc(&d_wavenumbers, h_wavenumbers.size() * sizeof(double)), "cudaMalloc");

        check_cuda(cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(),
                                   h_wavenumbers.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    }

    ~Impl() {
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
    }
};

BatchedTrackerStreamV5::BatchedTrackerStreamV5(
    const Dimensions& dims, const TrackerConfig& trk, std::size_t b_size, const V5ExecutionConfig& cfg)
    : impl_(std::make_unique<Impl>(dims, trk, b_size, cfg)) {}

BatchedTrackerStreamV5::~BatchedTrackerStreamV5() = default;

void BatchedTrackerStreamV5::process_batch(
    std::size_t first_window_index, const std::uint8_t* host_p, float* host_i) {
    const std::size_t p_bytes = voltage_sample_count(impl_->batch_dims) * sizeof(std::uint8_t);
    const std::size_t i_bytes = impl_->batch_dims.n_time * impl_->batch_dims.n_freq * sizeof(float);

    check_cuda(cudaMemcpyAsync(impl_->d_packed, host_p, p_bytes, cudaMemcpyHostToDevice, impl_->stream), "H2D");
    process_batch_kernel_only(first_window_index);
    check_cuda(cudaMemcpyAsync(host_i, impl_->d_intensity, i_bytes, cudaMemcpyDeviceToHost, impl_->stream), "D2H");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
}

void BatchedTrackerStreamV5::process_batch_kernel_only(std::size_t first_window_index) {
    for (std::size_t w = 0; w < impl_->batch_size; ++w) {
        const Vec3 direction = tracker_window_direction(
            impl_->tracker.trajectory, first_window_index + w, impl_->tracker.integration_spectra);
        impl_->h_window_directions[w * 3 + 0] = direction[0];
        impl_->h_window_directions[w * 3 + 1] = direction[1];
        impl_->h_window_directions[w * 3 + 2] = direction[2];
    }

    check_cuda(cudaMemcpyAsync(impl_->d_window_directions, impl_->h_window_directions.data(),
                               impl_->h_window_directions.size() * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync");

    check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "cudaEventRecord");

    constexpr float spacing_m = default_spacing_m;
    switch (impl_->batch_dims.n_ant) {
        case 32:
            dispatch_kernel_multibeam<32, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, impl_->tracker.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, 1, 1, impl_->total_warps, impl_->stream);
            break;
        case 64:
            dispatch_kernel_multibeam<64, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, impl_->tracker.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, 1, 1, impl_->total_warps, impl_->stream);
            break;
        case 128:
            dispatch_kernel_multibeam<128, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, impl_->tracker.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, 1, 1, impl_->total_warps, impl_->stream);
            break;
        case 256:
            dispatch_kernel_multibeam<256, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, impl_->tracker.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, 1, 1, impl_->total_warps, impl_->stream);
            break;
    }

    check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(impl_->stop_event), "cudaEventSynchronize");
    check_cuda(cudaEventElapsedTime(&impl_->last_time_ms, impl_->start_event, impl_->stop_event), "cudaEventElapsedTime");
}

float BatchedTrackerStreamV5::last_kernel_time_ms() const { return impl_->last_time_ms; }
std::uint8_t* BatchedTrackerStreamV5::device_packed_buffer() { return impl_->d_packed; }
float* BatchedTrackerStreamV5::device_intensity_buffer() { return impl_->d_intensity; }
void* BatchedTrackerStreamV5::device_stream() { return impl_->stream; }
std::size_t BatchedTrackerStreamV5::batch_size() const { return impl_->batch_size; }
std::size_t BatchedTrackerStreamV5::window_bytes() const { return voltage_sample_count(impl_->single_window_dims) * sizeof(std::uint8_t); }
std::size_t BatchedTrackerStreamV5::batch_voltage_bytes() const { return voltage_sample_count(impl_->batch_dims) * sizeof(std::uint8_t); }
std::size_t BatchedTrackerStreamV5::batch_output_floats() const { return impl_->batch_dims.n_time * impl_->batch_dims.n_freq; }

// ---------------------------------------------------------------------------
// MultiBeamBatchedTrackerStreamV5 Implementation
// ---------------------------------------------------------------------------
struct MultiBeamBatchedTrackerStreamV5::Impl {
    Dimensions single_window_dims;
    Dimensions batch_dims;
    std::size_t max_beams;
    MultiTrackerConfig tracker_config;
    V5ExecutionConfig config;
    std::size_t batch_size;
    std::mutex config_mutex;

    cudaStream_t stream = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float last_time_ms = 0.0F;

    std::size_t chunk_size = 0;
    std::size_t chunks_per_window = 0;
    std::vector<float> h_window_directions;
    std::vector<double> h_wavenumbers;

    Impl(const Dimensions& dims, std::size_t max_b, const MultiTrackerConfig& trk, std::size_t b_size, const V5ExecutionConfig& cfg)
        : single_window_dims(dims), max_beams(max_b), tracker_config(trk), config(cfg), batch_size(b_size) {

        batch_dims = single_window_dims;
        batch_dims.n_time = tracker_config.integration_spectra * batch_size;

        chunk_size = (config.time_chunk_size > 0)
            ? std::min(config.time_chunk_size, tracker_config.integration_spectra)
            : tracker_config.integration_spectra;
        chunks_per_window = (tracker_config.integration_spectra + chunk_size - 1) / chunk_size;

        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");

        check_cuda(cudaMalloc(&d_packed, voltage_sample_count(batch_dims) * sizeof(std::uint8_t)), "cudaMalloc");
        check_cuda(cudaMalloc(&d_intensity, batch_dims.n_time * batch_dims.n_freq * max_beams * sizeof(float)), "cudaMalloc");

        h_window_directions.resize(batch_size * MAX_TRACKER_BEAMS * 3);
        const auto freqs = channelized_frequencies(batch_dims.n_freq);
        h_wavenumbers.resize(batch_dims.n_freq);
        for (std::size_t f = 0; f < batch_dims.n_freq; ++f) {
            h_wavenumbers[f] = two_pi * static_cast<double>(freqs[f]) / speed_of_light_m_per_s;
        }

        check_cuda(cudaMalloc(&d_window_directions, h_window_directions.size() * sizeof(float)), "cudaMalloc");
        check_cuda(cudaMalloc(&d_wavenumbers, h_wavenumbers.size() * sizeof(double)), "cudaMalloc");

        check_cuda(cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(),
                                   h_wavenumbers.size() * sizeof(double),
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    }

    ~Impl() {
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
    }
};

MultiBeamBatchedTrackerStreamV5::MultiBeamBatchedTrackerStreamV5(
    const Dimensions& dims, std::size_t max_b, const MultiTrackerConfig& trk, std::size_t b_size, const V5ExecutionConfig& cfg)
    : impl_(std::make_unique<Impl>(dims, max_b, trk, b_size, cfg)) {}

MultiBeamBatchedTrackerStreamV5::~MultiBeamBatchedTrackerStreamV5() = default;

void MultiBeamBatchedTrackerStreamV5::set_trajectory(std::size_t beam_id, const TrackerTrajectoryConfig& traj) {
    if (beam_id < MAX_TRACKER_BEAMS) {
        std::lock_guard<std::mutex> lock(impl_->config_mutex);
        impl_->tracker_config.trajectories[beam_id] = traj;
    }
}

void MultiBeamBatchedTrackerStreamV5::set_num_active_beams(std::size_t count) {
    std::lock_guard<std::mutex> lock(impl_->config_mutex);
    impl_->tracker_config.num_active_beams = std::min(count, std::min(impl_->max_beams, MAX_TRACKER_BEAMS));
}

std::size_t MultiBeamBatchedTrackerStreamV5::num_active_beams() const {
    std::lock_guard<std::mutex> lock(impl_->config_mutex);
    return impl_->tracker_config.num_active_beams;
}

void MultiBeamBatchedTrackerStreamV5::process_batch(
    std::size_t first_window_index, const std::uint8_t* host_p, float* host_i) {
    const std::size_t p_bytes = voltage_sample_count(impl_->batch_dims) * sizeof(std::uint8_t);
    const std::size_t i_bytes = impl_->batch_dims.n_time * impl_->batch_dims.n_freq * impl_->max_beams * sizeof(float);

    check_cuda(cudaMemcpyAsync(impl_->d_packed, host_p, p_bytes, cudaMemcpyHostToDevice, impl_->stream), "H2D");
    process_batch_kernel_only(first_window_index);
    check_cuda(cudaMemcpyAsync(host_i, impl_->d_intensity, i_bytes, cudaMemcpyDeviceToHost, impl_->stream), "D2H");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
}

void MultiBeamBatchedTrackerStreamV5::process_batch_kernel_only(std::size_t first_window_index) {
    MultiTrackerConfig current_trk;
    {
        std::lock_guard<std::mutex> lock(impl_->config_mutex);
        current_trk = impl_->tracker_config;
    }

    if (current_trk.num_active_beams == 0) {
        impl_->last_time_ms = 0.0F;
        return; // Instantaneous bypass
    }

    const std::size_t num_active = current_trk.num_active_beams;
    for (std::size_t w = 0; w < impl_->batch_size; ++w) {
        for (std::size_t b = 0; b < num_active; ++b) {
            const Vec3 direction = tracker_window_direction(
                current_trk.trajectories[b], first_window_index + w, current_trk.integration_spectra);
            const std::size_t base = (w * num_active + b) * 3;
            impl_->h_window_directions[base + 0] = direction[0];
            impl_->h_window_directions[base + 1] = direction[1];
            impl_->h_window_directions[base + 2] = direction[2];
        }
    }

    check_cuda(cudaMemcpyAsync(impl_->d_window_directions, impl_->h_window_directions.data(),
                               impl_->batch_size * num_active * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync");

    check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "cudaEventRecord");

    const std::size_t total_warps = impl_->batch_size * impl_->chunks_per_window * impl_->batch_dims.n_freq * num_active;
    constexpr float spacing_m = default_spacing_m;

    switch (impl_->batch_dims.n_ant) {
        case 32:
            dispatch_kernel_multibeam<32, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, current_trk.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, num_active, impl_->max_beams, total_warps, impl_->stream);
            break;
        case 64:
            dispatch_kernel_multibeam<64, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, current_trk.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, num_active, impl_->max_beams, total_warps, impl_->stream);
            break;
        case 128:
            dispatch_kernel_multibeam<128, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, current_trk.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, num_active, impl_->max_beams, total_warps, impl_->stream);
            break;
        case 256:
            dispatch_kernel_multibeam<256, 8>(
                impl_->d_intensity, impl_->d_window_directions, impl_->d_wavenumbers, impl_->d_packed,
                impl_->batch_dims.n_time, impl_->batch_dims.n_freq, current_trk.integration_spectra,
                impl_->chunk_size, impl_->chunks_per_window, spacing_m, num_active, impl_->max_beams, total_warps, impl_->stream);
            break;
    }

    check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(impl_->stop_event), "cudaEventSynchronize");
    check_cuda(cudaEventElapsedTime(&impl_->last_time_ms, impl_->start_event, impl_->stop_event), "cudaEventElapsedTime");
}

float MultiBeamBatchedTrackerStreamV5::last_kernel_time_ms() const { return impl_->last_time_ms; }
std::uint8_t* MultiBeamBatchedTrackerStreamV5::device_packed_buffer() { return impl_->d_packed; }
float* MultiBeamBatchedTrackerStreamV5::device_intensity_buffer() { return impl_->d_intensity; }
void* MultiBeamBatchedTrackerStreamV5::device_stream() { return impl_->stream; }
std::size_t MultiBeamBatchedTrackerStreamV5::batch_size() const { return impl_->batch_size; }
std::size_t MultiBeamBatchedTrackerStreamV5::max_beams() const { return impl_->max_beams; }

} // namespace beamformer
