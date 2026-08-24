// src/cuda_beamformer_v2.cu
//
// CUDA Beamformer V2 — Unified Single-Engine Warp-Reduction Architecture for Fixed-Grid Beamforming.
//
// Target architectures: NVIDIA Blackwell, Hopper, Ada Lovelace, Ampere, Turing,
// Volta, and Pascal (SM_60+).

#include "beamformer/cuda_beamformer_v2.hpp"

#include "cuda_quantize.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

// ---------------------------------------------------------------------------
// Pinned Host Memory Helpers
// ---------------------------------------------------------------------------

void PinnedDeleterV2::operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

PinnedVectorV2<std::uint8_t> allocate_pinned_voltage_v2(const Dimensions& dims) {
    const std::size_t bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for packed voltage: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV2<std::uint8_t>(static_cast<std::uint8_t*>(ptr));
}

PinnedVectorV2<float> allocate_pinned_intensities_v2(const Dimensions& dims) {
    const std::size_t bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for intensities: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV2<float>(static_cast<float*>(ptr));
}

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

float event_elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

template <typename Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Value)),
                   "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Value* get() { return data_; }
    const Value* get() const { return data_; }
private:
    Value* data_ = nullptr;
};

// ---------------------------------------------------------------------------
// Fast Device Decoding Primitive (PTX bfe.s32)
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

// ---------------------------------------------------------------------------
// V2 Unified Direct Kernel Template (Unintegrated)
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
beamformer_v2_direct_kernel(
    float* __restrict__ intensity,
    const ComplexFloat* __restrict__ weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t n_beams,
    const std::size_t time_chunk_size,
    const std::size_t chunks_per_freq,
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

    const std::size_t beam = warp_id % n_beams;
    const std::size_t rest = warp_id / n_beams;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk = rest / n_freq;

    // Preload steering weights for this (beam, freq) pair and pre-negate imag component
    float w_r[ANT_PER_LANE];
    float w_i[ANT_PER_LANE];
    float nw_i[ANT_PER_LANE];

    const std::size_t weight_base = (beam * n_freq + freq) * N_ANT;
    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const std::size_t element = lane + a * 32U;
        const ComplexFloat w = weights[weight_base + element];
        w_r[a] = w.real;
        w_i[a] = w.imag;
        nw_i[a] = -w.imag;
    }

    const std::size_t t_chunk_start = chunk * time_chunk_size;
    if (t_chunk_start >= n_time) {
        return;
    }
    const std::size_t t_chunk_end = (t_chunk_start + time_chunk_size < n_time)
                                       ? (t_chunk_start + time_chunk_size)
                                       : n_time;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::size_t intensity_stride = n_freq * n_beams;

    const std::uint8_t* packed_ptr = packed + (t_chunk_start * n_freq + freq) * N_ANT + lane;
    float* intensity_ptr = intensity + (t_chunk_start * n_freq + freq) * n_beams + beam;

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
// V2 Unified Integrated Kernel Template (Fused Temporal Integration)
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
beamformer_v2_integrated_kernel(
    float* __restrict__ integrated_intensity,
    const ComplexFloat* __restrict__ weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t n_beams,
    const std::size_t integration_spectra,
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

    const std::size_t beam = warp_id % n_beams;
    const std::size_t rest = warp_id / n_beams;
    const std::size_t freq = rest % n_freq;
    const std::size_t window = rest / n_freq;

    // Preload steering weights for this (beam, freq) pair and pre-negate imag component
    float w_r[ANT_PER_LANE];
    float w_i[ANT_PER_LANE];
    float nw_i[ANT_PER_LANE];

    const std::size_t weight_base = (beam * n_freq + freq) * N_ANT;
    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const std::size_t element = lane + a * 32U;
        const ComplexFloat w = weights[weight_base + element];
        w_r[a] = w.real;
        w_i[a] = w.imag;
        nw_i[a] = -w.imag;
    }

    const std::size_t t_win_start = window * integration_spectra;
    const std::size_t t_win_end = (t_win_start + integration_spectra < n_time)
                                     ? (t_win_start + integration_spectra)
                                     : n_time;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::uint8_t* packed_ptr = packed + (t_win_start * n_freq + freq) * N_ANT + lane;

    float integrated_acc = 0.0F;
    std::size_t t = t_win_start;

    for (; t + (TIME_UNROLL - 1) < t_win_end; t += TIME_UNROLL) {
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
                integrated_acc += s_r[k] * s_r[k] + s_i[k] * s_i[k];
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
    }

    // Remainder loop
    for (; t < t_win_end; ++t) {
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
            integrated_acc += s_r * s_r + s_i * s_i;
        }

        packed_ptr += t_stride;
    }

    if (lane == 0) {
        integrated_intensity[(window * n_freq + freq) * n_beams + beam] = integrated_acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel Dispatchers
// ---------------------------------------------------------------------------

template <int N_ANT, int TIME_UNROLL>
void dispatch_direct_v2(
    float* d_intensity,
    const ComplexFloat* d_weights,
    const std::uint8_t* d_packed,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_beams,
    std::size_t time_chunk_size,
    std::size_t chunks_per_freq,
    std::size_t total_warps,
    cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    beamformer_v2_direct_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_intensity,
        d_weights,
        d_packed,
        n_time,
        n_freq,
        n_beams,
        time_chunk_size,
        chunks_per_freq,
        total_warps);
}

template <int N_ANT, int TIME_UNROLL>
void dispatch_integrated_v2(
    float* d_integrated_intensity,
    const ComplexFloat* d_weights,
    const std::uint8_t* d_packed,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_beams,
    std::size_t integration_spectra,
    std::size_t total_warps,
    cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    beamformer_v2_integrated_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_integrated_intensity,
        d_weights,
        d_packed,
        n_time,
        n_freq,
        n_beams,
        integration_spectra,
        total_warps);
}

void launch_direct_v2_auto(
    float* d_intensity,
    const ComplexFloat* d_weights,
    const std::uint8_t* d_packed,
    const Dimensions& dims,
    const V2BeamformerExecutionConfig& config,
    cudaStream_t stream) {

    const std::size_t chunk_size = (config.time_chunk_size > 0) ? config.time_chunk_size : dims.n_time;
    const std::size_t chunks_per_freq = (dims.n_time + chunk_size - 1) / chunk_size;
    const std::size_t total_warps = chunks_per_freq * dims.n_freq * dims.n_beams;

    auto dispatch_unroll = [&](auto n_ant_tag) {
        constexpr int N_ANT = decltype(n_ant_tag)::value;
        if (config.time_unroll == 2) {
            dispatch_direct_v2<N_ANT, 2>(d_intensity, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, chunk_size, chunks_per_freq, total_warps, stream);
        } else if (config.time_unroll == 4) {
            dispatch_direct_v2<N_ANT, 4>(d_intensity, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, chunk_size, chunks_per_freq, total_warps, stream);
        } else {
            dispatch_direct_v2<N_ANT, 8>(d_intensity, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, chunk_size, chunks_per_freq, total_warps, stream);
        }
    };

    if (dims.n_ant == 32) {
        dispatch_unroll(std::integral_constant<int, 32>{});
    } else if (dims.n_ant == 64) {
        dispatch_unroll(std::integral_constant<int, 64>{});
    } else if (dims.n_ant == 128) {
        dispatch_unroll(std::integral_constant<int, 128>{});
    } else if (dims.n_ant == 256) {
        dispatch_unroll(std::integral_constant<int, 256>{});
    } else {
        throw std::invalid_argument("CUDA Beamformer V2 supports n_ant = 32, 64, 128, 256");
    }
    check_cuda(cudaGetLastError(), "beamformer_v2_direct_kernel launch");
}

void launch_integrated_v2_auto(
    float* d_integrated,
    const ComplexFloat* d_weights,
    const std::uint8_t* d_packed,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    const V2BeamformerExecutionConfig& config,
    cudaStream_t stream) {

    validate_temporal_config(temporal_integration);
    const std::size_t window_count = integrated_time_count(dims.n_time, temporal_integration);
    const std::size_t total_warps = window_count * dims.n_freq * dims.n_beams;

    auto dispatch_unroll = [&](auto n_ant_tag) {
        constexpr int N_ANT = decltype(n_ant_tag)::value;
        if (config.time_unroll == 2) {
            dispatch_integrated_v2<N_ANT, 2>(d_integrated, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, temporal_integration.integration_spectra, total_warps, stream);
        } else if (config.time_unroll == 4) {
            dispatch_integrated_v2<N_ANT, 4>(d_integrated, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, temporal_integration.integration_spectra, total_warps, stream);
        } else {
            dispatch_integrated_v2<N_ANT, 8>(d_integrated, d_weights, d_packed, dims.n_time, dims.n_freq, dims.n_beams, temporal_integration.integration_spectra, total_warps, stream);
        }
    };

    if (dims.n_ant == 32) {
        dispatch_unroll(std::integral_constant<int, 32>{});
    } else if (dims.n_ant == 64) {
        dispatch_unroll(std::integral_constant<int, 64>{});
    } else if (dims.n_ant == 128) {
        dispatch_unroll(std::integral_constant<int, 128>{});
    } else if (dims.n_ant == 256) {
        dispatch_unroll(std::integral_constant<int, 256>{});
    } else {
        throw std::invalid_argument("CUDA Beamformer V2 supports n_ant = 32, 64, 128, 256");
    }
    check_cuda(cudaGetLastError(), "beamformer_v2_integrated_kernel launch");
}

bool same_shard(const ShardDescriptor& lhs, const ShardDescriptor& rhs) {
    return lhs.shard_id == rhs.shard_id
           && lhs.shard_count == rhs.shard_count
           && lhs.local_frequency_count == rhs.local_frequency_count
           && lhs.absolute_frequency_start == rhs.absolute_frequency_start
           && lhs.timestamp_start == rhs.timestamp_start
           && lhs.timestamp_step == rhs.timestamp_step
           && lhs.loss_mask_id == rhs.loss_mask_id
           && lhs.loss_mask_independent == rhs.loss_mask_independent;
}

void validate_matching_frame_metadata(const CudaFrameMetadata& reference,
                                      const CudaFrameMetadata& candidate,
                                      const char* label,
                                      const bool require_frame_id) {
    if (require_frame_id && reference.frame_id != candidate.frame_id) {
        throw std::invalid_argument(std::string(label) + " frame_id does not match input");
    }
    if (!same_shard(reference.shard, candidate.shard)) {
        throw std::invalid_argument(std::string(label) + " shard metadata does not match input");
    }
}

void validate_device_frame_shape(const CudaFrameView& frame,
                                 const CudaDataType type,
                                 const std::size_t rank,
                                 const std::array<std::size_t, cuda_frame_max_rank>& shape,
                                 const char* label) {
    validate_contiguous_cuda_buffer_view(frame.buffer);
    if (frame.buffer.type != type || frame.buffer.rank != rank
        || frame.buffer.shape != shape) {
        throw std::invalid_argument(std::string(label) + " shape or dtype does not match");
    }
}

} // namespace

// ---------------------------------------------------------------------------
// CudaBeamformerWorkspaceV2 Implementation
// ---------------------------------------------------------------------------
struct CudaBeamformerWorkspaceV2::Impl {
    explicit Impl(const Dimensions& requested_capacity,
                  const V2BeamformerExecutionConfig& exec_config,
                  const std::optional<TemporalIntegrationConfig>& selected_integration,
                  const CudaBeamformerOutputV2 selected_output)
        : capacity(requested_capacity),
          config(exec_config),
          temporal_integration(selected_integration),
          output(selected_output),
          device_packed_voltage(voltage_sample_count(capacity)),
          device_weights(capacity.n_beams * capacity.n_freq * capacity.n_ant),
          device_intensity(capacity.n_time * capacity.n_freq * capacity.n_beams) {

        if (temporal_integration) {
            device_integrated_intensity = std::make_unique<DeviceBuffer<float>>(
                integrated_intensity_count(capacity, *temporal_integration));
            if (output == CudaBeamformerOutputV2::QuantizedInt8) {
                const Dimensions integrated_capacity{
                    integrated_time_count(capacity.n_time, *temporal_integration),
                    capacity.n_freq, capacity.n_ant, capacity.n_beams};
                device_quantized_intensity = std::make_unique<DeviceBuffer<std::int8_t>>(
                    integrated_intensity_count(capacity, *temporal_integration));
                device_quantization_parameters =
                    std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
                        quantization_parameter_count(integrated_capacity));
            }
        }
    }

    void validate_request(const Dimensions& dims) const {
        validate_dimensions(dims);
        if (dims.n_freq != capacity.n_freq || dims.n_ant != capacity.n_ant
            || dims.n_time > capacity.n_time || dims.n_beams > capacity.n_beams) {
            throw std::invalid_argument("dimensions exceed CUDA workspace capacity");
        }
    }

    void require_loaded(const Dimensions& dims) const {
        if (loaded_packed_voltage_samples < voltage_sample_count(dims)) {
            throw std::logic_error("packed voltage has not been uploaded for these dimensions");
        }
        const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
        if (loaded_weights != required_weights) {
            throw std::logic_error("weights have not been uploaded for these dimensions");
        }
    }

    void require_temporal_integration() const {
        if (!temporal_integration || !device_integrated_intensity) {
            throw std::logic_error("CUDA workspace was not constructed with temporal integration");
        }
    }

    void require_quantization() const {
        require_temporal_integration();
        if (output != CudaBeamformerOutputV2::QuantizedInt8
            || !device_quantized_intensity || !device_quantization_parameters) {
            throw std::logic_error("CUDA workspace was not constructed with int8 quantization");
        }
    }

    Dimensions capacity;
    V2BeamformerExecutionConfig config;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutputV2 output;
    DeviceBuffer<std::uint8_t> device_packed_voltage;
    DeviceBuffer<ComplexFloat> device_weights;
    DeviceBuffer<float> device_intensity;
    std::unique_ptr<DeviceBuffer<float>> device_integrated_intensity;
    std::unique_ptr<DeviceBuffer<std::int8_t>> device_quantized_intensity;
    std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>> device_quantization_parameters;
    CudaStream stream;
    CudaEvent start;
    CudaEvent transfer_end;
    CudaEvent kernel_end;
    CudaEvent quantization_end;
    CudaEvent result_end;
    double measured_setup_ms = 0.0;
    std::size_t loaded_packed_voltage_samples = 0;
    std::size_t loaded_weights = 0;
};

CudaBeamformerWorkspaceV2::CudaBeamformerWorkspaceV2(
    const Dimensions& capacity,
    const V2BeamformerExecutionConfig& config,
    const std::optional<TemporalIntegrationConfig> temporal_integration,
    const CudaBeamformerOutputV2 output) {
    validate_dimensions(capacity);
    if (temporal_integration) {
        validate_temporal_config(*temporal_integration);
    }
    if (output == CudaBeamformerOutputV2::QuantizedInt8 && !temporal_integration) {
        throw std::invalid_argument(
            "int8 quantization requires a temporal-integration configuration");
    }
    const auto start = Clock::now();
    impl_ = std::make_unique<Impl>(capacity, config, temporal_integration, output);
    const auto end = Clock::now();
    impl_->measured_setup_ms = elapsed_ms(start, end);
}

CudaBeamformerWorkspaceV2::~CudaBeamformerWorkspaceV2() = default;

double CudaBeamformerWorkspaceV2::setup_ms() const {
    return impl_->measured_setup_ms;
}

const V2BeamformerExecutionConfig& CudaBeamformerWorkspaceV2::config() const {
    return impl_->config;
}

bool CudaBeamformerWorkspaceV2::has_temporal_integration() const {
    return impl_->temporal_integration.has_value();
}

bool CudaBeamformerWorkspaceV2::has_int8_quantization() const {
    return impl_->output == CudaBeamformerOutputV2::QuantizedInt8;
}

std::size_t CudaBeamformerWorkspaceV2::packed_voltage_capacity_bytes() const {
    return voltage_sample_count(impl_->capacity);
}

std::uint8_t* CudaBeamformerWorkspaceV2::device_packed_voltage() {
    return impl_->device_packed_voltage.get();
}

ComplexFloat* CudaBeamformerWorkspaceV2::device_weights() {
    return impl_->device_weights.get();
}

float* CudaBeamformerWorkspaceV2::device_intensity() {
    return impl_->device_intensity.get();
}

float* CudaBeamformerWorkspaceV2::device_integrated_intensity() {
    return impl_->device_integrated_intensity ? impl_->device_integrated_intensity->get() : nullptr;
}

void* CudaBeamformerWorkspaceV2::device_stream() {
    return static_cast<void*>(impl_->stream.get());
}

double CudaBeamformerWorkspaceV2::upload_packed_voltage(const PackedVoltage& packed,
                                                        const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = voltage_sample_count(dims);
    if (packed.size() < count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord packed voltage start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord packed voltage end");
    check_cuda(cudaEventSynchronize(impl_->transfer_end.get()),
               "cudaEventSynchronize packed voltage");
    impl_->loaded_packed_voltage_samples = count;
    return event_elapsed_ms(impl_->start, impl_->transfer_end);
}

double CudaBeamformerWorkspaceV2::upload_weights(const Weights& weights,
                                                 const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != count) {
        throw std::invalid_argument("weight count does not match dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord weights start");
    check_cuda(cudaMemcpyAsync(impl_->device_weights.get(), weights.data(),
                               count * sizeof(ComplexFloat), cudaMemcpyHostToDevice,
                               impl_->stream.get()),
               "cudaMemcpyAsync weights host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord weights end");
    check_cuda(cudaEventSynchronize(impl_->transfer_end.get()),
               "cudaEventSynchronize weights");
    impl_->loaded_weights = count;
    return event_elapsed_ms(impl_->start, impl_->transfer_end);
}

double CudaBeamformerWorkspaceV2::run_kernel(const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_loaded(dims);
    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord kernel start");
    launch_direct_v2_auto(
        impl_->device_intensity.get(),
        impl_->device_weights.get(),
        impl_->device_packed_voltage.get(),
        dims, impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord kernel end");
    check_cuda(cudaEventSynchronize(impl_->kernel_end.get()),
               "cudaEventSynchronize kernel");
    return event_elapsed_ms(impl_->start, impl_->kernel_end);
}

double CudaBeamformerWorkspaceV2::download_intensity(Intensities& intensity,
                                                     const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < count) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord intensity start");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity.get(),
                               count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord intensity end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize intensity");
    return event_elapsed_ms(impl_->start, impl_->result_end);
}

CudaBeamformerTimingsV2 CudaBeamformerWorkspaceV2::run_pipeline(
    const PackedVoltage& packed, Intensities& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.size() < output_count) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
    const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline transfer end");
    launch_direct_v2_auto(
        impl_->device_intensity.get(),
        impl_->device_weights.get(),
        impl_->device_packed_voltage.get(),
        dims, impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline kernel end");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity.get(),
                               output_count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync pipeline intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimingsV2 timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV2 CudaBeamformerWorkspaceV2::run_integrated_pipeline(
    const PackedVoltage& packed, IntegratedIntensities& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_temporal_integration();
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t output_count =
        integrated_intensity_count(dims, *impl_->temporal_integration);
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.size() < output_count) {
        throw std::invalid_argument("integrated intensity output is smaller than dimensions");
    }
    const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync integrated pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline transfer end");
    launch_integrated_v2_auto(
        impl_->device_integrated_intensity->get(),
        impl_->device_weights.get(),
        impl_->device_packed_voltage.get(),
        dims, *impl_->temporal_integration, impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline fused beamformer end");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_integrated_intensity->get(),
                               output_count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync integrated pipeline intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize integrated pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimingsV2 timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV2 CudaBeamformerWorkspaceV2::run_quantized_integrated_pipeline(
    const PackedVoltage& packed, QuantizedIntegratedOutput& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_quantization();
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t integrated_count =
        integrated_intensity_count(dims, *impl_->temporal_integration);
    const Dimensions integrated_dims{
        integrated_time_count(dims.n_time, *impl_->temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    const std::size_t parameter_count = quantization_parameter_count(integrated_dims);
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.codes.size() < integrated_count
        || intensity.parameters.size() < parameter_count) {
        throw std::invalid_argument("quantized output is smaller than dimensions");
    }
    const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline transfer end");
    launch_integrated_v2_auto(
        impl_->device_integrated_intensity->get(),
        impl_->device_weights.get(),
        impl_->device_packed_voltage.get(),
        dims, *impl_->temporal_integration, impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline fused beamformer end");
    launch_quantize_integrated_intensity(
        impl_->stream.get(), impl_->device_integrated_intensity->get(),
        impl_->device_quantized_intensity->get(),
        impl_->device_quantization_parameters->get(), integrated_dims);
    check_cuda(cudaEventRecord(impl_->quantization_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline quantization end");
    check_cuda(cudaMemcpyAsync(intensity.codes.data(), impl_->device_quantized_intensity->get(),
                               integrated_count * sizeof(std::int8_t), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline codes device to host");
    check_cuda(cudaMemcpyAsync(intensity.parameters.data(),
                               impl_->device_quantization_parameters->get(),
                               parameter_count * sizeof(Int8QuantizationParameters),
                               cudaMemcpyDeviceToHost, impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline parameters device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize quantized pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimingsV2 timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.quantization_ms =
        event_elapsed_ms(impl_->kernel_end, impl_->quantization_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->quantization_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV2 CudaBeamformerWorkspaceV2::run_device_frame(
    const CudaFrameView& voltage, const CudaFrameView& weights,
    CudaFrameView& output, CudaFrameView* quantization_parameters) {

    validate_cuda_frame_view(voltage);
    validate_cuda_frame_view(weights);
    validate_cuda_frame_view(output);
    validate_matching_frame_metadata(voltage.metadata, output.metadata, "output", true);

    if (voltage.buffer.type != CudaDataType::PackedInt4x2 || voltage.buffer.rank != 3) {
        throw std::invalid_argument("offline device frame voltage must be packed int4x2 with rank 3");
    }
    const Dimensions dims{
        voltage.buffer.shape[0], voltage.buffer.shape[1], voltage.buffer.shape[2],
        output.buffer.shape[2]};
    impl_->validate_request(dims);
    if (voltage.buffer.shape[1] != impl_->capacity.n_freq
        || output.buffer.rank != 3
        || output.buffer.shape[1] != dims.n_freq
        || output.buffer.shape[2] != dims.n_beams) {
        throw std::invalid_argument("offline device frame dimensions do not match");
    }

    const std::array<std::size_t, cuda_frame_max_rank> expected_weight_shape = {
        dims.n_beams, dims.n_freq, dims.n_ant, 0};
    validate_device_frame_shape(weights, CudaDataType::ComplexFloat32, 3,
                                expected_weight_shape, "offline device frame weights");
    if (!same_shard(voltage.metadata.shard, weights.metadata.shard)) {
        throw std::invalid_argument("offline device frame weights shard does not match input");
    }

    Dimensions output_dims = dims;
    if (impl_->temporal_integration) {
        output_dims.n_time = integrated_time_count(dims.n_time, *impl_->temporal_integration);
    }
    const auto expected_output_type =
        impl_->output == CudaBeamformerOutputV2::QuantizedInt8
            ? CudaDataType::SignedInt8
            : CudaDataType::Float32;
    validate_device_frame_shape(
        output, expected_output_type, 3,
        {output_dims.n_time, output_dims.n_freq, output_dims.n_beams, 0},
        "offline device frame output");

    if (impl_->output == CudaBeamformerOutputV2::QuantizedInt8) {
        if (!impl_->temporal_integration || quantization_parameters == nullptr) {
            throw std::invalid_argument("offline int8 output requires temporal integration parameters");
        }
        validate_cuda_frame_view(*quantization_parameters);
        validate_matching_frame_metadata(
            voltage.metadata, quantization_parameters->metadata, "quantization parameters", true);
        const auto layout = quantization_layout(output_dims);
        validate_device_frame_shape(
            *quantization_parameters, CudaDataType::Float32, 4,
            {layout.time_tiles, layout.frequency_tiles, layout.beam_tiles, 2},
            "offline quantization parameters");
    } else if (quantization_parameters != nullptr) {
        throw std::invalid_argument("float32 offline output must not provide quantization parameters");
    }

    const auto stream = impl_->stream.get();
    check_cuda(cudaEventRecord(impl_->start.get(), stream), "cudaEventRecord offline frame start");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), stream), "cudaEventRecord offline frame input end");

    if (impl_->temporal_integration) {
        launch_integrated_v2_auto(
            impl_->device_integrated_intensity->get(),
            static_cast<const ComplexFloat*>(weights.buffer.device_data),
            static_cast<const std::uint8_t*>(voltage.buffer.device_data),
            dims, *impl_->temporal_integration, impl_->config, stream);
    } else {
        launch_direct_v2_auto(
            impl_->device_intensity.get(),
            static_cast<const ComplexFloat*>(weights.buffer.device_data),
            static_cast<const std::uint8_t*>(voltage.buffer.device_data),
            dims, impl_->config, stream);
    }
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), stream), "cudaEventRecord offline frame beamformer end");

    CudaBeamformerTimingsV2 timings;
    timings.host_to_device_ms = 0.0;
    const CudaEvent* producer_end = &impl_->kernel_end;

    if (impl_->output == CudaBeamformerOutputV2::QuantizedInt8) {
        const Dimensions integrated_dims{
            integrated_time_count(dims.n_time, *impl_->temporal_integration),
            dims.n_freq, dims.n_ant, dims.n_beams};
        launch_quantize_integrated_intensity(
            stream, impl_->device_integrated_intensity->get(),
            impl_->device_quantized_intensity->get(),
            impl_->device_quantization_parameters->get(), integrated_dims);
        check_cuda(cudaEventRecord(impl_->quantization_end.get(), stream),
                   "cudaEventRecord offline frame quantization end");
        producer_end = &impl_->quantization_end;
    }

    const std::size_t output_bytes =
        impl_->output == CudaBeamformerOutputV2::QuantizedInt8
            ? quantized_intensity_bytes(output_dims)
            : intensity_bytes(output_dims);
    const void* source = impl_->output == CudaBeamformerOutputV2::QuantizedInt8
                             ? static_cast<const void*>(impl_->device_quantized_intensity->get())
                             : static_cast<const void*>(impl_->temporal_integration
                                                            ? impl_->device_integrated_intensity->get()
                                                            : impl_->device_intensity.get());
    check_cuda(cudaMemcpyAsync(output.buffer.device_data, source, output_bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync offline frame output device to device");
    if (impl_->output == CudaBeamformerOutputV2::QuantizedInt8) {
        const auto parameter_bytes =
            quantization_parameter_count(output_dims) * sizeof(Int8QuantizationParameters);
        check_cuda(cudaMemcpyAsync(
                       quantization_parameters->buffer.device_data,
                       impl_->device_quantization_parameters->get(), parameter_bytes,
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync offline frame parameters device to device");
    }
    check_cuda(cudaEventRecord(impl_->result_end.get(), stream), "cudaEventRecord offline frame output end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()), "cudaEventSynchronize offline frame");

    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    if (impl_->output == CudaBeamformerOutputV2::QuantizedInt8) {
        timings.quantization_ms = event_elapsed_ms(impl_->kernel_end, impl_->quantization_end);
    }
    timings.device_to_device_ms = event_elapsed_ms(*producer_end, impl_->result_end);
    timings.device_to_host_ms = 0.0;
    return timings;
}

// ---------------------------------------------------------------------------
// Standalone Functional APIs
// ---------------------------------------------------------------------------
Intensities cuda_beamform_v2_packed_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    CudaBeamformerTimingsV2* timings,
    const V2BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match dimensions");
    }

    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    CudaBeamformerWorkspaceV2 workspace(dims, config);
    CudaBeamformerTimingsV2 measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_packed_voltage(packed, dims)
                                 + workspace.upload_weights(weights, dims);
    measured.kernel_ms = workspace.run_kernel(dims);
    measured.device_to_host_ms = workspace.download_intensity(intensity, dims);
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

void cuda_beamform_v2_packed_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    Intensities& intensity,
    CudaBeamformerTimingsV2* timings,
    const V2BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    const std::size_t required_output = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity buffer is smaller than required output");
    }

    CudaBeamformerWorkspaceV2 workspace(dims, config);
    workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_pipeline(packed, intensity, dims);
    if (timings != nullptr) {
        *timings = pipeline;
        timings->setup_ms = workspace.setup_ms();
    }
}

IntegratedIntensities cuda_beamform_v2_packed_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV2* timings,
    const V2BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    validate_temporal_config(temporal_integration);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match dimensions");
    }

    IntegratedIntensities intensity(integrated_intensity_count(dims, temporal_integration));
    CudaBeamformerWorkspaceV2 workspace(dims, config, temporal_integration);
    CudaBeamformerTimingsV2 measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_integrated_pipeline(packed, intensity, dims);
    measured.host_to_device_ms += pipeline.host_to_device_ms;
    measured.kernel_ms = pipeline.kernel_ms;
    measured.device_to_host_ms = pipeline.device_to_host_ms;
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

void cuda_beamform_v2_packed_integrated_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    IntegratedIntensities& intensity,
    CudaBeamformerTimingsV2* timings,
    const V2BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    validate_temporal_config(temporal_integration);
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    const std::size_t required_output = integrated_intensity_count(dims, temporal_integration);
    if (intensity.size() < required_output) {
        throw std::invalid_argument("integrated intensity buffer is smaller than required output");
    }

    CudaBeamformerWorkspaceV2 workspace(dims, config, temporal_integration);
    workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_integrated_pipeline(packed, intensity, dims);
    if (timings != nullptr) {
        *timings = pipeline;
        timings->setup_ms = workspace.setup_ms();
    }
}

QuantizedIntegratedOutput cuda_beamform_v2_packed_quantized_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV2* timings,
    const V2BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    validate_temporal_config(temporal_integration);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    const Dimensions integrated_dims{
        integrated_time_count(dims.n_time, temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    QuantizedIntegratedOutput intensity{
        QuantizedIntensities(integrated_intensity_count(dims, temporal_integration)),
        std::vector<Int8QuantizationParameters>(quantization_parameter_count(integrated_dims)),
    };

    CudaBeamformerWorkspaceV2 workspace(
        dims, config, temporal_integration, CudaBeamformerOutputV2::QuantizedInt8);
    CudaBeamformerTimingsV2 measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_quantized_integrated_pipeline(packed, intensity, dims);
    measured.host_to_device_ms += pipeline.host_to_device_ms;
    measured.kernel_ms = pipeline.kernel_ms;
    measured.quantization_ms = pipeline.quantization_ms;
    measured.device_to_host_ms = pipeline.device_to_host_ms;
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

// ---------------------------------------------------------------------------
// Device-Resident APIs
// ---------------------------------------------------------------------------
void cuda_beamform_v2_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_intensity,
    const Dimensions& dims,
    const V2BeamformerExecutionConfig& config,
    void* stream_ptr) {

    validate_dimensions(dims);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    launch_direct_v2_auto(d_intensity, d_weights, d_packed, dims, config, stream);
}

void cuda_beamform_v2_integrated_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_integrated_intensity,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    const V2BeamformerExecutionConfig& config,
    void* stream_ptr) {

    validate_dimensions(dims);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    launch_integrated_v2_auto(d_integrated_intensity, d_weights, d_packed, dims,
                              temporal_integration, config, stream);
}

// ---------------------------------------------------------------------------
// Batched Persistent Pipeline & CUDA Graph Stream
// ---------------------------------------------------------------------------
struct BatchedBeamformerStreamV2::Impl {
    Impl(const Dimensions& d, const Weights& w, const V2BeamformerExecutionConfig& c)
        : dims(d), config(c) {
        validate_dimensions(dims);
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&event_start), "cudaEventCreate");
        check_cuda(cudaEventCreate(&event_end), "cudaEventCreate");

        voltage_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
        intensity_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
        const std::size_t weight_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);

        check_cuda(cudaMalloc(&d_packed, voltage_bytes), "cudaMalloc d_packed");
        check_cuda(cudaMalloc(&d_intensity, intensity_bytes), "cudaMalloc d_intensity");
        check_cuda(cudaMalloc(&d_weights, weight_bytes), "cudaMalloc d_weights");
        check_cuda(cudaMemcpyAsync(d_weights, w.data(), weight_bytes, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync d_weights");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize weights");
    }

    ~Impl() {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (d_packed) cudaFree(d_packed);
        if (d_intensity) cudaFree(d_intensity);
        if (d_weights) cudaFree(d_weights);
        if (event_start) cudaEventDestroy(event_start);
        if (event_end) cudaEventDestroy(event_end);
        if (stream) cudaStreamDestroy(stream);
    }

    Dimensions dims;
    V2BeamformerExecutionConfig config;
    cudaStream_t stream = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_end = nullptr;
    std::uint8_t* d_packed = nullptr;
    float* d_intensity = nullptr;
    ComplexFloat* d_weights = nullptr;
    std::size_t voltage_bytes = 0;
    std::size_t intensity_bytes = 0;
    float last_time_ms = 0.0F;

    bool graph_captured = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
};

BatchedBeamformerStreamV2::BatchedBeamformerStreamV2(
    const Dimensions& dims, const Weights& weights,
    const V2BeamformerExecutionConfig& config)
    : impl_(std::make_unique<Impl>(dims, weights, config)) {}

BatchedBeamformerStreamV2::~BatchedBeamformerStreamV2() = default;

void BatchedBeamformerStreamV2::process_batch(
    const std::uint8_t* host_packed, float* host_intensity) {

    check_cuda(cudaMemcpyAsync(impl_->d_packed, host_packed, impl_->voltage_bytes,
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync host to device");

    process_batch_kernel_only();

    check_cuda(cudaMemcpyAsync(host_intensity, impl_->d_intensity, impl_->intensity_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync device to host");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize batch");
}

void BatchedBeamformerStreamV2::process_batch_kernel_only() {
    check_cuda(cudaEventRecord(impl_->event_start, impl_->stream), "cudaEventRecord start");

    if (impl_->config.enable_cuda_graph) {
        if (!impl_->graph_captured) {
            check_cuda(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeGlobal),
                       "cudaStreamBeginCapture");
            launch_direct_v2_auto(
                impl_->d_intensity, impl_->d_weights, impl_->d_packed,
                impl_->dims, impl_->config, impl_->stream);
            check_cuda(cudaStreamEndCapture(impl_->stream, &impl_->graph),
                       "cudaStreamEndCapture");
            check_cuda(cudaGraphInstantiate(&impl_->graph_exec, impl_->graph, nullptr, nullptr, 0),
                       "cudaGraphInstantiate");
            impl_->graph_captured = true;
        }
        check_cuda(cudaGraphLaunch(impl_->graph_exec, impl_->stream), "cudaGraphLaunch");
    } else {
        launch_direct_v2_auto(
            impl_->d_intensity, impl_->d_weights, impl_->d_packed,
            impl_->dims, impl_->config, impl_->stream);
    }

    check_cuda(cudaEventRecord(impl_->event_end, impl_->stream), "cudaEventRecord end");
    check_cuda(cudaEventSynchronize(impl_->event_end), "cudaEventSynchronize kernel");
    check_cuda(cudaEventElapsedTime(&impl_->last_time_ms, impl_->event_start, impl_->event_end),
               "cudaEventElapsedTime");
}

float BatchedBeamformerStreamV2::last_kernel_time_ms() const {
    return impl_->last_time_ms;
}

std::uint8_t* BatchedBeamformerStreamV2::device_packed_buffer() {
    return impl_->d_packed;
}

float* BatchedBeamformerStreamV2::device_intensity_buffer() {
    return impl_->d_intensity;
}

void* BatchedBeamformerStreamV2::device_stream() {
    return static_cast<void*>(impl_->stream);
}

} // namespace beamformer
