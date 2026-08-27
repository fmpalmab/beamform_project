// src/cuda_beamformer_v3.cu
//
// CUDA Beamformer V3 — Next-Generation Multi-Beam Warp-Cooperative Architecture
// for High-Throughput Fixed-Grid Voltage Beamforming.
//
// Target architectures: NVIDIA Blackwell, Hopper, Ada Lovelace, Ampere, Turing,
// Volta, and Pascal (SM_60+).

#include "beamformer/cuda_beamformer_v3.hpp"

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

void PinnedDeleterV3::operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

PinnedVectorV3<std::uint8_t> allocate_pinned_voltage_v3(const Dimensions& dims) {
    const std::size_t bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for packed voltage: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV3<std::uint8_t>(static_cast<std::uint8_t*>(ptr));
}

PinnedVectorV3<float> allocate_pinned_intensities_v3(const Dimensions& dims) {
    const std::size_t bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
    void* ptr = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for intensities: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return PinnedVectorV3<float>(static_cast<float*>(ptr));
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
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
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
// V3 Dual-Beam Direct Kernel Template (Unintegrated, BEAMS_PER_WARP = 2)
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
beamformer_v3_dualbeam_direct_kernel(
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

    const std::size_t num_beam_pairs = (n_beams + 1) / 2;
    const std::size_t beam_pair = warp_id % num_beam_pairs;
    const std::size_t rest = warp_id / num_beam_pairs;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk = rest / n_freq;

    const std::size_t beam0 = beam_pair * 2;
    const std::size_t beam1 = beam0 + 1;
    const bool has_beam1 = (beam1 < n_beams);

    // Preload steering weights for beam0 and beam1
    float w_r0[ANT_PER_LANE], w_i0[ANT_PER_LANE], nw_i0[ANT_PER_LANE];
    float w_r1[ANT_PER_LANE], w_i1[ANT_PER_LANE], nw_i1[ANT_PER_LANE];

    const std::size_t weight_base0 = (beam0 * n_freq + freq) * N_ANT;
    const std::size_t weight_base1 = has_beam1 ? (beam1 * n_freq + freq) * N_ANT : 0;

    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const std::size_t element = lane + a * 32U;
        const ComplexFloat w0 = weights[weight_base0 + element];
        w_r0[a] = w0.real;
        w_i0[a] = w0.imag;
        nw_i0[a] = -w0.imag;

        if (has_beam1) {
            const ComplexFloat w1 = weights[weight_base1 + element];
            w_r1[a] = w1.real;
            w_i1[a] = w1.imag;
            nw_i1[a] = -w1.imag;
        }
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
    float* intensity_ptr0 = intensity + (t_chunk_start * n_freq + freq) * n_beams + beam0;
    float* intensity_ptr1 = has_beam1 ? (intensity + (t_chunk_start * n_freq + freq) * n_beams + beam1) : nullptr;

    std::size_t t = t_chunk_start;

    for (; t + (TIME_UNROLL - 1) < t_chunk_end; t += TIME_UNROLL) {
        float s_r0[TIME_UNROLL] = {0.0F};
        float s_i0[TIME_UNROLL] = {0.0F};
        float s_r1[TIME_UNROLL] = {0.0F};
        float s_i1[TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float wr0 = w_r0[a];
            const float wi0 = w_i0[a];
            const float nwi0 = nw_i0[a];

            const float wr1 = w_r1[a];
            const float wi1 = w_i1[a];
            const float nwi1 = nw_i1[a];

            const unsigned int a_offset = a * 32U;

            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                // Shared voltage fetch across both beams!
                const float2 p = unpack_int4_fast(packed_ptr[k * t_stride + a_offset]);
                
                s_r0[k] = fmaf(wr0, p.x, fmaf(nwi0, p.y, s_r0[k]));
                s_i0[k] = fmaf(wr0, p.y, fmaf(wi0, p.x, s_i0[k]));

                if (has_beam1) {
                    s_r1[k] = fmaf(wr1, p.x, fmaf(nwi1, p.y, s_r1[k]));
                    s_i1[k] = fmaf(wr1, p.y, fmaf(wi1, p.x, s_i1[k]));
                }
            }
        }

        // Intra-warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                s_r0[k] += __shfl_down_sync(full_mask, s_r0[k], offset);
                s_i0[k] += __shfl_down_sync(full_mask, s_i0[k], offset);
                if (has_beam1) {
                    s_r1[k] += __shfl_down_sync(full_mask, s_r1[k], offset);
                    s_i1[k] += __shfl_down_sync(full_mask, s_i1[k], offset);
                }
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                intensity_ptr0[k * intensity_stride] = s_r0[k] * s_r0[k] + s_i0[k] * s_i0[k];
                if (has_beam1) {
                    intensity_ptr1[k * intensity_stride] = s_r1[k] * s_r1[k] + s_i1[k] * s_i1[k];
                }
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
        intensity_ptr0 += TIME_UNROLL * intensity_stride;
        if (has_beam1) intensity_ptr1 += TIME_UNROLL * intensity_stride;
    }

    // Remainder loop
    for (; t < t_chunk_end; ++t) {
        float s_r0 = 0.0F, s_i0 = 0.0F;
        float s_r1 = 0.0F, s_i1 = 0.0F;

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_fast(packed_ptr[a * 32U]);
            s_r0 = fmaf(w_r0[a], p.x, fmaf(nw_i0[a], p.y, s_r0));
            s_i0 = fmaf(w_r0[a], p.y, fmaf(w_i0[a], p.x, s_i0));

            if (has_beam1) {
                s_r1 = fmaf(w_r1[a], p.x, fmaf(nw_i1[a], p.y, s_r1));
                s_i1 = fmaf(w_r1[a], p.y, fmaf(w_i1[a], p.x, s_i1));
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_r0 += __shfl_down_sync(full_mask, s_r0, offset);
            s_i0 += __shfl_down_sync(full_mask, s_i0, offset);
            if (has_beam1) {
                s_r1 += __shfl_down_sync(full_mask, s_r1, offset);
                s_i1 += __shfl_down_sync(full_mask, s_i1, offset);
            }
        }

        if (lane == 0) {
            *intensity_ptr0 = s_r0 * s_r0 + s_i0 * s_i0;
            if (has_beam1) {
                *intensity_ptr1 = s_r1 * s_r1 + s_i1 * s_i1;
            }
        }

        packed_ptr += t_stride;
        intensity_ptr0 += intensity_stride;
        if (has_beam1) intensity_ptr1 += intensity_stride;
    }
}

// ---------------------------------------------------------------------------
// V3 Dual-Beam Integrated Kernel Template (Fused Temporal Integration)
// ---------------------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
beamformer_v3_dualbeam_integrated_kernel(
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

    const std::size_t num_beam_pairs = (n_beams + 1) / 2;
    const std::size_t beam_pair = warp_id % num_beam_pairs;
    const std::size_t rest = warp_id / num_beam_pairs;
    const std::size_t freq = rest % n_freq;
    const std::size_t window = rest / n_freq;

    const std::size_t beam0 = beam_pair * 2;
    const std::size_t beam1 = beam0 + 1;
    const bool has_beam1 = (beam1 < n_beams);

    // Preload steering weights
    float w_r0[ANT_PER_LANE], w_i0[ANT_PER_LANE], nw_i0[ANT_PER_LANE];
    float w_r1[ANT_PER_LANE], w_i1[ANT_PER_LANE], nw_i1[ANT_PER_LANE];

    const std::size_t weight_base0 = (beam0 * n_freq + freq) * N_ANT;
    const std::size_t weight_base1 = has_beam1 ? (beam1 * n_freq + freq) * N_ANT : 0;

    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const std::size_t element = lane + a * 32U;
        const ComplexFloat w0 = weights[weight_base0 + element];
        w_r0[a] = w0.real;
        w_i0[a] = w0.imag;
        nw_i0[a] = -w0.imag;

        if (has_beam1) {
            const ComplexFloat w1 = weights[weight_base1 + element];
            w_r1[a] = w1.real;
            w_i1[a] = w1.imag;
            nw_i1[a] = -w1.imag;
        }
    }

    const std::size_t t_win_start = window * integration_spectra;
    const std::size_t t_win_end = (t_win_start + integration_spectra < n_time)
                                     ? (t_win_start + integration_spectra)
                                     : n_time;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::uint8_t* packed_ptr = packed + (t_win_start * n_freq + freq) * N_ANT + lane;

    float integrated_acc0 = 0.0F;
    float integrated_acc1 = 0.0F;
    std::size_t t = t_win_start;

    for (; t + (TIME_UNROLL - 1) < t_win_end; t += TIME_UNROLL) {
        float s_r0[TIME_UNROLL] = {0.0F};
        float s_i0[TIME_UNROLL] = {0.0F};
        float s_r1[TIME_UNROLL] = {0.0F};
        float s_i1[TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float wr0 = w_r0[a];
            const float wi0 = w_i0[a];
            const float nwi0 = nw_i0[a];

            const float wr1 = w_r1[a];
            const float wi1 = w_i1[a];
            const float nwi1 = nw_i1[a];

            const unsigned int a_offset = a * 32U;

            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                const float2 p = unpack_int4_fast(packed_ptr[k * t_stride + a_offset]);
                s_r0[k] = fmaf(wr0, p.x, fmaf(nwi0, p.y, s_r0[k]));
                s_i0[k] = fmaf(wr0, p.y, fmaf(wi0, p.x, s_i0[k]));

                if (has_beam1) {
                    s_r1[k] = fmaf(wr1, p.x, fmaf(nwi1, p.y, s_r1[k]));
                    s_i1[k] = fmaf(wr1, p.y, fmaf(wi1, p.x, s_i1[k]));
                }
            }
        }

        // Intra-warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                s_r0[k] += __shfl_down_sync(full_mask, s_r0[k], offset);
                s_i0[k] += __shfl_down_sync(full_mask, s_i0[k], offset);
                if (has_beam1) {
                    s_r1[k] += __shfl_down_sync(full_mask, s_r1[k], offset);
                    s_i1[k] += __shfl_down_sync(full_mask, s_i1[k], offset);
                }
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                integrated_acc0 += s_r0[k] * s_r0[k] + s_i0[k] * s_i0[k];
                if (has_beam1) {
                    integrated_acc1 += s_r1[k] * s_r1[k] + s_i1[k] * s_i1[k];
                }
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
    }

    // Remainder loop
    for (; t < t_win_end; ++t) {
        float s_r0 = 0.0F, s_i0 = 0.0F;
        float s_r1 = 0.0F, s_i1 = 0.0F;

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_fast(packed_ptr[a * 32U]);
            s_r0 = fmaf(w_r0[a], p.x, fmaf(nw_i0[a], p.y, s_r0));
            s_i0 = fmaf(w_r0[a], p.y, fmaf(w_i0[a], p.x, s_i0));

            if (has_beam1) {
                s_r1 = fmaf(w_r1[a], p.x, fmaf(nw_i1[a], p.y, s_r1));
                s_i1 = fmaf(w_r1[a], p.y, fmaf(w_i1[a], p.x, s_i1));
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_r0 += __shfl_down_sync(full_mask, s_r0, offset);
            s_i0 += __shfl_down_sync(full_mask, s_i0, offset);
            if (has_beam1) {
                s_r1 += __shfl_down_sync(full_mask, s_r1, offset);
                s_i1 += __shfl_down_sync(full_mask, s_i1, offset);
            }
        }

        if (lane == 0) {
            integrated_acc0 += s_r0 * s_r0 + s_i0 * s_i0;
            if (has_beam1) {
                integrated_acc1 += s_r1 * s_r1 + s_i1 * s_i1;
            }
        }

        packed_ptr += t_stride;
    }

    if (lane == 0) {
        const std::size_t out_base = (window * n_freq + freq) * n_beams;
        integrated_intensity[out_base + beam0] = integrated_acc0;
        if (has_beam1) {
            integrated_intensity[out_base + beam1] = integrated_acc1;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel Launch Dispatchers
// ---------------------------------------------------------------------------

template <int N_ANT, int TIME_UNROLL>
void dispatch_v3_direct(
    float* d_intensity, const ComplexFloat* d_weights, const std::uint8_t* d_packed,
    std::size_t n_time, std::size_t n_freq, std::size_t n_beams,
    std::size_t time_chunk_size, std::size_t chunks_per_freq,
    std::size_t total_warps, cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    beamformer_v3_dualbeam_direct_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_intensity, d_weights, d_packed,
        n_time, n_freq, n_beams, time_chunk_size, chunks_per_freq, total_warps);
}

template <int N_ANT, int TIME_UNROLL>
void dispatch_v3_integrated(
    float* d_integrated, const ComplexFloat* d_weights, const std::uint8_t* d_packed,
    std::size_t n_time, std::size_t n_freq, std::size_t n_beams,
    std::size_t integration_spectra, std::size_t total_warps, cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    beamformer_v3_dualbeam_integrated_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_integrated, d_weights, d_packed,
        n_time, n_freq, n_beams, integration_spectra, total_warps);
}

} // namespace

// ---------------------------------------------------------------------------
// Device-Resident APIs
// ---------------------------------------------------------------------------

void cuda_beamform_v3_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_intensity,
    const Dimensions& dims,
    const V3BeamformerExecutionConfig& config,
    void* stream_ptr) {

    validate_dimensions(dims);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    const std::size_t chunk_size = (config.time_chunk_size > 0) ? config.time_chunk_size : dims.n_time;
    const std::size_t chunks_per_freq = (dims.n_time + chunk_size - 1) / chunk_size;
    const std::size_t num_beam_pairs = (dims.n_beams + 1) / 2;
    const std::size_t total_warps = chunks_per_freq * dims.n_freq * num_beam_pairs;

    auto dispatch_ant = [&](auto n_ant_ic) {
        constexpr int N_A = decltype(n_ant_ic)::value;
        if (config.time_unroll >= 8) {
            dispatch_v3_direct<N_A, 8>(
                d_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                chunk_size, chunks_per_freq, total_warps, stream);
        } else if (config.time_unroll >= 4) {
            dispatch_v3_direct<N_A, 4>(
                d_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                chunk_size, chunks_per_freq, total_warps, stream);
        } else {
            dispatch_v3_direct<N_A, 2>(
                d_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                chunk_size, chunks_per_freq, total_warps, stream);
        }
    };

    switch (dims.n_ant) {
        case 32:  dispatch_ant(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_ant(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_ant(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_ant(std::integral_constant<int, 256>{}); break;
        default:
            throw std::invalid_argument("Unsupported n_ant: must be 32, 64, 128, or 256");
    }
}

void cuda_beamform_v3_integrated_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_integrated_intensity,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    const V3BeamformerExecutionConfig& config,
    void* stream_ptr) {

    validate_dimensions(dims);
    validate_temporal_config(temporal_integration);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    const std::size_t window_count =
        integrated_time_count(dims.n_time, temporal_integration);
    const std::size_t num_beam_pairs = (dims.n_beams + 1) / 2;
    const std::size_t total_warps = window_count * dims.n_freq * num_beam_pairs;

    auto dispatch_ant = [&](auto n_ant_ic) {
        constexpr int N_A = decltype(n_ant_ic)::value;
        if (config.time_unroll >= 8) {
            dispatch_v3_integrated<N_A, 8>(
                d_integrated_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                temporal_integration.integration_spectra, total_warps, stream);
        } else if (config.time_unroll >= 4) {
            dispatch_v3_integrated<N_A, 4>(
                d_integrated_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                temporal_integration.integration_spectra, total_warps, stream);
        } else {
            dispatch_v3_integrated<N_A, 2>(
                d_integrated_intensity, d_weights, d_packed,
                dims.n_time, dims.n_freq, dims.n_beams,
                temporal_integration.integration_spectra, total_warps, stream);
        }
    };

    switch (dims.n_ant) {
        case 32:  dispatch_ant(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_ant(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_ant(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_ant(std::integral_constant<int, 256>{}); break;
        default:
            throw std::invalid_argument("Unsupported n_ant: must be 32, 64, 128, or 256");
    }
}

// ---------------------------------------------------------------------------
// Standalone Functional APIs
// ---------------------------------------------------------------------------

void cuda_beamform_v3_packed_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    Intensities& intensity,
    CudaBeamformerTimingsV3* timings,
    const V3BeamformerExecutionConfig& config) {

    validate_dimensions(dims);

    const std::size_t packed_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t weights_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
    const std::size_t intensity_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);

    CudaStream stream;
    CudaEvent setup_end, h2d_end, kernel_end, d2h_end;

    std::uint8_t* d_packed = nullptr;
    ComplexFloat* d_weights = nullptr;
    float* d_intensity = nullptr;

    check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc d_packed");
    check_cuda(cudaMalloc(&d_weights, weights_bytes), "cudaMalloc d_weights");
    check_cuda(cudaMalloc(&d_intensity, intensity_bytes), "cudaMalloc d_intensity");

    check_cuda(cudaEventRecord(setup_end.get(), stream.get()), "setup_end");

    check_cuda(cudaMemcpyAsync(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice, stream.get()), "H2D packed");
    check_cuda(cudaMemcpyAsync(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice, stream.get()), "H2D weights");
    check_cuda(cudaEventRecord(h2d_end.get(), stream.get()), "h2d_end");

    cuda_beamform_v3_device_resident(d_packed, d_weights, d_intensity, dims, config, stream.get());
    check_cuda(cudaEventRecord(kernel_end.get(), stream.get()), "kernel_end");

    check_cuda(cudaMemcpyAsync(intensity.data(), d_intensity, intensity_bytes, cudaMemcpyDeviceToHost, stream.get()), "D2H intensity");
    check_cuda(cudaEventRecord(d2h_end.get(), stream.get()), "d2h_end");

    check_cuda(cudaStreamSynchronize(stream.get()), "streamSync");

    if (timings != nullptr) {
        timings->setup_ms = event_elapsed_ms(setup_end, h2d_end);
        timings->host_to_device_ms = event_elapsed_ms(setup_end, h2d_end);
        timings->kernel_ms = event_elapsed_ms(h2d_end, kernel_end);
        timings->device_to_host_ms = event_elapsed_ms(kernel_end, d2h_end);
    }

    cudaFree(d_packed);
    cudaFree(d_weights);
    cudaFree(d_intensity);
}

Intensities cuda_beamform_v3_packed_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    CudaBeamformerTimingsV3* timings,
    const V3BeamformerExecutionConfig& config) {
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    cuda_beamform_v3_packed_intensity_into(packed, weights, dims, intensity, timings, config);
    return intensity;
}

void cuda_beamform_v3_packed_integrated_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    IntegratedIntensities& intensity,
    CudaBeamformerTimingsV3* timings,
    const V3BeamformerExecutionConfig& config) {

    validate_dimensions(dims);
    validate_temporal_config(temporal_integration);

    const std::size_t packed_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t weights_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
    const std::size_t output_count = integrated_intensity_count(dims, temporal_integration);
    const std::size_t output_bytes = output_count * sizeof(float);

    CudaStream stream;
    CudaEvent setup_end, h2d_end, kernel_end, d2h_end;

    std::uint8_t* d_packed = nullptr;
    ComplexFloat* d_weights = nullptr;
    float* d_integrated = nullptr;

    check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc d_packed");
    check_cuda(cudaMalloc(&d_weights, weights_bytes), "cudaMalloc d_weights");
    check_cuda(cudaMalloc(&d_integrated, output_bytes), "cudaMalloc d_integrated");

    check_cuda(cudaEventRecord(setup_end.get(), stream.get()), "setup_end");

    check_cuda(cudaMemcpyAsync(d_packed, packed.data(), packed_bytes, cudaMemcpyHostToDevice, stream.get()), "H2D packed");
    check_cuda(cudaMemcpyAsync(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice, stream.get()), "H2D weights");
    check_cuda(cudaEventRecord(h2d_end.get(), stream.get()), "h2d_end");

    cuda_beamform_v3_integrated_device_resident(
        d_packed, d_weights, d_integrated, dims, temporal_integration, config, stream.get());
    check_cuda(cudaEventRecord(kernel_end.get(), stream.get()), "kernel_end");

    check_cuda(cudaMemcpyAsync(intensity.data(), d_integrated, output_bytes, cudaMemcpyDeviceToHost, stream.get()), "D2H integrated");
    check_cuda(cudaEventRecord(d2h_end.get(), stream.get()), "d2h_end");

    check_cuda(cudaStreamSynchronize(stream.get()), "streamSync");

    if (timings != nullptr) {
        timings->setup_ms = event_elapsed_ms(setup_end, h2d_end);
        timings->host_to_device_ms = event_elapsed_ms(setup_end, h2d_end);
        timings->temporal_integration_ms = event_elapsed_ms(h2d_end, kernel_end);
        timings->kernel_ms = timings->temporal_integration_ms;
        timings->device_to_host_ms = event_elapsed_ms(kernel_end, d2h_end);
    }

    cudaFree(d_packed);
    cudaFree(d_weights);
    cudaFree(d_integrated);
}

IntegratedIntensities cuda_beamform_v3_packed_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV3* timings,
    const V3BeamformerExecutionConfig& config) {
    const std::size_t output_count = integrated_intensity_count(dims, temporal_integration);
    IntegratedIntensities intensity(output_count, 0.0F);
    cuda_beamform_v3_packed_integrated_intensity_into(
        packed, weights, dims, temporal_integration, intensity, timings, config);
    return intensity;
}

QuantizedIntegratedOutput cuda_beamform_v3_packed_quantized_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV3* timings,
    const V3BeamformerExecutionConfig& config) {

    CudaBeamformerWorkspaceV3 workspace(
        dims, config, temporal_integration, CudaBeamformerOutputV3::QuantizedInt8);
    const Dimensions integrated_dims{
        integrated_time_count(dims.n_time, temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    QuantizedIntegratedOutput output{
        std::vector<std::int8_t>(integrated_intensity_count(dims, temporal_integration)),
        std::vector<Int8QuantizationParameters>(quantization_parameter_count(integrated_dims))};

    workspace.upload_weights(weights, dims);
    const auto pipeline_timings = workspace.run_quantized_integrated_pipeline(packed, output, dims);
    if (timings != nullptr) {
        *timings = pipeline_timings;
    }
    return output;
}

// ---------------------------------------------------------------------------
// Reusable Workspace Implementation (V3)
// ---------------------------------------------------------------------------
struct CudaBeamformerWorkspaceV3::Impl {
    Dimensions capacity;
    V3BeamformerExecutionConfig config;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutputV3 output;
    double setup_time_ms = 0.0;

    CudaStream stream;
    CudaEvent setup_end, upload_start, upload_end, kernel_start, kernel_end,
              quantization_end, result_end;

    std::unique_ptr<DeviceBuffer<std::uint8_t>> device_packed;
    std::unique_ptr<DeviceBuffer<ComplexFloat>> device_weights;
    std::unique_ptr<DeviceBuffer<float>> device_intensity;
    std::unique_ptr<DeviceBuffer<float>> device_integrated;
    std::unique_ptr<DeviceBuffer<std::int8_t>> device_quantized_intensity;
    std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>> device_quantization_parameters;

    Impl(const Dimensions& cap, const V3BeamformerExecutionConfig& cfg,
         std::optional<TemporalIntegrationConfig> t_int, CudaBeamformerOutputV3 out)
        : capacity(cap), config(cfg), temporal_integration(t_int), output(out) {

        const auto t0 = Clock::now();
        validate_dimensions(capacity);

        const std::size_t packed_bytes = voltage_sample_count(capacity);
        const std::size_t weights_count = capacity.n_beams * capacity.n_freq * capacity.n_ant;
        const std::size_t direct_intensity_count = capacity.n_time * capacity.n_freq * capacity.n_beams;

        device_packed = std::make_unique<DeviceBuffer<std::uint8_t>>(packed_bytes);
        device_weights = std::make_unique<DeviceBuffer<ComplexFloat>>(weights_count);

        if (temporal_integration.has_value()) {
            validate_temporal_config(*temporal_integration);
            const std::size_t int_count = integrated_intensity_count(capacity, *temporal_integration);
            device_integrated = std::make_unique<DeviceBuffer<float>>(int_count);

            if (output == CudaBeamformerOutputV3::QuantizedInt8) {
                const Dimensions int_dims{
                    integrated_time_count(capacity.n_time, *temporal_integration),
                    capacity.n_freq, capacity.n_ant, capacity.n_beams};
                device_quantized_intensity = std::make_unique<DeviceBuffer<std::int8_t>>(int_count);
                device_quantization_parameters = std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
                    quantization_parameter_count(int_dims));
            }
        } else {
            device_intensity = std::make_unique<DeviceBuffer<float>>(direct_intensity_count);
        }

        const auto t1 = Clock::now();
        setup_time_ms = elapsed_ms(t0, t1);
    }
};

CudaBeamformerWorkspaceV3::CudaBeamformerWorkspaceV3(
    const Dimensions& capacity, const V3BeamformerExecutionConfig& config,
    std::optional<TemporalIntegrationConfig> temporal_integration,
    CudaBeamformerOutputV3 output)
    : impl_(std::make_unique<Impl>(capacity, config, temporal_integration, output)) {}

CudaBeamformerWorkspaceV3::~CudaBeamformerWorkspaceV3() = default;

double CudaBeamformerWorkspaceV3::setup_ms() const { return impl_->setup_time_ms; }
const V3BeamformerExecutionConfig& CudaBeamformerWorkspaceV3::config() const { return impl_->config; }
bool CudaBeamformerWorkspaceV3::has_temporal_integration() const { return impl_->temporal_integration.has_value(); }
bool CudaBeamformerWorkspaceV3::has_int8_quantization() const { return impl_->output == CudaBeamformerOutputV3::QuantizedInt8; }
std::size_t CudaBeamformerWorkspaceV3::packed_voltage_capacity_bytes() const { return voltage_sample_count(impl_->capacity); }
std::uint8_t* CudaBeamformerWorkspaceV3::device_packed_voltage() { return impl_->device_packed->get(); }
ComplexFloat* CudaBeamformerWorkspaceV3::device_weights() { return impl_->device_weights->get(); }
float* CudaBeamformerWorkspaceV3::device_intensity() { return impl_->device_intensity ? impl_->device_intensity->get() : nullptr; }
float* CudaBeamformerWorkspaceV3::device_integrated_intensity() { return impl_->device_integrated ? impl_->device_integrated->get() : nullptr; }
void* CudaBeamformerWorkspaceV3::device_stream() { return impl_->stream.get(); }

double CudaBeamformerWorkspaceV3::upload_packed_voltage(const PackedVoltage& packed, const Dimensions& dims) {
    const std::size_t bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    check_cuda(cudaEventRecord(impl_->upload_start.get(), impl_->stream.get()), "upload_start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed->get(), packed.data(), bytes,
                               cudaMemcpyHostToDevice, impl_->stream.get()), "H2D packed");
    check_cuda(cudaEventRecord(impl_->upload_end.get(), impl_->stream.get()), "upload_end");
    check_cuda(cudaEventSynchronize(impl_->upload_end.get()), "upload_sync");
    return event_elapsed_ms(impl_->upload_start, impl_->upload_end);
}

double CudaBeamformerWorkspaceV3::upload_weights(const Weights& weights, const Dimensions& dims) {
    const std::size_t bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
    check_cuda(cudaEventRecord(impl_->upload_start.get(), impl_->stream.get()), "upload_start");
    check_cuda(cudaMemcpyAsync(impl_->device_weights->get(), weights.data(), bytes,
                               cudaMemcpyHostToDevice, impl_->stream.get()), "H2D weights");
    check_cuda(cudaEventRecord(impl_->upload_end.get(), impl_->stream.get()), "upload_end");
    check_cuda(cudaEventSynchronize(impl_->upload_end.get()), "upload_sync");
    return event_elapsed_ms(impl_->upload_start, impl_->upload_end);
}

double CudaBeamformerWorkspaceV3::run_kernel(const Dimensions& dims) {
    check_cuda(cudaEventRecord(impl_->kernel_start.get(), impl_->stream.get()), "kernel_start");
    if (impl_->temporal_integration.has_value()) {
        cuda_beamform_v3_integrated_device_resident(
            impl_->device_packed->get(), impl_->device_weights->get(),
            impl_->device_integrated->get(), dims, *impl_->temporal_integration,
            impl_->config, impl_->stream.get());
    } else {
        cuda_beamform_v3_device_resident(
            impl_->device_packed->get(), impl_->device_weights->get(),
            impl_->device_intensity->get(), dims, impl_->config, impl_->stream.get());
    }
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()), "kernel_end");
    check_cuda(cudaEventSynchronize(impl_->kernel_end.get()), "kernel_sync");
    return event_elapsed_ms(impl_->kernel_start, impl_->kernel_end);
}

double CudaBeamformerWorkspaceV3::download_intensity(Intensities& intensity, const Dimensions& dims) {
    const std::size_t bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()), "d2h_start");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity->get(), bytes,
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "D2H intensity");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()), "result_end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()), "d2h_sync");
    return event_elapsed_ms(impl_->kernel_end, impl_->result_end);
}

CudaBeamformerTimingsV3 CudaBeamformerWorkspaceV3::run_pipeline(
    const PackedVoltage& packed, Intensities& intensity, const Dimensions& dims) {

    const std::size_t p_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t i_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);

    check_cuda(cudaEventRecord(impl_->upload_start.get(), impl_->stream.get()), "upload_start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed->get(), packed.data(), p_bytes,
                               cudaMemcpyHostToDevice, impl_->stream.get()), "H2D");
    check_cuda(cudaEventRecord(impl_->upload_end.get(), impl_->stream.get()), "upload_end");

    cuda_beamform_v3_device_resident(
        impl_->device_packed->get(), impl_->device_weights->get(),
        impl_->device_intensity->get(), dims, impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()), "kernel_end");

    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity->get(), i_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "D2H");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()), "result_end");
    check_cuda(cudaStreamSynchronize(impl_->stream.get()), "streamSync");

    CudaBeamformerTimingsV3 timings;
    timings.setup_ms = impl_->setup_time_ms;
    timings.host_to_device_ms = event_elapsed_ms(impl_->upload_start, impl_->upload_end);
    timings.kernel_ms = event_elapsed_ms(impl_->upload_end, impl_->kernel_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV3 CudaBeamformerWorkspaceV3::run_integrated_pipeline(
    const PackedVoltage& packed, IntegratedIntensities& intensity, const Dimensions& dims) {

    if (!impl_->temporal_integration.has_value()) {
        throw std::logic_error("Workspace not configured with temporal integration");
    }

    const std::size_t p_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const std::size_t out_count = integrated_intensity_count(dims, *impl_->temporal_integration);
    const std::size_t out_bytes = out_count * sizeof(float);

    check_cuda(cudaEventRecord(impl_->upload_start.get(), impl_->stream.get()), "upload_start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed->get(), packed.data(), p_bytes,
                               cudaMemcpyHostToDevice, impl_->stream.get()), "H2D");
    check_cuda(cudaEventRecord(impl_->upload_end.get(), impl_->stream.get()), "upload_end");

    cuda_beamform_v3_integrated_device_resident(
        impl_->device_packed->get(), impl_->device_weights->get(),
        impl_->device_integrated->get(), dims, *impl_->temporal_integration,
        impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()), "kernel_end");

    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_integrated->get(), out_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "D2H");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()), "result_end");
    check_cuda(cudaStreamSynchronize(impl_->stream.get()), "streamSync");

    CudaBeamformerTimingsV3 timings;
    timings.setup_ms = impl_->setup_time_ms;
    timings.host_to_device_ms = event_elapsed_ms(impl_->upload_start, impl_->upload_end);
    timings.temporal_integration_ms = event_elapsed_ms(impl_->upload_end, impl_->kernel_end);
    timings.kernel_ms = timings.temporal_integration_ms;
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV3 CudaBeamformerWorkspaceV3::run_quantized_integrated_pipeline(
    const PackedVoltage& packed, QuantizedIntegratedOutput& intensity, const Dimensions& dims) {

    if (!impl_->temporal_integration.has_value() || impl_->output != CudaBeamformerOutputV3::QuantizedInt8) {
        throw std::logic_error("Workspace not configured with int8 quantization");
    }

    const std::size_t p_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
    const Dimensions int_dims{
        integrated_time_count(dims.n_time, *impl_->temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    const std::size_t int_count = integrated_intensity_count(dims, *impl_->temporal_integration);
    const std::size_t param_count = quantization_parameter_count(int_dims);

    check_cuda(cudaEventRecord(impl_->upload_start.get(), impl_->stream.get()), "upload_start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed->get(), packed.data(), p_bytes,
                               cudaMemcpyHostToDevice, impl_->stream.get()), "H2D");
    check_cuda(cudaEventRecord(impl_->upload_end.get(), impl_->stream.get()), "upload_end");

    cuda_beamform_v3_integrated_device_resident(
        impl_->device_packed->get(), impl_->device_weights->get(),
        impl_->device_integrated->get(), dims, *impl_->temporal_integration,
        impl_->config, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()), "kernel_end");

    launch_quantize_integrated_intensity(
        impl_->stream.get(),
        impl_->device_integrated->get(),
        impl_->device_quantized_intensity->get(),
        impl_->device_quantization_parameters->get(),
        int_dims);
    check_cuda(cudaEventRecord(impl_->quantization_end.get(), impl_->stream.get()), "quant_end");

    check_cuda(cudaMemcpyAsync(intensity.codes.data(), impl_->device_quantized_intensity->get(),
                               int_count * sizeof(std::int8_t),
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "D2H codes");
    check_cuda(cudaMemcpyAsync(intensity.parameters.data(), impl_->device_quantization_parameters->get(),
                               param_count * sizeof(Int8QuantizationParameters),
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "D2H params");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()), "result_end");
    check_cuda(cudaStreamSynchronize(impl_->stream.get()), "streamSync");

    CudaBeamformerTimingsV3 timings;
    timings.setup_ms = impl_->setup_time_ms;
    timings.host_to_device_ms = event_elapsed_ms(impl_->upload_start, impl_->upload_end);
    timings.temporal_integration_ms = event_elapsed_ms(impl_->upload_end, impl_->kernel_end);
    timings.kernel_ms = timings.temporal_integration_ms;
    timings.quantization_ms = event_elapsed_ms(impl_->kernel_end, impl_->quantization_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->quantization_end, impl_->result_end);
    return timings;
}

CudaBeamformerTimingsV3 CudaBeamformerWorkspaceV3::run_device_frame(
    const CudaFrameView& voltage, const CudaFrameView& weights,
    CudaFrameView& output, CudaFrameView* quantization_parameters) {

    validate_cuda_frame_view(voltage);
    validate_cuda_frame_view(weights);
    validate_cuda_frame_view(output);

    const Dimensions dims{
        voltage.buffer.shape[0], voltage.buffer.shape[1], voltage.buffer.shape[2],
        output.buffer.shape[2]};
    const auto stream = impl_->stream.get();

    CudaEvent k_start, k_end, q_end;
    check_cuda(cudaEventRecord(k_start.get(), stream), "k_start");

    if (impl_->temporal_integration.has_value()) {
        const Dimensions int_dims{
            integrated_time_count(dims.n_time, *impl_->temporal_integration),
            dims.n_freq, dims.n_ant, dims.n_beams};

        if (impl_->output == CudaBeamformerOutputV3::QuantizedInt8) {
            if (quantization_parameters == nullptr) {
                throw std::invalid_argument("Quantization parameters required for int8 output");
            }
            validate_cuda_frame_view(*quantization_parameters);

            cuda_beamform_v3_integrated_device_resident(
                static_cast<const std::uint8_t*>(voltage.buffer.device_data),
                static_cast<const ComplexFloat*>(weights.buffer.device_data),
                impl_->device_integrated->get(),
                dims, *impl_->temporal_integration, impl_->config, stream);

            check_cuda(cudaEventRecord(k_end.get(), stream), "k_end");

            launch_quantize_integrated_intensity(
                stream,
                impl_->device_integrated->get(),
                static_cast<std::int8_t*>(output.buffer.device_data),
                static_cast<Int8QuantizationParameters*>(quantization_parameters->buffer.device_data),
                int_dims);

            check_cuda(cudaEventRecord(q_end.get(), stream), "q_end");
        } else {
            cuda_beamform_v3_integrated_device_resident(
                static_cast<const std::uint8_t*>(voltage.buffer.device_data),
                static_cast<const ComplexFloat*>(weights.buffer.device_data),
                static_cast<float*>(output.buffer.device_data),
                dims, *impl_->temporal_integration, impl_->config, stream);
            check_cuda(cudaEventRecord(k_end.get(), stream), "k_end");
        }
    } else {
        cuda_beamform_v3_device_resident(
            static_cast<const std::uint8_t*>(voltage.buffer.device_data),
            static_cast<const ComplexFloat*>(weights.buffer.device_data),
            static_cast<float*>(output.buffer.device_data),
            dims, impl_->config, stream);
        check_cuda(cudaEventRecord(k_end.get(), stream), "k_end");
    }

    check_cuda(cudaStreamSynchronize(stream), "streamSync");

    CudaBeamformerTimingsV3 timings;
    timings.setup_ms = 0.0;
    timings.kernel_ms = event_elapsed_ms(k_start, k_end);
    if (impl_->output == CudaBeamformerOutputV3::QuantizedInt8) {
        timings.quantization_ms = event_elapsed_ms(k_end, q_end);
    }
    return timings;
}

// ---------------------------------------------------------------------------
// Batched Persistent Pipeline & CUDA Graph Stream (V3)
// ---------------------------------------------------------------------------
struct BatchedBeamformerStreamV3::Impl {
    Dimensions dims;
    Weights weights;
    V3BeamformerExecutionConfig config;

    cudaStream_t stream = nullptr;
    std::uint8_t* d_packed = nullptr;
    ComplexFloat* d_weights = nullptr;
    float* d_intensity = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float last_time_ms = 0.0F;

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;

    Impl(const Dimensions& d, const Weights& w, const V3BeamformerExecutionConfig& cfg)
        : dims(d), weights(w), config(cfg) {

        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");

        const std::size_t p_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
        const std::size_t w_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
        const std::size_t i_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);

        check_cuda(cudaMalloc(&d_packed, p_bytes), "cudaMalloc d_packed");
        check_cuda(cudaMalloc(&d_weights, w_bytes), "cudaMalloc d_weights");
        check_cuda(cudaMalloc(&d_intensity, i_bytes), "cudaMalloc d_intensity");

        check_cuda(cudaMemcpyAsync(d_weights, weights.data(), w_bytes,
                                   cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync weights");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    ~Impl() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (d_packed) cudaFree(d_packed);
        if (d_weights) cudaFree(d_weights);
        if (d_intensity) cudaFree(d_intensity);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
    }
};

BatchedBeamformerStreamV3::BatchedBeamformerStreamV3(
    const Dimensions& dims, const Weights& weights, const V3BeamformerExecutionConfig& config)
    : impl_(std::make_unique<Impl>(dims, weights, config)) {}

BatchedBeamformerStreamV3::~BatchedBeamformerStreamV3() = default;

void BatchedBeamformerStreamV3::process_batch(
    const std::uint8_t* host_packed, float* host_intensity) {

    const std::size_t p_bytes = voltage_sample_count(impl_->dims) * sizeof(std::uint8_t);
    const std::size_t i_bytes = impl_->dims.n_time * impl_->dims.n_freq * impl_->dims.n_beams * sizeof(float);

    check_cuda(cudaMemcpyAsync(impl_->d_packed, host_packed, p_bytes,
                               cudaMemcpyHostToDevice, impl_->stream), "H2D");
    process_batch_kernel_only();
    check_cuda(cudaMemcpyAsync(host_intensity, impl_->d_intensity, i_bytes,
                               cudaMemcpyDeviceToHost, impl_->stream), "D2H");
    check_cuda(cudaStreamSynchronize(impl_->stream), "streamSync");
}

void BatchedBeamformerStreamV3::process_batch_kernel_only() {
    if (impl_->config.enable_cuda_graph) {
        if (!impl_->graph_captured) {
            check_cuda(cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
            cuda_beamform_v3_device_resident(
                impl_->d_packed, impl_->d_weights, impl_->d_intensity, impl_->dims, impl_->config, impl_->stream);
            check_cuda(cudaStreamEndCapture(impl_->stream, &impl_->graph), "cudaStreamEndCapture");
            check_cuda(cudaGraphInstantiate(&impl_->graph_exec, impl_->graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
            impl_->graph_captured = true;
        }

        check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "cudaEventRecord");
        check_cuda(cudaGraphLaunch(impl_->graph_exec, impl_->stream), "cudaGraphLaunch");
        check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "cudaEventRecord");
        check_cuda(cudaEventSynchronize(impl_->stop_event), "cudaEventSynchronize");
        check_cuda(cudaEventElapsedTime(&impl_->last_time_ms, impl_->start_event, impl_->stop_event), "elapsed");
    } else {
        check_cuda(cudaEventRecord(impl_->start_event, impl_->stream), "cudaEventRecord");
        cuda_beamform_v3_device_resident(
            impl_->d_packed, impl_->d_weights, impl_->d_intensity, impl_->dims, impl_->config, impl_->stream);
        check_cuda(cudaEventRecord(impl_->stop_event, impl_->stream), "cudaEventRecord");
        check_cuda(cudaEventSynchronize(impl_->stop_event), "cudaEventSynchronize");
        check_cuda(cudaEventElapsedTime(&impl_->last_time_ms, impl_->start_event, impl_->stop_event), "elapsed");
    }
}

float BatchedBeamformerStreamV3::last_kernel_time_ms() const { return impl_->last_time_ms; }
std::uint8_t* BatchedBeamformerStreamV3::device_packed_buffer() { return impl_->d_packed; }
float* BatchedBeamformerStreamV3::device_intensity_buffer() { return impl_->d_intensity; }
void* BatchedBeamformerStreamV3::device_stream() { return impl_->stream; }

} // namespace beamformer
