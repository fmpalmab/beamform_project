// v5_unified_tracker_kernel_sketch.cu
//
// PROPOSED V5 DESIGN SKETCH -- single unified beam-tracker engine.
//
// This is an illustrative sketch to integrate into your existing
// beamformer/*.hpp headers, not a drop-in replacement -- it doesn't
// have access to your Dimensions / TrackerConfig / V4ExecutionConfig
// types as written, and the remainder (tail) loop is omitted for
// brevity. It shows the STRUCTURAL change:
//
//   ONE kernel template   (was 3: deep_ilp, half2_vector, block_reduction)
//   ONE reduction strategy (warp-shuffle only, at every antenna count)
//   ONE dispatch function  (was 6 engines: device-resident, batched
//                           kernel-only, multi-stream, batched-stream,
//                           batched-graph, block-reduction)
//
// WHY: tracker_v4_block_reduction_kernel (used for N_ANT = 128/256)
// needed inter-warp communication through shared memory: each warp
// reduces via shuffle, writes to smem, __syncthreads(), warp 0
// reduces again, __syncthreads() again -- TWO block-wide barriers
// per unrolled time-group, every group. That tax is almost
// certainly why GPU speedup vs CPU-naive fell from ~24x at 64
// antennas to ~11.8x at 256 antennas in your V4 numbers, even
// though 256 antennas is strictly more useful work per output
// sample. Below, every antenna count (32/64/128/256) uses the same
// register-accumulate + single-warp-shuffle-reduction pattern that
// was already your fastest design for N_ANT <= 64 -- it's just
// generalized so each lane owns N_ANT/32 antennas instead of 1 or 2.
//
// Also folded in:
//   - sincos() once instead of separate cos()+sin() calls in the
//     per-lane weight setup (halves transcendental work, zero
//     accuracy change -- still double precision).
//   - No __launch_bounds__ mismatch: every instantiation now shares
//     one occupancy target instead of the block-reduction kernel
//     having none.
//
// Deliberately NOT pursuing WMMA / tensor cores here: this tracker
// is single-beam (M=1 in GEMM terms), and tensor core tiles need
// M >= 8-16 to get real utilization -- that's why CHIME's real-time
// tracking beamformer forms several beams at once and DSA110's
// beamformer batches 256 beams through cuBLAS before it's worth it.
// The one tensor-core path that *would* fit your already-int4-packed
// data (wmma::experimental::precision::s4) is software-emulated with
// heavy lane divergence on current hardware, and on Hopper it's
// compiled down to ordinary IMAD on CUDA cores anyway -- no
// dedicated tensor-core path forward on the newer architectures your
// header already targets. Revisit tensor cores only if the tracker
// grows to track several directions simultaneously.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace beamformer {
namespace v5 {

__device__ __forceinline__ float2 unpack_int4_fast(std::uint32_t byte_val) {
    int r, i;
    asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
    asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
    return make_float2(__int2float_rn(r), __int2float_rn(i));
}

template <int N_ANT>
__device__ __forceinline__ float3 tracker_position_v5(unsigned int element, float spacing_m) {
    if constexpr (N_ANT == 32 || N_ANT == 64) {
        const unsigned int col = element & 7U, row = element >> 3U;
        return make_float3(col * spacing_m, row * spacing_m, 0.0F);
    } else { // 128, 256
        const unsigned int col = element & 15U, row = element >> 4U;
        return make_float3(col * spacing_m, row * spacing_m, 0.0F);
    }
}

// One sincos() call instead of V4's separate cos()+sin() calls --
// halves transcendental work per lane, same double precision.
__device__ __forceinline__ void tracker_weight_v5(float3 position, float3 direction,
                                                    double wave_number,
                                                    float* w_r, float* w_i) {
    const double delay_m = static_cast<double>(position.x) * direction.x
                          + static_cast<double>(position.y) * direction.y
                          + static_cast<double>(position.z) * direction.z;
    const double phase = wave_number * delay_m;
    double s, c;
    sincos(phase, &s, &c);
    *w_r = static_cast<float>(c);
    *w_i = static_cast<float>(s);
}

// ---------------------------------------------------------------
// THE single tracker kernel -- replaces tracker_v4_deep_ilp_kernel,
// tracker_v4_half2_vector_kernel, AND tracker_v4_block_reduction_kernel.
//
// Identical code path for N_ANT = 32, 64, 128, 256: each of the 32
// lanes owns ANT_PER_LANE = N_ANT/32 antennas in registers, and
// there is exactly one reduction step at the end -- a plain warp
// shuffle, no shared memory, no __syncthreads, at any antenna count.
// ---------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 8)
tracker_v5_kernel(
    float* __restrict__ intensity,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    std::size_t n_freq, std::size_t n_beams,
    std::size_t integration_spectra, std::size_t time_chunk_size,
    std::size_t chunks_per_window, float spacing_m,
    std::size_t total_warps) {

    constexpr unsigned int ANT_PER_LANE = N_ANT / 32U;
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id = static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;
    if (warp_id >= total_warps) return;

    const std::size_t freq = warp_id % n_freq;
    const std::size_t chunk_global = warp_id / n_freq;
    const std::size_t chunk_in_win = chunk_global % chunks_per_window;
    const std::size_t window = chunk_global / chunks_per_window;

    const float3 direction = make_float3(window_directions[window * 3 + 0],
                                          window_directions[window * 3 + 1],
                                          window_directions[window * 3 + 2]);
    const double wave_number = wavenumbers[freq];

    // ANT_PER_LANE registers per lane instead of V4's 1 (N_ANT<=32)
    // or 2 (N_ANT==64) -- same per-antenna trig cost, generalized.
    float w_r[ANT_PER_LANE], w_i[ANT_PER_LANE];
    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const float3 pos = tracker_position_v5<N_ANT>(lane + a * 32U, spacing_m);
        tracker_weight_v5(pos, direction, wave_number, &w_r[a], &w_i[a]);
    }

    const std::size_t t_win_start = window * integration_spectra;
    const std::size_t t_chunk_start = t_win_start + chunk_in_win * time_chunk_size;
    const std::size_t t_chunk_end = t_chunk_start + time_chunk_size; // clamp to n_time/window end upstream, as in V4

    std::size_t t = t_chunk_start;
    for (; t + (TIME_UNROLL - 1) < t_chunk_end; t += TIME_UNROLL) {
        float s_r[TIME_UNROLL] = {0.0F};
        float s_i[TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (int k = 0; k < TIME_UNROLL; ++k) {
            const std::size_t vk = ((t + k) * n_freq + freq) * N_ANT;
            #pragma unroll
            for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
                const float2 p = unpack_int4_fast(packed[vk + lane + a * 32U]);
                s_r[k] = fmaf(w_r[a], p.x, fmaf(-w_i[a], p.y, s_r[k]));
                s_i[k] = fmaf(w_r[a], p.y, fmaf( w_i[a], p.x, s_i[k]));
            }
        }

        // Single reduction step -- every antenna count, no shared
        // memory, no block-wide barrier.
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
                intensity[((t + k) * n_freq + freq) * n_beams] = s_r[k] * s_r[k] + s_i[k] * s_i[k];
            }
        }
    }
    // Remainder loop omitted for brevity -- same shape as V4's tail
    // loops, minus shared memory (single shfl reduction, ANT_PER_LANE
    // antennas per lane, same as the main loop above with TIME_UNROLL=1).
}

// ---------------------------------------------------------------
// ONE dispatcher, no engine selection: always device-resident,
// always this kernel; only N_ANT/TIME_UNROLL vary the template
// instantiation (a compile-time choice, not a runtime code-path
// fork). Drop BatchedTrackerStreamV4, CUDA-graph capture, and the
// multi-stream engine entirely -- in your own V4 numbers, none of
// them beat plain single-stream Device Resident at any antenna
// count (115ms vs 277-288ms at 256 antennas; 55ms vs 138-140ms at
// 128; roughly tied at 64). At ~4 FLOP/byte this workload is
// memory-bound, so there isn't enough compute per transfer for
// stream/graph overlap to pay for its own bookkeeping overhead.
// ---------------------------------------------------------------
template <int N_ANT, int TIME_UNROLL = 8>
void launch_v5(float* d_intensity, const float* d_window_directions,
               const double* d_wavenumbers, const std::uint8_t* d_packed,
               std::size_t n_freq, std::size_t n_beams,
               std::size_t integration_spectra, std::size_t time_chunk_size,
               std::size_t chunks_per_window, float spacing_m,
               std::size_t total_warps, cudaStream_t stream) {
    constexpr unsigned int warps_per_block = 4;
    const dim3 block_dim(32, warps_per_block);
    const unsigned int grid_dim = static_cast<unsigned int>(
        (total_warps + warps_per_block - 1) / warps_per_block);

    tracker_v5_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_intensity, d_window_directions, d_wavenumbers, d_packed,
        n_freq, n_beams, integration_spectra, time_chunk_size,
        chunks_per_window, spacing_m, total_warps);
}

} // namespace v5
} // namespace beamformer
