#pragma once

// CUDA Beam Tracker V5 (RTX 5090 / NVIDIA Blackwell Architecture Edition)
//
// Specialized high-throughput variant of the V5 Unified Beam Tracker architected
// for the NVIDIA GeForce RTX 5090 (Blackwell Architecture, GB202, SM 100/120).
//
// Architectural Highlights for Blackwell (RTX 5090):
//
// 1. Massive SM Saturation (170 Streaming Multiprocessors / 21,760 CUDA Cores):
//    Fine-grained grid partitioning and multi-warp sub-chunking engineered to
//    saturate 170 SMs simultaneously with optimal wave quantization.
//
// 2. 1.79 TB/s GDDR7 Memory Bus & 128 MB L2 Cache Exploitation:
//    L2 cache streaming directives (`ld.global.nc` / non-coherent cache bypass /
//    128-byte sector coalescing) designed to hit peak bandwidth on the 512-bit bus.
//
// 3. Deep 16-way and 8-way Instruction-Level Parallelism (ILP 16 / ILP 8):
//    Utilizes Blackwell's massive register file to keep up to 16 consecutive time
//    accumulators in registers without register spilling, completely saturating
//    the dual-issue FP32 execution pipeline.
//
// 4. Single-Cycle Hardware Bitfield Extraction (PTX `bfe.s32`):
//    Direct signed int4 nibble unpacking directly into 32-bit registers.
//
// 5. Zero Shared Memory Barriers:
//    Retains V5's unified register-accumulated warp-reduction tree (zero smem,
//    zero __syncthreads()), executing entirely within registers and warp shuffles.
//
// 6. Persistent Pipeline & CUDA Execution Graph Ready:
//    Persistent device buffers with hardware CUDA Graph capture for < 1 us dispatch.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace beamformer {
namespace rtx5090 {

// Execution configuration tuned for RTX 5090 (Blackwell)
struct RTX5090ExecutionConfig {
    // Number of time samples per warp sub-chunk (default: 40 or 80 for 170 SM saturation)
    std::size_t time_chunk_size = 40;

    // Inner loop unroll factor (default: 8 or 16 for Blackwell dual-issue FP32 pipeline)
    std::size_t time_unroll = 8; // 4, 8, or 16

    // Enable CUDA Graph capture for persistent execution
    bool enable_cuda_graph = false;
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocator Utilities
// ---------------------------------------------------------------------------
struct PinnedDeleterRTX5090 {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVectorRTX5090 = std::unique_ptr<T[], PinnedDeleterRTX5090>;

PinnedVectorRTX5090<std::uint8_t> allocate_pinned_voltage_rtx5090(const Dimensions& dims);
PinnedVectorRTX5090<float> allocate_pinned_intensities_rtx5090(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Core Functional API
// ---------------------------------------------------------------------------

// Allocate-and-return variant: runs the Blackwell-tuned V5 kernel and returns float32 intensities.
Intensities cuda_beam_tracker_v5_rtx5090(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const RTX5090ExecutionConfig& config = RTX5090ExecutionConfig{});

// Into-variant: writes directly into caller-allocated intensity buffer.
void cuda_beam_tracker_v5_rtx5090_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const RTX5090ExecutionConfig& config = RTX5090ExecutionConfig{});

// Device-Resident variant: executes directly on GPU device memory pointers (d_packed -> d_intensity).
void cuda_beam_tracker_v5_rtx5090_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const RTX5090ExecutionConfig& config = RTX5090ExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Persistent Streaming Engine (RTX 5090 Persistent Pipeline)
// ---------------------------------------------------------------------------
class BatchedTrackerStreamRTX5090 {
public:
    BatchedTrackerStreamRTX5090(const Dimensions& single_window_dims,
                                const TrackerConfig& tracker,
                                std::size_t batch_size,
                                const RTX5090ExecutionConfig& config = RTX5090ExecutionConfig{});
    ~BatchedTrackerStreamRTX5090();

    BatchedTrackerStreamRTX5090(const BatchedTrackerStreamRTX5090&) = delete;
    BatchedTrackerStreamRTX5090& operator=(const BatchedTrackerStreamRTX5090&) = delete;

    // Process a full batch of windows from host memory to host memory.
    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed,
                       float* host_intensity);

    // Executes ONLY GPU kernel computation on persistent device buffers.
    void process_batch_kernel_only(std::size_t first_window_index);

    // Returns the GPU execution time in milliseconds for the most recent batch.
    float last_kernel_time_ms() const;

    // Direct access to device buffers for zero-copy device-resident pipelines.
    std::uint8_t* device_packed_buffer();
    float* device_intensity_buffer();
    void* device_stream();

    std::size_t batch_size() const;
    std::size_t window_bytes() const;
    std::size_t batch_voltage_bytes() const;
    std::size_t batch_output_floats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rtx5090
} // namespace beamformer
