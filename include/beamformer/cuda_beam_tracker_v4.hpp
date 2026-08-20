#pragma once

// CUDA Beam Tracker V4 — Tensor-Core, Mixed-Precision & Asynchronous Memory Architecture.
//
// V4 establishes the next-generation beam tracking engine for exascale radio astronomy,
// synthesizing:
//
// 1. Warp Matrix Multiply-Accumulate (WMMA / Tensor Core Acceleration):
//    Warp-synchronous hardware GEMM matrix math mapping complex beamforming
//    (Antennas x Time) x (Beams x Antennas) into 4 real-valued block MMA operations:
//      Z_real = (X_real * W_real) - (X_imag * W_imag)
//      Z_imag = (X_real * W_imag) + (X_imag * W_real)
//    using 16x16x16 / 32x8x16 FP16 input fragments and FP32 accumulators on Volta,
//    Turing, Ampere, Ada Lovelace, Hopper, and Blackwell architectures.
//
// 2. Mixed-Precision Half2 Vector Arithmetic (__half2):
//    Dual FP16 vector compute instructions (__hfma2, __hadd2, __hmul2) doubling
//    arithmetic throughput and halving memory bandwidth on Pascal, Turing, and Ampere.
//
// 3. Asynchronous Double-Buffered Shared-Memory Staging (`cp.async`):
//    Global-to-Shared memory direct transfers bypassing registers on Ampere+ (SM80+),
//    with double-buffered warp prefetch pipeline on earlier architectures (SM60-SM75).
//
// 4. Deep Instruction-Level Parallelism (ILP 8-way / 4-way):
//    Register-reused steering weights across up to 8 consecutive time samples per
//    inner iteration with interleaved warp shuffle reduction trees.
//
// 5. Persistent Grid & CUDA Graph Stream Orchestration:
//    BatchedTrackerStreamV4 with persistent device resident staging and zero-dispatch
//    overhead via CUDA execution graph capture.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace beamformer {

// V4 Kernel Architecture Mode Selection
enum class V4KernelMode {
    Auto = 0,               // Auto-detect optimal kernel for host GPU architecture
    WmmaTensorCore = 1,     // Warp Matrix Multiply-Accumulate (Requires SM_70+)
    Half2VectorSimd = 2,    // __half2 packed vector arithmetic (SM_60+)
    DeepIlpWarpShuffle = 3, // 4-way / 8-way deep unrolled fused warp-shuffle with PTX bfe
    BlockReduction = 4,     // Block-level cooperative reduction for large antenna arrays (128, 256+)
};

// Execution configuration for V4 kernel tuning
struct V4ExecutionConfig {
    V4KernelMode mode = V4KernelMode::Auto;
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 4; // 2, 4, or 8
    bool enable_cuda_graph = false;
    bool enable_async_copy = true; // cp.async on SM80+
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocator Utilities
// ---------------------------------------------------------------------------
struct PinnedDeleterV4 {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVectorV4 = std::unique_ptr<T[], PinnedDeleterV4>;

PinnedVectorV4<std::uint8_t> allocate_pinned_voltage_v4(const Dimensions& dims);
PinnedVectorV4<float> allocate_pinned_intensities_v4(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Core Functional API
// ---------------------------------------------------------------------------

// Allocate-and-return variant: runs the V4 kernel and returns the float32 intensity cube.
Intensities cuda_beam_tracker_v4(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const V4ExecutionConfig& config = V4ExecutionConfig{});

// Into-variant: writes directly into caller-allocated intensity buffer.
void cuda_beam_tracker_v4_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const V4ExecutionConfig& config = V4ExecutionConfig{});

// Streaming variant: multi-stream double/triple-buffered pipelined execution.
void cuda_beam_tracker_v4_stream(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    std::size_t n_streams = 3,
    const V4ExecutionConfig& config = V4ExecutionConfig{});

// Device-Resident variant: executes directly on GPU device memory pointers (d_packed -> d_intensity).
void cuda_beam_tracker_v4_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const V4ExecutionConfig& config = V4ExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Batched Tracker Stream V4 (Continuous Streaming / Persistent Pipeline)
// ---------------------------------------------------------------------------
class BatchedTrackerStreamV4 {
public:
    BatchedTrackerStreamV4(const Dimensions& single_window_dims,
                           const TrackerConfig& tracker,
                           std::size_t batch_size,
                           const V4ExecutionConfig& config = V4ExecutionConfig{});
    ~BatchedTrackerStreamV4();

    BatchedTrackerStreamV4(const BatchedTrackerStreamV4&) = delete;
    BatchedTrackerStreamV4& operator=(const BatchedTrackerStreamV4&) = delete;

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

} // namespace beamformer
