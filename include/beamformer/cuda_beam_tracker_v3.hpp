#pragma once

// CUDA Beam Tracker V3 — High-Performance Fused ILP Warp-Shuffle Kernel.
//
// V3 extends and strictly improves upon Phase 4 (cuda_beam_tracker_fused_warp_shuffle)
// with the following architectural enhancements:
//
// 1. PTX Bit-Field Extraction (`bfe.s32`):
//    Single-cycle hardware sign-extended nibble decode into 32-bit registers,
//    eliminating branch penalties and multiple ALU/bitwise operations.
//
// 2. Dual and Quad Time-Sample Unrolling with Instruction-Level Parallelism (ILP):
//    Reuses register-cached steering weights across multiple consecutive time
//    samples (T_UNROLL = 2 or 4), amortizing steering weight storage, coalescing
//    global memory load transactions, and interleaving independent warp-shuffle
//    reduction instructions to eliminate warp scheduler pipeline stalls.
//
// 3. Compile-Time Antenna Specialization:
//    Templated over antenna count (N_ANT = 32, 64) with compile-time geometry
//    arithmetic (replacing dynamic integer division/modulo with single-cycle bitwise
//    shifts and masks) and eliminating dead lane branches.
//
// 4. Fine-Grained Time-Chunk Grid Tiling (T_CHUNK):
//    Allows single-window streaming executions (where window_count = 1 would otherwise
//    under-utilize massive GPUs like RTX 5090 / H100 / Trillium) to scale up active
//    blocks across SMs by partitioning the time dimension into sub-chunks.
//
// 5. Zero-Copy Device-Resident Execution:
//    Direct kernel launch on pre-staged GPU memory (e.g. from GPUDirect RDMA or
//    upstream pipeline stages) bypassing Host-to-Device transfer overhead.
//
// 6. CUDA Graph Acceleration:
//    BatchedTrackerStreamV3 supports capturing and instantiating CUDA execution graphs,
//    reducing host CPU dispatch overhead to < 3 microseconds per batch.
//
// 7. Pinned Host Memory Helpers:
//    Utilities to allocate and manage page-locked host memory buffers for peak PCIe
//    DMA transfer throughput.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, Vec3 via geometry
#include "beamformer/config.hpp"        // Dimensions, validate_dimensions
#include "beamformer/formats.hpp"        // PackedVoltage, Intensities

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace beamformer {

// Execution configuration options for V3 kernel tuning.
struct V3ExecutionConfig {
    // Number of time samples per warp sub-chunk (default: 80 for high occupancy,
    // 320 for single-warp-per-window mode). Must be divisible by time_unroll.
    std::size_t time_chunk_size = 80;

    // Number of time samples processed per inner-loop iteration (ILP unroll factor: 2 or 4).
    std::size_t time_unroll = 2;

    // Enable CUDA Graph capture in batched streaming mode.
    bool enable_cuda_graph = false;
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocators
// ---------------------------------------------------------------------------

struct PinnedDeleter {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVector = std::unique_ptr<T[], PinnedDeleter>;

// Allocate page-locked (pinned) host memory for packed int4 voltage.
PinnedVector<std::uint8_t> allocate_pinned_voltage(const Dimensions& dims);

// Allocate page-locked (pinned) host memory for float32 beamformed intensities.
PinnedVector<float> allocate_pinned_intensities(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Core Functional API
// ---------------------------------------------------------------------------

// Allocate-and-return variant: runs the V3 kernel on the GPU and returns
// the [time][freq][n_beams==1] float32 cube.
Intensities cuda_beam_tracker_v3(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const V3ExecutionConfig& config = V3ExecutionConfig{});

// Into-variant: writes directly into caller-owned intensity buffer.
void cuda_beam_tracker_v3_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const V3ExecutionConfig& config = V3ExecutionConfig{});

// Streaming variant: double/triple-buffered pipelined execution overlapping
// H2D copies, kernel execution, and D2H copies across `n_streams` CUDA streams.
void cuda_beam_tracker_v3_stream(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    std::size_t n_streams = 3,
    const V3ExecutionConfig& config = V3ExecutionConfig{});

// Device-Resident variant: executes V3 directly on device-allocated pointers
// (d_packed -> d_intensity). Zero host transfers.
void cuda_beam_tracker_v3_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const V3ExecutionConfig& config = V3ExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Batched Tracker Stream V3 (Continuous Streaming / Soak Pipeline)
// ---------------------------------------------------------------------------
class BatchedTrackerStreamV3 {
public:
    BatchedTrackerStreamV3(const Dimensions& single_window_dims,
                           const TrackerConfig& tracker,
                           std::size_t batch_size,
                           const V3ExecutionConfig& config = V3ExecutionConfig{});
    ~BatchedTrackerStreamV3();

    BatchedTrackerStreamV3(const BatchedTrackerStreamV3&) = delete;
    BatchedTrackerStreamV3& operator=(const BatchedTrackerStreamV3&) = delete;

    // Process a full batch of windows from host memory to host memory.
    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed,
                       float* host_intensity);

    // Executes ONLY the GPU kernel computation for the batch on persistent device
    // buffers without Host->Device or Device->Host PCIe data transfers.
    void process_batch_kernel_only(std::size_t first_window_index);

    // Returns the GPU device kernel execution time in milliseconds for the most
    // recent batch (measured via CUDA events).
    float last_kernel_time_ms() const;

    // Direct access to internal device buffers for zero-copy device-resident pipelines.
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
