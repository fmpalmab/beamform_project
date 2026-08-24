#pragma once

// CUDA Beam Tracker V5 — Unified Single-Engine Warp-Reduction Architecture with Dynamic Multi-Beam Support.
//
// V5 establishes the peak-efficiency beam tracking engine by unifying all array
// dimensions (32, 64, 128, 256 antennas) into a single, maximally optimized kernel:
//
// 1. Unified Warp-Reduction Strategy:
//    Eliminates block-level shared memory barriers and inter-warp synchronization.
//    Every antenna count (32, 64, 128, 256) is partitioned across the 32 warp lanes
//    (ANT_PER_LANE = N_ANT / 32) in registers, accumulating across unrolled time
//    steps and reducing in a single 5-step __shfl_down_sync tree.
//
// 2. Fused Transcendental Evaluation:
//    Precomputes steering weights per lane via a single sincos() call (halving
//    transcendental ALU operations while preserving full double-precision phase).
//
// 3. Register Pre-Negation & Fused Arithmetic:
//    Pre-negates imaginary steering weights in registers to issue back-to-back
//    FFMA instructions with zero sign-flip overhead.
//
// 4. Zero 64-bit Address Overhead:
//    Lifts all multi-dimensional indexing arithmetic out of the inner loop,
//    stepping pointers by uniform constant strides.
//
// 5. Hardware-Accelerated Bitfield Extraction:
//    PTX `bfe.s32` single-cycle signed nibble decoding directly into 32-bit registers.
//
// 6. Dynamic Multi-Beam Active Counter & 24/7 Live Reconfiguration:
//    Pre-allocates output slots with dynamic active beam count (0 to MAX_TRACKER_BEAMS)
//    and atomic zero-overhead trajectory updating.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace beamformer {

constexpr std::size_t MAX_TRACKER_BEAMS = 8;

// Execution configuration for V5 kernel tuning
struct V5ExecutionConfig {
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8; // 2, 4, or 8
    bool enable_cuda_graph = false;
};

// Multi-beam configuration for dynamic active slot management
struct MultiTrackerConfig {
    std::size_t num_active_beams = 1;
    std::array<TrackerTrajectoryConfig, MAX_TRACKER_BEAMS> trajectories;
    std::size_t integration_spectra = 320;
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocator Utilities
// ---------------------------------------------------------------------------
struct PinnedDeleterV5 {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVectorV5 = std::unique_ptr<T[], PinnedDeleterV5>;

PinnedVectorV5<std::uint8_t> allocate_pinned_voltage_v5(const Dimensions& dims);
PinnedVectorV5<float> allocate_pinned_intensities_v5(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Core Functional API
// ---------------------------------------------------------------------------

// Allocate-and-return variant: runs the V5 unified kernel and returns float32 intensities.
Intensities cuda_beam_tracker_v5(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    const V5ExecutionConfig& config = V5ExecutionConfig{});

// Into-variant: writes directly into caller-allocated intensity buffer.
void cuda_beam_tracker_v5_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    const V5ExecutionConfig& config = V5ExecutionConfig{});

// Device-Resident variant: executes directly on GPU device memory pointers (d_packed -> d_intensity).
void cuda_beam_tracker_v5_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, const TrackerConfig& tracker,
    const V5ExecutionConfig& config = V5ExecutionConfig{},
    void* stream = nullptr);

// Multi-Beam Device-Resident variant with pre-allocated max_beams output stride.
void cuda_beam_tracker_v5_multibeam_device_resident(
    const std::uint8_t* d_packed, float* d_intensity,
    const Dimensions& dims, std::size_t max_beams_allocated,
    const MultiTrackerConfig& multi_tracker,
    const V5ExecutionConfig& config = V5ExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Batched Tracker Stream V5 (Persistent Pipeline / CUDA Graph Execution)
// ---------------------------------------------------------------------------
class BatchedTrackerStreamV5 {
public:
    BatchedTrackerStreamV5(const Dimensions& single_window_dims,
                           const TrackerConfig& tracker,
                           std::size_t batch_size,
                           const V5ExecutionConfig& config = V5ExecutionConfig{});
    ~BatchedTrackerStreamV5();

    BatchedTrackerStreamV5(const BatchedTrackerStreamV5&) = delete;
    BatchedTrackerStreamV5& operator=(const BatchedTrackerStreamV5&) = delete;

    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed,
                       float* host_intensity);

    void process_batch_kernel_only(std::size_t first_window_index);

    float last_kernel_time_ms() const;

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

// ---------------------------------------------------------------------------
// Multi-Beam Batched Tracker Stream with Live Dynamic Updates
// ---------------------------------------------------------------------------
class MultiBeamBatchedTrackerStreamV5 {
public:
    MultiBeamBatchedTrackerStreamV5(
        const Dimensions& single_window_dims,
        std::size_t max_beams,
        const MultiTrackerConfig& initial_tracker,
        std::size_t batch_size,
        const V5ExecutionConfig& config = V5ExecutionConfig{});
    ~MultiBeamBatchedTrackerStreamV5();

    MultiBeamBatchedTrackerStreamV5(const MultiBeamBatchedTrackerStreamV5&) = delete;
    MultiBeamBatchedTrackerStreamV5& operator=(const MultiBeamBatchedTrackerStreamV5&) = delete;

    // Dynamic live controls (0 ns / thread-safe host updates)
    void set_trajectory(std::size_t beam_id, const TrackerTrajectoryConfig& traj);
    void set_num_active_beams(std::size_t count);
    std::size_t num_active_beams() const;

    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed,
                       float* host_intensity);

    void process_batch_kernel_only(std::size_t first_window_index);

    float last_kernel_time_ms() const;

    std::uint8_t* device_packed_buffer();
    float* device_intensity_buffer();
    void* device_stream();

    std::size_t batch_size() const;
    std::size_t max_beams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
