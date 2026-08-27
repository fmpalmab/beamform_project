#pragma once

// CUDA Dual Parallel Pipeline: Unified Concurrent Execution Engine for
// Simultaneous Fixed-Grid Beamforming (V3) and Multi-Beam Dynamic Tracking (V5).
//
// Key Optimizations for Maximum Runtime and Resource Efficiency:
// 1. Zero-Copy Input Sharing:
//    Uploads the packed voltage frame to device memory EXACTLY ONCE over PCIe.
//    Both Beamformer and Tracker kernels read directly from this shared buffer,
//    eliminating 50% of PCIe transfer overhead and 50% of input VRAM footprint.
//
// 2. Hardware Stream Concurrency:
//    Dispatches the Beamformer and Tracker onto separate non-blocking CUDA streams
//    using cross-stream events (`cudaStreamWaitEvent`), enabling simultaneous
//    warp execution and overlapping compute across GPU Streaming Multiprocessors.
//
// 3. Consolidated CUDA Graph Fork-Join Execution:
//    Captures the H2D upload, parallel kernel fork, and D2H download join into a
//    single execution graph, reducing CPU launch latency to near zero.
//
// 4. Live Multi-Target Tracker Reconfiguration:
//    Allows dynamic updates to tracking trajectories (l0, m0, dl, dm) and active
//    beam counts on the fly without interrupting the fixed-grid survey beamformer.

#include "beamformer/config.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/cuda_beam_tracker_v5.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/temporal_integration.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace beamformer {

// Performance diagnostics for joint parallel execution
struct DualParallelTimings {
    double host_to_device_ms = 0.0;
    double beamformer_kernel_ms = 0.0;
    double tracker_kernel_ms = 0.0;
    double concurrent_gpu_ms = 0.0;  // Total concurrent GPU execution time
    double sequential_sum_ms = 0.0;  // BF kernel ms + Tracker kernel ms (for speedup calc)
    double device_to_host_ms = 0.0;
    double total_wall_ms = 0.0;
    double parallel_efficiency = 1.0; // sequential_sum_ms / concurrent_gpu_ms
};

// Configuration for joint execution
struct DualParallelConfig {
    V3BeamformerExecutionConfig beamformer_config;
    V5ExecutionConfig tracker_config;
    bool enable_cuda_graph = false;
    bool direct_device_resident = false; // true = caller manages GPU buffers
};

class DualEngineParallelPipeline {
public:
    DualEngineParallelPipeline(
        const Dimensions& bf_dims,
        const Weights& bf_weights,
        const Dimensions& tracker_dims,
        const std::vector<TrackerTrajectoryConfig>& tracker_trajectories,
        const TemporalIntegrationConfig& temporal_integration,
        const DualParallelConfig& config = DualParallelConfig{});
    ~DualEngineParallelPipeline();

    DualEngineParallelPipeline(const DualEngineParallelPipeline&) = delete;
    DualEngineParallelPipeline& operator=(const DualEngineParallelPipeline&) = delete;

    // Process one input frame: Host-to-Device -> Concurrent Kernels -> Device-to-Host
    DualParallelTimings process_frame(
        const std::uint8_t* host_packed_voltage,
        float* host_bf_integrated_output,
        float* host_tracker_integrated_output);

    // Device-resident process: runs concurrently on already-uploaded d_packed_voltage
    DualParallelTimings process_device_resident(
        const std::uint8_t* d_packed_voltage,
        float* d_bf_integrated_output,
        float* d_tracker_integrated_output);

    // Live update for tracking trajectory without reallocating or stopping pipeline
    void update_tracker_trajectory(
        std::size_t beam_index, const TrackerTrajectoryConfig& new_traj);

    // Dynamic active beam count update for tracker (0 to MAX_BEAMS)
    void set_tracker_active_beams(std::size_t active_count);

    // Memory savings and resource query
    std::size_t shared_input_bytes() const;
    std::size_t memory_saved_bytes() const;
    const Dimensions& beamformer_dimensions() const;
    const Dimensions& tracker_dimensions() const;

    // Accessors to device pointers
    std::uint8_t* device_shared_packed_voltage();
    float* device_beamformer_output();
    float* device_tracker_output();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
