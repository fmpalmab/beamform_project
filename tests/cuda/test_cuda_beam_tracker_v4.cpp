// tests/cuda/test_cuda_beam_tracker_v4.cpp
//
// Unit tests and numerical validation for CUDA Beam Tracker V4.
// Validates:
// 1. Deep ILP Unrolling (T_UNROLL = 8, 4, 2)
// 2. Half2 Vector SIMD mode
// 3. Time-Chunk Tiling (80 vs 320)
// 4. Multi-Stream Pipelined Execution
// 5. Batched Continuous Execution & CUDA Graph Stream
// 6. Device-Resident Zero-Copy Execution
// 7. Pinned Memory Allocation Helpers
// 8. 32-antenna and 64-antenna array geometries
// 9. Numerical parity against CPU Opt v2 reference.

#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_beam_tracker_v4.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace beamformer;

namespace {

bool check_tolerance(const Intensities& ref, const Intensities& test, 
                     const float rel_tol = 1e-3F, const float abs_tol = 1e-4F) {
    if (ref.size() != test.size()) {
        std::fprintf(stderr, "Size mismatch: ref=%zu, test=%zu\n", ref.size(), test.size());
        return false;
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float r = ref[i];
        const float t = test[i];
        const float diff = std::abs(r - t);
        if (diff > abs_tol && diff > rel_tol * r) {
            std::fprintf(stderr, "Mismatch at index %zu: ref=%f, test=%f, diff=%f\n", i, r, t, diff);
            return false;
        }
    }
    return true;
}

bool run_test_configuration(const std::size_t n_ant, const std::size_t n_time = 3200, 
                            const std::size_t integration_spectra = 320) {
    std::printf("=================================================================\n");
    std::printf("Testing V4: n_ant=%zu, n_time=%zu, integration_spectra=%zu\n",
                n_ant, n_time, integration_spectra);
    std::printf("=================================================================\n");

    const Dimensions dims{n_time, default_frequency_channels, n_ant, tracker_beam_count};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

    TrackerTrajectoryConfig source_traj{
        direction_from_lm(0.0F, 0.0F),
        {1.0e-5F, 0.0F}
    };
    const auto packed = beam_tracker_make_moving_point_source(
        dims, positions, frequencies, source_traj, 4.0F);

    TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
    tracker_cfg.trajectory.direction_rate_per_sample = {1.0e-5F, 0.0F};
    tracker_cfg.integration_spectra = integration_spectra;

    const std::size_t total_cells = dims.n_time * dims.n_freq * dims.n_beams;

    // 1. CPU Reference
    Intensities cpu_ref(total_cells);
    beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_ref);

    // 2. V4 Deep ILP (Unroll 8, Chunk 80)
    {
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::DeepIlpWarpShuffle;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 8;
        Intensities gpu_v4_u8(total_cells);
        cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, gpu_v4_u8, cfg);
        if (!check_tolerance(cpu_ref, gpu_v4_u8)) {
            std::fprintf(stderr, "FAIL: V4 Deep ILP (Unroll 8) failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Deep ILP (Unroll 8, Chunk 80) matches CPU reference.\n");
    }

    // 3. V4 Deep ILP (Unroll 4, Chunk 80)
    {
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::DeepIlpWarpShuffle;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 4;
        Intensities gpu_v4_u4(total_cells);
        cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, gpu_v4_u4, cfg);
        if (!check_tolerance(cpu_ref, gpu_v4_u4)) {
            std::fprintf(stderr, "FAIL: V4 Deep ILP (Unroll 4) failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Deep ILP (Unroll 4, Chunk 80) matches CPU reference.\n");
    }

    // 4. V4 Deep ILP (Unroll 2, Chunk 320 - Single Warp per Channel)
    {
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::DeepIlpWarpShuffle;
        cfg.time_chunk_size = 320;
        cfg.time_unroll = 2;
        Intensities gpu_v4_c320(total_cells);
        cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, gpu_v4_c320, cfg);
        if (!check_tolerance(cpu_ref, gpu_v4_c320)) {
            std::fprintf(stderr, "FAIL: V4 Deep ILP (Chunk 320) failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Deep ILP (Chunk 320) matches CPU reference.\n");
    }

    // 5. V4 Half2 Vector SIMD Mode (Available for 32 and 64 antennas)
    if (n_ant <= 64) {
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::Half2VectorSimd;
        cfg.time_chunk_size = 80;
        Intensities gpu_v4_half2(total_cells);
        cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, gpu_v4_half2, cfg);
        if (!check_tolerance(cpu_ref, gpu_v4_half2, 5e-3F, 1e-3F)) {
            std::fprintf(stderr, "FAIL: V4 Half2 Vector SIMD failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Half2 Vector SIMD matches CPU reference within FP16 precision.\n");
    }

    // 6. V4 Streaming Multi-Stream Pipeline
    {
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::Auto;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 4;
        Intensities gpu_v4_stream(total_cells);
        cuda_beam_tracker_v4_stream(packed, dims, tracker_cfg, gpu_v4_stream, /*n_streams=*/3, cfg);
        if (!check_tolerance(cpu_ref, gpu_v4_stream)) {
            std::fprintf(stderr, "FAIL: V4 Streaming pipeline failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Multi-Stream pipeline matches CPU reference.\n");
    }

    // 7. BatchedTrackerStreamV4 (Standard Batch Execution)
    if (dims.n_time % tracker_cfg.integration_spectra == 0) {
        const std::size_t window_count = tracker_window_count(dims.n_time, tracker_cfg.integration_spectra);
        Dimensions win_dims{tracker_cfg.integration_spectra, dims.n_freq, dims.n_ant, dims.n_beams};
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::Auto;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 4;
        cfg.enable_cuda_graph = false;

        BatchedTrackerStreamV4 batched_stream(win_dims, tracker_cfg, window_count, cfg);
        Intensities gpu_batched(total_cells);
        batched_stream.process_batch(0, packed.data(), gpu_batched.data());

        if (!check_tolerance(cpu_ref, gpu_batched)) {
            std::fprintf(stderr, "FAIL: BatchedTrackerStreamV4 failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: BatchedTrackerStreamV4 matches CPU reference (Kernel time: %.3f ms).\n",
                    batched_stream.last_kernel_time_ms());
    }

    // 8. BatchedTrackerStreamV4 (CUDA Graph Execution)
    if (dims.n_time % tracker_cfg.integration_spectra == 0) {
        const std::size_t window_count = tracker_window_count(dims.n_time, tracker_cfg.integration_spectra);
        Dimensions win_dims{tracker_cfg.integration_spectra, dims.n_freq, dims.n_ant, dims.n_beams};
        V4ExecutionConfig cfg;
        cfg.mode = V4KernelMode::Auto;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 4;
        cfg.enable_cuda_graph = true;

        BatchedTrackerStreamV4 graph_stream(win_dims, tracker_cfg, window_count, cfg);
        Intensities gpu_graph(total_cells);
        // First iteration captures graph, second executes instantiated graph
        graph_stream.process_batch(0, packed.data(), gpu_graph.data());
        graph_stream.process_batch(0, packed.data(), gpu_graph.data());

        if (!check_tolerance(cpu_ref, gpu_graph)) {
            std::fprintf(stderr, "FAIL: BatchedTrackerStreamV4 (CUDA Graph) failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: BatchedTrackerStreamV4 with CUDA Graph matches CPU reference (Kernel time: %.3f ms).\n",
                    graph_stream.last_kernel_time_ms());
    }

    // 9. Device-Resident Zero-Copy Execution
    {
        std::uint8_t* d_packed = nullptr;
        float* d_intensity = nullptr;
        const std::size_t v_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
        const std::size_t out_bytes = total_cells * sizeof(float);

        cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);
        cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_intensity, 0, out_bytes);

        V4ExecutionConfig cfg;
        cfg.time_unroll = 4;
        cuda_beam_tracker_v4_device_resident(d_packed, d_intensity, dims, tracker_cfg, cfg);

        Intensities gpu_resident(total_cells);
        cudaMemcpy(gpu_resident.data(), d_intensity, out_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_packed);
        cudaFree(d_intensity);

        if (!check_tolerance(cpu_ref, gpu_resident)) {
            std::fprintf(stderr, "FAIL: V4 Device-Resident execution failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Device-Resident execution matches CPU reference.\n");
    }

    // 10. Pinned Memory Allocation Test
    {
        auto pinned_v = allocate_pinned_voltage_v4(dims);
        auto pinned_i = allocate_pinned_intensities_v4(dims);
        std::memcpy(pinned_v.get(), packed.data(), voltage_sample_count(dims) * sizeof(std::uint8_t));

        PackedVoltage pinned_packed_view(pinned_v.get(), pinned_v.get() + voltage_sample_count(dims));
        Intensities gpu_pinned(total_cells);
        cuda_beam_tracker_v4_into(pinned_packed_view, dims, tracker_cfg, gpu_pinned);

        if (!check_tolerance(cpu_ref, gpu_pinned)) {
            std::fprintf(stderr, "FAIL: V4 Pinned memory test failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V4 Pinned memory allocators verified.\n");
    }

    return true;
}

} // namespace

int main() {
    try {
        if (!run_test_configuration(256)) return 1;
        if (!run_test_configuration(128)) return 1;
        if (!run_test_configuration(64)) return 1;
        if (!run_test_configuration(32)) return 1;

        // Partial final window test
        if (!run_test_configuration(64, /*n_time=*/3300, /*integration_spectra=*/320)) return 1;

        std::printf("\n=================================================================\n");
        std::printf("SUCCESS: All CUDA Beam Tracker V4 unit tests passed successfully!\n");
        std::printf("=================================================================\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}
