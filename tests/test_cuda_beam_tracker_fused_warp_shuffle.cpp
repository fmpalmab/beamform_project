// tests/test_cuda_beam_tracker_fused_warp_shuffle.cpp
//
// Correctness unit tests for Phase 4 Fused Warp-Shuffle Tracker.
// Validates Direct Registers, Shared Memory staging, and Double-Buffered
// Streaming against the CPU reference for both 32-antenna and 64-antenna arrays.

#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace beamformer;

bool check_tolerance(const Intensities& ref, const Intensities& test, 
                     const float rel_tol = 1e-4F, const float abs_tol = 1e-5F) {
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
    std::printf("--- Testing n_ant=%zu (n_time=%zu, window=%zu) ---\n", 
                n_ant, n_time, integration_spectra);

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

    // 1. CPU Baseline
    Intensities cpu_ref(total_cells);
    beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_ref);

    // 2. Direct Registers
    Intensities gpu_direct(total_cells);
    cuda_beam_tracker_fused_warp_shuffle_into(
        packed, dims, tracker_cfg, gpu_direct, FwsLoadStrategy::DirectRegisters);
    if (!check_tolerance(cpu_ref, gpu_direct)) {
        std::fprintf(stderr, "  -> FAIL: Direct Registers mode failed for n_ant=%zu\n", n_ant);
        return false;
    }
    std::printf("  -> PASS: Direct Registers matches CPU baseline.\n");

    // 3. Shared Memory Staging
    Intensities gpu_smem(total_cells);
    cuda_beam_tracker_fused_warp_shuffle_into(
        packed, dims, tracker_cfg, gpu_smem, FwsLoadStrategy::SharedMemory);
    if (!check_tolerance(cpu_ref, gpu_smem)) {
        std::fprintf(stderr, "  -> FAIL: Shared Memory mode failed for n_ant=%zu\n", n_ant);
        return false;
    }
    std::printf("  -> PASS: Shared Memory staging matches CPU baseline.\n");

    // 4. Double-Buffered Streaming
    Intensities gpu_stream(total_cells);
    cuda_beam_tracker_fused_warp_shuffle_stream(
        packed, dims, tracker_cfg, gpu_stream, /*n_streams=*/2, FwsLoadStrategy::DirectRegisters);
    if (!check_tolerance(cpu_ref, gpu_stream)) {
        std::fprintf(stderr, "  -> FAIL: Streamed pipeline failed for n_ant=%zu\n", n_ant);
        return false;
    }
    std::printf("  -> PASS: Streamed double-buffered pipeline matches CPU baseline.\n");

    return true;
}

int main() {
    try {
        // Test both supported antenna configurations (32: 1 ant/lane, 64: 2 ant/lane)
        if (!run_test_configuration(64)) return 1;
        if (!run_test_configuration(32)) return 1;

        std::printf("\nAll CUDA Fused Warp-Shuffle unit tests passed successfully.\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}