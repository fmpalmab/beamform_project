// tests/cuda/test_cuda_tracker_v2.cpp
//
// Correctness test for the CUDA v2 beam tracker implementations.
// Validates three kernels against the optimized CPU reference path
// using a floating-point tolerance check, accounting for architecture
// and compiler-level math optimizations (like GPU Fused Multiply-Add).

#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_tracker_v2.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace beamformer;

// Tolerance checker to account for CPU vs GPU floating-point rounding differences
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

int main() {
    try {
        const Dimensions dims{3200, default_frequency_channels, 64, tracker_beam_count};
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
        tracker_cfg.integration_spectra = 320;

        const std::size_t total_cells = dims.n_time * dims.n_freq * dims.n_beams;

        // 1. Generate CPU Baseline
        std::printf("Running CPU baseline (beam_tracker_opt_v2)...\n");
        Intensities cpu_ref(total_cells);
        beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_ref);

        // 2. Test CUDA TwoPass Kernel
        std::printf("Running CUDA TwoPass Kernel...\n");
        Intensities cuda_twopass(total_cells);
        cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_twopass, 
                                              CudaTrackerKernelV2::TwoPass);
        
        if (!check_tolerance(cpu_ref, cuda_twopass, 1e-4F, 1e-5F)) {
            std::fprintf(stderr, "FAIL: CUDA TwoPass output exceeds floating-point tolerance.\n");
            return 1;
        }
        std::printf("  -> PASS: TwoPass output is within floating-point tolerance of CPU.\n");

        // 3. Test CUDA Fused Kernel
        std::printf("Running CUDA Fused Kernel...\n");
        Intensities cuda_fused(total_cells);
        cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_fused, 
                                              CudaTrackerKernelV2::Fused);
        
        if (!check_tolerance(cpu_ref, cuda_fused, 1e-4F, 1e-5F)) {
            std::fprintf(stderr, "FAIL: CUDA Fused output exceeds floating-point tolerance.\n");
            return 1;
        }
        std::printf("  -> PASS: Fused output is within floating-point tolerance of CPU.\n");

        // 4. Test CUDA WarpReduction Kernel
        std::printf("Running CUDA WarpReduction Kernel...\n");
        Intensities cuda_warp(total_cells);
        cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_warp, 
                                              CudaTrackerKernelV2::WarpReduction);
        
        if (!check_tolerance(cpu_ref, cuda_warp, 1e-4F, 1e-5F)) {
            std::fprintf(stderr, "FAIL: CUDA WarpReduction output exceeds floating-point tolerance.\n");
            return 1;
        }
        std::printf("  -> PASS: WarpReduction output is within floating-point tolerance of CPU.\n");

        std::printf("\nAll CUDA tracker v2 tests passed successfully.\n");
        
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }

    return 0;
}