// tests/cuda/test_cuda_dual_parallel_pipeline.cpp
//
// Comprehensive unit tests and numerical validation for DualEngineParallelPipeline:
// 1. Zero-copy memory savings and resource verification
// 2. Parity validation: Concurrent execution vs independent CPU/GPU ground truth
// 3. Multi-antenna scaling across N_ANT in {32, 64, 128, 256}
// 4. Live tracker trajectory update without pipeline interruption
// 5. Multi-frame continuous streaming stability

#include "beamformer/cuda_dual_parallel_pipeline.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/cuda_beam_tracker_v5.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace beamformer;

namespace {

bool check_tolerance(const std::vector<float>& ref, const std::vector<float>& test,
                     const float rel_tol = 1e-3F, const float abs_tol = 1e-4F) {
    if (ref.size() != test.size()) {
        std::fprintf(stderr, "Size mismatch: ref=%zu, test=%zu\n", ref.size(), test.size());
        return false;
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float r = ref[i];
        const float t = test[i];
        const float diff = std::abs(r - t);
        if (diff > abs_tol && diff > rel_tol * std::abs(r)) {
            std::fprintf(stderr, "Mismatch at index %zu: ref=%f, test=%f, diff=%f\n", i, r, t, diff);
            return false;
        }
    }
    return true;
}

Weights make_test_weights(const Dimensions& dims) {
    const std::size_t count = dims.n_beams * dims.n_freq * dims.n_ant;
    Weights weights(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float angle = static_cast<float>(i % 360) * 3.14159265358979323846F / 180.0F;
        weights[i] = {std::cos(angle) * 0.125F, std::sin(angle) * 0.125F};
    }
    return weights;
}

bool test_resource_savings() {
    std::printf("=================================================================\n");
    std::printf("Testing Dual Pipeline Resource & Memory Savings\n");
    std::printf("=================================================================\n");

    const Dimensions bf_dims{320, default_frequency_channels, 64, 32};
    const Dimensions tr_dims{320, default_frequency_channels, 64, 1};
    const auto bf_weights = make_test_weights(bf_dims);
    const std::vector<TrackerTrajectoryConfig> trajs{
        {direction_from_lm(0.05F, 0.05F), {0.0001F, 0.0001F}}
    };
    const TemporalIntegrationConfig tint{80};

    DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);

    const std::size_t expected_shared_bytes = voltage_sample_count(bf_dims) * sizeof(std::uint8_t);
    if (pipeline.shared_input_bytes() != expected_shared_bytes) {
        std::fprintf(stderr, "FAIL: Shared input bytes mismatch\n");
        return false;
    }
    if (pipeline.memory_saved_bytes() != expected_shared_bytes) {
        std::fprintf(stderr, "FAIL: Memory saved bytes mismatch\n");
        return false;
    }
    std::printf("  -> PASS: Shared input %zu bytes (saved 50%% PCIe & VRAM)\n", expected_shared_bytes);
    return true;
}

bool test_concurrent_parity(const std::size_t n_ant = 64) {
    std::printf("=================================================================\n");
    std::printf("Testing Dual Pipeline Concurrent Parity (n_ant=%zu)\n", n_ant);
    std::printf("=================================================================\n");

    const Dimensions bf_dims{320, default_frequency_channels, n_ant, 16};
    const Dimensions tr_dims{320, default_frequency_channels, n_ant, 1};
    const auto packed = make_noise(bf_dims, 1234 + n_ant);
    const auto bf_weights = make_test_weights(bf_dims);

    const std::vector<TrackerTrajectoryConfig> trajs{
        {direction_from_lm(0.02F, -0.03F), {0.00005F, 0.00005F}}
    };
    const TemporalIntegrationConfig tint{80};

    // Calculate ground truth references independently
    const auto expected_bf = cuda_beamform_v3_packed_integrated_intensity(
        packed, bf_weights, bf_dims, tint);

    TrackerConfig tr_cfg;
    tr_cfg.trajectory = trajs[0];
    tr_cfg.integration_spectra = tint.integration_spectra;
    const auto expected_tr = cuda_beam_tracker_v5(packed, tr_dims, tr_cfg);

    DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);

    std::vector<float> actual_bf(integrated_intensity_count(bf_dims, tint));
    std::vector<float> actual_tr(tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams);

    const auto timings = pipeline.process_frame(packed.data(), actual_bf.data(), actual_tr.data());

    if (!check_tolerance(expected_bf, actual_bf)) {
        std::fprintf(stderr, "FAIL: Beamformer output mismatch during concurrent execution\n");
        return false;
    }
    if (!check_tolerance(expected_tr, actual_tr)) {
        std::fprintf(stderr, "FAIL: Tracker output mismatch during concurrent execution\n");
        return false;
    }

    std::printf("  -> PASS: Both BF and Tracker outputs match independent ground truth!\n");
    std::printf("     Timings: H2D=%.2f ms, BF_Kernel=%.2f ms, TR_Kernel=%.2f ms, Concurrent_GPU=%.2f ms (Efficiency: %.2fx)\n",
                timings.host_to_device_ms, timings.beamformer_kernel_ms, timings.tracker_kernel_ms,
                timings.concurrent_gpu_ms, timings.parallel_efficiency);
    return true;
}

bool test_live_trajectory_update() {
    std::printf("=================================================================\n");
    std::printf("Testing Live Dynamic Tracker Trajectory Update\n");
    std::printf("=================================================================\n");

    const Dimensions bf_dims{320, default_frequency_channels, 64, 8};
    const Dimensions tr_dims{320, default_frequency_channels, 64, 1};
    const auto packed = make_noise(bf_dims, 8888);
    const auto bf_weights = make_test_weights(bf_dims);

    std::vector<TrackerTrajectoryConfig> trajs{
        {direction_from_lm(0.0F, 0.0F), {0.0F, 0.0F}}
    };
    const TemporalIntegrationConfig tint{80};

    DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);

    std::vector<float> actual_bf(integrated_intensity_count(bf_dims, tint));
    std::vector<float> actual_tr1(tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams);
    std::vector<float> actual_tr2(tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams);

    pipeline.process_frame(packed.data(), actual_bf.data(), actual_tr1.data());

    // Update target trajectory
    const TrackerTrajectoryConfig new_traj{direction_from_lm(0.1F, 0.1F), {0.001F, 0.001F}};
    pipeline.update_tracker_trajectory(0, new_traj);

    pipeline.process_frame(packed.data(), actual_bf.data(), actual_tr2.data());

    bool any_difference = false;
    for (std::size_t i = 0; i < actual_tr1.size(); ++i) {
        if (std::abs(actual_tr1[i] - actual_tr2[i]) > 1e-4F) {
            any_difference = true;
            break;
        }
    }
    if (!any_difference) {
        std::fprintf(stderr, "FAIL: Trajectory update produced identical output\n");
        return false;
    }
    std::printf("  -> PASS: Live trajectory update modified tracking beam steering seamlessly\n");
    return true;
}

bool test_multi_frame_stability() {
    std::printf("=================================================================\n");
    std::printf("Testing Multi-Frame Continuous Streaming Stability\n");
    std::printf("=================================================================\n");

    const Dimensions bf_dims{160, default_frequency_channels, 64, 8};
    const Dimensions tr_dims{160, default_frequency_channels, 64, 1};
    const auto packed = make_noise(bf_dims, 4321);
    const auto bf_weights = make_test_weights(bf_dims);

    const std::vector<TrackerTrajectoryConfig> trajs{
        {direction_from_lm(0.05F, 0.02F), {0.0001F, 0.0001F}}
    };
    const TemporalIntegrationConfig tint{80};

    DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);

    std::vector<float> actual_bf(integrated_intensity_count(bf_dims, tint));
    std::vector<float> actual_tr(tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams);

    for (int frame = 0; frame < 10; ++frame) {
        const auto timings = pipeline.process_frame(packed.data(), actual_bf.data(), actual_tr.data());
        if (timings.concurrent_gpu_ms <= 0.0) {
            std::fprintf(stderr, "FAIL: Concurrent GPU timing invalid on frame %d\n", frame);
            return false;
        }
    }
    std::printf("  -> PASS: 10 consecutive frames executed stably\n");
    return true;
}

} // namespace

int main() {
    try {
        if (!test_resource_savings()) return 1;

        if (!test_concurrent_parity(32)) return 1;
        if (!test_concurrent_parity(64)) return 1;
        if (!test_concurrent_parity(128)) return 1;
        if (!test_concurrent_parity(256)) return 1;

        if (!test_live_trajectory_update()) return 1;
        if (!test_multi_frame_stability()) return 1;

        std::printf("\n=================================================================\n");
        std::printf("SUCCESS: All Dual Parallel Pipeline unit tests passed successfully!\n");
        std::printf("=================================================================\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}
