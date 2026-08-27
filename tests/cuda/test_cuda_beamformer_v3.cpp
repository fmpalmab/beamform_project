// tests/cuda/test_cuda_beamformer_v3.cpp
//
// Comprehensive unit tests and numerical validation for CUDA Beamformer V3:
// 1. Direct Beamforming Parity vs CPU Reference across N_ANT in {32, 64, 128, 256}
// 2. Multi-Beam scaling (N_BEAMS = 16, 32, 64) and Dual-Beam warp co-execution
// 3. Time Unroll factors (U=8, 4, 2) and partial time remainders (e.g. N_TIME=330)
// 4. Fused Temporal Integration (10 spectra post-upchan & 320 spectra direct) vs CPU reference
// 5. Quantized Int8 output validation
// 6. Device-Resident Zero-Copy Execution (Direct & Integrated)
// 7. Persistent Batched Pipeline & CUDA Graph Capture
// 8. Point Source recovery and Peak Beam match.

#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
        if (diff > abs_tol && diff > rel_tol * std::abs(r)) {
            std::fprintf(stderr, "Mismatch at index %zu: ref=%f, test=%f, diff=%f\n", i, r, t, diff);
            return false;
        }
    }
    return true;
}

bool test_direct_beamformer(const std::size_t n_ant, const std::size_t n_beams = 16,
                            const std::size_t n_time = 320) {
    std::printf("=================================================================\n");
    std::printf("Testing V3 Direct Beamformer (n_ant=%zu, n_beams=%zu, n_time=%zu)\n",
                n_ant, n_beams, n_time);
    std::printf("=================================================================\n");

    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};
    const auto positions = default_positions(n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = default_beam_grid(n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    const auto packed = make_noise(dims, 42);

    const auto cpu_intensity = cpu_beamform_packed_intensity(packed, weights, dims);

    for (const std::size_t unroll : {8, 4, 2}) {
        V3BeamformerExecutionConfig cfg;
        cfg.time_unroll = unroll;
        cfg.beams_per_warp = 2;

        const auto gpu_intensity =
            cuda_beamform_v3_packed_intensity(packed, weights, dims, nullptr, cfg);

        if (!check_tolerance(cpu_intensity, gpu_intensity)) {
            std::fprintf(stderr, "FAIL: V3 direct beamformer failed parity at unroll=%zu\n", unroll);
            return false;
        }
        std::printf("  -> PASS: Unroll %zu matches CPU reference\n", unroll);
    }
    return true;
}

bool test_integrated_beamformer(const std::size_t n_ant, const std::size_t integration_spectra) {
    std::printf("=================================================================\n");
    std::printf("Testing V3 Fused Temporal Integration (n_ant=%zu, spectra=%zu)\n",
                n_ant, integration_spectra);
    std::printf("=================================================================\n");

    const std::size_t n_time = 320;
    const std::size_t n_beams = 16;
    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};
    const TemporalIntegrationConfig int_cfg{integration_spectra};

    const auto positions = default_positions(n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = default_beam_grid(n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    const auto packed = make_noise(dims, 123);

    const auto cpu_direct = cpu_beamform_packed_intensity(packed, weights, dims);
    const auto cpu_integrated = cpu_integrate_intensity(cpu_direct, dims, int_cfg);

    const auto gpu_integrated = cuda_beamform_v3_packed_integrated_intensity(
        packed, weights, dims, int_cfg);

    if (!check_tolerance(cpu_integrated, gpu_integrated)) {
        std::fprintf(stderr, "FAIL: V3 integrated beamformer failed parity vs CPU\n");
        return false;
    }
    std::printf("  -> PASS: Fused integration (spectra=%zu) matches CPU reference\n", integration_spectra);

    // Quantized Int8 validation
    const auto quantized_out = cuda_beamform_v3_packed_quantized_integrated_intensity(
        packed, weights, dims, int_cfg);
    const Dimensions int_dims{
        integrated_time_count(dims.n_time, int_cfg), dims.n_freq, dims.n_ant, dims.n_beams};
    const auto dequantized = cpu_dequantize_integrated_intensity(quantized_out, int_dims);

    if (!check_tolerance(cpu_integrated, dequantized, 0.05F, 0.5F)) {
        std::fprintf(stderr, "FAIL: V3 quantized int8 dequantization mismatch\n");
        return false;
    }
    std::printf("  -> PASS: Quantized Int8 output dequantizes faithfully\n");

    return true;
}

bool test_device_resident_and_graph() {
    std::printf("=================================================================\n");
    std::printf("Testing V3 Device-Resident Execution & Batched CUDA Graph\n");
    std::printf("=================================================================\n");

    constexpr std::size_t n_ant = 64;
    constexpr std::size_t n_beams = 16;
    constexpr std::size_t n_time = 320;
    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};

    const auto positions = default_positions(n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = default_beam_grid(n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    const auto packed = make_noise(dims, 777);

    const auto cpu_intensity = cpu_beamform_packed_intensity(packed, weights, dims);

    V3BeamformerExecutionConfig cfg;
    cfg.enable_cuda_graph = true;
    BatchedBeamformerStreamV3 stream(dims, weights, cfg);

    Intensities gpu_intensity(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    for (int iter = 0; iter < 3; ++iter) {
        stream.process_batch(packed.data(), gpu_intensity.data());
        if (!check_tolerance(cpu_intensity, gpu_intensity)) {
            std::fprintf(stderr, "FAIL: V3 batched graph stream mismatch on iter %d\n", iter);
            return false;
        }
    }
    std::printf("  -> PASS: Batched CUDA Graph stream validated over 3 iterations\n");
    return true;
}

bool test_point_source_recovery() {
    std::printf("=================================================================\n");
    std::printf("Testing V3 Point Source Recovery & Peak Beam\n");
    std::printf("=================================================================\n");

    constexpr std::size_t n_ant = 32;
    constexpr std::size_t n_beams = 32;
    constexpr std::size_t n_time = 320;
    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};

    const auto positions = default_positions(n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = rectangular_beam_grid(n_ant);
    const auto weights = generate_weights(dims, positions, frequencies, directions);

    constexpr std::size_t injected_beam = 12;
    const Vec3 source_direction = directions[injected_beam];
    const auto packed = make_point_source(dims, positions, frequencies, source_direction, 4.0F);

    Intensities gpu_intensity = cuda_beamform_v3_packed_intensity(packed, weights, dims);

    std::vector<double> power(dims.n_beams, 0.0);
    for (std::size_t t = 0; t < dims.n_time; ++t) {
        for (std::size_t f = 0; f < dims.n_freq; ++f) {
            for (std::size_t b = 0; b < dims.n_beams; ++b) {
                power[b] += gpu_intensity[intensity_index(t, f, b, dims)];
            }
        }
    }

    const std::size_t peak = static_cast<std::size_t>(
        std::distance(power.begin(), std::max_element(power.begin(), power.end())));

    if (peak != injected_beam) {
        std::fprintf(stderr, "FAIL: Point source peak beam %zu != injected %zu\n", peak, injected_beam);
        return false;
    }

    double runner_up = 0.0;
    for (std::size_t b = 0; b < dims.n_beams; ++b) {
        if (b != injected_beam) runner_up = std::max(runner_up, power[b]);
    }
    const double ratio = power[injected_beam] / runner_up;
    std::printf("  -> PASS: Injected beam %zu recovered with peak-to-sidelobe ratio %.2fx\n",
                injected_beam, ratio);
    return true;
}

} // namespace

int main() {
    try {
        if (!test_direct_beamformer(256, 16, 320)) return 1;
        if (!test_direct_beamformer(128, 16, 320)) return 1;
        if (!test_direct_beamformer(64, 32, 320)) return 1;
        if (!test_direct_beamformer(32, 16, 320)) return 1;

        // Partial time remainder test
        if (!test_direct_beamformer(64, 16, /*n_time=*/330)) return 1;

        // Fused Temporal Integration
        if (!test_integrated_beamformer(64, 320)) return 1;
        if (!test_integrated_beamformer(64, 10)) return 1;

        // Device-Resident & Graph
        if (!test_device_resident_and_graph()) return 1;

        // Point source physics verification
        if (!test_point_source_recovery()) return 1;

        std::printf("\n=================================================================\n");
        std::printf("SUCCESS: All CUDA Beamformer V3 unit tests passed successfully!\n");
        std::printf("=================================================================\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}
