// tests/cuda/test_cuda_beamformer_v2.cpp
//
// Comprehensive unit tests and numerical validation for CUDA Beamformer V2:
// 1. Direct Beamforming Parity vs CPU Reference across N_ANT in {32, 64, 128, 256}
// 2. Multi-Beam scaling (N_BEAMS = 16, 32, 64)
// 3. Time Unroll factors (U=8, 4, 2) and partial time remainders (e.g. N_TIME=3300)
// 4. Fused Temporal Integration (10 spectra post-upchan & 320 spectra direct) vs CPU reference
// 5. Quantized Int8 output validation
// 6. Device-Resident Zero-Copy Execution (Direct & Integrated)
// 7. Persistent Batched Pipeline & CUDA Graph Capture
// 8. Point Source recovery and Peak Beam match.

#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer_v2.hpp"
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
    std::printf("Testing V2 Direct: n_ant=%zu, n_beams=%zu, n_time=%zu\n",
                n_ant, n_beams, n_time);
    std::printf("=================================================================\n");

    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = (dims.n_ant <= 64)
                                ? fft_beam_grid(dims.n_ant, dims.n_beams)
                                : default_beam_grid(dims.n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);

    const auto packed = make_noise(dims, 42);
    const std::size_t total_outputs = dims.n_time * dims.n_freq * dims.n_beams;

    // 1. CPU Reference
    Intensities cpu_ref(total_outputs);
    cpu_beamform_packed_intensity_into(packed, weights, dims, cpu_ref);

    // 2. V2 Unroll 8, Chunk 80
    {
        V2BeamformerExecutionConfig cfg;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 8;
        Intensities gpu_v2_u8(total_outputs);
        cuda_beamform_v2_packed_intensity_into(packed, weights, dims, gpu_v2_u8, nullptr, cfg);
        if (!check_tolerance(cpu_ref, gpu_v2_u8)) {
            std::fprintf(stderr, "FAIL: V2 Unroll 8 failed for n_ant=%zu, n_beams=%zu\n", n_ant, n_beams);
            return false;
        }
        std::printf("  -> PASS: V2 Direct (Unroll 8, Chunk 80) matches CPU reference.\n");
    }

    // 3. V2 Unroll 4, Chunk 80
    {
        V2BeamformerExecutionConfig cfg;
        cfg.time_chunk_size = 80;
        cfg.time_unroll = 4;
        Intensities gpu_v2_u4(total_outputs);
        cuda_beamform_v2_packed_intensity_into(packed, weights, dims, gpu_v2_u4, nullptr, cfg);
        if (!check_tolerance(cpu_ref, gpu_v2_u4)) {
            std::fprintf(stderr, "FAIL: V2 Unroll 4 failed for n_ant=%zu, n_beams=%zu\n", n_ant, n_beams);
            return false;
        }
        std::printf("  -> PASS: V2 Direct (Unroll 4, Chunk 80) matches CPU reference.\n");
    }

    // 4. V2 Unroll 2, Chunk 320
    {
        V2BeamformerExecutionConfig cfg;
        cfg.time_chunk_size = 320;
        cfg.time_unroll = 2;
        Intensities gpu_v2_u2(total_outputs);
        cuda_beamform_v2_packed_intensity_into(packed, weights, dims, gpu_v2_u2, nullptr, cfg);
        if (!check_tolerance(cpu_ref, gpu_v2_u2)) {
            std::fprintf(stderr, "FAIL: V2 Unroll 2 failed for n_ant=%zu, n_beams=%zu\n", n_ant, n_beams);
            return false;
        }
        std::printf("  -> PASS: V2 Direct (Unroll 2, Chunk 320) matches CPU reference.\n");
    }

    // 5. Device-Resident Zero-Copy Execution
    {
        const std::size_t v_bytes = voltage_sample_count(dims) * sizeof(std::uint8_t);
        const std::size_t w_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
        const std::size_t out_bytes = total_outputs * sizeof(float);

        std::uint8_t* d_packed = nullptr;
        ComplexFloat* d_weights = nullptr;
        float* d_intensity = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_weights), w_bytes);
        cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);

        cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights.data(), w_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_intensity, 0, out_bytes);

        V2BeamformerExecutionConfig cfg;
        cfg.time_unroll = 8;
        cuda_beamform_v2_device_resident(d_packed, d_weights, d_intensity, dims, cfg);

        Intensities gpu_resident(total_outputs);
        cudaMemcpy(gpu_resident.data(), d_intensity, out_bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_packed);
        cudaFree(d_weights);
        cudaFree(d_intensity);

        if (!check_tolerance(cpu_ref, gpu_resident)) {
            std::fprintf(stderr, "FAIL: V2 Device-Resident failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: V2 Device-Resident matches CPU reference.\n");
    }

    // 6. BatchedBeamformerStreamV2 with CUDA Graph Capture
    {
        V2BeamformerExecutionConfig cfg;
        cfg.time_unroll = 8;
        cfg.enable_cuda_graph = true;

        BatchedBeamformerStreamV2 stream(dims, weights, cfg);
        Intensities gpu_stream(total_outputs);
        // First iteration captures graph, second executes graph
        stream.process_batch(packed.data(), gpu_stream.data());
        stream.process_batch(packed.data(), gpu_stream.data());

        if (!check_tolerance(cpu_ref, gpu_stream)) {
            std::fprintf(stderr, "FAIL: BatchedBeamformerStreamV2 (Graph) failed for n_ant=%zu\n", n_ant);
            return false;
        }
        std::printf("  -> PASS: BatchedBeamformerStreamV2 (CUDA Graph) matches CPU ref (Kernel time: %.3f ms).\n",
                    stream.last_kernel_time_ms());
    }

    return true;
}

bool test_integrated_beamformer(const std::size_t n_ant, const std::size_t integration_spectra) {
    std::printf("=================================================================\n");
    std::printf("Testing V2 Fused Temporal Integration: n_ant=%zu, spectra=%zu\n",
                n_ant, integration_spectra);
    std::printf("=================================================================\n");

    const std::size_t n_time = 320;
    const std::size_t n_beams = 16;
    const Dimensions dims{n_time, default_frequency_channels, n_ant, n_beams};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = (dims.n_ant <= 64)
                                ? fft_beam_grid(dims.n_ant, dims.n_beams)
                                : default_beam_grid(dims.n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    const auto packed = make_noise(dims, 123);

    TemporalIntegrationConfig temporal_cfg{integration_spectra};
    const std::size_t integrated_outputs = integrated_intensity_count(dims, temporal_cfg);

    // 1. CPU Reference: Compute direct intensity, then integrate
    Intensities cpu_direct(dims.n_time * dims.n_freq * dims.n_beams);
    cpu_beamform_packed_intensity_into(packed, weights, dims, cpu_direct);
    IntegratedIntensities cpu_integrated = cpu_integrate_intensity(cpu_direct, dims, temporal_cfg);

    // 2. V2 Fused Integration
    IntegratedIntensities gpu_integrated(integrated_outputs);
    cuda_beamform_v2_packed_integrated_intensity_into(
        packed, weights, dims, temporal_cfg, gpu_integrated);

    if (!check_tolerance(cpu_integrated, gpu_integrated)) {
        std::fprintf(stderr, "FAIL: V2 Fused Temporal Integration failed for n_ant=%zu, spectra=%zu\n",
                     n_ant, integration_spectra);
        return false;
    }
    std::printf("  -> PASS: V2 Fused Temporal Integration matches CPU reference.\n");

    // 3. Quantized Int8 Output
    {
        const Dimensions integrated_dims{
            integrated_time_count(dims.n_time, temporal_cfg),
            dims.n_freq, dims.n_ant, dims.n_beams};
        const auto cpu_quantized = cpu_quantize_integrated_intensity(cpu_integrated, integrated_dims);

        auto gpu_quantized = cuda_beamform_v2_packed_quantized_integrated_intensity(
            packed, weights, dims, temporal_cfg);

        if (gpu_quantized.codes.size() != cpu_quantized.codes.size()) {
            std::fprintf(stderr, "FAIL: Quantized code size mismatch\n");
            return false;
        }

        std::size_t code_mismatches = 0;
        for (std::size_t i = 0; i < cpu_quantized.codes.size(); ++i) {
            if (std::abs(static_cast<int>(cpu_quantized.codes[i]) - static_cast<int>(gpu_quantized.codes[i])) > 1) {
                ++code_mismatches;
            }
        }
        if (code_mismatches != 0) {
            std::fprintf(stderr, "FAIL: Quantized code mismatches = %zu\n", code_mismatches);
            return false;
        }
        std::printf("  -> PASS: V2 Quantized Int8 output verified.\n");
    }

    return true;
}

bool test_point_source_recovery() {
    std::printf("=================================================================\n");
    std::printf("Testing V2 Point Source Recovery & Peak Beam\n");
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

    Intensities gpu_intensity = cuda_beamform_v2_packed_intensity(packed, weights, dims);

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

        // Point source physics verification
        if (!test_point_source_recovery()) return 1;

        std::printf("\n=================================================================\n");
        std::printf("SUCCESS: All CUDA Beamformer V2 unit tests passed successfully!\n");
        std::printf("=================================================================\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}
