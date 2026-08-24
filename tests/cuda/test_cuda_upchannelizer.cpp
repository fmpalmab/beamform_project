// tests/cuda/test_cuda_upchannelizer.cpp
//
// Comprehensive unit tests and numerical validation for CUDA Upchannelizer:
// 1. Voltage Upchannelizer Parity vs CPU Reference across M in {4, 8, 16, 32}
// 2. Window Function verification (Rectangular, Hann, Hamming, Blackman)
// 3. Single-tone CW injection and fine-channel spectral peak recovery
// 4. Fused Beam Tracker + Upchannelizer numerical parity vs CPU reference
// 5. CudaUpchannelizerWorkspace verification

#include "beamformer/config.hpp"
#include "beamformer/cuda_upchannelizer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace beamformer;

namespace {

constexpr double PI = 3.14159265358979323846;

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

bool test_voltage_upchannelizer(std::size_t M, UpchannelizerWindowType window_type, const char* win_name) {
    std::printf("=================================================================\n");
    std::printf("Testing Voltage Upchannelizer: M=%zu, Window=%s\n", M, win_name);
    std::printf("=================================================================\n");

    const std::size_t n_time = 3200;
    const std::size_t n_freq = default_frequency_channels; // 336
    const std::size_t n_beams = 1;
    const Dimensions dims{n_time, n_freq, 64, n_beams};

    // Synthetic complex voltage
    ComplexVoltage voltage(n_time * n_freq * n_beams);
    for (std::size_t i = 0; i < voltage.size(); ++i) {
        const float t_val = static_cast<float>(i % n_time);
        voltage[i] = {
            std::cos(0.05F * t_val),
            std::sin(0.05F * t_val)
        };
    }

    UpchannelizerConfig cfg;
    cfg.upchan_factor = M;
    cfg.window = window_type;

    // 1. CPU Reference
    Intensities cpu_ref = cpu_upchannelize_voltage(voltage, dims, cfg);

    // 2. CUDA Output
    Intensities gpu_out = cuda_upchannelize_voltage(voltage, dims, cfg);

    if (!check_tolerance(cpu_ref, gpu_out, 1e-3F, 1e-3F)) {
        std::fprintf(stderr, "FAIL: Voltage Upchannelizer parity failed for M=%zu, Window=%s\n", M, win_name);
        return false;
    }
    std::printf("  -> PASS: CUDA Upchannelizer matches CPU reference for M=%zu (%s).\n", M, win_name);
    return true;
}

bool test_spectral_tone_recovery(std::size_t M = 32) {
    std::printf("=================================================================\n");
    std::printf("Testing Spectral Tone Peak Recovery: M=%zu\n", M);
    std::printf("=================================================================\n");

    const std::size_t n_time = 3200;
    const std::size_t n_freq = default_frequency_channels;
    const Dimensions dims{n_time, n_freq, 64, 1};

    // Inject a pure tone at sub-channel bin k_target = 7 in coarse channel f_target = 3
    const std::size_t f_target = 3;
    const std::size_t k_target = 7;
    const double target_phase_step = 2.0 * PI * static_cast<double>(k_target) / static_cast<double>(M);

    ComplexVoltage voltage(n_time * n_freq, {0.0F, 0.0F});
    for (std::size_t t = 0; t < n_time; ++t) {
        const double phase = target_phase_step * static_cast<double>(t);
        voltage[t * n_freq + f_target] = {
            static_cast<float>(std::cos(phase)),
            static_cast<float>(std::sin(phase))
        };
    }

    UpchannelizerConfig cfg;
    cfg.upchan_factor = M;
    cfg.window = UpchannelizerWindowType::Rectangular; // exact delta function response

    Intensities fine = cuda_upchannelize_voltage(voltage, dims, cfg);

    const std::size_t n_fine_time = n_time / M;
    const std::size_t fine_freq = n_freq * M;

    // Verify for each fine-time spectrum that peak is exactly at (f_target * M + k_target)
    for (std::size_t tau = 0; tau < n_fine_time; ++tau) {
        const std::size_t offset = tau * fine_freq;
        std::size_t max_bin = 0;
        float max_val = -1.0F;
        for (std::size_t bin = 0; bin < fine_freq; ++bin) {
            if (fine[offset + bin] > max_val) {
                max_val = fine[offset + bin];
                max_bin = bin;
            }
        }
        const std::size_t expected_bin = f_target * M + k_target;
        if (max_bin != expected_bin) {
            std::fprintf(stderr, "FAIL: Tone peak at bin %zu, expected %zu\n", max_bin, expected_bin);
            return false;
        }
    }

    std::printf("  -> PASS: Pure sinusoid tone recovered cleanly at fine channel %zu.\n",
                f_target * M + k_target);
    return true;
}

bool test_tracker_upchannelize_fused() {
    std::printf("=================================================================\n");
    std::printf("Testing Fused Beam Tracker + Upchannelizer (M=32, N_ANT=64)\n");
    std::printf("=================================================================\n");

    const std::size_t n_time = 3200;
    const std::size_t n_ant = 64;
    const Dimensions dims{n_time, default_frequency_channels, n_ant, 1};

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

    UpchannelizerConfig upchan_cfg;
    upchan_cfg.upchan_factor = 32;
    upchan_cfg.window = UpchannelizerWindowType::Hann;

    // 1. CPU Reference
    Intensities cpu_ref = cpu_tracker_upchannelize(packed, dims, tracker_cfg, upchan_cfg);

    // 2. CUDA Fused Kernel
    Intensities gpu_out = cuda_tracker_upchannelize(packed, dims, tracker_cfg, upchan_cfg);

    if (!check_tolerance(cpu_ref, gpu_out, 2e-2F, 5e-3F)) {
        std::fprintf(stderr, "FAIL: Fused Tracker + Upchannelizer failed\n");
        return false;
    }
    std::printf("  -> PASS: Fused Tracker + Upchannelizer matches CPU reference.\n");

    // 3. Workspace test
    {
        CudaUpchannelizerWorkspace workspace(dims, upchan_cfg);
        Intensities ws_out(workspace.dimensions().fine_intensity_count());
        workspace.process_tracker(packed, tracker_cfg, ws_out);
        if (!check_tolerance(cpu_ref, ws_out, 2e-2F, 5e-3F)) {
            std::fprintf(stderr, "FAIL: CudaUpchannelizerWorkspace failed\n");
            return false;
        }
        std::printf("  -> PASS: CudaUpchannelizerWorkspace verified (setup_ms = %.3f ms).\n",
                    workspace.setup_ms());
    }

    return true;
}

} // namespace

int main() {
    try {
        if (!test_voltage_upchannelizer(32, UpchannelizerWindowType::Hann, "Hann")) return 1;
        if (!test_voltage_upchannelizer(32, UpchannelizerWindowType::Hamming, "Hamming")) return 1;
        if (!test_voltage_upchannelizer(32, UpchannelizerWindowType::Blackman, "Blackman")) return 1;
        if (!test_voltage_upchannelizer(32, UpchannelizerWindowType::Rectangular, "Rectangular")) return 1;

        if (!test_voltage_upchannelizer(16, UpchannelizerWindowType::Hann, "Hann")) return 1;
        if (!test_voltage_upchannelizer(8, UpchannelizerWindowType::Hann, "Hann")) return 1;
        if (!test_voltage_upchannelizer(4, UpchannelizerWindowType::Hann, "Hann")) return 1;

        if (!test_spectral_tone_recovery(32)) return 1;
        if (!test_tracker_upchannelize_fused()) return 1;

        std::printf("\n=================================================================\n");
        std::printf("SUCCESS: All CUDA Upchannelizer unit tests passed successfully!\n");
        std::printf("=================================================================\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception caught during testing: %s\n", e.what());
        return 1;
    }
    return 0;
}
