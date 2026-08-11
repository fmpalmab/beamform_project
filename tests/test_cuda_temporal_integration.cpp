#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double absolute_tolerance = 1.0e-3;
constexpr double relative_tolerance = 1.0e-5;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare_cpu_cuda(const std::string& label,
                      const beamformer::IntegratedIntensities& expected,
                      const beamformer::IntegratedIntensities& actual) {
    require(expected.size() == actual.size(), label + ": output sizes differ");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::isfinite(expected[index]) && std::isfinite(actual[index]),
                label + ": non-finite intensity");
        const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
        const double allowed = absolute_tolerance
                               + relative_tolerance * std::abs(static_cast<double>(expected[index]));
        require(error <= allowed, label + ": CPU/CUDA integration tolerance exceeded");
    }
}

beamformer::Weights select_weights(const beamformer::CudaBeamformerKernel kernel,
                                   const beamformer::Dimensions& dims,
                                   const beamformer::Weights& canonical,
                                   const std::vector<beamformer::Vec3>& directions) {
    if (kernel == beamformer::CudaBeamformerKernel::Direct) {
        return canonical;
    }
    return beamformer::generate_tiled_weights(
        dims, beamformer::default_positions(dims.n_ant),
        beamformer::channelized_frequencies(dims.n_freq), directions);
}

void verify_integrated_case(const std::string& label,
                            const beamformer::CudaBeamformerKernel kernel,
                            const beamformer::Dimensions& dims,
                            const beamformer::TemporalIntegrationConfig& integration,
                            const beamformer::PackedVoltage& packed,
                            const std::vector<beamformer::Vec3>& directions) {
    const auto positions = beamformer::default_positions(dims.n_ant);
    const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);
    const auto canonical = beamformer::generate_weights(
        dims, positions, frequencies, directions);
    const auto weights = select_weights(kernel, dims, canonical, directions);
    const auto cpu_raw = beamformer::cpu_beamform_packed_intensity(packed, canonical, dims);
    const auto expected = beamformer::cpu_integrate_intensity(cpu_raw, dims, integration);

    beamformer::CudaBeamformerTimings timings;
    const auto actual = beamformer::cuda_beamform_packed_integrated_intensity(
        packed, weights, dims, integration, &timings, kernel);
    compare_cpu_cuda(label, expected, actual);
    require(timings.kernel_ms >= 0.0 && timings.temporal_integration_ms == 0.0,
            label + ": fused integration must not report a separate stage");
    require(actual.size() == beamformer::integrated_intensity_count(dims, integration),
            label + ": integrated output count is incorrect");
}

bool direct_rejects_10_spectra(const beamformer::Dimensions& dims) {
    try {
        beamformer::CudaBeamformerWorkspace workspace(
            dims, beamformer::CudaBeamformerKernel::Direct,
            beamformer::integration_after_upchan);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    try {
        using namespace beamformer;

        require(integrated_time_count(15360, integration_direct) == 48,
                "15360 / 320 must produce 48 windows");
        require(integrated_time_count(480, integration_after_upchan) == 48,
                "480 / 10 must produce 48 windows");

        const Dimensions direct_dims{320, default_frequency_channels, 32, 1};
        const std::vector<Vec3> broadside{direction_from_lm(0.0F, 0.0F)};
        verify_integrated_case("Direct 320 constant", CudaBeamformerKernel::Direct,
                               direct_dims, integration_direct,
                               make_constant(direct_dims, {7, -8}), broadside);
        require(direct_rejects_10_spectra(direct_dims),
                "Direct must reject 10-spectrum temporal integration");

        const Dimensions tiled_320_dims{321, default_frequency_channels, 32, 8};
        const auto tiled_320_directions = default_beam_grid(tiled_320_dims.n_beams);
        verify_integrated_case(
            "Tiled 320 point source with final partial window",
            CudaBeamformerKernel::Tiled, tiled_320_dims, integration_direct,
            make_point_source(tiled_320_dims, default_positions(tiled_320_dims.n_ant),
                              channelized_frequencies(tiled_320_dims.n_freq),
                              tiled_320_directions[3], 4.0F), tiled_320_directions);

        const Dimensions tiled_10_dims{480, default_frequency_channels, 32, 1};
        verify_integrated_case("Tiled 10 constant", CudaBeamformerKernel::Tiled,
                               tiled_10_dims, integration_after_upchan,
                               make_constant(tiled_10_dims, {7, -8}), broadside);

        const Dimensions tiled_one_hot_dims{1, default_frequency_channels, 32, 3};
        const auto one_hot_directions = default_beam_grid(tiled_one_hot_dims.n_beams);
        verify_integrated_case("Tiled 10 one-hot partial window",
                               CudaBeamformerKernel::Tiled, tiled_one_hot_dims,
                               integration_after_upchan,
                               make_one_hot(tiled_one_hot_dims, 0, 335, 31, {-8, 7}),
                               one_hot_directions);

        const Dimensions tiled_noise_dims{11, default_frequency_channels, 32, 7};
        const auto noise_directions = default_beam_grid(tiled_noise_dims.n_beams);
        verify_integrated_case("Tiled 10 seeded noise with final partial window",
                               CudaBeamformerKernel::Tiled, tiled_noise_dims,
                               integration_after_upchan,
                               make_noise(tiled_noise_dims, 9876), noise_directions);
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_temporal_integration: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
