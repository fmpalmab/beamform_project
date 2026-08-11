#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/quantization.hpp"
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

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare_quantized(const std::string& label,
                       const beamformer::QuantizedIntegratedOutput& expected,
                       const beamformer::QuantizedIntegratedOutput& actual) {
    require(expected.codes == actual.codes, label + ": int8 codes differ from CPU reference");
    require(expected.parameters.size() == actual.parameters.size(),
            label + ": parameter counts differ");
    for (std::size_t index = 0; index < expected.parameters.size(); ++index) {
        const float offset_error =
            std::abs(expected.parameters[index].offset - actual.parameters[index].offset);
        const float offset_allowed = 1.0e-3F
                                     + 1.0e-6F * std::abs(expected.parameters[index].offset);
        require(offset_error <= offset_allowed,
                label + ": offset differs from CPU reference");
        const float scale_error =
            std::abs(expected.parameters[index].scale - actual.parameters[index].scale);
        const float scale_allowed = 1.0e-5F
                                    + 1.0e-6F * std::abs(expected.parameters[index].scale);
        require(scale_error <= scale_allowed,
                label + ": scale differs from CPU reference");
    }
}

beamformer::Weights kernel_weights(const beamformer::CudaBeamformerKernel kernel,
                                   const beamformer::Dimensions& dims,
                                   const std::vector<beamformer::Vec3>& directions,
                                   const beamformer::Weights& canonical) {
    if (kernel == beamformer::CudaBeamformerKernel::Direct) {
        return canonical;
    }
    return beamformer::generate_tiled_weights(
        dims, beamformer::default_positions(dims.n_ant),
        beamformer::channelized_frequencies(dims.n_freq), directions);
}

void verify_case(const std::string& label,
                 const beamformer::CudaBeamformerKernel kernel,
                 const beamformer::Dimensions& dims,
                 const beamformer::TemporalIntegrationConfig& integration,
                 const beamformer::PackedVoltage& packed,
                 const std::vector<beamformer::Vec3>& directions) {
    const auto positions = beamformer::default_positions(dims.n_ant);
    const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);
    const auto canonical = beamformer::generate_weights(
        dims, positions, frequencies, directions);
    const auto selected_weights = kernel_weights(kernel, dims, directions, canonical);
    const auto cpu_raw = beamformer::cpu_beamform_packed_intensity(packed, canonical, dims);
    const auto cpu_integrated = beamformer::cpu_integrate_intensity(cpu_raw, dims, integration);
    const beamformer::Dimensions integrated_dims{
        beamformer::integrated_time_count(dims.n_time, integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    const auto gpu_integrated = beamformer::cuda_beamform_packed_integrated_intensity(
        packed, selected_weights, dims, integration, nullptr, kernel);
    const auto expected = beamformer::cpu_quantize_integrated_intensity(
        gpu_integrated, integrated_dims);

    beamformer::CudaBeamformerTimings timings;
    const auto actual = beamformer::cuda_beamform_packed_quantized_integrated_intensity(
        packed, selected_weights, dims, integration, &timings, kernel);
    compare_quantized(label, expected, actual);
    const auto stats = beamformer::quantization_stats(cpu_integrated, actual, integrated_dims);
    require(stats.finite_samples == cpu_integrated.size() && stats.invalid_samples == 0,
            label + ": unexpected invalid quantized values");
    require(timings.quantization_ms >= 0.0, label + ": invalid quantization timing");
    require(actual.codes.size() == beamformer::quantized_intensity_bytes(integrated_dims),
            label + ": int8 output byte count is incorrect");
    require(actual.parameters.size()
                == beamformer::quantization_parameter_count(integrated_dims),
            label + ": parameter count is incorrect");
}

} // namespace

int main() {
    try {
        using namespace beamformer;

        const Dimensions direct_dims{320, default_frequency_channels, 32, 1};
        const std::vector<Vec3> broadside{direction_from_lm(0.0F, 0.0F)};
        verify_case("Direct 320 constant", CudaBeamformerKernel::Direct, direct_dims,
                    integration_direct, make_constant(direct_dims, {7, -8}), broadside);

        const Dimensions tiled_point_dims{321, default_frequency_channels, 32, 8};
        const auto point_directions = default_beam_grid(tiled_point_dims.n_beams);
        verify_case("Tiled 320 point source", CudaBeamformerKernel::Tiled, tiled_point_dims,
                    integration_direct,
                    make_point_source(tiled_point_dims,
                                      default_positions(tiled_point_dims.n_ant),
                                      channelized_frequencies(tiled_point_dims.n_freq),
                                      point_directions[3], 4.0F), point_directions);

        const Dimensions tiled_noise_dims{11, default_frequency_channels, 32, 7};
        const auto noise_directions = default_beam_grid(tiled_noise_dims.n_beams);
        verify_case("Tiled 10 seeded noise", CudaBeamformerKernel::Tiled, tiled_noise_dims,
                    integration_after_upchan, make_noise(tiled_noise_dims, 9876),
                    noise_directions);
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_quantization: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
