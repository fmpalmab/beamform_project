#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_offline_runner.hpp"
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

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare_float_output(const beamformer::Intensities& expected,
                          const beamformer::Intensities& actual,
                          const std::string& label) {
    require(expected.size() == actual.size(), label + ": output sizes differ");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = std::abs(static_cast<double>(actual[index])
                                      - static_cast<double>(expected[index]));
        const double allowed = 1.0e-3
                               + 1.0e-5 * std::abs(static_cast<double>(expected[index]));
        require(error <= allowed, label + ": output exceeds CPU tolerance");
    }
}

void compare_quantized(const beamformer::QuantizedIntegratedOutput& expected,
                       const beamformer::QuantizedIntegratedOutput& actual) {
    require(expected.codes == actual.codes, "offline int8 codes differ");
    require(expected.parameters.size() == actual.parameters.size(),
            "offline int8 parameter sizes differ");
    for (std::size_t index = 0; index < expected.parameters.size(); ++index) {
        require(std::abs(expected.parameters[index].offset
                         - actual.parameters[index].offset) <= 1.0e-3F
                    + 1.0e-6F * std::abs(expected.parameters[index].offset),
                "offline int8 offset differs");
        require(std::abs(expected.parameters[index].scale
                         - actual.parameters[index].scale) <= 1.0e-5F
                    + 1.0e-6F * std::abs(expected.parameters[index].scale),
                "offline int8 scale differs");
    }
}

} // namespace

int main() {
    try {
        using namespace beamformer;
        const auto shard = default_shard_descriptors()[1];

        const Dimensions direct_dims{3, default_frequency_channels, 32, 5};
        const auto direct_directions = default_beam_grid(direct_dims.n_beams);
        const auto direct_packed = make_noise(direct_dims, 1001);
        const auto direct_weights = generate_weights(
            direct_dims, default_positions(direct_dims.n_ant),
            channelized_frequencies(direct_dims.n_freq), direct_directions);
        const auto direct_expected = cpu_beamform_packed_intensity(
            direct_packed, direct_weights, direct_dims);
        CudaOfflineFrameRunner direct_runner(
            direct_dims, CudaBeamformerKernel::Direct);
        const auto direct_result = direct_runner.run(
            direct_packed, direct_weights, direct_dims, 17, shard);
        compare_float_output(direct_expected, direct_result.float32_output,
                             "offline Direct float32");
        require(direct_result.output_dims.n_time == direct_dims.n_time,
                "offline Direct output time dimension changed");
        require(direct_result.timings.host_to_device_ms >= 0.0
                    && direct_result.timings.device_to_host_ms >= 0.0,
                "offline transfer timings must be non-negative");

        const Dimensions integrated_dims{11, default_frequency_channels, 32, 7};
        const auto integrated_directions = default_beam_grid(integrated_dims.n_beams);
        const auto integrated_packed = make_noise(integrated_dims, 2002);
        const auto canonical_weights = generate_weights(
            integrated_dims, default_positions(integrated_dims.n_ant),
            channelized_frequencies(integrated_dims.n_freq), integrated_directions);
        const auto tiled_weights = generate_tiled_weights(
            integrated_dims, default_positions(integrated_dims.n_ant),
            channelized_frequencies(integrated_dims.n_freq), integrated_directions);
        const auto raw_reference = cpu_beamform_packed_intensity(
            integrated_packed, canonical_weights, integrated_dims);
        const auto integrated_expected = cpu_integrate_intensity(
            raw_reference, integrated_dims, integration_after_upchan);
        const Dimensions integrated_output_dims{
            integrated_time_count(integrated_dims.n_time, integration_after_upchan),
            integrated_dims.n_freq, integrated_dims.n_ant, integrated_dims.n_beams};
        CudaOfflineFrameRunner integrated_runner(
            integrated_dims, CudaBeamformerKernel::Tiled, integration_after_upchan);
        const auto integrated_result = integrated_runner.run(
            integrated_packed, tiled_weights, integrated_dims, 18, shard);
        compare_float_output(integrated_expected, integrated_result.float32_output,
                             "offline Tiled integrated float32");
        require(integrated_result.output_dims.n_time == integrated_output_dims.n_time,
                "offline integrated output window count is incorrect");

        const Dimensions quantized_dims{321, default_frequency_channels, 32, 8};
        const auto quantized_directions = default_beam_grid(quantized_dims.n_beams);
        const auto quantized_packed = make_point_source(
            quantized_dims, default_positions(quantized_dims.n_ant),
            channelized_frequencies(quantized_dims.n_freq), quantized_directions[3], 4.0F);
        const auto quantized_weights = generate_tiled_weights(
            quantized_dims, default_positions(quantized_dims.n_ant),
            channelized_frequencies(quantized_dims.n_freq), quantized_directions);
        const auto expected_quantized =
            cuda_beamform_packed_quantized_integrated_intensity(
                quantized_packed, quantized_weights, quantized_dims,
                integration_direct, nullptr, CudaBeamformerKernel::Tiled);
        CudaOfflineFrameRunner quantized_runner(
            quantized_dims, CudaBeamformerKernel::Tiled, integration_direct,
            CudaBeamformerOutput::QuantizedInt8);
        const auto quantized_result = quantized_runner.run(
            quantized_packed, quantized_weights, quantized_dims, 19, shard);
        compare_quantized(expected_quantized, quantized_result.quantized_output);
        require(quantized_result.float32_output.empty(),
                "offline int8 runner unexpectedly returned float32 output");
        require(quantized_result.quantized_output.codes.size()
                    == quantized_intensity_bytes(quantized_result.output_dims),
                "offline int8 output byte count is incorrect");

        bool rejected_input_size = false;
        try {
            auto invalid_input = direct_packed;
            invalid_input.pop_back();
            direct_runner.run(invalid_input, direct_weights, direct_dims, 20, shard);
        } catch (const std::invalid_argument&) {
            rejected_input_size = true;
        }
        require(rejected_input_size, "offline runner accepted an undersized input");
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_offline_runner: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
