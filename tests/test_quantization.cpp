#include "beamformer/config.hpp"
#include "beamformer/io.hpp"
#include "beamformer/quantization.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

bool close(const float actual, const float expected, const float tolerance = 1.0e-5F) {
    return std::abs(actual - expected) <= tolerance;
}

std::size_t index(const std::size_t time, const std::size_t frequency,
                  const std::size_t beam, const beamformer::Dimensions& dims) {
    return (time * dims.n_freq + frequency) * dims.n_beams + beam;
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace beamformer;

    const Dimensions dims{2, default_frequency_channels, 32, 17};
    Intensities input(dims.n_time * dims.n_freq * dims.n_beams);
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                input[index(time, frequency, beam, dims)] = static_cast<float>(
                    1000 * time + 10 * frequency + beam);
            }
        }
    }

    const auto layout = quantization_layout(dims);
    assert(layout.time_tiles == 2);
    assert(layout.frequency_tiles == 21);
    assert(layout.beam_tiles == 2);
    assert(quantization_parameter_count(dims) == 84);
    assert(quantized_intensity_bytes(dims) == input.size());

    const auto quantized = cpu_quantize_integrated_intensity(input, dims);
    assert(quantized.codes.size() == input.size());
    assert(quantized.parameters.size() == 84);
    assert(quantized.codes[index(0, 0, 0, dims)] == quantized_min_code);
    assert(quantized.codes[index(0, 15, 15, dims)] == quantized_max_code);
    assert(quantized.codes[index(0, 0, 16, dims)] == quantized_min_code);
    assert(quantized.codes[index(0, 15, 16, dims)] == quantized_max_code);

    const auto reconstructed = cpu_dequantize_integrated_intensity(quantized, dims);
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                const auto parameter = quantized.parameters[quantization_parameter_index(
                    time, frequency / quantization_chunk_frequency,
                    beam / quantization_chunk_beam, dims)];
                const float error = std::abs(reconstructed[index(time, frequency, beam, dims)]
                                             - input[index(time, frequency, beam, dims)]);
                if (error > parameter.scale * 0.5F + 5.0e-4F) {
                    throw std::runtime_error(
                        "reconstruction exceeds half a quantization bin: error="
                        + std::to_string(error) + " scale="
                        + std::to_string(parameter.scale));
                }
            }
        }
    }
    const auto stats = quantization_stats(input, quantized, dims);
    assert(stats.finite_samples == input.size());
    assert(stats.invalid_samples == 0);
    assert(stats.minimum_code_samples > 0);
    assert(stats.maximum_code_samples > 0);

    const Dimensions constant_dims{1, default_frequency_channels, 32, 1};
    Intensities constant(constant_dims.n_freq, 42.0F);
    const auto constant_quantized = cpu_quantize_integrated_intensity(constant, constant_dims);
    assert(constant_quantized.parameters.size() == 21);
    for (const auto& parameter : constant_quantized.parameters) {
        assert(parameter.scale == 0.0F);
        assert(parameter.offset == 42.0F);
    }
    for (const auto code : constant_quantized.codes) {
        assert(code == 0);
    }
    for (const auto value : cpu_dequantize_integrated_intensity(
             constant_quantized, constant_dims)) {
        assert(value == 42.0F);
    }

    Intensities non_finite = constant;
    non_finite[0] = std::numeric_limits<float>::quiet_NaN();
    const auto non_finite_quantized = cpu_quantize_integrated_intensity(
        non_finite, constant_dims);
    assert(non_finite_quantized.codes[0] == quantized_invalid_code);
    assert(non_finite_quantized.parameters[0].offset == 0.0F);
    assert(non_finite_quantized.parameters[0].scale == 0.0F);
    assert(std::isnan(cpu_dequantize_integrated_intensity(
        non_finite_quantized, constant_dims)[0]));

    const auto temporary_directory = std::filesystem::temp_directory_path();
    const auto codes_path = temporary_directory / "beamformer_quantization_codes.bin";
    const auto parameters_path = temporary_directory / "beamformer_quantization_params.bin";
    const auto metadata_path = temporary_directory / "beamformer_quantization_metadata.txt";
    write_quantized_intensities(codes_path, quantized.codes, dims);
    write_quantization_parameters(parameters_path, quantized.parameters, dims);
    assert(read_quantized_intensities(codes_path, dims) == quantized.codes);
    const auto loaded_parameters = read_quantization_parameters(parameters_path, dims);
    assert(loaded_parameters.size() == quantized.parameters.size());
    for (std::size_t item = 0; item < loaded_parameters.size(); ++item) {
        assert(close(loaded_parameters[item].offset, quantized.parameters[item].offset));
        assert(close(loaded_parameters[item].scale, quantized.parameters[item].scale));
    }

    const Dimensions metadata_dims{1, default_frequency_channels, 32, 17};
    const QuantizedOutputMetadata metadata{
        metadata_dims, 320, integration_direct, default_shard_descriptors()[1],
        parameters_path.filename().string(),
    };
    write_quantized_output_metadata(metadata_path, metadata);
    const auto loaded_metadata = read_quantized_output_metadata(metadata_path);
    assert(loaded_metadata.output_dims.n_time == metadata_dims.n_time);
    assert(loaded_metadata.temporal_integration.integration_spectra == 320);
    assert(loaded_metadata.shard.shard_id == 1);
    assert(loaded_metadata.shard.absolute_frequency_start == 336);
    assert(loaded_metadata.parameters_file == parameters_path.filename().string());
    assert(throws_invalid_argument([&] {
        validate_quantized_output_metadata({metadata_dims, 321, integration_direct,
                                            default_shard_descriptors()[0], "params.bin"});
    }));
    return 0;
}
