#include "beamformer/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace beamformer {
namespace {

std::size_t divide_round_up(const std::size_t value, const std::size_t divisor) {
    return value / divisor + (value % divisor == 0 ? 0U : 1U);
}

std::size_t intensity_index(const std::size_t time, const std::size_t frequency,
                            const std::size_t beam, const Dimensions& dims) {
    return (time * dims.n_freq + frequency) * dims.n_beams + beam;
}

int quantize_value(const float value, const Int8QuantizationParameters& parameters) {
    if (!std::isfinite(value)) {
        return quantized_invalid_code;
    }
    if (parameters.scale == 0.0F) {
        return 0;
    }
    const long rounded = std::lrint((value - parameters.offset) / parameters.scale);
    return static_cast<int>(std::max<long>(
        quantized_min_code, std::min<long>(quantized_max_code, rounded)));
}

} // namespace

QuantizationLayout quantization_layout(const Dimensions& integrated_dims) {
    validate_dimensions(integrated_dims);
    return {
        divide_round_up(integrated_dims.n_time, quantization_chunk_time),
        divide_round_up(integrated_dims.n_freq, quantization_chunk_frequency),
        divide_round_up(integrated_dims.n_beams, quantization_chunk_beam),
    };
}

// We operate on a flattened [time][frequency][beam] tensor, so the number of quantization parameters is the number of 1 x 16 x 16 local chunks in the integrated output.
std::size_t quantization_parameter_count(const Dimensions& integrated_dims) {
    const auto layout = quantization_layout(integrated_dims);
    return layout.time_tiles * layout.frequency_tiles * layout.beam_tiles;
}

std::size_t quantization_parameter_index(
    const std::size_t time_tile, const std::size_t frequency_tile,
    const std::size_t beam_tile, const Dimensions& integrated_dims) {
    const auto layout = quantization_layout(integrated_dims);
    if (time_tile >= layout.time_tiles || frequency_tile >= layout.frequency_tiles
        || beam_tile >= layout.beam_tiles) {
        throw std::out_of_range("quantization parameter tile is outside dimensions");
    }
    return (time_tile * layout.frequency_tiles + frequency_tile) * layout.beam_tiles
           + beam_tile;
}

QuantizedIntegratedOutput cpu_quantize_integrated_intensity(
    const IntegratedIntensities& intensity, const Dimensions& integrated_dims) {
    validate_dimensions(integrated_dims);
    const std::size_t count = integrated_dims.n_time * integrated_dims.n_freq
                              * integrated_dims.n_beams;
    if (intensity.size() != count) {
        throw std::invalid_argument("integrated intensity count does not match dimensions");
    }

    QuantizedIntegratedOutput output;
    output.codes.resize(count);
    output.parameters.resize(quantization_parameter_count(integrated_dims));
    const auto layout = quantization_layout(integrated_dims);
    constexpr float output_min = static_cast<float>(quantized_min_code) - 0.5F;
    constexpr float output_max = static_cast<float>(quantized_max_code) + 0.5F;
    constexpr float output_range = output_max - output_min;

    for (std::size_t time_tile = 0; time_tile < layout.time_tiles; ++time_tile) {
        for (std::size_t frequency_tile = 0; frequency_tile < layout.frequency_tiles;
             ++frequency_tile) {
            for (std::size_t beam_tile = 0; beam_tile < layout.beam_tiles; ++beam_tile) {
                float minimum = std::numeric_limits<float>::infinity();
                float maximum = -std::numeric_limits<float>::infinity();
                bool non_finite_chunk = false;
                for (std::size_t local_time = 0; local_time < quantization_chunk_time;
                     ++local_time) {
                    const std::size_t time = time_tile * quantization_chunk_time + local_time;
                    if (time >= integrated_dims.n_time) {
                        continue;
                    }
                    for (std::size_t local_frequency = 0;
                         local_frequency < quantization_chunk_frequency; ++local_frequency) {
                        const std::size_t frequency = frequency_tile
                                                      * quantization_chunk_frequency
                                                      + local_frequency;
                        if (frequency >= integrated_dims.n_freq) {
                            continue;
                        }
                        for (std::size_t local_beam = 0;
                             local_beam < quantization_chunk_beam; ++local_beam) {
                            const std::size_t beam = beam_tile * quantization_chunk_beam
                                                     + local_beam;
                            if (beam >= integrated_dims.n_beams) {
                                continue;
                            }
                            const float value = intensity[intensity_index(
                                time, frequency, beam, integrated_dims)];
                            if (!std::isfinite(value)) {
                                non_finite_chunk = true;
                            } else {
                                minimum = std::min(minimum, value);
                                maximum = std::max(maximum, value);
                            }
                        }
                    }
                }

                Int8QuantizationParameters parameters;
                if (!non_finite_chunk && std::isfinite(minimum) && std::isfinite(maximum)) {
                    parameters.scale = (maximum - minimum) / output_range;
                    parameters.offset = minimum - output_min * parameters.scale;
                    if (!std::isfinite(parameters.offset) || !std::isfinite(parameters.scale)) {
                        parameters = {};
                    }
                }
                output.parameters[quantization_parameter_index(
                    time_tile, frequency_tile, beam_tile, integrated_dims)] = parameters;

                for (std::size_t local_time = 0; local_time < quantization_chunk_time;
                     ++local_time) {
                    const std::size_t time = time_tile * quantization_chunk_time + local_time;
                    if (time >= integrated_dims.n_time) {
                        continue;
                    }
                    for (std::size_t local_frequency = 0;
                         local_frequency < quantization_chunk_frequency; ++local_frequency) {
                        const std::size_t frequency = frequency_tile
                                                      * quantization_chunk_frequency
                                                      + local_frequency;
                        if (frequency >= integrated_dims.n_freq) {
                            continue;
                        }
                        for (std::size_t local_beam = 0;
                             local_beam < quantization_chunk_beam; ++local_beam) {
                            const std::size_t beam = beam_tile * quantization_chunk_beam
                                                     + local_beam;
                            if (beam >= integrated_dims.n_beams) {
                                continue;
                            }
                            const std::size_t index = intensity_index(
                                time, frequency, beam, integrated_dims);
                            output.codes[index] = static_cast<std::int8_t>(
                                quantize_value(intensity[index], parameters));
                        }
                    }
                }
            }
        }
    }
    return output;
}

IntegratedIntensities cpu_dequantize_integrated_intensity(
    const QuantizedIntegratedOutput& quantized, const Dimensions& integrated_dims) {
    validate_dimensions(integrated_dims);
    const std::size_t count = integrated_dims.n_time * integrated_dims.n_freq
                              * integrated_dims.n_beams;
    if (quantized.codes.size() != count
        || quantized.parameters.size() != quantization_parameter_count(integrated_dims)) {
        throw std::invalid_argument("quantized output does not match dimensions");
    }

    IntegratedIntensities output(count);
    for (std::size_t time = 0; time < integrated_dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < integrated_dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < integrated_dims.n_beams; ++beam) {
                const std::size_t index = intensity_index(time, frequency, beam, integrated_dims);
                const std::int8_t code = quantized.codes[index];
                if (code == quantized_invalid_code) {
                    output[index] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                const auto& parameters = quantized.parameters[quantization_parameter_index(
                    time / quantization_chunk_time,
                    frequency / quantization_chunk_frequency,
                    beam / quantization_chunk_beam, integrated_dims)];
                output[index] = parameters.offset
                                + parameters.scale * static_cast<float>(code);
            }
        }
    }
    return output;
}

QuantizationStats quantization_stats(
    const IntegratedIntensities& reference,
    const QuantizedIntegratedOutput& quantized, const Dimensions& integrated_dims) {
    const auto reconstructed = cpu_dequantize_integrated_intensity(quantized, integrated_dims);
    if (reference.size() != reconstructed.size()) {
        throw std::invalid_argument("reference intensity count does not match dimensions");
    }

    QuantizationStats stats;
    double absolute_sum = 0.0;
    double squared_error_sum = 0.0;
    double squared_reference_sum = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const std::int8_t code = quantized.codes[index];
        if (!std::isfinite(reference[index]) || code == quantized_invalid_code) {
            ++stats.invalid_samples;
            continue;
        }
        const double error = std::abs(static_cast<double>(reconstructed[index]) - reference[index]);
        stats.maximum_absolute_error = std::max(stats.maximum_absolute_error, error);
        absolute_sum += error;
        squared_error_sum += error * error;
        squared_reference_sum += static_cast<double>(reference[index]) * reference[index];
        ++stats.finite_samples;
        if (code == quantized_min_code) {
            ++stats.minimum_code_samples;
        }
        if (code == quantized_max_code) {
            ++stats.maximum_code_samples;
        }
    }
    if (stats.finite_samples > 0) {
        stats.mean_absolute_error = absolute_sum / stats.finite_samples;
        stats.normalized_rmse = squared_reference_sum > 0.0
                                    ? std::sqrt(squared_error_sum / squared_reference_sum)
                                    : std::sqrt(squared_error_sum / stats.finite_samples);
    }
    return stats;
}

void validate_quantized_output_metadata(const QuantizedOutputMetadata& metadata) {
    validate_dimensions(metadata.output_dims);
    validate_temporal_config(metadata.temporal_integration);
    validate_shard_descriptor(metadata.shard);
    if (metadata.input_n_time == 0) {
        throw std::invalid_argument("quantized metadata input_n_time must be positive");
    }
    if (integrated_time_count(metadata.input_n_time, metadata.temporal_integration)
        != metadata.output_dims.n_time) {
        throw std::invalid_argument("quantized metadata time dimensions are inconsistent");
    }
    if (metadata.parameters_file.empty()) {
        throw std::invalid_argument("quantized metadata parameters_file must not be empty");
    }
}

} // namespace beamformer
