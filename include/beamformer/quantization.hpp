#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/temporal_integration.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace beamformer {

// Adapted from CHIME's 8-bit quantizer: one warp processes 256 integrated
// intensities and writes one local affine reconstruction pair.
inline constexpr std::size_t quantization_chunk_time = 1;
inline constexpr std::size_t quantization_chunk_frequency = 16;
inline constexpr std::size_t quantization_chunk_beam = 16;
inline constexpr std::int8_t quantized_invalid_code = -128;
inline constexpr std::int8_t quantized_min_code = -127;
inline constexpr std::int8_t quantized_max_code = 127;

struct Int8QuantizationParameters {
    float offset = 0.0F;
    float scale = 0.0F;
};

static_assert(sizeof(Int8QuantizationParameters) == 2 * sizeof(float));

struct QuantizationLayout {
    std::size_t time_tiles = 0;
    std::size_t frequency_tiles = 0;
    std::size_t beam_tiles = 0;
};

QuantizationLayout quantization_layout(const Dimensions& integrated_dims);
std::size_t quantization_parameter_count(const Dimensions& integrated_dims);

// Flattened order is [time_tile][frequency_tile][beam_tile]. Each parameter
// maps its corresponding 1 x 16 x 16 output chunk.
std::size_t quantization_parameter_index(
    std::size_t time_tile, std::size_t frequency_tile, std::size_t beam_tile,
    const Dimensions& integrated_dims);

struct QuantizedIntegratedOutput {
    QuantizedIntensities codes;
    std::vector<Int8QuantizationParameters> parameters;
};

struct QuantizationStats {
    std::size_t finite_samples = 0;
    std::size_t invalid_samples = 0;
    std::size_t minimum_code_samples = 0;
    std::size_t maximum_code_samples = 0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double normalized_rmse = 0.0;
};

// The CPU reference is deliberately straightforward. It mirrors the CUDA
// chunk contract: non-finite inputs become quantized_invalid_code. A
// degenerate chunk gets scale=0 and offset=the constant value, so finite
// values map to code zero and reconstruct exactly.
QuantizedIntegratedOutput cpu_quantize_integrated_intensity(
    const IntegratedIntensities& intensity, const Dimensions& integrated_dims);

IntegratedIntensities cpu_dequantize_integrated_intensity(
    const QuantizedIntegratedOutput& quantized, const Dimensions& integrated_dims);

QuantizationStats quantization_stats(
    const IntegratedIntensities& reference,
    const QuantizedIntegratedOutput& quantized,
    const Dimensions& integrated_dims);

struct QuantizedOutputMetadata {
    Dimensions output_dims;
    std::size_t input_n_time = 0;
    TemporalIntegrationConfig temporal_integration{0};
    ShardDescriptor shard;
    std::string parameters_file;
};

void validate_quantized_output_metadata(const QuantizedOutputMetadata& metadata);

} // namespace beamformer
