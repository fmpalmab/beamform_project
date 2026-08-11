#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/quantization.hpp"

#include <filesystem>

namespace beamformer {

PackedVoltage read_packed_voltage(const std::filesystem::path& path,
                                  const Dimensions& dims);
void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims);

void write_packed_shard(const std::filesystem::path& payload_path,
                        const std::filesystem::path& metadata_path,
                        const std::filesystem::path& loss_mask_path,
                        const PackedShard& shard, const Dimensions& dims);
PackedShard read_packed_shard(const std::filesystem::path& payload_path,
                              const std::filesystem::path& metadata_path,
                              const std::filesystem::path& loss_mask_path,
                              const Dimensions& dims);

Weights read_weights(const std::filesystem::path& path, const Dimensions& dims);
void write_weights(const std::filesystem::path& path, const Weights& weights,
                   const Dimensions& dims);
TiledWeights read_tiled_weights(const std::filesystem::path& path,
                                const Dimensions& dims);
void write_tiled_weights(const std::filesystem::path& path,
                         const TiledWeights& weights, const Dimensions& dims);

void write_intensities(const std::filesystem::path& path,
                       const Intensities& intensities,
                       const Dimensions& dims);

QuantizedIntensities read_quantized_intensities(const std::filesystem::path& path,
                                                const Dimensions& dims);
void write_quantized_intensities(const std::filesystem::path& path,
                                 const QuantizedIntensities& intensities,
                                 const Dimensions& dims);
std::vector<Int8QuantizationParameters> read_quantization_parameters(
    const std::filesystem::path& path, const Dimensions& integrated_dims);
void write_quantization_parameters(
    const std::filesystem::path& path,
    const std::vector<Int8QuantizationParameters>& parameters,
    const Dimensions& integrated_dims);
void write_quantized_output_metadata(const std::filesystem::path& path,
                                     const QuantizedOutputMetadata& metadata);
QuantizedOutputMetadata read_quantized_output_metadata(const std::filesystem::path& path);

} // namespace beamformer
