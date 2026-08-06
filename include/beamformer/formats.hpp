#pragma once

#include "beamformer/config.hpp"
#include "beamformer/complex.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace beamformer {

using PackedVoltage = std::vector<std::uint8_t>;
using ComplexVoltage = std::vector<ComplexFloat>;
using Weights = std::vector<ComplexFloat>;
// The vector is interpreted as [frequency][beam_tile][antenna][local_beam]
// when passed to the Tiled CUDA kernel. The final beam tile is zero-padded.
using TiledWeights = std::vector<ComplexFloat>;
using Intensities = std::vector<float>;

// One validity byte per [time][local_frequency] frame. A zero marks a lost
// frame; the payload remains byte-for-byte RFSoC data and is not rewritten.
using LossMask = std::vector<std::uint8_t>;

struct PackedShard {
    ShardDescriptor descriptor;
    PackedVoltage payload;
    LossMask loss_mask;
};

using PackedShardSet = std::array<PackedShard, frequency_shard_count>;

constexpr std::size_t voltage_sample_count(const Dimensions& dims) {
    return dims.n_time * dims.n_freq * dims.n_ant;
}

constexpr std::size_t packed_voltage_bytes(const Dimensions& dims) {
    return voltage_sample_count(dims);
}

constexpr std::size_t loss_mask_count(const Dimensions& dims) {
    return dims.n_time * dims.n_freq;
}

// little-endian float32 [beam][frequency][antenna][real, imag]
constexpr std::size_t weight_bytes(const Dimensions& dims) {
    return dims.n_beams * dims.n_freq * dims.n_ant * 2 * sizeof(float);
}

constexpr std::size_t tiled_weight_beam_tile = 32;

constexpr std::size_t tiled_weight_beam_tiles(const Dimensions& dims) {
    return (dims.n_beams + tiled_weight_beam_tile - 1) / tiled_weight_beam_tile;
}

// little-endian float32 [frequency][beam_tile][antenna][local_beam][real, imag]
constexpr std::size_t tiled_weight_count(const Dimensions& dims) {
    return dims.n_freq * tiled_weight_beam_tiles(dims) * dims.n_ant
           * tiled_weight_beam_tile;
}

constexpr std::size_t tiled_weight_bytes(const Dimensions& dims) {
    return tiled_weight_count(dims) * 2 * sizeof(float);
}

// little-endian float32 [time][frequency][beam]
constexpr std::size_t intensity_bytes(const Dimensions& dims) {
    return dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
}

} // namespace beamformer
