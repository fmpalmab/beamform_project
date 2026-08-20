#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace beamformer {

// A production input is one local frequency shard, not the concatenated band.
// Each NIC contributes 336 channels; the two shards cover the 672-channel band.
inline constexpr std::size_t local_frequency_channels = 336;
inline constexpr std::size_t frequency_shard_count = 2;
inline constexpr std::size_t full_band_frequency_channels =
    local_frequency_channels * frequency_shard_count;
// Compatibility name: n_freq in Dimensions means local frequency count.
inline constexpr std::size_t default_frequency_channels = local_frequency_channels;
inline constexpr std::size_t maximum_beams = 128;
inline constexpr std::size_t rfsoc_channels_per_subband = 168;
inline constexpr std::size_t rfsoc_subbands_per_nic = 2;
inline constexpr std::size_t rfsoc_channels_per_nic =
    rfsoc_channels_per_subband * rfsoc_subbands_per_nic;
inline constexpr std::size_t rfsoc_elements_per_device = 32;
inline constexpr float default_frequency_start_hz = 300'000'000.0F;
inline constexpr float default_channel_width_hz = 300'000.0F;
inline constexpr float beam_grid_design_frequency_hz = 400'000'000.0F;
inline constexpr float default_spacing_m = 0.6F;

struct ShardDescriptor {
    // Identity is deliberately explicit: shard buffers must remain independent.
    std::size_t shard_id = 0;
    std::size_t shard_count = frequency_shard_count;
    std::size_t local_frequency_count = local_frequency_channels;
    std::size_t absolute_frequency_start = 0;
    // Timestamp units are supplied by the producer; step must be positive.
    std::uint64_t timestamp_start = 0;
    std::uint64_t timestamp_step = 1;
    // The loss mask is external metadata and has one identity per shard.
    std::uint64_t loss_mask_id = 0;
    bool loss_mask_independent = true;
};

inline void validate_shard_descriptor(const ShardDescriptor& shard) {
    if (shard.shard_count != frequency_shard_count) {
        throw std::invalid_argument("shard_count must be exactly 2");
    }
    if (shard.shard_id >= shard.shard_count) {
        throw std::invalid_argument("shard_id is outside the two-shard band");
    }
    if (shard.local_frequency_count != local_frequency_channels) {
        throw std::invalid_argument("local_frequency_count must be exactly 336");
    }
    if (shard.absolute_frequency_start >
        full_band_frequency_channels - local_frequency_channels) {
        throw std::invalid_argument("absolute frequency start is outside the full band");
    }
    if (shard.timestamp_step == 0) {
        throw std::invalid_argument("timestamp_step must be positive");
    }
    if (!shard.loss_mask_independent) {
        throw std::invalid_argument("loss masks must be independent per shard");
    }
}

inline constexpr std::array<ShardDescriptor, frequency_shard_count>
default_shard_descriptors() {
    return {{
        ShardDescriptor{0, frequency_shard_count, local_frequency_channels, 0, 0, 1, 0, true},
        ShardDescriptor{1, frequency_shard_count, local_frequency_channels,
                        local_frequency_channels, 0, 1, 1, true},
    }};
}

struct Dimensions {
    std::size_t n_time = 15360;
    std::size_t n_freq = default_frequency_channels;
    std::size_t n_ant = 64;
    std::size_t n_beams = 64;
};

inline void validate_dimensions(const Dimensions& dims) {
    if (dims.n_time == 0) {
        throw std::invalid_argument("n_time must be positive");
    }
    if (dims.n_freq != default_frequency_channels) {
        throw std::invalid_argument("the local-shard contract requires exactly 336 frequency channels");
    }
    if (dims.n_ant != 32 && dims.n_ant != 64 && dims.n_ant != 128 && dims.n_ant != 256) {
        throw std::invalid_argument("n_ant must be 32, 64, 128, or 256");
    }
    if (dims.n_beams == 0 || dims.n_beams > maximum_beams) {
        throw std::invalid_argument("n_beams must be between 1 and 128");
    }
}

} // namespace beamformer
