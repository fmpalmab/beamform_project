#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/int4.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace beamformer {

PackedVoltage make_one_hot(const Dimensions& dims, std::size_t active_time,
                           std::size_t active_frequency, std::size_t active_element,
                           ComplexInt4 value = ComplexInt4{3, -2});

PackedVoltage make_constant(const Dimensions& dims,
                            ComplexInt4 value = ComplexInt4{1, 0});

PackedVoltage make_point_source(const Dimensions& dims,
                                const std::vector<Vec3>& positions_m,
                                const std::vector<float>& frequencies_hz,
                                const Vec3& source_direction,
                                float amplitude = 4.0F);

PackedVoltage make_noise(const Dimensions& dims, std::uint32_t seed = 1);

LossMask make_loss_mask(const Dimensions& dims, std::uint32_t seed = 1,
                        float loss_probability = 0.0F);

void validate_packed_shard(const PackedShard& shard, const Dimensions& dims);
void validate_packed_shards(const PackedShardSet& shards, const Dimensions& dims);

PackedShardSet make_two_shard_one_hot(
    const Dimensions& dims, std::size_t active_time,
    const std::array<std::size_t, frequency_shard_count>& active_frequencies,
    const std::array<std::size_t, frequency_shard_count>& active_elements,
    ComplexInt4 value = ComplexInt4{3, -2},
    std::array<std::uint32_t, frequency_shard_count> loss_seeds = {1, 2},
    float loss_probability = 0.0F);

PackedShardSet make_two_shard_constant(
    const Dimensions& dims, ComplexInt4 value = ComplexInt4{1, 0},
    std::array<std::uint32_t, frequency_shard_count> loss_seeds = {1, 2},
    float loss_probability = 0.0F);

PackedShardSet make_two_shard_noise(
    const Dimensions& dims, std::uint32_t seed = 1,
    float loss_probability = 0.0F);

PackedShardSet make_two_shard_point_source(
    const Dimensions& dims, const std::vector<Vec3>& positions_m,
    const Vec3& source_direction, float amplitude = 4.0F,
    std::array<std::uint32_t, frequency_shard_count> loss_seeds = {1, 2},
    float loss_probability = 0.0F);

} // namespace beamformer
