#include "beamformer/synthetic_data.hpp"

#include "beamformer/formats.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/physics.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace beamformer {
namespace {

void validate_point_source_inputs(const Dimensions& dims,
                                  const std::vector<Vec3>& positions_m,
                                  const std::vector<float>& frequencies_hz,
                                  const Vec3& source_direction,
                                  const float amplitude) {
    validate_dimensions(dims);
    if (positions_m.size() != dims.n_ant) {
        throw std::invalid_argument("position count must match n_ant");
    }
    if (frequencies_hz.size() != dims.n_freq) {
        throw std::invalid_argument("frequency count must match n_freq");
    }
    if (!std::isfinite(amplitude) || amplitude <= 0.0F || amplitude > 7.0F) {
        throw std::invalid_argument("point-source amplitude must be in (0, 7]");
    }

    double norm_squared = 0.0;
    for (const float component : source_direction) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument("source direction must be finite");
        }
        norm_squared += static_cast<double>(component) * component;
    }
    if (std::abs(norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument("source direction must be a unit vector");
    }
    for (const float frequency : frequencies_hz) {
        if (!std::isfinite(frequency) || frequency <= 0.0F) {
            throw std::invalid_argument("frequencies must be positive and finite");
        }
    }
}

} // namespace

PackedVoltage make_one_hot(const Dimensions& dims, const std::size_t active_time,
                           const std::size_t active_frequency,
                           const std::size_t active_element, const ComplexInt4 value) {
    validate_dimensions(dims);
    if (active_time >= dims.n_time || active_frequency >= dims.n_freq
        || active_element >= dims.n_ant) {
        throw std::out_of_range("one-hot index is outside the voltage dimensions");
    }

    PackedVoltage voltage(voltage_sample_count(dims), pack_complex_int4(0, 0));
    voltage[voltage_index(active_time, active_frequency, active_element, dims)] =
        pack_complex_int4(value.real, value.imag);
    return voltage;
}

PackedVoltage make_constant(const Dimensions& dims, const ComplexInt4 value) {
    validate_dimensions(dims);
    return PackedVoltage(voltage_sample_count(dims),
                         pack_complex_int4(value.real, value.imag));
}

PackedVoltage make_point_source(const Dimensions& dims,
                                const std::vector<Vec3>& positions_m,
                                const std::vector<float>& frequencies_hz,
                                const Vec3& source_direction,
                                const float amplitude) {
    validate_point_source_inputs(dims, positions_m, frequencies_hz, source_direction,
                                 amplitude);

    const std::size_t spectrum_size = dims.n_freq * dims.n_ant;
    PackedVoltage spectrum(spectrum_size);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        const double wave_number = two_pi * static_cast<double>(frequencies_hz[frequency])
                                   / speed_of_light_m_per_s;
        for (std::size_t element = 0; element < dims.n_ant; ++element) {
            const auto& position = positions_m[element];
            const double delay_m =
                static_cast<double>(position[0]) * source_direction[0]
                + static_cast<double>(position[1]) * source_direction[1]
                + static_cast<double>(position[2]) * source_direction[2];
            const double phase = wave_number * delay_m;
            const auto real = static_cast<std::int8_t>(
                std::lround(static_cast<double>(amplitude) * std::cos(phase)));
            const auto imag = static_cast<std::int8_t>(
                std::lround(-static_cast<double>(amplitude) * std::sin(phase)));
            spectrum[frequency * dims.n_ant + element] = pack_complex_int4(real, imag);
        }
    }

    PackedVoltage voltage(voltage_sample_count(dims));
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        std::copy(spectrum.begin(), spectrum.end(),
                  voltage.begin() + static_cast<std::ptrdiff_t>(time * spectrum_size));
    }
    return voltage;
}

PackedVoltage make_noise(const Dimensions& dims, const std::uint32_t seed) {
    validate_dimensions(dims);
    PackedVoltage voltage(voltage_sample_count(dims));
    std::mt19937 random(seed);
    for (auto& packed : voltage) {
        const std::uint32_t bits = random();
        const auto real = static_cast<std::int8_t>(static_cast<int>(bits & 0xFFFFU) % 15 - 7);
        const auto imag =
            static_cast<std::int8_t>(static_cast<int>((bits >> 16) & 0xFFFFU) % 15 - 7);
        packed = pack_complex_int4(real, imag);
    }
    return voltage;
}


LossMask make_loss_mask(const Dimensions& dims, const std::uint32_t seed,
                        const float loss_probability) {
    validate_dimensions(dims);
    if (!std::isfinite(loss_probability) || loss_probability < 0.0F
        || loss_probability > 1.0F) {
        throw std::invalid_argument("loss probability must be in [0, 1]");
    }

    LossMask mask(loss_mask_count(dims), 1);
    std::mt19937 random(seed);
    std::bernoulli_distribution lost(loss_probability);
    for (auto& valid : mask) {
        valid = lost(random) ? 0U : 1U;
    }
    return mask;
}

void validate_packed_shard(const PackedShard& shard, const Dimensions& dims) {
    validate_dimensions(dims);
    validate_shard_descriptor(shard.descriptor);
    if (shard.payload.size() != packed_voltage_bytes(dims)) {
        throw std::invalid_argument("shard payload size does not match dimensions");
    }
    if (shard.loss_mask.size() != loss_mask_count(dims)) {
        throw std::invalid_argument("loss mask size does not match dimensions");
    }
    for (const auto valid : shard.loss_mask) {
        if (valid > 1U) {
            throw std::invalid_argument("loss mask entries must be 0 or 1");
        }
    }
}

void validate_packed_shards(const PackedShardSet& shards, const Dimensions& dims) {
    validate_dimensions(dims);
    for (std::size_t index = 0; index < frequency_shard_count; ++index) {
        validate_packed_shard(shards[index], dims);
        if (shards[index].descriptor.shard_id != index) {
            throw std::invalid_argument("shard IDs must identify their array slot");
        }
    }
    if (shards[0].descriptor.absolute_frequency_start
        == shards[1].descriptor.absolute_frequency_start) {
        throw std::invalid_argument("the two shards must have distinct frequency origins");
    }
    if (shards[0].descriptor.loss_mask_id
        == shards[1].descriptor.loss_mask_id) {
        throw std::invalid_argument("the two shards must have independent loss masks");
    }
}

namespace {

PackedShardSet make_shard_set(
    const Dimensions& dims,
    const std::array<PackedVoltage, frequency_shard_count>& payloads,
    const std::array<std::uint32_t, frequency_shard_count>& loss_seeds,
    const float loss_probability) {
    const auto descriptors = default_shard_descriptors();
    PackedShardSet shards;
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        shards[shard_id] = PackedShard{
            descriptors[shard_id], payloads[shard_id],
            make_loss_mask(dims, loss_seeds[shard_id], loss_probability),
        };
    }
    validate_packed_shards(shards, dims);
    return shards;
}

} // namespace

PackedShardSet make_two_shard_one_hot(
    const Dimensions& dims, const std::size_t active_time,
    const std::array<std::size_t, frequency_shard_count>& active_frequencies,
    const std::array<std::size_t, frequency_shard_count>& active_elements,
    const ComplexInt4 value,
    const std::array<std::uint32_t, frequency_shard_count> loss_seeds,
    const float loss_probability) {
    validate_dimensions(dims);
    std::array<PackedVoltage, frequency_shard_count> payloads;
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        payloads[shard_id] = make_one_hot(
            dims, active_time, active_frequencies[shard_id],
            active_elements[shard_id], value);
    }
    return make_shard_set(dims, payloads, loss_seeds, loss_probability);
}

PackedShardSet make_two_shard_constant(
    const Dimensions& dims, const ComplexInt4 value,
    const std::array<std::uint32_t, frequency_shard_count> loss_seeds,
    const float loss_probability) {
    validate_dimensions(dims);
    std::array<PackedVoltage, frequency_shard_count> payloads;
    for (auto& payload : payloads) {
        payload = make_constant(dims, value);
    }
    return make_shard_set(dims, payloads, loss_seeds, loss_probability);
}

PackedShardSet make_two_shard_noise(const Dimensions& dims, const std::uint32_t seed,
                                    const float loss_probability) {
    validate_dimensions(dims);
    std::array<PackedVoltage, frequency_shard_count> payloads;
    std::array<std::uint32_t, frequency_shard_count> loss_seeds{seed, seed + 1U};
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        // Distinct stream IDs keep shard payloads independent while preserving
        // reproducibility for a fixed base seed.
        payloads[shard_id] = make_noise(dims, seed + 0x9E3779B9U * static_cast<std::uint32_t>(shard_id));
    }
    return make_shard_set(dims, payloads, loss_seeds, loss_probability);
}

PackedShardSet make_two_shard_point_source(
    const Dimensions& dims, const std::vector<Vec3>& positions_m,
    const Vec3& source_direction, const float amplitude,
    const std::array<std::uint32_t, frequency_shard_count> loss_seeds,
    const float loss_probability) {
    validate_dimensions(dims);
    const auto descriptors = default_shard_descriptors();
    std::array<PackedVoltage, frequency_shard_count> payloads;
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        const float start_hz = default_frequency_start_hz
            + static_cast<float>(descriptors[shard_id].absolute_frequency_start)
                * default_channel_width_hz;
        const auto frequencies = channelized_frequencies(
            dims.n_freq, start_hz, default_channel_width_hz);
        payloads[shard_id] = make_point_source(
            dims, positions_m, frequencies, source_direction, amplitude);
    }
    return make_shard_set(dims, payloads, loss_seeds, loss_probability);
}

} // namespace beamformer
