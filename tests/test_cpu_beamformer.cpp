#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float actual, const float expected, const float tolerance = 1.0e-3F) {
    return std::abs(actual - expected) <= tolerance;
}

std::int8_t independent_decode_nibble(const std::uint8_t nibble) {
    const auto bits = static_cast<std::uint8_t>(nibble & 0x0F);
    return bits < 8 ? static_cast<std::int8_t>(bits)
                    : static_cast<std::int8_t>(static_cast<int>(bits) - 16);
}

float independently_calculated_power(const beamformer::PackedVoltage& packed,
                                     const beamformer::Weights& weights,
                                     const beamformer::Dimensions& dims,
                                     const std::size_t time,
                                     const std::size_t frequency,
                                     const std::size_t beam) {
    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t element = 0; element < dims.n_ant; ++element) {
        const auto byte = packed[beamformer::voltage_index(time, frequency, element, dims)];
        const float sample_real = static_cast<float>(independent_decode_nibble(byte));
        const float sample_imag = static_cast<float>(
            independent_decode_nibble(static_cast<std::uint8_t>(byte >> 4)));
        const auto& weight = weights[beamformer::weight_index(beam, frequency, element, dims)];
        sum_real += weight.real * sample_real - weight.imag * sample_imag;
        sum_imag += weight.real * sample_imag + weight.imag * sample_real;
    }
    return sum_real * sum_real + sum_imag * sum_imag;
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

    // Exhaustive raw-byte coverage: every packed byte must decode to the
    // independently calculated signed two's-complement nibble pair.
    for (int raw = 0; raw < 256; ++raw) {
        const auto byte = static_cast<std::uint8_t>(raw);
        const auto decoded = unpack_complex_int4(byte);
        assert(decoded.real == independent_decode_nibble(byte));
        assert(decoded.imag == independent_decode_nibble(
                                  static_cast<std::uint8_t>(byte >> 4)));
    }

    const Dimensions dims{1, default_frequency_channels, 32, 5};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = default_beam_grid(dims.n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    assert(weights.size() == dims.n_beams * dims.n_freq * dims.n_ant);
    for (const auto& weight : weights) {
        assert(close(weight.real * weight.real + weight.imag * weight.imag, 1.0F));
    }
    const auto tiled_weights = generate_tiled_weights(
        dims, positions, frequencies, directions);
    assert(tiled_weights.size() == tiled_weight_count(dims));
    for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t antenna = 0; antenna < dims.n_ant; ++antenna) {
                const auto tiled = tiled_weights[tiled_weight_index(
                    frequency, beam / tiled_weight_beam_tile, antenna,
                    beam % tiled_weight_beam_tile, dims)];
                const auto canonical = weights[weight_index(beam, frequency, antenna, dims)];
                assert(tiled.real == canonical.real);
                assert(tiled.imag == canonical.imag);
            }
        }
    }
    for (std::size_t beam = dims.n_beams;
         beam < tiled_weight_beam_tiles(dims) * tiled_weight_beam_tile; ++beam) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t antenna = 0; antenna < dims.n_ant; ++antenna) {
                const auto padded = tiled_weights[tiled_weight_index(
                    frequency, beam / tiled_weight_beam_tile, antenna,
                    beam % tiled_weight_beam_tile, dims)];
                assert(padded.real == 0.0F && padded.imag == 0.0F);
            }
        }
    }

    const std::size_t active_frequency = 17;
    const auto packed_one_hot =
        make_one_hot(dims, 0, active_frequency, 7, ComplexInt4{3, -2});
    const auto expanded_one_hot = cpu_beamform_intensity(
        unpack_voltage(packed_one_hot, dims), weights, dims);
    const auto one_hot = cpu_beamform_packed_intensity(packed_one_hot, weights, dims);
    assert(one_hot == expanded_one_hot);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
            const float expected = frequency == active_frequency ? 13.0F : 0.0F;
            assert(close(one_hot[intensity_index(0, frequency, beam, dims)], expected));
        }
    }

    ComplexVoltage extended_voltage = unpack_voltage(packed_one_hot, dims);
    extended_voltage.push_back({99.0F, 99.0F});
    Intensities preallocated(one_hot.size() + 1, -1.0F);
    cpu_beamform_intensity_into(extended_voltage, weights, dims, preallocated);
    assert(std::equal(one_hot.begin(), one_hot.end(), preallocated.begin()));
    assert(preallocated.back() == -1.0F);

    PackedVoltage extended_packed = packed_one_hot;
    extended_packed.push_back(pack_complex_int4(7, 7));
    Intensities packed_preallocated(one_hot.size() + 1, -2.0F);
    cpu_beamform_packed_intensity_into(
        extended_packed, weights, dims, packed_preallocated);
    assert(std::equal(one_hot.begin(), one_hot.end(), packed_preallocated.begin()));
    assert(packed_preallocated.back() == -2.0F);
    assert(throws_invalid_argument([&] {
        cpu_beamform_packed_intensity(PackedVoltage(1, 0), weights, dims);
    }));

    const Dimensions broadside_dims{1, default_frequency_channels, 32, 1};
    const auto broadside_weights = generate_weights(
        broadside_dims, positions, frequencies,
        std::vector<Vec3>{direction_from_lm(0.0F, 0.0F)});
    const auto constant_packed = make_constant(broadside_dims, ComplexInt4{1, 0});
    const auto constant = cpu_beamform_packed_intensity(
        constant_packed, broadside_weights, broadside_dims);
    const auto constant_expanded = cpu_beamform_intensity(
        unpack_voltage(constant_packed, broadside_dims),
        broadside_weights, broadside_dims);
    assert(constant == constant_expanded);
    const float coherent_intensity =
        static_cast<float>(broadside_dims.n_ant * broadside_dims.n_ant);
    assert(std::all_of(constant.begin(), constant.end(),
                       [coherent_intensity](const float value) {
                           return close(value, coherent_intensity);
                       }));

    const auto source_direction = direction_from_lm(0.04F, 0.0F);
    const auto point_source = make_point_source(
        dims, positions, frequencies, source_direction, 4.0F);
    const auto point_intensity = cpu_beamform_packed_intensity(
        point_source, weights, dims);
    const auto point_intensity_expanded = cpu_beamform_intensity(
        unpack_voltage(point_source, dims), weights, dims);
    for (std::size_t index = 0; index < point_intensity.size(); ++index) {
        assert(close(point_intensity[index], point_intensity_expanded[index], 1.0e-5F));
    }
    std::vector<double> integrated(dims.n_beams, 0.0);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
            integrated[beam] +=
                point_intensity[intensity_index(0, frequency, beam, dims)];
        }
    }
    const auto peak = std::distance(
        integrated.begin(), std::max_element(integrated.begin(), integrated.end()));
    assert(peak == 4);

    const Dimensions final_dims{1, default_frequency_channels, 32, 32};
    const auto final_directions = rectangular_beam_grid(final_dims.n_ant);
    const auto final_weights =
        generate_weights(final_dims, positions, frequencies, final_directions);
    const std::size_t injected_beam = 12;
    const auto final_source = make_point_source(
        final_dims, positions, frequencies, final_directions[injected_beam], 4.0F);
    const auto final_intensity = cpu_beamform_packed_intensity(
        final_source, final_weights, final_dims);
    std::vector<double> final_integrated(final_dims.n_beams, 0.0);
    for (std::size_t frequency = 0; frequency < final_dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < final_dims.n_beams; ++beam) {
            final_integrated[beam] +=
                final_intensity[intensity_index(0, frequency, beam, final_dims)];
        }
    }
    const auto final_peak = std::distance(
        final_integrated.begin(),
        std::max_element(final_integrated.begin(), final_integrated.end()));
    assert(static_cast<std::size_t>(final_peak) == injected_beam);

    // Seeded noise is reproducible, and selected outputs agree with an
    // independently calculated packed-byte reference. This also exercises
    // [time][frequency][beam] output indexing over more than one time sample.
    const Dimensions noise_dims{2, default_frequency_channels, 32, 2};
    Weights noise_weights(noise_dims.n_beams * noise_dims.n_freq * noise_dims.n_ant);
    for (std::size_t beam = 0; beam < noise_dims.n_beams; ++beam) {
        for (std::size_t frequency = 0; frequency < noise_dims.n_freq; ++frequency) {
            for (std::size_t element = 0; element < noise_dims.n_ant; ++element) {
                noise_weights[weight_index(beam, frequency, element, noise_dims)] =
                    beam == 0 ? ComplexFloat{1.0F, 0.0F}
                              : ComplexFloat{0.25F, -0.5F};
            }
        }
    }
    const auto noise_a = make_noise(noise_dims, 4242);
    const auto noise_b = make_noise(noise_dims, 4242);
    const auto noise_c = make_noise(noise_dims, 4243);
    const auto noise_intensity_a =
        cpu_beamform_packed_intensity(noise_a, noise_weights, noise_dims);
    const auto noise_intensity_b =
        cpu_beamform_packed_intensity(noise_b, noise_weights, noise_dims);
    const auto noise_intensity_c =
        cpu_beamform_packed_intensity(noise_c, noise_weights, noise_dims);
    assert(noise_intensity_a == noise_intensity_b);
    assert(noise_intensity_a != noise_intensity_c);
    for (const auto time : {std::size_t{0}, std::size_t{1}}) {
        for (const auto frequency : {std::size_t{0}, std::size_t{335}}) {
            for (std::size_t beam = 0; beam < noise_dims.n_beams; ++beam) {
                const auto output = intensity_index(time, frequency, beam, noise_dims);
                assert(noise_intensity_a[output] == independently_calculated_power(
                    noise_a, noise_weights, noise_dims, time, frequency, beam));
            }
        }
    }

    // The two local frequency shards use separate allocations and are
    // beamformed independently; the same local index has no cross-shard effect.
    const auto shard_set = make_two_shard_one_hot(
        dims, 0, {3, 7}, {1, 2}, {3, -2}, {17, 18});
    const auto shard_zero_intensity =
        cpu_beamform_packed_intensity(shard_set[0].payload, weights, dims);
    const auto shard_one_intensity =
        cpu_beamform_packed_intensity(shard_set[1].payload, weights, dims);
    for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
        assert(shard_zero_intensity[intensity_index(0, 3, beam, dims)] > 0.0F);
        assert(shard_zero_intensity[intensity_index(0, 7, beam, dims)] == 0.0F);
        assert(shard_one_intensity[intensity_index(0, 3, beam, dims)] == 0.0F);
        assert(shard_one_intensity[intensity_index(0, 7, beam, dims)] > 0.0F);
    }

    const auto temporary = std::filesystem::temp_directory_path();
    const auto voltage_path = temporary / "beamformer_cpu_voltage_test.bin";
    const auto weights_path = temporary / "beamformer_cpu_weights_test.bin";
    const auto tiled_weights_path = temporary / "beamformer_cpu_tiled_weights_test.bin";
    const auto malformed_path = temporary / "beamformer_cpu_malformed_test.bin";
    write_packed_voltage(voltage_path, point_source, dims);
    write_weights(weights_path, weights, dims);
    write_tiled_weights(tiled_weights_path, tiled_weights, dims);
    assert(read_packed_voltage(voltage_path, dims) == point_source);
    const auto loaded_weights = read_weights(weights_path, dims);
    assert(loaded_weights.size() == weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
        assert(loaded_weights[index].real == weights[index].real);
        assert(loaded_weights[index].imag == weights[index].imag);
    }
    const auto loaded_tiled_weights = read_tiled_weights(tiled_weights_path, dims);
    assert(loaded_tiled_weights.size() == tiled_weights.size());
    for (std::size_t index = 0; index < tiled_weights.size(); ++index) {
        assert(loaded_tiled_weights[index].real == tiled_weights[index].real);
        assert(loaded_tiled_weights[index].imag == tiled_weights[index].imag);
    }

    {
        std::ofstream malformed(malformed_path, std::ios::binary | std::ios::trunc);
        malformed.put('\0');
    }
    bool malformed_rejected = false;
    try {
        static_cast<void>(read_packed_voltage(malformed_path, dims));
    } catch (const std::runtime_error&) {
        malformed_rejected = true;
    }
    assert(malformed_rejected);

    std::filesystem::remove(voltage_path);
    std::filesystem::remove(weights_path);
    std::filesystem::remove(tiled_weights_path);
    std::filesystem::remove(malformed_path);
    return 0;
}
