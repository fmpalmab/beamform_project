#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

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

    for (int real = -8; real <= 7; ++real) {
        for (int imag = -8; imag <= 7; ++imag) {
            const auto packed = pack_complex_int4(static_cast<std::int8_t>(real),
                                                  static_cast<std::int8_t>(imag));
            const auto decoded = unpack_complex_int4(packed);
            assert(decoded.real == real);
            assert(decoded.imag == imag);
        }
    }

    assert(unpack_complex_int4(0x07).real == 7);
    assert(unpack_complex_int4(0x08).real == -8);
    assert(unpack_complex_int4(0x70).imag == 7);
    assert(unpack_complex_int4(0x80).imag == -8);

    Dimensions dims{3, default_frequency_channels, 64, 5};
    validate_dimensions(dims);

    const Dimensions defaults;
    assert(defaults.n_time == 15360);
    assert(defaults.n_freq == local_frequency_channels);
    assert(defaults.n_ant == 64);
    assert(defaults.n_beams == 64);

    const auto shards = default_shard_descriptors();
    assert(shards.size() == frequency_shard_count);
    for (const auto& shard : shards) {
        validate_shard_descriptor(shard);
        assert(shard.local_frequency_count == dims.n_freq);
        assert(shard.timestamp_step > 0);
        assert(shard.loss_mask_independent);
    }
    assert(shards[0].shard_id == 0);
    assert(shards[1].shard_id == 1);
    assert(shards[0].loss_mask_id != shards[1].loss_mask_id);
    assert(absolute_frequency(shards[0], 0) == 0);
    assert(absolute_frequency(shards[0], 335) == 335);
    assert(absolute_frequency(shards[1], 0) == 336);
    assert(absolute_frequency(shards[1], 335) == 671);

    const std::size_t local_index =
        shard_voltage_index(2, 17, 63, dims, shards[0]);
    assert(local_index == voltage_index(2, 17, 63, dims));
    assert(shard_voltage_index(2, 17, 63, dims, shards[1]) == local_index);

    assert(throws_invalid_argument([] {
        validate_shard_descriptor(ShardDescriptor{2, 2, 336, 672, 0, 1, 2, true});
    }));
    assert(throws_invalid_argument([] {
        validate_shard_descriptor(ShardDescriptor{0, 2, 335, 0, 0, 1, 0, true});
    }));
    assert(throws_invalid_argument([] {
        validate_shard_descriptor(ShardDescriptor{0, 2, 336, 0, 0, 0, 0, true});
    }));
    assert(throws_invalid_argument([] {
        validate_shard_descriptor(ShardDescriptor{0, 2, 336, 0, 0, 1, 0, false});
    }));

    const std::size_t sample_count = voltage_sample_count(dims);
    std::vector<bool> visited(sample_count, false);
    for (std::size_t t = 0; t < dims.n_time; ++t) {
        for (std::size_t f = 0; f < dims.n_freq; ++f) {
            for (std::size_t a = 0; a < dims.n_ant; ++a) {
                const auto physical = voltage_index(t, f, a, dims);
                assert(physical < sample_count);
                assert(!visited[physical]);
                visited[physical] = true;
            }
        }
    }
    for (const bool value : visited) {
        assert(value);
    }

    assert(voltage_index(0, 0, 0, dims) == 0);
    assert(voltage_index(0, 0, 63, dims) == 63);
    assert(voltage_index(0, 1, 0, dims) == 64);
    assert(voltage_index(1, 0, 0, dims) == dims.n_freq * dims.n_ant);
    assert(rfsoc_output_element(1, 0) == 0);
    assert(rfsoc_output_element(1, 31) == 31);
    assert(rfsoc_output_element(0, 0) == 32);
    assert(rfsoc_output_element(0, 31) == 63);
    assert(full_band_frequency(0, 335) == 335);
    assert(full_band_frequency(1, 0) == 336);
    assert(full_band_frequency(1, 335) == 671);
    assert(weight_index(1, 0, 0, dims) == dims.n_freq * dims.n_ant);
    assert(intensity_index(1, 0, 0, dims) == dims.n_freq * dims.n_beams);

    assert(packed_voltage_bytes(dims) == 3U * 336U * 64U);
    assert(weight_bytes(dims) == 5U * 336U * 64U * 2U * sizeof(float));
    assert(intensity_bytes(dims) == 3U * 336U * 5U * sizeof(float));

    assert(throws_invalid_argument([] { validate_dimensions({0, default_frequency_channels, 32, 1}); }));
    assert(throws_invalid_argument([] { validate_dimensions({32, 335, 32, 1}); }));
    assert(throws_invalid_argument([] { validate_dimensions({32, default_frequency_channels, 48, 1}); }));
    assert(throws_invalid_argument([] { validate_dimensions({32, default_frequency_channels, 32, 0}); }));
    validate_dimensions({32, default_frequency_channels, 32, 11});
    validate_dimensions({32, default_frequency_channels, 32, 16});
    validate_dimensions({32, default_frequency_channels, 64, 48});
    validate_dimensions({32, default_frequency_channels, 32, 32});
    validate_dimensions({32, default_frequency_channels, 64, 64});
    validate_dimensions({32, default_frequency_channels, 32, 128});
    validate_dimensions({32, default_frequency_channels, 64, 128});
    assert(throws_invalid_argument([] { validate_dimensions({32, default_frequency_channels, 32, 129}); }));
    assert(throws_invalid_argument([] { validate_dimensions({32, default_frequency_channels, 64, 129}); }));

    return 0;
}
