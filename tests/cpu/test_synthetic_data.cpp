#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main() {
    using namespace beamformer;

    const Dimensions dims{2, default_frequency_channels, 64, 1};
    const auto zero = pack_complex_int4(0, 0);
    const auto active = pack_complex_int4(3, -2);

    const auto one_hot = make_one_hot(dims, 1, 335, 63, {3, -2});
    assert(one_hot.size() == dims.n_time * dims.n_freq * dims.n_ant);
    assert(one_hot[voltage_index(1, 335, 63, dims)] == active);
    assert(std::count(one_hot.begin(), one_hot.end(), active) == 1);
    assert(std::count(one_hot.begin(), one_hot.end(), zero)
           == static_cast<std::ptrdiff_t>(one_hot.size() - 1));

    const auto constant = make_constant(dims, {-4, 7});
    const auto constant_value = pack_complex_int4(-4, 7);
    assert(std::all_of(constant.begin(), constant.end(),
                       [constant_value](const auto value) {
                           return value == constant_value;
                       }));

    for (const auto real : std::array<std::int8_t, 5>{-8, -1, 0, 1, 7}) {
        for (const auto imag : std::array<std::int8_t, 5>{-8, -1, 0, 1, 7}) {
            const auto packed = pack_complex_int4(real, imag);
            const auto decoded = unpack_complex_int4(packed);
            assert(decoded.real == real);
            assert(decoded.imag == imag);
        }
    }

    const auto shards = make_two_shard_one_hot(
        dims, 1, {0, 335}, {0, 63}, {-8, 7}, {21, 22});
    validate_packed_shards(shards, dims);
    assert(shards[0].payload.size() == dims.n_time * 336U * dims.n_ant);
    assert(shards[1].payload.size() == dims.n_time * 336U * dims.n_ant);
    assert(shards[0].descriptor.shard_id == 0);
    assert(shards[1].descriptor.shard_id == 1);
    assert(absolute_frequency(shards[0].descriptor, 0) == 0);
    assert(absolute_frequency(shards[1].descriptor, 335) == 671);
    assert(shards[0].payload[voltage_index(1, 0, 0, dims)]
           == pack_complex_int4(-8, 7));
    assert(shards[1].payload[voltage_index(1, 335, 63, dims)]
           == pack_complex_int4(-8, 7));
    assert(shards[0].payload[voltage_index(1, 335, 63, dims)]
           == pack_complex_int4(0, 0));
    assert(shards[1].payload[voltage_index(1, 0, 0, dims)]
           == pack_complex_int4(0, 0));
    assert(shards[0].loss_mask.size() == loss_mask_count(dims));
    assert(shards[1].loss_mask.size() == loss_mask_count(dims));
    assert(shards[0].loss_mask.data() != shards[1].loss_mask.data());

    const auto constant_shards = make_two_shard_constant(dims, {-4, 7});
    assert(constant_shards[0].payload == constant_shards[1].payload);
    assert(constant_shards[0].payload.data() != constant_shards[1].payload.data());

    const auto two_noise_a = make_two_shard_noise(dims, 1234);
    const auto two_noise_b = make_two_shard_noise(dims, 1234);
    const auto two_noise_c = make_two_shard_noise(dims, 1235);
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        assert(two_noise_a[shard_id].payload == two_noise_b[shard_id].payload);
        assert(two_noise_a[shard_id].loss_mask == two_noise_b[shard_id].loss_mask);
        assert(two_noise_a[shard_id].payload != two_noise_c[shard_id].payload);
    }
    assert(two_noise_a[0].payload != two_noise_a[1].payload);

    auto isolated_loss = make_two_shard_constant(dims, {1, 0});
    const auto shard0_payload_before = isolated_loss[0].payload;
    const auto shard1_payload_before = isolated_loss[1].payload;
    const auto shard1_mask_before = isolated_loss[1].loss_mask;
    isolated_loss[0].loss_mask[0] = 0;
    assert(isolated_loss[0].loss_mask[0] == 0);
    assert(isolated_loss[1].loss_mask[0] == 1);
    assert(isolated_loss[0].payload == shard0_payload_before);
    assert(isolated_loss[1].payload == shard1_payload_before);
    assert(isolated_loss[1].loss_mask == shard1_mask_before);

    const auto two_point_source = make_two_shard_point_source(
        dims, default_positions(dims.n_ant), direction_from_lm(0.0F, 0.0F));
    const auto point_source_value = pack_complex_int4(4, 0);
    assert(std::all_of(two_point_source[0].payload.begin(),
                       two_point_source[0].payload.end(),
                       [point_source_value](const auto value) {
                           return value == point_source_value;
                       }));
    assert(std::all_of(two_point_source[1].payload.begin(),
                       two_point_source[1].payload.end(),
                       [point_source_value](const auto value) {
                           return value == point_source_value;
                       }));

    const auto noise_a = make_noise(dims, 1234);
    const auto noise_b = make_noise(dims, 1234);
    const auto noise_c = make_noise(dims, 1235);
    assert(noise_a == noise_b);
    assert(noise_a != noise_c);
    for (const auto packed : noise_a) {
        const auto sample = unpack_complex_int4(packed);
        assert(sample.real >= -7 && sample.real <= 7);
        assert(sample.imag >= -7 && sample.imag <= 7);
    }

    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto broadside =
        make_point_source(dims, positions, frequencies, direction_from_lm(0.0F, 0.0F));
    const auto broadside_value = pack_complex_int4(4, 0);
    assert(std::all_of(broadside.begin(), broadside.end(),
                       [broadside_value](const auto value) {
                           return value == broadside_value;
                       }));

    const auto off_axis =
        make_point_source(dims, positions, frequencies, direction_from_lm(0.04F, 0.0F));
    const std::size_t spectrum_size = dims.n_freq * dims.n_ant;
    assert(std::equal(off_axis.begin(), off_axis.begin() + spectrum_size,
                      off_axis.begin() + spectrum_size));
    assert(std::any_of(off_axis.begin(), off_axis.begin() + spectrum_size,
                       [broadside_value](const auto value) {
                           return value != broadside_value;
                       }));

    const auto path =
        std::filesystem::temp_directory_path() / "beamformer_poc_one_hot_test.bin";
    write_packed_voltage(path, one_hot, dims);
    assert(std::filesystem::file_size(path) == one_hot.size());
    std::ifstream input(path, std::ios::binary);
    const std::vector<std::uint8_t> loaded{std::istreambuf_iterator<char>(input),
                                           std::istreambuf_iterator<char>()};
    assert(loaded == one_hot);
    std::filesystem::remove(path);

    const auto payload_path =
        std::filesystem::temp_directory_path() / "beamformer_poc_shard_payload_test.bin";
    const auto metadata_path =
        std::filesystem::temp_directory_path() / "beamformer_poc_shard_metadata_test.txt";
    const auto mask_path =
        std::filesystem::temp_directory_path() / "beamformer_poc_shard_mask_test.bin";
    write_packed_shard(payload_path, metadata_path, mask_path, shards[1], dims);
    assert(std::filesystem::file_size(payload_path) == packed_voltage_bytes(dims));
    assert(std::filesystem::file_size(mask_path) == loss_mask_count(dims));
    const auto loaded_shard =
        read_packed_shard(payload_path, metadata_path, mask_path, dims);
    assert(loaded_shard.descriptor.shard_id == shards[1].descriptor.shard_id);
    assert(loaded_shard.descriptor.absolute_frequency_start
           == shards[1].descriptor.absolute_frequency_start);
    assert(loaded_shard.descriptor.timestamp_start
           == shards[1].descriptor.timestamp_start);
    assert(loaded_shard.descriptor.loss_mask_id == shards[1].descriptor.loss_mask_id);
    assert(loaded_shard.payload == shards[1].payload);
    assert(loaded_shard.loss_mask == shards[1].loss_mask);
    std::filesystem::remove(payload_path);
    std::filesystem::remove(metadata_path);
    std::filesystem::remove(mask_path);

    bool invalid_index_rejected = false;
    try {
        static_cast<void>(make_one_hot(dims, dims.n_time, 0, 0));
    } catch (const std::out_of_range&) {
        invalid_index_rejected = true;
    }
    assert(invalid_index_rejected);

    bool invalid_geometry_rejected = false;
    try {
        static_cast<void>(
            make_point_source(dims, std::vector<Vec3>(1), frequencies,
                              direction_from_lm(0.0F, 0.0F)));
    } catch (const std::invalid_argument&) {
        invalid_geometry_rejected = true;
    }
    assert(invalid_geometry_rejected);

    return 0;
}
