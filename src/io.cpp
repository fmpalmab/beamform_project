#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>

namespace beamformer {
namespace {

void require_file_size(const std::filesystem::path& path,
                       const std::uintmax_t expected_bytes) {
    std::error_code error;
    const auto actual_bytes = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error("cannot inspect input file: " + path.string());
    }
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(path.string() + " has " + std::to_string(actual_bytes)
                                 + " bytes; expected "
                                 + std::to_string(expected_bytes));
    }
}

template <typename Value>
std::vector<Value> read_binary(const std::filesystem::path& path,
                               const std::size_t count) {
    require_file_size(path, count * sizeof(Value));
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path.string());
    }
    std::vector<Value> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(Value)));
    if (!input) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

template <typename Value>
void write_binary(const std::filesystem::path& path,
                  const std::vector<Value>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(Value)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

} // namespace

PackedVoltage read_packed_voltage(const std::filesystem::path& path,
                                  const Dimensions& dims) {
    validate_dimensions(dims);
    return read_binary<std::uint8_t>(path, voltage_sample_count(dims));
}

void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims) {
    validate_dimensions(dims);
    if (voltage.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    write_binary(path, voltage);
}

Weights read_weights(const std::filesystem::path& path, const Dimensions& dims) {
    validate_dimensions(dims);
    return read_binary<ComplexFloat>(path, dims.n_beams * dims.n_freq * dims.n_ant);
}

void write_weights(const std::filesystem::path& path, const Weights& weights,
                   const Dimensions& dims) {
    validate_dimensions(dims);
    if (weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    write_binary(path, weights);
}

void write_intensities(const std::filesystem::path& path,
                       const Intensities& intensities,
                       const Dimensions& dims) {
    validate_dimensions(dims);
    if (intensities.size() != dims.n_time * dims.n_freq * dims.n_beams) {
        throw std::invalid_argument("intensity count does not match dimensions");
    }
    write_binary(path, intensities);
}

void write_packed_shard(const std::filesystem::path& payload_path,
                        const std::filesystem::path& metadata_path,
                        const std::filesystem::path& loss_mask_path,
                        const PackedShard& shard, const Dimensions& dims) {
    validate_packed_shard(shard, dims);
    write_packed_voltage(payload_path, shard.payload, dims);
    write_binary(loss_mask_path, shard.loss_mask);

    std::ofstream metadata(metadata_path, std::ios::trunc);
    if (!metadata) {
        throw std::runtime_error("cannot open shard metadata file: "
                                 + metadata_path.string());
    }
    const auto& descriptor = shard.descriptor;
    metadata << "format=beamformer-packed-shard-v1\n"
             << "n_time=" << dims.n_time << '\n'
             << "n_freq=" << dims.n_freq << '\n'
             << "n_ant=" << dims.n_ant << '\n'
             << "n_beams=" << dims.n_beams << '\n'
             << "shard_id=" << descriptor.shard_id << '\n'
             << "shard_count=" << descriptor.shard_count << '\n'
             << "local_frequency_count=" << descriptor.local_frequency_count << '\n'
             << "absolute_frequency_start=" << descriptor.absolute_frequency_start << '\n'
             << "timestamp_start=" << descriptor.timestamp_start << '\n'
             << "timestamp_step=" << descriptor.timestamp_step << '\n'
             << "loss_mask_id=" << descriptor.loss_mask_id << '\n'
             << "loss_mask_independent=" << (descriptor.loss_mask_independent ? 1 : 0)
             << '\n'
             << "payload_bytes=" << shard.payload.size() << '\n'
             << "loss_mask_bytes=" << shard.loss_mask.size() << '\n';
    if (!metadata) {
        throw std::runtime_error("failed to write shard metadata file: "
                                 + metadata_path.string());
    }
}

PackedShard read_packed_shard(const std::filesystem::path& payload_path,
                              const std::filesystem::path& metadata_path,
                              const std::filesystem::path& loss_mask_path,
                              const Dimensions& dims) {
    validate_dimensions(dims);
    std::ifstream metadata_file(metadata_path);
    if (!metadata_file) {
        throw std::runtime_error("cannot open shard metadata file: "
                                 + metadata_path.string());
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(metadata_file, line)) {
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0) {
            throw std::runtime_error("invalid shard metadata line in: "
                                     + metadata_path.string());
        }
        fields[line.substr(0, separator)] = line.substr(separator + 1);
    }
    const auto required = [&fields, &metadata_path](const char* name) -> const std::string& {
        const auto found = fields.find(name);
        if (found == fields.end()) {
            throw std::runtime_error("missing shard metadata field "
                                     + std::string(name) + " in: "
                                     + metadata_path.string());
        }
        return found->second;
    };
    const auto number = [&required](const char* name) {
        return static_cast<std::size_t>(std::stoull(required(name)));
    };
    if (required("format") != "beamformer-packed-shard-v1"
        || number("n_time") != dims.n_time || number("n_freq") != dims.n_freq
        || number("n_ant") != dims.n_ant || number("n_beams") != dims.n_beams
        || number("payload_bytes") != packed_voltage_bytes(dims)
        || number("loss_mask_bytes") != loss_mask_count(dims)) {
        throw std::runtime_error("shard metadata does not match dimensions: "
                                 + metadata_path.string());
    }

    const auto boolean = [&required](const char* name) {
        const auto value = required(name);
        if (value != "0" && value != "1") {
            throw std::runtime_error("invalid boolean shard metadata field: "
                                     + std::string(name));
        }
        return value == "1";
    };
    PackedShard shard;
    shard.descriptor = ShardDescriptor{
        number("shard_id"), number("shard_count"), number("local_frequency_count"),
        number("absolute_frequency_start"),
        static_cast<std::uint64_t>(std::stoull(required("timestamp_start"))),
        static_cast<std::uint64_t>(std::stoull(required("timestamp_step"))),
        static_cast<std::uint64_t>(std::stoull(required("loss_mask_id"))),
        boolean("loss_mask_independent"),
    };
    shard.payload = read_packed_voltage(payload_path, dims);
    shard.loss_mask = read_binary<std::uint8_t>(loss_mask_path, loss_mask_count(dims));
    validate_packed_shard(shard, dims);
    return shard;
}

} // namespace beamformer
