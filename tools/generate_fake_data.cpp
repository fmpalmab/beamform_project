#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string type = "point-source";
    std::filesystem::path output;
    std::optional<std::filesystem::path> shard_output_prefix;
    std::optional<std::filesystem::path> positions_file;
    std::optional<std::filesystem::path> frequencies_file;
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t active_time = 0;
    std::size_t active_frequency = 0;
    std::size_t active_element = 0;
    std::int8_t value_real = 3;
    std::int8_t value_imag = -2;
    std::uint32_t seed = 1;
    float spacing_m = beamformer::default_spacing_m;
    std::optional<float> frequency_hz;
    float source_l = 0.04F;
    float source_m = 0.0F;
    float amplitude = 4.0F;
    float loss_probability = 0.0F;
    // Moving point source (synthetic tracker validation target).
    float track_l0 = 0.0F;
    float track_m0 = 0.0F;
    float dl_per_sample = 0.0F;
    float dm_per_sample = 0.0F;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --output FILE [options]\n"
        << "       " << program << " --shard-output-prefix PREFIX [options]\n"
        << "\n"
        << "Data types: one-hot, constant, point-source, moving-point-source, noise\n"
        << "\n"
        << "Common options:\n"
        << "  --type TYPE             default: point-source\n"
        << "  --n-time N              default: 15360\n"
        << "  --n-ant N               32 or 64; default: 64\n"
        << "  --value-real N          int4 value; default: 3\n"
        << "  --value-imag N          int4 value; default: -2\n"
        << "  --seed N                noise seed; default: 1\n"
        << "  --shard-output-prefix P write P.shard{0,1}.{bin,meta,mask}\n"
        << "  --loss-probability P    lost [T][F] frames; default: 0\n"
        << "\n"
        << "One-hot options:\n"
        << "  --active-time N --active-frequency N --active-element N\n"
        << "\n"
        << "Point-source options (and moving-point-source base options):\n"
        << "  --source-l L --source-m M --amplitude A\n"
        << "  --spacing-m M           default geometry spacing: 0.6 m\n"
        << "  --frequency-hz HZ       optional constant-frequency override\n"
        << "                          default centers: 300 + 0.3*channel MHz\n"
        << "  --positions FILE        optional x,y,z rows indexed by output element\n"
        << "  --frequencies FILE      optional one-Hz-value-per-line override\n"
        << "\n"
        << "Moving-point-source options (single shard only):\n"
        << "  --track-l0 L --track-m0 M   source direction at t=0\n"
        << "  --dl-per-sample D         linear drift in l per time sample\n"
        << "  --dm-per-sample D         linear drift in m per time sample\n"
        << "                            (the source re-projects onto the unit disk per t)\n";
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

std::size_t parse_size(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stoull(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

std::int8_t parse_int4(const char* value, const char* option) {
    const int parsed = std::stoi(value);
    if (parsed < -8 || parsed > 7) {
        throw std::invalid_argument(std::string(option) + " must be in [-8, 7]");
    }
    return static_cast<std::int8_t>(parsed);
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i);
        } else if (argument == "--shard-output-prefix") {
            options.shard_output_prefix = require_value(argc, argv, i);
        } else if (argument == "--type") {
            options.type = require_value(argc, argv, i);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--active-time") {
            options.active_time = parse_size(require_value(argc, argv, i), "--active-time");
        } else if (argument == "--active-frequency") {
            options.active_frequency =
                parse_size(require_value(argc, argv, i), "--active-frequency");
        } else if (argument == "--active-element") {
            options.active_element =
                parse_size(require_value(argc, argv, i), "--active-element");
        } else if (argument == "--value-real") {
            options.value_real = parse_int4(require_value(argc, argv, i), "--value-real");
        } else if (argument == "--value-imag") {
            options.value_imag = parse_int4(require_value(argc, argv, i), "--value-imag");
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                parse_size(require_value(argc, argv, i), "--seed"));
        } else if (argument == "--spacing-m") {
            options.spacing_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--frequency-hz") {
            options.frequency_hz = std::stof(require_value(argc, argv, i));
        } else if (argument == "--source-l") {
            options.source_l = std::stof(require_value(argc, argv, i));
        } else if (argument == "--source-m") {
            options.source_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--amplitude") {
            options.amplitude = std::stof(require_value(argc, argv, i));
        } else if (argument == "--loss-probability") {
            options.loss_probability = std::stof(require_value(argc, argv, i));
        } else if (argument == "--track-l0") {
            options.track_l0 = std::stof(require_value(argc, argv, i));
        } else if (argument == "--track-m0") {
            options.track_m0 = std::stof(require_value(argc, argv, i));
        } else if (argument == "--dl-per-sample") {
            options.dl_per_sample = std::stof(require_value(argc, argv, i));
        } else if (argument == "--dm-per-sample") {
            options.dm_per_sample = std::stof(require_value(argc, argv, i));
        } else if (argument == "--positions") {
            options.positions_file = require_value(argc, argv, i);
        } else if (argument == "--frequencies") {
            options.frequencies_file = require_value(argc, argv, i);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.output.empty() && !options.shard_output_prefix) {
        throw std::invalid_argument("--output or --shard-output-prefix is required");
    }
    if (!options.output.empty() && options.shard_output_prefix) {
        throw std::invalid_argument("choose --output or --shard-output-prefix, not both");
    }
    return options;
}

beamformer::PackedVoltage generate(const Options& options,
                                   const beamformer::Dimensions& dims) {
    const beamformer::ComplexInt4 value{options.value_real, options.value_imag};
    if (options.type == "one-hot") {
        return beamformer::make_one_hot(dims, options.active_time, options.active_frequency,
                                        options.active_element, value);
    }
    if (options.type == "constant") {
        return beamformer::make_constant(dims, value);
    }
    if (options.type == "noise") {
        return beamformer::make_noise(dims, options.seed);
    }
    if (options.type == "point-source") {
        const auto positions = options.positions_file
                                   ? beamformer::load_positions(*options.positions_file, dims.n_ant)
                                   : beamformer::default_positions(dims.n_ant, options.spacing_m);
        const auto frequencies = options.frequencies_file
                                     ? beamformer::load_frequencies(
                                           *options.frequencies_file, dims.n_freq)
                                 : options.frequency_hz
                                     ? beamformer::constant_frequencies(
                                           dims.n_freq, *options.frequency_hz)
                                     : beamformer::channelized_frequencies(dims.n_freq);
        const auto direction =
            beamformer::direction_from_lm(options.source_l, options.source_m);
        return beamformer::make_point_source(dims, positions, frequencies, direction,
                                             options.amplitude);
    }
    if (options.type == "moving-point-source") {
        // Single-shard moving source: the natural validation target for the
        // tracker beam. Positions/frequencies resolution matches point-source.
        const auto positions = options.positions_file
                                   ? beamformer::load_positions(*options.positions_file, dims.n_ant)
                                   : beamformer::default_positions(dims.n_ant, options.spacing_m);
        const auto frequencies = options.frequencies_file
                                     ? beamformer::load_frequencies(
                                           *options.frequencies_file, dims.n_freq)
                                 : options.frequency_hz
                                     ? beamformer::constant_frequencies(
                                           dims.n_freq, *options.frequency_hz)
                                     : beamformer::channelized_frequencies(dims.n_freq);
        beamformer::TrackerTrajectoryConfig trajectory;
        trajectory.direction_start =
            beamformer::direction_from_lm(options.track_l0, options.track_m0);
        trajectory.direction_rate_per_sample = {options.dl_per_sample,
                                                 options.dm_per_sample};
        return beamformer::beam_tracker_make_moving_point_source(
            dims, positions, frequencies, trajectory, options.amplitude);
    }
    throw std::invalid_argument("unknown synthetic type: " + options.type);
}


beamformer::PackedShardSet generate_shards(
    const Options& options, const beamformer::Dimensions& dims) {
    // The tracker runs per node on a single local shard; the moving point source
    // is therefore a single-shard-only synthetic. The two-shard path keeps the
    // pre-existing one-hot/constant/noise/point-source catalog untouched.
    if (options.type == "moving-point-source") {
        throw std::invalid_argument(
            "moving-point-source is single-shard only; use --output, not --shard-output-prefix");
    }
    if (options.frequencies_file || options.frequency_hz) {
        throw std::invalid_argument(
            "two-shard output uses canonical absolute frequency mapping; custom frequency overrides are not supported");
    }
    const std::array<std::uint32_t, beamformer::frequency_shard_count> loss_seeds{
        options.seed, options.seed + 1U};
    if (options.type == "one-hot") {
        return beamformer::make_two_shard_one_hot(
            dims, options.active_time, {options.active_frequency, options.active_frequency},
            {options.active_element, options.active_element},
            {options.value_real, options.value_imag}, loss_seeds,
            options.loss_probability);
    }
    if (options.type == "constant") {
        return beamformer::make_two_shard_constant(
            dims, {options.value_real, options.value_imag}, loss_seeds,
            options.loss_probability);
    }
    if (options.type == "noise") {
        return beamformer::make_two_shard_noise(
            dims, options.seed, options.loss_probability);
    }
    if (options.type == "point-source") {
        const auto positions = options.positions_file
                                   ? beamformer::load_positions(*options.positions_file, dims.n_ant)
                                   : beamformer::default_positions(dims.n_ant, options.spacing_m);
        return beamformer::make_two_shard_point_source(
            dims, positions,
            beamformer::direction_from_lm(options.source_l, options.source_m),
            options.amplitude, loss_seeds, options.loss_probability);
    }
    throw std::invalid_argument("unknown synthetic type: " + options.type);
}

std::filesystem::path shard_file(const std::filesystem::path& prefix,
                                 const std::size_t shard_id,
                                 const char* suffix) {
    return prefix.string() + ".shard" + std::to_string(shard_id) + suffix;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const beamformer::Dimensions dims{
            options.n_time,
            beamformer::default_frequency_channels,
            options.n_ant,
            1,
        };
        beamformer::validate_dimensions(dims);
        if (options.shard_output_prefix) {
            const auto shards = generate_shards(options, dims);
            for (std::size_t shard_id = 0;
                 shard_id < beamformer::frequency_shard_count; ++shard_id) {
                const auto payload = shard_file(*options.shard_output_prefix, shard_id, ".bin");
                const auto metadata = shard_file(*options.shard_output_prefix, shard_id, ".meta");
                const auto mask = shard_file(*options.shard_output_prefix, shard_id, ".mask");
                beamformer::write_packed_shard(
                    payload, metadata, mask, shards[shard_id], dims);
                std::cout << "Wrote shard " << shard_id << ": "
                          << shards[shard_id].payload.size() << " payload bytes, "
                          << shards[shard_id].loss_mask.size() << " mask bytes\n"
                          << "  payload=" << payload << "\n"
                          << "  metadata=" << metadata << "\n"
                          << "  loss-mask=" << mask << "\n";
            }
            std::cout << "layout=[T=" << dims.n_time << "][F_local=" << dims.n_freq
                      << "][E=" << dims.n_ant << "] shards=2 type="
                      << options.type << "\n";
        } else {
            const auto voltage = generate(options, dims);
            beamformer::write_packed_voltage(options.output, voltage, dims);
            std::cout << "Wrote " << voltage.size() << " bytes to " << options.output << "\n"
                      << "layout=[T=" << dims.n_time << "][F_local=" << dims.n_freq
                      << "][E=" << dims.n_ant << "] type=" << options.type << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generate_fake_data: " << error.what() << "\n";
        return 1;
    }
}
