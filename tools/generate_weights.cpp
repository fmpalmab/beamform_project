#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/io.hpp"
#include "beamformer/weights.hpp"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path output;
    std::optional<std::filesystem::path> positions_file;
    std::optional<std::filesystem::path> frequencies_file;
    std::optional<std::filesystem::path> directions_file;
    std::size_t n_ant = 64;
    std::size_t n_beams = 5;
    float spacing_m = beamformer::default_spacing_m;
    std::optional<float> frequency_hz;
    float beam_l_step = 0.02F;
    float beam_m = 0.0F;
    std::string beam_grid = "fft";
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --output FILE [options]\n\n"
        << "  --n-ant N               32 or 64; default: 64\n"
        << "  --n-beams N             1 to 128; default: 5\n"
        << "  --spacing-m M           default geometry spacing: 0.6 m\n"
        << "  --frequency-hz HZ       optional constant-frequency override\n"
        << "                          default centers: 300 + 0.3*channel MHz\n"
        << "  --beam-grid GRID        fft, line, or legacy-rectangular; default: fft\n"
        << "  --beam-l-step L         line-grid spacing in l: 0.02\n"
        << "  --beam-m M              line-grid m direction cosine: 0\n"
        << "  --positions FILE        optional x,y,z rows\n"
        << "  --frequencies FILE      optional one-Hz-value-per-line override\n"
        << "  --directions FILE       optional unit x,y,z rows, one per beam\n";
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

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i);
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--n-beams") {
            options.n_beams = parse_size(require_value(argc, argv, i), "--n-beams");
        } else if (argument == "--spacing-m") {
            options.spacing_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--frequency-hz") {
            options.frequency_hz = std::stof(require_value(argc, argv, i));
        } else if (argument == "--beam-l-step") {
            options.beam_l_step = std::stof(require_value(argc, argv, i));
        } else if (argument == "--beam-m") {
            options.beam_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--beam-grid") {
            options.beam_grid = require_value(argc, argv, i);
        } else if (argument == "--positions") {
            options.positions_file = require_value(argc, argv, i);
        } else if (argument == "--frequencies") {
            options.frequencies_file = require_value(argc, argv, i);
        } else if (argument == "--directions") {
            options.directions_file = require_value(argc, argv, i);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.output.empty()) {
        throw std::invalid_argument("--output is required");
    }
    if (options.beam_grid != "fft" && options.beam_grid != "line"
        && options.beam_grid != "legacy-rectangular") {
        throw std::invalid_argument(
            "--beam-grid must be fft, line, or legacy-rectangular");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const beamformer::Dimensions dims{
            1,
            beamformer::default_frequency_channels,
            options.n_ant,
            options.n_beams,
        };
        beamformer::validate_dimensions(dims);
        if (options.beam_grid == "legacy-rectangular"
            && dims.n_beams != dims.n_ant) {
            throw std::invalid_argument(
                "legacy-rectangular requires n_beams == n_ant");
        }

        const auto positions = options.positions_file
                                   ? beamformer::load_positions(*options.positions_file,
                                                                dims.n_ant)
                                   : beamformer::default_positions(dims.n_ant,
                                                                   options.spacing_m);
        const auto frequencies = options.frequencies_file
                                     ? beamformer::load_frequencies(
                                           *options.frequencies_file, dims.n_freq)
                                 : options.frequency_hz
                                     ? beamformer::constant_frequencies(
                                           dims.n_freq, *options.frequency_hz)
                                     : beamformer::channelized_frequencies(dims.n_freq);
        const auto directions = options.directions_file
                                    ? beamformer::load_directions(*options.directions_file,
                                                                  dims.n_beams)
                                : options.beam_grid == "fft"
                                    ? beamformer::fft_beam_grid(
                                          dims.n_ant, dims.n_beams,
                                          options.spacing_m)
                                : options.beam_grid == "legacy-rectangular"
                                    ? beamformer::rectangular_beam_grid(
                                          dims.n_beams, options.spacing_m)
                                    : beamformer::default_beam_grid(
                                          dims.n_beams, options.beam_l_step,
                                          options.beam_m);
        const auto weights =
            beamformer::generate_weights(dims, positions, frequencies, directions);
        beamformer::write_weights(options.output, weights, dims);

        std::cout << "Wrote " << weights.size() << " complex float32 weights ("
                  << weights.size() * sizeof(beamformer::ComplexFloat) << " bytes) to "
                  << options.output << "\n"
                  << "layout=[B=" << dims.n_beams << "][F=" << dims.n_freq
                  << "][E=" << dims.n_ant << "]\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generate_weights: " << error.what() << "\n";
        return 1;
    }
}
