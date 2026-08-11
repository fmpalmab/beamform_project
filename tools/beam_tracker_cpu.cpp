// CPU tracker beam CLI.
//
// Reads a headerless packed-int4 voltage shard, runs the naive CPU tracker
// beamformer over it, and writes the single-beam float32 intensity in the
// standard [time][frequency][beam] layout with n_beams == 1. The trajectory is
// the placeholder linear model from tracker.hpp: the source starts at
// direction cosines (l0, m0) and advances by (dl, dm) per time sample. One
// weight set is generated per integration window (default 320 spectra).

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/io.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    std::optional<std::filesystem::path> metrics;
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    float track_l0 = 0.0F;
    float track_m0 = 0.0F;
    float dl_per_sample = 0.0F;
    float dm_per_sample = 0.0F;
    std::size_t integration_spectra =
        beamformer::integration_direct.integration_spectra;
};

struct Timings {
    double load_ms = 0.0;
    double compute_ms = 0.0;
    double write_ms = 0.0;
    double total_ms = 0.0;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --input FILE --output FILE [options]\n\n"
        << "  --n-time N              default: 15360\n"
        << "  --n-ant N               32 or 64; default: 64\n"
        << "  --track-l0 F            initial direction cosine l; default: 0.0\n"
        << "  --track-m0 F            initial direction cosine m; default: 0.0\n"
        << "  --dl-per-sample F       linear drift in l per sample; default: 0.0\n"
        << "  --dm-per-sample F       linear drift in m per sample; default: 0.0\n"
        << "  --integration-spectra N one weight set per window; default: 320\n"
        << "  --metrics FILE          append timing row to CSV\n";
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

float parse_float(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stof(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid float for ") + option);
    }
    return parsed;
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--input") {
            options.input = require_value(argc, argv, i);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i);
        } else if (argument == "--metrics") {
            options.metrics = require_value(argc, argv, i);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--track-l0") {
            options.track_l0 = parse_float(require_value(argc, argv, i), "--track-l0");
        } else if (argument == "--track-m0") {
            options.track_m0 = parse_float(require_value(argc, argv, i), "--track-m0");
        } else if (argument == "--dl-per-sample") {
            options.dl_per_sample =
                parse_float(require_value(argc, argv, i), "--dl-per-sample");
        } else if (argument == "--dm-per-sample") {
            options.dm_per_sample =
                parse_float(require_value(argc, argv, i), "--dm-per-sample");
        } else if (argument == "--integration-spectra") {
            options.integration_spectra =
                parse_size(require_value(argc, argv, i), "--integration-spectra");
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::invalid_argument("--input and --output are required");
    }
    return options;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void append_metrics(const std::filesystem::path& path,
                    const beamformer::Dimensions& dims,
                    const beamformer::TrackerConfig& tracker,
                    const Timings& timings) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw std::runtime_error("cannot inspect metrics file: " + path.string());
    }
    const auto file_size = exists ? std::filesystem::file_size(path, error) : 0;
    if (error) {
        throw std::runtime_error("cannot inspect metrics file: " + path.string());
    }
    const bool needs_header = !exists || file_size == 0;

    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open metrics file: " + path.string());
    }
    if (needs_header) {
        output << "backend,n_time,n_freq,n_ant,n_beams,integration_spectra,"
                  "track_l0,track_m0,dl_per_sample,dm_per_sample,"
                  "load_ms,compute_ms,write_ms,total_ms\n";
    }
    output << std::fixed << std::setprecision(6) << "tracker_cpu," << dims.n_time << ','
           << dims.n_freq << ',' << dims.n_ant << ',' << dims.n_beams << ','
           << tracker.integration_spectra << ',' << tracker.trajectory.direction_start[0]
           << ',' << tracker.trajectory.direction_start[1] << ','
           << tracker.trajectory.direction_rate_per_sample[0] << ','
           << tracker.trajectory.direction_rate_per_sample[1] << ','
           << timings.load_ms << ',' << timings.compute_ms << ',' << timings.write_ms
           << ',' << timings.total_ms << '\n';
    if (!output) {
        throw std::runtime_error("failed to write metrics file: " + path.string());
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const beamformer::Dimensions dims{
            options.n_time,
            beamformer::default_frequency_channels,
            options.n_ant,
            beamformer::tracker_beam_count,
        };
        beamformer::validate_dimensions(dims);

        beamformer::TrackerConfig tracker;
        tracker.trajectory.direction_start =
            beamformer::direction_from_lm(options.track_l0, options.track_m0);
        tracker.trajectory.direction_rate_per_sample = {
            options.dl_per_sample, options.dm_per_sample};
        tracker.integration_spectra = options.integration_spectra;

        const auto total_start = Clock::now();
        const auto packed = beamformer::read_packed_voltage(options.input, dims);
        const auto load_end = Clock::now();
        const auto intensity =
            beamformer::beam_tracker_cpu_packed_intensity(packed, dims, tracker);
        const auto compute_end = Clock::now();
        beamformer::write_intensities(options.output, intensity, dims);
        const auto write_end = Clock::now();

        Timings timings;
        timings.load_ms = elapsed_ms(total_start, load_end);
        timings.compute_ms = elapsed_ms(load_end, compute_end);
        timings.write_ms = elapsed_ms(compute_end, write_end);
        timings.total_ms = elapsed_ms(total_start, write_end);

        if (options.metrics) {
            append_metrics(*options.metrics, dims, tracker, timings);
        }

        const std::size_t window_count =
            beamformer::tracker_window_count(dims.n_time, tracker.integration_spectra);
        std::cout << std::fixed << std::setprecision(3)
                  << "CPU tracker beamforming complete: layout=[T=" << dims.n_time
                  << "][F=" << dims.n_freq << "][B=" << dims.n_beams << "]\n"
                  << "windows=" << window_count
                  << " integration_spectra=" << tracker.integration_spectra
                  << " start=(" << tracker.trajectory.direction_start[0] << ','
                  << tracker.trajectory.direction_start[1] << ") rate=("
                  << tracker.trajectory.direction_rate_per_sample[0] << ','
                  << tracker.trajectory.direction_rate_per_sample[1] << ") per sample\n"
                  << "load_ms=" << timings.load_ms
                  << " compute_ms=" << timings.compute_ms
                  << " write_ms=" << timings.write_ms
                  << " total_ms=" << timings.total_ms << "\n"
                  << "Wrote " << intensity.size() * sizeof(float) << " bytes to "
                  << options.output << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "beam_tracker_cpu: " << error.what() << "\n";
        return 1;
    }
}
