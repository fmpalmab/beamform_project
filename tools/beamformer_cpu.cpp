#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
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
#include <system_error>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path input;
    std::filesystem::path weights;
    std::filesystem::path output;
    std::optional<std::filesystem::path> metrics;
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t n_beams = 5;
};

struct Timings {
    double load_ms = 0.0;
    double unpack_ms = 0.0;
    double setup_ms = 0.0;
    double host_to_device_ms = 0.0;
    double compute_ms = 0.0;
    double device_to_host_ms = 0.0;
    double write_ms = 0.0;
    double total_ms = 0.0;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --input FILE --weights FILE --output FILE [options]\n\n"
        << "  --n-time N              default: 15360\n"
        << "  --n-ant N               32 or 64; default: 64\n"
        << "  --n-beams N             1 to 128; default: 5\n"
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

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--input") {
            options.input = require_value(argc, argv, i);
        } else if (argument == "--weights") {
            options.weights = require_value(argc, argv, i);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i);
        } else if (argument == "--metrics") {
            options.metrics = require_value(argc, argv, i);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--n-beams") {
            options.n_beams = parse_size(require_value(argc, argv, i), "--n-beams");
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.input.empty() || options.weights.empty() || options.output.empty()) {
        throw std::invalid_argument("--input, --weights, and --output are required");
    }
    return options;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t peak_beam(const beamformer::Intensities& intensity,
                      const beamformer::Dimensions& dims) {
    std::vector<double> integrated(dims.n_beams, 0.0);
    for (std::size_t index = 0; index < intensity.size(); ++index) {
        integrated[index % dims.n_beams] += intensity[index];
    }
    return static_cast<std::size_t>(
        std::distance(integrated.begin(),
                      std::max_element(integrated.begin(), integrated.end())));
}

void append_metrics(const std::filesystem::path& path,
                    const beamformer::Dimensions& dims, const Timings& timings,
                    const double output_elements_per_second,
                    const double complex_gmac_per_second) {
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
        output << "backend,n_time,n_freq,n_ant,n_beams,load_ms,unpack_ms,setup_ms,"
                  "host_to_device_ms,compute_ms,device_to_host_ms,write_ms,total_ms,"
                  "output_elements_per_second,complex_gmac_per_second\n";
    }
    output << std::fixed << std::setprecision(6) << "cpu," << dims.n_time << ','
           << dims.n_freq << ',' << dims.n_ant << ',' << dims.n_beams << ','
           << timings.load_ms << ',' << timings.unpack_ms << ',' << timings.setup_ms
           << ',' << timings.host_to_device_ms << ',' << timings.compute_ms << ','
           << timings.device_to_host_ms << ',' << timings.write_ms << ','
           << timings.total_ms << ',' << output_elements_per_second << ','
           << complex_gmac_per_second << '\n';
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
            options.n_beams,
        };
        beamformer::validate_dimensions(dims);

        const auto total_start = Clock::now();
        const auto packed = beamformer::read_packed_voltage(options.input, dims);
        const auto weights = beamformer::read_weights(options.weights, dims);
        const auto load_end = Clock::now();
        const auto intensity =
            beamformer::cpu_beamform_packed_intensity(packed, weights, dims);
        const auto compute_end = Clock::now();
        beamformer::write_intensities(options.output, intensity, dims);
        const auto write_end = Clock::now();

        Timings timings;
        timings.load_ms = elapsed_ms(total_start, load_end);
        // The CSV field is retained for compatibility; decoding is fused into
        // the CPU accumulation and is therefore not a separate stage.
        timings.unpack_ms = 0.0;
        timings.compute_ms = elapsed_ms(load_end, compute_end);
        timings.write_ms = elapsed_ms(compute_end, write_end);
        timings.total_ms = elapsed_ms(total_start, write_end);

        const double compute_seconds = timings.compute_ms / 1000.0;
        const double outputs = static_cast<double>(intensity.size());
        const double complex_macs = outputs * static_cast<double>(dims.n_ant);
        const double output_rate = compute_seconds > 0.0 ? outputs / compute_seconds : 0.0;
        const double gmac_rate =
            compute_seconds > 0.0 ? complex_macs / compute_seconds / 1.0e9 : 0.0;

        if (options.metrics) {
            append_metrics(*options.metrics, dims, timings, output_rate, gmac_rate);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "CPU beamforming complete: layout=[T=" << dims.n_time
                  << "][F=" << dims.n_freq << "][B=" << dims.n_beams << "]\n"
                  << "load_ms=" << timings.load_ms
                  << " unpack_ms=" << timings.unpack_ms
                  << " compute_ms=" << timings.compute_ms
                  << " write_ms=" << timings.write_ms
                  << " total_ms=" << timings.total_ms << "\n"
                  << "output_elements_per_second=" << output_rate
                  << " complex_GMAC_per_second=" << gmac_rate
                  << " peak_integrated_beam=" << peak_beam(intensity, dims) << "\n"
                  << "Wrote " << intensity.size() * sizeof(float) << " bytes to "
                  << options.output << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "beamformer_cpu: " << error.what() << "\n";
        return 1;
    }
}
