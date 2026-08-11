#include "beamformer/config.hpp"
#include "beamformer/cuda_beamformer.hpp"
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
    std::optional<std::size_t> integration_spectra;
    bool output_int8 = false;
    std::optional<std::filesystem::path> quantization_parameters;
    std::optional<std::filesystem::path> quantization_metadata;
    std::size_t shard_id = 0;
    beamformer::CudaBeamformerKernel kernel =
        beamformer::CudaBeamformerKernel::Direct;
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
        << "  --kernel NAME           direct or tiled; default: direct\n"
        << "  --integration-spectra N GPU integration: 10 or 320\n"
        << "  --output-format NAME    float32 or int8; default: float32\n"
        << "  --quantization-params F int8 parameter output; required for int8\n"
        << "  --quantization-metadata F int8 metadata output; required for int8\n"
        << "  --shard-id N            0 or 1; default: 0\n"
        << "  --metrics FILE          append timing row to CSV\n";
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

beamformer::CudaBeamformerKernel parse_kernel(const char* value) {
    const std::string name(value);
    if (name == "direct") {
        return beamformer::CudaBeamformerKernel::Direct;
    }
    if (name == "tiled") {
        return beamformer::CudaBeamformerKernel::Tiled;
    }
    throw std::invalid_argument("kernel must be direct or tiled");
}

bool parse_output_int8(const char* value) {
    const std::string name(value);
    if (name == "float32") {
        return false;
    }
    if (name == "int8") {
        return true;
    }
    throw std::invalid_argument("output format must be float32 or int8");
}

const char* kernel_name(const beamformer::CudaBeamformerKernel kernel) {
    return kernel == beamformer::CudaBeamformerKernel::Direct ? "direct" : "tiled";
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
        } else if (argument == "--kernel") {
            options.kernel = parse_kernel(require_value(argc, argv, i));
        } else if (argument == "--integration-spectra") {
            options.integration_spectra = parse_size(
                require_value(argc, argv, i), "--integration-spectra");
        } else if (argument == "--output-format") {
            options.output_int8 = parse_output_int8(require_value(argc, argv, i));
        } else if (argument == "--quantization-params") {
            options.quantization_parameters = require_value(argc, argv, i);
        } else if (argument == "--quantization-metadata") {
            options.quantization_metadata = require_value(argc, argv, i);
        } else if (argument == "--shard-id") {
            options.shard_id = parse_size(require_value(argc, argv, i), "--shard-id");
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.input.empty() || options.weights.empty() || options.output.empty()) {
        throw std::invalid_argument("--input, --weights, and --output are required");
    }
    if (options.output_int8 && (!options.integration_spectra
                                || !options.quantization_parameters
                                || !options.quantization_metadata)) {
        throw std::invalid_argument(
            "int8 output requires --integration-spectra, --quantization-params, and --quantization-metadata");
    }
    if (options.shard_id >= beamformer::frequency_shard_count) {
        throw std::invalid_argument("--shard-id must be 0 or 1");
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
    output << std::fixed << std::setprecision(6) << "cuda," << dims.n_time << ','
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
        const auto weights = options.kernel == beamformer::CudaBeamformerKernel::Tiled
                                 ? beamformer::read_tiled_weights(options.weights, dims)
                                 : beamformer::read_weights(options.weights, dims);
        const auto load_end = Clock::now();
        beamformer::CudaBeamformerTimings cuda_timings;
        const std::optional<beamformer::TemporalIntegrationConfig> temporal_integration =
            options.integration_spectra
                ? std::optional<beamformer::TemporalIntegrationConfig>(
                      beamformer::TemporalIntegrationConfig{*options.integration_spectra})
                : std::nullopt;
        const auto output_dims = temporal_integration
            ? beamformer::Dimensions{
                  beamformer::integrated_time_count(dims.n_time, *temporal_integration),
                  dims.n_freq, dims.n_ant, dims.n_beams}
            : dims;
        std::size_t output_bytes = 0;
        std::size_t output_elements = 0;
        std::optional<std::size_t> output_peak_beam;
        Clock::time_point compute_end;
        if (options.output_int8) {
            const auto quantized =
                beamformer::cuda_beamform_packed_quantized_integrated_intensity(
                    packed, weights, dims, *temporal_integration, &cuda_timings,
                    options.kernel);
            compute_end = Clock::now();
            beamformer::write_quantized_intensities(options.output, quantized.codes, output_dims);
            beamformer::write_quantization_parameters(
                *options.quantization_parameters, quantized.parameters, output_dims);
            const beamformer::QuantizedOutputMetadata metadata{
                output_dims, dims.n_time, *temporal_integration,
                beamformer::default_shard_descriptors()[options.shard_id],
                options.quantization_parameters->filename().string(),
            };
            beamformer::write_quantized_output_metadata(
                *options.quantization_metadata, metadata);
            output_elements = quantized.codes.size();
            output_bytes = quantized.codes.size() * sizeof(std::int8_t)
                           + quantized.parameters.size()
                                 * sizeof(beamformer::Int8QuantizationParameters);
        } else {
            const auto intensity = temporal_integration
                ? beamformer::cuda_beamform_packed_integrated_intensity(
                      packed, weights, dims, *temporal_integration, &cuda_timings,
                      options.kernel)
                : beamformer::cuda_beamform_packed_intensity(
                      packed, weights, dims, &cuda_timings, options.kernel);
            compute_end = Clock::now();
            beamformer::write_intensities(options.output, intensity, output_dims);
            output_elements = intensity.size();
            output_bytes = intensity.size() * sizeof(float);
            output_peak_beam = peak_beam(intensity, output_dims);
        }
        const auto write_end = Clock::now();

        Timings timings;
        timings.load_ms = elapsed_ms(total_start, load_end);
        // Packed bytes are decoded in the CUDA kernel, not on the host.
        timings.unpack_ms = 0.0;
        timings.setup_ms = cuda_timings.setup_ms;
        timings.host_to_device_ms = cuda_timings.host_to_device_ms;
        timings.compute_ms = cuda_timings.kernel_ms + cuda_timings.temporal_integration_ms
                             + cuda_timings.quantization_ms;
        timings.device_to_host_ms = cuda_timings.device_to_host_ms;
        timings.write_ms = elapsed_ms(compute_end, write_end);
        timings.total_ms = elapsed_ms(total_start, write_end);

        const double compute_seconds = timings.compute_ms / 1000.0;
        const double outputs = static_cast<double>(output_elements);
        const double raw_outputs = static_cast<double>(
            dims.n_time * dims.n_freq * dims.n_beams);
        const double complex_macs = raw_outputs * static_cast<double>(dims.n_ant);
        const double output_rate = compute_seconds > 0.0 ? outputs / compute_seconds : 0.0;
        const double gmac_rate =
            compute_seconds > 0.0 ? complex_macs / compute_seconds / 1.0e9 : 0.0;
        const double cuda_call_ms = elapsed_ms(load_end, compute_end);
        const double pipeline_ms = timings.host_to_device_ms + timings.compute_ms
                                   + timings.device_to_host_ms;
        const double pipeline_output_rate =
            pipeline_ms > 0.0 ? outputs / (pipeline_ms / 1000.0) : 0.0;

        if (options.metrics) {
            append_metrics(*options.metrics, dims, timings, output_rate, gmac_rate);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "CUDA beamforming complete: kernel=" << kernel_name(options.kernel)
                  << " input_layout=[T=" << dims.n_time
                  << "][F=" << dims.n_freq << "][B=" << dims.n_beams << "]"
                  << " output_layout=[T=" << output_dims.n_time
                  << "][F=" << output_dims.n_freq << "][B=" << output_dims.n_beams
                  << "]\n"
                  << "load_ms=" << timings.load_ms
                  << " unpack_ms=" << timings.unpack_ms
                  << " setup_ms=" << timings.setup_ms
                  << " host_to_device_ms=" << timings.host_to_device_ms
                  << " kernel_ms=" << timings.compute_ms
                  << " beamformer_ms=" << cuda_timings.kernel_ms
                  << " temporal_integration_ms=" << cuda_timings.temporal_integration_ms
                  << " quantization_ms=" << cuda_timings.quantization_ms
                  << " device_to_host_ms=" << timings.device_to_host_ms
                  << " cuda_call_ms=" << cuda_call_ms
                  << " write_ms=" << timings.write_ms
                  << " total_ms=" << timings.total_ms << "\n"
                  << "kernel_output_elements_per_second=" << output_rate
                  << " pipeline_output_elements_per_second=" << pipeline_output_rate
                  << " complex_GMAC_per_second=" << gmac_rate;
        if (output_peak_beam) {
            std::cout << " peak_integrated_beam=" << *output_peak_beam;
        }
        std::cout << "\nWrote " << output_bytes << " bytes of output data to "
                  << options.output << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "beamformer_cuda: " << error.what() << "\n";
        return 1;
    }
}
