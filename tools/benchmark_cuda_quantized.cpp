#include "beamformer/config.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/io.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path input;
    std::filesystem::path weights;
    std::filesystem::path log;
    std::filesystem::path summary;
    std::size_t n_time = 15360;
    std::size_t integration_spectra = 320;
    std::size_t warmups = 2;
    std::size_t repetitions = 5;
    beamformer::CudaBeamformerKernel kernel = beamformer::CudaBeamformerKernel::Tiled;
};

const char* kernel_name(const beamformer::CudaBeamformerKernel kernel) {
    return kernel == beamformer::CudaBeamformerKernel::Direct ? "direct" : "tiled";
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

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " --input FILE --weights FILE --log FILE --summary FILE [options]\n\n"
        << "  --n-time N                 default: 15360\n"
        << "  --integration-spectra N    10 or 320; default: 320\n"
        << "  --kernel NAME              direct or tiled; default: tiled\n"
        << "  --warmups N                default: 2\n"
        << "  --repetitions N            default: 5\n";
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--input") {
            options.input = require_value(argc, argv, index);
        } else if (argument == "--weights") {
            options.weights = require_value(argc, argv, index);
        } else if (argument == "--log") {
            options.log = require_value(argc, argv, index);
        } else if (argument == "--summary") {
            options.summary = require_value(argc, argv, index);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, index), "--n-time");
        } else if (argument == "--integration-spectra") {
            options.integration_spectra = parse_size(
                require_value(argc, argv, index), "--integration-spectra");
        } else if (argument == "--kernel") {
            options.kernel = parse_kernel(require_value(argc, argv, index));
        } else if (argument == "--warmups") {
            options.warmups = parse_size(require_value(argc, argv, index), "--warmups");
        } else if (argument == "--repetitions") {
            options.repetitions = parse_size(require_value(argc, argv, index), "--repetitions");
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.input.empty() || options.weights.empty() || options.log.empty()
        || options.summary.empty()) {
        throw std::invalid_argument(
            "--input, --weights, --log, and --summary are required");
    }
    if (options.integration_spectra != 10 && options.integration_spectra != 320) {
        throw std::invalid_argument("--integration-spectra must be 10 or 320");
    }
    if (options.repetitions == 0) {
        throw std::invalid_argument("--repetitions must be positive");
    }
    if (options.kernel == beamformer::CudaBeamformerKernel::Direct
        && options.integration_spectra != 320) {
        throw std::invalid_argument("direct quantized benchmark only supports 320 spectra");
    }
    return options;
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Measurement {
    double h2d_ms = 0.0;
    double beamformer_ms = 0.0;
    double integration_ms = 0.0;
    double quantization_ms = 0.0;
    double d2h_ms = 0.0;
    double pipeline_event_ms = 0.0;
    double pipeline_wall_ms = 0.0;
};

Measurement run_once(beamformer::CudaBeamformerWorkspace& workspace,
                     const beamformer::PackedVoltage& packed,
                     beamformer::QuantizedIntegratedOutput& output,
                     const beamformer::Dimensions& dims) {
    const auto start = Clock::now();
    const auto timings = workspace.run_quantized_integrated_pipeline(
        packed, output, dims);
    const auto end = Clock::now();
    Measurement measurement;
    measurement.h2d_ms = timings.host_to_device_ms;
    measurement.beamformer_ms = timings.kernel_ms;
    measurement.integration_ms = timings.temporal_integration_ms;
    measurement.quantization_ms = timings.quantization_ms;
    measurement.d2h_ms = timings.device_to_host_ms;
    measurement.pipeline_event_ms = measurement.h2d_ms + measurement.beamformer_ms
                                    + measurement.integration_ms
                                    + measurement.quantization_ms + measurement.d2h_ms;
    measurement.pipeline_wall_ms = elapsed_ms(start, end);
    return measurement;
}

void write_measurement(std::ostream& output, const std::size_t repeat,
                       const Measurement& measurement) {
    output << std::fixed << std::setprecision(3)
           << "repeat=" << repeat << " h2d_ms=" << measurement.h2d_ms
           << " beamformer_ms=" << measurement.beamformer_ms
           << " integration_ms=" << measurement.integration_ms
           << " quantization_ms=" << measurement.quantization_ms
           << " d2h_ms=" << measurement.d2h_ms
           << " pipeline_event_ms=" << measurement.pipeline_event_ms
           << " pipeline_wall_ms=" << measurement.pipeline_wall_ms << '\n';
}

Measurement mean_measurement(const std::vector<Measurement>& measurements) {
    Measurement mean;
    for (const auto& measurement : measurements) {
        mean.h2d_ms += measurement.h2d_ms;
        mean.beamformer_ms += measurement.beamformer_ms;
        mean.integration_ms += measurement.integration_ms;
        mean.quantization_ms += measurement.quantization_ms;
        mean.d2h_ms += measurement.d2h_ms;
        mean.pipeline_event_ms += measurement.pipeline_event_ms;
        mean.pipeline_wall_ms += measurement.pipeline_wall_ms;
    }
    const double count = static_cast<double>(measurements.size());
    mean.h2d_ms /= count;
    mean.beamformer_ms /= count;
    mean.integration_ms /= count;
    mean.quantization_ms /= count;
    mean.d2h_ms /= count;
    mean.pipeline_event_ms /= count;
    mean.pipeline_wall_ms /= count;
    return mean;
}

bool has_content(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error)
           && std::filesystem::file_size(path, error) != 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const beamformer::Dimensions dims{
            options.n_time, beamformer::default_frequency_channels, 64, 64};
        beamformer::validate_dimensions(dims);

        const auto packed = beamformer::read_packed_voltage(options.input, dims);
        const auto weights = options.kernel == beamformer::CudaBeamformerKernel::Tiled
                                 ? beamformer::read_tiled_weights(options.weights, dims)
                                 : beamformer::read_weights(options.weights, dims);
        const beamformer::TemporalIntegrationConfig integration{
            options.integration_spectra};
        const beamformer::Dimensions integrated_dims{
            beamformer::integrated_time_count(options.n_time, integration),
            dims.n_freq, dims.n_ant, dims.n_beams};
        beamformer::QuantizedIntegratedOutput output{
            beamformer::QuantizedIntensities(
                beamformer::integrated_intensity_count(dims, integration)),
            std::vector<beamformer::Int8QuantizationParameters>(
                beamformer::quantization_parameter_count(integrated_dims))};

        beamformer::CudaBeamformerWorkspace workspace(
            dims, options.kernel, integration,
            beamformer::CudaBeamformerOutput::QuantizedInt8);
        const double weights_h2d_ms = workspace.upload_weights(weights, dims);

        std::ofstream log(options.log, std::ios::trunc);
        if (!log) {
            throw std::runtime_error("cannot open benchmark log: " + options.log.string());
        }
        log << "kernel=" << kernel_name(options.kernel) << " T=" << options.n_time
            << " integration=" << options.integration_spectra
            << " output=int8 setup_ms=" << std::fixed << std::setprecision(3)
            << workspace.setup_ms() << " weights_h2d_ms=" << weights_h2d_ms
            << " output_bytes="
            << (output.codes.size() * sizeof(std::int8_t)
                + output.parameters.size()
                      * sizeof(beamformer::Int8QuantizationParameters))
            << '\n';

        for (std::size_t repeat = 0; repeat < options.warmups; ++repeat) {
            static_cast<void>(run_once(workspace, packed, output, dims));
        }

        std::vector<Measurement> measurements;
        measurements.reserve(options.repetitions);
        for (std::size_t repeat = 0; repeat < options.repetitions; ++repeat) {
            const auto measurement = run_once(workspace, packed, output, dims);
            measurements.push_back(measurement);
            write_measurement(log, repeat + 1, measurement);
        }
        const auto mean = mean_measurement(measurements);
        log << std::fixed << std::setprecision(3)
            << "mean_h2d_ms=" << mean.h2d_ms
            << " mean_beamformer_ms=" << mean.beamformer_ms
            << " mean_integration_ms=" << mean.integration_ms
            << " mean_quantization_ms=" << mean.quantization_ms
            << " mean_d2h_ms=" << mean.d2h_ms
            << " mean_pipeline_event_ms=" << mean.pipeline_event_ms
            << " mean_pipeline_wall_ms=" << mean.pipeline_wall_ms << '\n';

        std::ofstream summary(options.summary, std::ios::app);
        if (!summary) {
            throw std::runtime_error("cannot open benchmark summary: "
                                     + options.summary.string());
        }
        if (!has_content(options.summary)) {
            summary << "configuration,n_time,integration_spectra,n_ant,n_freq,n_beams,"
                       "setup_ms,weights_h2d_ms,mean_h2d_ms,mean_beamformer_ms,"
                       "mean_integration_ms,mean_quantization_ms,mean_d2h_ms,"
                       "mean_pipeline_event_ms,mean_pipeline_wall_ms,warmups,repetitions,"
                       "output_int8_bytes,parameter_count\n";
        }
        summary << std::fixed << std::setprecision(3)
                << kernel_name(options.kernel) << '_' << options.integration_spectra << ','
                << options.n_time << ',' << options.integration_spectra << ','
                << dims.n_ant << ',' << dims.n_freq << ',' << dims.n_beams << ','
                << workspace.setup_ms() << ',' << weights_h2d_ms << ','
                << mean.h2d_ms << ',' << mean.beamformer_ms << ','
                << mean.integration_ms << ',' << mean.quantization_ms << ','
                << mean.d2h_ms << ',' << mean.pipeline_event_ms << ','
                << mean.pipeline_wall_ms << ',' << options.warmups << ','
                << options.repetitions << ',' << output.codes.size() << ','
                << output.parameters.size() << '\n';

        log.close();
        std::ifstream complete_log(options.log);
        std::cout << complete_log.rdbuf();
        std::cout << "Benchmark complete: " << options.log << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cuda_quantized: " << error.what() << '\n';
        return 1;
    }
}
