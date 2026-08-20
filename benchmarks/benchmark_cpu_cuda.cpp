#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
constexpr double real_flops_per_complex_mac = 8.0;
constexpr double real_flops_per_intensity = 3.0;

struct Options {
    std::filesystem::path output_prefix = "results/gpu_benchmark_fft";
    std::vector<std::size_t> antenna_values{32, 64};
    std::vector<std::size_t> time_values{15360, 24576, 30720};
    std::vector<std::size_t> beam_values{16, 32, 64, 128};
    std::size_t validation_time = 16;
    std::size_t warmup = 3;
    std::size_t repetitions = 10;
    std::uint32_t seed = 1;
    double absolute_tolerance = 1.0e-3;
    double relative_tolerance = 1.0e-5;
    bool dry_run = false;
    beamformer::CudaBeamformerKernel kernel =
        beamformer::CudaBeamformerKernel::Direct;
};

struct ValidationStats {
    std::size_t output_count = 0;
    double max_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
    double max_relative_error = 0.0;
    double mean_relative_error = 0.0;
    double sampled_p99_relative_error = 0.0;
    double normalized_rmse = 0.0;
    double correlation = 1.0;
    std::size_t outside_tolerance = 0;
    std::size_t cpu_peak_beam = 0;
    std::size_t gpu_peak_beam = 0;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --output-prefix PATH    default: results/gpu_benchmark_fft\n"
        << "  --n-ant LIST            comma list from {32,64}; default: 32,64\n"
        << "  --times LIST            GPU times; default: 15360,24576,30720\n"
        << "  --beams LIST            default: 16,32,64,128\n"
        << "  --validation-time N     compact CPU/CUDA check; default: 16\n"
        << "  --warmup N              default: 3\n"
        << "  --repetitions N         default: 10\n"
        << "  --seed N                default: 1\n"
        << "  --absolute-tolerance X  default: 1e-3\n"
        << "  --relative-tolerance X  default: 1e-5\n"
        << "  --kernel NAME           direct or tiled; default: direct\n"
        << "  --dry-run               validate and print the matrix only\n";
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

std::size_t parse_size(const std::string& text, const char* option) {
    std::size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + option);
    }
    return static_cast<std::size_t>(value);
}

beamformer::CudaBeamformerKernel parse_kernel(const std::string& text) {
    if (text == "direct") {
        return beamformer::CudaBeamformerKernel::Direct;
    }
    if (text == "tiled") {
        return beamformer::CudaBeamformerKernel::Tiled;
    }
    throw std::invalid_argument("kernel must be direct or tiled");
}

const char* kernel_name(const beamformer::CudaBeamformerKernel kernel) {
    return kernel == beamformer::CudaBeamformerKernel::Direct ? "direct" : "tiled";
}

double parse_double(const std::string& text, const char* option) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("invalid number for ") + option);
    }
    return value;
}

std::vector<std::size_t> parse_list(const std::string& text,
                                    const char* option) {
    std::vector<std::size_t> values;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument(std::string("empty value in ") + option);
        }
        values.push_back(parse_size(item, option));
    }
    if (values.empty()) {
        throw std::invalid_argument(std::string(option) + " cannot be empty");
    }
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--output-prefix") {
            options.output_prefix = require_value(argc, argv, i);
        } else if (argument == "--n-ant") {
            options.antenna_values = parse_list(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--times") {
            options.time_values = parse_list(require_value(argc, argv, i), "--times");
        } else if (argument == "--beams") {
            options.beam_values = parse_list(require_value(argc, argv, i), "--beams");
        } else if (argument == "--validation-time") {
            options.validation_time =
                parse_size(require_value(argc, argv, i), "--validation-time");
        } else if (argument == "--warmup") {
            options.warmup = parse_size(require_value(argc, argv, i), "--warmup");
        } else if (argument == "--repetitions") {
            options.repetitions =
                parse_size(require_value(argc, argv, i), "--repetitions");
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                parse_size(require_value(argc, argv, i), "--seed"));
        } else if (argument == "--absolute-tolerance") {
            options.absolute_tolerance = parse_double(
                require_value(argc, argv, i), "--absolute-tolerance");
        } else if (argument == "--relative-tolerance") {
            options.relative_tolerance = parse_double(
                require_value(argc, argv, i), "--relative-tolerance");
        } else if (argument == "--kernel") {
            options.kernel = parse_kernel(require_value(argc, argv, i));
        } else if (argument == "--dry-run") {
            options.dry_run = true;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

void validate_options(const Options& options) {
    if (options.output_prefix.empty()) {
        throw std::invalid_argument("output prefix cannot be empty");
    }
    if (options.repetitions == 0) {
        throw std::invalid_argument("repetitions must be positive");
    }
    if (options.absolute_tolerance < 0.0 || options.relative_tolerance < 0.0) {
        throw std::invalid_argument("validation tolerances cannot be negative");
    }
    for (const std::size_t n_ant : options.antenna_values) {
        if (n_ant != 32 && n_ant != 64) {
            throw std::invalid_argument("n-ant values must be 32 or 64");
        }
    }
    for (const std::size_t n_time : options.time_values) {
        if (n_time == 0) {
            throw std::invalid_argument("time values must be positive");
        }
    }
    if (options.validation_time == 0) {
        throw std::invalid_argument("validation-time must be positive");
    }
    for (const std::size_t beams : options.beam_values) {
        if (beams == 0 || beams > beamformer::maximum_beams) {
            throw std::invalid_argument("beam values must be between 1 and 128");
        }
    }
    for (const std::size_t n_ant : options.antenna_values) {
        for (const std::size_t beams : options.beam_values) {
            static_cast<void>(beamformer::fft_beam_grid(n_ant, beams));
        }
    }
}

std::filesystem::path with_suffix(const std::filesystem::path& prefix,
                                  const std::string& suffix) {
    return prefix.parent_path() / (prefix.filename().string() + suffix);
}

double wall_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::uint64_t output_count(const beamformer::Dimensions& dims) {
    return static_cast<std::uint64_t>(dims.n_time) * dims.n_freq * dims.n_beams;
}

std::uint64_t complex_mac_count(const beamformer::Dimensions& dims) {
    return output_count(dims) * dims.n_ant;
}

double estimated_flop_count(const beamformer::Dimensions& dims) {
    return real_flops_per_complex_mac * static_cast<double>(complex_mac_count(dims))
           + real_flops_per_intensity * static_cast<double>(output_count(dims));
}

double rate_per_second(const double count, const double milliseconds) {
    return milliseconds > 0.0 ? count / (milliseconds / 1000.0) : 0.0;
}

beamformer::PackedVoltage make_benchmark_voltage(const std::size_t n_ant,
                                                 const std::size_t max_time,
                                                 const std::uint32_t seed) {
    const beamformer::Dimensions one_time{
        1, beamformer::default_frequency_channels, n_ant, 1};
    const auto spectrum = beamformer::make_noise(one_time, seed);
    beamformer::PackedVoltage packed(spectrum.size() * max_time);
    for (std::size_t time = 0; time < max_time; ++time) {
        std::copy(spectrum.begin(), spectrum.end(),
                  packed.begin()
                      + static_cast<std::ptrdiff_t>(time * spectrum.size()));
    }
    return packed;
}

beamformer::Weights make_benchmark_weights(
    const std::size_t n_ant, const std::size_t n_beams,
    const beamformer::CudaBeamformerKernel kernel) {
    const beamformer::Dimensions dims{
        1, beamformer::default_frequency_channels, n_ant, n_beams};
    const auto positions = beamformer::default_positions(n_ant);
    const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);
    const auto directions = beamformer::fft_beam_grid(n_ant, n_beams);
    return kernel == beamformer::CudaBeamformerKernel::Tiled
               ? beamformer::generate_tiled_weights(
                     dims, positions, frequencies, directions)
               : beamformer::generate_weights(
                     dims, positions, frequencies, directions);
}

ValidationStats compare_outputs(const beamformer::Intensities& cpu,
                                const beamformer::Intensities& gpu,
                                const beamformer::Dimensions& dims,
                                const double absolute_tolerance,
                                const double relative_tolerance) {
    const std::size_t count = static_cast<std::size_t>(output_count(dims));
    if (cpu.size() < count || gpu.size() < count) {
        throw std::invalid_argument("validation outputs are smaller than dimensions");
    }

    ValidationStats stats;
    stats.output_count = count;
    std::vector<double> relative_sample;
    constexpr std::size_t maximum_sample = 1'000'000;
    const std::size_t sample_stride = std::max<std::size_t>(1, count / maximum_sample);
    relative_sample.reserve(std::min(count, maximum_sample + 1));
    std::size_t next_sample = 0;
    std::size_t relative_count = 0;
    double absolute_sum = 0.0;
    double relative_sum = 0.0;
    double squared_difference_sum = 0.0;
    double squared_reference_sum = 0.0;
    double reference_sum = 0.0;
    double candidate_sum = 0.0;
    double reference_squared_sum = 0.0;
    double candidate_squared_sum = 0.0;
    double cross_sum = 0.0;
    std::vector<double> cpu_integrated(dims.n_beams, 0.0);
    std::vector<double> gpu_integrated(dims.n_beams, 0.0);

    for (std::size_t index = 0; index < count; ++index) {
        const double reference = cpu[index];
        const double candidate = gpu[index];
        const double difference = candidate - reference;
        const double absolute_error = std::abs(difference);
        const double reference_absolute = std::abs(reference);
        stats.max_absolute_error = std::max(stats.max_absolute_error, absolute_error);
        absolute_sum += absolute_error;
        squared_difference_sum += difference * difference;
        squared_reference_sum += reference * reference;
        reference_sum += reference;
        candidate_sum += candidate;
        reference_squared_sum += reference * reference;
        candidate_squared_sum += candidate * candidate;
        cross_sum += reference * candidate;
        if (absolute_error
            > absolute_tolerance + relative_tolerance * reference_absolute) {
            ++stats.outside_tolerance;
        }
        if (reference_absolute > absolute_tolerance) {
            const double relative_error = absolute_error / reference_absolute;
            stats.max_relative_error =
                std::max(stats.max_relative_error, relative_error);
            relative_sum += relative_error;
            ++relative_count;
            if (index >= next_sample) {
                relative_sample.push_back(relative_error);
                next_sample = index + sample_stride;
            }
        }
        const std::size_t beam = index % dims.n_beams;
        cpu_integrated[beam] += reference;
        gpu_integrated[beam] += candidate;
    }

    stats.mean_absolute_error = absolute_sum / static_cast<double>(count);
    stats.mean_relative_error = relative_count > 0
                                    ? relative_sum / static_cast<double>(relative_count)
                                    : 0.0;
    if (!relative_sample.empty()) {
        std::sort(relative_sample.begin(), relative_sample.end());
        const std::size_t p99_index = static_cast<std::size_t>(
            std::ceil(0.99 * static_cast<double>(relative_sample.size()))) - 1;
        stats.sampled_p99_relative_error = relative_sample[p99_index];
    }
    stats.normalized_rmse = squared_reference_sum > 0.0
                                ? std::sqrt(squared_difference_sum
                                            / squared_reference_sum)
                                : std::sqrt(squared_difference_sum
                                            / static_cast<double>(count));
    const double count_double = static_cast<double>(count);
    const double covariance = count_double * cross_sum
                              - reference_sum * candidate_sum;
    const double reference_variance = count_double * reference_squared_sum
                                      - reference_sum * reference_sum;
    const double candidate_variance = count_double * candidate_squared_sum
                                      - candidate_sum * candidate_sum;
    const double denominator =
        std::sqrt(std::max(0.0, reference_variance)
                  * std::max(0.0, candidate_variance));
    stats.correlation = denominator > 0.0 ? covariance / denominator
                                         : stats.max_absolute_error == 0.0 ? 1.0 : 0.0;
    stats.cpu_peak_beam = static_cast<std::size_t>(
        std::distance(cpu_integrated.begin(),
                      std::max_element(cpu_integrated.begin(), cpu_integrated.end())));
    stats.gpu_peak_beam = static_cast<std::size_t>(
        std::distance(gpu_integrated.begin(),
                      std::max_element(gpu_integrated.begin(), gpu_integrated.end())));
    return stats;
}

void write_timing_header(std::ofstream& output) {
    output
        << "n_ant,n_freq,n_beams,n_time,repeat,n_outputs,n_cmac,estimated_flop,"
           "cuda_setup_ms,weights_h2d_ms,gpu_kernel_ms,gpu_h2d_ms,"
           "gpu_pipeline_kernel_ms,gpu_d2h_ms,gpu_pipeline_event_ms,"
           "gpu_pipeline_wall_ms,gpu_kernel_cmac_per_s,"
           "gpu_pipeline_cmac_per_s,gpu_kernel_estimated_flop_per_s,"
           "gpu_pipeline_estimated_flop_per_s\n";
}

void write_timing_row(std::ofstream& output, const beamformer::Dimensions& dims,
                      const std::size_t repeat, const double setup_ms,
                      const double weights_h2d_ms,
                      const double gpu_kernel_ms,
                      const beamformer::CudaBeamformerTimings& pipeline,
                      const double pipeline_wall_ms) {
    const auto outputs = output_count(dims);
    const auto cmac = complex_mac_count(dims);
    const double flops = estimated_flop_count(dims);
    const double pipeline_event_ms = pipeline.host_to_device_ms + pipeline.kernel_ms
                                     + pipeline.device_to_host_ms;
    output << dims.n_ant << ',' << dims.n_freq << ',' << dims.n_beams << ','
           << dims.n_time << ',' << repeat << ',' << outputs << ',' << cmac << ','
           << flops << ',' << setup_ms << ',' << weights_h2d_ms << ','
           << gpu_kernel_ms << ',' << pipeline.host_to_device_ms << ','
           << pipeline.kernel_ms << ',' << pipeline.device_to_host_ms << ','
           << pipeline_event_ms << ',' << pipeline_wall_ms << ','
           << rate_per_second(static_cast<double>(cmac), gpu_kernel_ms) << ','
           << rate_per_second(static_cast<double>(cmac), pipeline_wall_ms) << ','
           << rate_per_second(flops, gpu_kernel_ms) << ','
           << rate_per_second(flops, pipeline_wall_ms) << '\n';
    output.flush();
}

void write_validation_header(std::ofstream& output) {
    output
        << "n_ant,n_freq,n_beams,n_time,n_outputs,max_absolute_error,"
           "mean_absolute_error,max_relative_error,mean_relative_error,"
           "sampled_p99_relative_error,normalized_rmse,correlation,"
           "outside_tolerance,cpu_peak_beam,gpu_peak_beam\n";
}

void write_validation_row(std::ofstream& output,
                          const beamformer::Dimensions& dims,
                          const ValidationStats& stats) {
    output << dims.n_ant << ',' << dims.n_freq << ',' << dims.n_beams << ','
           << dims.n_time << ',' << stats.output_count << ','
           << stats.max_absolute_error << ',' << stats.mean_absolute_error << ','
           << stats.max_relative_error << ',' << stats.mean_relative_error << ','
           << stats.sampled_p99_relative_error << ',' << stats.normalized_rmse << ','
           << stats.correlation << ',' << stats.outside_tolerance << ','
           << stats.cpu_peak_beam << ',' << stats.gpu_peak_beam << '\n';
    output.flush();
}

void print_memory_estimate(const std::size_t n_ant, const std::size_t max_time,
                           const std::size_t max_beams) {
    const beamformer::Dimensions capacity{
        max_time, beamformer::default_frequency_channels, n_ant, max_beams};
    const double voltage_bytes = static_cast<double>(
        beamformer::packed_voltage_bytes(capacity));
    const double weight_bytes = static_cast<double>(max_beams * capacity.n_freq * n_ant
                                                     * sizeof(beamformer::ComplexFloat));
    const double intensity_bytes = static_cast<double>(output_count(capacity)
                                                        * sizeof(float));
    std::cout << "n_ant=" << n_ant << " max host working set ~= "
              << (voltage_bytes + intensity_bytes + weight_bytes) / bytes_per_gib
              << " GiB; max GPU workspace ~= "
              << (voltage_bytes + intensity_bytes + weight_bytes) / bytes_per_gib
              << " GiB\n";
}

void run_antenna_series(const Options& options, const std::size_t n_ant,
                        std::ofstream& timings_output,
                        std::ofstream& validation_output,
                        beamformer::CudaDeviceInfo& device_info) {
    const auto& beam_values = options.beam_values;
    const std::size_t max_time = std::max(
        options.validation_time,
        *std::max_element(options.time_values.begin(), options.time_values.end()));
    const std::size_t max_beams = *std::max_element(beam_values.begin(),
                                                   beam_values.end());
    print_memory_estimate(n_ant, max_time, max_beams);
    std::cout << "Preparing deterministic packed-voltage prefix for n_ant=" << n_ant
              << "..." << std::endl;
    auto packed_voltage = make_benchmark_voltage(n_ant, max_time,
                                          options.seed + static_cast<std::uint32_t>(n_ant));
    const beamformer::Dimensions capacity{
        max_time, beamformer::default_frequency_channels, n_ant, max_beams};
    beamformer::CudaBeamformerWorkspace workspace(capacity, options.kernel);
    if (device_info.name.empty()) {
        device_info = beamformer::cuda_device_info();
    }
    std::cout << "CUDA workspace ready: kernel=" << kernel_name(options.kernel)
              << " setup_ms=" << workspace.setup_ms() << std::endl;

    for (const std::size_t n_beams : beam_values) {
        const auto cpu_weights = make_benchmark_weights(
            n_ant, n_beams, beamformer::CudaBeamformerKernel::Direct);
        const auto gpu_weights = make_benchmark_weights(
            n_ant, n_beams, options.kernel);
        const beamformer::Dimensions weight_dims{
            1, beamformer::default_frequency_channels, n_ant, n_beams};
        const double weights_h2d_ms = workspace.upload_weights(gpu_weights, weight_dims);

        const beamformer::Dimensions validation_dims{
            options.validation_time, beamformer::default_frequency_channels,
            n_ant, n_beams};
        const std::size_t validation_outputs =
            static_cast<std::size_t>(output_count(validation_dims));
        beamformer::Intensities cpu_validation(validation_outputs);
        beamformer::Intensities gpu_validation(validation_outputs);
        const auto cpu_start = Clock::now();
        beamformer::cpu_beamform_packed_intensity_into(
            packed_voltage, cpu_weights, validation_dims, cpu_validation);
        const auto cpu_end = Clock::now();
        const auto gpu_validation_timing =
            workspace.run_pipeline(packed_voltage, gpu_validation, validation_dims);
        const auto validation = compare_outputs(
            cpu_validation, gpu_validation, validation_dims,
            options.absolute_tolerance, options.relative_tolerance);
        write_validation_row(validation_output, validation_dims, validation);
        std::cout << "validate A=" << n_ant << " B=" << n_beams
                  << " T=" << options.validation_time
                  << " cpu_ms=" << wall_ms(cpu_start, cpu_end)
                  << " gpu_pipeline_event_ms="
                  << gpu_validation_timing.host_to_device_ms
                         + gpu_validation_timing.kernel_ms
                         + gpu_validation_timing.device_to_host_ms
                  << " max_rel=" << validation.max_relative_error
                  << " outside=" << validation.outside_tolerance
                  << " peaks=" << validation.cpu_peak_beam << '/'
                  << validation.gpu_peak_beam << std::endl;
        if (validation.outside_tolerance != 0
            || validation.cpu_peak_beam != validation.gpu_peak_beam) {
            throw std::runtime_error(
                "compact CPU/CUDA validation failed for benchmark configuration");
        }

        for (const std::size_t n_time : options.time_values) {
            const beamformer::Dimensions dims{
                n_time, beamformer::default_frequency_channels, n_ant, n_beams};
            const std::size_t outputs = static_cast<std::size_t>(output_count(dims));
            beamformer::Intensities gpu_output(outputs);
            workspace.upload_packed_voltage(packed_voltage, dims);

            for (std::size_t warmup = 0; warmup < options.warmup; ++warmup) {
                const double kernel_ms = workspace.run_kernel(dims);
                const auto pipeline = workspace.run_pipeline(packed_voltage, gpu_output, dims);
                std::cout << "gpu warmup " << warmup + 1 << '/' << options.warmup
                          << " A=" << n_ant << " B=" << n_beams
                          << " T=" << n_time << " kernel_ms=" << kernel_ms
                          << " pipeline_event_ms="
                          << pipeline.host_to_device_ms + pipeline.kernel_ms
                                 + pipeline.device_to_host_ms
                          << std::endl;
            }

            std::vector<double> gpu_kernel_times(options.repetitions);
            std::vector<beamformer::CudaBeamformerTimings> pipeline_times(
                options.repetitions);
            std::vector<double> pipeline_wall_times(options.repetitions);
            for (std::size_t repeat = 0; repeat < options.repetitions; ++repeat) {
                gpu_kernel_times[repeat] = workspace.run_kernel(dims);
                const auto pipeline_start = Clock::now();
                pipeline_times[repeat] =
                    workspace.run_pipeline(packed_voltage, gpu_output, dims);
                const auto pipeline_end = Clock::now();
                pipeline_wall_times[repeat] = wall_ms(pipeline_start, pipeline_end);
                write_timing_row(timings_output, dims, repeat, workspace.setup_ms(),
                                 weights_h2d_ms, gpu_kernel_times[repeat],
                                 pipeline_times[repeat],
                                 pipeline_wall_times[repeat]);
                std::cout << "gpu measure " << repeat + 1 << '/'
                          << options.repetitions << " A=" << n_ant
                          << " B=" << n_beams << " T=" << n_time
                          << " kernel_ms=" << gpu_kernel_times[repeat]
                          << " pipeline_ms=" << pipeline_wall_times[repeat]
                          << std::endl;
            }
        }
    }
}

void write_metadata(const std::filesystem::path& path, const Options& options,
                    const beamformer::CudaDeviceInfo& device) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open metadata file: " + path.string());
    }
    const auto write_list = [&output](const std::vector<std::size_t>& values) {
        output << '[';
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            output << values[index];
        }
        output << ']';
    };
    output << "{\n  \"gpu_name\": \"" << device.name << "\",\n"
           << "  \"compute_capability\": \"" << device.compute_major << '.'
           << device.compute_minor << "\",\n"
           << "  \"gpu_global_memory_bytes\": " << device.global_memory_bytes << ",\n"
           << "  \"cuda_driver_version\": " << device.driver_version << ",\n"
           << "  \"cuda_runtime_version\": " << device.runtime_version << ",\n"
           << "  \"n_freq\": " << beamformer::default_frequency_channels << ",\n"
           << "  \"n_ant\": ";
    write_list(options.antenna_values);
    output << ",\n  \"n_time\": ";
    write_list(options.time_values);
    output << ",\n  \"n_beams\": ";
    write_list(options.beam_values);
    output << ",\n  \"validation_n_time\": " << options.validation_time
           << ",\n  \"warmup\": " << options.warmup
           << ",\n  \"repetitions\": " << options.repetitions
           << ",\n  \"seed\": " << options.seed
           << ",\n  \"absolute_tolerance\": " << options.absolute_tolerance
           << ",\n  \"relative_tolerance\": " << options.relative_tolerance
           << ",\n  \"complex_mac_real_flops\": " << real_flops_per_complex_mac
           << ",\n  \"intensity_real_flops\": " << real_flops_per_intensity
           << ",\n  \"timed_backend\": \"cuda\""
           << ",\n  \"kernel\": \"" << kernel_name(options.kernel) << "\""
           << ",\n  \"cpu_validation_threads\": 1"
           << ",\n  \"beam_grid\": \"centered zero-padded rectangular FFT bins\""
           << ",\n  \"temporal_chunking\": false,\n"
           << "  \"voltage_input\": \"packed int4x2; one seeded spectrum repeated over time\"\n"
           << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        validate_options(options);
        if (options.dry_run) {
            const std::size_t max_time = std::max(
                options.validation_time,
                *std::max_element(options.time_values.begin(),
                                  options.time_values.end()));
            const std::size_t max_beams = *std::max_element(
                options.beam_values.begin(), options.beam_values.end());
            std::cout << "GPU-only timing matrix: "
                      << options.antenna_values.size() * options.beam_values.size()
                             * options.time_values.size()
                      << " configurations; CPU/CUDA validation T="
                      << options.validation_time << '\n';
            for (const std::size_t n_ant : options.antenna_values) {
                print_memory_estimate(n_ant, max_time, max_beams);
            }
            return 0;
        }
        const auto parent = options.output_prefix.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        const auto timings_path = with_suffix(options.output_prefix, "_timings.csv");
        const auto validation_path = with_suffix(options.output_prefix,
                                                 "_validation.csv");
        const auto metadata_path = with_suffix(options.output_prefix, "_metadata.json");
        std::ofstream timings_output(timings_path, std::ios::trunc);
        std::ofstream validation_output(validation_path, std::ios::trunc);
        if (!timings_output || !validation_output) {
            throw std::runtime_error("cannot open benchmark CSV outputs");
        }
        timings_output << std::setprecision(12);
        validation_output << std::setprecision(12);
        write_timing_header(timings_output);
        write_validation_header(validation_output);

        beamformer::CudaDeviceInfo device_info;
        for (const std::size_t n_ant : options.antenna_values) {
            run_antenna_series(options, n_ant, timings_output, validation_output,
                               device_info);
        }
        write_metadata(metadata_path, options, device_info);
        std::cout << "Benchmark complete\nWrote " << timings_path << "\nWrote "
                  << validation_path << "\nWrote " << metadata_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cpu_cuda: " << error.what() << '\n';
        return 1;
    }
}
