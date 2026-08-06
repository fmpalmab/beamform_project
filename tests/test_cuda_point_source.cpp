#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double absolute_tolerance = 1.0e-3;
constexpr double relative_tolerance = 1.0e-5;
constexpr std::size_t injected_beam = 12;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<double> integrated_power(const beamformer::Intensities& intensity,
                                     const beamformer::Dimensions& dims,
                                     const std::size_t first_time,
                                     const std::size_t time_count) {
    require(first_time + time_count <= dims.n_time,
            "requested integration range is outside n_time");
    std::vector<double> power(dims.n_beams, 0.0);
    for (std::size_t time = first_time; time < first_time + time_count; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                power[beam] += intensity[beamformer::intensity_index(
                    time, frequency, beam, dims)];
            }
        }
    }
    return power;
}

std::size_t peak_beam(const std::vector<double>& power) {
    return static_cast<std::size_t>(
        std::distance(power.begin(), std::max_element(power.begin(), power.end())));
}

double peak_to_runner_up(const std::vector<double>& power,
                         const std::size_t expected_peak) {
    double runner_up = 0.0;
    for (std::size_t beam = 0; beam < power.size(); ++beam) {
        if (beam != expected_peak) {
            runner_up = std::max(runner_up, power[beam]);
        }
    }
    require(power[expected_peak] > runner_up,
            "injected beam is not strictly stronger than every other beam");
    return power[expected_peak] / runner_up;
}

double compare_cpu_gpu(const beamformer::Intensities& cpu,
                       const beamformer::Intensities& gpu) {
    require(cpu.size() == gpu.size(), "CPU and GPU output sizes differ");
    double maximum_absolute_error = 0.0;
    std::size_t outside_tolerance = 0;
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        require(std::isfinite(cpu[index]) && std::isfinite(gpu[index]),
                "CPU or GPU produced a non-finite intensity");
        const double absolute_error =
            std::abs(static_cast<double>(gpu[index]) - cpu[index]);
        maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
        if (absolute_error
            > absolute_tolerance
                  + relative_tolerance * std::abs(static_cast<double>(cpu[index]))) {
            ++outside_tolerance;
        }
    }
    require(outside_tolerance == 0,
            "CPU and GPU point-source outputs exceed the numerical tolerance");
    return maximum_absolute_error;
}

} // namespace

int main() {
    try {
        constexpr std::size_t n_ant = 32;
        constexpr std::size_t n_beams = 32;
        constexpr std::size_t maximum_time = 12;

        const auto positions = beamformer::default_positions(n_ant);
        const auto frequencies =
            beamformer::channelized_frequencies(beamformer::default_frequency_channels);
        // The same 32 beam centers are used for every time sample and frequency channel.
        const auto directions = beamformer::rectangular_beam_grid(n_ant);
        require(directions[injected_beam][0] != 0.0F
                    && directions[injected_beam][1] != 0.0F,
                "test source must have non-zero l and m direction cosines");

        const beamformer::Dimensions weight_dims{
            1, beamformer::default_frequency_channels, n_ant, n_beams};
        const auto weights = beamformer::generate_weights(
            weight_dims, positions, frequencies, directions);
        const auto tiled_weights = beamformer::generate_tiled_weights(
            weight_dims, positions, frequencies, directions);

        for (std::size_t n_time = 1; n_time <= maximum_time; ++n_time) {
            const beamformer::Dimensions dims{
                n_time, beamformer::default_frequency_channels, n_ant, n_beams};
            const auto packed = beamformer::make_point_source(
                dims, positions, frequencies, directions[injected_beam], 4.0F);
            const auto cpu =
                beamformer::cpu_beamform_packed_intensity(packed, weights, dims);

            for (const auto kernel : std::array{
                     beamformer::CudaBeamformerKernel::Direct,
                     beamformer::CudaBeamformerKernel::Tiled}) {
                const auto& kernel_weights =
                    kernel == beamformer::CudaBeamformerKernel::Direct
                        ? static_cast<const beamformer::Weights&>(weights)
                        : static_cast<const beamformer::Weights&>(tiled_weights);
                const auto gpu = beamformer::cuda_beamform_packed_intensity(
                    packed, kernel_weights, dims, nullptr, kernel);
                const double maximum_absolute_error = compare_cpu_gpu(cpu, gpu);
                const auto cpu_integrated = integrated_power(cpu, dims, 0, n_time);
                const auto gpu_integrated = integrated_power(gpu, dims, 0, n_time);
                require(peak_beam(cpu_integrated) == injected_beam,
                        "CPU integrated peak does not match the injected beam");
                require(peak_beam(gpu_integrated) == injected_beam,
                        "GPU integrated peak does not match the injected beam");

                for (std::size_t time = 0; time < n_time; ++time) {
                    require(peak_beam(integrated_power(cpu, dims, time, 1))
                                == injected_beam,
                            "CPU per-time peak does not match the injected beam");
                    require(peak_beam(integrated_power(gpu, dims, time, 1))
                                == injected_beam,
                            "GPU per-time peak does not match the injected beam");
                }

                const double cpu_contrast =
                    peak_to_runner_up(cpu_integrated, injected_beam);
                const double gpu_contrast =
                    peak_to_runner_up(gpu_integrated, injected_beam);
                std::cout << "point_source T=" << n_time
                          << " kernel="
                          << (kernel == beamformer::CudaBeamformerKernel::Direct
                                  ? "Direct" : "Tiled")
                          << " injected_beam=" << injected_beam
                          << " cpu_peak=" << peak_beam(cpu_integrated)
                          << " gpu_peak=" << peak_beam(gpu_integrated)
                          << " cpu_peak_to_runner_up=" << cpu_contrast
                          << " gpu_peak_to_runner_up=" << gpu_contrast
                          << " max_abs_error=" << maximum_absolute_error << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_point_source: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
