#include "beamformer/temporal_integration.hpp"

#include "beamformer/indexing.hpp"

#include <algorithm>
#include <stdexcept>

namespace beamformer {

void validate_temporal_config(const TemporalIntegrationConfig& config) {
    if (config.integration_spectra == 0) {
        throw std::invalid_argument("integration_spectra must be positive");
    }
}

std::size_t integrated_time_count(const std::size_t n_time,
                                  const std::size_t integration_spectra) {
    if (integration_spectra == 0) {
        throw std::invalid_argument("integration_spectra must be positive");
    }
    if (n_time == 0) {
        throw std::invalid_argument("n_time must be positive");
    }
    return n_time / integration_spectra
           + (n_time % integration_spectra == 0 ? 0U : 1U);
}

std::size_t integrated_time_count(
    const std::size_t n_time, const TemporalIntegrationConfig& config) {
    validate_temporal_config(config);
    return integrated_time_count(n_time, config.integration_spectra);
}

std::size_t integrated_intensity_count(const Dimensions& dims,
                                       const std::size_t integration_spectra) {
    return integrated_intensity_count(
        dims, TemporalIntegrationConfig{integration_spectra});
}

std::size_t integrated_intensity_count(
    const Dimensions& dims, const TemporalIntegrationConfig& config) {
    validate_dimensions(dims);
    return integrated_time_count(dims.n_time, config)
           * dims.n_freq * dims.n_beams;
}

IntegratedIntensities cpu_integrate_intensity(
    const Intensities& intensity, const Dimensions& dims,
    const std::size_t integration_spectra) {
    validate_dimensions(dims);
    const std::size_t required_input = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() != required_input) {
        throw std::invalid_argument("intensity count does not match dimensions");
    }

    IntegratedIntensities integrated(integrated_intensity_count(
        dims, integration_spectra));
    cpu_integrate_intensity_into(
        intensity, dims, integration_spectra, integrated);
    return integrated;
}

IntegratedIntensities cpu_integrate_intensity(
    const Intensities& intensity, const Dimensions& dims,
    const TemporalIntegrationConfig& config) {
    validate_temporal_config(config);
    validate_dimensions(dims);
    const std::size_t required_input = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() != required_input) {
        throw std::invalid_argument("intensity count does not match dimensions");
    }

    IntegratedIntensities integrated(integrated_intensity_count(dims, config));
    cpu_integrate_intensity_into(
        intensity, dims, config.integration_spectra, integrated);
    return integrated;
}

void cpu_integrate_intensity_into(
    const Intensities& intensity, const Dimensions& dims,
    const std::size_t integration_spectra, IntegratedIntensities& integrated) {
    validate_dimensions(dims);
    const std::size_t required_input = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_input) {
        throw std::invalid_argument("intensity is smaller than dimensions");
    }

    const std::size_t output_time =
        integrated_time_count(dims.n_time, integration_spectra);
    const std::size_t required_output = output_time * dims.n_freq * dims.n_beams;
    if (integrated.size() < required_output) {
        throw std::invalid_argument("integrated output is smaller than dimensions");
    }

    for (std::size_t window = 0; window < output_time; ++window) {
        const std::size_t first_time = window * integration_spectra;
        const std::size_t last_time =
            std::min(first_time + integration_spectra, dims.n_time);
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                float sum = 0.0F;
                for (std::size_t time = first_time; time < last_time; ++time) {
                    sum += intensity[intensity_index(time, frequency, beam, dims)];
                }
                integrated[(window * dims.n_freq + frequency) * dims.n_beams + beam] =
                    sum;
            }
        }
    }
}

} // namespace beamformer
