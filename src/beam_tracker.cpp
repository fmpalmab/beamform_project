#include "beamformer/beam_tracker.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/weights.hpp"

#include <cmath>
#include <stdexcept>

namespace beamformer {

namespace {

void validate_tracker_config(const Dimensions& dims,
                              const TrackerConfig& tracker) {
    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (tracker.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
    }
    const auto& start = tracker.trajectory.direction_start;
    const double norm_squared =
        static_cast<double>(start[0]) * start[0]
        + static_cast<double>(start[1]) * start[1]
        + static_cast<double>(start[2]) * start[2];
    if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument(
            "tracker direction_start must be a finite unit vector");
    }
    for (const float component : tracker.trajectory.direction_rate_per_sample) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "tracker direction_rate_per_sample must be finite");
        }
    }
}

} // namespace

std::size_t tracker_window_count(const std::size_t n_time,
                                 const std::size_t integration_spectra) {
    if (integration_spectra == 0) {
        throw std::invalid_argument("integration_spectra must be positive");
    }
    return (n_time + integration_spectra - 1) / integration_spectra;
}

Vec3 tracker_direction(const TrackerTrajectoryConfig& trajectory,
                      const std::size_t t) {
    // The linear model advances the direction cosines (l, m) by t * rate and
    // re-projects onto the unit sphere via direction_from_lm. That helper
    // throws when the cosines leave the unit disk, so a trajectory that would
    // point the beam below the horizon is rejected explicitly rather than
    // clamped: callers must keep the full (start, t*rate) path valid. The
    // initial n component is ignored because direction_from_lm recomputes it.
    const float l = trajectory.direction_start[0]
                    + static_cast<float>(t) * trajectory.direction_rate_per_sample[0];
    const float m = trajectory.direction_start[1]
                    + static_cast<float>(t) * trajectory.direction_rate_per_sample[1];
    return direction_from_lm(l, m);
}

Vec3 tracker_window_direction(const TrackerTrajectoryConfig& trajectory,
                              const std::size_t window,
                              const std::size_t integration_spectra) {
    if (integration_spectra == 0) {
        throw std::invalid_argument("integration_spectra must be positive");
    }
    // The per-window direction is evaluated at the first sample of the window,
    // matching the agreed "one direction per integration window" cadence.
    return tracker_direction(trajectory, window * integration_spectra);
}

Intensities beam_tracker_cpu_packed_intensity(const PackedVoltage& packed,
                                             const Dimensions& dims,
                                             const TrackerConfig& tracker) {
    validate_tracker_config(dims, tracker);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    beam_tracker_cpu_packed_intensity_into(packed, dims, tracker, intensity);
    return intensity;
}

void beam_tracker_cpu_packed_intensity_into(const PackedVoltage& packed,
                                            const Dimensions& dims,
                                            const TrackerConfig& tracker,
                                            Intensities& intensity) {
    validate_tracker_config(dims, tracker);
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    const std::size_t required_output =
        dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }

    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);

    // Strategy A: one weight set per integration window, computed by reusing
    // the existing generate_weights formula with a single-element direction
    // vector. The single tracker beam is always beam index 0, so we keep the
    // standard weight_index(beam=0, frequency, element, dims) layout and the
    // standard intensity_index(time, frequency, beam=0, dims) output slot.
    for (std::size_t window = 0; window < window_count; ++window) {
        const Vec3 direction =
            tracker_window_direction(tracker.trajectory, window,
                                     tracker.integration_spectra);
        const auto window_weights =
            generate_weights(dims, positions, frequencies, std::vector<Vec3>{direction});

        const std::size_t first_time = window * tracker.integration_spectra;
        const std::size_t last_time = std::min(first_time + tracker.integration_spectra,
                                               dims.n_time);
        for (std::size_t time = first_time; time < last_time; ++time) {
            for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
                float sum_real = 0.0F;
                float sum_imag = 0.0F;
                for (std::size_t element = 0; element < dims.n_ant; ++element) {
                    const auto sample = unpack_complex_int4(
                        packed[voltage_index(time, frequency, element, dims)]);
                    const float sample_real = static_cast<float>(sample.real);
                    const float sample_imag = static_cast<float>(sample.imag);
                    const auto& weight = window_weights[weight_index(
                        0, frequency, element, dims)];
                    sum_real += weight.real * sample_real
                                - weight.imag * sample_imag;
                    sum_imag += weight.real * sample_imag
                                + weight.imag * sample_real;
                }
                intensity[intensity_index(time, frequency, 0, dims)] =
                    sum_real * sum_real + sum_imag * sum_imag;
            }
        }
    }
}

} // namespace beamformer
