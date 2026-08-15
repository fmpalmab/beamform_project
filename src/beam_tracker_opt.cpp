#include "beamformer/beam_tracker_opt.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace beamformer {

namespace {

// Same validation the naive tracker applies, hoisted out of the per-window
// loop so it runs once per call instead of once per window (the naive path
// re-validates dims / positions / frequencies / direction inside
// generate_weights every window).
//
// The checks here are deliberately the union of what validate_tracker_config
// and generate_weights check, so any input that the naive path would reject is
// rejected by this path too, with the same exceptions, BEFORE any work begins.
void validate_opt_inputs(const Dimensions& dims,
                         const TrackerConfig& tracker,
                         const PackedVoltage& packed,
                         const Intensities& intensity) {
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
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    const std::size_t required_output =
        dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
}

// Direction precomputed-per-window geometry helper: compute the phase-axis
// accumulation basis (the actual steering weights) for direction `dir` using
// the SAME phase expression as generate_weights in weights.cpp:
//   phase = two_pi * f / c * (pos . dir);  weight = {cosf(phase), sinf(phase)}.
//
// The double-precision intermediate and the float casts are identical to
// generate_weights, so for a given (dims, positions, frequencies, direction)
// the weights written here are bit-equal to those produce by generate_weights
// (and therefore to what the naive tracker consumes) for every (frequency,
// element). The output layout is the standard
// `weight_index(beam=0, frequency, element, dims)` indexing, but with
// n_beams == 1 that reduces to `frequency * n_ant + element` and slots into
// `weights` below contiguously per frequency — which also makes the inner
// element loop a unit-stride load over `weights[f*n_ant : (f+1)*n_ant]`,
// friendly to auto-vectorization.
//
// NOTE: this helper computes weights from a KNOWN direction. It does NOT read
// the packed data, and it contains no covariance / search / grid scan logic.
void compute_window_weights(const Dimensions& dims,
                            const std::vector<Vec3>& positions,
                            const std::vector<double>& wavenumbers,
                            const Vec3& direction,
                            Weights& weights) {
    (void)weight_index;  // indexing is referenced via the inline expression below
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        const double wave_number = wavenumbers[frequency];
        const std::size_t base = frequency * dims.n_ant;  // == weight_index(0, f, 0, dims) with n_beams==1
        for (std::size_t element = 0; element < dims.n_ant; ++element) {
            const auto& position = positions[element];
            const double delay_m =
                static_cast<double>(position[0]) * direction[0]
                + static_cast<double>(position[1]) * direction[1]
                + static_cast<double>(position[2]) * direction[2];
            const double phase = wave_number * delay_m;
            weights[base + element] = {
                static_cast<float>(std::cos(phase)),
                static_cast<float>(std::sin(phase)),
            };
        }
    }
}

} // namespace

Intensities beam_tracker_opt_cpu_packed_intensity(const PackedVoltage& packed,
                                                 const Dimensions& dims,
                                                 const TrackerConfig& tracker) {
    validate_dimensions(dims);
    if (dims.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker, intensity);
    return intensity;
}

void beam_tracker_opt_cpu_packed_intensity_into(const PackedVoltage& packed,
                                                const Dimensions& dims,
                                                const TrackerConfig& tracker,
                                                Intensities& intensity) {
    validate_opt_inputs(dims, tracker, packed, intensity);

    // Hoisted, loop-invariant setup. The naive path re-derives these every
    // window via generate_weights (which re-validates dims and re-allocates a
    // fresh Weights vector each window). Here we do all of it once.
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

    // Per-frequency wavenumber, precomputed once (it does not depend on the
    // per-window direction). Uses the same double precision expression as
    // generate_weights so the resulting phases are bit-equal for the same
    // frequency.
    std::vector<double> wavenumbers(dims.n_freq);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        if (!std::isfinite(frequencies[frequency])
            || frequencies[frequency] <= 0.0F) {
            throw std::invalid_argument("frequencies must be positive and finite");
        }
        wavenumbers[frequency] =
            two_pi * static_cast<double>(frequencies[frequency])
            / speed_of_light_m_per_s;
    }

    const std::size_t window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);

    // One reusable weights buffer (n_beams == 1 ⇒ n_freq * n_ant entries),
    // rewritten per window. Avoids the per-window allocation that the naive
    // path performs through generate_weights.
    Weights window_weights(dims.n_freq * dims.n_ant);

    for (std::size_t window = 0; window < window_count; ++window) {
        // Direction comes EXCLUSIVELY from the known trajectory model — the
        // same call the naive tracker makes. No data is read here; nothing is
        // estimated, no search grid is scanned.
        const Vec3 direction =
            tracker_window_direction(tracker.trajectory, window,
                                     tracker.integration_spectra);

        // Recompute the weight set for this window's direction using the exact
        // phase formula of generate_weights (see compute_window_weights). The
        // per-frequency wavenumbers were precomputed once from `frequencies`
        // in the loop above, so this helper consumes wavenumbers only.
        compute_window_weights(dims, positions, wavenumbers,
                               direction, window_weights);

        const std::size_t first_time = window * tracker.integration_spectra;
        const std::size_t last_time = std::min(first_time + tracker.integration_spectra,
                                                dims.n_time);

        // Parallelize across frequency. The (time, frequency) intensity cells
        // are independent: each writes to intensity_index(time, frequency, 0,
        // dims), a disjoint slot, so the parallel region has no cross-iteration
        // dependencies and needs no scratch buffers or atomics. OpenMP
        // distributes the frequency loop across threads; the element inner loop
        // (unit-stride loads over weights and packed, fixed range 0..n_ant-1)
        // is left to the compiler's auto-vectorizer.
        //
        // The per-cell accumulation over `element` is computed in the SAME
        // order as the naive tracker (0 .. n_ant-1, left-to-right float adds)
        // and uses the SAME float MACs, so bit-for-bit equality with the naive
        // path is preserved per cell. Parallelism is only across frequency,
        // which never aliases another cell, so no per-cell value can change.
        #pragma omp parallel for collapse(2) schedule(static)
        for (std::ptrdiff_t time_pt = static_cast<ptrdiff_t>(first_time);
             time_pt < static_cast<ptrdiff_t>(last_time); ++time_pt) {
            for (std::ptrdiff_t frequency = 0;
                 frequency < static_cast<ptrdiff_t>(dims.n_freq); ++frequency) {
                const std::size_t time = static_cast<std::size_t>(time_pt);
                const std::size_t freq = static_cast<std::size_t>(frequency);

                const std::size_t weight_base = freq * dims.n_ant;
                const std::size_t voltage_base =
                    (time * dims.n_freq + freq) * dims.n_ant;

                float sum_real = 0.0F;
                float sum_imag = 0.0F;
                const auto* w = &window_weights[weight_base];
                const auto* v = &packed[voltage_base];
                for (std::size_t element = 0; element < dims.n_ant; ++element) {
                    const auto sample = unpack_complex_int4(v[element]);
                    const float sample_real = static_cast<float>(sample.real);
                    const float sample_imag = static_cast<float>(sample.imag);
                    const float weight_real = w[element].real;
                    const float weight_imag = w[element].imag;
                    sum_real += weight_real * sample_real - weight_imag * sample_imag;
                    sum_imag += weight_real * sample_imag + weight_imag * sample_real;
                }
                intensity[intensity_index(time, freq, 0, dims)] =
                    sum_real * sum_real + sum_imag * sum_imag;
            }
        }
    }
}

} // namespace beamformer
