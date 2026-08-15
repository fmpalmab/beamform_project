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

    // Parallelization strategy (two-pass, fully-parallel across windows AND
    // frequency, single fork/join per pass)
    // -----------------------------------------------------------------------
    // The naive tracker is serial; its per-window overhead is generate_weights
    // (re-validation + reallocation), which the hoisting above removes. For the
    // multi-core win we use TWO OpenMP parallel regions, each collapsed over a
    // large, uniform iteration space so the machine stays saturated regardless
    // of window_count (warned against earlier: window_count alone — e.g. ~48 at
    // the default n_time=15360, integration_spectra=320 — can underutilize 96+
    // threads; collapsing across (window, freq) and (window, time, freq) keeps
    // the iteration count at window_count * n_freq and
    // window_count * integration_spectra * n_freq respectively).
    //
    // Pass 1 (weights, collapse(2) over (window, freq)): precompute every
    // (window, freq, element) steering weight into one contiguous buffer using
    // the exact phase formula of generate_weights (double -> float cos/sin).
    // Pass 2 (accumulation, collapse(3) over (window, time_in_window, freq)):
    // weight . int4-sample complex MAC + intensity = sum_real^2 + sum_imag^2.
    //
    // schedule(static) is correct: every iteration does essentially identical
    // work (same integration_spectra, n_freq, n_ant) except possibly a shorter
    // last window, so dynamic scheduling would add atomic counter traffic for
    // zero load-balancing benefit.
    //
    // Bit-equality preservation: the per-cell computation is IDENTICAL to the
    // naive tracker — same per-window direction via the same
    // tracker_window_direction call, the same weight phase formula, and the
    // SAME element-axis accumulation order (0 .. n_ant-1, left-to-right float
    // MACs). Weights are computed once per (window, freq, element) and consumed
    // once; the per-cell float result is therefore bit-identical to the serial
    // naive path. Each collapsed iteration writes a disjoint intensity slot
    // (intensity_index(time, freq, 0, dims)), so no per-cell value can differ.
    // Perhaps-note: a future optimization is to make `all_weights` a caller-
    // owned persistent buffer (reused frame-to-frame) instead of allocating it
    // per call — left out here to keep the API byte-compatible with the naive
    // path and because frame-rate-driven allocation churn is unmeasured.

    Weights all_weights(window_count * dims.n_freq * dims.n_ant);

    // Pass 1: precompute steering weights for every (window, freq) pair.
    // Iteration space = window_count * n_freq (≥ 48*336 >> typical thread
    // count), so this fills the machine even when window_count alone would not.
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::ptrdiff_t w = 0;
         w < static_cast<ptrdiff_t>(window_count); ++w) {
        for (std::ptrdiff_t f = 0;
             f < static_cast<ptrdiff_t>(dims.n_freq); ++f) {
            const std::size_t ws = static_cast<std::size_t>(w);
            const std::size_t fs = static_cast<std::size_t>(f);
            // Direction comes EXCLUSIVELY from the known trajectory model —
            // the same call the naive tracker makes. No data is read here;
            // nothing is estimated, no search grid is scanned.
            const Vec3 direction = tracker_window_direction(
                tracker.trajectory, ws, tracker.integration_spectra);
            const double wave_number = wavenumbers[fs];
            const std::size_t base = (ws * dims.n_freq + fs) * dims.n_ant;
            for (std::size_t element = 0; element < dims.n_ant; ++element) {
                const auto& position = positions[element];
                const double delay_m =
                    static_cast<double>(position[0]) * direction[0]
                    + static_cast<double>(position[1]) * direction[1]
                    + static_cast<double>(position[2]) * direction[2];
                const double phase = wave_number * delay_m;
                all_weights[base + element] = {
                    static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase)),
                };
            }
        }
    }

    // Pass 2: accumulation over all windows. collapse(3) over
    // (window, time_in_window, freq); the inner element loop stays serial and
    // unit-stride over both all_weights[(w*n_freq+f)*n_ant ..] and
    // packed[(time*n_freq+f)*n_ant ..], so the compiler can auto-vectorize the
    // 64-wide complex MAC with AVX. The iteration space is
    // window_count * integration_spectra * n_freq — maximally parallel.
    // The last window may be partial, so guard `time >= dims.n_time`.
    #pragma omp parallel for collapse(3) schedule(static)
    for (std::ptrdiff_t w = 0;
         w < static_cast<ptrdiff_t>(window_count); ++w) {
        for (std::ptrdiff_t t = 0;
             t < static_cast<ptrdiff_t>(tracker.integration_spectra); ++t) {
            for (std::ptrdiff_t f = 0;
                 f < static_cast<ptrdiff_t>(dims.n_freq); ++f) {
                const std::size_t ws = static_cast<std::size_t>(w);
                const std::size_t fs = static_cast<std::size_t>(f);
                const std::size_t time =
                    ws * tracker.integration_spectra
                    + static_cast<std::size_t>(t);
                if (time >= dims.n_time) continue;  // partial final window

                const std::size_t weight_base = (ws * dims.n_freq + fs) * dims.n_ant;
                const std::size_t voltage_base =
                    (time * dims.n_freq + fs) * dims.n_ant;

                float sum_real = 0.0F;
                float sum_imag = 0.0F;
                const auto* w_ptr = &all_weights[weight_base];
                const auto* v = &packed[voltage_base];
                for (std::size_t element = 0; element < dims.n_ant; ++element) {
                    const auto sample = unpack_complex_int4(v[element]);
                    const float sample_real = static_cast<float>(sample.real);
                    const float sample_imag = static_cast<float>(sample.imag);
                    const float weight_real = w_ptr[element].real;
                    const float weight_imag = w_ptr[element].imag;
                    sum_real += weight_real * sample_real
                                - weight_imag * sample_imag;
                    sum_imag += weight_real * sample_imag
                                + weight_imag * sample_real;
                }
                intensity[intensity_index(time, fs, 0, dims)] =
                    sum_real * sum_real + sum_imag * sum_imag;
            }
        }
    }
}

} // namespace beamformer
