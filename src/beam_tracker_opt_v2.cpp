#include "beamformer/beam_tracker_opt_v2.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// PROJECT_CONSTRAINTS
// ------------------
// This file is an ADDITIVE performance experiment layered on the already-working
// v1 optimized tracker (src/beam_tracker_opt.cpp). The following MUST remain
// bit-for-bit untouched (confirmed via `git diff --stat HEAD` / `git status
// --short` before and after this work):
//   * src/beam_tracker.cpp            (naive reference)
//   * src/beam_tracker_opt.cpp        (v1)
//   * src/cpu_opt_beam_tracker.cpp    (the separate DOA estimator)
//   * include/beamformer/beam_tracker_opt.hpp (v1's header)
//
// v2 implements the EXACT SAME algorithm as v1 and the naive tracker: one
// steering direction per integration window from tracker_window_direction
// (trajectory model only), one pass, NO covariance / search / estimation of
// any kind. The direction is an INPUT (from the trajectory), never an output.
//
// The four v2-only changes vs. v1 (each documented where applied below):
//   FIX 1 — NUMA first-touch: allocate `all_weights` WITHOUT value-
//           initialization so Pass 1's parallel writes, not a serial memset,
//           first-touch each page.
//   FIX 2 — Eliminate redundant tracker_window_direction calls in Pass 1 by
//           reusing the per-window directions the v2-local validation pre-scan
//           already has to compute anyway.
//   FIX 3 — Merge the two #pragma omp parallel regions into one, with two
//           #pragma omp for sections inside (one implicit barrier between
//           them, which is required regardless).
//   FIX 4 — __restrict__ on the Pass 2 inner-loop pointers (w_ptr, v) so the
//           compiler can prove all_weights and packed never alias.
//
// None of these four changes alter per-cell floating-point computation or the
// element-axis accumulation order (0 .. n_ant-1, left-to-right float MACs), so
// the per-cell result is bit-identical to BOTH v1 and the naive path. Each
// collapsed iteration writes a disjoint intensity slot, so parallel
// reassociation across (window,freq)/(window,time,freq) cannot change any
// per-cell value.

namespace beamformer {

namespace {

// v2-local duplicate of v1's validate_opt_inputs (NOT modifying v1's shared
// copy, to keep v1's file diff-free per PROJECT_CONSTRAINTS).
//
// IDENTICAL behavior to v1's validate_opt_inputs EXCEPT for one additive
// feature required by FIX 2: instead of discarding each window's computed
// direction in the serial trajectory pre-scan, it populates the caller-provided
// `window_directions` vector (one Vec3 per integration window). Pass 1 then
// indexes into `window_directions[w]` instead of recomputing
// tracker_window_direction(tracker.trajectory, w, integration_spectra) inside
// the collapsed (window, freq) loop — collapsing over (w, f) would otherwise
// recompute the SAME direction n_freq times per window.
//
// The side-effect (throw on an off-disk cosine, from the main thread, exactly
// reproducing the naive path's "throw on the first bad window, from the main
// thread" contract) is unchanged: we still call tracker_window_direction for
// every window here, serially, before any OpenMP region.
void validate_opt_v2_inputs(const Dimensions& dims,
                            const TrackerConfig& tracker,
                            const PackedVoltage& packed,
                            const Intensities& intensity,
                            std::vector<Vec3>& window_directions) {
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

    // FIX 2 (setup half): serial pre-scan over EVERY integration window that
    // reproduces the naive path's main-thread throw-on-first-bad-window
    // contract exactly (same as v1's pre-scan), AND additionally keeps the
    // computed direction for each window so Pass 1 doesn't have to recompute it.
    const std::size_t pre_window_count =
        tracker_window_count(dims.n_time, tracker.integration_spectra);
    window_directions.resize(pre_window_count);
    for (std::size_t w = 0; w < pre_window_count; ++w) {
        window_directions[w] = tracker_window_direction(
            tracker.trajectory, w, tracker.integration_spectra);
    }
}

} // namespace

Intensities beam_tracker_opt_v2_cpu_packed_intensity(
    const PackedVoltage& packed, const Dimensions& dims,
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
    beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker, intensity);
    return intensity;
}

void beam_tracker_opt_v2_cpu_packed_intensity_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity) {
    // FIX 2 (capture half): the pre-scan populates per-window directions once
    // instead of discarding them. This vector is consumed by Pass 1 below.
    std::vector<Vec3> window_directions;
    validate_opt_v2_inputs(dims, tracker, packed, intensity, window_directions);

    // Hoisted, loop-invariant setup — identical to v1 (and to the naive path's
    // per-window generate_weights, hoisted once here).
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

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

    // -------------------------------------------------------------------
    // FIX 1 — NUMA first-touch: allocate `all_weights` WITHOUT value-
    // initialization so the constructor does NOT zero-fill the buffer.
    //
    // Why: `Weights` is `std::vector<ComplexFloat>`, and `ComplexFloat` is a
    // trivial POD struct { float real; float imag; } (see complex.hpp; both
    // sizeof == 2*sizeof(float) and std::is_trivially_copyable_v are asserted).
    // A size-only `std::vector<ComplexFloat>` constructor VALUE-INITIALIZES
    // every element (no user-provided default ctor), so v1's
    //     Weights all_weights(window_count * dims.n_freq * dims.n_ant);
    // performs a single-threaded memset of ~8.26MB on the calling thread before
    // any OpenMP region runs. That memset (a) is wasted — Pass 1 writes every
    // single element unconditionally with no gaps, overwriting the zeroes in
    // full, and (b) forces first-touch NUMA placement of every page onto the
    // calling thread's socket, so Pass 1's writes (and Pass 2's reads) from
    // other sockets pay a remote-memory penalty regardless of how well the
    // parallel loops themselves are structured.
    //
    // Fix: allocate raw storage with default-init (no value-init) via
    // std::make_unique_for_overwrite<ComplexFloat[]>(n) (C++20; this TU is
    // compiled under CMAKE_CXX_STANDARD=17 from CMakeLists.txt, but gcc/clang
    // ship make_unique_for_overwrite as an extension in C++17 mode, and the
    // fallback below covers any compiler that doesn't). Pass 1 writes every
    // element unconditionally before any reader, so reading uninitialized
    // storage never happens — the first real write to each byte becomes Pass
    // 1's parallel write, letting first-touch NUMA placement follow Pass 1's
    // actual write distribution instead of landing on one socket.
    //
    // Bit-equality is preserved: a freed-from-memset allocation does not
    // change which values Pass 1 writes — Pass 1 unconditionally assigns every
    // element {cos(phase), sin(phase)}, the same assignment v1 does, so the
    // contents are bit-identical to v1's after Pass 1 completes.
    const std::size_t weight_count =
        window_count * dims.n_freq * dims.n_ant;
#if __cpp_lib_make_unique_for_overwrite >= 202002L
    const auto weights_owner =
        std::make_unique_for_overwrite<ComplexFloat[]>(weight_count);
#else
    // Fallback: operator new[] default-initializes (no zero-fill) for a trivial
    // type. This matches make_unique_for_overwrite's contract for ComplexFloat.
    struct DefaultInitDelete {
        void operator()(ComplexFloat* p) const noexcept { ::operator delete[](p); }
    };
    const std::unique_ptr<ComplexFloat, DefaultInitDelete> weights_owner(
        static_cast<ComplexFloat*>(::operator new[](weight_count * sizeof(ComplexFloat))));
#endif
    ComplexFloat* const all_weights = weights_owner.get();

    // --- Single parallel region for both passes (FIX 3) -----------------
    // One #pragma omp parallel block with two #pragma omp for sections inside
    // it, replacing v1's two separate #pragma omp parallel for regions. The
    // implicit barrier between the two `for`s is REQUIRED anyway (Pass 2 reads
    // every weight Pass 1 writes), so the merge removes one team fork/join with
    // zero loss of correctness. schedule(static) stays correct on both: every
    // iteration does identical work (except the optionally-shorter last window
    // in Pass 2), so static partitioning adds no atomic-counter overhead for
    // no load-balancing loss.
    #pragma omp parallel default(none) \
        shared(all_weights, window_directions, wavenumbers, positions, \
               packed, intensity, dims, tracker, window_count)
    {
        // --- Pass 1: precompute steering weights for every (window, freq) pair.
        // FIX 2 (consume half): `direction` is INDEXED from window_directions
        // (computed once in validate_opt_v2_inputs's serial pre-scan) instead
        // of recomputed by tracker_window_direction for every (w, f) pair —
        // collapsing over (w, f) would otherwise call tracker_window_direction
        // window_count * n_freq times instead of window_count.
        #pragma omp for collapse(2) schedule(static)
        for (std::ptrdiff_t w = 0;
             w < static_cast<ptrdiff_t>(window_count); ++w) {
            for (std::ptrdiff_t f = 0;
                 f < static_cast<ptrdiff_t>(dims.n_freq); ++f) {
                const std::size_t ws = static_cast<std::size_t>(w);
                const std::size_t fs = static_cast<std::size_t>(f);
                const Vec3& direction = window_directions[ws];
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
        }  // implicit barrier: required — Pass 2 consumes Pass 1's output.

        // --- Pass 2: accumulation. Identical math to v1/naive; FIX 4 adds
        // __restrict__ to the inner-loop pointers so the compiler can prove
        // all_weights and packed never alias (they are distinct arrays), which
        // may change autovectorization codegen. Bit-equality holds because
        // __restrict__ is a promise about aliasing, not a reordering of the
        // element-axis float MACs — the accumulation order over element is
        // still 0 .. n_ant-1 left-to-right, identical to v1/naive.
        #pragma omp for collapse(3) schedule(static)
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
                    // FIX 4: __restrict__ on both hot pointers.
                    ComplexFloat* __restrict__ const w_ptr =
                        &all_weights[weight_base];
                    const std::uint8_t* __restrict__ const v =
                        &packed[voltage_base];
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
    }  // end single parallel region
}

} // namespace beamformer
