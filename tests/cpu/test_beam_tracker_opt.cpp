// Regression test for the optimized naive tracker
// (src/beam_tracker_opt.cpp) against the reference naive tracker
// (src/beam_tracker.cpp).
//
// Contract: beam_tracker_opt_cpu_packed_intensity must be **byte-for-bit
// (exact)** equal to beam_tracker_cpu_packed_intensity for the same inputs:
//
//     assert(naive == opt);
//
// This follows the established "naive == opt byte-equality" precedent already
// used in this codebase (tests/cpu/test_cpu_opt_beam_tracker.cpp, block 2). Bit-
// equality is the right tolerance here (not a float epsilon) because the opt
// path preserves the exact same per-cell float MAC order over the element axis
// as the naive path; the only reassociation is across frequency (parallelized),
// and each (time, frequency) cell writes a disjoint output slot, so no per-cell
// value can change. Computing the same weights via the same double->float phase
// formula as generate_weights keeps the weighted sums bit-equal.
//
// The opt tracker does NO direction-of-arrival estimation: the direction is
// taken from the trajectory model (tracker_window_direction) exactly as in the
// naive path. There is intentionally no search/covariance/grid coverage here —
// that is out of scope and lives in the separate DOA tracker tests.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Run both trackers over the same (packed, dims, tracker) and assert their
// Intensities outputs are exactly equal element-for-element. The into-variant
// is exercised too so the reusable-buffer path is covered by the same
// byte-equality contract.
void assert_byte_equal(const beamformer::PackedVoltage& packed,
                       const beamformer::Dimensions& dims,
                       const beamformer::TrackerConfig& tracker) {
    const auto naive =
        beamformer::beam_tracker_cpu_packed_intensity(packed, dims, tracker);
    const auto opt =
        beamformer::beam_tracker_opt_cpu_packed_intensity(packed, dims, tracker);
    assert(naive.size() == opt.size());
    assert(naive == opt);

    // Also exercise the into-variant against the naive into-variant.
    beamformer::Intensities naive_into(dims.n_time * dims.n_freq * dims.n_beams);
    beamformer::Intensities opt_into(dims.n_time * dims.n_freq * dims.n_beams);
    beamformer::beam_tracker_cpu_packed_intensity_into(packed, dims, tracker,
                                                       naive_into);
    beamformer::beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker,
                                                           opt_into);
    assert(naive_into == opt_into);
    // The into-variant must match the allocate-and-return variant too.
    assert(naive_into == naive);
}

} // namespace

int main() {
    using namespace beamformer;

    // ------------------------------------------------------------------
    // 1. Stationary point source, single integration window over the whole
    //    run: one weight set, reused across all time. This is the simplest
    //    "does the parallel path still produce identical output" test.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto source_dir = direction_from_lm(0.04F, 0.0F);
        TrackerConfig stationary;
        stationary.trajectory.direction_start = source_dir;
        stationary.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        stationary.integration_spectra = dims.n_time;
        const auto packed =
            make_point_source(dims, positions, frequencies, source_dir, 4.0F);
        assert_byte_equal(packed, dims, stationary);
    }

    // ------------------------------------------------------------------
    // 2. Drifting trajectory, multiple integration windows. The opt path must
    //    recompute one weight set per window from the trajectory, identically
    //    to the naive path, and the per-window direction must be used for the
    //    right subset of time samples.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{6, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig moving;
        moving.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        moving.trajectory.direction_rate_per_sample = {1.0e-3F, 0.0F};
        moving.integration_spectra = 3;
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, moving.trajectory, 4.0F);
        assert_byte_equal(packed, dims, moving);
    }

    // ------------------------------------------------------------------
    // 3. integration_spectra == 1: one weight set per time sample (worst case
    //    for the "hoist per-window work" optimization, since generate_weights
    //    is called n_time times in the naive path). Confirms byte-equality
    //    holds when the per-window direction changes every spectrum.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{3, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig per_spectrum;
        per_spectrum.trajectory.direction_start = direction_from_lm(0.02F, 0.02F);
        per_spectrum.trajectory.direction_rate_per_sample = {2.0e-4F, -1.0e-4F};
        per_spectrum.integration_spectra = 1;
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, per_spectrum.trajectory, 4.0F);
        assert_byte_equal(packed, dims, per_spectrum);
    }

    // ------------------------------------------------------------------
    // 4. Noise input with a stationary tracker. Exercises arbitrary int4
    //    samples (not the structured point-source pattern), so the inner
    //    complex MAC runs over the full nibble range.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{2, default_frequency_channels, 32, tracker_beam_count};
        const auto noise = make_noise(dims, 4242);
        TrackerConfig noise_tracker;
        noise_tracker.trajectory.direction_start =
            direction_from_lm(0.02F, 0.0F);
        noise_tracker.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        noise_tracker.integration_spectra = 1;
        assert_byte_equal(noise, dims, noise_tracker);
    }

    // ------------------------------------------------------------------
    // 5. n_ant == 64: byte-equality must hold for the larger element count too
    //    (the inner loop over element is the auto-vectorization target).
    // ------------------------------------------------------------------
    {
        const Dimensions dims{3, default_frequency_channels, 64, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig moving64;
        moving64.trajectory.direction_start = direction_from_lm(0.01F, 0.0F);
        moving64.trajectory.direction_rate_per_sample = {5.0e-4F, 0.0F};
        moving64.integration_spectra = 2;
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, moving64.trajectory, 4.0F);
        assert_byte_equal(packed, dims, moving64);
    }

    // ------------------------------------------------------------------
    // 6. Non-trivial drift in 2D (both l and m rates) with windows that are not
    //    a clean divisor of n_time (partial final window). Byte-equality must
    //    cover the unintegrated tail of the last window identically.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{7, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig two_d;
        two_d.trajectory.direction_start = direction_from_lm(0.01F, -0.01F);
        two_d.trajectory.direction_rate_per_sample = {7.0e-5F, -3.0e-5F};
        two_d.trajectory.direction_rate_per_sample = {7.0e-5F, -3.0e-5F};
        two_d.integration_spectra = 3;  // 7 = 2*3 + 1 → partial final window
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, two_d.trajectory, 4.0F);
        assert_byte_equal(packed, dims, two_d);
    }

    // ------------------------------------------------------------------
    // 7. Validation-rejection parity: every input that the naive tracker
    //    rejects must also be rejected by the opt tracker (same exceptions),
    //    and vice versa. This guarantees the hoisted validation did not drop
    //    a guard the naive path enforces.
    // ------------------------------------------------------------------
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto source_dir = direction_from_lm(0.04F, 0.0F);
        const auto packed =
            make_point_source(dims, positions, frequencies, source_dir, 4.0F);
        TrackerConfig stationary;
        stationary.trajectory.direction_start = source_dir;
        stationary.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        stationary.integration_spectra = dims.n_time;

        // n_beams != 1 rejected by both.
        assert(throws_invalid_argument([&] {
            const Dimensions bad{2, default_frequency_channels, 32, 2};
            static_cast<void>(
                beam_tracker_cpu_packed_intensity(packed, bad, stationary));
        }));
        assert(throws_invalid_argument([&] {
            const Dimensions bad{2, default_frequency_channels, 32, 2};
            static_cast<void>(
                beam_tracker_opt_cpu_packed_intensity(packed, bad, stationary));
        }));

        // integration_spectra == 0 rejected by both.
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary; bad.integration_spectra = 0;
            static_cast<void>(
                beam_tracker_cpu_packed_intensity(packed, dims, bad));
        }));
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary; bad.integration_spectra = 0;
            static_cast<void>(
                beam_tracker_opt_cpu_packed_intensity(packed, dims, bad));
        }));

        // Non-unit direction_start rejected by both.
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_start = {2.0F, 0.0F, 0.0F};
            static_cast<void>(
                beam_tracker_cpu_packed_intensity(packed, dims, bad));
        }));
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_start = {2.0F, 0.0F, 0.0F};
            static_cast<void>(
                beam_tracker_opt_cpu_packed_intensity(packed, dims, bad));
        }));

        // (a) Non-finite direction_rate rejected by BOTH paths identically.
        //     validate_tracker_config (naive) and validate_opt_inputs (opt) both
        //     gate on finiteness of every rate component, so a NaN is rejected by
        //     both before either tracker ever touches the trajectory.
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_rate_per_sample = {
                std::numeric_limits<float>::quiet_NaN(), 0.0F};
            static_cast<void>(
                beam_tracker_cpu_packed_intensity(packed, dims, bad));
        }));
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_rate_per_sample = {
                std::numeric_limits<float>::quiet_NaN(), 0.0F};
            static_cast<void>(
                beam_tracker_opt_cpu_packed_intensity(packed, dims, bad));
        }));

        // (b) Parallel parity: a trajectory whose (l, m) leaves the unit disk
        //     at an *evaluated* window must be rejected by BOTH paths at the
        //     same point (the shared tracker_window_direction -> tracker_direction
        //     -> direction_from_lm chain). Use integration_spectra=1 so window
        //     1 evaluates tracker_direction(t=1): start {0.5, 0.0} + rate
        //     {0.5, 0.5} -> l=1.0, m=0.5 -> l^2+m^2 = 1.25 > 1 -> throws. This
        //     guards that the opt path exercises the exact same per-window
        //     trajectory evaluation as the naive path (and does NOT silently stop
        //     early, skip a window, or clamp instead of throw).
        {
            Dimensions parity_dims{4, default_frequency_channels, 32,
                                    tracker_beam_count};
            TrackerConfig offdisk;
            offdisk.trajectory.direction_start = direction_from_lm(0.5F, 0.0F);
            offdisk.trajectory.direction_rate_per_sample = {0.5F, 0.5F};
            offdisk.integration_spectra = 1;
            assert(throws_invalid_argument([&] {
                static_cast<void>(beam_tracker_cpu_packed_intensity(
                    packed, parity_dims, offdisk));
            }));
            assert(throws_invalid_argument([&] {
                static_cast<void>(beam_tracker_opt_cpu_packed_intensity(
                    packed, parity_dims, offdisk));
            }));
        }

        // Wrong packed size rejected by both.
        assert(throws_invalid_argument([&] {
            static_cast<void>(beam_tracker_cpu_packed_intensity(
                PackedVoltage(1, 0), dims, stationary));
        }));
        assert(throws_invalid_argument([&] {
            static_cast<void>(beam_tracker_opt_cpu_packed_intensity(
                PackedVoltage(1, 0), dims, stationary));
        }));

        // Into-variant with an undersized intensity buffer rejected by both.
        assert(throws_invalid_argument([&] {
            Intensities too_small(dims.n_freq);  // far below required
            static_cast<void>(beam_tracker_cpu_packed_intensity_into(
                packed, dims, stationary, too_small));
        }));
        assert(throws_invalid_argument([&] {
            Intensities too_small(dims.n_freq);
            static_cast<void>(beam_tracker_opt_cpu_packed_intensity_into(
                packed, dims, stationary, too_small));
        }));
    }

    // ------------------------------------------------------------------
    // 8. Output-size contract. Both paths must produce identical-size
    //    Intensities cubes (n_time * n_freq * tracker_beam_count).
    // ------------------------------------------------------------------
    {
        const Dimensions dims{5, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig moving;
        moving.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        moving.trajectory.direction_rate_per_sample = {5.0e-4F, 0.0F};
        moving.integration_spectra = 2;
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, moving.trajectory, 4.0F);
        const auto naive =
            beam_tracker_cpu_packed_intensity(packed, dims, moving);
        const auto opt =
            beam_tracker_opt_cpu_packed_intensity(packed, dims, moving);
        assert(naive.size() == dims.n_time * dims.n_freq * tracker_beam_count);
        assert(opt.size() == naive.size());
        assert(naive == opt);
    }

    return 0;
}
