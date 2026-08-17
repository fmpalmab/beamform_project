// Regression test for the v2 optimized naive tracker
// (src/beam_tracker_opt_v2.cpp) against both:
//   * the reference naive tracker (src/beam_tracker.cpp), and
//   * the v1 optimized tracker (src/beam_tracker_opt.cpp).
//
// Contract (two-way, exact):
//     assert(naive == v2);
//     assert(v1    == v2);
//
// Same byte-equality standard as v1's test_beam_tracker_opt.cpp — exact
// std::vector<float> equality (no epsilon), because the four v2 changes only
// touch memory initialization, call redundancy, region structure, and aliasing
// hints — none of them alter per-cell floating-point computation or the
// element-axis accumulation order.
//
// Scenarios mirror tests/test_beam_tracker_opt.cpp verbatim (stationary,
// drifting multi-window, integration_spectra=1, noise, n_ant=64, 2D drift with
// partial final window, validation-rejection parity, output-size contract), plus
// the explicit v1==v2 parity assertion for each.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"      // v1
#include "beamformer/beam_tracker_opt_v2.hpp"   // v2
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

// Run naive, v1, and v2 over the same (packed, dims, tracker) and assert all
// three Intensities outputs are exactly equal element-for-element. The
// into-variants are exercised too so the reusable-buffer paths are covered by
// the same byte-equality contract (naive_into == opt_into == v2_into, and each
// into-variant matches its own allocate-and-return variant).
void assert_byte_equal(const beamformer::PackedVoltage& packed,
                        const beamformer::Dimensions& dims,
                        const beamformer::TrackerConfig& tracker) {
    const auto naive =
        beamformer::beam_tracker_cpu_packed_intensity(packed, dims, tracker);
    const auto opt   =
        beamformer::beam_tracker_opt_cpu_packed_intensity(packed, dims, tracker);
    const auto v2    =
        beamformer::beam_tracker_opt_v2_cpu_packed_intensity(packed, dims, tracker);

    assert(naive.size() == opt.size());
    assert(opt.size()  == v2.size());
    assert(naive == opt);   // v1 vs naive (parity with v1's own test)
    assert(naive == v2);    // v2 vs naive (primary contract)
    assert(opt   == v2);    // v2 vs v1 (explicit v1==v2 parity)

    // Into-variants: all three match each other and match the return variants.
    beamformer::Intensities naive_into(dims.n_time * dims.n_freq * dims.n_beams);
    beamformer::Intensities opt_into(dims.n_time * dims.n_freq * dims.n_beams);
    beamformer::Intensities v2_into(dims.n_time * dims.n_freq * dims.n_beams);
    beamformer::beam_tracker_cpu_packed_intensity_into(packed, dims, tracker,
                                                       naive_into);
    beamformer::beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker,
                                                          opt_into);
    beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker,
                                                             v2_into);
    assert(naive_into == opt_into);
    assert(naive_into == v2_into);
    assert(opt_into   == v2_into);
    assert(naive_into == naive);
}

} // namespace

int main() {
    using namespace beamformer;

    // ------------------------------------------------------------------
    // 1. Stationary point source, single integration window over the whole
    //    run: one weight set, reused across all time.
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
    // 2. Drifting trajectory, multiple integration windows.
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
    // 3. integration_spectra == 1: one weight set per time sample.
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
    // 4. Noise input with a stationary tracker.
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
    //    (the inner loop over element is the auto-vectorization target, and
    //    the __restrict__ hint from FIX 4 is load-bearing here).
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
    // 6. Non-trivial drift in 2D with a partial final window.
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
    // 7. Validation-rejection parity: every input the naive tracker rejects
    //    must also be rejected by v1 AND v2 (same exceptions), and vice versa.
    //    This guards that v2's local duplicate of validate_opt_inputs (which
    //    FIX 2 extended to capture per-window directions) did not drop or relax
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

        // n_beams != 1 rejected by all three.
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
        assert(throws_invalid_argument([&] {
            const Dimensions bad{2, default_frequency_channels, 32, 2};
            static_cast<void>(
                beam_tracker_opt_v2_cpu_packed_intensity(packed, bad, stationary));
        }));

        // integration_spectra == 0 rejected by all three.
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
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary; bad.integration_spectra = 0;
            static_cast<void>(
                beam_tracker_opt_v2_cpu_packed_intensity(packed, dims, bad));
        }));

        // Non-unit direction_start rejected by all three.
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
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_start = {2.0F, 0.0F, 0.0F};
            static_cast<void>(
                beam_tracker_opt_v2_cpu_packed_intensity(packed, dims, bad));
        }));

        // Non-finite direction_rate rejected by all three.
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
        assert(throws_invalid_argument([&] {
            TrackerConfig bad = stationary;
            bad.trajectory.direction_rate_per_sample = {
                std::numeric_limits<float>::quiet_NaN(), 0.0F};
            static_cast<void>(
                beam_tracker_opt_v2_cpu_packed_intensity(packed, dims, bad));
        }));

        // Parallel parity: a trajectory whose (l, m) leaves the unit disk at an
        // evaluated window must be rejected by all three at the same point.
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
            assert(throws_invalid_argument([&] {
                static_cast<void>(beam_tracker_opt_v2_cpu_packed_intensity(
                    packed, parity_dims, offdisk));
            }));
        }

        // Wrong packed size rejected by all three.
        assert(throws_invalid_argument([&] {
            static_cast<void>(beam_tracker_cpu_packed_intensity(
                PackedVoltage(1, 0), dims, stationary));
        }));
        assert(throws_invalid_argument([&] {
            static_cast<void>(beam_tracker_opt_cpu_packed_intensity(
                PackedVoltage(1, 0), dims, stationary));
        }));
        assert(throws_invalid_argument([&] {
            static_cast<void>(beam_tracker_opt_v2_cpu_packed_intensity(
                PackedVoltage(1, 0), dims, stationary));
        }));

        // Into-variant with an undersized intensity buffer rejected by all three.
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
        assert(throws_invalid_argument([&] {
            Intensities too_small(dims.n_freq);
            static_cast<void>(beam_tracker_opt_v2_cpu_packed_intensity_into(
                packed, dims, stationary, too_small));
        }));
    }

    // ------------------------------------------------------------------
    // 8. Output-size contract. All three paths must produce identical-size
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
        const auto v2 =
            beam_tracker_opt_v2_cpu_packed_intensity(packed, dims, moving);
        assert(naive.size() == dims.n_time * dims.n_freq * tracker_beam_count);
        assert(opt.size()   == naive.size());
        assert(v2.size()    == naive.size());
        assert(naive == opt);
        assert(naive == v2);
    }

    return 0;
}
