// Unit tests for the naive CPU tracker beam (beam_tracker.cpp).
//
// These tests deliberately exercise only the new tracker components. They reuse
// the existing geometry/weights/synthetic-data machinery to build a numerical
// reference, so a regression in the shared engine surfaces here too, but the
// tracker code itself is isolated and never modifies the fixed-grid path.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float actual, const float expected,
           const float tolerance = 1.0e-4F) {
    return std::abs(actual - expected) <= tolerance;
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

std::int8_t independent_decode_nibble(const std::uint8_t nibble) {
    const auto bits = static_cast<std::uint8_t>(nibble & 0x0F);
    return bits < 8 ? static_cast<std::int8_t>(bits)
                    : static_cast<std::int8_t>(static_cast<int>(bits) - 16);
}

float independently_calculated_power(const beamformer::PackedVoltage& packed,
                                     const beamformer::Weights& weights,
                                     const beamformer::Dimensions& dims,
                                     std::size_t time, std::size_t frequency,
                                     std::size_t beam) {
    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t element = 0; element < dims.n_ant; ++element) {
        const auto byte =
            packed[beamformer::voltage_index(time, frequency, element, dims)];
        const float sample_real =
            static_cast<float>(independent_decode_nibble(byte));
        const float sample_imag = static_cast<float>(
            independent_decode_nibble(static_cast<std::uint8_t>(byte >> 4)));
        const auto& weight =
            weights[beamformer::weight_index(beam, frequency, element, dims)];
        sum_real += weight.real * sample_real - weight.imag * sample_imag;
        sum_imag += weight.real * sample_imag + weight.imag * sample_real;
    }
    return sum_real * sum_real + sum_imag * sum_imag;
}

} // namespace

int main() {
    using namespace beamformer;

    // --- Trajectory projection --------------------------------------------
    // The linear model advances (l, m) linearly with t and re-projects via
    // direction_from_lm, so a zero rate reproduces the start exactly, and a
    // simple one-step displacement matches an explicit direction_from_lm.
    TrackerTrajectoryConfig traj_zero{
        direction_from_lm(0.0F, 0.0F), {0.0F, 0.0F}};
    assert(tracker_direction(traj_zero, 0) == direction_from_lm(0.0F, 0.0F));
    assert(tracker_direction(traj_zero, 123) == direction_from_lm(0.0F, 0.0F));

    // Rate is small enough that the trajectory stays on the unit disk even when
    // evaluated at the 320-spectrum window offsets used below (t up to 640).
    TrackerTrajectoryConfig traj_drift{
        direction_from_lm(0.02F, -0.03F), {1.0e-4F, -2.0e-4F}};
    assert(tracker_direction(traj_drift, 0) == direction_from_lm(0.02F, -0.03F));
    assert(tracker_direction(traj_drift, 10) ==
           direction_from_lm(0.02F + 10.0F * 1.0e-4F,
                             -0.03F + 10.0F * -2.0e-4F));

    // Per-window direction uses the first sample of each window.
    const std::size_t spectra = 320;
    assert(tracker_window_direction(traj_drift, 0, spectra) ==
           tracker_direction(traj_drift, 0));
    assert(tracker_window_direction(traj_drift, 1, spectra) ==
           tracker_direction(traj_drift, spectra));
    assert(tracker_window_direction(traj_drift, 2, spectra) ==
           tracker_direction(traj_drift, 2 * spectra));

    // A trajectory that leaves the unit disk is rejected (direction_from_lm
    // throws rather than clamping), surfacing bad configs early.
    TrackerTrajectoryConfig traj_offsky{direction_from_lm(0.9F, 0.0F),
                                         {0.01F, 0.0F}};
    assert(throws_invalid_argument(
        [&] { static_cast<void>(tracker_direction(traj_offsky, 20)); }));

    // --- window_count -----------------------------------------------------
    assert(tracker_window_count(0, 320) == 0);
    assert(tracker_window_count(1, 320) == 1);
    assert(tracker_window_count(320, 320) == 1);
    assert(tracker_window_count(321, 320) == 2);
    assert(tracker_window_count(640, 320) == 2);
    assert(tracker_window_count(641, 320) == 3);
    assert(throws_invalid_argument([] { static_cast<void>(tracker_window_count(8, 0)); }));

    // --- Constant trajectory equals the one-beam fixed grid ---------------
    // With zero drift and n_beams == 1, the tracker must produce byte-identical
    // output to the existing cpu_beamform_packed_intensity for that single beam,
    // for the whole (unintegrated) [t][f][b] cube. This proves the tracker
    // reuses the same index/decode/weight contract without divergence.
    const Dimensions grid_dims{4, default_frequency_channels, 32, 1};
    const auto positions = default_positions(grid_dims.n_ant);
    const auto frequencies = channelized_frequencies(grid_dims.n_freq);
    const auto source_dir = direction_from_lm(0.04F, 0.0F);

    TrackerConfig stationary;
    stationary.trajectory.direction_start = source_dir;
    stationary.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
    stationary.integration_spectra = grid_dims.n_time; // one window = whole run

    const auto packed = make_point_source(
        grid_dims, positions, frequencies, source_dir, 4.0F);
    const auto grid_weights =
        generate_weights(grid_dims, positions, frequencies, std::vector<Vec3>{source_dir});
    const auto grid_intensity =
        cpu_beamform_packed_intensity(packed, grid_weights, grid_dims);
    const auto tracker_intensity =
        beam_tracker_cpu_packed_intensity(packed, grid_dims, stationary);
    assert(grid_intensity == tracker_intensity);

    // --- Packed decode matches independent reference ----------------------
    // For a moving source on a non-trivial grid, every emitted sample must
    // match an independently calculated power using the per-window weight set.
    const Dimensions noise_dims{2, default_frequency_channels, 32, 1};
    const auto noise = make_noise(noise_dims, 4242);
    TrackerConfig noise_tracker;
    noise_tracker.trajectory.direction_start =
        direction_from_lm(0.02F, 0.02F);
    noise_tracker.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
    noise_tracker.integration_spectra = 1; // one weight set per spectrum
    const auto noise_intensity =
        beam_tracker_cpu_packed_intensity(noise, noise_dims, noise_tracker);
    const auto noise_positions = default_positions(noise_dims.n_ant);
    const auto noise_frequencies = channelized_frequencies(noise_dims.n_freq);
    for (std::size_t t = 0; t < noise_dims.n_time; ++t) {
        const auto dir = tracker_direction(noise_tracker.trajectory, t);
        const auto w = generate_weights(
            noise_dims, noise_positions, noise_frequencies, std::vector<Vec3>{dir});
        for (std::size_t f = 0; f < noise_dims.n_freq; ++f) {
            const auto expected =
                independently_calculated_power(noise, w, noise_dims, t, f, 0);
            const auto actual = noise_intensity[intensity_index(t, f, 0, noise_dims)];
            assert(close(actual, expected, 1.0e-4F));
        }
    }

    // --- Per-window weight change actually affects output -----------------
    // With a non-zero drift and integration_spectra < n_time, the tracker uses
    // a different direction for each window. A point source centered on the
    // mid-run direction should peak in the window whose direction matches it.
    const Dimensions track_dims{4, default_frequency_channels, 32, 1};
    const auto track_positions = default_positions(track_dims.n_ant);
    const auto track_frequencies = channelized_frequencies(track_dims.n_freq);
    // Build a point source at the direction of window 1's first sample.
    TrackerConfig moving;
    moving.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
    moving.trajectory.direction_rate_per_sample = {0.01F, 0.0F};
    moving.integration_spectra = 2;
    const auto target_dir = tracker_window_direction(moving.trajectory, 1, 2);
    const auto moving_source = make_point_source(
        track_dims, track_positions, track_frequencies, target_dir, 4.0F);
    const auto moving_intensity =
        beam_tracker_cpu_packed_intensity(moving_source, track_dims, moving);

    // Integrate per window and confirm window 1 has the larger total power.
    double window0 = 0.0;
    double window1 = 0.0;
    for (std::size_t t = 0; t < track_dims.n_time; ++t) {
        const bool in_window1 = t / moving.integration_spectra == 1;
        for (std::size_t f = 0; f < track_dims.n_freq; ++f) {
            const double value =
                moving_intensity[intensity_index(t, f, 0, track_dims)];
            (in_window1 ? window1 : window0) += value;
        }
    }
    assert(window1 > window0);

    // --- Validation rejections -------------------------------------------
    assert(throws_invalid_argument([&] {
        // n_beams must be exactly 1 for the tracker contract.
        const Dimensions bad{2, default_frequency_channels, 32, 2};
        beam_tracker_cpu_packed_intensity(packed, bad, stationary);
    }));
    assert(throws_invalid_argument([&] {
        TrackerConfig bad = stationary;
        bad.integration_spectra = 0;
        beam_tracker_cpu_packed_intensity(packed, grid_dims, bad);
    }));
    assert(throws_invalid_argument([&] {
        TrackerConfig bad = stationary;
        bad.trajectory.direction_start = {2.0F, 0.0F, 0.0F}; // not a unit vector
        beam_tracker_cpu_packed_intensity(packed, grid_dims, bad);
    }));
    assert(throws_invalid_argument([&] {
        // Wrong packed size.
        beam_tracker_cpu_packed_intensity(PackedVoltage(1, 0), grid_dims,
                                           stationary);
    }));

    // --- Moving point source generator -----------------------------------
    // A tracker aligned with the source's linear track must recover the bulk of
    // the integrated power; a deliberately misaligned (zero-drift) tracker must
    // recover less. This exercises beam_tracker_make_moving_point_source and
    // confirms the tracker actually "follows" the source.
    const Dimensions move_dims{4, default_frequency_channels, 32, 1};
    const auto move_positions = default_positions(move_dims.n_ant);
    const auto move_frequencies = channelized_frequencies(move_dims.n_freq);

    TrackerTrajectoryConfig source_traj{
        direction_from_lm(0.0F, 0.0F), {0.01F, 0.0F}};
    const auto moving_voltage = beam_tracker_make_moving_point_source(
        move_dims, move_positions, move_frequencies, source_traj, 4.0F);
    assert(moving_voltage.size() == voltage_sample_count(move_dims));

    TrackerConfig aligned;
    aligned.trajectory = source_traj;
    aligned.integration_spectra = 1; // recompute steering every spectrum
    const auto aligned_intensity =
        beam_tracker_cpu_packed_intensity(moving_voltage, move_dims, aligned);

    TrackerConfig misaligned;
    misaligned.trajectory.direction_start = direction_from_lm(-0.2F, 0.2F);
    misaligned.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
    misaligned.integration_spectra = 1;
    const auto misaligned_intensity =
        beam_tracker_cpu_packed_intensity(moving_voltage, move_dims, misaligned);

    double aligned_total = 0.0;
    double misaligned_total = 0.0;
    for (std::size_t index = 0; index < aligned_intensity.size(); ++index) {
        aligned_total += aligned_intensity[index];
        misaligned_total += misaligned_intensity[index];
    }
    assert(aligned_total > misaligned_total);

    // A zero-drift moving source must byte-match the fixed-grid point source,
    // since direction(t) is constant and the per-t recompute collapses to one
    // repeated spectrum (same as make_point_source's behavior).
    TrackerTrajectoryConfig static_traj{
        direction_from_lm(0.04F, 0.0F), {0.0F, 0.0F}};
    const auto static_moving = beam_tracker_make_moving_point_source(
        grid_dims, positions, frequencies, static_traj, 4.0F);
    assert(make_point_source(grid_dims, positions, frequencies,
                             static_traj.direction_start, 4.0F) == static_moving);

    return 0;
}
