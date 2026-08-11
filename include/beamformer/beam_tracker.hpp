#pragma once

// Tracker beam components.
//
// The tracker beam is a single beam whose pointing direction changes over time.
// These components are deliberately separate from the fixed-grid beamformer in
// cpu_beamformer.hpp so the tested grid path stays untouched. The tracker reuses
// the existing packed-int4 input contract, geometry, phase-steering weight
// formula, and [time][frequency][beam] intensity layout with n_beams==1.
//
// Design decisions (see info/tracker_beam_first_look.md):
//   * Direction update granularity: one direction per integration window
//     (320 spectra at a ~3.33 us spectrum period ~= 1.07 ms).
//   * Trajectory model: linear parametric placeholder until a final ephemeris
//     format is defined.
//   * One simultaneously tracked beam (n_beams == 1).
//   * Output product: float32 intensity in the same [time][freq][beam] layout
//     as the fixed-grid beamformer, so downstream stages stay unchanged.
//   * The tracker runs per node on a single local frequency shard. The two
//     shards are combined downstream at the classification node.

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace beamformer {

// Configuration of the (placeholder) linear trajectory. Directions are unit
// vectors on the celestial sphere; the source moves uniformly from
// direction_start at t == 0 toward direction_start + time_index * direction_rate
// (re-normalized to a unit vector) for every requested time index.
//
// direction_rate is expressed per *time sample* (not per integration window).
// The tracker driver is responsible for evaluating the direction once per
// integration window; the direction itself is a function of the absolute time
// sample index so the trajectory is well defined regardless of block size.
struct TrackerTrajectoryConfig {
    Vec3 direction_start = direction_from_lm(0.0F, 0.0F);
    // Linear drift per time sample, in direction-cosine space. A zero rate
    // reproduces a stationary beam (equivalent to one-beam fixed grid).
    Vec2 direction_rate_per_sample = {0.0F, 0.0F};
};

// Snapshot of the tracker configuration used by a single run. The unit of
// direction update is the integration window: the tracker computes one weight
// set per window and uses it for every spectrum in that window.
struct TrackerConfig {
    TrackerTrajectoryConfig trajectory;
    std::size_t integration_spectra = integration_direct.integration_spectra;
};

inline constexpr std::size_t tracker_beam_count = 1;

// Number of integration windows covered by n_time with the given cadence.
std::size_t tracker_window_count(std::size_t n_time,
                                 std::size_t integration_spectra);

// Direction unit vector at absolute time sample t, using the linear model.
// The direction-cosine (l, m) pair is advanced by t * rate and then projected
// back onto the unit sphere via direction_from_lm (which fixes n = sqrt(1-ll-mm)
// and clamps the cosine pair if it leaves the unit disk).
Vec3 tracker_direction(const TrackerTrajectoryConfig& trajectory, std::size_t t);

// The single direction used for integration window w. This is the
// direction at the first sample of the window, matching the "per-window
// update" cadence.
Vec3 tracker_window_direction(const TrackerTrajectoryConfig& trajectory,
                              std::size_t window,
                              std::size_t integration_spectra);

// Minimal naive CPU tracker beamformer.
//
// Strategy A from the design note: loop over integration windows, generate one
// n_beams==1 weight set per window via the existing generate_weights formula,
// and accumulate the single-beam intensity for every spectrum in the window,
// decoding signed int4 samples inline exactly like cpu_beamform_packed_intensity.
//
// The output uses the standard [time][freq][beam] layout with n_beams == 1 so
// it remains byte-compatible with write_intensities / read_weights style IO.
// Unintegrated spectra within a partial final window are still emitted (no
// truncation), matching the existing temporal-integration reference's
// "prefix-only" contract.
//
// The tracker never touches the existing cpu_beamformer.hpp code; it calls
// generate_weights with a 1-element direction vector and reuses indexing.hpp.
Intensities beam_tracker_cpu_packed_intensity(const PackedVoltage& packed,
                                             const Dimensions& dims,
                                             const TrackerConfig& tracker);

// Into-variant for reusable output buffers, mirroring the established
// cpu_beamform_*_into naming convention.
void beam_tracker_cpu_packed_intensity_into(const PackedVoltage& packed,
                                            const Dimensions& dims,
                                            const TrackerConfig& tracker,
                                            Intensities& intensity);

} // namespace beamformer
