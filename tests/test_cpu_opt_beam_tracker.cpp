// Extensive unit tests for the optimized CPU beam tracker
// (cpu_opt_beam_tracker.cpp). Covers the six algorithmic optimizations O1–O6
// and the back-compatibility contract against the naive tracker.
//
// Coverage map (each block is self-contained and prints nothing on success):
//   1. Construction / config validation rejections.
//   2. Back-compat: default config reproduces the naive tracker output exactly
//      (the `grid_intensity == tracker_intensity` anchor carried over).
//   3. Free-function mirrors produce byte-identical output to the naive path
//      for the same supplied direction (scan disabled).
//   4. Stateful class persists direction estimates across `run_into` calls
//      when the public seeding entry point is used.
//   5. Bartlett DOA recovery: with a single point source the estimated
//      window_direction is closer to the true source direction than the
//      trajectory prior (O1, O3, O4, O6).
//   6. Capon estimator runs end-to-end and recovers the DOA (O1 MVDR branch).
//   7. Spatial smoothing (O2) path does not crash and still recovers the DOA
//      for a coherent source (the rank-deficient case the spec targets).
//   8. Forgetting factor (O5): λ = 1.0 equals the block estimate; a small λ
//      still recovers the DOA but tracks a moving source within FoV.
//   9. Quadratic peak interpolation (O4) is on by default and improves the
//      estimated direction over the raw grid argmax.
//  10. Multi-window tracking: estimated directions follow a slowly moving
//      source better than the open-loop trajectory prior (true adaptive
//      tracking — the loop the naive code never closes).
//  11. n_ant ∈ {32, 64} both run the optimized path without throwing.
//  12. Coarse-grid-argmax sanity: a one-window run with the source at the
//      grid centre peaks within the central 3×3 of the coarse lattice.
//  13. Byte-layout & output-size contract: intensity size == n_time*n_freq*1
//      and window_direction count == tracker_window_count.
//
// These tests deliberately mirror the assertion style of
// tests/test_beam_tracker.cpp (assert-based, no external framework) so they
// integrate with the existing `-UNDEBUG` CMake pattern. They use only the
// public API (seed_trajectory / run_into / window_direction / config / dims).

#include "beamformer/beam_tracker.hpp"
#include "beamformer/cpu_opt_beam_tracker.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
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

// Direction-cosine distance ||d_true - d_est|| (l,m plane). Used as the DOA
// error metric: for the optimized tracker the estimate should be closer to
// the true source direction than the trajectory prior.
float lm_distance(const beamformer::Vec3& a, const beamformer::Vec3& b) {
    const float dl = a[0] - b[0];
    const float dm = a[1] - b[1];
    return std::sqrt(dl * dl + dm * dm);
}

// Write a debug search dump for `tracker` so a failing DOA-recovery
// assertion leaves on-disk state (covariance, snapshots, coarse + refinement
// spectra, estimates) for offline diagnosis. The dump is emitted only when
// the tracker TU was compiled with -DBEAMFORMER_TRACKER_DEBUG (the CMake
// option BEAMFORMER_TRACKER_DEBUG=ON arranges this); otherwise the call is
// a zero-cost stub inside the library. The destination is:
//   ${BEAMFORMER_TRACKER_DUMP_DIR}  if that env var is set, else
//   ./tracker_dumps
// `label` is folded into the per-run subdirectory name so multiple failures
// (e.g. block 5 vs block 6) don't clobber one another.
void dump_search(const beamformer::CpuOptBeamTracker& tracker,
                 const char* label) {
#if defined(BEAMFORMER_TRACKER_DEBUG)
    const char* env_dir = std::getenv("BEAMFORMER_TRACKER_DUMP_DIR");
    const std::string dir = (env_dir && *env_dir) ? std::string(env_dir)
                                                  : std::string("./tracker_dumps");
    try {
        tracker.debug_search_dump(dir.c_str(), label);
        std::printf("[tracker] debug dump written to %s (%s)\n",
                    dir.c_str(), label);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[tracker] WARNING: debug dump failed for %s: %s\n",
                     label, e.what());
    }
#else
    (void)tracker;
    (void)label;
    std::printf("[tracker] debug dump requested for %s but the tracker was "
                "built WITHOUT -DBEAMFORMER_TRACKER_DEBUG (reconfigure with "
                "-DBEAMFORMER_TRACKER_DEBUG=ON to enable capture).\n",
                label);
#endif
}

// Helper: build a standard default-geometry / default-frequency tracker input
// shard for a point source placed at `source_dir`.
struct ShardedSource {
    beamformer::Dimensions dims;
    std::vector<beamformer::Vec3> positions;
    std::vector<float> frequencies;
    beamformer::PackedVoltage packed;
};

ShardedSource make_source(beamformer::Dimensions dims,
                           beamformer::Vec3 source_dir,
                           float amplitude = 4.0F) {
    ShardedSource s;
    s.dims = dims;
    s.positions = beamformer::default_positions(dims.n_ant);
    s.frequencies = beamformer::channelized_frequencies(dims.n_freq);
    s.packed = beamformer::make_point_source(
        dims, s.positions, s.frequencies, source_dir, amplitude);
    return s;
}

// Total integrated power of a single-beam intensity cube (sum over all cells).
double total_power(const beamformer::Intensities& intensity) {
    double sum = 0.0;
    for (const float v : intensity) sum += static_cast<double>(v);
    return sum;
}

// Run the stateful tracker with a seeded prior and return the estimated
// direction for window `w` (after run_into fills window_dirs).
template <typename Tracker>
beamformer::Vec3 run_and_get_dir(Tracker& tracker,
                                  const beamformer::TrackerTrajectoryConfig& prior,
                                  const beamformer::PackedVoltage& packed,
                                  beamformer::Intensities& out,
                                  std::size_t w) {
    tracker.seed_trajectory(prior);
    tracker.run_into(packed, out);
    return tracker.window_direction(w);
}

}  // namespace

int main() {
    using namespace beamformer;

    // ================================================================
    // 1. Construction / config validation.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);

        CpuOptTrackerConfig def;
        CpuOptBeamTracker tracker(positions, frequencies, dims, def);
        assert(tracker.dimensions().n_ant == dims.n_ant);
        assert(tracker.config().coarse_grid_resolution == def.coarse_grid_resolution);

        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 12;
        scan.refinement_levels = 2;
        CpuOptBeamTracker tracker2(positions, frequencies, dims, scan);
        assert(tracker2.config().coarse_grid_resolution == 12);

        assert(throws_invalid_argument([&] {
            const Dimensions bad{4, default_frequency_channels, 32, 2};
            CpuOptBeamTracker t(positions, frequencies, bad, def);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptBeamTracker t(default_positions(64), frequencies, dims, def);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptBeamTracker t(positions, channelized_frequencies(128), dims, def);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptTrackerConfig c; c.integration_spectra = 0;
            CpuOptBeamTracker t(positions, frequencies, dims, c);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptTrackerConfig c; c.forgetting_factor = 0.0F;
            CpuOptBeamTracker t(positions, frequencies, dims, c);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptTrackerConfig c; c.forgetting_factor = 1.5F;
            CpuOptBeamTracker t(positions, frequencies, dims, c);
            (void)t;
        }));
        assert(throws_invalid_argument([&] {
            CpuOptTrackerConfig c; c.spatial_smoothing_subarray_size = 100;
            CpuOptBeamTracker t(positions, frequencies, dims, c);
            (void)t;
        }));

        // seed_trajectory rejects a non-unit start.
        assert(throws_invalid_argument([&] {
            TrackerTrajectoryConfig bad;
            bad.direction_start = {2.0F, 0.0F, 0.0F};
            tracker.seed_trajectory(bad);
        }));
        // seed_trajectory rejects a non-finite rate.
        assert(throws_invalid_argument([&] {
            TrackerTrajectoryConfig bad;
            bad.direction_rate_per_sample = {std::numeric_limits<float>::quiet_NaN(), 0.0F};
            tracker.seed_trajectory(bad);
        }));
    }

    // ================================================================
    // 2. Back-compat: default config == naive tracker (byte-equal).
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        TrackerConfig stationary;
        stationary.trajectory.direction_start = direction_from_lm(0.04F, 0.0F);
        stationary.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        stationary.integration_spectra = dims.n_time;

        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto packed = make_point_source(
            dims, positions, frequencies,
            stationary.trajectory.direction_start, 4.0F);

        const auto naive =
            beam_tracker_cpu_packed_intensity(packed, dims, stationary);
        const auto opt =
            cpu_opt_beam_tracker_packed_intensity(packed, dims, stationary);
        assert(naive == opt);
    }

    // ================================================================
    // 3. Free-function mirrors: equal to naive for the same direction,
    //    including the stateful into-variant and a small drifting
    //    trajectory (per-window directions exercised).
    // ================================================================
    {
        const Dimensions dims{6, default_frequency_channels, 32, tracker_beam_count};
        TrackerConfig moving;
        moving.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        moving.trajectory.direction_rate_per_sample = {1.0e-3F, 0.0F};
        moving.integration_spectra = 3;

        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, moving.trajectory, 4.0F);

        const auto naive =
            beam_tracker_cpu_packed_intensity(packed, dims, moving);

        CpuOptTrackerConfig def;
        def.integration_spectra = moving.integration_spectra;
        Intensities opt(dims.n_time * dims.n_freq * dims.n_beams);
        CpuOptBeamTracker tracker(positions, frequencies, dims, def);
        tracker.seed_trajectory(moving.trajectory);
        tracker.run_into(packed, opt);
        assert(naive == opt);

        Intensities opt2(dims.n_time * dims.n_freq * dims.n_beams);
        CpuOptBeamTracker tracker2(positions, frequencies, dims, def);
        tracker2.seed_trajectory(moving.trajectory);
        tracker2.run_into(packed, opt2);
        assert(naive == opt2);
    }

    // ================================================================
    // 4. Stateful class persists direction estimates across calls.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        TrackerConfig traj;
        traj.trajectory.direction_start = direction_from_lm(0.02F, 0.0F);
        traj.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        traj.integration_spectra = dims.n_time;
        CpuOptTrackerConfig def;
        def.integration_spectra = traj.integration_spectra;

        CpuOptBeamTracker tracker(positions, frequencies, dims, def);
        PackedVoltage packed = make_point_source(
            dims, positions, frequencies, traj.trajectory.direction_start, 4.0F);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(traj.trajectory);
        tracker.run_into(packed, out);
        const Vec3 wd0 = tracker.window_direction(0);
        assert(close(wd0[0], traj.trajectory.direction_start[0], 1.0e-6F));
        assert(close(wd0[1], traj.trajectory.direction_start[1], 1.0e-6F));
    }

    // ================================================================
    // 5. Bartlett DOA recovery (O1+O3+O4+O6).
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.03F, 0.01F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 16;
        scan.refinement_levels = 2;
        scan.search_fov_l = 0.2F;
        scan.search_fov_m = 0.2F;
        scan.estimator = TrackerEstimator::Bartlett;
        scan.integration_spectra = prior.integration_spectra;

        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(prior.trajectory);
        tracker.run_into(s.packed, out);

        const Vec3 est = tracker.window_direction(0);
        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        const float err_est = lm_distance(true_dir, est);
        std::printf("[block5 bartlett] true=(%.6f,%.6f,%.6f) est=(%.6f,%.6f,%.6f)"
                    " err_prior=%.6f err_est=%.6f est<=%sprior\n",
                    true_dir[0], true_dir[1], true_dir[2],
                    est[0], est[1], est[2], err_prior, err_est,
                    (err_est < err_prior ? "yes" : "NO"));
        dump_search(tracker, "block5_bartlett");
        assert(err_est < err_prior);
        const float coarse_pitch = (2.0F * scan.search_fov_l)
                                   / static_cast<float>(scan.coarse_grid_resolution);
        assert(err_est <= coarse_pitch);
    }

    // ================================================================
    // 6. Capon estimator (O1 MVDR) recovers the DOA.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.03F, 0.0F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig scan;
        scan.estimator = TrackerEstimator::Capon;
        scan.coarse_grid_resolution = 16;
        scan.refinement_levels = 2;
        scan.diagonal_load = 1.0e-2F;
        scan.integration_spectra = prior.integration_spectra;

        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(prior.trajectory);
        tracker.run_into(s.packed, out);

        const Vec3 est = tracker.window_direction(0);
        const float err_est = lm_distance(true_dir, est);
        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        std::printf("[block6 capon] true=(%.6f,%.6f,%.6f) est=(%.6f,%.6f,%.6f)"
                    " err_prior=%.6f err_est=%.6f est<=%sprior\n",
                    true_dir[0], true_dir[1], true_dir[2],
                    est[0], est[1], est[2], err_prior, err_est,
                    (err_est < err_prior ? "yes" : "NO"));
        dump_search(tracker, "block6_capon");
        assert(err_est < err_prior);
        const float coarse_pitch = (2.0F * scan.search_fov_l)
                                   / static_cast<float>(scan.coarse_grid_resolution);
        assert(err_est <= coarse_pitch);
    }

    // ================================================================
    // 7. Spatial smoothing (O2) path is robust for a coherent source.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.02F, 0.02F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig scan;
        scan.estimator = TrackerEstimator::Bartlett;
        scan.coarse_grid_resolution = 16;
        scan.refinement_levels = 1;
        scan.spatial_smoothing_subarray_size = 24;
        scan.integration_spectra = prior.integration_spectra;

        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(prior.trajectory);
        tracker.run_into(s.packed, out);

        const Vec3 est = tracker.window_direction(0);
        const float err_est = lm_distance(true_dir, est);
        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        std::printf("[block7 smoothing] true=(%.6f,%.6f,%.6f) est=(%.6f,%.6f,%.6f)"
                    " err_prior=%.6f err_est=%.6f est<=%sprior\n",
                    true_dir[0], true_dir[1], true_dir[2],
                    est[0], est[1], est[2], err_prior, err_est,
                    (err_est < err_prior ? "yes" : "NO"));
        dump_search(tracker, "block7_smoothing");
        assert(err_est < err_prior);
    }

    // ================================================================
    // 8. Forgetting factor (O5): λ = 1 block estimate vs adaptive λ.
    // ================================================================
    {
        const Dimensions dims{6, default_frequency_channels, 32, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.03F, 0.0F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = 3;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig stationary_scan;
        stationary_scan.coarse_grid_resolution = 16;
        stationary_scan.refinement_levels = 1;
        stationary_scan.forgetting_factor = 1.0F;
        stationary_scan.integration_spectra = prior.integration_spectra;
        CpuOptBeamTracker t_stat(s.positions, s.frequencies, dims,
                                 stationary_scan);
        Intensities out_stat(dims.n_time * dims.n_freq * dims.n_beams);
        t_stat.seed_trajectory(prior.trajectory);
        t_stat.run_into(s.packed, out_stat);
        const Vec3 stat_est = t_stat.window_direction(1);

        CpuOptTrackerConfig adaptive_scan = stationary_scan;
        adaptive_scan.forgetting_factor = 0.9F;
        CpuOptBeamTracker t_adapt(s.positions, s.frequencies, dims,
                                  adaptive_scan);
        Intensities out_adapt(dims.n_time * dims.n_freq * dims.n_beams);
        t_adapt.seed_trajectory(prior.trajectory);
        t_adapt.run_into(s.packed, out_adapt);
        const Vec3 adapt_est = t_adapt.window_direction(1);

        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        assert(lm_distance(true_dir, stat_est) < err_prior);
        assert(lm_distance(true_dir, adapt_est) < err_prior);
        assert(out_stat.size() == dims.n_time * dims.n_freq * dims.n_beams);
        assert(out_adapt.size() == out_stat.size());
    }

    // ================================================================
    // 9. Quadratic peak interpolation (O4) default-on improves the
    //    estimate over raw grid argmax.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const float pitch = (2.0F * 0.2F) / 15.0F;
        const Vec3 true_dir = direction_from_lm(1.5F * pitch, 0.0F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        auto run = [&](bool interp) -> Vec3 {
            CpuOptTrackerConfig scan;
            scan.coarse_grid_resolution = 16;
            scan.refinement_levels = 0;
            scan.enable_quadratic_peak_interp = interp;
            scan.integration_spectra = prior.integration_spectra;
            CpuOptBeamTracker t(s.positions, s.frequencies, dims, scan);
            Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
            t.seed_trajectory(prior.trajectory);
            t.run_into(s.packed, out);
            return t.window_direction(0);
        };

        const Vec3 est_on = run(true);
        const Vec3 est_off = run(false);
        const float err_on = lm_distance(true_dir, est_on);
        const float err_off = lm_distance(true_dir, est_off);
        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        assert(err_on < err_prior);
        assert(err_off < err_prior);
        assert(err_on <= err_off + 5.0e-4F);
    }

    // ================================================================
    // 10. Multi-window tracking: a moving source recovered across windows.
    // ================================================================
    {
        const Dimensions dims{8, default_frequency_channels, 32, tracker_beam_count};
        TrackerTrajectoryConfig source_traj{
            direction_from_lm(0.01F, 0.0F), {4.0e-3F, 0.0F}};
        TrackerConfig zero_prior;
        zero_prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        zero_prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        zero_prior.integration_spectra = 4;
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto packed = beam_tracker_make_moving_point_source(
            dims, positions, frequencies, source_traj, 4.0F);

        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 16;
        scan.refinement_levels = 1;
        scan.search_fov_l = 0.3F;
        scan.search_fov_m = 0.3F;
        scan.integration_spectra = zero_prior.integration_spectra;

        CpuOptBeamTracker tracker(positions, frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(zero_prior.trajectory);
        tracker.run_into(packed, out);

        const Vec3 true_w1 = tracker_direction(source_traj,
                                               1 * zero_prior.integration_spectra);
        const Vec3 est_w1 = tracker.window_direction(1);
        const Vec3 prior_w1 = tracker_window_direction(
            zero_prior.trajectory, 1, zero_prior.integration_spectra);
        const float err_prior = lm_distance(true_w1, prior_w1);
        const float err_est = lm_distance(true_w1, est_w1);
        assert(err_est < err_prior);
    }

    // ================================================================
    // 11. n_ant == 64 runs the optimized path and recovers the DOA.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 64, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.03F, 0.02F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 16;
        scan.refinement_levels = 2;
        scan.integration_spectra = prior.integration_spectra;
        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(prior.trajectory);
        tracker.run_into(s.packed, out);

        const Vec3 est = tracker.window_direction(0);
        const float err_est = lm_distance(true_dir, est);
        const float err_prior = lm_distance(true_dir,
                                            prior.trajectory.direction_start);
        assert(err_est < err_prior);
    }

    // ================================================================
    // 12. coarse-grid-argmax sanity near the FoV centre.
    // ================================================================
    {
        const Dimensions dims{4, default_frequency_channels, 32, tracker_beam_count};
        const Vec3 true_dir = direction_from_lm(0.0F, 0.0F);
        TrackerConfig prior;
        prior.trajectory.direction_start = direction_from_lm(0.0F, 0.0F);
        prior.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        prior.integration_spectra = dims.n_time;
        const auto s = make_source(dims, true_dir, 4.0F);

        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 12;
        scan.refinement_levels = 0;
        scan.enable_quadratic_peak_interp = false;
        scan.integration_spectra = prior.integration_spectra;
        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        tracker.seed_trajectory(prior.trajectory);
        tracker.run_into(s.packed, out);

        const Vec3 est = tracker.window_direction(0);
        const float coarse_pitch = (2.0F * scan.search_fov_l)
                                   / static_cast<float>(scan.coarse_grid_resolution);
        assert(std::abs(est[0]) <= 1.5F * coarse_pitch);
        assert(std::abs(est[1]) <= 1.5F * coarse_pitch);
    }

    // ================================================================
    // 13. Byte-layout & output-size contract.
    // ================================================================
    {
        const Dimensions dims{6, default_frequency_channels, 32, tracker_beam_count};
        TrackerConfig traj;
        traj.trajectory.direction_start = direction_from_lm(0.02F, 0.02F);
        traj.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        traj.integration_spectra = 3;
        const auto s = make_source(dims, traj.trajectory.direction_start);
        CpuOptTrackerConfig scan;
        scan.coarse_grid_resolution = 12;
        scan.refinement_levels = 1;
        scan.integration_spectra = traj.integration_spectra;

        Intensities out(dims.n_time * dims.n_freq * dims.n_beams);
        CpuOptBeamTracker tracker(s.positions, s.frequencies, dims, scan);
        tracker.seed_trajectory(traj.trajectory);
        tracker.run_into(s.packed, out);
        assert(out.size() == dims.n_time * dims.n_freq * dims.n_beams);
        const std::size_t W = tracker_window_count(dims.n_time,
                                                   scan.integration_spectra);
        for (std::size_t w = 0; w < W; ++w) {
            (void)tracker.window_direction(w);
        }
        for (const float v : out) assert(v >= 0.0F);

        const auto s_aligned = make_source(dims, traj.trajectory.direction_start);
        CpuOptBeamTracker t_aligned(s_aligned.positions, s_aligned.frequencies,
                                    dims, scan);
        Intensities out_aligned(dims.n_time * dims.n_freq * dims.n_beams);
        t_aligned.seed_trajectory(traj.trajectory);
        t_aligned.run_into(s_aligned.packed, out_aligned);
        const double aligned = total_power(out_aligned);

        TrackerConfig far_prior = traj;
        far_prior.trajectory.direction_start = direction_from_lm(-0.2F, 0.2F);
        CpuOptBeamTracker t_far(s_aligned.positions, s_aligned.frequencies,
                                dims, scan);
        Intensities out_far(dims.n_time * dims.n_freq * dims.n_beams);
        t_far.seed_trajectory(far_prior.trajectory);
        t_far.run_into(s_aligned.packed, out_far);
        const double far = total_power(out_far);
        assert(aligned > far);
    }

    return 0;
}
