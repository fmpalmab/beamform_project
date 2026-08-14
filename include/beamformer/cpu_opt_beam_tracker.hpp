#pragma once

// Optimized CPU beam tracker — algorithmic-only counterpart to the naive
// `beamformer::BeamTracker`. Implements the six algorithmic optimizations
// (O1–O6) described in `include/beamformer/cpu_opt_beam_tracker.md`:
//
//   O1 per-frequency sample covariance + Bartlett / Capon (MVDR) spectrum
//   O2 spatial smoothing + forward-backward averaging
//   O3 coarse-to-fine hierarchical direction search
//   O4 quadratic 3-point peak interpolation
//   O5 recursive covariance with exponential forgetting (RLS-style)
//   O6 precomputed steering phase tables + incremental response
//
// The class and free functions are a byte-compatible, drop-in alternative to
// the naive `beamformer::BeamTracker` family: they reuse the existing
// `PackedVoltage` / `Intensities` / `Dimensions` / `Vec3` data contracts, the
// `read_packed_voltage` / `write_intensities` IO, and emit the standard
// `[time][freq][beam=1]` float32 intensity cube so the existing CLI
// (`tools/beam_tracker_cpu.cpp`), test fixtures (`tests/test_beam_tracker.cpp`)
// and CSV metrics (`tools/plot_tracker_results.py`) carry over unchanged.
//
// Only algorithmic optimizations are in scope here. SIMD/AVX, threading and
// FFT libraries are explicitly deferred (see the spec's "Data layout & search
// plan"). Plain portable C++17, `std::complex<float>` math preserved.
//
// Default-constructed `CpuOptTrackerConfig` reproduces the naive open-loop
// output exactly (scan disabled → supplied trajectory direction), preserving
// the regression anchor in `tests/test_beam_tracker.cpp`.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, tracker_beam_count
#include "beamformer/complex.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace beamformer {

// Estimator family for the per-frequency spatial spectrum scan (O1).
enum class TrackerEstimator {
    Bartlett,  // P(θ) = a(θ)^H R a(θ)            (no matrix inverse; default)
    Capon      // P(θ) = 1 / (a(θ)^H (R+εI)^{-1} a(θ))   (MVDR; higher resolution)
};

// Configuration of the optimized search/estimation. All knobs carry sensible
// defaults so a default-constructed struct reproduces the naive open-loop
// output exactly (zero search grid → direction supplied by the trajectory,
// same as the existing behavior; this is the back-compat / equivalence test
// anchor in tests/test_beam_tracker.cpp).
struct CpuOptTrackerConfig {
    // Steering / estimation.
    TrackerEstimator estimator = TrackerEstimator::Bartlett;

    // Coarse-to-fine search hierarchy (O3).
    //   coarse_grid_resolution: number of cells per FoV axis at level 0 (e.g. 12).
    //   refinement_levels:      number of 3x3 refinement halvings after level 0.
    //   search_fov_l, search_fov_m: half-extent of the scanned FoV in (l, m).
    // Setting coarse_grid_resolution <= 1 disables the scan entirely and the
    // tracker falls back to the supplied trajectory direction (naive behaviour).
    std::size_t coarse_grid_resolution = 1;
    std::size_t refinement_levels = 0;
    float search_fov_l = 0.2F;
    float search_fov_m = 0.2F;

    // Spatial smoothing sub-array size (O2); 0 disables smoothing.
    // Must satisfy 1 <= subarray <= n_ant; typically ~2/3 * n_ant.
    std::size_t spatial_smoothing_subarray_size = 0;

    // Recursive covariance forgetting factor (O5), 0 < λ <= 1.
    // λ = 1.0  => stationary covariance (no forgetting; equals block estimate).
    // λ < 1    => adaptive tracking with effective memory 1/(1-λ) snapshots.
    float forgetting_factor = 1.0F;

    // Diagonal loading of R for Capon numerical stability (O1): R + ε I, with
    // ε = diagonal_load * trace(R)/M. 0 disables loading (not recommended for Capon).
    float diagonal_load = 1.0e-3F;

    // Sub-grid quadratic interpolation switch (O4).
    bool enable_quadratic_peak_interp = true;

    // Per-window snapshot count reused from the naive API (one direction
    // decision per window). Mirrors TrackerConfig.integration_spectra.
    std::size_t integration_spectra = integration_direct.integration_spectra;
};

// Stateful optimized tracker: caches the precomputed steering phase tables
// (O6) and holds the recursive covariance state across windows (O5).
class CpuOptBeamTracker {
public:
    // Construct with the array geometry + frequency plan reused from the naive
    // path; the tracker precomputes the coarse steering table here (O6).
    CpuOptBeamTracker(std::vector<Vec3> positions_m,
                      std::vector<float> frequencies_hz,
                      Dimensions dims,
                      CpuOptTrackerConfig config);

    // Run the optimized pipeline over a packed-int4 shard and write the standard
    // [time][freq][beam=1] float32 intensity cube. Byte-compatible output with
    // the naive beam_tracker_cpu_packed_intensity_into.
    void run_into(const PackedVoltage& packed, Intensities& intensity);

    // Direction estimated for window w (post-search + O4 interpolation).
    // Returns the trajectory-supplied direction when scanning is disabled
    // (coarse_grid_resolution <= 1), giving the naive back-compat path.
    Vec3 window_direction(std::size_t window) const;

    const Dimensions& dimensions() const noexcept;
    const CpuOptTrackerConfig& config() const noexcept;

    // Seed the trajectory prior (the open-loop initial guess used as the
    // per-window direction when scanning is disabled, and as the level-0
    // coarse-grid centre for the first window when scanning is enabled).
    // This is the public counterpart of the stateful free-function overload's
    // implicit seeding; callers that own a `CpuOptBeamTracker` and `run_into`
    // directly must seed it before the first run.
    void seed_trajectory(const TrackerTrajectoryConfig& trajectory);

    // Per-window ("per-frame") kernel latencies in milliseconds, populated by
    // the last `run_into` call. Empty unless the translation unit was compiled
    // with `-DBEAMFORMER_TRACKER_PERF` (the benchmark does this; production
    // builds keep the vector empty → zero overhead). Read-only view.
    const std::vector<double>& per_frame_ms() const noexcept;

    // Stateful free-function overload seeds the trajectory prior here.
    friend void cpu_opt_beam_tracker_packed_intensity_into(
        const PackedVoltage& packed, const TrackerConfig& trajectory,
        CpuOptBeamTracker& tracker, Intensities& intensity);

private:
    std::vector<Vec3> positions_m_;
    std::vector<float> frequencies_hz_;
    Dimensions dims_;
    CpuOptTrackerConfig config_;
    // Cached coarse steering vectors A[f][cell][a]  (O6),
    // per-window recursive covariance R_w[f]  (O5),
    // per-window estimated direction vector.
    // (Layout / container types are an implementation detail of the .cpp.)
    struct Impl;
    std::unique_ptr<Impl> impl_;  // opaque per-run mutable state (pimpl)
};

// ---- Free-function drop-in mirrors of the naive API -----------------------

// Returns a freshly-constructed intensity cube; identical byte layout to
// beam_tracker_cpu_packed_intensity. Uses default geometry + frequencies from
// default_positions / channelized_frequencies, matching the naive CLI path.
Intensities cpu_opt_beam_tracker_packed_intensity(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& trajectory,                  // initial guess / prior
    const CpuOptTrackerConfig& opt = CpuOptTrackerConfig{});

// Into-variant for reusable output buffers.
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& trajectory,
    Intensities& intensity,
    const CpuOptTrackerConfig& opt = CpuOptTrackerConfig{});

// Stateful variant: caller owns the tracker (preferred for streaming, since
// the covariance recursion and steering cache persist across calls).
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed,
    const TrackerConfig& trajectory,
    CpuOptBeamTracker& tracker,
    Intensities& intensity);

}  // namespace beamformer
