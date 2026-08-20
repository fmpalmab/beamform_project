#pragma once

// beam_tracker_opt_v2 — performance-only experiment layered ON TOP of the
// already-working v1 optimized tracker (src/beam_tracker_opt.cpp). v1 stays
// untouched (see PROJECT_CONSTRAINTS in beam_tracker_opt_v2.cpp); this header
// and src/beam_tracker_opt_v2.cpp are an ADDITIVE control surface.
//
// Like v1, this implements the EXACT SAME algorithm as the naive tracker
// (src/beam_tracker.cpp): one steering direction per integration window from
// tracker_window_direction (trajectory model only), one pass, no covariance /
// search / estimation of any kind. The direction is an INPUT (from the
// trajectory), never an output.
//
// The four changes vs. v1 (documented in beam_tracker_opt_v2.cpp) only touch:
//   1. how `all_weights` storage is initialized (NUMA first-touch),
//   2. redundancy of per-window tracker_window_direction calls in Pass 1,
//   3. the OpenMP region structure (one fork/join vs. two), and
//   4. __restrict__ aliasing hints on the Pass 2 inner-loop pointers.
//
// None of these four changes alter per-cell floating-point computation or the
// element-axis accumulation order. The output is therefore bit-for-bit
// identical to BOTH the naive tracker and v1 — asserted in
// tests/cpu/test_beam_tracker_opt_v2.cpp as exact `std::vector<float>` equality
// (no epsilon), mirroring the v1 regression contract.
//
// Scope guardrails
// ----------------
// Same hard scope as beam_tracker_opt.hpp: do NOT introduce covariance
// matrices, matrix factorization, grid search, quadratic peak interpolation,
// or any other data-driven direction estimation here. This file is a
// perf-micro-variant of v1, not an estimator.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, trackers
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

namespace beamformer {

// Allocate-and-return variant mirroring beam_tracker_opt_cpu_packed_intensity
// (v1) and beam_tracker_cpu_packed_intensity (naive). The returned Intensities
// is bit-equal to what the naive path (and v1) returns for the same
// (packed, dims, tracker).
Intensities beam_tracker_opt_v2_cpu_packed_intensity(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker);

// Into-variant mirroring beam_tracker_opt_cpu_packed_intensity_into (v1)
// and beam_tracker_cpu_packed_intensity_into (naive), for reusable output
// buffers (same naming convention as the existing *_into family).
void beam_tracker_opt_v2_cpu_packed_intensity_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity);

} // namespace beamformer
