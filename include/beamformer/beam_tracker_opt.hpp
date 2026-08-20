#pragma once

// Optimized (faster) drop-in variant of the *naive* beam tracker
// (src/beam_tracker.cpp). This is a performance-only reimplementation of the
// SAME algorithm: it applies one steering direction per integration window,
// one pass over the data, and emits the standard [time][freq][beam=1] float32
// intensity cube. It performs NO direction-of-arrival estimation — the direction
// comes exclusively from the supplied trajectory model
// (tracker_window_direction), exactly as in beam_tracker.cpp.
//
// Optimizations applied here vs. the naive serial path:
//   * Loop-invariant validation hoisted out of the per-window loop (dims,
//     positions, frequencies, trajectory start/rate are validated once, not
//     once per window via generate_weights).
//   * Per-frequency wavenumber precomputed once (it does not depend on the
//     trajectory window).
//   * Weights are computed inline with the identical phase formula
//     `cos/sin(two_pi * f / c * (pos . dir))` (double precision then cast to
//     float) used by generate_weights in weights.cpp, so per-cell float outputs
//     are bit-identical to the naive path.
//   * The hot (time x frequency x element) loop is parallelized across
//     frequency with OpenMP. Once per-window weights are generated, there are
//     no cross-iteration dependencies (each (time, frequency) cell writes a
//     disjoint intensity slot), so OpenMP cleanly distributes the work without
//     any scratch-buffer complexity.
//   * The element inner loop is auto-vectorization friendly: weights are a
//     contiguous [freq][n_ant] block of ComplexFloat and the packed voltage is
//     laid out [time][freq][n_ant] with element fastest, so the weight . sample
//     complex MAC has unit-stride loads on both operands. No intrinsics are
//     used; the compiler is relied upon to vectorize the contiguous MAC (the
//     per-cell result is identical to the serial order regardless of whether
//     the compiler vectorizes, because float accumulation order over the
//     element axis is preserved element-by-element left-to-right; we do NOT
//     reorder the inner sum, which is what keeps the result bit-exact).
//
// Output contract and tolerance
// ----------------------------
// The output of beam_tracker_opt_cpu_packed_intensity is **bit-for-bit
// identical** to beam_tracker_cpu_packed_intensity for the same inputs,
// asserted in tests/cpu/test_beam_tracker_opt.cpp as `assert(naive == opt)`. This
// follows the established "naive == opt byte-equality" precedent already used
// in this codebase (tests/cpu/test_cpu_opt_beam_tracker.cpp, block 2). Bit-equality
// holds because:
//   1. The per-window direction is the same call to tracker_window_direction,
//      so the same trajectory input produces the same direction vector.
//   2. Each weight is computed by the same double-precision phase expression and
//      the same float cast used by generate_weights, so weights are bit-equal.
//   3. Each (time, frequency) cell accumulates sum_real/sum_imag over element
//      in the same order (0 .. n_ant-1) using the same float MACs and then
//      computes intensity = sum_real*sum_real + sum_imag*sum_imag. Floating
//      addition is deterministic for a fixed order; we do not reorder the
//      element-axis sum, so the per-cell result is deterministic and equal.
//   4. The ONLY reassociation is across frequency (parallelized), and each
//      (time, frequency) cell writes a disjoint output slot, so parallelism
//      cannot change any per-cell value.
//
// Scope guardrails
// ----------------
// This file must NOT introduce covariance matrices, Cholesky / matrix inverse
// factorization, grid search, quadratic peak interpolation, or any other form
// of data-driven direction estimation. The direction is an INPUT (from the
// trajectory), not an output. If you find yourself writing any of those, stop
// — that is the exact scope-drift failure this file exists to prevent. The
// out-of-scope DOA tracker (src/cpu_opt_beam_tracker.cpp) answers a different
// question and is deliberately not built upon here.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, trackers
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

namespace beamformer {

// Allocate-and-return variant mirroring
// beam_tracker_cpu_packed_intensity. The returned Intensities is bit-equal to
// what the naive path returns for the same (packed, dims, tracker).
Intensities beam_tracker_opt_cpu_packed_intensity(const PackedVoltage& packed,
                                                 const Dimensions& dims,
                                                 const TrackerConfig& tracker);

// Into-variant mirroring beam_tracker_cpu_packed_intensity_into, for reusable
// output buffers (same naming convention as the existing *_into family).
void beam_tracker_opt_cpu_packed_intensity_into(const PackedVoltage& packed,
                                                const Dimensions& dims,
                                                const TrackerConfig& tracker,
                                                Intensities& intensity);

} // namespace beamformer
