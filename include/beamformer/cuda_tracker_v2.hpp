#pragma once

// CUDA tracker — additive GPU port of the v2 optimized CPU tracker
// (src/beam_tracker_opt_v2.cpp), layered ON TOP of the existing beamformer CUDA
// core. v2's CPU TU stays untouched; this header and src/cuda_beam_tracker_v2.cu
// are an additive control surface for GPU evaluation.
//
// Like the CPU v2 path (and the naive tracker), this implements the EXACT SAME
// algorithm: one steering direction per integration window from
// tracker_window_direction (trajectory model only), NO covariance / search /
// estimation. The direction is an INPUT (from the trajectory), never an output.
//
// The output intensity cube uses the standard [time][freq][beam] layout with
// n_beams == 1 (tracker_beam_count), so it is byte-compatible with the CPU v2
// output path for the same (packed, dims, tracker) input. Phase 1 of the GPU
// implementation is a literal 1:1 port of the CPU two-pass kernel and aspires to
// bit-for-bit agreement with the CPU v2 path for examined scenarios.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, Vec3 via geometry
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cstddef>
#include <vector>

namespace beamformer {

// Selector for which CUDA implementation of the v2 tracker to launch.
enum class CudaTrackerKernelV2 {
    // Phase 1: literal 1:1 port — two kernels (weights precompute +
    // accumulation) over a device-resident weight buffer. Correctness baseline.
    TwoPass,
    // Phase 2: single fused kernel — weights recomputed in registers,
    // no device weight buffer. Performance baseline.
    Fused,
    // Phase 3: fused + warp-shuffle reduction over the n_ant axis
    // (one warp per (time, freq) cell).
    WarpReduction,
};

// Allocate-and-return variant mirroring beam_tracker_opt_v2_cpu_packed_intensity.
// Runs `kernel` on the GPU and returns the [time][freq][n_beams==1] float32 cube.
Intensities cuda_tracker_v2_packed_intensity(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    CudaTrackerKernelV2 kernel = CudaTrackerKernelV2::TwoPass);

// Into-variant mirroring beam_tracker_opt_v2_cpu_packed_intensity_into, for
// reusable output buffers (same naming convention as the *_into family).
void cuda_tracker_v2_packed_intensity_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    CudaTrackerKernelV2 kernel = CudaTrackerKernelV2::TwoPass);

} // namespace beamformer
