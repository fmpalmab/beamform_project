#pragma once

// CUDA tracker — Fused Warp-Shuffle optimized kernel (Phase 4).
//
// This header is the additive control surface for the *optimized* GPU tracker
// that merges the single-pass memory layout of the Phase 2 `CUDA Fused` kernel
// with the register-level `__shfl_down_sync` reduction primitives of the Phase
// 3 `CUDA WarpReduction` kernel. The existing `cuda_tracker_v2.hpp` API and its
// three legacy kernels (TwoPass / Fused / WarpReduction) remain untouched; this
// module exposes a separate, dedicated entry-point family so benchmarks and
// tests can evaluate the optimized path in isolation without disturbing the
// established v2 surface.
//
// Algorithm (bit-compatible at the float-cell level with the v2 WarpReduction
// kernel, i.e. same double-precision phase + single-precision MAC, same
// __shfl_down_sync pairwise reduction order — so it inherits the same tolerance
// contract rather than the strict bit-for-bit equality of the TwoPass/Fused
// paths):
//
//   * Grid:  one warp (32 lanes) per (window, freq) channel. `blockDim(32, 4)`
//     packs 4 warps / 128 threads per block to saturate the SM scheduler.
//   * Per warp: the steering weight for the warp's *own* antennas is computed
//     once per window in registers (zero DRAM weight traffic), then reused
//     across the 320 (integration_spectra) time samples of that window.
//   * Per time sample: each lane decodes its int4 antenna sample(s), accumulates
//     a partial complex voltage in registers, then the warp reduces the 32
//     lane-partials to a single complex beamformed voltage via a 5-step
//     `__shfl_down_sync` tree. Lane 0 computes |V|^2 and writes one float32 per
//     (time, freq) cell directly to global memory.
//
// The output cube uses the standard [time][freq][beam] layout with
// n_beams == 1 (tracker_beam_count), byte-compatible with the CPU v2 output
// path for the same (packed, dims, tracker) input.

#include "beamformer/beam_tracker.hpp"  // TrackerConfig, tracker_*, Vec3 via geometry
#include "beamformer/config.hpp"        // Dimensions, validate_dimensions
#include "beamformer/formats.hpp"        // PackedVoltage, Intensities

#include <cstddef>
#include <vector>

namespace beamformer {

// Selector for the optional shared-memory staging strategy used by the fused
// warp-shuffle kernel when loading the per-window voltage window from VRAM.
enum class FwsLoadStrategy {
    // Direct coalesced loads from global memory into registers (default for the
    // RTX 5090 — the int4 voltage window is small enough that the L2 cache
    // absorbs the working set, and avoiding shared-memory staging removes one
    // synchronization barrier from the hot loop).
    Direct,
    // Stage the per-time-sample antenna row through shared memory before the
    // warp reduction (useful on architectures with a thinner L2 or when several
    // warps in the same block share frequency-locality).
    SharedMemory,
};

// Allocate-and-return variant mirroring cuda_tracker_v2_packed_intensity.
// Runs the fused warp-shuffle kernel on the GPU and returns the
// [time][freq][n_beams==1] float32 cube.
Intensities cuda_beam_tracker_fused_warp_shuffle(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker,
    FwsLoadStrategy load_strategy = FwsLoadStrategy::Direct);

// Into-variant mirroring cuda_tracker_v2_packed_intensity_into, for reusable
// output buffers (same *_into naming convention as the rest of the codebase).
void cuda_beam_tracker_fused_warp_shuffle_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    FwsLoadStrategy load_strategy = FwsLoadStrategy::Direct);

// Streaming variant: pipelines the Host->Device transfer of the packed voltage
// with kernel execution across `n_streams` CUDA streams so the PCIe Gen 5 H2D
// copy latency is fully hidden behind compute on the RTX 5090. `n_streams`
// must be >= 2; values are clamped to [2, 4]. Writes into the caller-owned
// `intensity` cube (same byte layout as the non-streaming variants).
void cuda_beam_tracker_fused_warp_shuffle_stream(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& tracker, Intensities& intensity,
    std::size_t n_streams = 3,
    FwsLoadStrategy load_strategy = FwsLoadStrategy::Direct);

} // namespace beamformer
