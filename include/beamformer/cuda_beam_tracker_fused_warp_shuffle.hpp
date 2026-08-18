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
#include <memory>
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

// ---------------------------------------------------------------------------
// Batched streaming variant for the continuous 24/7 soak test.
//
// The single-window streaming path launches one kernel per integration window,
// which yields only `n_freq / 4` == 84 blocks on a 336-channel shard -- far too
// few to saturate the SMs of an H100 / RTX 5090. This class instead accumulates
// a configurable batch of `batch_size` integration windows and dispatches them
// with a SINGLE H2D copy + SINGLE kernel launch whose grid is
// `batch_size * n_freq / 4` blocks, so the GPU stays busy while the producer
// keeps feeding windows at the 1.05 ms cadence.
//
// The class owns persistent device buffers (batch voltage, per-window steering
// metadata, output) and one CUDA stream, so per-batch dispatch has no
// cudaMalloc / cudaFree / stream-create overhead. The caller owns the host
// batch buffer (e.g. a pinned ring-buffer span) and the host output buffer.
//
// The kernel is the SAME fused warp-shuffle kernel as the other variants; the
// batch is presented as a single `n_time == batch_size * integration_spectra`
// cube, so each warp still owns exactly one (window, freq) channel and the
// per-window numerical contract is unchanged.
// ---------------------------------------------------------------------------
class BatchedTrackerStream {
public:
    // `dims` describes ONE integration window (n_time == integration_spectra).
    // `tracker` carries the continuous trajectory + integration_spectra.
    // `batch_size` is the number of windows dispatched per kernel launch.
    BatchedTrackerStream(const Dimensions& dims, const TrackerConfig& tracker,
                         std::size_t batch_size,
                         FwsLoadStrategy load_strategy = FwsLoadStrategy::Direct);
    ~BatchedTrackerStream();

    BatchedTrackerStream(const BatchedTrackerStream&) = delete;
    BatchedTrackerStream& operator=(const BatchedTrackerStream&) = delete;

    // Dispatch one batch. `first_window_index` is the ABSOLUTE index of the
    // first window in the batch along the continuous trajectory (used to derive
    // each window's steering direction). `host_packed` points at `batch_size`
    // contiguous integration windows of packed int4 voltage (window-major);
    // `host_intensity` receives `batch_size * integration_spectra * n_freq`
    // float32 intensities ([time][freq][beam==1] within the batch). Both are
    // caller-owned. Synchronous: returns after the D2H copy completes.
    void process_batch(std::size_t first_window_index,
                       const std::uint8_t* host_packed, float* host_intensity);

    // Executes ONLY the GPU kernel computation for the batch on persistent device
    // buffers without Host->Device or Device->Host PCIe data transfers.
    void process_batch_kernel_only(std::size_t first_window_index);

    // Returns the GPU device kernel execution time in milliseconds from the most
    // recent process_batch or process_batch_kernel_only call (measured via CUDA events).
    float last_kernel_time_ms() const;

    std::size_t batch_size() const;
    std::size_t window_bytes() const;          // bytes per integration window
    std::size_t batch_voltage_bytes() const;   // bytes for the whole batch
    std::size_t batch_output_floats() const;   // floats for the whole batch

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
