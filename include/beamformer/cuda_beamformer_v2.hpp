#pragma once

// CUDA Beamformer V2 — Unified Single-Engine Warp-Reduction Architecture for Fixed-Grid Beamforming.
//
// V2 applies the peak-efficiency GPU optimizations from CUDA Beam Tracker V5
// to the fixed-grid voltage beamformer:
//
// 1. Unified Single-Engine Warp Reduction:
//    Eliminates block-level shared memory barriers and inter-warp synchronization.
//    Antenna elements (32, 64, 128, 256) are partitioned across the 32 warp lanes
//    (ANT_PER_LANE = N_ANT / 32) in registers, accumulating across unrolled time
//    steps and reducing in a single 5-step __shfl_down_sync tree.
//
// 2. Register Pre-Negation & Fused Multiply-Add (FFMA):
//    Pre-negates imaginary steering weights in registers to issue back-to-back
//    FFMA instructions with zero sign-flip overhead.
//
// 3. Hardware-Accelerated Bitfield Extraction:
//    PTX `bfe.s32` single-cycle signed nibble decoding directly into 32-bit registers.
//
// 4. Pointer Stride Arithmetic:
//    Eliminates 64-bit integer multiplication and division inside inner loops.
//
// 5. Fused Temporal Integration in Registers:
//    Directly accumulates power sums over integration windows in registers,
//    eliminating intermediate global memory allocations.
//
// 6. Full Pipeline Support:
//    Compatible with Float32 intensities, Fused Temporal Integration,
//    Chunk-Quantized Int8 output, CudaFrameView offline interfaces,
//    and persistent CUDA Graph execution.

#include "beamformer/config.hpp"
#include "beamformer/cuda_frame.hpp"
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/temporal_integration.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace beamformer {

// Output format selector
enum class CudaBeamformerOutputV2 {
    Float32,
    QuantizedInt8,
};

// Timing diagnostics
struct CudaBeamformerTimingsV2 {
    double setup_ms = 0.0;
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
    double temporal_integration_ms = 0.0;
    double quantization_ms = 0.0;
    double device_to_device_ms = 0.0;
    double device_to_host_ms = 0.0;
};

// Tuning configuration for V2 kernel execution
struct V2BeamformerExecutionConfig {
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8; // 2, 4, or 8
    bool enable_cuda_graph = false;
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocator Utilities
// ---------------------------------------------------------------------------
struct PinnedDeleterV2 {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVectorV2 = std::unique_ptr<T[], PinnedDeleterV2>;

PinnedVectorV2<std::uint8_t> allocate_pinned_voltage_v2(const Dimensions& dims);
PinnedVectorV2<float> allocate_pinned_intensities_v2(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Reusable Workspace Class (V2)
// ---------------------------------------------------------------------------
class CudaBeamformerWorkspaceV2 {
public:
    explicit CudaBeamformerWorkspaceV2(
        const Dimensions& capacity,
        const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{},
        std::optional<TemporalIntegrationConfig> temporal_integration = std::nullopt,
        CudaBeamformerOutputV2 output = CudaBeamformerOutputV2::Float32);
    ~CudaBeamformerWorkspaceV2();

    CudaBeamformerWorkspaceV2(const CudaBeamformerWorkspaceV2&) = delete;
    CudaBeamformerWorkspaceV2& operator=(const CudaBeamformerWorkspaceV2&) = delete;

    double setup_ms() const;
    const V2BeamformerExecutionConfig& config() const;
    
    double upload_packed_voltage(const PackedVoltage& packed, const Dimensions& dims);
    double upload_weights(const Weights& weights, const Dimensions& dims);
    double run_kernel(const Dimensions& dims);
    double download_intensity(Intensities& intensity, const Dimensions& dims);
    
    CudaBeamformerTimingsV2 run_pipeline(const PackedVoltage& packed,
                                         Intensities& intensity,
                                         const Dimensions& dims);
    CudaBeamformerTimingsV2 run_integrated_pipeline(
        const PackedVoltage& packed, IntegratedIntensities& intensity,
        const Dimensions& dims);
    CudaBeamformerTimingsV2 run_quantized_integrated_pipeline(
        const PackedVoltage& packed, QuantizedIntegratedOutput& intensity,
        const Dimensions& dims);

    // Offline single-frame path with CudaFrameView
    CudaBeamformerTimingsV2 run_device_frame(
        const CudaFrameView& voltage, const CudaFrameView& weights,
        CudaFrameView& output, CudaFrameView* quantization_parameters = nullptr);

    bool has_temporal_integration() const;
    bool has_int8_quantization() const;
    std::size_t packed_voltage_capacity_bytes() const;

    // Direct access to device buffers for streaming/pipelining
    std::uint8_t* device_packed_voltage();
    ComplexFloat* device_weights();
    float* device_intensity();
    float* device_integrated_intensity();
    void* device_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Standalone Functional APIs
// ---------------------------------------------------------------------------

// Full allocate-and-return direct beamformer using V2 unified warp reduction
Intensities cuda_beamform_v2_packed_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    CudaBeamformerTimingsV2* timings = nullptr,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});

// Into-variant: writes directly into pre-allocated output buffer
void cuda_beamform_v2_packed_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    Intensities& intensity,
    CudaBeamformerTimingsV2* timings = nullptr,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});

// Fused temporal integration variant
IntegratedIntensities cuda_beamform_v2_packed_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV2* timings = nullptr,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});

// Fused temporal integration into preallocated buffer
void cuda_beamform_v2_packed_integrated_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    IntegratedIntensities& intensity,
    CudaBeamformerTimingsV2* timings = nullptr,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});

// Quantized integrated intensity variant
QuantizedIntegratedOutput cuda_beamform_v2_packed_quantized_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV2* timings = nullptr,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});

// ---------------------------------------------------------------------------
// Device-Resident APIs (Direct GPU-to-GPU execution)
// ---------------------------------------------------------------------------
void cuda_beamform_v2_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_intensity,
    const Dimensions& dims,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{},
    void* stream = nullptr);

void cuda_beamform_v2_integrated_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_integrated_intensity,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Batched Persistent Pipeline & CUDA Graph Stream
// ---------------------------------------------------------------------------
class BatchedBeamformerStreamV2 {
public:
    BatchedBeamformerStreamV2(
        const Dimensions& dims,
        const Weights& weights,
        const V2BeamformerExecutionConfig& config = V2BeamformerExecutionConfig{});
    ~BatchedBeamformerStreamV2();

    BatchedBeamformerStreamV2(const BatchedBeamformerStreamV2&) = delete;
    BatchedBeamformerStreamV2& operator=(const BatchedBeamformerStreamV2&) = delete;

    void process_batch(const std::uint8_t* host_packed, float* host_intensity);
    void process_batch_kernel_only();

    float last_kernel_time_ms() const;
    std::uint8_t* device_packed_buffer();
    float* device_intensity_buffer();
    void* device_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
