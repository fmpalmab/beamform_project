#pragma once

// CUDA Beamformer V3 — Next-Generation Multi-Beam Warp-Cooperative Architecture
// for High-Throughput Fixed-Grid Voltage Beamforming.
//
// Key Architectural Enhancements over V2:
//
// 1. Dual-Beam Warp Co-Execution (Memory Bandwidth Halving):
//    In fixed-grid beamforming, all beams evaluate identical input voltage tensors
//    for a given (time, frequency) cell. V3 introduces dual-beam warp co-execution
//    (`BEAMS_PER_WARP = 2`), preloading steering weights for two beams into registers
//    and sharing each unpacked voltage sample across both beam accumulations.
//    This cuts GPU global memory voltage traffic by exactly 50% and doubles arithmetic intensity.
//
// 2. Hardware-Accelerated PTX Bitfield Extraction:
//    PTX `bfe.s32` single-cycle signed nibble decoding directly into 32-bit registers.
//
// 3. Register Pre-Negation & Back-to-Back FFMA:
//    Pre-negates imaginary steering weights in registers (`nw_i = -w_i`) to issue
//    continuous fused multiply-accumulate instructions without sign flips.
//
// 4. Stride-Based Pointer Arithmetic:
//    Lifts all multi-dimensional indexing arithmetic out of the inner loop,
//    stepping pointers by uniform constant strides.
//
// 5. Dual-Beam Fused Temporal Integration in Registers:
//    Directly accumulates power sums over integration windows in registers across
//    both co-executed beams simultaneously, eliminating intermediate global memory allocations.
//
// 6. Complete Production Pipeline Support:
//    Fully compatible with Float32 direct intensities, Fused Temporal Integration,
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
enum class CudaBeamformerOutputV3 {
    Float32,
    QuantizedInt8,
};

// Timing diagnostics
struct CudaBeamformerTimingsV3 {
    double setup_ms = 0.0;
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
    double temporal_integration_ms = 0.0;
    double quantization_ms = 0.0;
    double device_to_device_ms = 0.0;
    double device_to_host_ms = 0.0;
};

// Tuning configuration for V3 kernel execution
struct V3BeamformerExecutionConfig {
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8; // 2, 4, or 8
    std::size_t beams_per_warp = 2; // 1 or 2
    bool enable_cuda_graph = false;
};

// ---------------------------------------------------------------------------
// Pinned Host Memory Allocator Utilities
// ---------------------------------------------------------------------------
struct PinnedDeleterV3 {
    void operator()(void* ptr) const noexcept;
};

template <typename T>
using PinnedVectorV3 = std::unique_ptr<T[], PinnedDeleterV3>;

PinnedVectorV3<std::uint8_t> allocate_pinned_voltage_v3(const Dimensions& dims);
PinnedVectorV3<float> allocate_pinned_intensities_v3(const Dimensions& dims);

// ---------------------------------------------------------------------------
// Reusable Workspace Class (V3)
// ---------------------------------------------------------------------------
class CudaBeamformerWorkspaceV3 {
public:
    explicit CudaBeamformerWorkspaceV3(
        const Dimensions& capacity,
        const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{},
        std::optional<TemporalIntegrationConfig> temporal_integration = std::nullopt,
        CudaBeamformerOutputV3 output = CudaBeamformerOutputV3::Float32);
    ~CudaBeamformerWorkspaceV3();

    CudaBeamformerWorkspaceV3(const CudaBeamformerWorkspaceV3&) = delete;
    CudaBeamformerWorkspaceV3& operator=(const CudaBeamformerWorkspaceV3&) = delete;

    double setup_ms() const;
    const V3BeamformerExecutionConfig& config() const;
    
    double upload_packed_voltage(const PackedVoltage& packed, const Dimensions& dims);
    double upload_weights(const Weights& weights, const Dimensions& dims);
    double run_kernel(const Dimensions& dims);
    double download_intensity(Intensities& intensity, const Dimensions& dims);
    
    CudaBeamformerTimingsV3 run_pipeline(const PackedVoltage& packed,
                                         Intensities& intensity,
                                         const Dimensions& dims);
    CudaBeamformerTimingsV3 run_integrated_pipeline(
        const PackedVoltage& packed, IntegratedIntensities& intensity,
        const Dimensions& dims);
    CudaBeamformerTimingsV3 run_quantized_integrated_pipeline(
        const PackedVoltage& packed, QuantizedIntegratedOutput& intensity,
        const Dimensions& dims);

    // Offline single-frame path with CudaFrameView
    CudaBeamformerTimingsV3 run_device_frame(
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

// Full allocate-and-return direct beamformer using V3 dual-beam warp co-execution
Intensities cuda_beamform_v3_packed_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    CudaBeamformerTimingsV3* timings = nullptr,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});

// Into-variant: writes directly into pre-allocated output buffer
void cuda_beamform_v3_packed_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    Intensities& intensity,
    CudaBeamformerTimingsV3* timings = nullptr,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});

// Fused temporal integration variant
IntegratedIntensities cuda_beamform_v3_packed_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV3* timings = nullptr,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});

// Fused temporal integration into preallocated buffer
void cuda_beamform_v3_packed_integrated_intensity_into(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    IntegratedIntensities& intensity,
    CudaBeamformerTimingsV3* timings = nullptr,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});

// Quantized integrated intensity variant
QuantizedIntegratedOutput cuda_beamform_v3_packed_quantized_integrated_intensity(
    const PackedVoltage& packed,
    const Weights& weights,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimingsV3* timings = nullptr,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});

// ---------------------------------------------------------------------------
// Device-Resident APIs (Direct GPU-to-GPU execution)
// ---------------------------------------------------------------------------
void cuda_beamform_v3_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_intensity,
    const Dimensions& dims,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{},
    void* stream = nullptr);

void cuda_beamform_v3_integrated_device_resident(
    const std::uint8_t* d_packed,
    const ComplexFloat* d_weights,
    float* d_integrated_intensity,
    const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{},
    void* stream = nullptr);

// ---------------------------------------------------------------------------
// Batched Persistent Pipeline & CUDA Graph Stream (V3)
// ---------------------------------------------------------------------------
class BatchedBeamformerStreamV3 {
public:
    BatchedBeamformerStreamV3(
        const Dimensions& dims,
        const Weights& weights,
        const V3BeamformerExecutionConfig& config = V3BeamformerExecutionConfig{});
    ~BatchedBeamformerStreamV3();

    BatchedBeamformerStreamV3(const BatchedBeamformerStreamV3&) = delete;
    BatchedBeamformerStreamV3& operator=(const BatchedBeamformerStreamV3&) = delete;

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
