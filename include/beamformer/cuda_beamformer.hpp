#pragma once

#include "beamformer/config.hpp"
#include "beamformer/cuda_frame.hpp"
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/temporal_integration.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace beamformer {

enum class CudaBeamformerOutput {
    Float32,
    QuantizedInt8,
};

struct CudaBeamformerTimings {
    double setup_ms = 0.0;
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
    // Kept for compatibility; fused modes report zero and include this work in kernel_ms.
    double temporal_integration_ms = 0.0;
    double quantization_ms = 0.0;
    double device_to_device_ms = 0.0;
    double device_to_host_ms = 0.0;
};

struct CudaDeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t global_memory_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

CudaDeviceInfo cuda_device_info();

// Reuses device buffers, stream, and events across benchmark iterations.
// Weights are uploaded separately so they can remain resident while voltage
// blocks and intensity products move for every pipeline iteration.
class CudaBeamformerWorkspace {
  public:
    explicit CudaBeamformerWorkspace(
        const Dimensions& capacity,
        CudaBeamformerKernel kernel = CudaBeamformerKernel::Direct,
        std::optional<TemporalIntegrationConfig> temporal_integration = std::nullopt,
        CudaBeamformerOutput output = CudaBeamformerOutput::Float32);
    ~CudaBeamformerWorkspace();

    CudaBeamformerWorkspace(const CudaBeamformerWorkspace&) = delete;
    CudaBeamformerWorkspace& operator=(const CudaBeamformerWorkspace&) = delete;

    double setup_ms() const;
    CudaBeamformerKernel kernel() const;
    double upload_packed_voltage(const PackedVoltage& packed, const Dimensions& dims);
    double upload_weights(const Weights& weights, const Dimensions& dims);
    double run_kernel(const Dimensions& dims);
    double download_intensity(Intensities& intensity, const Dimensions& dims);
    CudaBeamformerTimings run_pipeline(const PackedVoltage& packed,
                                       Intensities& intensity,
                                       const Dimensions& dims);
    CudaBeamformerTimings run_integrated_pipeline(
        const PackedVoltage& packed, IntegratedIntensities& intensity,
        const Dimensions& dims);
    CudaBeamformerTimings run_quantized_integrated_pipeline(
        const PackedVoltage& packed, QuantizedIntegratedOutput& intensity,
        const Dimensions& dims);

    // Offline single-frame path. Input, weights, and final output are
    // non-owned device buffers described by CudaFrameView. Intermediate
    // beamformed/integrated buffers remain workspace-owned.
    CudaBeamformerTimings run_device_frame(
        const CudaFrameView& voltage, const CudaFrameView& weights,
        CudaFrameView& output, CudaFrameView* quantization_parameters = nullptr);

    bool has_temporal_integration() const;
    bool has_int8_quantization() const;

    // Structural validation guard: production input storage is one byte per
    // packed complex sample, never a ComplexFloat voltage tensor.
    std::size_t packed_voltage_capacity_bytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Direct packed-voltage beamformer with one CUDA thread per
// [time][local_frequency][beam] output. Each thread decodes signed complex
// int4 bytes inside its element loop; no full complex-float voltage tensor is
// allocated or transferred. Weights and output remain float32.
Intensities cuda_beamform_packed_intensity(const PackedVoltage& packed,
                                           const Weights& weights,
                                           const Dimensions& dims,
                                           CudaBeamformerTimings* timings = nullptr,
                                           CudaBeamformerKernel kernel =
                                               CudaBeamformerKernel::Direct);

// GPU temporal integration is fused into the selected beamforming kernel, so
// this path writes [integration_window][frequency][beam] directly without a
// full per-spectrum intensity tensor. Tiled supports 10 and 320 spectra;
// Direct accepts only 320 as a debugging path. The upchannelizer is not
// implemented or coupled here.
IntegratedIntensities cuda_beamform_packed_integrated_intensity(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimings* timings = nullptr,
    CudaBeamformerKernel kernel = CudaBeamformerKernel::Tiled);

// CHIME-style chunk quantization happens after float32 temporal integration.
// The returned int8 tensor preserves [integration_window][frequency][beam];
// one offset/scale pair is returned for every 1 x 16 x 16 local chunk.
QuantizedIntegratedOutput cuda_beamform_packed_quantized_integrated_intensity(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimings* timings = nullptr,
    CudaBeamformerKernel kernel = CudaBeamformerKernel::Tiled);

} // namespace beamformer
