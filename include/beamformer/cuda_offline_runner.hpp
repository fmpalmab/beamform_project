#pragma once

#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/formats.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace beamformer {

// Host-side result returned by one offline frame execution. When the runner is
// configured for Float32, float32_output contains either the per-spectrum or
// integrated [time][local_frequency][beam] result. When it is configured for
// QuantizedInt8, quantized_output contains the signed codes and reconstruction
// parameters instead.
struct CudaOfflineFrameResult {
    Dimensions output_dims;
    CudaBeamformerTimings timings;
    Intensities float32_output;
    QuantizedIntegratedOutput quantized_output;
};

// Convenience layer for one offline frame. It owns only temporary device
// copies of caller-owned host input/output data; it does not create a ring
// buffer or retain frame state. The underlying run_device_frame call remains
// the external device-buffer contract used by the PoC.
class CudaOfflineFrameRunner {
  public:
    explicit CudaOfflineFrameRunner(
        const Dimensions& capacity,
        CudaBeamformerKernel kernel = CudaBeamformerKernel::Direct,
        std::optional<TemporalIntegrationConfig> temporal_integration = std::nullopt,
        CudaBeamformerOutput output = CudaBeamformerOutput::Float32);
    ~CudaOfflineFrameRunner();

    CudaOfflineFrameRunner(const CudaOfflineFrameRunner&) = delete;
    CudaOfflineFrameRunner& operator=(const CudaOfflineFrameRunner&) = delete;

    CudaOfflineFrameResult run(
        const PackedVoltage& packed, const Weights& weights,
        const Dimensions& dims, std::uint64_t frame_id,
        const ShardDescriptor& shard);

    CudaBeamformerKernel kernel() const;
    CudaBeamformerOutput output() const;
    bool has_temporal_integration() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
