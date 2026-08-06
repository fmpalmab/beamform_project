#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace beamformer {

enum class CudaBeamformerKernel {
    Direct,
    Tiled,
};

struct CudaBeamformerTimings {
    double setup_ms = 0.0;
    double host_to_device_ms = 0.0;
    double kernel_ms = 0.0;
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
        CudaBeamformerKernel kernel = CudaBeamformerKernel::Direct);
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

} // namespace beamformer
