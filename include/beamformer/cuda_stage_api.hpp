#pragma once

#include "beamformer/complex.hpp"
#include "beamformer/config.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/temporal_integration.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace beamformer {

enum class CudaBeamformerKernel {
    Direct,
    Tiled,
};

// Launches one packed-voltage beamforming stage on caller-owned device buffers
// and stream. The function neither allocates memory nor synchronizes the stream.
void launch_packed_beamformer(CudaBeamformerKernel kernel, cudaStream_t stream,
                              const std::uint8_t* packed_voltage,
                              const ComplexFloat* weights, float* intensity,
                              const Dimensions& dims);

// Launches a fused beamforming-plus-temporal-integration stage. It decodes
// packed int4x2 voltage bytes at point of use and writes [T_integrated][F][B]
// directly, without materializing a [T][F][B] intermediate. The function
// neither allocates memory nor synchronizes the stream.
void launch_packed_integrated_beamformer(
    CudaBeamformerKernel kernel, cudaStream_t stream,
    const std::uint8_t* packed_voltage, const ComplexFloat* weights,
    float* integrated_intensity, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration);

// Launches the independent int8 output stage on caller-owned device buffers
// and stream. The input uses [T_integrated][F][B] float32 ordering; the codes
// preserve that ordering and parameters use the documented chunk layout.
void launch_quantize_integrated_intensity(
    cudaStream_t stream, const float* integrated_intensity,
    std::int8_t* quantized_intensity, Int8QuantizationParameters* parameters,
    const Dimensions& integrated_dims);

} // namespace beamformer
