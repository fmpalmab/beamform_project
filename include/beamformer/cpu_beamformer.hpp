#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

namespace beamformer {

ComplexVoltage unpack_voltage(const PackedVoltage& packed,
                              const Dimensions& dims);

// Compact numerical reference for the production input contract. Samples are
// decoded from packed signed-int4 bytes inside the direct accumulation loop;
// no full ComplexVoltage input tensor is materialized.
Intensities cpu_beamform_packed_intensity(const PackedVoltage& packed,
                                          const Weights& weights,
                                          const Dimensions& dims);

// Writes the required output prefix and permits a larger reusable output
// buffer, matching cpu_beamform_intensity_into.
void cpu_beamform_packed_intensity_into(const PackedVoltage& packed,
                                        const Weights& weights,
                                        const Dimensions& dims,
                                        Intensities& intensity);

Intensities cpu_beamform_intensity(const ComplexVoltage& voltage,
                                   const Weights& weights,
                                   const Dimensions& dims);

// Benchmark-oriented variant that writes into a preallocated output. Voltage
// may contain additional time samples after the prefix described by dims.
void cpu_beamform_intensity_into(const ComplexVoltage& voltage,
                                 const Weights& weights,
                                 const Dimensions& dims,
                                 Intensities& intensity);

} // namespace beamformer
