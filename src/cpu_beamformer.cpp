#include "beamformer/cpu_beamformer.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"

#include <stdexcept>

namespace beamformer {

ComplexVoltage unpack_voltage(const PackedVoltage& packed,
                              const Dimensions& dims) {
    validate_dimensions(dims);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }

    ComplexVoltage voltage(packed.size());
    for (std::size_t index = 0; index < packed.size(); ++index) {
        const auto sample = unpack_complex_int4(packed[index]);
        voltage[index] = {
            static_cast<float>(sample.real),
            static_cast<float>(sample.imag),
        };
    }
    return voltage;
}

Intensities cpu_beamform_packed_intensity(const PackedVoltage& packed,
                                          const Weights& weights,
                                          const Dimensions& dims) {
    validate_dimensions(dims);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    cpu_beamform_packed_intensity_into(packed, weights, dims, intensity);
    return intensity;
}

void cpu_beamform_packed_intensity_into(const PackedVoltage& packed,
                                        const Weights& weights,
                                        const Dimensions& dims,
                                        Intensities& intensity) {
    validate_dimensions(dims);
    if (packed.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    const std::size_t required_output = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                float sum_real = 0.0F;
                float sum_imag = 0.0F;
                for (std::size_t element = 0; element < dims.n_ant; ++element) {
                    // Decode the RFSoC byte at the point of use. The two
                    // signed-int4 values are widened exactly to float32;
                    // there is no intermediate ComplexVoltage tensor.
                    const auto sample = unpack_complex_int4(
                        packed[voltage_index(time, frequency, element, dims)]);
                    const float sample_real = static_cast<float>(sample.real);
                    const float sample_imag = static_cast<float>(sample.imag);
                    const auto& weight =
                        weights[weight_index(beam, frequency, element, dims)];
                    sum_real += weight.real * sample_real - weight.imag * sample_imag;
                    sum_imag += weight.real * sample_imag + weight.imag * sample_real;
                }
                intensity[intensity_index(time, frequency, beam, dims)] =
                    sum_real * sum_real + sum_imag * sum_imag;
            }
        }
    }
}

Intensities cpu_beamform_intensity(const ComplexVoltage& voltage,
                                   const Weights& weights,
                                   const Dimensions& dims) {
    validate_dimensions(dims);
    if (voltage.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("voltage count does not match dimensions");
    }
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    cpu_beamform_intensity_into(voltage, weights, dims, intensity);
    return intensity;
}

void cpu_beamform_intensity_into(const ComplexVoltage& voltage,
                                 const Weights& weights,
                                 const Dimensions& dims,
                                 Intensities& intensity) {
    validate_dimensions(dims);
    if (voltage.size() < voltage_sample_count(dims)) {
        throw std::invalid_argument("voltage count is smaller than dimensions");
    }
    if (weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    const std::size_t required_output = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }

    for (std::size_t time = 0; time < dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                float sum_real = 0.0F;
                float sum_imag = 0.0F;
                for (std::size_t element = 0; element < dims.n_ant; ++element) {
                    const auto& sample =
                        voltage[voltage_index(time, frequency, element, dims)];
                    const auto& weight =
                        weights[weight_index(beam, frequency, element, dims)];
                    sum_real += weight.real * sample.real - weight.imag * sample.imag;
                    sum_imag += weight.real * sample.imag + weight.imag * sample.real;
                }
                intensity[intensity_index(time, frequency, beam, dims)] =
                    sum_real * sum_real + sum_imag * sum_imag;
            }
        }
    }
}

} // namespace beamformer
