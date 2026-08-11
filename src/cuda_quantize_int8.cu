#include "beamformer/cuda_stage_api.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beamformer {
namespace {

constexpr unsigned int quantization_threads = 32;
constexpr unsigned int samples_per_thread = 8;
static_assert(quantization_threads * samples_per_thread
              == quantization_chunk_time * quantization_chunk_frequency
                     * quantization_chunk_beam);

__device__ float sane_fmin(const float lhs, const float rhs) {
    return (isnan(lhs) || isnan(rhs)) ? nanf("") : fminf(lhs, rhs);
}

__device__ float sane_fmax(const float lhs, const float rhs) {
    return (isnan(lhs) || isnan(rhs)) ? nanf("") : fmaxf(lhs, rhs);
}


// The kernels operate on a chunk of the integrated intensity tensor, which is a 1 x 16 x 16 local chunk of the [time][frequency][beam] tensor. Each thread processes 8 samples, and there are 32 threads per block, so each block processes a total of 256 samples (1 x 16 x 16). The kernel computes the minimum and maximum values in the chunk, and then uses those to compute the quantization parameters (scale and offset) for that chunk. Finally, it quantizes the samples in the chunk using those parameters.
__global__ void quantize_integrated_intensity_kernel(
    const float* __restrict__ integrated_intensity,
    std::int8_t* __restrict__ quantized_intensity,
    Int8QuantizationParameters* __restrict__ parameters,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_beams, const std::size_t frequency_tiles,
    const std::size_t beam_tiles) {
    const std::size_t chunk = blockIdx.x;
    const std::size_t lane = threadIdx.x;
    const std::size_t beam_tile = chunk % beam_tiles;
    const std::size_t time_frequency_tile = chunk / beam_tiles;
    const std::size_t frequency_tile = time_frequency_tile % frequency_tiles;
    const std::size_t time_tile = time_frequency_tile / frequency_tiles;

    float values[samples_per_thread];
    std::size_t output_indices[samples_per_thread];
    bool valid[samples_per_thread];
    constexpr float float_max = 3.402823466e+38F;
    float minimum = float_max;
    float maximum = -float_max;
    bool non_finite = false;

    for (unsigned int item = 0; item < samples_per_thread; ++item) {
        const std::size_t sample = lane + item * quantization_threads;
        const std::size_t local_time = sample
                                       / (quantization_chunk_frequency
                                          * quantization_chunk_beam);
        const std::size_t local_frequency =
            (sample / quantization_chunk_beam) % quantization_chunk_frequency;
        const std::size_t local_beam = sample % quantization_chunk_beam;
        const std::size_t time = time_tile * quantization_chunk_time + local_time;
        const std::size_t frequency = frequency_tile * quantization_chunk_frequency
                                      + local_frequency;
        const std::size_t beam = beam_tile * quantization_chunk_beam + local_beam;
        valid[item] = time < n_time && frequency < n_freq && beam < n_beams;
        if (!valid[item]) {
            continue;
        }
        output_indices[item] = (time * n_freq + frequency) * n_beams + beam;
        values[item] = integrated_intensity[output_indices[item]];
        if (!isfinite(values[item])) {
            non_finite = true;
        } else {
            minimum = sane_fmin(minimum, values[item]);
            maximum = sane_fmax(maximum, values[item]);
        }
    }

    constexpr unsigned int mask = 0xffffffffU;
    non_finite = __any_sync(mask, non_finite);
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x01));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x01));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x02));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x02));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x04));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x04));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x08));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x08));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x10));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x10));

    constexpr float output_min = static_cast<float>(quantized_min_code) - 0.5F;
    constexpr float output_max = static_cast<float>(quantized_max_code) + 0.5F;
    constexpr float output_range = output_max - output_min;
    Int8QuantizationParameters chunk_parameters{};
    if (!non_finite && isfinite(minimum) && isfinite(maximum)) {
        chunk_parameters.scale = (maximum - minimum) / output_range;
        chunk_parameters.offset = minimum - output_min * chunk_parameters.scale;
        if (!isfinite(chunk_parameters.offset) || !isfinite(chunk_parameters.scale)) {
            chunk_parameters = {};
        }
    }
    if (lane == 0) {
        parameters[chunk] = chunk_parameters;
    }

    for (unsigned int item = 0; item < samples_per_thread; ++item) {
        if (!valid[item]) {
            continue;
        }
        int code = 0;
        if (!isfinite(values[item])) {
            code = quantized_invalid_code;
        } else if (chunk_parameters.scale != 0.0F) {
            code = __float2int_rn(
                (values[item] - chunk_parameters.offset) / chunk_parameters.scale);
            code = max(static_cast<int>(quantized_min_code),
                       min(static_cast<int>(quantized_max_code), code));
        }
        quantized_intensity[output_indices[item]] = static_cast<std::int8_t>(code);
    }
}

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(result));
    }
}

} // namespace

void launch_quantize_integrated_intensity(
    const cudaStream_t stream, const float* integrated_intensity,
    std::int8_t* quantized_intensity, Int8QuantizationParameters* parameters,
    const Dimensions& integrated_dims) {
    const auto layout = quantization_layout(integrated_dims);
    const std::size_t chunk_count = quantization_parameter_count(integrated_dims);
    if (chunk_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA quantization grid exceeds the supported size");
    }
    quantize_integrated_intensity_kernel<<<static_cast<unsigned int>(chunk_count),
                                            quantization_threads, 0, stream>>>(
        integrated_intensity, quantized_intensity, parameters,
        integrated_dims.n_time, integrated_dims.n_freq, integrated_dims.n_beams,
        layout.frequency_tiles, layout.beam_tiles);
    check_cuda(cudaGetLastError(), "quantize_integrated_intensity_kernel launch");
}

} // namespace beamformer
