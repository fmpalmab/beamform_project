#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double absolute_tolerance = 1.0e-3;
constexpr double relative_tolerance = 1.0e-5;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

__global__ void decode_bytes_kernel(const std::uint8_t* input,
                                    std::int8_t* real,
                                    std::int8_t* imag,
                                    const std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                              + threadIdx.x;
    if (index >= count) {
        return;
    }
    // Uses the same __host__ __device__ decoder as the production packed
    // beamforming kernel.
    const auto sample = beamformer::unpack_complex_int4(input[index]);
    real[index] = sample.real;
    imag[index] = sample.imag;
}

void verify_all_256_cuda_decodes() {
    std::vector<std::uint8_t> input(256);
    std::vector<std::int8_t> real(input.size());
    std::vector<std::int8_t> imag(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>(index);
    }

    std::uint8_t* device_input = nullptr;
    std::int8_t* device_real = nullptr;
    std::int8_t* device_imag = nullptr;
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input.size()),
                   "cudaMalloc decode input");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_real), real.size()),
                   "cudaMalloc decode real");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_imag), imag.size()),
                   "cudaMalloc decode imag");
        check_cuda(cudaMemcpy(device_input, input.data(), input.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy decode input");
        decode_bytes_kernel<<<1, 256>>>(device_input, device_real, device_imag,
                                        input.size());
        check_cuda(cudaGetLastError(), "decode_bytes_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "decode_bytes_kernel synchronize");
        check_cuda(cudaMemcpy(real.data(), device_real, real.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy decode real");
        check_cuda(cudaMemcpy(imag.data(), device_imag, imag.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy decode imag");
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_real != nullptr) {
            cudaFree(device_real);
        }
        if (device_imag != nullptr) {
            cudaFree(device_imag);
        }
        throw;
    }
    check_cuda(cudaFree(device_input), "cudaFree decode input");
    check_cuda(cudaFree(device_real), "cudaFree decode real");
    check_cuda(cudaFree(device_imag), "cudaFree decode imag");

    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto expected = beamformer::unpack_complex_int4(input[index]);
        require(real[index] == expected.real && imag[index] == expected.imag,
                "CUDA packed-byte decode differs from signed-nibble contract");
    }
}

void compare_cpu_gpu(const std::string& label,
                     const beamformer::Intensities& cpu,
                     const beamformer::Intensities& gpu) {
    require(cpu.size() == gpu.size(), label + ": output sizes differ");
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        require(std::isfinite(cpu[index]) && std::isfinite(gpu[index]),
                label + ": non-finite output");
        const double error = std::abs(static_cast<double>(gpu[index]) - cpu[index]);
        const double allowed = absolute_tolerance
                               + relative_tolerance * std::abs(static_cast<double>(cpu[index]));
        require(error <= allowed, label + ": packed CPU/CUDA tolerance exceeded");
    }
}

void verify_case(const std::string& label,
                 const beamformer::PackedVoltage& packed,
                 const beamformer::Weights& weights,
                 const beamformer::Dimensions& dims) {
    beamformer::TiledWeights tiled_weights(
        beamformer::tiled_weight_count(dims), beamformer::ComplexFloat{0.0F, 0.0F});
    for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t antenna = 0; antenna < dims.n_ant; ++antenna) {
                tiled_weights[beamformer::tiled_weight_index(
                    frequency, beam / beamformer::tiled_weight_beam_tile,
                    antenna, beam % beamformer::tiled_weight_beam_tile, dims)] =
                    weights[beamformer::weight_index(beam, frequency, antenna, dims)];
            }
        }
    }
    const auto cpu = beamformer::cpu_beamform_packed_intensity(packed, weights, dims);
    for (const auto kernel : std::array{
             beamformer::CudaBeamformerKernel::Direct,
             beamformer::CudaBeamformerKernel::Tiled}) {
        const auto& kernel_weights =
            kernel == beamformer::CudaBeamformerKernel::Direct
                ? static_cast<const beamformer::Weights&>(weights)
                : static_cast<const beamformer::Weights&>(tiled_weights);
        const auto gpu = beamformer::cuda_beamform_packed_intensity(
            packed, kernel_weights, dims, nullptr, kernel);
        compare_cpu_gpu(label + (kernel == beamformer::CudaBeamformerKernel::Direct
                                     ? " [Direct]" : " [Tiled]"),
                        cpu, gpu);
    }
}

beamformer::Weights make_weights(const beamformer::Dimensions& dims,
                                 const std::vector<beamformer::Vec3>& directions) {
    return beamformer::generate_weights(
        dims, beamformer::default_positions(dims.n_ant),
        beamformer::channelized_frequencies(dims.n_freq), directions);
}

} // namespace

int main() {
    try {
        using namespace beamformer;

        verify_all_256_cuda_decodes();

        const Dimensions one_hot_dims{1, default_frequency_channels, 32, 5};
        const auto one_hot_weights = make_weights(
            one_hot_dims, default_beam_grid(one_hot_dims.n_beams));
        const auto one_hot = make_one_hot(
            one_hot_dims, 0, 335, 31, {-8, 7});
        verify_case("one-hot boundary index", one_hot, one_hot_weights, one_hot_dims);

        const Dimensions constant_dims{2, default_frequency_channels, 32, 1};
        const auto constant_weights = make_weights(
            constant_dims, std::vector<Vec3>{direction_from_lm(0.0F, 0.0F)});
        const auto constant = make_constant(constant_dims, {7, -8});
        verify_case("constant", constant, constant_weights, constant_dims);
        const auto constant_cpu =
            cpu_beamform_packed_intensity(constant, constant_weights, constant_dims);
        const float expected_constant = static_cast<float>(
            constant_dims.n_ant * constant_dims.n_ant * (7 * 7 + 8 * 8));
        for (const float value : constant_cpu) {
            require(value == expected_constant, "constant analytical intensity mismatch");
        }

        const Dimensions point_dims{3, default_frequency_channels, 32, 32};
        const auto point_directions = rectangular_beam_grid(point_dims.n_ant);
        const auto point_weights = make_weights(point_dims, point_directions);
        const auto point_source = make_point_source(
            point_dims, default_positions(point_dims.n_ant),
            channelized_frequencies(point_dims.n_freq), point_directions[12], 4.0F);
        verify_case("point source", point_source, point_weights, point_dims);

        const Dimensions noise_dims{10, default_frequency_channels, 64, 7};
        const auto noise_weights = make_weights(
            noise_dims, fft_beam_grid(noise_dims.n_ant, noise_dims.n_beams));
        const auto noise = make_noise(noise_dims, 9876);
        verify_case("seeded noise", noise, noise_weights, noise_dims);

        // Boundary values exercise the signed extremes in direct accumulation.
        for (const auto value : std::array<ComplexInt4, 4>{
                 ComplexInt4{-8, -8}, ComplexInt4{-8, 7},
                 ComplexInt4{7, -8}, ComplexInt4{7, 7}}) {
            const auto boundary = make_constant(constant_dims, value);
            verify_case("boundary nibble", boundary, constant_weights, constant_dims);
        }

        const Dimensions shard_dims{2, default_frequency_channels, 32, 4};
        const auto shard_weights = make_weights(
            shard_dims, default_beam_grid(shard_dims.n_beams));
        const auto shards = make_two_shard_one_hot(
            shard_dims, 1, {0, 335}, {0, 31}, {-8, 7}, {11, 12});
        validate_packed_shards(shards, shard_dims);
        verify_case("shard zero", shards[0].payload, shard_weights, shard_dims);
        verify_case("shard one", shards[1].payload, shard_weights, shard_dims);

        // The production workspace reserves exactly one byte per voltage sample.
        for (const auto kernel : std::array{
                 CudaBeamformerKernel::Direct, CudaBeamformerKernel::Tiled}) {
            CudaBeamformerWorkspace workspace(noise_dims, kernel);
            require(workspace.kernel() == kernel,
                    "CUDA workspace did not retain the selected kernel");
            require(workspace.packed_voltage_capacity_bytes()
                        == packed_voltage_bytes(noise_dims),
                    "CUDA workspace input allocation is not packed-byte sized");
        }
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_packed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
