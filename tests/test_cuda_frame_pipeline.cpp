#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_frame.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                   "cudaMalloc test frame buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from(const std::vector<T>& host) {
        if (host.size() != count_) {
            throw std::invalid_argument("test frame host/device sizes differ");
        }
        check_cuda(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice),
                   "cudaMemcpy test frame host to device");
    }

    void copy_to(std::vector<T>& host) const {
        if (host.size() != count_) {
            throw std::invalid_argument("test frame device/host sizes differ");
        }
        check_cuda(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost),
                   "cudaMemcpy test frame device to host");
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        using namespace beamformer;

        const Dimensions dims{11, default_frequency_channels, 32, 7};
        const auto directions = default_beam_grid(dims.n_beams);
        const auto positions = default_positions(dims.n_ant);
        const auto frequencies = channelized_frequencies(dims.n_freq);
        const auto packed = make_noise(dims, 1234);
        const auto canonical_weights =
            generate_weights(dims, positions, frequencies, directions);
        const auto tiled_weights =
            generate_tiled_weights(dims, positions, frequencies, directions);
        const auto raw_reference =
            cpu_beamform_packed_intensity(packed, canonical_weights, dims);
        const auto expected =
            cpu_integrate_intensity(raw_reference, dims, integration_after_upchan);

        Dimensions output_dims = dims;
        output_dims.n_time = integrated_time_count(
            dims.n_time, integration_after_upchan);
        const auto shard = default_shard_descriptors()[0];
        DeviceBuffer<std::uint8_t> device_voltage(packed.size());
        DeviceBuffer<ComplexFloat> device_weights(tiled_weights.size());
        DeviceBuffer<float> device_output(expected.size());
        device_voltage.copy_from(packed);
        device_weights.copy_from(tiled_weights);

        const auto voltage_frame = make_packed_voltage_frame_view(
            device_voltage.data(), dims, 42, shard);
        const auto weights_frame = make_tiled_weights_frame_view(
            device_weights.data(), dims, 42, shard);
        auto output_frame = make_intensity_frame_view(
            device_output.data(), output_dims, 42, shard);

        CudaBeamformerWorkspace workspace(
            dims, CudaBeamformerKernel::Tiled, integration_after_upchan);
        const auto timings = workspace.run_device_frame(
            voltage_frame, weights_frame, output_frame);

        std::vector<float> actual(expected.size());
        device_output.copy_to(actual);
        require(actual.size() == expected.size(), "offline frame output size differs");
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double error = std::abs(static_cast<double>(actual[index])
                                          - static_cast<double>(expected[index]));
            const double allowed = 1.0e-3
                                   + 1.0e-5 * std::abs(static_cast<double>(expected[index]));
            require(error <= allowed, "offline frame output exceeds CPU tolerance");
        }

        require(timings.host_to_device_ms == 0.0,
                "offline frame must not report host-to-device work");
        require(timings.device_to_host_ms == 0.0,
                "offline frame must not report device-to-host work");
        require(timings.kernel_ms >= 0.0 && timings.temporal_integration_ms == 0.0
                    && timings.device_to_device_ms >= 0.0,
                "offline frame must report integration as part of the beamformer kernel");

        cudaStream_t stage_stream = nullptr;
        check_cuda(cudaStreamCreate(&stage_stream), "cudaStreamCreate stage API test");
        launch_packed_integrated_beamformer(
            CudaBeamformerKernel::Tiled, stage_stream, device_voltage.data(),
            device_weights.data(), device_output.data(), dims,
            integration_after_upchan);
        check_cuda(cudaStreamSynchronize(stage_stream),
                   "cudaStreamSynchronize stage API test");
        std::vector<float> stage_actual(expected.size());
        device_output.copy_to(stage_actual);
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const double error = std::abs(static_cast<double>(stage_actual[index])
                                          - static_cast<double>(expected[index]));
            const double allowed = 1.0e-3
                                   + 1.0e-5 * std::abs(static_cast<double>(expected[index]));
            require(error <= allowed,
                    "stage API fused output exceeds CPU tolerance");
        }
        check_cuda(cudaStreamDestroy(stage_stream), "cudaStreamDestroy stage API test");
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_frame_pipeline: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
