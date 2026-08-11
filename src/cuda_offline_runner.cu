#include "beamformer/cuda_offline_runner.hpp"

#include "beamformer/cuda_frame.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/temporal_integration.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace beamformer {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::invalid_argument("offline CUDA buffer has an invalid element count");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                   "cudaMalloc offline frame buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from(const std::vector<T>& host, const char* operation) {
        if (host.size() != count_) {
            throw std::invalid_argument("offline host/device buffer sizes differ");
        }
        check_cuda(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice), operation);
    }

    void copy_to(std::vector<T>& host, const char* operation) const {
        if (host.size() != count_) {
            throw std::invalid_argument("offline device/host buffer sizes differ");
        }
        check_cuda(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost), operation);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

std::size_t weight_count(const Dimensions& dims, const CudaBeamformerKernel kernel) {
    if (kernel == CudaBeamformerKernel::Direct) {
        return dims.n_beams * dims.n_freq * dims.n_ant;
    }
    return tiled_weight_count(dims);
}

Dimensions output_dimensions(const Dimensions& dims,
                             const std::optional<TemporalIntegrationConfig>& integration) {
    Dimensions output = dims;
    if (integration) {
        output.n_time = integrated_time_count(dims.n_time, *integration);
    }
    return output;
}

} // namespace

struct CudaOfflineFrameRunner::Impl {
    Impl(const Dimensions& requested_capacity, const CudaBeamformerKernel selected_kernel,
         const std::optional<TemporalIntegrationConfig>& selected_integration,
         const CudaBeamformerOutput selected_output)
        : capacity(requested_capacity),
          kernel(selected_kernel),
          temporal_integration(selected_integration),
          output(selected_output),
          workspace(capacity, kernel, temporal_integration, output) {}

    Dimensions capacity;
    CudaBeamformerKernel kernel;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutput output;
    CudaBeamformerWorkspace workspace;
};

CudaOfflineFrameRunner::CudaOfflineFrameRunner(
    const Dimensions& capacity, const CudaBeamformerKernel kernel,
    const std::optional<TemporalIntegrationConfig> temporal_integration,
    const CudaBeamformerOutput output)
    : impl_(std::make_unique<Impl>(capacity, kernel, temporal_integration, output)) {}

CudaOfflineFrameRunner::~CudaOfflineFrameRunner() = default;

CudaBeamformerKernel CudaOfflineFrameRunner::kernel() const {
    return impl_->kernel;
}

CudaBeamformerOutput CudaOfflineFrameRunner::output() const {
    return impl_->output;
}

bool CudaOfflineFrameRunner::has_temporal_integration() const {
    return impl_->temporal_integration.has_value();
}

CudaOfflineFrameResult CudaOfflineFrameRunner::run(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const std::uint64_t frame_id, const ShardDescriptor& shard) {
    validate_dimensions(dims);
    validate_shard_descriptor(shard);
    if (dims.n_freq != impl_->capacity.n_freq || dims.n_ant != impl_->capacity.n_ant
        || dims.n_time > impl_->capacity.n_time || dims.n_beams > impl_->capacity.n_beams) {
        throw std::invalid_argument("offline frame dimensions exceed runner capacity");
    }

    const std::size_t packed_count = packed_voltage_bytes(dims);
    if (packed.size() != packed_count) {
        throw std::invalid_argument("offline packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = weight_count(dims, impl_->kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("offline weights size does not match selected layout");
    }

    const Dimensions output_dims = output_dimensions(dims, impl_->temporal_integration);
    CudaOfflineFrameResult result;
    result.output_dims = output_dims;
    result.timings.setup_ms = impl_->workspace.setup_ms();

    DeviceBuffer<std::uint8_t> device_voltage(packed.size());
    DeviceBuffer<ComplexFloat> device_weights(weights.size());

    const auto transfer_start = Clock::now();
    device_voltage.copy_from(packed, "cudaMemcpy offline voltage host to device");
    device_weights.copy_from(weights, "cudaMemcpy offline weights host to device");
    const auto transfer_end = Clock::now();
    result.timings.host_to_device_ms = elapsed_ms(transfer_start, transfer_end);

    const auto voltage_frame = make_packed_voltage_frame_view(
        device_voltage.data(), dims, frame_id, shard);
    const auto weights_frame = impl_->kernel == CudaBeamformerKernel::Direct
                                   ? make_weights_frame_view(
                                         device_weights.data(), dims, frame_id, shard)
                                   : make_tiled_weights_frame_view(
                                         device_weights.data(), dims, frame_id, shard);

    DeviceBuffer<float>* device_float_output = nullptr;
    DeviceBuffer<std::int8_t>* device_int8_output = nullptr;
    DeviceBuffer<Int8QuantizationParameters>* device_parameters = nullptr;
    std::unique_ptr<DeviceBuffer<float>> float_output;
    std::unique_ptr<DeviceBuffer<std::int8_t>> int8_output;
    std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>> parameters;

    CudaFrameView output_frame;
    std::unique_ptr<CudaFrameView> parameter_frame;
    if (impl_->output == CudaBeamformerOutput::Float32) {
        result.float32_output.resize(output_dims.n_time * output_dims.n_freq
                                     * output_dims.n_beams);
        float_output = std::make_unique<DeviceBuffer<float>>(result.float32_output.size());
        device_float_output = float_output.get();
        output_frame = make_intensity_frame_view(
            device_float_output->data(), output_dims, frame_id, shard);
    } else {
        if (!impl_->temporal_integration) {
            throw std::logic_error("offline int8 output requires temporal integration");
        }
        result.quantized_output.codes.resize(quantized_intensity_bytes(output_dims));
        result.quantized_output.parameters.resize(
            quantization_parameter_count(output_dims));
        int8_output = std::make_unique<DeviceBuffer<std::int8_t>>(
            result.quantized_output.codes.size());
        parameters = std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
            result.quantized_output.parameters.size());
        device_int8_output = int8_output.get();
        device_parameters = parameters.get();
        output_frame = make_quantized_intensity_frame_view(
            device_int8_output->data(), output_dims, frame_id, shard);
        parameter_frame = std::make_unique<CudaFrameView>(
            make_quantization_parameters_frame_view(
                device_parameters->data(), output_dims, frame_id, shard));
    }

    const auto device_timings = impl_->workspace.run_device_frame(
        voltage_frame, weights_frame, output_frame, parameter_frame.get());
    result.timings.kernel_ms = device_timings.kernel_ms;
    result.timings.temporal_integration_ms = device_timings.temporal_integration_ms;
    result.timings.quantization_ms = device_timings.quantization_ms;
    result.timings.device_to_device_ms = device_timings.device_to_device_ms;

    const auto result_start = Clock::now();
    if (impl_->output == CudaBeamformerOutput::Float32) {
        device_float_output->copy_to(
            result.float32_output, "cudaMemcpy offline float output device to host");
    } else {
        device_int8_output->copy_to(
            result.quantized_output.codes, "cudaMemcpy offline int8 output device to host");
        device_parameters->copy_to(
            result.quantized_output.parameters,
            "cudaMemcpy offline quantization parameters device to host");
    }
    const auto result_end = Clock::now();
    result.timings.device_to_host_ms = elapsed_ms(result_start, result_end);
    return result;
}

} // namespace beamformer
