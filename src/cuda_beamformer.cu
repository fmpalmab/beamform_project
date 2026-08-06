#include "beamformer/cuda_beamformer.hpp"

#include "beamformer/formats.hpp"
#include "beamformer/int4.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace beamformer {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(result));
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class CudaStream {
  public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const {
        return stream_;
    }

  private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

  private:
    cudaEvent_t event_ = nullptr;
};

template <typename Value>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Value)),
                   "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Value* get() {
        return data_;
    }

    const Value* get() const {
        return data_;
    }

  private:
    Value* data_ = nullptr;
};

__global__ void direct_packed_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights, float* intensity,
    const std::size_t output_count, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t beam = output_index % n_beams;
    const std::size_t time_frequency = output_index / n_beams;
    const std::size_t frequency = time_frequency % n_freq;
    const std::size_t voltage_offset = time_frequency * n_ant;
    const std::size_t weight_offset = (beam * n_freq + frequency) * n_ant;

    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t element = 0; element < n_ant; ++element) {
        // Decode at the point of use using the shared CHARTS int4x2
        // convention: real in the low nibble, imaginary in the high nibble.
        const ComplexInt4 packed_sample =
            unpack_complex_int4(packed_voltage[voltage_offset + element]);
        const float sample_real = static_cast<float>(packed_sample.real);
        const float sample_imag = static_cast<float>(packed_sample.imag);
        const ComplexFloat weight = weights[weight_offset + element];
        sum_real += weight.real * sample_real - weight.imag * sample_imag;
        sum_imag += weight.real * sample_imag + weight.imag * sample_real;
    }
    intensity[output_index] = sum_real * sum_real + sum_imag * sum_imag;
}

float event_elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

void launch_direct_kernel(const std::uint8_t* packed_voltage,
                          const ComplexFloat* weights, float* intensity,
                          const Dimensions& dims, cudaStream_t stream) {
    constexpr std::size_t threads_per_block = 256;
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;
    const std::size_t block_count =
        (output_count + threads_per_block - 1) / threads_per_block;
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA grid exceeds the supported one-dimensional size");
    }
    direct_packed_voltage_beamformer_kernel<<<static_cast<unsigned int>(block_count),
                                              static_cast<unsigned int>(threads_per_block), 0,
                                              stream>>>(packed_voltage, weights, intensity,
                                                        output_count, dims.n_freq,
                                                        dims.n_ant, dims.n_beams);
    check_cuda(cudaGetLastError(), "direct_packed_voltage_beamformer_kernel launch");
}

} // namespace

struct CudaBeamformerWorkspace::Impl {
    explicit Impl(const Dimensions& requested_capacity)
        : capacity(requested_capacity),
          device_packed_voltage(voltage_sample_count(capacity)),
          device_weights(capacity.n_beams * capacity.n_freq * capacity.n_ant),
          device_intensity(capacity.n_time * capacity.n_freq * capacity.n_beams) {}

    void validate_request(const Dimensions& dims) const {
        validate_dimensions(dims);
        if (dims.n_freq != capacity.n_freq || dims.n_ant != capacity.n_ant
            || dims.n_time > capacity.n_time || dims.n_beams > capacity.n_beams) {
            throw std::invalid_argument("dimensions exceed CUDA workspace capacity");
        }
    }

    void require_loaded(const Dimensions& dims) const {
        if (loaded_packed_voltage_samples < voltage_sample_count(dims)) {
            throw std::logic_error("packed voltage has not been uploaded for these dimensions");
        }
        const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
        if (loaded_weights != required_weights) {
            throw std::logic_error("weights have not been uploaded for these dimensions");
        }
    }

    Dimensions capacity;
    DeviceBuffer<std::uint8_t> device_packed_voltage;
    DeviceBuffer<ComplexFloat> device_weights;
    DeviceBuffer<float> device_intensity;
    CudaStream stream;
    CudaEvent start;
    CudaEvent transfer_end;
    CudaEvent kernel_end;
    CudaEvent result_end;
    double measured_setup_ms = 0.0;
    std::size_t loaded_packed_voltage_samples = 0;
    std::size_t loaded_weights = 0;
};

CudaDeviceInfo cuda_device_info() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");

    CudaDeviceInfo info;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.global_memory_bytes = properties.totalGlobalMem;
    check_cuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
    return info;
}

CudaBeamformerWorkspace::CudaBeamformerWorkspace(const Dimensions& capacity) {
    validate_dimensions(capacity);
    const auto start = Clock::now();
    impl_ = std::make_unique<Impl>(capacity);
    const auto end = Clock::now();
    impl_->measured_setup_ms = elapsed_ms(start, end);
}

CudaBeamformerWorkspace::~CudaBeamformerWorkspace() = default;

double CudaBeamformerWorkspace::setup_ms() const {
    return impl_->measured_setup_ms;
}

std::size_t CudaBeamformerWorkspace::packed_voltage_capacity_bytes() const {
    return voltage_sample_count(impl_->capacity);
}

double CudaBeamformerWorkspace::upload_packed_voltage(const PackedVoltage& packed,
                                                      const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = voltage_sample_count(dims);
    if (packed.size() < count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord packed voltage start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord packed voltage end");
    check_cuda(cudaEventSynchronize(impl_->transfer_end.get()),
               "cudaEventSynchronize packed voltage");
    impl_->loaded_packed_voltage_samples = count;
    return event_elapsed_ms(impl_->start, impl_->transfer_end);
}

double CudaBeamformerWorkspace::upload_weights(const Weights& weights,
                                               const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = dims.n_beams * dims.n_freq * dims.n_ant;
    if (weights.size() != count) {
        throw std::invalid_argument("weight count does not match dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord weights start");
    check_cuda(cudaMemcpyAsync(impl_->device_weights.get(), weights.data(),
                               count * sizeof(ComplexFloat), cudaMemcpyHostToDevice,
                               impl_->stream.get()),
               "cudaMemcpyAsync weights host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord weights end");
    check_cuda(cudaEventSynchronize(impl_->transfer_end.get()),
               "cudaEventSynchronize weights");
    impl_->loaded_weights = count;
    return event_elapsed_ms(impl_->start, impl_->transfer_end);
}

double CudaBeamformerWorkspace::run_kernel(const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_loaded(dims);
    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord kernel start");
    launch_direct_kernel(impl_->device_packed_voltage.get(), impl_->device_weights.get(),
                         impl_->device_intensity.get(), dims, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord kernel end");
    check_cuda(cudaEventSynchronize(impl_->kernel_end.get()),
               "cudaEventSynchronize kernel");
    return event_elapsed_ms(impl_->start, impl_->kernel_end);
}

double CudaBeamformerWorkspace::download_intensity(Intensities& intensity,
                                                   const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t count = dims.n_time * dims.n_freq * dims.n_beams;
    if (intensity.size() < count) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord intensity start");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity.get(),
                               count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord intensity end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize intensity");
    return event_elapsed_ms(impl_->start, impl_->result_end);
}

CudaBeamformerTimings CudaBeamformerWorkspace::run_pipeline(
    const PackedVoltage& packed, Intensities& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t output_count = dims.n_time * dims.n_freq * dims.n_beams;
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.size() < output_count) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }
    const std::size_t required_weights = dims.n_beams * dims.n_freq * dims.n_ant;
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline transfer end");
    launch_direct_kernel(impl_->device_packed_voltage.get(), impl_->device_weights.get(),
                         impl_->device_intensity.get(), dims, impl_->stream.get());
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline kernel end");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_intensity.get(),
                               output_count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync pipeline intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimings timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end,
                                                 impl_->result_end);
    return timings;
}

Intensities cuda_beamform_packed_intensity(const PackedVoltage& packed,
                                           const Weights& weights,
                                           const Dimensions& dims,
                                           CudaBeamformerTimings* timings) {
    validate_dimensions(dims);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    if (weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("weight count does not match dimensions");
    }

    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    CudaBeamformerWorkspace workspace(dims);
    CudaBeamformerTimings measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_packed_voltage(packed, dims)
                                 + workspace.upload_weights(weights, dims);
    measured.kernel_ms = workspace.run_kernel(dims);
    measured.device_to_host_ms = workspace.download_intensity(intensity, dims);
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

} // namespace beamformer
