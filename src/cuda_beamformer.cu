#include "beamformer/cuda_beamformer.hpp"

#include "beamformer/formats.hpp"
#include "beamformer/indexing.hpp"
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

constexpr unsigned int tiled_beam_tile = 32;
constexpr unsigned int tiled_time_tile = 8;
constexpr unsigned int tiled_threads = tiled_beam_tile * tiled_time_tile;
constexpr unsigned int tiled_max_antennas = 64;

__global__ void tiled_packed_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights, float* intensity,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams) {
    extern __shared__ unsigned char shared_storage[];
    auto* shared_weights = reinterpret_cast<ComplexFloat*>(shared_storage);
    auto* shared_voltage = shared_storage
                           + tiled_max_antennas * tiled_beam_tile
                                 * sizeof(ComplexFloat);

    const std::size_t local_beam = threadIdx.x;
    const std::size_t local_time = threadIdx.y;
    const std::size_t frequency = blockIdx.y;
    const std::size_t beam = static_cast<std::size_t>(blockIdx.x)
                             * tiled_beam_tile + local_beam;
    const std::size_t time = static_cast<std::size_t>(blockIdx.z)
                             * tiled_time_tile + local_time;
    const std::size_t linear_thread = local_time * tiled_beam_tile + local_beam;

    // Global weights are laid out [frequency][beam_tile][antenna][local_beam].
    // Mapping each warp to one antenna makes its local-beam reads contiguous,
    // while the shared representation remains [antenna][local_beam].
    for (std::size_t index = linear_thread;
         index < n_ant * tiled_beam_tile; index += tiled_threads) {
        const std::size_t antenna = index / tiled_beam_tile;
        const std::size_t tile_beam = index % tiled_beam_tile;
        const std::size_t global_beam = static_cast<std::size_t>(blockIdx.x)
                                        * tiled_beam_tile + tile_beam;
        const auto shared_index = antenna * tiled_beam_tile + tile_beam;
        if (global_beam < n_beams) {
            const std::size_t tiled_weight_index =
                (((frequency * ((n_beams + tiled_beam_tile - 1) / tiled_beam_tile)
                   + static_cast<std::size_t>(blockIdx.x))
                  * n_ant + antenna) * tiled_beam_tile + tile_beam);
            shared_weights[shared_index] = weights[tiled_weight_index];
        } else {
            shared_weights[shared_index] = ComplexFloat{0.0F, 0.0F};
        }
    }

    // Global voltage is laid out [time][frequency][antenna]. Keep the packed
    // byte in shared memory; decode it only when the output thread consumes it.
    for (std::size_t index = linear_thread;
         index < tiled_time_tile * n_ant; index += tiled_threads) {
        const std::size_t tile_time = index / n_ant;
        const std::size_t antenna = index % n_ant;
        const std::size_t global_time = static_cast<std::size_t>(blockIdx.z)
                                        * tiled_time_tile + tile_time;
        shared_voltage[tile_time * n_ant + antenna] =
            global_time < n_time
                ? packed_voltage[(global_time * n_freq + frequency) * n_ant + antenna]
                : 0U;
    }
    __syncthreads();

    if (time >= n_time || beam >= n_beams) {
        return;
    }

    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t antenna = 0; antenna < n_ant; ++antenna) {
        const ComplexInt4 packed_sample = unpack_complex_int4(
            shared_voltage[local_time * n_ant + antenna]);
        const float sample_real = static_cast<float>(packed_sample.real);
        const float sample_imag = static_cast<float>(packed_sample.imag);
        const ComplexFloat weight =
            shared_weights[antenna * tiled_beam_tile + local_beam];
        sum_real += weight.real * sample_real - weight.imag * sample_imag;
        sum_imag += weight.real * sample_imag + weight.imag * sample_real;
    }
    intensity[(time * n_freq + frequency) * n_beams + beam] =
        sum_real * sum_real + sum_imag * sum_imag;
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

void launch_tiled_kernel(const std::uint8_t* packed_voltage,
                         const ComplexFloat* weights, float* intensity,
                         const Dimensions& dims, cudaStream_t stream) {
    const auto grid_x = (dims.n_beams + tiled_beam_tile - 1) / tiled_beam_tile;
    const auto grid_z = (dims.n_time + tiled_time_tile - 1) / tiled_time_tile;
    if (grid_x > std::numeric_limits<unsigned int>::max()
        || dims.n_freq > std::numeric_limits<unsigned int>::max()
        || grid_z > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA tiled grid exceeds the supported size");
    }
    constexpr std::size_t shared_bytes =
        tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat)
        + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t);
    const dim3 grid(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(dims.n_freq),
                    static_cast<unsigned int>(grid_z));
    const dim3 block(tiled_beam_tile, tiled_time_tile, 1);
    tiled_packed_voltage_beamformer_kernel<<<grid, block, shared_bytes, stream>>>(
        packed_voltage, weights, intensity, dims.n_time, dims.n_freq,
        dims.n_ant, dims.n_beams);
    check_cuda(cudaGetLastError(), "tiled_packed_voltage_beamformer_kernel launch");
}

void launch_selected_kernel(const CudaBeamformerKernel kernel,
                            const std::uint8_t* packed_voltage,
                            const ComplexFloat* weights, float* intensity,
                            const Dimensions& dims, cudaStream_t stream) {
    switch (kernel) {
    case CudaBeamformerKernel::Direct:
        launch_direct_kernel(packed_voltage, weights, intensity, dims, stream);
        return;
    case CudaBeamformerKernel::Tiled:
        launch_tiled_kernel(packed_voltage, weights, intensity, dims, stream);
        return;
    }
    throw std::invalid_argument("unknown CUDA beamformer kernel selector");
}

std::size_t weight_storage_count(const Dimensions& dims,
                                 const CudaBeamformerKernel kernel) {
    return kernel == CudaBeamformerKernel::Tiled
               ? tiled_weight_count(dims)
               : dims.n_beams * dims.n_freq * dims.n_ant;
}

} // namespace

struct CudaBeamformerWorkspace::Impl {
    explicit Impl(const Dimensions& requested_capacity,
                  const CudaBeamformerKernel selected_kernel)
        : capacity(requested_capacity),
          kernel(selected_kernel),
          device_packed_voltage(voltage_sample_count(capacity)),
          device_weights(weight_storage_count(capacity, kernel)),
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
        const std::size_t required_weights = weight_storage_count(dims, kernel);
        if (loaded_weights != required_weights) {
            throw std::logic_error("weights have not been uploaded for these dimensions");
        }
    }

    Dimensions capacity;
    CudaBeamformerKernel kernel;
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

CudaBeamformerWorkspace::CudaBeamformerWorkspace(
    const Dimensions& capacity, const CudaBeamformerKernel kernel) {
    validate_dimensions(capacity);
    const auto start = Clock::now();
    impl_ = std::make_unique<Impl>(capacity, kernel);
    const auto end = Clock::now();
    impl_->measured_setup_ms = elapsed_ms(start, end);
}

CudaBeamformerWorkspace::~CudaBeamformerWorkspace() = default;

double CudaBeamformerWorkspace::setup_ms() const {
    return impl_->measured_setup_ms;
}

CudaBeamformerKernel CudaBeamformerWorkspace::kernel() const {
    return impl_->kernel;
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
    const std::size_t count = weight_storage_count(dims, impl_->kernel);
    if (weights.size() != count) {
        throw std::invalid_argument("weight count does not match selected kernel layout");
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
    launch_selected_kernel(impl_->kernel, impl_->device_packed_voltage.get(),
                            impl_->device_weights.get(), impl_->device_intensity.get(),
                            dims, impl_->stream.get());
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
    const std::size_t required_weights = weight_storage_count(dims, impl_->kernel);
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
    launch_selected_kernel(impl_->kernel, impl_->device_packed_voltage.get(),
                            impl_->device_weights.get(), impl_->device_intensity.get(),
                            dims, impl_->stream.get());
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

Intensities cuda_beamform_packed_intensity(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    CudaBeamformerTimings* timings, const CudaBeamformerKernel kernel) {
    validate_dimensions(dims);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const auto expected_weights = weight_storage_count(dims, kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match selected kernel layout");
    }

    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    CudaBeamformerWorkspace workspace(dims, kernel);
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
