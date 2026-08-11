#include "beamformer/cuda_beamformer.hpp"

#include "cuda_quantize.hpp"
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

__global__ void direct_packed_integrated_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights,
    float* integrated_intensity, const std::size_t output_count,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams,
    const std::size_t integration_spectra) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t beam = output_index % n_beams;
    const std::size_t window_frequency = output_index / n_beams;
    const std::size_t frequency = window_frequency % n_freq;
    const std::size_t window = window_frequency / n_freq;
    const std::size_t first_time = window * integration_spectra;
    const std::size_t last_time = first_time + integration_spectra < n_time
                                      ? first_time + integration_spectra
                                      : n_time;

    // Keep the CPU reference order: antenna accumulation, conversion to power,
    // then increasing-time float32 integration.
    float integrated = 0.0F;
    for (std::size_t time = first_time; time < last_time; ++time) {
        float sum_real = 0.0F;
        float sum_imag = 0.0F;
        const std::size_t voltage_offset = (time * n_freq + frequency) * n_ant;
        const std::size_t weight_offset = (beam * n_freq + frequency) * n_ant;
        for (std::size_t element = 0; element < n_ant; ++element) {
            const ComplexInt4 packed_sample =
                unpack_complex_int4(packed_voltage[voltage_offset + element]);
            const float sample_real = static_cast<float>(packed_sample.real);
            const float sample_imag = static_cast<float>(packed_sample.imag);
            const ComplexFloat weight = weights[weight_offset + element];
            sum_real += weight.real * sample_real - weight.imag * sample_imag;
            sum_imag += weight.real * sample_imag + weight.imag * sample_real;
        }
        integrated += sum_real * sum_real + sum_imag * sum_imag;
    }
    integrated_intensity[output_index] = integrated;
}

__global__ void tiled_packed_integrated_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights,
    float* integrated_intensity, const std::size_t n_time,
    const std::size_t n_freq, const std::size_t n_ant,
    const std::size_t n_beams, const std::size_t integration_spectra) {
    extern __shared__ unsigned char shared_storage[];
    auto* shared_weights = reinterpret_cast<ComplexFloat*>(shared_storage);
    auto* shared_voltage = shared_storage
                           + tiled_max_antennas * tiled_beam_tile
                                 * sizeof(ComplexFloat);
    auto* shared_power = reinterpret_cast<float*>(
        shared_voltage + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t));

    const std::size_t local_beam = threadIdx.x;
    const std::size_t local_time = threadIdx.y;
    const std::size_t frequency = blockIdx.y;
    const std::size_t beam = static_cast<std::size_t>(blockIdx.x)
                             * tiled_beam_tile + local_beam;
    const std::size_t window = blockIdx.z;
    const std::size_t linear_thread = local_time * tiled_beam_tile + local_beam;

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
    __syncthreads();

    const std::size_t first_time = window * integration_spectra;
    const std::size_t last_time = first_time + integration_spectra < n_time
                                      ? first_time + integration_spectra
                                      : n_time;
    float integrated = 0.0F;
    for (std::size_t group_offset = 0; group_offset < integration_spectra;
         group_offset += tiled_time_tile) {
        for (std::size_t index = linear_thread;
             index < tiled_time_tile * n_ant; index += tiled_threads) {
            const std::size_t tile_time = index / n_ant;
            const std::size_t antenna = index % n_ant;
            const std::size_t time = first_time + group_offset + tile_time;
            shared_voltage[tile_time * n_ant + antenna] = time < last_time
                ? packed_voltage[(time * n_freq + frequency) * n_ant + antenna]
                : 0U;
        }
        __syncthreads();

        float power = 0.0F;
        const std::size_t time = first_time + group_offset + local_time;
        if (beam < n_beams && time < last_time) {
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
            power = sum_real * sum_real + sum_imag * sum_imag;
        }
        shared_power[local_time * tiled_beam_tile + local_beam] = power;
        __syncthreads();

        if (local_time == 0 && beam < n_beams) {
            for (std::size_t tile_time = 0; tile_time < tiled_time_tile; ++tile_time) {
                if (first_time + group_offset + tile_time < last_time) {
                    integrated += shared_power[tile_time * tiled_beam_tile + local_beam];
                }
            }
        }
        __syncthreads();
    }

    if (local_time == 0 && beam < n_beams) {
        integrated_intensity[(window * n_freq + frequency) * n_beams + beam] = integrated;
    }
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

void validate_cuda_temporal_integration(
    const CudaBeamformerKernel kernel,
    const TemporalIntegrationConfig& temporal_integration) {
    validate_temporal_config(temporal_integration);
    const std::size_t spectra = temporal_integration.integration_spectra;
    if (spectra != integration_after_upchan.integration_spectra
        && spectra != integration_direct.integration_spectra) {
        throw std::invalid_argument(
            "CUDA temporal integration supports only 10 or 320 spectra");
    }
    if (kernel == CudaBeamformerKernel::Direct
        && spectra != integration_direct.integration_spectra) {
        throw std::invalid_argument(
            "the Direct CUDA kernel supports only 320-spectrum temporal integration");
    }
}


std::size_t weight_storage_count(const Dimensions& dims,
                                 const CudaBeamformerKernel kernel) {
    return kernel == CudaBeamformerKernel::Tiled
               ? tiled_weight_count(dims)
               : dims.n_beams * dims.n_freq * dims.n_ant;
}

bool same_shard(const ShardDescriptor& lhs, const ShardDescriptor& rhs) {
    return lhs.shard_id == rhs.shard_id
           && lhs.shard_count == rhs.shard_count
           && lhs.local_frequency_count == rhs.local_frequency_count
           && lhs.absolute_frequency_start == rhs.absolute_frequency_start
           && lhs.timestamp_start == rhs.timestamp_start
           && lhs.timestamp_step == rhs.timestamp_step
           && lhs.loss_mask_id == rhs.loss_mask_id
           && lhs.loss_mask_independent == rhs.loss_mask_independent;
}

void validate_matching_frame_metadata(const CudaFrameMetadata& reference,
                                      const CudaFrameMetadata& candidate,
                                      const char* label,
                                      const bool require_frame_id) {
    if (require_frame_id && reference.frame_id != candidate.frame_id) {
        throw std::invalid_argument(std::string(label) + " frame_id does not match input");
    }
    if (!same_shard(reference.shard, candidate.shard)) {
        throw std::invalid_argument(std::string(label) + " shard metadata does not match input");
    }
}

void validate_device_frame_shape(const CudaFrameView& frame,
                                 const CudaDataType type,
                                 const std::size_t rank,
                                 const std::array<std::size_t, cuda_frame_max_rank>& shape,
                                 const char* label) {
    validate_contiguous_cuda_buffer_view(frame.buffer);
    if (frame.buffer.type != type || frame.buffer.rank != rank
        || frame.buffer.shape != shape) {
        throw std::invalid_argument(std::string(label) + " shape or dtype does not match");
    }
}

} // namespace
void launch_packed_beamformer(const CudaBeamformerKernel kernel,
                              const cudaStream_t stream,
                              const std::uint8_t* packed_voltage,
                              const ComplexFloat* weights, float* intensity,
                              const Dimensions& dims) {
    validate_dimensions(dims);
    launch_selected_kernel(kernel, packed_voltage, weights, intensity, dims, stream);
}

void launch_packed_integrated_beamformer(
    const CudaBeamformerKernel kernel, const cudaStream_t stream,
    const std::uint8_t* packed_voltage, const ComplexFloat* weights,
    float* integrated_intensity, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration) {
    validate_dimensions(dims);
    validate_cuda_temporal_integration(kernel, temporal_integration);
    constexpr std::size_t threads_per_block = 256;
    const std::size_t output_count =
        integrated_intensity_count(dims, temporal_integration);
    const std::size_t block_count =
        (output_count + threads_per_block - 1) / threads_per_block;
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error(
            "CUDA integrated beamformer grid exceeds the supported one-dimensional size");
    }

    if (kernel == CudaBeamformerKernel::Direct) {
        direct_packed_integrated_voltage_beamformer_kernel<<<
            static_cast<unsigned int>(block_count),
            static_cast<unsigned int>(threads_per_block), 0, stream>>>(
            packed_voltage, weights, integrated_intensity, output_count,
            dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
            temporal_integration.integration_spectra);
        check_cuda(cudaGetLastError(),
                   "direct_packed_integrated_voltage_beamformer_kernel launch");
        return;
    }
    if (kernel == CudaBeamformerKernel::Tiled) {
        const auto grid_x = (dims.n_beams + tiled_beam_tile - 1) / tiled_beam_tile;
        const auto window_count =
            integrated_time_count(dims.n_time, temporal_integration);
        if (grid_x > std::numeric_limits<unsigned int>::max()
            || dims.n_freq > std::numeric_limits<unsigned int>::max()
            || window_count > std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error(
                "CUDA tiled integrated beamformer grid exceeds the supported size");
        }
        constexpr std::size_t shared_bytes =
            tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat)
            + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t)
            + tiled_time_tile * tiled_beam_tile * sizeof(float);
        const dim3 grid(static_cast<unsigned int>(grid_x),
                        static_cast<unsigned int>(dims.n_freq),
                        static_cast<unsigned int>(window_count));
        const dim3 block(tiled_beam_tile, tiled_time_tile, 1);
        tiled_packed_integrated_voltage_beamformer_kernel<<<grid, block, shared_bytes,
                                                              stream>>>(
            packed_voltage, weights, integrated_intensity, dims.n_time, dims.n_freq,
            dims.n_ant, dims.n_beams, temporal_integration.integration_spectra);
        check_cuda(cudaGetLastError(),
                   "tiled_packed_integrated_voltage_beamformer_kernel launch");
        return;
    }
    throw std::invalid_argument("unknown CUDA beamformer kernel selector");
}

struct CudaBeamformerWorkspace::Impl {
    explicit Impl(const Dimensions& requested_capacity,
                  const CudaBeamformerKernel selected_kernel,
                  const std::optional<TemporalIntegrationConfig>& selected_integration,
                  const CudaBeamformerOutput selected_output)
        : capacity(requested_capacity),
          kernel(selected_kernel),
          temporal_integration(selected_integration),
          output(selected_output),
          device_packed_voltage(voltage_sample_count(capacity)),
          device_weights(weight_storage_count(capacity, kernel)),
          device_intensity(capacity.n_time * capacity.n_freq * capacity.n_beams) {
        if (temporal_integration) {
            device_integrated_intensity = std::make_unique<DeviceBuffer<float>>(
                integrated_intensity_count(capacity, *temporal_integration));
            if (output == CudaBeamformerOutput::QuantizedInt8) {
                const Dimensions integrated_capacity{
                    integrated_time_count(capacity.n_time, *temporal_integration),
                    capacity.n_freq, capacity.n_ant, capacity.n_beams};
                device_quantized_intensity = std::make_unique<DeviceBuffer<std::int8_t>>(
                    integrated_intensity_count(capacity, *temporal_integration));
                device_quantization_parameters =
                    std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
                        quantization_parameter_count(integrated_capacity));
            }
        }
    }

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

    void require_temporal_integration() const {
        if (!temporal_integration || !device_integrated_intensity) {
            throw std::logic_error(
                "CUDA workspace was not constructed with temporal integration");
        }
    }

    void require_quantization() const {
        require_temporal_integration();
        if (output != CudaBeamformerOutput::QuantizedInt8
            || !device_quantized_intensity || !device_quantization_parameters) {
            throw std::logic_error(
                "CUDA workspace was not constructed with int8 quantization");
        }
    }

    Dimensions capacity;
    CudaBeamformerKernel kernel;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutput output;
    DeviceBuffer<std::uint8_t> device_packed_voltage;
    DeviceBuffer<ComplexFloat> device_weights;
    DeviceBuffer<float> device_intensity;
    std::unique_ptr<DeviceBuffer<float>> device_integrated_intensity;
    std::unique_ptr<DeviceBuffer<std::int8_t>> device_quantized_intensity;
    std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>>
        device_quantization_parameters;
    CudaStream stream;
    CudaEvent start;
    CudaEvent transfer_end;
    CudaEvent kernel_end;
    CudaEvent integration_end;
    CudaEvent quantization_end;
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
    const Dimensions& capacity, const CudaBeamformerKernel kernel,
    const std::optional<TemporalIntegrationConfig> temporal_integration,
    const CudaBeamformerOutput output) {
    validate_dimensions(capacity);
    if (temporal_integration) {
        validate_cuda_temporal_integration(kernel, *temporal_integration);
    }
    if (output == CudaBeamformerOutput::QuantizedInt8 && !temporal_integration) {
        throw std::invalid_argument(
            "int8 quantization requires a temporal-integration configuration");
    }
    const auto start = Clock::now();
    impl_ = std::make_unique<Impl>(capacity, kernel, temporal_integration, output);
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

bool CudaBeamformerWorkspace::has_temporal_integration() const {
    return impl_->temporal_integration.has_value();
}

bool CudaBeamformerWorkspace::has_int8_quantization() const {
    return impl_->output == CudaBeamformerOutput::QuantizedInt8;
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

CudaBeamformerTimings CudaBeamformerWorkspace::run_integrated_pipeline(
    const PackedVoltage& packed, IntegratedIntensities& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_temporal_integration();
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t output_count =
        integrated_intensity_count(dims, *impl_->temporal_integration);
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.size() < output_count) {
        throw std::invalid_argument("integrated intensity output is smaller than dimensions");
    }
    const std::size_t required_weights = weight_storage_count(dims, impl_->kernel);
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync integrated pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline transfer end");
    launch_packed_integrated_beamformer(
        impl_->kernel, impl_->stream.get(), impl_->device_packed_voltage.get(),
        impl_->device_weights.get(), impl_->device_integrated_intensity->get(), dims,
        *impl_->temporal_integration);
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline fused beamformer end");
    check_cuda(cudaMemcpyAsync(intensity.data(), impl_->device_integrated_intensity->get(),
                               output_count * sizeof(float), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync integrated pipeline intensity device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord integrated pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize integrated pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimings timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->kernel_end,
                                                 impl_->result_end);
    return timings;
}

CudaBeamformerTimings CudaBeamformerWorkspace::run_quantized_integrated_pipeline(
    const PackedVoltage& packed, QuantizedIntegratedOutput& intensity,
    const Dimensions& dims) {
    impl_->validate_request(dims);
    impl_->require_quantization();
    const std::size_t packed_count = voltage_sample_count(dims);
    const std::size_t integrated_count =
        integrated_intensity_count(dims, *impl_->temporal_integration);
    const Dimensions integrated_dims{
        integrated_time_count(dims.n_time, *impl_->temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    const std::size_t parameter_count = quantization_parameter_count(integrated_dims);
    if (packed.size() < packed_count) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    if (intensity.codes.size() < integrated_count
        || intensity.parameters.size() < parameter_count) {
        throw std::invalid_argument("quantized output is smaller than dimensions");
    }
    const std::size_t required_weights = weight_storage_count(dims, impl_->kernel);
    if (impl_->loaded_weights != required_weights) {
        throw std::logic_error("weights have not been uploaded for these dimensions");
    }

    check_cuda(cudaEventRecord(impl_->start.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline start");
    check_cuda(cudaMemcpyAsync(impl_->device_packed_voltage.get(), packed.data(), packed_count,
                               cudaMemcpyHostToDevice, impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline packed voltage host to device");
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline transfer end");
    launch_packed_integrated_beamformer(
        impl_->kernel, impl_->stream.get(), impl_->device_packed_voltage.get(),
        impl_->device_weights.get(), impl_->device_integrated_intensity->get(), dims,
        *impl_->temporal_integration);
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline fused beamformer end");
    launch_quantize_integrated_intensity(
        impl_->stream.get(), impl_->device_integrated_intensity->get(),
        impl_->device_quantized_intensity->get(),
        impl_->device_quantization_parameters->get(), integrated_dims);
    check_cuda(cudaEventRecord(impl_->quantization_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline quantization end");
    check_cuda(cudaMemcpyAsync(intensity.codes.data(), impl_->device_quantized_intensity->get(),
                               integrated_count * sizeof(std::int8_t), cudaMemcpyDeviceToHost,
                               impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline codes device to host");
    check_cuda(cudaMemcpyAsync(intensity.parameters.data(),
                               impl_->device_quantization_parameters->get(),
                               parameter_count * sizeof(Int8QuantizationParameters),
                               cudaMemcpyDeviceToHost, impl_->stream.get()),
               "cudaMemcpyAsync quantized pipeline parameters device to host");
    check_cuda(cudaEventRecord(impl_->result_end.get(), impl_->stream.get()),
               "cudaEventRecord quantized pipeline result end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize quantized pipeline");
    impl_->loaded_packed_voltage_samples = packed_count;

    CudaBeamformerTimings timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->transfer_end);
    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    timings.quantization_ms =
        event_elapsed_ms(impl_->kernel_end, impl_->quantization_end);
    timings.device_to_host_ms = event_elapsed_ms(impl_->quantization_end,
                                                 impl_->result_end);
    return timings;
}

CudaBeamformerTimings CudaBeamformerWorkspace::run_device_frame(
    const CudaFrameView& voltage, const CudaFrameView& weights,
    CudaFrameView& output, CudaFrameView* quantization_parameters) {
    validate_cuda_frame_view(voltage);
    validate_cuda_frame_view(weights);
    validate_cuda_frame_view(output);
    validate_matching_frame_metadata(voltage.metadata, output.metadata,
                                     "output", true);

    if (voltage.buffer.type != CudaDataType::PackedInt4x2
        || voltage.buffer.rank != 3) {
        throw std::invalid_argument(
            "offline device frame voltage must be packed int4x2 with rank 3");
    }
    const Dimensions dims{
        voltage.buffer.shape[0], voltage.buffer.shape[1], voltage.buffer.shape[2],
        output.buffer.shape[2]};
    impl_->validate_request(dims);
    if (voltage.buffer.shape[1] != impl_->capacity.n_freq
        || output.buffer.rank != 3
        || output.buffer.shape[1] != dims.n_freq
        || output.buffer.shape[2] != dims.n_beams) {
        throw std::invalid_argument("offline device frame dimensions do not match");
    }

    const std::array<std::size_t, cuda_frame_max_rank> expected_weight_shape =
        impl_->kernel == CudaBeamformerKernel::Direct
            ? std::array<std::size_t, cuda_frame_max_rank>{
                  dims.n_beams, dims.n_freq, dims.n_ant, 0}
            : std::array<std::size_t, cuda_frame_max_rank>{
                  dims.n_freq, tiled_weight_beam_tiles(dims), dims.n_ant,
                  tiled_weight_beam_tile};
    validate_device_frame_shape(weights, CudaDataType::ComplexFloat32,
                                impl_->kernel == CudaBeamformerKernel::Direct ? 3 : 4,
                                expected_weight_shape, "offline device frame weights");
    if (!same_shard(voltage.metadata.shard, weights.metadata.shard)) {
        throw std::invalid_argument("offline device frame weights shard does not match input");
    }

    Dimensions output_dims = dims;
    if (impl_->temporal_integration) {
        output_dims.n_time = integrated_time_count(dims.n_time, *impl_->temporal_integration);
    }
    const auto expected_output_type =
        impl_->output == CudaBeamformerOutput::QuantizedInt8
            ? CudaDataType::SignedInt8
            : CudaDataType::Float32;
    validate_device_frame_shape(
        output, expected_output_type, 3,
        {output_dims.n_time, output_dims.n_freq, output_dims.n_beams, 0},
        "offline device frame output");

    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        if (!impl_->temporal_integration || quantization_parameters == nullptr) {
            throw std::invalid_argument(
                "offline int8 output requires temporal integration parameters");
        }
        validate_cuda_frame_view(*quantization_parameters);
        validate_matching_frame_metadata(
            voltage.metadata, quantization_parameters->metadata,
            "quantization parameters", true);
        const auto layout = quantization_layout(output_dims);
        validate_device_frame_shape(
            *quantization_parameters, CudaDataType::Float32, 4,
            {layout.time_tiles, layout.frequency_tiles, layout.beam_tiles, 2},
            "offline quantization parameters");
    } else if (quantization_parameters != nullptr) {
        throw std::invalid_argument(
            "float32 offline output must not provide quantization parameters");
    }

    const auto stream = impl_->stream.get();
    check_cuda(cudaEventRecord(impl_->start.get(), stream),
               "cudaEventRecord offline frame start");
    // The input and weights are already device-resident. This event pair keeps
    // the timing structure compatible with the regular workspace path.
    check_cuda(cudaEventRecord(impl_->transfer_end.get(), stream),
               "cudaEventRecord offline frame input end");
    if (impl_->temporal_integration) {
        launch_packed_integrated_beamformer(
            impl_->kernel, stream,
            static_cast<const std::uint8_t*>(voltage.buffer.device_data),
            static_cast<const ComplexFloat*>(weights.buffer.device_data),
            impl_->device_integrated_intensity->get(), dims,
            *impl_->temporal_integration);
    } else {
        launch_packed_beamformer(
            impl_->kernel, stream,
            static_cast<const std::uint8_t*>(voltage.buffer.device_data),
            static_cast<const ComplexFloat*>(weights.buffer.device_data),
            impl_->device_intensity.get(), dims);
    }
    check_cuda(cudaEventRecord(impl_->kernel_end.get(), stream),
               "cudaEventRecord offline frame beamformer end");

    CudaBeamformerTimings timings;
    timings.host_to_device_ms = 0.0;
    const CudaEvent* producer_end = &impl_->kernel_end;

    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        const Dimensions integrated_dims{
            integrated_time_count(dims.n_time, *impl_->temporal_integration),
            dims.n_freq, dims.n_ant, dims.n_beams};
        launch_quantize_integrated_intensity(
            stream, impl_->device_integrated_intensity->get(),
            impl_->device_quantized_intensity->get(),
            impl_->device_quantization_parameters->get(), integrated_dims);
        check_cuda(cudaEventRecord(impl_->quantization_end.get(), stream),
                   "cudaEventRecord offline frame quantization end");
        producer_end = &impl_->quantization_end;
    }

    const std::size_t output_bytes =
        impl_->output == CudaBeamformerOutput::QuantizedInt8
            ? quantized_intensity_bytes(output_dims)
            : intensity_bytes(output_dims);
    const void* source = impl_->output == CudaBeamformerOutput::QuantizedInt8
                             ? static_cast<const void*>(
                                   impl_->device_quantized_intensity->get())
                             : static_cast<const void*>(
                                   impl_->temporal_integration
                                       ? impl_->device_integrated_intensity->get()
                                       : impl_->device_intensity.get());
    check_cuda(cudaMemcpyAsync(output.buffer.device_data, source, output_bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync offline frame output device to device");
    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        const auto parameter_bytes =
            quantization_parameter_count(output_dims) * sizeof(Int8QuantizationParameters);
        check_cuda(cudaMemcpyAsync(
                       quantization_parameters->buffer.device_data,
                       impl_->device_quantization_parameters->get(), parameter_bytes,
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync offline frame parameters device to device");
    }
    check_cuda(cudaEventRecord(impl_->result_end.get(), stream),
               "cudaEventRecord offline frame output end");
    check_cuda(cudaEventSynchronize(impl_->result_end.get()),
               "cudaEventSynchronize offline frame");

    timings.kernel_ms = event_elapsed_ms(impl_->transfer_end, impl_->kernel_end);
    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        timings.quantization_ms =
            event_elapsed_ms(impl_->kernel_end, impl_->quantization_end);
    }
    timings.device_to_device_ms =
        event_elapsed_ms(*producer_end, impl_->result_end);
    timings.device_to_host_ms = 0.0;
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

IntegratedIntensities cuda_beamform_packed_integrated_intensity(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimings* timings, const CudaBeamformerKernel kernel) {
    validate_dimensions(dims);
    validate_cuda_temporal_integration(kernel, temporal_integration);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const auto expected_weights = weight_storage_count(dims, kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match selected kernel layout");
    }

    IntegratedIntensities intensity(
        integrated_intensity_count(dims, temporal_integration));
    CudaBeamformerWorkspace workspace(dims, kernel, temporal_integration);
    CudaBeamformerTimings measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_integrated_pipeline(packed, intensity, dims);
    measured.host_to_device_ms += pipeline.host_to_device_ms;
    measured.kernel_ms = pipeline.kernel_ms;
    measured.temporal_integration_ms = pipeline.temporal_integration_ms;
    measured.device_to_host_ms = pipeline.device_to_host_ms;
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

QuantizedIntegratedOutput cuda_beamform_packed_quantized_integrated_intensity(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const TemporalIntegrationConfig& temporal_integration,
    CudaBeamformerTimings* timings, const CudaBeamformerKernel kernel) {
    validate_dimensions(dims);
    validate_cuda_temporal_integration(kernel, temporal_integration);
    if (packed.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    const auto expected_weights = weight_storage_count(dims, kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("weight count does not match selected kernel layout");
    }
    const Dimensions integrated_dims{
        integrated_time_count(dims.n_time, temporal_integration),
        dims.n_freq, dims.n_ant, dims.n_beams};
    QuantizedIntegratedOutput intensity{
        QuantizedIntensities(integrated_intensity_count(dims, temporal_integration)),
        std::vector<Int8QuantizationParameters>(quantization_parameter_count(integrated_dims)),
    };
    CudaBeamformerWorkspace workspace(
        dims, kernel, temporal_integration, CudaBeamformerOutput::QuantizedInt8);
    CudaBeamformerTimings measured;
    measured.setup_ms = workspace.setup_ms();
    measured.host_to_device_ms = workspace.upload_weights(weights, dims);
    const auto pipeline = workspace.run_quantized_integrated_pipeline(packed, intensity, dims);
    measured.host_to_device_ms += pipeline.host_to_device_ms;
    measured.kernel_ms = pipeline.kernel_ms;
    measured.temporal_integration_ms = pipeline.temporal_integration_ms;
    measured.quantization_ms = pipeline.quantization_ms;
    measured.device_to_host_ms = pipeline.device_to_host_ms;
    if (timings != nullptr) {
        *timings = measured;
    }
    return intensity;
}

} // namespace beamformer
