#include "beamformer/cuda_frame.hpp"
#include "beamformer/quantization.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace beamformer;
    const auto shard = default_shard_descriptors()[1];
    const Dimensions dims{3, default_frequency_channels, 64, 64};
    void* const fake_device_pointer = reinterpret_cast<void*>(0x1000);

    const auto voltage =
        make_packed_voltage_frame_view(fake_device_pointer, dims, 42, shard);
    validate_cuda_frame_view(voltage);
    assert(voltage.buffer.type == CudaDataType::PackedInt4x2);
    assert(voltage.buffer.rank == 3);
    assert(voltage.buffer.shape[0] == 3);
    assert(voltage.buffer.shape[1] == 336);
    assert(voltage.buffer.shape[2] == 64);
    assert(voltage.buffer.strides_bytes[2] == 1);
    assert(voltage.buffer.strides_bytes[1] == 64);
    assert(voltage.buffer.strides_bytes[0] == 336 * 64);
    assert(voltage.buffer.bytes == packed_voltage_bytes(dims));
    assert(voltage.metadata.frame_id == 42);
    assert(voltage.metadata.shard.shard_id == 1);

    const auto intensity =
        make_intensity_frame_view(fake_device_pointer, dims, 42, shard);
    assert(intensity.buffer.type == CudaDataType::Float32);
    assert(intensity.buffer.strides_bytes[2] == sizeof(float));
    assert(intensity.buffer.strides_bytes[1] == 64 * sizeof(float));
    assert(intensity.buffer.bytes == intensity_bytes(dims));

    const auto weights =
        make_weights_frame_view(fake_device_pointer, dims, 0, shard);
    assert(weights.buffer.rank == 3);
    assert(weights.buffer.shape[0] == 64);
    assert(weights.buffer.shape[1] == 336);
    assert(weights.buffer.shape[2] == 64);
    assert(weights.buffer.bytes == weight_bytes(dims));

    const auto tiled_weights =
        make_tiled_weights_frame_view(fake_device_pointer, dims, 0, shard);
    assert(tiled_weights.buffer.rank == 4);
    assert(tiled_weights.buffer.shape[0] == 336);
    assert(tiled_weights.buffer.shape[1] == 2);
    assert(tiled_weights.buffer.shape[2] == 64);
    assert(tiled_weights.buffer.shape[3] == 32);
    assert(tiled_weights.buffer.bytes == tiled_weight_bytes(dims));

    const Dimensions integrated_dims{48, default_frequency_channels, 64, 64};
    const auto parameters = make_quantization_parameters_frame_view(
        fake_device_pointer, integrated_dims, 42, shard);
    validate_cuda_frame_view(parameters);
    assert(parameters.buffer.rank == 4);
    assert(parameters.buffer.shape[3] == 2);
    assert(parameters.buffer.bytes
           == quantization_parameter_count(integrated_dims) * 2 * sizeof(float));

    assert(throws_invalid_argument([&] {
        CudaBufferView invalid = voltage.buffer;
        invalid.bytes -= 1;
        validate_contiguous_cuda_buffer_view(invalid);
    }));
    assert(throws_invalid_argument([&] {
        CudaBufferView invalid = voltage.buffer;
        invalid.device_data = nullptr;
        validate_cuda_buffer_view(invalid);
    }));
    assert(throws_invalid_argument([&] {
        CudaBufferView invalid = voltage.buffer;
        invalid.strides_bytes[1] += 1;
        validate_contiguous_cuda_buffer_view(invalid);
    }));
    assert(throws_invalid_argument([&] {
        CudaFrameView invalid = voltage;
        invalid.metadata.shard.shard_id = 2;
        validate_cuda_frame_view(invalid);
    }));

    return 0;
}
