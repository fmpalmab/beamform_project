#pragma once

#include "beamformer/complex.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/quantization.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beamformer {

// Non-owning offline description of one device-side array. It does not
// allocate, free, synchronize, or copy memory.
inline constexpr std::size_t cuda_frame_max_rank = 4;

enum class CudaDataType {
    PackedInt4x2,
    ComplexFloat32,
    Float32,
    SignedInt8,
};

constexpr std::size_t cuda_data_type_size(const CudaDataType type) {
    switch (type) {
        case CudaDataType::PackedInt4x2:
        case CudaDataType::SignedInt8:
            return sizeof(std::uint8_t);
        case CudaDataType::ComplexFloat32:
            return sizeof(ComplexFloat);
        case CudaDataType::Float32:
            return sizeof(float);
    }
    throw std::invalid_argument("unknown CUDA data type");
}

struct CudaBufferView {
    // The stage using this descriptor determines read/write direction.
    // Ownership remains with the caller.
    void* device_data = nullptr;
    std::size_t bytes = 0;
    CudaDataType type = CudaDataType::Float32;
    std::size_t rank = 0;
    std::array<std::size_t, cuda_frame_max_rank> shape{};
    // Strides are measured in bytes.
    std::array<std::size_t, cuda_frame_max_rank> strides_bytes{};
};

struct CudaFrameMetadata {
    std::uint64_t frame_id = 0;
    ShardDescriptor shard;
};

struct CudaFrameView {
    CudaBufferView buffer;
    CudaFrameMetadata metadata;
};

inline std::size_t cuda_frame_element_count(
    const std::array<std::size_t, cuda_frame_max_rank>& shape,
    const std::size_t rank) {
    if (rank == 0 || rank > cuda_frame_max_rank) {
        throw std::invalid_argument("CUDA frame rank must be between 1 and 4");
    }
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] == 0
            || count > std::numeric_limits<std::size_t>::max() / shape[axis]) {
            throw std::invalid_argument("CUDA frame shape is empty or overflows size_t");
        }
        count *= shape[axis];
    }
    for (std::size_t axis = rank; axis < cuda_frame_max_rank; ++axis) {
        if (shape[axis] != 0) {
            throw std::invalid_argument("unused CUDA frame shape axes must be zero");
        }
    }
    return count;
}

inline std::array<std::size_t, cuda_frame_max_rank>
cuda_contiguous_strides_bytes(
    const CudaDataType type,
    const std::array<std::size_t, cuda_frame_max_rank>& shape,
    const std::size_t rank) {
    cuda_frame_element_count(shape, rank);
    std::array<std::size_t, cuda_frame_max_rank> strides{};
    std::size_t stride = cuda_data_type_size(type);
    for (std::size_t axis = rank; axis-- > 0;) {
        strides[axis] = stride;
        if (shape[axis] > std::numeric_limits<std::size_t>::max() / stride) {
            throw std::invalid_argument("CUDA frame strides overflow size_t");
        }
        stride *= shape[axis];
    }
    return strides;
}

inline std::size_t cuda_frame_minimum_storage_bytes(
    const CudaBufferView& buffer) {
    if (buffer.rank == 0 || buffer.rank > cuda_frame_max_rank) {
        throw std::invalid_argument("CUDA buffer rank must be between 1 and 4");
    }
    cuda_frame_element_count(buffer.shape, buffer.rank);
    const std::size_t element_size = cuda_data_type_size(buffer.type);
    std::size_t last_byte = element_size;
    for (std::size_t axis = 0; axis < buffer.rank; ++axis) {
        if (buffer.strides_bytes[axis] == 0) {
            throw std::invalid_argument("CUDA buffer strides must be positive");
        }
        const std::size_t extent = buffer.shape[axis] - 1;
        if (extent > (std::numeric_limits<std::size_t>::max() - last_byte)
                          / buffer.strides_bytes[axis]) {
            throw std::invalid_argument("CUDA buffer storage size overflows size_t");
        }
        last_byte += extent * buffer.strides_bytes[axis];
    }
    for (std::size_t axis = buffer.rank; axis < cuda_frame_max_rank; ++axis) {
        if (buffer.strides_bytes[axis] != 0) {
            throw std::invalid_argument("unused CUDA buffer strides must be zero");
        }
    }
    return last_byte;
}

inline bool cuda_buffer_view_is_contiguous(const CudaBufferView& buffer) {
    return buffer.strides_bytes
           == cuda_contiguous_strides_bytes(buffer.type, buffer.shape, buffer.rank);
}

inline void validate_cuda_buffer_view(const CudaBufferView& buffer) {
    if (buffer.device_data == nullptr) {
        throw std::invalid_argument("CUDA buffer device_data must not be null");
    }
    if (buffer.bytes < cuda_frame_minimum_storage_bytes(buffer)) {
        throw std::invalid_argument(
            "CUDA buffer bytes are smaller than its shape and strides");
    }
}

inline void validate_contiguous_cuda_buffer_view(const CudaBufferView& buffer) {
    validate_cuda_buffer_view(buffer);
    if (!cuda_buffer_view_is_contiguous(buffer)) {
        throw std::invalid_argument("CUDA frame must use contiguous storage");
    }
}

inline CudaFrameView make_contiguous_cuda_frame_view(
    void* device_data, std::size_t bytes, CudaDataType type,
    const std::array<std::size_t, cuda_frame_max_rank>& shape,
    std::size_t rank, std::uint64_t frame_id, const ShardDescriptor& shard) {
    validate_shard_descriptor(shard);
    CudaFrameView frame;
    frame.buffer.device_data = device_data;
    frame.buffer.bytes = bytes;
    frame.buffer.type = type;
    frame.buffer.rank = rank;
    frame.buffer.shape = shape;
    frame.buffer.strides_bytes = cuda_contiguous_strides_bytes(type, shape, rank);
    frame.metadata.frame_id = frame_id;
    frame.metadata.shard = shard;
    validate_contiguous_cuda_buffer_view(frame.buffer);
    return frame;
}

inline CudaFrameView make_packed_voltage_frame_view(
    void* device_data, const Dimensions& dims, std::uint64_t frame_id,
    const ShardDescriptor& shard) {
    validate_dimensions(dims);
    return make_contiguous_cuda_frame_view(
        device_data, packed_voltage_bytes(dims), CudaDataType::PackedInt4x2,
        {dims.n_time, dims.n_freq, dims.n_ant, 0}, 3, frame_id, shard);
}

inline CudaFrameView make_intensity_frame_view(
    void* device_data, const Dimensions& dims, std::uint64_t frame_id,
    const ShardDescriptor& shard) {
    validate_dimensions(dims);
    return make_contiguous_cuda_frame_view(
        device_data, intensity_bytes(dims), CudaDataType::Float32,
        {dims.n_time, dims.n_freq, dims.n_beams, 0}, 3, frame_id, shard);
}

inline CudaFrameView make_quantized_intensity_frame_view(
    void* device_data, const Dimensions& dims, std::uint64_t frame_id,
    const ShardDescriptor& shard) {
    validate_dimensions(dims);
    return make_contiguous_cuda_frame_view(
        device_data, quantized_intensity_bytes(dims), CudaDataType::SignedInt8,
        {dims.n_time, dims.n_freq, dims.n_beams, 0}, 3, frame_id, shard);
}

inline CudaFrameView make_weights_frame_view(
    void* device_data, const Dimensions& dims, std::uint64_t frame_id,
    const ShardDescriptor& shard) {
    validate_dimensions(dims);
    return make_contiguous_cuda_frame_view(
        device_data, weight_bytes(dims), CudaDataType::ComplexFloat32,
        {dims.n_beams, dims.n_freq, dims.n_ant, 0}, 3, frame_id, shard);
}

inline CudaFrameView make_tiled_weights_frame_view(
    void* device_data, const Dimensions& dims, std::uint64_t frame_id,
    const ShardDescriptor& shard) {
    validate_dimensions(dims);
    return make_contiguous_cuda_frame_view(
        device_data, tiled_weight_bytes(dims), CudaDataType::ComplexFloat32,
        {dims.n_freq, tiled_weight_beam_tiles(dims), dims.n_ant,
         tiled_weight_beam_tile}, 4, frame_id, shard);
}

inline CudaFrameView make_quantization_parameters_frame_view(
    void* device_data, const Dimensions& integrated_dims,
    std::uint64_t frame_id, const ShardDescriptor& shard) {
    validate_dimensions(integrated_dims);
    const auto layout = quantization_layout(integrated_dims);
    const auto parameter_count = quantization_parameter_count(integrated_dims);
    return make_contiguous_cuda_frame_view(
        device_data, parameter_count * 2 * sizeof(float), CudaDataType::Float32,
        {layout.time_tiles, layout.frequency_tiles, layout.beam_tiles, 2},
        4, frame_id, shard);
}

inline void validate_cuda_frame_view(const CudaFrameView& frame) {
    validate_cuda_buffer_view(frame.buffer);
    validate_shard_descriptor(frame.metadata.shard);
}

} // namespace beamformer
