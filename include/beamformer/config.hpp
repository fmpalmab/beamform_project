#pragma once

#include <cstddef>
#include <stdexcept>

namespace beamformer {

inline constexpr std::size_t default_frequency_channels = 672;
inline constexpr std::size_t maximum_beams = 128;
inline constexpr std::size_t rfsoc_channels_per_subband = 168;
inline constexpr std::size_t rfsoc_subbands_per_nic = 2;
inline constexpr std::size_t rfsoc_channels_per_nic =
    rfsoc_channels_per_subband * rfsoc_subbands_per_nic;
inline constexpr std::size_t rfsoc_elements_per_device = 32;
inline constexpr float default_frequency_start_hz = 300'000'000.0F;
inline constexpr float default_channel_width_hz = 300'000.0F;
inline constexpr float beam_grid_design_frequency_hz = 400'000'000.0F;
inline constexpr float default_spacing_m = 0.6F;

struct Dimensions {
    std::size_t n_time = 1024;
    std::size_t n_freq = default_frequency_channels;
    std::size_t n_ant = 32;
    std::size_t n_beams = 1;
};

inline void validate_dimensions(const Dimensions& dims) {
    if (dims.n_time == 0) {
        throw std::invalid_argument("n_time must be positive");
    }
    if (dims.n_freq != default_frequency_channels) {
        throw std::invalid_argument("the v1 PoC requires exactly 672 frequency channels");
    }
    if (dims.n_ant != 32 && dims.n_ant != 64) {
        throw std::invalid_argument("n_ant must be either 32 or 64");
    }
    if (dims.n_beams == 0 || dims.n_beams > maximum_beams) {
        throw std::invalid_argument("n_beams must be between 1 and 128");
    }
}

} // namespace beamformer
