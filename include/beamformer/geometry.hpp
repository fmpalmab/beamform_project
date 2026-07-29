#pragma once

#include "beamformer/config.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace beamformer {

using Vec3 = std::array<float, 3>;
using Vec2 = std::array<float, 2>;

struct FftBeamSelection {
    std::vector<Vec2> centers;
    std::size_t n_u = 0;
    std::size_t n_v = 0;
    std::size_t n_bank_u = 0;
    std::size_t n_bank_v = 0;
};

std::vector<Vec3> regular_array(std::size_t rows, std::size_t columns,
                                float spacing_m = default_spacing_m);
std::vector<Vec3> default_positions(std::size_t n_ant,
                                    float spacing_m = default_spacing_m);
std::vector<float> constant_frequencies(
    std::size_t n_freq = default_frequency_channels,
    float frequency_hz = beam_grid_design_frequency_hz);
std::vector<float> channelized_frequencies(
    std::size_t n_freq = default_frequency_channels,
    float start_hz = default_frequency_start_hz,
    float channel_width_hz = default_channel_width_hz);
Vec3 direction_from_lm(float l, float m);
std::vector<Vec3> default_beam_grid(std::size_t n_beams, float l_step = 0.02F,
                                    float m = 0.0F);
std::vector<long long> centered_integer_range(std::size_t count);
FftBeamSelection select_fft_beam_centers_rectangular(
    float du_fft_u, float dv_fft_v, std::size_t n_beams,
    std::size_t m_side, std::size_t n_side);
// Selects direction centers from the zero-padded 2D FFT bin geometry, but the
// beamformer itself remains the direct voltage-summing implementation.
std::vector<Vec3> fft_beam_grid(
    std::size_t n_ant, std::size_t n_beams,
    float spacing_m = default_spacing_m,
    float design_frequency_hz = beam_grid_design_frequency_hz);
std::size_t estimate_nbeams_hex_formula(
    float spacing_m, std::size_t m_side, std::size_t n_side,
    float bw_e_deg, float bw_h_deg, float wavelength_m);
std::vector<Vec2> generate_hex_targets_cropped_fov(
    std::size_t n_beams, float u_max, float v_max);

// Legacy N_beams=N_ant grid retained for reproducibility of existing
// point-source validation artifacts.
std::vector<Vec3> rectangular_beam_grid(
    std::size_t n_ant, float spacing_m = default_spacing_m,
    float design_frequency_hz = beam_grid_design_frequency_hz);

// Text files accept whitespace or commas and optional comments starting with '#'.
std::vector<Vec3> load_positions(const std::filesystem::path& path,
                                 std::size_t expected_count);
std::vector<float> load_frequencies(const std::filesystem::path& path,
                                    std::size_t expected_count);
std::vector<Vec3> load_directions(const std::filesystem::path& path,
                                  std::size_t expected_count);

} // namespace beamformer
