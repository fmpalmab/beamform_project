#pragma once

#include <cstddef>

namespace beamformer {
namespace constants {

// Physical and Astrophysical Constants
inline constexpr double speed_of_light_m_per_s = 299792458.0;
inline constexpr double k_dm = 4148.741601; // MHz^2 s / (pc cm^-3)
inline constexpr double two_pi = 6.283185307179586476925286766559;

// Instrumental & Sampling Constants
inline constexpr double adc_sampling_freq_hz = 2.5e+09;
inline constexpr double adc_sampling_freq_mhz = 2457.6;
inline constexpr std::size_t fpga_num_samp_fft = 8192;
inline constexpr double fpga_time_resolution_s = 3.33333333333333332e-06; // ~3.333 us

// Band & Channel Specifications
inline constexpr float charts_channel_width_hz = 300000.0F;
inline constexpr float charts_channel_width_mhz = 0.3F;
inline constexpr float charts_frequency_start_hz = 300000000.0F;
inline constexpr float charts_design_frequency_hz = 400000000.0F;
inline constexpr std::size_t charts_full_band_channels = 672;
inline constexpr std::size_t charts_shard_count = 2;
inline constexpr std::size_t charts_local_channels = 336; // 336
inline constexpr std::size_t charts_upchannelized_factor = 32;
inline constexpr std::size_t charts_upchannelized_channels = 21504; // 21504

// Telescope Geometry & Array Defaults
inline constexpr float charts_default_spacing_m = 0.6F;
inline constexpr std::size_t charts_total_antennas = 256;

// Site Coordinates (Carén Observatory Site)
inline constexpr double charts_caren_lat_deg = -33.4211146;
inline constexpr double charts_caren_lon_deg = -70.8634710;
inline constexpr double charts_caren_alt_m = 458.0;

// CPT Dual-Band Networking Constants
inline constexpr double cpt_sample_rate_mhz = 4915.2;
inline constexpr double cpt_delta_time_s = 3.33333333333333332e-06;
inline constexpr std::size_t cpt_spectra_per_packet = 4;
inline constexpr std::size_t cpt_udp_payload_bytes = 5440;
inline constexpr std::size_t cpt_total_raw_packet_bytes = 5482;

} // namespace constants
} // namespace beamformer
