// src/frb_dispersion_helpers.cpp
//
// CPU dispersion-injection helpers for fast end-to-end classifier tests.
// Mirrors tools/astronomical_validation/injector.py's cold-plasma dispersion
// math (K_DM * dm * (f^-2 - f_ref^-2)) so C++ and Python tests cross-check.

#include "frb_dispersion_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace beamformer {

long long dispersion_shift_samples(const double dm_pc_cm3, const double freq_hz,
                                    const double f_ref_hz, const double dt_s) {
    if (dm_pc_cm3 < 0.0) {
        return 0;
    }
    const double f_mhz = freq_hz / 1.0e6;
    const double f_ref_mhz = f_ref_hz / 1.0e6;
    if (f_mhz <= 0.0 || f_ref_mhz <= 0.0) {
        return 0;
    }
    const double delay_s = k_dm_dispersion * dm_pc_cm3
                          * (1.0 / (f_mhz * f_mhz) - 1.0 / (f_ref_mhz * f_ref_mhz));
    if (!std::isfinite(delay_s)) {
        return 0;
    }
    const double samples = delay_s / dt_s;
    // Delays are non-negative for f < f_ref (cold plasma smears lower freq
    // later). Round-half-up to match the CUDA round() used in the kernel.
    return static_cast<long long>(std::floor(samples + 0.5));
}

// Maximum dispersion delay samples across the band, gated to n_time.
namespace {

long long max_shift_in_band(const Dimensions& dims, const FRBClassifierConfig& config,
                             float dm_pc_cm3, double& f_ref_hz_out,
                             std::vector<long long>& shifts_out) {
    const std::vector<float> freqs = config.frequencies(dims.n_freq);
    if (freqs.empty()) {
        f_ref_hz_out = config.freq_start_hz;
        return 0;
    }
    // f_ref = high edge of the band (last channel), per task spec.
    f_ref_hz_out = static_cast<double>(freqs[dims.n_freq - 1]);
    const double dt = spectrum_period_s;
    shifts_out.assign(dims.n_freq, 0);
    long long max_shift = 0;
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        const long long s = dispersion_shift_samples(
            static_cast<double>(dm_pc_cm3), static_cast<double>(freqs[f]), f_ref_hz_out, dt);
        shifts_out[f] = s;
        if (s > max_shift) {
            max_shift = s;
        }
    }
    return max_shift;
}

void inject_pulse(std::vector<float>& out, const Dimensions& dims,
                  const std::vector<long long>& shifts, std::size_t peak_time_index,
                  float amplitude, int width_samples) {
    const double sigma = std::max(1.0, static_cast<double>(width_samples) / 2.355);
    const double two_sigma_sq = 2.0 * sigma * sigma;
    const long long n_t = static_cast<long long>(dims.n_time);
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        long long center = (static_cast<long long>(peak_time_index) + shifts[f]) % n_t;
        if (center < 0) center += n_t;
        const int half = std::max(1, width_samples);
        for (int k = -half; k <= half; ++k) {
            long long t = (center + k) % n_t;
            if (t < 0) t += n_t;
            const double g = std::exp(-(static_cast<double>(k * k)) / two_sigma_sq);
            const float add = amplitude * static_cast<float>(g);
            for (std::size_t b = 0; b < dims.n_beams; ++b) {
                const std::size_t idx =
                    (static_cast<std::size_t>(t) * dims.n_freq + f) * dims.n_beams + b;
                out[idx] += add;
            }
        }
    }
}

} // namespace

Intensities add_dispersion_to_intensity(const Intensities& beamformed_intensity,
                                         const Dimensions& dims,
                                         const FRBClassifierConfig& config,
                                         const float dm_pc_cm3,
                                         const std::size_t peak_time_index,
                                         const float amplitude) {
    if (beamformed_intensity.size() != dims.n_time * dims.n_freq * dims.n_beams) {
        throw std::invalid_argument(
            "add_dispersion_to_intensity: intensity size does not match dims");
    }
    double f_ref_hz = config.freq_start_hz;
    std::vector<long long> shifts;
    max_shift_in_band(dims, config, dm_pc_cm3, f_ref_hz, shifts);

    Intensities out = beamformed_intensity;
    inject_pulse(out, dims, shifts, peak_time_index, amplitude, 4);
    return out;
}

Intensities make_dispersed_frb_intensity(const Dimensions& dims,
                                          const FRBClassifierConfig& config,
                                          const float dm_pc_cm3,
                                          const std::size_t peak_time_index,
                                          const float amplitude,
                                          const int width_samples) {
    std::vector<float> zeros(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    double f_ref_hz = config.freq_start_hz;
    std::vector<long long> shifts;
    max_shift_in_band(dims, config, dm_pc_cm3, f_ref_hz, shifts);
    inject_pulse(zeros, dims, shifts, peak_time_index, amplitude, width_samples);
    return zeros;
}

} // namespace beamformer
