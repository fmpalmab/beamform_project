#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cstddef>
#include <cstdint>

namespace beamformer {

inline constexpr std::uint64_t spectrum_period_us_numerator = 10;
inline constexpr std::uint64_t spectrum_period_us_denominator = 3;

struct TemporalIntegrationConfig {
    std::size_t integration_spectra;
};

// The upchannelizer is a separate stage. These configurations describe only
// the number of spectra present in the input tensor for one integration sum.
inline constexpr TemporalIntegrationConfig integration_after_upchan{10};
inline constexpr TemporalIntegrationConfig integration_direct{320};

// Source-compatible names retained for callers that still use the old labels.
inline constexpr TemporalIntegrationConfig integration_with_upchan =
    integration_after_upchan;
inline constexpr TemporalIntegrationConfig integration_without_upchan =
    integration_direct;

inline constexpr std::size_t default_integration_spectra =
    integration_without_upchan.integration_spectra;

void validate_temporal_config(const TemporalIntegrationConfig& config);

std::size_t integrated_time_count(
    std::size_t n_time,
    std::size_t integration_spectra = default_integration_spectra);

std::size_t integrated_time_count(
    std::size_t n_time, const TemporalIntegrationConfig& config);

std::size_t integrated_intensity_count(
    const Dimensions& dims,
    std::size_t integration_spectra = default_integration_spectra);

std::size_t integrated_intensity_count(
    const Dimensions& dims, const TemporalIntegrationConfig& config);

// Output layout: [integration_window][local_frequency][beam]. Each value is
// the float32 sum of per-spectrum intensity values in that window.
using IntegratedIntensities = Intensities;

IntegratedIntensities cpu_integrate_intensity(
    const Intensities& intensity, const Dimensions& dims,
    std::size_t integration_spectra = default_integration_spectra);

IntegratedIntensities cpu_integrate_intensity(
    const Intensities& intensity, const Dimensions& dims,
    const TemporalIntegrationConfig& config);

// Writes the required output prefix and permits a larger reusable output
// buffer. Input and output remain independent; in-place integration is not
// supported.
void cpu_integrate_intensity_into(
    const Intensities& intensity, const Dimensions& dims,
    std::size_t integration_spectra, IntegratedIntensities& integrated);

} // namespace beamformer
