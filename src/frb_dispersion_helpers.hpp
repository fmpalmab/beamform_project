// src/frb_dispersion_helpers.hpp
//
// CPU dispersion-injection helpers for fast end-to-end classifier tests.
// Produces a synthetic dispersed FRB in beamformed intensity space (not voltage
// space), matching the Python injector's cold-plasma dispersion math so tests
// can cross-check.

#pragma once

#include "beamformer/config.hpp"            // Dimensions
#include "beamformer/formats.hpp"           // Intensities
#include "beamformer/frb_classifier.hpp"    // FRBClassifierConfig (frequencies)

#include <cstddef>

namespace beamformer {

// Dispersion constant K_DM in (s * MHz^2) / (pc * cm^-3). Matches official CHARTS constants.
inline constexpr double k_dm_dispersion = constants::k_dm;

// Sample period in seconds: 10/3 us (defined in temporal_integration.hpp and charts_constants).
inline constexpr double spectrum_period_s = constants::cpt_delta_time_s;

// Returns floor(K_DM * dm * (f_mhz^-2 - f_ref_mhz^-2) / dt) samples for the
// given channel frequency in Hz, using f_ref_hz as the high-band reference.
// Phase 1 uses the instrument's local 300-400 MHz high edge as f_ref; pass
// FRBClassifierConfig::frequencies(n_freq)[n_freq-1] for that.
long long dispersion_shift_samples(double dm_pc_cm3, double freq_hz,
                                    double f_ref_hz, double dt_s);

// Produces a synthetic dispersed FRB in beamformed intensity space.
//
// Output layout matches dims: [time][freq][beam], float32, total == dims.n_time
// * dims.n_freq * dims.n_beams. Handles n_beams == 1 (typical V5 output shape);
// for n_beams > 1 the pulse is replicated across beams.
//
// Injects a Gaussian + boxcar profile of the requested amplitude at each
// frequency channel at time `peak_time_index + shift_samples[f]`, where the
// shift is the cold-plasma dispersion delay (low frequencies arrive later).
// This matches the kernel's dedisp shift table for the same DM, so the kernel
// realigns the pulse at peak_time_index.
//
// DM budget: the maximum shift across the band (at the lowest freq) must fit
// inside [0, n_time). The helper gates dm to the largest value that keeps the
// highest-delay channel inside n_time and asserts if the caller's dm exceeds
// that budget (documented in the .cpp).
Intensities add_dispersion_to_intensity(const Intensities& beamformed_intensity,
                                         const Dimensions& dims,
                                         const FRBClassifierConfig& config,
                                         float dm_pc_cm3,
                                         std::size_t peak_time_index,
                                         float amplitude);

// Convenience: same as above but starting from a flat zero background, which
// is what the Phase 1 unit tests use (clean SNR recovery). Provided so tests
// don't have to zero-fill a vector first.
Intensities make_dispersed_frb_intensity(const Dimensions& dims,
                                          const FRBClassifierConfig& config,
                                          float dm_pc_cm3,
                                          std::size_t peak_time_index,
                                          float amplitude,
                                          int width_samples);

} // namespace beamformer
