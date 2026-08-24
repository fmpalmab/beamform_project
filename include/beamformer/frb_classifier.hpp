// include/beamformer/frb_classifier.hpp
//
// Phase 1 real-time FRB candidate detector + rule-based classifier header.
//
// Operates on the V5 tracker's device-resident intensity buffer
// (layout [time][freq][beam], float32, n_beams == 1 for the tracked beam).
// Stage 1 (this phase): hand-written dedispersion + boxcar CUDA kernels and an
// NMS/writer kernel produce Candidate structs into a device ring; the host
// rule classifier (R1-R7, Phase 1 subset) assigns labels and a free function
// emits the JSON candidate document.
//
// Phase 2 hooks (MAD RFI kernel, candidate-ring overflow, async consumer) are
// wired as slots but intentionally inert here; see the inline notes.

#pragma once

#include "beamformer/config.hpp"   // Dimensions
#include "beamformer/formats.hpp"  // Intensities

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace beamformer {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct FRBClassifierConfig {
    // DM-trial grid: dms[i] = dm_min + i * (dm_max - dm_min) / n_dm.
    // Phase 1 defaults: 0..1500 pc/cm^3 over 1024 trials. High-DM bursts
    // (> ~1500 pc/cm^3) are an accepted v1 miss.
    std::size_t n_dm = 1024;
    float dm_min = 0.0F;
    float dm_max = 1500.0F;

    // Boxcar widths (samples). Physical widths at dt = 10/3 us are
    // 0.0033..1.71 ms. The normalized boxcar divides the accumulated sum by
    // sqrt(W) for cross-width SNR comparability.
    std::vector<int> boxcar_widths = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    // Detection threshold on the normalized boxcar SNR.
    float snr_threshold = 6.0F;

    // Phase 2 hook: MAD RFI rejection n-sigma. The Phase 1 classifier does NOT
    // run the MAD kernel; the value is retained for forward-compatibility.
    float rfi_mad_nsigma = 5.0F;

    // Frequency descriptor for the device-local 300-400 MHz shard.
    // freq_start_hz + i * channel_width_hz mirrors channelized_frequencies().
    float freq_start_hz = 300e6F;
    float channel_width_hz = 300e3F;

    // Phase 2 hook: zero-DM subtraction before dedispersion. Default off.
    bool enable_zero_dm_subtraction = false;

    // Device candidate-ring capacity (one batch's survivors).
    std::size_t candidate_ring_capacity = 4096;

    // Reimplements channelized_frequencies() inline (K_DM math, classifier
    // tests) so the classifier stays self-contained — it does NOT call into
    // src/geometry.cpp. Same formula: f[c] = freq_start_hz + c * channel_width_hz.
    std::vector<float> frequencies(std::size_t n_freq) const {
        std::vector<float> f(n_freq);
        for (std::size_t c = 0; c < n_freq; ++c) {
            f[c] = freq_start_hz + static_cast<float>(c) * channel_width_hz;
        }
        return f;
    }

    // DM step on the configured grid (n_dm > 1). Zero if degenerate.
    float dm_step() const {
        return (n_dm > 1) ? (dm_max - dm_min) / static_cast<float>(n_dm) : 0.0F;
    }
};

// ---------------------------------------------------------------------------
// Candidate + labels
// ---------------------------------------------------------------------------
enum class CandidateLabel { Unknown, AstrophyicalFRB, RFI, Borderline };

struct Candidate {
    float snr = 0.0F;                // normalized boxcar SNR at (dm,t,W)
    float dm = 0.0F;                 // DM trial value (pc/cm^3)
    std::size_t time_index = 0;      // window-relative time sample
    int width_idx = 0;               // index into config.boxcar_widths
    int width_samples = 1;           // decoded width = boxcar_widths[width_idx]
    float baseline_mean = 0.0F;      // per-block running-average estimate
    float baseline_std = 1.0F;       // per-block std estimate
    CandidateLabel label = CandidateLabel::Unknown;
    // Normalized y_W at (dm, time_index) across the 10 boxcar widths, for the
    // host width-unimodality rule (R4) and downstream Python/NN consumption.
    std::array<float, 10> width_curve{};
};

// ---------------------------------------------------------------------------
// FRBClassifierStreamV5 — mirrors BatchedTrackerStreamV5 (Phase 1 subset).
//
// Non-owning / zero-copy on the existing V5 device intensity buffer. It does
// NOT take ownership of the V5 stream; it merely uses it for ordering.
// ---------------------------------------------------------------------------
class FRBClassifierStreamV5 {
public:
    // device_intensity_buffer: pointer to float32 [n_time][n_freq][n_beams]
    //   device-resident intensity (V5's d_intensity for the tracked beam).
    // device_stream: the V5 cudaStream_t (as void*); used only for ordering,
    //   not owned. Phase 1 fences synchronously via cudaStreamSynchronize.
    // dims: layout dims; Phase 1 classifier assumes the tracked beam slice
    //   (n_beams == 1 is the typical V5 output shape).
    FRBClassifierStreamV5(float* device_intensity_buffer, void* device_stream,
                          const Dimensions& dims, FRBClassifierConfig config);
    ~FRBClassifierStreamV5();

    FRBClassifierStreamV5(const FRBClassifierStreamV5&) = delete;
    FRBClassifierStreamV5& operator=(const FRBClassifierStreamV5&) = delete;

    // Launch dedisp+boxcar kernel then NMS kernel on the provided stream and
    // fence synchronously (Phase 1). Phase 2 will add an async-consumer path
    // triggered from here; left as a comment in the .cu implementation.
    void run(std::size_t window_index = 0);

    // Copies the surviving candidates out after a run() (host-side snapshot).
    std::vector<Candidate> candidates() const;

    float last_kernel_time_ms() const;   // dedisp+boxcar+NMS wall time
    float last_total_time_ms() const;    // total (incl. D2H copy)
    std::size_t ring_overflow_count() const; // Phase 1: always 0 (basic NMS)

private:
    struct Impl;
    Impl* impl_;
};

// ---------------------------------------------------------------------------
// Host rule classifier (Phase 1 subset R1, R2; R3/R4 placeholders).
// ---------------------------------------------------------------------------
std::vector<Candidate> classify_candidates(std::vector<Candidate> candidates,
                                           const FRBClassifierConfig& config);

// ---------------------------------------------------------------------------
// JSON emitter (schema_version 1, hand-rolled, ASCII, no external json dep).
// ---------------------------------------------------------------------------
std::string candidates_to_json(const std::vector<Candidate>& candidates,
                                const FRBClassifierConfig& config,
                                std::size_t window_index);

} // namespace beamformer
