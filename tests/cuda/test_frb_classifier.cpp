// tests/cuda/test_frb_classifier.cpp
//
// Phase 1 FRB classifier end-to-end tests (sub-tests a-d only). Builds the
// input by allocating a device intensity buffer directly (cudaMalloc/cudaMemcpy)
// and constructing a cudaStream_t — does NOT instantiate BatchedTrackerStreamV5.
// Plain assert/printf style; no external test framework. main() runs all four
// sub-tests, returns nonzero on failure.
//
// (a) synthetic FRB injection end-to-end
// (b) known-DM recovery across dm ∈ {50, 200, 500, 800, 1200}
// (c) boxcar-width recovery across injected width ∈ {4, 16, 64, 256}
// (d) RFI rejection of a zero-DM undispersed narrow-band constant pulse

#include "beamformer/frb_classifier.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include "frb_dispersion_helpers.hpp" // make_dispersed_frb_intensity

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace beamformer;

namespace {

// Project-style CUDA error helper (mirrors check_cuda in the V5 TU).
void check_cuda(const cudaError_t e, const char* op) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", op, cudaGetErrorString(e));
        std::abort();
    }
}

// Test dimensions kept small for fast runs; the classifier is config-driven
// so this is legitimate. n_beams == 1 (typical V5 tracked-beam output).
Dimensions make_test_dims(std::size_t n_time) {
    Dimensions d;
    d.n_time = n_time;
    d.n_freq = default_frequency_channels; // 336
    d.n_ant = 64;                          // classifier ignores antennas
    d.n_beams = 1;
    return d;
}

// Snap the desired DM onto the nearest grid trial so the injected pulse aligns
// exactly with a kernel DM trial -> clean recovery.
float snap_dm_to_grid(const FRBClassifierConfig& cfg, float dm) {
    if (cfg.n_dm <= 1) return cfg.dm_min;
    const float step = cfg.dm_step();
    const float idx = std::round((dm - cfg.dm_min) / step);
    return cfg.dm_min + idx * step;
}

// Convenience: build intensities on device, run classifier, return survivors
// after host classification.
std::vector<Candidate> run_classifier_for_intensity(
    const std::vector<float>& intensity_host, const Dimensions& dims,
    const FRBClassifierConfig& cfg) {
    const std::size_t bytes = intensity_host.size() * sizeof(float);
    float* d_intensity = nullptr;
    check_cuda(cudaMalloc(&d_intensity, bytes), "cudaMalloc test intensity");
    assert(d_intensity != nullptr);
    check_cuda(cudaMemcpy(d_intensity, intensity_host.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D intensity");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate test");

    FRBClassifierStreamV5 classifier(d_intensity, static_cast<void*>(stream), dims, cfg);
    classifier.run(/*window_index*/ 0);

    std::vector<Candidate> raw = classifier.candidates();
    // Host rule classifier runs on the kernel survivors (R1/R2 labels).
    std::vector<Candidate> labeled = classify_candidates(std::move(raw), cfg);

    check_cuda(cudaStreamSynchronize(stream), "final sync");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(d_intensity), "cudaFree intensity");
    return labeled;
}

// Locate the highest-SNR candidate of a given (or any) label. Returns nullptr
// if none match.
const Candidate* best_candidate(const std::vector<Candidate>& cs,
                                 CandidateLabel want = CandidateLabel::AstrophyicalFRB) {
    const Candidate* best = nullptr;
    for (const auto& c : cs) {
        if (c.label == want) {
            if (!best || c.snr > best->snr) best = &c;
        }
    }
    return best;
}

// Find the global max-SNR candidate regardless of label.
const Candidate* max_snr_candidate(const std::vector<Candidate>& cs) {
    const Candidate* best = nullptr;
    for (const auto& c : cs) {
        if (!best || c.snr > best->snr) best = &c;
    }
    return best;
}

// ---------------------------------------------------------------------------
// (a) synthetic FRB injection end-to-end
// ---------------------------------------------------------------------------
bool test_a_end_to_end() {
    std::printf("=== (a) synthetic FRB injection end-to-end ===\n");
    const std::size_t n_time = 16384;
    const Dimensions dims = make_test_dims(n_time);
    FRBClassifierConfig cfg;
    cfg.n_dm = 1024;
    cfg.dm_min = 0.0F;
    cfg.dm_max = 1500.0F;
    cfg.snr_threshold = 6.0F;

    const float injected_dm = snap_dm_to_grid(cfg, 312.5F);
    const std::size_t peak = 2000;
    const float amplitude = 64.0F; // strong, clean SNR recovery
    const int width = 8;

    Intensities intensity = make_dispersed_frb_intensity(dims, cfg, injected_dm, peak, amplitude, width);
    auto candidates = run_classifier_for_intensity(intensity, dims, cfg);

    if (candidates.empty()) {
        std::fprintf(stderr, "FAIL (a): no candidate returned\n");
        return false;
    }
    // Print the JSON skeleton for the recovered batch (informational).
    const std::string json = candidates_to_json(candidates, cfg, /*window_index*/ 0);
    std::printf("%s", json.c_str());

    const Candidate* best = max_snr_candidate(candidates);
    if (!best) {
        std::fprintf(stderr, "FAIL (a): no max-SNR candidate\n");
        return false;
    }
    // Recovery tolerances: DM within 1 grid step; time within a few samples;
    // SNR > threshold.
    const float dm_step = cfg.dm_step();
    const float dm_err = std::abs(best->dm - injected_dm);
    const std::size_t time_err = static_cast<std::size_t>(std::abs(
        static_cast<double>(best->time_index) - static_cast<double>(peak)));
    const bool dm_ok = dm_err <= (1.5F * dm_step);
    const bool time_ok = time_err <= 4;
    const bool snr_ok = best->snr >= cfg.snr_threshold;
    std::printf("  injected dm=%.3f peak=%zu width=%d | recovered dm=%.3f t=%zu snr=%.3f width=%d label=%d\n",
                injected_dm, peak, width, best->dm, best->time_index, best->snr,
                best->width_samples, static_cast<int>(best->label));
    if (!dm_ok || !time_ok || !snr_ok) {
        std::fprintf(stderr, "FAIL (a): dm_err=%.3f time_err=%zu snr_ok=%d\n", dm_err, time_err, snr_ok);
        return false;
    }
    std::printf("  -> PASS (a)\n");
    return true;
}

// ---------------------------------------------------------------------------
// (b) known-DM recovery
// ---------------------------------------------------------------------------
bool test_b_known_dm() {
    std::printf("=== (b) known-DM recovery ===\n");
    const std::size_t n_time = 16384;
    const Dimensions dims = make_test_dims(n_time);
    FRBClassifierConfig cfg;
    cfg.n_dm = 1024;
    cfg.dm_min = 0.0F;
    cfg.dm_max = 1500.0F;
    cfg.snr_threshold = 6.0F;
    const float dm_step = cfg.dm_step();

    const float injected_dms[] = {50.0F, 200.0F, 500.0F, 800.0F, 1200.0F};
    bool all_ok = true;
    for (float requested : injected_dms) {
        const float injected_dm = snap_dm_to_grid(cfg, requested);
        const std::size_t peak = 1000; // small; pulse tail = peak + max_shift
        Intensities intensity = make_dispersed_frb_intensity(dims, cfg, injected_dm, peak, 64.0F, 8);
        auto cands = run_classifier_for_intensity(intensity, dims, cfg);
        const Candidate* best = max_snr_candidate(cands);
        if (!best || best->label != CandidateLabel::AstrophyicalFRB) {
            std::fprintf(stderr, "FAIL (b): requested_dm=%.1f no Astrophysical candidate (label=%d)\n",
                         requested, best ? static_cast<int>(best->label) : -1);
            all_ok = false;
            continue;
        }
        const float err = std::abs(best->dm - injected_dm);
        const bool ok = err <= (1.5F * dm_step) && best->snr >= cfg.snr_threshold;
        std::printf("  requested=%.1f injected=%.3f recovered dm=%.3f err=%.3f snr=%.3f label=%d %s\n",
                    requested, injected_dm, best->dm, err, best->snr,
                    static_cast<int>(best->label), ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }
    std::printf("  -> %s (b)\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ---------------------------------------------------------------------------
// (c) boxcar-width recovery
// ---------------------------------------------------------------------------
bool test_c_width_recovery() {
    std::printf("=== (c) boxcar-width recovery ===\n");
    const std::size_t n_time = 16384;
    const Dimensions dims = make_test_dims(n_time);
    FRBClassifierConfig cfg;
    cfg.n_dm = 1024;
    cfg.dm_min = 0.0F;
    cfg.dm_max = 1500.0F;
    cfg.snr_threshold = 6.0F;
    const float injected_dm = snap_dm_to_grid(cfg, 250.0F);
    const std::size_t peak = 2000;
    const int widths[] = {4, 16, 64, 256};

    // Find a width-index tolerance: recovered width_idx within ±1 of injected.
    bool all_ok = true;
    for (int w : widths) {
        // Scale amplitude so wider pulses stay above threshold; we want
        // comparable normalized SNRs (the kernel normalizes by sqrt(W)).
        const float amplitude = 64.0F;
        Intensities intensity = make_dispersed_frb_intensity(dims, cfg, injected_dm, peak, amplitude, w);
        auto cands = run_classifier_for_intensity(intensity, dims, cfg);
        const Candidate* best = max_snr_candidate(cands);
        if (!best) {
            std::fprintf(stderr, "FAIL (c): width=%d no candidate\n", w);
            all_ok = false;
            continue;
        }
        // Map injected width to a width_idx on the boxcar list.
        int target_idx = 0;
        for (std::size_t i = 0; i < cfg.boxcar_widths.size(); ++i) {
            if (cfg.boxcar_widths[i] == w) { target_idx = static_cast<int>(i); break; }
        }
        const int width_err = best->width_idx - target_idx;
        const bool ok = (std::abs(width_err) <= 1) && (best->snr >= cfg.snr_threshold);
        std::printf("  injected width=%d (idx=%d) | recovered idx=%d (width=%d) snr=%.3f label=%d %s\n",
                    w, target_idx, best->width_idx, best->width_samples, best->snr,
                    static_cast<int>(best->label), ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }
    std::printf("  -> %s (c)\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ---------------------------------------------------------------------------
// (d) RFI rejection: zero-DM undispersed narrow-band constant pulse
// ---------------------------------------------------------------------------
bool test_d_rfi_rejection() {
    std::printf("=== (d) RFI rejection ===\n");
    const std::size_t n_time = 8192;
    const Dimensions dims = make_test_dims(n_time);
    FRBClassifierConfig cfg;
    cfg.n_dm = 1024;
    cfg.dm_min = 0.0F;
    cfg.dm_max = 1500.0F;
    cfg.snr_threshold = 6.0F;

    // Build an UNDISPERSED (dm=0) constant-band pulse: same amplitude in every
    // frequency channel at the same time sample. This aligns perfectly at
    // DM=0 (the dm_min grid trial), so the kernel will produce a strong
    // candidate whose dm is within one DM-step of dm_min -> R1 fires.
    std::vector<float> intensity(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    const std::size_t pulse_t = 1000;
    const float amplitude = 64.0F;
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        for (std::size_t b = 0; b < dims.n_beams; ++b) {
            std::size_t idx = (pulse_t * dims.n_freq + f) * dims.n_beams + b;
            intensity[idx] += amplitude;
        }
    }
    auto cands = run_classifier_for_intensity(intensity, dims, cfg);

    // Assert: every surviving candidate is labeled RFI; NONE labeled AstrophyicalFRB.
    bool has_rfi = false;
    bool has_astro = false;
    for (const auto& c : cands) {
        if (c.label == CandidateLabel::RFI) has_rfi = true;
        if (c.label == CandidateLabel::AstrophyicalFRB) has_astro = true;
        std::printf("  cand dm=%.3f t=%zu snr=%.3f label=%d\n", c.dm, c.time_index, c.snr,
                    static_cast<int>(c.label));
    }
    const bool ok = has_rfi && !has_astro;
    std::printf("  has_rfi=%d has_astro=%d\n", has_rfi, has_astro);
    std::printf("  -> %s (d)\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("No CUDA device available; skipping Phase 1 FRB classifier tests.\n");
        return 0; // gated; CI without a GPU reports success (build is the gate).
    }

    int failures = 0;
    if (!test_a_end_to_end()) ++failures;
    if (!test_b_known_dm()) ++failures;
    if (!test_c_width_recovery()) ++failures;
    if (!test_d_rfi_rejection()) ++failures;

    if (failures == 0) {
        std::printf("\nALL PHASE 1 FRB CLASSIFIER SUB-TESTS (a)-(d) PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d PHASE 1 FRB CLASSIFIER SUB-TEST(S) FAILED.\n", failures);
    return 1;
}
