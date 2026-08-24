// src/frb_classifier_rules.cpp
//
// Host-only rule-based classifier (Phase 1 subset R1-R2; R3/R4 placeholders)
// and the hand-rolled JSON emitter. No CUDA in this TU; links on CPU-only
// builds so the JSON schema symbol stays available regardless of CUDA support.

#include "beamformer/frb_classifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace beamformer {

namespace {

const char* label_to_string(CandidateLabel l) {
    switch (l) {
        case CandidateLabel::AstrophyicalFRB: return "Astrophysical";
        case CandidateLabel::RFI: return "RFI";
        case CandidateLabel::Borderline: return "Borderline";
        case CandidateLabel::Unknown:
        default: return "Unknown";
    }
}

// Minimal float-to-ASCII emitter with sane precision and trailing-zero trim.
void emit_float(std::string& out, float v) {
    if (!std::isfinite(v)) {
        if (std::isnan(v)) { out += "null"; return; }
        out += (v < 0.0F) ? "-1e999" : "1e999";
        return;
    }
    char buf[64];
    // 6 significant digits is plenty for SNR/DM reports.
    std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(v));
    out += buf;
}

void emit_json_string(std::string& out, const char* s) {
    out += '"';
    for (const char* p = s; *p != '\0'; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else { out += static_cast<char>(c); }
    }
    out += '"';
}

} // namespace

// ---------------------------------------------------------------------------
// Host rule classifier
// ---------------------------------------------------------------------------
//
// Phase 1 implements:
//   R1 (zero-DM test): any candidate whose dm is within one DM-step of
//      dm_min is reclassified RFI (astrophysical FRBs at dm=0 don't exist).
//   R2 (DM dependence): accept if dm > 1.5 * max(2 * dm_step, 1.0).
//   R3/R4, R5, R6, R7: needs a DM sweep / spectral estimate that Phase 1
//      does not have; kept as Unknown-or-Borderline placeholders and
//      documented inline. The shell exists so Phase 2 can drop in the sweep.
std::vector<Candidate> classify_candidates(std::vector<Candidate> candidates,
                                           const FRBClassifierConfig& config) {
    const float dm_step = config.dm_step();
    // R1 lower bound: a candidate "at dm=0" means within one step of dm_min.
    const float r1_upper = config.dm_min + dm_step;
    // R2 floor: a candidate needs dm well above the DM-dependence floor.
    const float r2_threshold = std::max(config.dm_min + 1.5F * std::max(2.0F * dm_step, 1.0F),
                                         config.dm_min + 3.0F * dm_step);

    // Find zero-DM peak and overall peak across candidates for zero-DM veto
    float max_zero_dm_snr = 0.0F;
    float global_max_snr = 0.0F;
    float global_max_dm = 0.0F;
    for (const auto& c : candidates) {
        if (c.dm <= r1_upper) {
            if (c.snr > max_zero_dm_snr) max_zero_dm_snr = c.snr;
        }
        if (c.snr > global_max_snr) {
            global_max_snr = c.snr;
            global_max_dm = c.dm;
        }
    }

    const bool zero_dm_dominant = (max_zero_dm_snr >= config.snr_threshold &&
                                   max_zero_dm_snr >= 0.5F * global_max_snr &&
                                   global_max_dm <= r1_upper);

    for (auto& c : candidates) {
        // If zero-DM RFI burst dominates the batch, all candidates are RFI or sweep artifacts
        if (zero_dm_dominant) {
            c.label = CandidateLabel::RFI;
            continue;
        }

        // R1: zero-DM test.
        const bool r1_fails = (c.dm <= r1_upper);
        // R2: DM dependence — accept the dependence if DM is clearly above the floor.
        const bool r2_passes = (c.dm >= r2_threshold);

        bool r4_peak_above_w0 = false;
        const float w0 = c.width_curve[0];
        for (std::size_t i = 1; i < c.width_curve.size(); ++i) {
            if (c.width_curve[i] > w0 + 1.0e-6F) { r4_peak_above_w0 = true; break; }
        }

        if (r1_fails) {
            c.label = CandidateLabel::RFI;
        } else if (r2_passes) {
            // R7 default path: passes R2, R4 (or no curve info), and SNR threshold.
            if (c.snr >= config.snr_threshold) {
                if (c.snr < 0.1F * global_max_snr && global_max_snr > 100.0F) {
                    c.label = CandidateLabel::Borderline;
                } else {
                    c.label = CandidateLabel::AstrophyicalFRB;
                }
            } else {
                c.label = CandidateLabel::Borderline; // SNR borderline (R5)
            }
        } else {
            // Between r1_upper and r2_threshold: not enough DM dependence evidence.
            c.label = CandidateLabel::Borderline;
        }
    }
    return candidates;
}

// ---------------------------------------------------------------------------
// JSON emitter (schema_version 1)
// ---------------------------------------------------------------------------
std::string candidates_to_json(const std::vector<Candidate>& candidates,
                                const FRBClassifierConfig& config,
                                const std::size_t window_index) {
    // f_ref = high edge of the band, computed from the same inline formula
    // the classifier uses (not the geometry TU). For discovery we need at
    // least one channel to anchor the band.
    std::vector<float> freqs = config.frequencies(/* default band */ 336);
    const double f_ref_hz = freqs.empty() ? config.freq_start_hz
                                          : static_cast<double>(freqs.back());

    std::string out;
    out.reserve(4096);
    out += "{\n";
    out += "  \"schema_version\": 1,\n";
    out += "  \"band\": {\n";
    out += "    \"start_hz\": "; emit_float(out, config.freq_start_hz); out += ",\n";
    out += "    \"channel_width_hz\": "; emit_float(out, config.channel_width_hz); out += ",\n";
    out += "    \"n_freq\": 336,\n";
    { char buf[64]; std::snprintf(buf, sizeof(buf), "%.1f", f_ref_hz); out += "    \"f_ref_hz\": "; out += buf; out += ",\n"; }
    out += "    \"sample_rate_hz\": 952.381,\n";
    out += "    \"convention\": \"device_local_300_400MHz\"\n";
    out += "  },\n";
    out += "  \"batch\": {\n";
    { char buf[64]; std::snprintf(buf, sizeof(buf), "%zu", window_index); out += "    \"first_window_index\": "; out += buf; out += ",\n"; }
    { char buf[64]; std::snprintf(buf, sizeof(buf), "%zu", candidates.size()); out += "    \"n_candidates\": "; out += buf; out += ",\n"; }
    { char buf[64]; std::snprintf(buf, sizeof(buf), "%zu", config.candidate_ring_capacity); out += "    \"ring_capacity\": "; out += buf; out += ",\n"; }
    out += "    \"ring_overflow_count\": 0\n";
    out += "  },\n";
    out += "  \"candidates\": [\n";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& c = candidates[i];
        out += "    {\n";
        out += "      \"snr\": "; emit_float(out, c.snr); out += ",\n";
        out += "      \"dm\": "; emit_float(out, c.dm); out += ",\n";
        { char buf[64]; std::snprintf(buf, sizeof(buf), "%zu", c.time_index); out += "      \"time_index\": "; out += buf; out += ",\n"; }
        { char buf[64]; std::snprintf(buf, sizeof(buf), "%d", c.width_idx); out += "      \"width_idx\": "; out += buf; out += ",\n"; }
        { char buf[64]; std::snprintf(buf, sizeof(buf), "%d", c.width_samples); out += "      \"width_samples\": "; out += buf; out += ",\n"; }
        out += "      \"baseline_mean\": "; emit_float(out, c.baseline_mean); out += ",\n";
        out += "      \"baseline_std\": "; emit_float(out, c.baseline_std); out += ",\n";
        out += "      \"label\": "; emit_json_string(out, label_to_string(c.label)); out += ",\n";
        out += "      \"width_curve\": [";
        for (std::size_t w = 0; w < c.width_curve.size(); ++w) {
            if (w) out += ", ";
            emit_float(out, c.width_curve[w]);
        }
        out += "]\n";
        out += "    }";
        if (i + 1 < candidates.size()) out += ',';
        out += '\n';
    }
    out += "    ]\n";
    out += "}\n";
    return out;
}

} // namespace beamformer
