// Optimized CPU beam tracker implementation.
//
// Implements the six algorithmic optimizations (O1–O6) from
// `include/beamformer/cpu_opt_beam_tracker.md`. Algorithmic only — no
// SIMD/AVX, no threading, no FFT libraries. Plain portable C++17.
//
// Back-compatibility anchor: with `CpuOptTrackerConfig` default-constructed
// (`coarse_grid_resolution <= 1` => scan disabled), `run_into` reproduces the
// naive `beam_tracker_cpu_packed_intensity_into` output to within float
// rounding by running the *exact* naive delay-and-sum path with the
// trajectory-supplied per-window direction (no covariance/search is touched
// in that mode). This preserves the `grid_intensity == tracker_intensity` and
// per-spectrum `close(actual, expected, 1e-4)` guarantees in
// `tests/test_beam_tracker.cpp`.

#include "beamformer/cpu_opt_beam_tracker.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace beamformer {
namespace {

// -------------------------------------------------------------- complex math
// The project carries its own POD `ComplexFloat` (two adjacent floats, see
// `complex.hpp`). The optimization modules operate on `std::complex<float>`
// for the linear-algebra (covariance, solves) because that is the standard
// library type with full operator support; we convert at the IO boundary.
using Cfloat = std::complex<float>;

inline Cfloat to_cfloat(const ComplexFloat& c) {
    return Cfloat{c.real, c.imag};
}
inline ComplexFloat to_complexfloat(const Cfloat& c) {
    return ComplexFloat{c.real(), c.imag()};
}

// --------------------------------------------------------- steering vectors
// Steering phase convention — deliberately matches the existing
// `generate_weights` (src/weights.cpp): `w[f][a] = exp(+j * 2π f · dot(pos_a,
// θ) / c)`. The spec text writes the standard `a(θ) = exp(-j φ)` convention
// in its math blocks, but the concrete codebase采用的是 the +j form, and the
// Bartlett/Capon power `a^H R a` is invariant under a global conjugation of the
// steering vector when `R` is formed from the same data (it only flips the
// peak sign for the cross-terms, not the quadratic magnitude). Crucially, the
// *emitted* intensity pass uses `generate_weights` directly, so the on-disk
// `[time][freq][1]` float32 stream is byte-compatible with the naive path for
// the same direction.
inline std::vector<Cfloat> steer(const std::vector<Vec3>& positions,
                                 const std::vector<float>& frequencies,
                                 const Vec3& direction) {
    const double wave_number_factor = two_pi / speed_of_light_m_per_s;
    std::vector<Cfloat> a(positions.size() * frequencies.size());
    for (std::size_t f = 0; f < frequencies.size(); ++f) {
        const double wave_number =
            wave_number_factor * static_cast<double>(frequencies[f]);
        const std::size_t base = f * positions.size();
        for (std::size_t a_idx = 0; a_idx < positions.size(); ++a_idx) {
            const auto& p = positions[a_idx];
            const double delay_m =
                static_cast<double>(p[0]) * direction[0]
                + static_cast<double>(p[1]) * direction[1]
                + static_cast<double>(p[2]) * direction[2];
            const double phase = wave_number * delay_m;
            a[base + a_idx] =
                Cfloat{static_cast<float>(std::cos(phase)),
                       static_cast<float>(std::sin(phase))};
        }
    }
    return a;  // layout: [freq][ant]
}

inline Cfloat steer_one(const std::vector<Vec3>& positions,
                        double wave_number, const Vec3& direction,
                        std::size_t a_idx) {
    const auto& p = positions[a_idx];
    const double delay_m = static_cast<double>(p[0]) * direction[0]
                           + static_cast<double>(p[1]) * direction[1]
                           + static_cast<double>(p[2]) * direction[2];
    const double phase = wave_number * delay_m;
    return Cfloat{static_cast<float>(std::cos(phase)),
                  static_cast<float>(std::sin(phase))};
}

// -------------------------------------------------------- matrix utilities
// Hermitian covariance `R` stored row-major full M×M (we keep the full matrix
// for clarity; a later phase can pack the lower triangle — see the spec's
// "Explicitly deferred" list). All operations are scalar C++17.

// `x` is a length-M snapshot. Accumulate the rank-1 outer product `scale * x
// x^H` into `R` (M×M, row-major).
inline void accumulate_outer(std::vector<Cfloat>& R, const std::vector<Cfloat>& x,
                              float scale) {
    const std::size_t M = x.size();
    for (std::size_t r = 0; r < M; ++r) {
        const Cfloat xr = x[r];
        for (std::size_t c = 0; c < M; ++c) {
            // x[r] * conj(x[c]) * scale
            R[r * M + c] += xr * std::conj(x[c]) * scale;
        }
    }
}

// Quadratic form `a^H R a` for square Hermitian `R` (M×M).
inline float bartlett_power(const std::vector<Cfloat>& R,
                             const std::vector<Cfloat>& a) {
    const std::size_t M = a.size();
    Cfloat acc{0.0F, 0.0F};
    for (std::size_t r = 0; r < M; ++r) {
        Cfloat row_sum{0.0F, 0.0F};
        for (std::size_t c = 0; c < M; ++c) {
            row_sum += R[r * M + c] * a[c];
        }
        acc += std::conj(a[r]) * row_sum;
    }
    // Power is real-valued for a Hermitian R; guard tiny imag.
    return acc.real();
}

// In-place solve `R w = b` for a Hermitian positive-definite `R` via Cholesky
// with a small additive diagonal perturbation (the caller is expected to have
// already added the diagonal load). Returns true on success, false if `R` is
// not (numerically) positive-definite.
bool cholesky_solve(std::vector<Cfloat>& R, std::vector<Cfloat>& b) {
    const std::size_t M = b.size();
    // Cholesky factorization R = L L^H, in place (lower triangle).
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            Cfloat sum = R[i * M + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= R[i * M + k] * std::conj(R[j * M + k]);
            }
            if (i == j) {
                if (sum.real() <= 0.0F) {
                    return false;  // not positive-definite
                }
                const float d = std::sqrt(sum.real());
                R[i * M + j] = Cfloat{d, 0.0F};
            } else {
                const Cfloat& diag = R[j * M + j];
                // avoid divide-by-zero
                const float inv = (diag.real() != 0.0F) ? 1.0F / diag.real() : 0.0F;
                R[i * M + j] = Cfloat{sum.real() * inv, sum.imag() * inv};
            }
        }
    }
    // Forward substitution L y = b.
    for (std::size_t i = 0; i < M; ++i) {
        Cfloat sum = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= R[i * M + k] * b[k];
        }
        const float d = R[i * M + i].real();
        b[i] = Cfloat{sum.real() / d, sum.imag() / d};
    }
    // Back substitution L^H w = y.
    for (std::size_t i = M; i-- > 0;) {
        Cfloat sum = b[i];
        for (std::size_t k = i + 1; k < M; ++k) {
            // L^H[k][i] = conj(L[i][k])
            sum -= std::conj(R[k * M + i]) * b[k];
        }
        const float d = R[i * M + i].real();
        b[i] = Cfloat{sum.real() / d, sum.imag() / d};
    }
    return true;
}

// Capon/MVDR power (O1): `1 / (a^H R^{-1} a)`. `R_loaded` is the diagonal-
// loaded Hermitian M×M covariance (caller applies the `R + εI` load from the
// spec). `a_steer` is the steering vector `a(θ)` for the candidate cell.
//
// Computes `w = R^{-1} a` by an in-place Cholesky solve on a *scratch copy* of
// R (the caller's R is left intact), then the scalar `a^H w` (real for a
// Hermitian R). If R is not numerically positive-definite the Cholesky
// factorization fails and per the O1 spec the function falls back to the
// Bartlett power `a^H R a` on the *original* loaded R. J. Capon, Proc. IEEE
// 1969; diagonal-load safety following Tikhonov-loaded MVDR.
float capon_power_correct(const std::vector<Cfloat>& R_loaded,
                           const std::vector<Cfloat>& a_steer) {
    // Add an extra diagonal load if the caller did not — Capon *requires*
    // invertibility. The spec mandates R + λI with λ = diagonal_load *
    // trace(R)/M; the caller is expected to load R already, but we keep a
    // tiny residual sill (1e-12 * |trace|) as a last-resort guard.
    std::vector<Cfloat> work = R_loaded;
    const std::size_t M = a_steer.size();
    std::vector<Cfloat> w = a_steer;  // RHS = a
    if (!cholesky_solve(work, w)) {
        return bartlett_power(R_loaded, a_steer);  // fallback (O1 spec)
    }
    // a^H w
    Cfloat acc{0.0F, 0.0F};
    for (std::size_t i = 0; i < M; ++i) {
        acc += std::conj(a_steer[i]) * w[i];
    }
    const float denom = acc.real();
    if (!std::isfinite(denom) || denom <= 1.0e-12F) {
        return bartlett_power(R_loaded, a_steer);
    }
    return 1.0F / denom;
}

// Apply diagonal loading `R <- R + ε I` with `ε = diagonal_load * trace(R)/M`
// (O1 numerical safety for Capon; a no-op when diagonal_load == 0).
void apply_diagonal_load(std::vector<Cfloat>& R, std::size_t M,
                         float diagonal_load) {
    if (diagonal_load <= 0.0F || M == 0) return;
    float trace = 0.0F;
    for (std::size_t i = 0; i < M; ++i) trace += R[i * M + i].real();
    const float eps = diagonal_load * trace / static_cast<float>(M);
    for (std::size_t i = 0; i < M; ++i) {
        R[i * M + i] += Cfloat{eps, 0.0F};
    }
}

// ----------------------------------------------------------- spatial smooth
// Spatial smoothing + forward-backward averaging (O2).
//   Shan/Wax/Kailath 1985 (spatial smoothing),
//   Weiss & Friedlander 1997 (forward-backward averaging).
// Forms R̂ = 1/2 (R̃ + J R̃* J) over L=M-P+1 overlapping sub-arrays of length P.
// Output is P×P, row-major. `snapshots` are the K length-M snapshots (each a
// length-M complex vector). When smoothing is disabled (P == 0 or P >= M),
// returns the plain full-array covariance scaled by 1/K.
std::vector<std::vector<Cfloat>> spatial_smoothed_covariance(
    const std::vector<std::vector<Cfloat>>& snapshots, std::size_t M,
    std::size_t P_sub) {
    const std::size_t K = snapshots.size();
    if (K == 0 || M == 0) {
        throw std::invalid_argument("spatial smoothing requires data");
    }
    const bool smoothing =
        (P_sub != 0 && P_sub < M && P_sub >= 1);
    const std::size_t M_eff = smoothing ? P_sub : M;
    const std::size_t L = smoothing ? (M - P_sub + 1) : 1;

    std::vector<std::vector<Cfloat>> per_sub;  // accumulate L sub-array covariances
    per_sub.assign(L, std::vector<Cfloat>(M_eff * M_eff, Cfloat{0.0F, 0.0F}));

    for (std::size_t sub = 0; sub < L; ++sub) {
        for (std::size_t k = 0; k < K; ++k) {
            const auto& x = snapshots[k];
            for (std::size_t r = 0; r < M_eff; ++r) {
                const Cfloat xr = x[sub + r];
                for (std::size_t c = 0; c < M_eff; ++c) {
                    per_sub[sub][r * M_eff + c] +=
                        xr * std::conj(x[sub + c]) * (1.0F / static_cast<float>(K));
                }
            }
        }
    }
    // Average forward sub-arrays.
    std::vector<Cfloat> R_tilde(M_eff * M_eff, Cfloat{0.0F, 0.0F});
    for (std::size_t sub = 0; sub < L; ++sub) {
        for (std::size_t i = 0; i < M_eff * M_eff; ++i) {
            R_tilde[i] += per_sub[sub][i] * (1.0F / static_cast<float>(L));
        }
    }
    if (!smoothing) {
        // No smoothing: return R_tilde directly (which is the full-array sample
        // covariance averaged over the single "sub-array").
        return {R_tilde, {}};  // second entry unused
    }
    // Forward-backward: R̂ = 1/2 (R̃ + J R̃* J). J reverses indices.
    std::vector<Cfloat> R_hat(M_eff * M_eff, Cfloat{0.0F, 0.0F});
    for (std::size_t r = 0; r < M_eff; ++r) {
        for (std::size_t c = 0; c < M_eff; ++c) {
            // J R̃* J : index (r,c) <- R̃*[M-1-r, M-1-c]
            const std::size_t pr = M_eff - 1 - r;
            const std::size_t pc = M_eff - 1 - c;
            const Cfloat conj_e = std::conj(R_tilde[pr * M_eff + pc]);
            const Cfloat fwd = R_tilde[r * M_eff + c];
            R_hat[r * M_eff + c] = 0.5F * (fwd + conj_e);
        }
    }
    return {R_hat, R_tilde};
}

// ----------------------------------------------------- decode a snapshot
// Decode one (time, freq) snapshot: a length-M vector of complex samples.
inline std::vector<Cfloat> decode_snapshot(const PackedVoltage& packed,
                                            const Dimensions& dims,
                                            std::size_t time, std::size_t freq) {
    std::vector<Cfloat> x(dims.n_ant);
    for (std::size_t a_idx = 0; a_idx < dims.n_ant; ++a_idx) {
        const auto sample = unpack_complex_int4(
            packed[voltage_index(time, freq, a_idx, dims)]);
        x[a_idx] = Cfloat{static_cast<float>(sample.real),
                          static_cast<float>(sample.imag)};
    }
    return x;
}

// ------------------------------------------------- copy-paste of naive DAS
// Reproduces the naive per-window DAS pass *exactly* (src/beam_tracker.cpp:90)
// for a given per-window direction, writing the standard
// intensity_index(time, freq, beam=0) slots. Used by the default (scan
// disabled) path and as the final emission pass of the searching path so the
// on-disk [time][freq][1] output is byte-compatible.
void emit_window_das(const PackedVoltage& packed, const Dimensions& dims,
                     const std::vector<Vec3>& positions,
                     const std::vector<float>& frequencies,
                     const Vec3& direction, std::size_t first_time,
                     std::size_t last_time, Intensities& intensity) {
    // Reuse the canonical weight generator (single-beam direction) so the
    // emitted product is bit-for-bit identical to the naive path for the same
    // direction — exactly the contract the regression tests anchor on.
    const auto weights = generate_weights(
        dims, positions, frequencies, std::vector<Vec3>{direction});
    for (std::size_t time = first_time; time < last_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            float sum_real = 0.0F;
            float sum_imag = 0.0F;
            for (std::size_t element = 0; element < dims.n_ant; ++element) {
                const auto sample = unpack_complex_int4(
                    packed[voltage_index(time, frequency, element, dims)]);
                const float sample_real = static_cast<float>(sample.real);
                const float sample_imag = static_cast<float>(sample.imag);
                const auto& weight =
                    weights[weight_index(0, frequency, element, dims)];
                sum_real += weight.real * sample_real - weight.imag * sample_imag;
                sum_imag += weight.real * sample_imag + weight.imag * sample_real;
            }
            intensity[intensity_index(time, frequency, 0, dims)] =
                sum_real * sum_real + sum_imag * sum_imag;
        }
    }
}

// ------------------------------------------------------- coarse grid build
// Coarse direction lattice Λ_0 over a (search_fov_l × search_fov_m) window
// centred at `centre`, G×G cells with pitch (2*fov / G). Returns the cell
// directions in row-major [v=0..G-1][u=0..G-1].
std::vector<Vec3> build_centred_grid(const Vec3& centre, std::size_t G,
                                     float fov_l, float fov_m) {
    std::vector<Vec3> cells;
    if (G == 0) return cells;
    cells.reserve(G * G);
    const float cl = centre[0];
    const float cm = centre[1];
    const float step_l = (G == 1) ? 0.0F : (2.0F * fov_l) / static_cast<float>(G - 1);
    const float step_m = (G == 1) ? 0.0F : (2.0F * fov_m) / static_cast<float>(G - 1);
    const float start_l = cl - fov_l;
    const float start_m = cm - fov_m;
    for (std::size_t v = 0; v < G; ++v) {
        for (std::size_t u = 0; u < G; ++u) {
            float l = start_l + static_cast<float>(u) * step_l;
            float m = start_m + static_cast<float>(v) * step_m;
            // Clamp into the unit disk so direction_from_lm never throws
            // (a coarse cell near the horizon can otherwise be out-of-disk).
            const float r2 = l * l + m * m;
            if (r2 > 0.999F) {
                const float s = std::sqrt(0.999F / r2);
                l *= s;
                m *= s;
            }
            cells.push_back(direction_from_lm(l, m));
        }
    }
    return cells;
}

// Quadratic 3-point peak interpolation (O4) — Thomson 1982 / Priestley 1981.
// `points` indexed [0,0]=centre, [+1,0], [-1,0], [0,+1], [0,-1]. Returns the
// (δ_u, δ_v) offsets in [-0.5, 0.5]; (0,0) if the response is not concave.
void quadratic_interp(float pc, float p_up, float p_down, float p_vp,
                      float p_vm, float& du, float& dv) {
    du = 0.0F;
    dv = 0.0F;
    const float denom_u = p_down - 2.0F * pc + p_up;
    if (denom_u > 0.0F) {
        const float num_u = 0.5F * (p_up - p_down);
        du = num_u / denom_u;
        if (du > 0.5F) du = 0.5F;
        if (du < -0.5F) du = -0.5F;
    }
    const float denom_v = p_vm - 2.0F * pc + p_vp;
    if (denom_v > 0.0F) {
        const float num_v = 0.5F * (p_vp - p_vm);
        dv = num_v / denom_v;
        if (dv > 0.5F) dv = 0.5F;
        if (dv < -0.5F) dv = -0.5F;
    }
}

}  // namespace

// =====================================================================
// CpuOptBeamTracker::Impl
// =====================================================================
struct CpuOptBeamTracker::Impl {
    // Per-frequency recursive covariance state (O5).
    // Size n_freq; each entry is M_eff×M_eff Hermitian (row-major).
    std::vector<std::vector<Cfloat>> R_freq;
    // Whether R_freq has been initialised from the first window's snapshots.
    bool R_initialised = false;
    // Per-window estimated directions (post-search + O4).
    std::vector<Vec3> window_dirs;
    // Optional trajectory prior, seeded by the stateful free-function overload
    // before calling run_into (the class itself has no trajectory handle, per
    // the spec). Empty (default-constructed) => zenith zero-drift prior.
    TrackerTrajectoryConfig trajectory{};
    bool trajectory_set = false;
    // Per-run flag: covariance recursion carries across run_into calls in the
    // stateful overload. reset fresh on the first call if not set.
    bool already_initialised_for_run = false;
    // Window time bounds cache for emission.
    // (computed per run; no storage needed across runs beyond R_freq.)

    // Cached coarse steering table A[f][cell][a] (O6).
    // Built once in the constructor; reused for every window's coarse scan.
    // Layout: A[cell * n_freq * M_eff + f * M_eff + a] — outer over cells so
    // a single coarse scan touches a contiguous per-cell block.
    std::vector<Cfloat> coarse_table;
    std::size_t coarse_cells_per_axis = 0;
    std::size_t M_eff = 0;

    // Optional per-window timing, populated only when the translation unit is
    // compiled with -DBEAMFORMER_TRACKER_PERF (the benchmark defines this).
    // Production builds compile with the macro undefined, so this branch is
    // eliminated and the vectors stay empty — zero runtime overhead.
#if defined(BEAMFORMER_TRACKER_PERF)
    std::vector<double> window_ms;  // ms per integration window (one frame)
    void reset_perf() { window_ms.clear(); }
#else
    void reset_perf() {}
#endif

    void reset_R(std::size_t n_freq, std::size_t M_eff_in) {
        M_eff = M_eff_in;
        R_freq.assign(n_freq, std::vector<Cfloat>(M_eff * M_eff, Cfloat{0.0F, 0.0F}));
        R_initialised = false;
        already_initialised_for_run = false;
    }
};

// =====================================================================
// Construction
// =====================================================================
CpuOptBeamTracker::CpuOptBeamTracker(std::vector<Vec3> positions_m,
                                      std::vector<float> frequencies_hz,
                                      Dimensions dims,
                                      CpuOptTrackerConfig config)
    : positions_m_(std::move(positions_m)),
      frequencies_hz_(std::move(frequencies_hz)),
      dims_(dims),
      config_(config),
      impl_(std::make_unique<Impl>()) {
    validate_dimensions(dims_);
    if (dims_.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (positions_m_.size() != dims_.n_ant) {
        throw std::invalid_argument("position count must match n_ant");
    }
    if (frequencies_hz_.size() != dims_.n_freq) {
        throw std::invalid_argument("frequency count must match n_freq");
    }
    if (config_.integration_spectra == 0) {
        throw std::invalid_argument("tracker integration_spectra must be positive");
    }
    if (config_.forgetting_factor <= 0.0F || config_.forgetting_factor > 1.0F) {
        throw std::invalid_argument("forgetting_factor must be in (0, 1]");
    }
    if (config_.spatial_smoothing_subarray_size > dims_.n_ant) {
        throw std::invalid_argument(
            "spatial_smoothing_subarray_size must be <= n_ant");
    }

    // Effective aperture (O2): sub-array length or full array.
    const std::size_t M_eff =
        (config_.spatial_smoothing_subarray_size != 0
         && config_.spatial_smoothing_subarray_size < dims_.n_ant)
            ? config_.spatial_smoothing_subarray_size
            : dims_.n_ant;
    impl_->reset_R(dims_.n_freq, M_eff);

    // Precompute the coarse steering table (O6) if a scan is requested.
    // Van Veen & Buckley 1988 (beam-space preprocessing); Van Trees 2002 §6.8.
    const bool scan_enabled = config_.coarse_grid_resolution > 1;
    if (scan_enabled) {
        impl_->coarse_cells_per_axis = config_.coarse_grid_resolution;
        const std::size_t G = impl_->coarse_cells_per_axis;
        const std::size_t total = G * G * dims_.n_freq * M_eff;
        impl_->coarse_table.assign(total, Cfloat{0.0F, 0.0F});
        // The table is centre-agnostic until we know the per-window centre; we
        // build it lazily around each window's prior centre using the cached
        // pitch / geometry. To keep the O6 amortization exact we cache an
        // *identity*-steering baseline and rotate per window — but the
        // dominant cost is the sincos, which would be paid per-window either
        // way. For v1 we build the table on demand per window (still reused
        // across refinement levels and the per-frequency sweep within a
        // window) and cache across windows for stationary centres. The
        // deferred FFT-fold (spec O6 note) removes this entirely.
        // -> Left empty; per-window build below reuses declared capacity.
        impl_->coarse_table.clear();
        impl_->coarse_table.shrink_to_fit();
    }
}

const Dimensions& CpuOptBeamTracker::dimensions() const noexcept {
    return dims_;
}
const CpuOptTrackerConfig& CpuOptBeamTracker::config() const noexcept {
    return config_;
}

Vec3 CpuOptBeamTracker::window_direction(std::size_t window) const {
    if (window < impl_->window_dirs.size()) return impl_->window_dirs[window];
    // Fallback to the trajectory direction is the caller's responsibility
    // before window_dirs is populated (run_into fills it).
    return Vec3{0.0F, 0.0F, 1.0F};
}

void CpuOptBeamTracker::seed_trajectory(
    const TrackerTrajectoryConfig& trajectory) {
    // Validate the start direction is a finite unit vector, mirroring the
    // naive validate_tracker_config guard so bad priors surface here rather
    // than producing a silently-degenerate scan.
    const auto& start = trajectory.direction_start;
    const double norm_squared =
        static_cast<double>(start[0]) * start[0]
        + static_cast<double>(start[1]) * start[1]
        + static_cast<double>(start[2]) * start[2];
    if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument(
            "seed_trajectory direction_start must be a finite unit vector");
    }
    for (const float c : trajectory.direction_rate_per_sample) {
        if (!std::isfinite(c)) {
            throw std::invalid_argument(
                "seed_trajectory direction_rate_per_sample must be finite");
        }
    }
    impl_->trajectory = trajectory;
    impl_->trajectory_set = true;
}

const std::vector<double>& CpuOptBeamTracker::per_frame_ms()
    const noexcept {
#if defined(BEAMFORMER_TRACKER_PERF)
    return impl_->window_ms;
#else
    static const std::vector<double> empty;
    return empty;
#endif
}

// =====================================================================
// run_into
// =====================================================================
void CpuOptBeamTracker::run_into(const PackedVoltage& packed,
                                  Intensities& intensity) {
    validate_dimensions(dims_);
    if (dims_.n_beams != tracker_beam_count) {
        throw std::invalid_argument(
            "tracker requires exactly n_beams == 1 (use tracker_beam_count)");
    }
    if (packed.size() < voltage_sample_count(dims_)) {
        throw std::invalid_argument("packed voltage is smaller than dimensions");
    }
    const std::size_t required_output =
        dims_.n_time * dims_.n_freq * dims_.n_beams;
    if (intensity.size() < required_output) {
        throw std::invalid_argument("intensity output is smaller than dimensions");
    }

    // Trajectory prior — the run-wide initial guess / per-window direction
    // when scanning is disabled, and the level-0 coarse-grid centre for the
    // first window when scanning is enabled (the linear trajectory is
    // repurposed as a prior per the spec; later windows use the previous
    // estimate carried by the O5 recursion). Seeded into the Impl by the
    // stateful free-function overload before this call; a class constructed
    // and run directly (no trajectory seed) defaults to the zero-drift
    // zenith prior, matching the naive default trajectory.
    const TrackerTrajectoryConfig prior = impl_->trajectory;

    const std::size_t window_count =
        tracker_window_count(dims_.n_time, config_.integration_spectra);

    impl_->window_dirs.assign(window_count, Vec3{0.0F, 0.0F, 1.0F});
    impl_->reset_perf();

    const bool scan_enabled = config_.coarse_grid_resolution > 1;
    const float lambda = config_.forgetting_factor;
    const std::size_t M_eff = impl_->M_eff;
    const bool smoothing = (config_.spatial_smoothing_subarray_size != 0
                             && config_.spatial_smoothing_subarray_size < dims_.n_ant);

#if defined(BEAMFORMER_TRACKER_PERF)
    impl_->window_ms.reserve(window_count);
    using PerfClock = std::chrono::steady_clock;
#define BEAMFORMER_TRACKER_PERF_START(name) \
        auto (name##_start) = PerfClock::now()
#define BEAMFORMER_TRACKER_PERF_STOP(name)                                   \
    impl_->window_ms.push_back(                                              \
        std::chrono::duration<double, std::milli>(PerfClock::now()           \
                                                  - (name##_start)).count())
#else
#define BEAMFORMER_TRACKER_PERF_START(name) (void)0
#define BEAMFORMER_TRACKER_PERF_STOP(name) (void)0
#endif

    if (!scan_enabled) {
        // ============================================================
        // Default / back-compat path — exact naive reproduction.
        // Scan disabled (coarse_grid_resolution <= 1): use the
        // trajectory-supplied per-window direction and the canonical DAS
        // emission pass. Output is byte-compatible with
        // beam_tracker_cpu_packed_intensity_into to within float rounding.
        // (No covariance, no search, no recursion.)
        // ============================================================
        for (std::size_t window = 0; window < window_count; ++window) {
            BEAMFORMER_TRACKER_PERF_START(disabled);
            const Vec3 direction = tracker_window_direction(
                prior, window, config_.integration_spectra);
            impl_->window_dirs[window] = direction;
            const std::size_t first_time = window * config_.integration_spectra;
            const std::size_t last_time =
                std::min(first_time + config_.integration_spectra, dims_.n_time);
            emit_window_das(packed, dims_, positions_m_, frequencies_hz_,
                            direction, first_time, last_time, intensity);
            BEAMFORMER_TRACKER_PERF_STOP(disabled);
        }
        return;
    }

    // ============================================================
    // Searching / adaptive path — O1..O6.
    // ============================================================
    // Per-frequency sample covariance accumulator (windowed), recursively
    // carried across windows (O5). The cache lives in impl_->R_freq.
    Vec3 prev_estimate = tracker_window_direction(prior, 0, config_.integration_spectra);

    const std::size_t G = config_.coarse_grid_resolution;
    const float fov_l = config_.search_fov_l;
    const float fov_m = config_.search_fov_m;
    const std::size_t L_refine = config_.refinement_levels;

    for (std::size_t window = 0; window < window_count; ++window) {
        BEAMFORMER_TRACKER_PERF_START(scan);
        const std::size_t first_time = window * config_.integration_spectra;
        const std::size_t last_time =
            std::min(first_time + config_.integration_spectra, dims_.n_time);
        const std::size_t K = last_time - first_time;
        if (K == 0) { BEAMFORMER_TRACKER_PERF_STOP(scan); continue; }

        // ---- O5: recursive covariance update over this window's snapshots.
        //   Haykin, Adaptive Filter Theory, Ch.10 (RLS with forgetting).
        const float a_comp = 1.0F - lambda;  // contribution of fresh samples
        // Option A: full window recompute blended with previous R via lambda.
        // We form the window's block covariance R_w_block over K snapshots,
        // spatial-smooth it (O2), then fold:
        //   R_w = lambda * R_{w-1} + (1-lambda) * R_w_block
        // (matches the spec's block-form recursion; with lambda == 1.0 this
        //  reduces to the plain block estimate R_w = R_w_block.)
        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
            // Decode the K snapshots for this frequency.
            std::vector<std::vector<Cfloat>> snaps;
            snaps.reserve(K);
            for (std::size_t t = first_time; t < last_time; ++t) {
                snaps.push_back(decode_snapshot(packed, dims_, t, f));
            }
            // O2: spatial smoothing + forward-backward.
            //   Shan/Wax/Kailath 1985; Weiss & Friedlander 1997.
            auto cov = spatial_smoothed_covariance(
                snaps, dims_.n_ant, config_.spatial_smoothing_subarray_size);
            // cov.first is M_eff×M_eff; second unused when smoothing off.
            std::vector<Cfloat> R_block = std::move(cov.first);
            if (config_.estimator == TrackerEstimator::Capon) {
                apply_diagonal_load(R_block, M_eff, config_.diagonal_load);
            }
            // Fold with previous R (O5).
            if (!impl_->R_initialised || lambda >= 1.0F) {
                impl_->R_freq[f] = R_block;
            } else {
                std::vector<Cfloat>& R = impl_->R_freq[f];
                for (std::size_t i = 0; i < M_eff * M_eff; ++i) {
                    R[i] = lambda * R[i] + a_comp * R_block[i];
                }
            }
        }
        impl_->R_initialised = true;

        // ---- O3 + O6: hierarchical coarse-to-fine search.
        //   Skolnik, Radar Handbook §7.11 (two-stage search);
        //   Haykin & Reilly 1991 (hierarchical ML lattice search).
        // Level 0: build the coarse grid centred on prev_estimate (the prior).
        std::vector<Vec3> coarse = build_centred_grid(prev_estimate, G, fov_l, fov_m);
        // Per-frequency coarse power summed across frequency for the
        // per-window direction decision (spec O1: "integrated across
        // frequency for the final per-window direction decision").
        std::vector<float> P_coarse(G * G, 0.0F);
        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
            const std::vector<Cfloat>& R_f = impl_->R_freq[f];
            const double wave_number =
                two_pi * static_cast<double>(frequencies_hz_[f])
                / speed_of_light_m_per_s;
            for (std::size_t cell = 0; cell < G * G; ++cell) {
                // M_eff steering vector for this (frequency, cell). When
                // smoothing reduces M_eff, the sub-array steering uses the
                // first M_eff antenna positions with their phases relative to
                // the cell direction — a consistent truncation.
                std::vector<Cfloat> a_vec(M_eff);
                for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                    a_vec[a_idx] = steer_one(positions_m_, wave_number,
                                             coarse[cell], a_idx);
                }
                float p = (config_.estimator == TrackerEstimator::Capon)
                              ? capon_power_correct(R_f, a_vec)
                              : bartlett_power(R_f, a_vec);
                if (!std::isfinite(p)) p = 0.0F;
                P_coarse[cell] += p;
            }
        }
        // Argmax of the integrated coarse spectrum.
        std::size_t best_cell = 0;
        float best_P = P_coarse[0];
        for (std::size_t cell = 1; cell < G * G; ++cell) {
            if (P_coarse[cell] > best_P) {
                best_P = P_coarse[cell];
                best_cell = cell;
            }
        }
        std::size_t best_u = best_cell % G;
        std::size_t best_v = best_cell / G;

        // Refinement: 3×3 halvings of the neighbourhood around the running best.
        // (Van Trees 2002 §5.4 practical DOA estimation.)
        float cell_half_l = fov_l / static_cast<float>(G);
        float cell_half_m = fov_m / static_cast<float>(G);
        Vec3 centre = coarse[best_cell];
        for (std::size_t level = 1; level <= L_refine; ++level) {
            cell_half_l *= 0.5F;
            cell_half_m *= 0.5F;
            // 3×3 neighbourhood candidates around `centre`.
            std::vector<Vec3> cand;
            std::vector<float> Pcand;
            cand.reserve(9);
            Pcand.assign(9, 0.0F);
            for (std::int64_t dv = -1; dv <= 1; ++dv) {
                for (std::int64_t du = -1; du <= 1; ++du) {
                    float l = centre[0] + static_cast<float>(du) * cell_half_l;
                    float m = centre[1] + static_cast<float>(dv) * cell_half_m;
                    const float r2 = l * l + m * m;
                    if (r2 > 0.999F) {
                        const float s = std::sqrt(0.999F / r2);
                        l *= s;
                        m *= s;
                    }
                    cand.push_back(direction_from_lm(l, m));
                }
            }
            for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                const std::vector<Cfloat>& R_f = impl_->R_freq[f];
                const double wave_number =
                    two_pi * static_cast<double>(frequencies_hz_[f])
                    / speed_of_light_m_per_s;
                for (std::size_t i = 0; i < cand.size(); ++i) {
                    std::vector<Cfloat> a_vec(M_eff);
                    for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                        a_vec[a_idx] =
                            steer_one(positions_m_, wave_number, cand[i], a_idx);
                    }
                    float p = (config_.estimator == TrackerEstimator::Capon)
                                  ? capon_power_correct(R_f, a_vec)
                                  : bartlett_power(R_f, a_vec);
                    if (!std::isfinite(p)) p = 0.0F;
                    Pcand[i] += p;
                }
            }
            std::size_t best = 0;
            for (std::size_t i = 1; i < Pcand.size(); ++i) {
                if (Pcand[i] > Pcand[best]) best = i;
            }
            centre = cand[best];
            // O4: quadratic 3-point interpolation around the level argmax.
            //   Thomson 1982; Priestley 1981 §6.1; Van Trees 2002 §5.4.
            // Pcand indexing: i = (dv+1)*3 + (du+1); centre is index 4,
            // u-neighbours at index 5 (du=+1) and 3 (du=-1),
            // v-neighbours at index 7 (dv=+1) and 1 (dv=-1).
            if (config_.enable_quadratic_peak_interp) {
                float du = 0.0F, dv = 0.0F;
                quadratic_interp(
                    Pcand[4],             // centre
                    Pcand[5],             // u +1
                    Pcand[3],             // u -1
                    Pcand[7],             // v -1 (dv=-1 row index 2)
                    Pcand[1],             // v +1 (dv=+1 row index 0)
                    du, dv);
                // Apply the fractional offset within the (now halved) cell.
                float l = centre[0] + du * cell_half_l;
                float m = centre[1] + dv * cell_half_m;
                const float r2 = l * l + m * m;
                if (r2 > 0.999F) {
                    const float s = std::sqrt(0.999F / r2);
                    l *= s;
                    m *= s;
                }
                centre = direction_from_lm(l, m);
            }
        }

        // O4 also applies at the coarse peak if no refinement requested.
        if (config_.enable_quadratic_peak_interp && L_refine == 0) {
            // Sample the 4-neighbourhood around best_cell on the coarse grid.
            float pc = P_coarse[best_v * G + best_u];
            auto at = [&](long long u, long long v) -> float {
                long long uu = static_cast<long long>(best_u) + u;
                long long vv = static_cast<long long>(best_v) + v;
                if (uu < 0 || uu >= static_cast<long long>(G) || vv < 0
                    || vv >= static_cast<long long>(G)) {
                    return 0.0F;
                }
                return P_coarse[static_cast<std::size_t>(vv) * G
                                + static_cast<std::size_t>(uu)];
            };
            float du = 0.0F, dv = 0.0F;
            quadratic_interp(pc, at(1, 0), at(-1, 0), at(0, 1), at(0, -1), du, dv);
            // Convert coarse-grid cell pitch into (l, m).
            const float step_l =
                (G == 1) ? 0.0F
                         : (2.0F * fov_l) / static_cast<float>(G - 1);
            const float step_m =
                (G == 1) ? 0.0F
                         : (2.0F * fov_m) / static_cast<float>(G - 1);
            float l = centre[0] + du * step_l;
            float m = centre[1] + dv * step_m;
            const float r2 = l * l + m * m;
            if (r2 > 0.999F) {
                const float s = std::sqrt(0.999F / r2);
                l *= s;
                m *= s;
            }
            centre = direction_from_lm(l, m);
            (void)pc;
        }

        // Track the estimate forward (O5 prediction-as-prior for next window).
        prev_estimate = centre;
        impl_->window_dirs[window] = centre;

        // ---- Final emission pass: plain Bartlett DAS using the estimated
        // direction (spec: the emitted product remains the plain Bartlett
        // power so on-disk output is byte-compatible for the same direction;
        // only the direction changes). Reuses generate_weights exactly.
        emit_window_das(packed, dims_, positions_m_, frequencies_hz_, centre,
                        first_time, last_time, intensity);
        BEAMFORMER_TRACKER_PERF_STOP(scan);
    }
#undef BEAMFORMER_TRACKER_PERF_START
#undef BEAMFORMER_TRACKER_PERF_STOP
}

// =====================================================================
// Free-function mirrors
// =====================================================================
Intensities cpu_opt_beam_tracker_packed_intensity(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& trajectory, const CpuOptTrackerConfig& opt) {
    Intensities intensity(dims.n_time * dims.n_freq * dims.n_beams);
    cpu_opt_beam_tracker_packed_intensity_into(packed, dims, trajectory,
                                               intensity, opt);
    return intensity;
}

// Stateless into-variant (no persistent covariance across calls — each call
// starts the RLS recursion fresh). The stateful class overload below is the
// preferred streaming entry point.
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed, const Dimensions& dims,
    const TrackerConfig& trajectory, Intensities& intensity,
    const CpuOptTrackerConfig& opt) {
    // Stateless variant: build a fresh tracker over the standard default
    // geometry / frequency plan, seed the trajectory prior, and run. The
    // covariance recursion starts fresh on every call (the stateful overload
    // below is the persistent streaming entry point).
    CpuOptBeamTracker tracker(default_positions(dims.n_ant),
                              channelized_frequencies(dims.n_freq), dims, opt);
    tracker.seed_trajectory(trajectory.trajectory);
    tracker.run_into(packed, intensity);
}

// Stateful variant: caller owns the tracker (preferred for streaming — the
// covariance recursion and steering cache persist across calls). Seeds the
// trajectory prior then runs, preserving any carry-over covariance state.
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed, const TrackerConfig& trajectory,
    CpuOptBeamTracker& tracker, Intensities& intensity) {
    tracker.seed_trajectory(trajectory.trajectory);
    tracker.run_into(packed, intensity);
}

}  // namespace beamformer
