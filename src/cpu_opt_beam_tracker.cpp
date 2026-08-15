#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

// Threading: the N_freq channels are fully independent, so the dominant
// per-frame cost (covariance formation + coarse grid scan + refinement) is
// embarrassingly parallel across frequencies. OpenMP is the project's chosen
// concurrency layer (linked via OpenMP::OpenMP_CXX in CMake); guarded so the
// TU still compiles cleanly when the feature is unavailable.
#if defined(_OPENMP)
#include <omp.h>
#define BEAMFORMER_TRACKER_OMP_ENABLED 1
#else
#define BEAMFORMER_TRACKER_OMP_ENABLED 0
#endif

// --- Project Includes ---
#include "beamformer/cpu_opt_beam_tracker.hpp"
#include "beamformer/complex.hpp"
#include "beamformer/weights.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"

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

// Steering vector for the covariance-search scan. NOTE the sign convention:
// the synthetic / received voltage is x_a = A * exp(-j φ_a) (see
// make_point_source / beam_tracker_make_moving_point_source, both set
// imag = -A sin φ). For the Hermitian quadratic form a(θ)^H R a(θ) with
// R = (1/K) Σ x x^H to peak at the TRUE source direction we need a(θ_true)
// to match the *data*, i.e. a(θ) = exp(-j φ(θ)) so that conj(a_a) x_a =
// exp(+j φ_a(θ)) exp(-j φ_a(source)) sums coherently when θ = source.
// (The emission pass uses generate_weights directly, so this sign choice
// is local to the scan and does NOT affect the on-disk byte-compat output.)
inline Cfloat steer_one(const std::vector<Vec3>& positions,
                        double wave_number, const Vec3& direction,
                        std::size_t a_idx) {
    const auto& p = positions[a_idx];
    const double delay_m = static_cast<double>(p[0]) * direction[0]
                           + static_cast<double>(p[1]) * direction[1]
                           + static_cast<double>(p[2]) * direction[2];
    const double phase = wave_number * delay_m;
    return Cfloat{static_cast<float>(std::cos(phase)),
                  static_cast<float>(-std::sin(phase))};
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
[[maybe_unused]] float capon_power_correct(const std::vector<Cfloat>& R_loaded,
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
[[maybe_unused]] void apply_diagonal_load(std::vector<Cfloat>& R, std::size_t M,
                         float diagonal_load) {
    if (diagonal_load <= 0.0F || M == 0) return;
    float trace = 0.0F;
    for (std::size_t i = 0; i < M; ++i) trace += R[i * M + i].real();
    const float eps = diagonal_load * trace / static_cast<float>(M);
    for (std::size_t i = 0; i < M; ++i) {
        R[i * M + i] += Cfloat{eps, 0.0F};
    }
}

// ------------------------------------------------------- pointer overloads
// Phase 1: the per-window hot path calls these over the preallocated
// `R_block_scratch` / `steer_scratch` / `L_factors` scratchpads so no
// std::vector is constructed per candidate cell. The arithmetic is identical
// to the vector-based references above (same indexing, same numerics).
inline void apply_diagonal_load_into(Cfloat* __restrict R, std::size_t M,
                                    float diagonal_load) {
    if (diagonal_load <= 0.0F || M == 0) return;
    float trace = 0.0F;
    for (std::size_t i = 0; i < M; ++i) trace += R[i * M + i].real();
    const float eps = diagonal_load * trace / static_cast<float>(M);
    for (std::size_t i = 0; i < M; ++i) {
        R[i * M + i] += Cfloat{eps, 0.0F};
    }
}

inline float bartlett_power_into(const Cfloat* __restrict R,
                                const Cfloat* __restrict a, std::size_t M) {
    Cfloat acc{0.0F, 0.0F};
    for (std::size_t r = 0; r < M; ++r) {
        Cfloat row_sum{0.0F, 0.0F};
        for (std::size_t c = 0; c < M; ++c) {
            row_sum += R[r * M + c] * a[c];
        }
        acc += std::conj(a[r]) * row_sum;
    }
    return acc.real();
}

// Hermitian-symmetric, trace-precomputed Bartlett quadratic form
// `a^H R a` for a Hermitian R (M×M row-major) and a unit-modulus steering
// vector `a` (|a_i|^2 == 1, the Bartlett/Capon scan case). Exploiting the
// Hermitian symmetry R_{c,r} = R_{r,c}*:
//
//   a^H R a = Σ_r |a_r|^2 R_{rr} + Σ_{r≠c} conj(a_r) R_{rc} a_c
//           = trace(R) + 2·Re( Σ_{r<c} conj(a_r) * R_{rc} * a_c )
//
// so only the upper triangle (M(M-1)/2 cross terms) is accumulated instead of
// the full M×M sweep, halving the inner-product FMAs and turning the diagonal
// sum into a single precomputed `trace` (constant across all search cells of
// a given frequency — recomputed once per freq in run_into, never per cell).
//
// The steering vectors built here have |a_i|^2 == 1 to float precision
// (exp(-jφ) = cos φ - j sin φ), so the `|a_r|^2 == 1` substitution is exact in
// float arithmetic. The cross-term reduction uses `2·Re(...)` which is the
// standard Hermitian-quadratic identity; the tiny float-reordering versus the
// full-sweep form does not affect the well-separated DOA argmax the search
// resolves (the source peak dominates by orders of magnitude).
inline float bartlett_power_hermitian_trace(const Cfloat* __restrict R,
                                            const Cfloat* __restrict a,
                                            std::size_t M, float trace_R) {
    Cfloat cross{0.0F, 0.0F};
    for (std::size_t r = 0; r < M; ++r) {
        const Cfloat ar_conj = std::conj(a[r]);
        // Upper triangle only: c from r+1 .. M-1. R[r*M+c] is the upper
        // entry read directly; the lower half (R_{c,r}) is its conjugate.
        const std::size_t base = r * M;
        for (std::size_t c = r + 1; c < M; ++c) {
            cross += ar_conj * R[base + c] * a[c];
        }
    }
    // 2·Re(cross) accounts for both (r<c) and its conjugate (c<r) half.
    return trace_R + 2.0F * cross.real();
}

// In-place Cholesky solve `R w = b` operating on a *copy* of R passed in
// `work` (M×M row-major) and writing the solution back into `b`. Returns true
// on success, false if R is not numerically positive-definite. Identical
// arithmetic to the vector `cholesky_solve` above.
inline bool cholesky_solve_into(Cfloat* __restrict work, Cfloat* __restrict b,
                                std::size_t M) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            Cfloat sum = work[i * M + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= work[i * M + k] * std::conj(work[j * M + k]);
            }
            if (i == j) {
                if (sum.real() <= 0.0F) {
                    return false;  // not positive-definite
                }
                const float d = std::sqrt(sum.real());
                work[i * M + j] = Cfloat{d, 0.0F};
            } else {
                const Cfloat& diag = work[j * M + j];
                const float inv = (diag.real() != 0.0F) ? 1.0F / diag.real() : 0.0F;
                work[i * M + j] = Cfloat{sum.real() * inv, sum.imag() * inv};
            }
        }
    }
    for (std::size_t i = 0; i < M; ++i) {
        Cfloat sum = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= work[i * M + k] * b[k];
        }
        const float d = work[i * M + i].real();
        b[i] = Cfloat{sum.real() / d, sum.imag() / d};
    }
    for (std::size_t i = M; i-- > 0;) {
        Cfloat sum = b[i];
        for (std::size_t k = i + 1; k < M; ++k) {
            sum -= std::conj(work[k * M + i]) * b[k];
        }
        const float d = work[i * M + i].real();
        b[i] = Cfloat{sum.real() / d, sum.imag() / d};
    }
    return true;
}

// Capon/MVDR power using preallocated scratch: copies the (already
// diagonal-loaded) R into `work` (M×M), seeds `b` = a, solves R w = a, and
// returns 1 / (a^H w). Falls back to Bartlett on the loaded R if the Cholesky
// factorization fails (matches capon_power_correct semantics). `a` is read
// only; `work` and `b` are caller-owned scratch (typically steer_scratch as b
// and L_factors as work).
inline float capon_power_into(const Cfloat* __restrict R_loaded,
                              const Cfloat* __restrict a, std::size_t M,
                              Cfloat* __restrict work, Cfloat* __restrict b) {
    std::copy(R_loaded, R_loaded + M * M, work);
    for (std::size_t i = 0; i < M; ++i) b[i] = a[i];
    if (!cholesky_solve_into(work, b, M)) {
        return bartlett_power_into(R_loaded, a, M);  // fallback (O1 spec)
    }
    Cfloat acc{0.0F, 0.0F};
    for (std::size_t i = 0; i < M; ++i) {
        acc += std::conj(a[i]) * b[i];
    }
    const float denom = acc.real();
    if (!std::isfinite(denom) || denom <= 1.0e-12F) {
        return bartlett_power_into(R_loaded, a, M);
    }
    return 1.0F / denom;
}

// ------------------------------------------------------- Phase 2: FOSM
// Factor-Once / Solve-Many Cholesky decoupling. The pre-Phase-2 Capon hot
// path re-ran the full O(M^3) factorization for every candidate cell (coarse
// grid + every refinement stage) by way of `capon_power_into`. FOSM instead
// factorizes the (diagonal-loaded) per-frequency covariance R = L L^H exactly
// ONCE per frequency per window, then for each candidate cell evaluates the
// MVDR power as the squared norm of a single O(M^2) forward substitution:
//   a^H R^{-1} a = || L^{-1} a ||_2^2 = || v ||_2^2,   L v = a,  P = 1/||v||^2.
// This is mathematically identical to the per-cell Cholesky solve (it is the
// same factorization used internally by `cholesky_solve_into` — only the back
// substitution, which produced `w = R^{-1} a` only to take `a^H w`, is elided
// since `a^H R^{-1} a = ||L^{-1} a||^2` for a Hermitian PD R). The storage
// convention matches the existing solve: `R` is the full M×M row-major
// Hermitian covariance, only the lower triangle is read during factorization
// and only the lower triangle is valid (L) afterwards — the quadratic-form
// power path reads the lower triangle exclusively.

// In-place Cholesky factorization R = L L^H of a Hermitian M×M covariance
// stored row-major (only the lower triangle is referenced; the strict upper
// triangle is left stale and must not be read by callers afterward). On
// success `R` holds the lower-triangular factor L (real positive diagonal).
// Returns false if `R` is not numerically positive-definite (the caller
// falls back to the per-cell solve for that frequency so degenerate windows
// retain the prior Phase-1 numerics). The tiny `1e-12` positive definiteness
// sill matches the existing `capon_power_into` denominator guard.
bool cholesky_factorize(Cfloat* __restrict R, std::size_t M) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            Cfloat sum = R[i * M + j];
            for (std::size_t k = 0; k < j; ++k) {
                sum -= R[i * M + k] * std::conj(R[j * M + k]);
            }
            if (i == j) {
                if (sum.real() <= 1.0e-12F) return false;
                R[i * M + j] = Cfloat{std::sqrt(sum.real()), 0.0F};
            } else {
                const float inv = 1.0F / R[j * M + j].real();
                R[i * M + j] = Cfloat{sum.real() * inv, sum.imag() * inv};
            }
        }
    }
    return true;
}

// MVDR / Capon power from a precomputed Cholesky factor L (lower triangular,
// the output of `cholesky_factorize`). Solves L v = a by forward substitution
// into the caller-owned `v_scratch` (length M) and returns 1 / ||v||^2. The
// factor is read-only; `a` is read-only. `v_scratch` is mutated in place and
// must be distinct from `a`. Falls back to 0 on a degenerate (near-zero)
// denominator, mirroring the `> 1e-12` guard in `capon_power_into`.
float capon_power_from_factor(const Cfloat* __restrict L, const Cfloat* __restrict a,
                              std::size_t M, Cfloat* __restrict v_scratch) {
    float norm_sq = 0.0F;
    for (std::size_t i = 0; i < M; ++i) {
        Cfloat sum = a[i];
        for (std::size_t k = 0; k < i; ++k) {
            sum -= L[i * M + k] * v_scratch[k];
        }
        const float d_inv = 1.0F / L[i * M + i].real();
        const Cfloat vi{sum.real() * d_inv, sum.imag() * d_inv};
        v_scratch[i] = vi;
        norm_sq += vi.real() * vi.real() + vi.imag() * vi.imag();
    }
    return (norm_sq > 1.0e-12F) ? (1.0F / norm_sq) : 0.0F;
}

// ----------------------------------------------------------- spatial smooth
// Spatial smoothing + forward-backward averaging (O2).
//   Shan/Wax/Kailath 1985 (spatial smoothing),
//   Weiss & Friedlander 1997 (forward-backward averaging).
// Forms R̂ = 1/2 (R̃ + J R̃* J) over L=M-P+1 overlapping sub-arrays of length P.
// Output is P×P, row-major. `snapshots` are the K length-M snapshots (each a
// length-M complex vector). When smoothing is disabled (P == 0 or P >= M),
// returns the plain full-array covariance scaled by 1/K.
[[maybe_unused]] std::pair<std::vector<Cfloat>, std::vector<Cfloat>>
spatial_smoothed_covariance(
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

// ----------------------------------------------------- decode a snapshot (into)
// Phase 1: zero-allocation counterpart that decodes one (time, freq) snapshot
// into a caller-provided destination rather than returning a heap-allocated
// vector. Used by the `run_into` hot path against the preallocated
// `snapshot_buffer` slice so the per-window covariance build does no heap
// allocation. Indexing matches `decode_snapshot` exactly (uses `voltage_index`).
inline void decode_snapshot_into(const PackedVoltage& packed,
                                 const Dimensions& dims,
                                 std::size_t time, std::size_t freq,
                                 Cfloat* __restrict out) {
    for (std::size_t a_idx = 0; a_idx < dims.n_ant; ++a_idx) {
        const auto sample = unpack_complex_int4(
            packed[voltage_index(time, freq, a_idx, dims)]);
        out[a_idx] = Cfloat{static_cast<float>(sample.real),
                            static_cast<float>(sample.imag)};
    }
}

// Compute the spatially-smoothed + forward-backward-averaged covariance of the
// K length-M snapshots that are laid out contiguously in `snap_flat` (each
// snapshot strided by `snap_stride` floats) directly into `R_out` (an
// M_eff x M_eff row-major, fully-populated — both triangles written — row-major
// buffer). When smoothing is disabled (P_sub == 0 or P_sub >= M) `M_eff == M`
// and the plain 1/K sample covariance is written.
//
// Optimizations versus the prior scalar form (defects flagged in the analysis):
//  • Exploits Hermitian symmetry — only the upper triangle (incl. the
//    diagonal) of each sub-array outer product is accumulated; the lower
//    triangle is filled by conjugate mirroring (a no-op bit-exact math step
//    since R_{c,r} == R_{r,c}* exactly for an outer-product average). This
//    halves the per-snapshot complex FMAs from M^2 to M(M+1)/2.
//  • When smoothing is disabled (L == 1, the dominant benchmark path:
//    default config, M_eff == n_ant), accumulates straight into `R_out`
//    instead of into `per_sub_scratch` and copying. No sub-array scratch is
//    touched, removing the per-window `assign`-the-whole-M^2-buffer cost.
//  • Smoothing path still accumulates the L sub-arrays separately (they have
//    distinct offsets and so cannot share one accumulator), but into a
//    per-sub upper-triangle slab — half the memory traffic and FMAs.
//
// The numeric result is identical up to single-precision floating-point
// reordering of the upper-triangle accumulation, which does not perturb the
// well-separated DOA argmax the search resolves.
inline void spatial_smoothed_covariance_into(
    const Cfloat* __restrict snap_flat, std::size_t K, std::size_t M,
    std::size_t snap_stride, std::size_t P_sub, std::size_t M_eff,
    Cfloat* __restrict R_out,
    Cfloat* __restrict per_sub_scratch,
    std::size_t /*per_sub_capacity*/) {
    const bool smoothing = (P_sub != 0 && P_sub < M && P_sub >= 1);
    const std::size_t L = smoothing ? (M - P_sub + 1) : 1;
    const float inv_K = 1.0F / static_cast<float>(K);

    if (!smoothing) {
        // L == 1: accumulate the single (full-array) sample covariance straight
        // into `R_out`. Zero the whole M_eff^2 buffer once (cheap relative to
        // the K * M(M+1)/2 FMAs), then fill the upper triangle and mirror.
        for (std::size_t i = 0; i < M_eff * M_eff; ++i) {
            R_out[i] = Cfloat{0.0F, 0.0F};
        }
        for (std::size_t k = 0; k < K; ++k) {
            const Cfloat* __restrict x = snap_flat + k * snap_stride;
            for (std::size_t r = 0; r < M_eff; ++r) {
                const Cfloat xr = x[r];
                const std::size_t base = r * M_eff;
                // Diagonal: real-valued term |x_r|^2 / K.
                {
                    const Cfloat& xc = x[r];
                    R_out[base + r] += xr * std::conj(xc) * inv_K;
                }
                // Strict upper triangle c = r+1 .. M_eff-1.
                for (std::size_t c = r + 1; c < M_eff; ++c) {
                    R_out[base + c] += xr * std::conj(x[c]) * inv_K;
                }
            }
        }
        // Mirror the upper triangle into the strict lower triangle via the
        // Hermitian identity R_{c,r} = R_{r,c}*. The diagonal is already
        // correct (and real). For row r, column c with c < r: the upper entry
        // is R[c][r] (column r > row c), and we write it into the lower entry
        // R[r][c] = conj(R[c][r]).
        for (std::size_t r = 1; r < M_eff; ++r) {
            for (std::size_t c = 0; c < r; ++c) {
                R_out[r * M_eff + c] = std::conj(R_out[c * M_eff + r]);
            }
        }
        return;
    }

    // --- Smoothing path: L >= 2 sub-arrays, distinct offsets. ---
    const float inv_L = 1.0F / static_cast<float>(L);
    // Per-sub upper-triangle slab (only the upper triangle is stored, packed
    // contiguously per sub-array). Capacity L * M_eff^2 is reserved by the
    // caller; we only touch the upper-triangle region, zeroing it once.
    std::fill_n(per_sub_scratch, L * M_eff * M_eff, Cfloat{0.0F, 0.0F});
    for (std::size_t sub = 0; sub < L; ++sub) {
        Cfloat* __restrict sub_R =
            per_sub_scratch + sub * M_eff * M_eff;
        for (std::size_t k = 0; k < K; ++k) {
            const Cfloat* __restrict x =
                snap_flat + k * snap_stride + sub;
            for (std::size_t r = 0; r < M_eff; ++r) {
                const Cfloat xr = x[r];
                const std::size_t base = r * M_eff;
                sub_R[base + r] += xr * std::conj(x[r]) * inv_K;
                for (std::size_t c = r + 1; c < M_eff; ++c) {
                    sub_R[base + c] += xr * std::conj(x[c]) * inv_K;
                }
            }
        }
    }
    // Average the L forward sub-arrays into R_out (the forward R̃), filling the
    // upper triangle; the strict lower triangle is mirrored below.
    for (std::size_t r = 0; r < M_eff; ++r) {
        const std::size_t base = r * M_eff;
        for (std::size_t c = r; c < M_eff; ++c) {
            Cfloat acc{0.0F, 0.0F};
            for (std::size_t sub = 0; sub < L; ++sub) {
                acc += per_sub_scratch[sub * M_eff * M_eff + base + c] * inv_L;
            }
            R_out[base + c] = acc;
        }
    }
    // Mirror the populated upper triangle into the strict lower (R̃ is
    // Hermitian): R[r][c] = conj(R[c][r]) for c < r, where R[c][r] (row c,
    // column r with r > c) is the populated upper entry.
    for (std::size_t r = 1; r < M_eff; ++r) {
        for (std::size_t c = 0; c < r; ++c) {
            R_out[r * M_eff + c] = std::conj(R_out[c * M_eff + r]);
        }
    }
    // Forward-backward: R̂ = 1/2 (R̃ + J R̃* J). J reverses indices. Fold in
    // place into R_out (R̃ -> R̂).
    for (std::size_t r = 0; r < M_eff; ++r) {
        for (std::size_t c = 0; c < M_eff; ++c) {
            const std::size_t pr = M_eff - 1 - r;
            const std::size_t pc = M_eff - 1 - c;
            const Cfloat conj_e = std::conj(R_out[pr * M_eff + pc]);
            const Cfloat fwd = R_out[r * M_eff + c];
            R_out[r * M_eff + c] = 0.5F * (fwd + conj_e);
        }
    }
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

// Parallel emission counterpart to `emit_window_das` (fix-plan defect #8):
// distributes the (time, frequency) DAS grid across the OpenMP team. Each
// output `intensity_index(time, frequency, 0, dims)` cell is independent and
// its inner n_ant element-sum is computed with the identical serial arithmetic
// as `emit_window_das`, so the result is bit-identical per cell — only the
// scheduling of cells across threads differs. `weights` is read-only and safe
// to share. Used ONLY by the searching/adaptive path (the scan-disabled
// back-compat path keeps the serial `emit_window_das` so the byte-equal
// `naive == opt` regression contract is preserved regardless of thread count).
void emit_window_das_parallel(const PackedVoltage& packed,
                               const Dimensions& dims,
                               const std::vector<Vec3>& positions,
                               const std::vector<float>& frequencies,
                               const Vec3& direction, std::size_t first_time,
                               std::size_t last_time, Intensities& intensity) {
    const auto weights = generate_weights(
        dims, positions, frequencies, std::vector<Vec3>{direction});
    const std::size_t n_time_cells = last_time - first_time;
    const std::size_t n_freq = dims.n_freq;
    const std::size_t total_cells = n_time_cells * n_freq;
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t ci = 0;
         ci < static_cast<std::ptrdiff_t>(total_cells); ++ci) {
        const std::size_t t_off =
            static_cast<std::size_t>(ci) / n_freq;
        const std::size_t frequency =
            static_cast<std::size_t>(ci) % n_freq;
        const std::size_t time = first_time + t_off;
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

// Fused emission counterpart that reuses the already-decoded snapshot buffer
// populated during covariance formation, instead of re-decoding the int4
// voltage a second time. `snap_buf_base` holds the [freq][K][n_ant] complex
// float snapshots exactly as `spatial_smoothed_covariance_into` consumed them
// (`snap_buf_base + f*snap_slot + t*snap_stride`). `weights_c` is the standard
// beam-0 weight set for the emission direction, produced once per window from a
// single thread using `generate_weights` (so the emitted product matches
// `emit_window_das`/`emit_window_das_parallel` bit-for-bit per cell; the only
// freedom is (time,freq) scheduling across threads).
//
// Per-cell inner sum is byte-identical to the serial DAS: each summand pair
// `(sum_re, sum_im)` accumulates `w.re*s.re - w.im*s.im` and
// `w.re*s.im + w.im*s.re` in fixed element order. Since the snapshot buffer was
// decoded with the identical `unpack_complex_int4` path that the second-pass
// decode would have used, every float summand is the same — only the int4
// re-decode and the redundant `generate_weights` trig are eliminated.
void emit_window_das_fused(const Cfloat* __restrict snap_buf_base,
                            std::size_t K_this, std::size_t snap_stride,
                            std::size_t snap_slot,
                            const Dimensions& dims,
                            const Cfloat* __restrict weights_c,
                            std::size_t first_time,
                            Intensities& intensity) {
    const std::size_t n_freq = dims.n_freq;
    const std::size_t total_cells = K_this * n_freq;
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t ci = 0;
         ci < static_cast<std::ptrdiff_t>(total_cells); ++ci) {
        const std::size_t t_off =
            static_cast<std::size_t>(ci) / n_freq;
        const std::size_t frequency =
            static_cast<std::size_t>(ci) % n_freq;
        const std::size_t time = first_time + t_off;
        const Cfloat* __restrict snap_f =
            snap_buf_base + frequency * snap_slot + t_off * snap_stride;
        const std::size_t w_base = weight_index(0, frequency, 0, dims);
        float sum_real = 0.0F;
        float sum_imag = 0.0F;
        for (std::size_t element = 0; element < dims.n_ant; ++element) {
            const Cfloat& s = snap_f[element];
            const Cfloat& w = weights_c[w_base + element];
            sum_real += w.real() * s.real() - w.imag() * s.imag();
            sum_imag += w.real() * s.imag() + w.imag() * s.real();
        }
        intensity[intensity_index(time, frequency, 0, dims)] =
            sum_real * sum_real + sum_imag * sum_imag;
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

    // --- Phase 3: precomputed relative-grid phasor tables (O6, trig kill). ---
    // The steering phase is `phi_a(l,m) = k(f) * (p_x*l + p_y*m + p_z*n(l,m))`
    // (see `steer_one` above). For a PLANAR array lying in the z=0 plane every
    // antenna has p_z == 0, so the nonlinear n(l,m)=sqrt(1-l*l-m*m) term never
    // enters the phase: `phi_a(l,m) = k(f)*(p_x*l + p_y*m)` — LINEAR in (l,m).
    // The additive phase decomposition then holds exactly:
    //   phi_a(u0 + du) = phi_a(u0) + k(f)*(p_x*du_l + p_y*du_m)
    // so the absolute steering phasor factors as
    //   a_a(u0+du) = centre_phasor_a(f) * rel_phasor_a(f, du)
    // The relative lattice offsets `du` are IDENTICAL for every window (only
    // the coarse pitch / refinement pitches and antenna geometry set them, all
    // static), so `rel_phasor` is built ONCE in the constructor — the entire
    // coarse scan + every refinement candidate drops from 2 trig calls/antenna
    // to a single complex multiply/antenna. This is the Phase 3 objective.
    //
    // `phasor_fast_path` is false iff the array is not coplanar (some |p_z|>eps):
    // the additive decomposition is then not exact, so the hot path keeps the
    // original `steer_one` trig call per cell (no regression — `default_positions`
    // is planar so tests take the fast path).
    bool phasor_fast_path = false;

    // Coarse-grid relative phasor per frequency: `rel_coarse[f][cell][a]`, flat
    // layout `rel_coarse[cell * n_freq * M_eff + f * M_eff + a]`
    // = exp(-j * k(f) * (p_x*du_l[cell] + p_y*du_m[cell])).
    // `du_l[cell] = (u-(G-1)/2)*step_l`, `du_m[cell] = (v-(G-1)/2)*step_m`: the
    // offset of cell (v,u) from the grid centre (the prior). At the centre cell
    // the offset is exactly zero so the phasor is +1 — matching the identity
    // `a == centre_phasor` there. Sized in the ctor only when the fast path is
    // taken; left empty otherwise.
    std::vector<Cfloat> rel_coarse;

    // Refinement relative phasors per level per frequency:
    // `rel_refine[L][o][f][a]`, flat `rel_refine[L][o * n_freq * M_eff + f*M_eff + a]`,
    // `o = (dv+1)*3 + (du+1) in [0,9)` for the 3x3 candidate pattern at the
    // parent-cell pitch. At level `L` (1-indexed) the candidate offset from the
    // parent centre is `(du*half_L, dv*half_L)` with
    // `half_L = (search_fov*/G) / 2^L`. `rel_refine[L][4]` (centre) is +1.
    std::vector<std::vector<Cfloat>> rel_refine;  // size: L_refine, each 9*n_freq*M
    // Per-level half-cell pitch cached at build time (1-indexed levels). Read by
    // the scan to detect the (rare) unit-disk clamp that invalidates the static
    // offset and triggers the trig fallback for that candidate only.
    std::vector<float> refine_half_l;  // size: L_refine
    std::vector<float> refine_half_m;  // size: L_refine

    // Per-window centre phasor scratch: `centre_phasor[f][a]`
    // = exp(-j * k(f) * (p_x*l0 + p_y*m0)) for the current search centre (l0,m0).
    // Recomputed ONCE per window per frequency (n_freq*M_eff trig calls), stored
    // flat as `centre_phasor[f * M_eff + a]`. Sized in the ctor (fast path only).
    std::vector<Cfloat> centre_phasor;

    // Per-antenna planar position components (copies of the first M_eff rows of
    // positions_m_), cached so the hot path indexes a tight contiguous buffer
    // when smoothing truncates the aperture to M_eff antennas.
    std::vector<float> pos_x;  // size: M_eff
    std::vector<float> pos_y;  // size: M_eff

    // --- Phase 1 preallocated scratchpad buffers (zero-allocation hot path) ---
    // Sized once in the constructor from the configured dimensions; the
    // `run_into` hot path reuses these slices instead of allocating per
    // window. All sizes are worst-case capacities (independent of window K,
    // which is bounded by integration_spectra).
    // Snapshot decode buffer for the covariance build, sized for per-frequency
    // parallelism: `n_freq * K_max * n_ant` contiguous slots, so each thread
    // owning frequency `f` decodes its K snapshots into a private slice
    // `f * K_max * n_ant` with no cross-thread aliasing. (Hermitian-symmetric
    // covariance formation reads snapshots in [k][antenna] order, which is the
    // contiguous strided layout the existing inner kernel already expects.)
    std::vector<Cfloat> snapshot_buffer;  // size: n_freq * K_max * n_ant
    // Per-frequency smoothed-covariance block buffers (M_eff×M_eff each, flat
    // `R_block_freq[f * M_eff * M_eff]`), so the parallel frequency loop writes
    // each frequency's smoothed covariance into a private slot — no false
    // sharing with neighbouring threads. The per-fold step `lambda R + (1-lambda)
    // R_block` then writes `R_freq[f]` (also per-frequency-independent).
    std::vector<Cfloat> R_block_freq;  // size: n_freq * M_eff * M_eff
    std::vector<Cfloat> L_factors;        // size: n_freq * M_eff * M_eff (Capon
                                           //        Cholesky scratch per frequency)
    // Per-thread steering-vector scratch, allocated at OpenMP init. With N
    // threads and `n_threads * M_eff` contiguous slots, thread `t` owns the slice
    // `t * M_eff`. Thread private (no aliasing across the parallel coarse /
    // refinement scan).
    std::vector<Cfloat> steer_scratch;    // size: n_threads * M_eff
    // Capon RHS/solution scratch (M_eff): the Cholesky solve writes R^{-1} a
    // here. Kept separate from `steer_scratch` (which holds the read-only `a`)
    // so the a^H w step reads the original steering vector after the solve.
    // Sized `n_threads * M_eff` for the same reason as `steer_scratch`.
    std::vector<Cfloat> capon_b_scratch;  // size: n_threads * M_eff
    // Per-subarray covariance accumulation scratch for spatial smoothing,
    // per-folder (one folder per OpenMP thread so the parallel frequency loop
    // can each accumulate its sub-array covariances without aliasing). Capacity
    // per folder: L_max * M_eff^2 where L_max = n_ant - M_eff + 1 (1 when M_eff
    // == n_ant, i.e. smoothing disabled). Threads index `t * L_max * M_eff^2`.
    std::vector<Cfloat> per_sub_scratch;
    // Cached per-frequency trace(R_freq[f]) (post-fold, pre-diagonal-load) — the
    // diagonal-sum half of a^H R a Bartlett power, constant across all search
    // cells of a frequency, computed ONCE per window per frequency rather than
    // inside every Bartlett grid cell. Sized `n_freq`.
    std::vector<float> R_trace;
    // Number of OpenMP worker threads the parallel `run_into` paths use.
    // Captured at hot-path entry (omp_get_max_threads()) once per window so it
    // is consistent across the cov-formation, capon fold, and the scan loops that
    // all use the same per-thread scratch indexing scheme.
    int n_threads = 1;
    std::vector<float>  power_grid;       // size: G * G (coarse integrated power)
    // Beam-0 DAS weight set for the per-window emission direction, reused by
    // the fused emission pass (emit_window_das_fused) instead of re-running
    // generate_weights on a second pass. Populated once per window under
    // #pragma omp single on the persistent team, then read-only across the
    // emission for loop. Sized n_freq * batch * n_ant in the ctor (max batch).
    std::vector<Cfloat> emit_weights;
    // Per-thread coarse power accumulator (G_max*G_max floats per thread) +
    // per-thread refinement power accumulator (9 floats per thread). Owned
    // per-thread inside the persistent parallel region (index by
    // omp_get_thread_num() * stride). Avoids per-window vector<vector> allocs.
    std::vector<float>  P_local_all;     // size: n_threads * G_max * G_max
    std::vector<float>  Pcand_local_all; // size: n_threads * 9
    std::size_t G_threads_stride = 0;   // G_max*G_max per thread
    // Phase 2: per-frequency positive-definite flag for the FOSM Cholesky
    // factorization. Set once per window (after the O5 fold + diagonal load)
    // and read by both the coarse scan and every refinement stage to decide
    // whether to use the amortized `capon_power_from_factor` path or fall
    // back to the per-cell `capon_power_into` solve for a degenerate freq.
    // Sized once in the constructor (off the hot path) — no per-window alloc.
    std::vector<char> capon_factor_ok;    // size: n_freq
    // Per-subarray covariance accumulation scratch for spatial smoothing:
    // capacity L_max * M_eff^2 where L_max = n_ant - M_eff + 1 (max sub-array
    // count; 1 when smoothing disabled since M_eff == n_ant).
    std::size_t K_max = 0;                // integration_spectra capacity bound
    std::size_t G_max = 0;               // coarse grid cells-per-axis capacity

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

    // -----------------------------------------------------------------
    // Debug search capture (only with -DBEAMFORMER_TRACKER_DEBUG).
    // Populated by run_into during the scan path so a post-hoc dump can write
    // every intermediate quantity needed to diagnose a failed DOA assertion.
    // Empty / unused when the macro is undefined → zero overhead.
    // -----------------------------------------------------------------
#if defined(BEAMFORMER_TRACKER_DEBUG)
    // Per window: the coarse grid centres (G*G Vec3, row-major) and the
    // integrated coarse power spectrum (G*G floats) used for the argmax.
    std::vector<std::vector<Vec3>> dbg_coarse_centres;
    std::vector<std::vector<float>> dbg_coarse_power;
    // Per window, per refinement level: the 9 candidate directions and 9
    // integrated powers (row-major 3x3), plus the level's half-cell pitch.
    std::vector<std::vector<std::vector<Vec3>>> dbg_refine_cands;
    std::vector<std::vector<std::vector<float>>> dbg_refine_power;
    // Per window, per frequency: the M_eff×M_eff Hermitian covariance used
    // for the spectrum evaluation (after O2 smoothing + O5 fold + Capon load).
    std::vector<std::vector<std::vector<Cfloat>>> dbg_R_freq_per_window;
    // Per window, per frequency, per snapshot: decoded length-M_ant snapshot
    // (full array, pre-smoothing) so the source phase can be re-derived.
    std::vector<std::vector<std::vector<std::vector<Cfloat>>>> dbg_snapshots;
    // Per-window carrying direction (the prev_estimate fed to level 0).
    std::vector<Vec3> dbg_prior_centre;
    // Per-window final estimate written into window_dirs.
    std::vector<Vec3> dbg_final_estimate;
    // The trajectory prior seeded before the run.
    TrackerTrajectoryConfig dbg_seeded_prior;
    bool dbg_prior_captured = false;

    void reset_dbg() {
        dbg_coarse_centres.clear();
        dbg_coarse_power.clear();
        dbg_refine_cands.clear();
        dbg_refine_power.clear();
        dbg_R_freq_per_window.clear();
        dbg_snapshots.clear();
        dbg_prior_centre.clear();
        dbg_final_estimate.clear();
    }
#else
    void reset_dbg() {}
#endif

    void reset_R(std::size_t n_freq, std::size_t M_eff_in) {
        M_eff = M_eff_in;
        R_freq.assign(n_freq, std::vector<Cfloat>(M_eff * M_eff, Cfloat{0.0F, 0.0F}));
        R_initialised = false;
        already_initialised_for_run = false;
    }
};

// =====================================================================
// Special members — defined here (not in the header) so `Impl` is complete.
// =====================================================================
CpuOptBeamTracker::~CpuOptBeamTracker() = default;
CpuOptBeamTracker::CpuOptBeamTracker(CpuOptBeamTracker&&) noexcept = default;
CpuOptBeamTracker& CpuOptBeamTracker::operator=(CpuOptBeamTracker&&)
    noexcept = default;

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

    // --- Phase 1: preallocate the zero-allocation hot-path scratchpads. ---
    // K_max bounds the number of snapshots integrated per window (the final
    // window may be shorter, but never longer than integration_spectra). The
    // snapshot buffer holds K_max decoded length-M_eff vectors back-to-back so
    // the covariance accumulation can iterate contiguous memory. The remaining
    // buffers are sized from M_eff / the coarse grid; they are reused (not
    // grown) on every window. The capacities below are conservative upper
    // bounds and never change after construction.
    const std::size_t K_max = config_.integration_spectra;
    const std::size_t G_cap =
        std::max<std::size_t>(config_.coarse_grid_resolution, 1);
    impl_->K_max = K_max;
    impl_->G_max = G_cap;
    // Per-frequency parallel decode buffer: n_freq * K_max * n_ant contiguous
    // slots, so each window's parallel frequency loop decodes its K snapshots
    // into a private `f * K_max * n_ant` slice (no cross-thread aliasing).
    impl_->snapshot_buffer.assign(dims_.n_freq * K_max * dims_.n_ant,
                                  Cfloat{0.0F, 0.0F});
    impl_->R_block_freq.assign(dims_.n_freq * M_eff * M_eff, Cfloat{0.0F, 0.0F});
    impl_->L_factors.assign(dims_.n_freq * M_eff * M_eff,
                            Cfloat{0.0F, 0.0F});
    // Per-thread steering / Capon-RHS scratch: grown lazily at run_into entry
    // to `omp_get_max_threads()` so a freshly-built tracker is usable on a
    // machine with a different thread cap than construction time. Baseline of
    // 1 keeps the buffers non-empty for the OpenMP-disabled fallback path.
    impl_->steer_scratch.assign(M_eff, Cfloat{0.0F, 0.0F});
    impl_->capon_b_scratch.assign(M_eff, Cfloat{0.0F, 0.0F});
    impl_->power_grid.assign(G_cap * G_cap, 0.0F);
    impl_->capon_factor_ok.assign(dims_.n_freq, 0);
    impl_->R_trace.assign(dims_.n_freq, 0.0F);
    impl_->n_threads = 1;
    // Persistent-team scratch (Phase 4): the emission weight set for the
    // current window's emit direction (beam 0) and the per-thread coarse /
    // refinement power accumulators. Sized conservatively to the OpenMP cap
    // at construction (1 thread → minimal allocation), grown lazily at run_into
    // entry to omp_get_max_threads() so a tracker moved across machines still
    // works. The emit weights are sized for the full n_freq * K_max * n_ant
    // beam-0 weight set (one weight per (freq, antenna) per snapshot is NOT
    // needed — weights are direction/freq/antenna-indexed, not time-indexed).
    {
        const std::size_t max_nt = 1;  // baseline; run_into grows to true cap
        impl_->emit_weights.assign(dims_.n_freq * 1 * dims_.n_ant,
                                   Cfloat{0.0F, 0.0F});
        impl_->G_threads_stride = G_cap * G_cap;
        impl_->P_local_all.assign(max_nt * impl_->G_threads_stride, 0.0F);
        impl_->Pcand_local_all.assign(max_nt * 9, 0.0F);
    }
    // Per-subarray accumulation scratch (per thread): cap L_max * M_eff^2 per
    // folder. Grown lazily at run_into entry to `n_threads` folders so each
    // OpenMP worker in the parallel frequency loop gets its own slab.
    {
        const std::size_t L_max = dims_.n_ant - M_eff + 1;
        impl_->per_sub_scratch.assign(L_max * M_eff * M_eff,
                                      Cfloat{0.0F, 0.0F});
    }

    // Phase 3: precompute the O6 relative-grid phasor tables (off the hot
    // path) when a scan is requested AND the aperture is coplanar (every
    // antenna z within float eps of 0). For a planar array the steering phase
    // `phi_a = k(f)*(p_x*l + p_y*m + p_z*n(l,m))` collapses to a function LINEAR
    // in (l,m) (p_z==0 kills the n term), so `phi_a(u0+du)=phi_a(u0)+phi_a(du)`
    // exactly and each candidate's steering vector is `centre * rel` with no
    // per-cell trig. A non-coplanar aperture keeps the original `steer_one` path
    // (`phasor_fast_path` stays false) — numerically identical to before.
    const bool scan_enabled = config_.coarse_grid_resolution > 1;
    if (scan_enabled) {
        impl_->coarse_cells_per_axis = config_.coarse_grid_resolution;
        const std::size_t G = impl_->coarse_cells_per_axis;

        // Coplanarity guard: the fast phase-decomposition path is exact iff no
        // antenna has a meaningful z component (otherwise the nonlinear
        // n=sqrt(1-l*l-m*m) term contributes and per-offset tabulation would
        // not reproduce steer_one to float tolerance). Use the first M_eff
        // antennas — exactly those the sub-array aperture steers with.
        const float coplanar_eps = 1.0e-6F;
        bool coplanar = true;
        for (std::size_t a = 0; a < M_eff; ++a) {
            if (std::fabs(positions_m_[a][2]) > coplanar_eps) {
                coplanar = false;
                break;
            }
        }
        impl_->phasor_fast_path = coplanar;

        // Cache the planar position components used by both the table build
        // (here, off-hot-path) and the centre-phasor recompute (per window).
        impl_->pos_x.assign(M_eff, 0.0F);
        impl_->pos_y.assign(M_eff, 0.0F);
        for (std::size_t a = 0; a < M_eff; ++a) {
            impl_->pos_x[a] = positions_m_[a][0];
            impl_->pos_y[a] = positions_m_[a][1];
        }

        if (coplanar) {
            const double two_pi_over_c = two_pi / speed_of_light_m_per_s;

            // --- Coarse relative phasors: rel_coarse[cell][f][a] ---
            // `du_l[u] = (u - (G-1)/2) * step_l`, `step_l = 2*fov_l/(G-1)`; same
            // for m. The coarse grid centres on the prior, so the centre cell
            // offset is exactly 0 => rel phasor +1 (matches a == centre there).
            const double step_l =
                (G == 1) ? 0.0 : (2.0 * static_cast<double>(config_.search_fov_l))
                                 / static_cast<double>(G - 1);
            const double step_m =
                (G == 1) ? 0.0 : (2.0 * static_cast<double>(config_.search_fov_m))
                                 / static_cast<double>(G - 1);
            const double mid =
                static_cast<double>(G - 1) * 0.5;  // fractional centre index
            impl_->rel_coarse.assign(G * G * dims_.n_freq * M_eff,
                                     Cfloat{0.0F, 0.0F});
            for (std::size_t v = 0; v < G; ++v) {
                const double du_m = (static_cast<double>(v) - mid) * step_m;
                for (std::size_t u = 0; u < G; ++u) {
                    const double du_l =
                        (static_cast<double>(u) - mid) * step_l;
                    const std::size_t cell = v * G + u;
                    Cfloat* __restrict cell_base =
                        impl_->rel_coarse.data()
                        + cell * dims_.n_freq * M_eff;
                    for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                        const double k =
                            two_pi_over_c
                            * static_cast<double>(frequencies_hz_[f]);
                        Cfloat* __restrict f_base =
                            cell_base + f * M_eff;
                        for (std::size_t a = 0; a < M_eff; ++a) {
                            const double phi =
                                k * (static_cast<double>(impl_->pos_x[a]) * du_l
                                     + static_cast<double>(impl_->pos_y[a])
                                       * du_m);
                            // steer_one sign convention: imag = -sin.
                            f_base[a] = Cfloat{
                                static_cast<float>(std::cos(phi)),
                                static_cast<float>(-std::sin(phi))};
                        }
                    }
                }
            }

            // --- Refinement relative phasors: rel_refine[L][o][f][a] ---
            // Level L (1-indexed) half-cell pitch `half = (fov/G) / 2^L`. The
            // 3x3 candidate offsets are `(du*half, dv*half)` with o=(dv+1)*3+(du+1).
            const std::size_t L_refine = config_.refinement_levels;
            impl_->rel_refine.assign(L_refine, {});
            impl_->refine_half_l.assign(L_refine, 0.0F);
            impl_->refine_half_m.assign(L_refine, 0.0F);
            const float base_half_l = config_.search_fov_l / static_cast<float>(G);
            const float base_half_m = config_.search_fov_m / static_cast<float>(G);
            for (std::size_t L = 1; L <= L_refine; ++L) {
                const float half_l =
                    base_half_l / static_cast<float>(1u << L);  // /2^L
                const float half_m =
                    base_half_m / static_cast<float>(1u << L);
                impl_->refine_half_l[L - 1] = half_l;
                impl_->refine_half_m[L - 1] = half_m;
                std::vector<Cfloat>& tbl = impl_->rel_refine[L - 1];
                tbl.assign(9 * dims_.n_freq * M_eff, Cfloat{0.0F, 0.0F});
                for (std::int64_t dv = -1; dv <= 1; ++dv) {
                    for (std::int64_t du = -1; du <= 1; ++du) {
                        const std::size_t o =
                            static_cast<std::size_t>((dv + 1) * 3 + (du + 1));
                        const double du_l =
                            static_cast<double>(du) * static_cast<double>(half_l);
                        const double du_m =
                            static_cast<double>(dv) * static_cast<double>(half_m);
                        Cfloat* __restrict o_base =
                            tbl.data() + o * dims_.n_freq * M_eff;
                        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                            const double k =
                                two_pi_over_c
                                * static_cast<double>(frequencies_hz_[f]);
                            Cfloat* __restrict f_base = o_base + f * M_eff;
                            for (std::size_t a = 0; a < M_eff; ++a) {
                                const double phi =
                                    k
                                    * (static_cast<double>(impl_->pos_x[a]) * du_l
                                       + static_cast<double>(impl_->pos_y[a])
                                         * du_m);
                                f_base[a] = Cfloat{
                                    static_cast<float>(std::cos(phi)),
                                    static_cast<float>(-std::sin(phi))};
                            }
                        }
                    }
                }
            }

            // Per-window centre-phasor scratch (sized once; written per window).
            impl_->centre_phasor.assign(dims_.n_freq * M_eff,
                                        Cfloat{0.0F, 0.0F});
        }
        // (The legacy `coarse_table` slot is left empty — Phase 3 supersedes it.
        // It remains declared for ABI/back-compat with any external reader.)
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

#if defined(BEAMFORMER_TRACKER_DEBUG)
// ---------------------------------------------------------------------
// debug_search_dump — write everything needed to diagnose a failing
// DOA-recovery assertion to a self-describing directory. Enabled only
// in debug builds (CMake passes -DBEAMFORMER_TRACKER_DEBUG to the test
// target); a no-op stub below keeps the symbol resolvable in production.
// ---------------------------------------------------------------------
namespace {

void ensure_dir(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
}

std::ofstream open_text(const std::filesystem::path& p) {
    std::ofstream out(p);
    if (!out) {
        throw std::runtime_error("debug_search_dump cannot open " + p.string());
    }
    out << std::fixed << std::setprecision(9);
    return out;  // NRVO
}

std::ofstream open_bin(const std::filesystem::path& p) {
    std::ofstream out(p, std::ios::binary);
    if (!out) {
        throw std::runtime_error("debug_search_dump cannot open " + p.string());
    }
    return out;
}

// Write a flat float vector as raw little-endian float32 (the project is
// x86/Linux on Trillium), with a one-line count comment is not possible in
// binary — instead we rely on the README listing the shape.
void write_floats_bin(const std::filesystem::path& p,
                      const float* data, std::size_t n) {
    auto out = open_bin(p);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(n * sizeof(float)));
}

void write_cfloats_bin(const std::filesystem::path& p,
                       const std::vector<std::complex<float>>& data) {
    auto out = open_bin(p);
    // std::complex<float> is layout-compatible with two adjacent floats.
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(
                  data.size() * sizeof(std::complex<float>)));
}

}  // namespace

void CpuOptBeamTracker::debug_search_dump(const char* dir,
                                            const char* extra) const {
    // Unique directory so repeated runs don't clobber: <dir>/<extra>_<timestamp>
    std::ostringstream nm;
    nm << (extra && *extra ? extra : "dump");
    nm << "_" << std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::path(dir) / nm.str();
    ensure_dir(root);

    // ---- README ---------------------------------------------------------
    {
        auto out = open_text(root / "README.txt");
        out << "CpuOptBeamTracker debug search dump\n";
        out << "==================================\n\n";
        out << "Build: BEAMFORMER_TRACKER_DEBUG defined.\n";
        out << "n_time=" << dims_.n_time << " n_freq=" << dims_.n_freq
            << " n_ant=" << dims_.n_ant << " n_beams=" << dims_.n_beams
            << " M_eff=" << impl_->M_eff << '\n';
        out << "config: estimator="
            << (config_.estimator == TrackerEstimator::Capon ? "capon"
                                                              : "bartlett")
            << " coarse_grid_resolution=" << config_.coarse_grid_resolution
            << " refinement_levels=" << config_.refinement_levels
            << " search_fov_l=" << config_.search_fov_l
            << " search_fov_m=" << config_.search_fov_m
            << " forgetting_factor=" << config_.forgetting_factor
            << " diagonal_load=" << config_.diagonal_load
            << " smoothing_subarray="
            << config_.spatial_smoothing_subarray_size
            << " enable_quad_interp="
            << (config_.enable_quadratic_peak_interp ? 1 : 0)
            << " integration_spectra=" << config_.integration_spectra << '\n';
        out << "seeded_prior: start=(" << impl_->dbg_seeded_prior.direction_start[0]
            << ',' << impl_->dbg_seeded_prior.direction_start[1] << ','
            << impl_->dbg_seeded_prior.direction_start[2] << ") rate=("
            << impl_->dbg_seeded_prior.direction_rate_per_sample[0] << ','
            << impl_->dbg_seeded_prior.direction_rate_per_sample[1] << ")\n";
        out << "windows captured=" << impl_->dbg_final_estimate.size() << '\n';
        out << "\nFile layout:\n";
        out << "  positions.csv         : array geometry, one row per ant (x,y,z).\n";
        out << "  frequencies.csv       : channel frequencies, one row.\n";
        out << "  estimates.csv         : per-window prior centre + final estimate.\n";
        out << "  coarse_<w>.csv        : per-window coarse grid cell dirs + integrated power (row-major GxG).\n";
        out << "  refine_<w>_<lvl>.csv  : per-window per-level 3x3 candidate dirs + integrated power.\n";
        out << "  R_<w>_<f>.bin         : per-window per-frequency M_eff^2 Hermitian covariance (complex<float>, row-major).\n";
        out << "  snaps_<w>_<f>.bin     : per-window per-frequency K length-n_ant decoded snapshots (complex<float>).\n";
        out << "  config.txt            : same config block as this README, machine-friendly.\n";
    }
    {
        auto out = open_text(root / "config.txt");
        out << "estimator "
            << (config_.estimator == TrackerEstimator::Capon ? "capon"
                                                              : "bartlett") << '\n';
        out << "coarse_grid_resolution " << config_.coarse_grid_resolution << '\n';
        out << "refinement_levels " << config_.refinement_levels << '\n';
        out << "search_fov_l " << config_.search_fov_l << '\n';
        out << "search_fov_m " << config_.search_fov_m << '\n';
        out << "forgetting_factor " << config_.forgetting_factor << '\n';
        out << "diagonal_load " << config_.diagonal_load << '\n';
        out << "spatial_smoothing_subarray_size "
            << config_.spatial_smoothing_subarray_size << '\n';
        out << "enable_quadratic_peak_interp "
            << (config_.enable_quadratic_peak_interp ? 1 : 0) << '\n';
        out << "integration_spectra " << config_.integration_spectra << '\n';
        out << "n_time " << dims_.n_time << '\n';
        out << "n_freq " << dims_.n_freq << '\n';
        out << "n_ant " << dims_.n_ant << '\n';
        out << "M_eff " << impl_->M_eff << '\n';
    }

    // ---- positions / frequencies ---------------------------------------
    {
        auto out = open_text(root / "positions.csv");
        for (std::size_t a = 0; a < positions_m_.size(); ++a) {
            out << positions_m_[a][0] << ',' << positions_m_[a][1] << ','
                << positions_m_[a][2] << '\n';
        }
    }
    {
        auto out = open_text(root / "frequencies.csv");
        for (std::size_t f = 0; f < frequencies_hz_.size(); ++f) {
            out << frequencies_hz_[f] << '\n';
        }
    }

    // ---- estimates ------------------------------------------------------
    {
        auto out = open_text(root / "estimates.csv");
        out << "window,prior_l,prior_m,prior_n,est_l,est_m,est_n\n";
        const std::size_t W = impl_->dbg_final_estimate.size();
        for (std::size_t w = 0; w < W; ++w) {
            const Vec3& pc = (w < impl_->dbg_prior_centre.size())
                                 ? impl_->dbg_prior_centre[w]
                                 : Vec3{0, 0, 1};
            const Vec3& est = impl_->dbg_final_estimate[w];
            out << w << ',' << pc[0] << ',' << pc[1] << ',' << pc[2] << ','
                << est[0] << ',' << est[1] << ',' << est[2] << '\n';
        }
    }

    // ---- coarse spectra (dirs + power) ----------------------------------
    for (std::size_t w = 0; w < impl_->dbg_coarse_centres.size(); ++w) {
        std::ostringstream nm2;
        nm2 << "coarse_" << w << ".csv";
        auto out = open_text(root / nm2.str());
        out << "v,u,l,m,n,power\n";
        const auto& cells = impl_->dbg_coarse_centres[w];
        const auto& power = impl_->dbg_coarse_power[w];
        const std::size_t G = config_.coarse_grid_resolution;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            const std::size_t u = i % G;
            const std::size_t v = i / G;
            const float pw = (i < power.size()) ? power[i] : 0.0F;
            out << v << ',' << u << ',' << cells[i][0] << ',' << cells[i][1]
                << ',' << cells[i][2] << ',' << pw << '\n';
        }
    }

    // ---- refinement spectra ---------------------------------------------
    for (std::size_t w = 0; w < impl_->dbg_refine_cands.size(); ++w) {
        for (std::size_t lvl = 0; lvl < impl_->dbg_refine_cands[w].size(); ++lvl) {
            std::ostringstream nm2;
            nm2 << "refine_" << w << '_' << lvl << ".csv";
            auto out = open_text(root / nm2.str());
            out << "idx,dv,du,l,m,n,power\n";
            const auto& cand = impl_->dbg_refine_cands[w][lvl];
            const auto& pw = impl_->dbg_refine_power[w][lvl];
            for (std::size_t i = 0; i < cand.size(); ++i) {
                const std::int64_t dv = static_cast<std::int64_t>(i / 3) - 1;
                const std::int64_t du = static_cast<std::int64_t>(i % 3) - 1;
                const float p = (i < pw.size()) ? pw[i] : 0.0F;
                out << i << ',' << dv << ',' << du << ',' << cand[i][0] << ','
                    << cand[i][1] << ',' << cand[i][2] << ',' << p << '\n';
            }
        }
    }

    // ---- covariances + snapshots (binary) -------------------------------
    const std::size_t M = impl_->M_eff;
    for (std::size_t w = 0; w < impl_->dbg_R_freq_per_window.size(); ++w) {
        for (std::size_t f = 0; f < impl_->dbg_R_freq_per_window[w].size(); ++f) {
            {
                std::ostringstream nm2;
                nm2 << "R_" << w << '_' << f << ".bin";
                write_cfloats_bin(root / nm2.str(),
                                  impl_->dbg_R_freq_per_window[w][f]);
            }
            if (w < impl_->dbg_snapshots.size()
                && f < impl_->dbg_snapshots[w].size()) {
                std::ostringstream nm2;
                nm2 << "snaps_" << w << '_' << f << ".bin";
                // Write K snapshots of length n_ant, concatenated complex<float>.
                const auto& snaps = impl_->dbg_snapshots[w][f];
                std::vector<std::complex<float>> flat;
                flat.reserve(snaps.size() * (snaps.empty() ? 0 : snaps[0].size()));
                for (const auto& s : snaps) {
                    for (const auto& c : s) flat.push_back(c);
                }
                write_cfloats_bin(root / nm2.str(), flat);
            }
        }
        // Per-window shape manifest so binary files are interpretable.
        std::ostringstream nm2;
        nm2 << "shapes_" << w << ".txt";
        auto out = open_text(root / nm2.str());
        out << "window " << w << '\n';
        out << "M_eff " << M << '\n';
        out << "R_nfreq " << impl_->dbg_R_freq_per_window[w].size() << '\n';
        if (w < impl_->dbg_snapshots.size()) {
            for (std::size_t f = 0; f < impl_->dbg_snapshots[w].size(); ++f) {
                out << "snaps_n_freq" << f << ' '
                    << impl_->dbg_snapshots[w][f].size();
                if (!impl_->dbg_snapshots[w][f].empty()) {
                    out << ' ' << impl_->dbg_snapshots[w][f][0].size();
                }
                out << '\n';
            }
        }
    }

    // Emit an obvious marker so a grep across the dump is easy.
    auto done = open_text(root / "DUMP_COMPLETE");
    done << "ok\n";
}

#else  // !BEAMFORMER_TRACKER_DEBUG
void CpuOptBeamTracker::debug_search_dump(const char* /*dir*/,
                                            const char* /*extra*/) const {
    // Production build stub: no capture buffers exist (impl_->dbg_* members
    // are compiled out), so there is nothing to write. Keeping the symbol
    // resolvable means callers don't need their own macro guards.
}
#endif  // BEAMFORMER_TRACKER_DEBUG

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
    impl_->reset_dbg();
#if defined(BEAMFORMER_TRACKER_DEBUG)
    impl_->dbg_seeded_prior = prior;
    impl_->dbg_prior_captured = true;
    impl_->dbg_final_estimate.assign(window_count, Vec3{0.0F, 0.0F, 1.0F});
#endif

    const bool scan_enabled = config_.coarse_grid_resolution > 1;
    const float lambda = config_.forgetting_factor;
    const std::size_t M_eff = impl_->M_eff;  // full n_ant, or sub-array size if
                                             // O2 smoothing was enabled in the
                                             // constructor (smoothing decision
                                             // is baked into M_eff there).

#if defined(BEAMFORMER_TRACKER_PERF)
    impl_->window_ms.reserve(window_count);
    using PerfClock = std::chrono::steady_clock;
#define BEAMFORMER_TRACKER_PERF_START(name) \
        auto name##_start = PerfClock::now()
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

    // ---- Per-thread scratch warm-up (once per run_into, not per window): the
    // parallel frequency loops index thread-private steer / capon-RHS slabs by
    // `omp_get_thread_num()`, so they must be sized to the current OpenMP cap
    // before the first window. Cheap (one resize if the cap grew) and done off
    // the per-window hot path.
#if BEAMFORMER_TRACKER_OMP_ENABLED
    {
        const int nthread = omp_get_max_threads();
        if (nthread > 0) impl_->n_threads = nthread;
    }
#else
    impl_->n_threads = 1;
#endif
    {
        const std::size_t nt =
            static_cast<std::size_t>(std::max<int>(impl_->n_threads, 1));
        if (impl_->steer_scratch.size() < nt * M_eff)
            impl_->steer_scratch.assign(nt * M_eff, Cfloat{0.0F, 0.0F});
        if (impl_->capon_b_scratch.size() < nt * M_eff)
            impl_->capon_b_scratch.assign(nt * M_eff, Cfloat{0.0F, 0.0F});
        const std::size_t L_max = dims_.n_ant - M_eff + 1;
        const std::size_t per_folder = L_max * M_eff * M_eff;
        if (impl_->per_sub_scratch.size() < nt * per_folder)
            impl_->per_sub_scratch.assign(nt * per_folder, Cfloat{0.0F, 0.0F});
        // Phase 4 persistent per-thread power accumulators: G_max*G_max
        // (coarse) and 9 (refine) floats per thread. Sized once per run_into
        // so the per-window scan loops zero a slab slice instead of
        // heap-allocating std::vector<std::vector<float>> / array<float,9> per
        // thread per window (the original allocation alone was a measurable
        // fraction of the per-frame cost at high thread counts).
        const std::size_t Gg = impl_->G_max * impl_->G_max;
        if (impl_->G_threads_stride < Gg) impl_->G_threads_stride = Gg;
        const std::size_t need_P = nt * impl_->G_threads_stride;
        if (impl_->P_local_all.size() < need_P)
            impl_->P_local_all.assign(need_P, 0.0F);
        if (impl_->Pcand_local_all.size() < nt * 9)
            impl_->Pcand_local_all.assign(nt * 9, 0.0F);
    }

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
        const bool capon = (config_.estimator == TrackerEstimator::Capon);
        const std::size_t snap_stride = dims_.n_ant;
        const std::size_t cov_slot = M_eff * M_eff;
        const std::size_t snap_slot = impl_->K_max * snap_stride;
        const std::size_t L_max = dims_.n_ant - M_eff + 1;
        const std::size_t per_folder = L_max * cov_slot;
        const bool first_window = !impl_->R_initialised;
        const bool lambda_one = (lambda >= 1.0F);
        Cfloat* __restrict Rf_block_base = impl_->R_block_freq.data();
        Cfloat* __restrict snap_buf_base = impl_->snapshot_buffer.data();
        Cfloat* __restrict per_sub_base = impl_->per_sub_scratch.data();

#if defined(BEAMFORMER_TRACKER_DEBUG)
        // Reserve debug per-window containers on the first window.
        if (window == 0) {
            impl_->dbg_snapshots.assign(window_count,
                std::vector<std::vector<std::vector<Cfloat>>>(dims_.n_freq));
            impl_->dbg_R_freq_per_window.assign(window_count,
                std::vector<std::vector<Cfloat>>(dims_.n_freq));
        }
        impl_->dbg_prior_centre.push_back(prev_estimate);
        std::vector<std::vector<std::vector<Cfloat>>>& dbg_w_snaps =
            impl_->dbg_snapshots[window];
        std::vector<std::vector<Cfloat>>& dbg_w_R =
            impl_->dbg_R_freq_per_window[window];
        // Debug builds keep the serial covariance path so the per-frequency
        // snapshot capture (which heap-allocates vectors) is preserved verbatim.
        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
            Cfloat* __restrict snap_flat = snap_buf_base + f * snap_slot;
            for (std::size_t t = first_time; t < last_time; ++t) {
                decode_snapshot_into(packed, dims_, t, f,
                                     snap_flat + (t - first_time) * snap_stride);
            }
            Cfloat* __restrict R_block = Rf_block_base + f * cov_slot;
            spatial_smoothed_covariance_into(
                snap_flat, K, dims_.n_ant, snap_stride,
                config_.spatial_smoothing_subarray_size, M_eff,
                R_block, per_sub_base, per_folder);
            if (capon) {
                apply_diagonal_load_into(R_block, M_eff, config_.diagonal_load);
            }
            std::vector<Cfloat>& R = impl_->R_freq[f];
            if (first_window || lambda_one) {
                for (std::size_t i = 0; i < cov_slot; ++i) R[i] = R_block[i];
            } else {
                for (std::size_t i = 0; i < cov_slot; ++i) {
                    R[i] = lambda * R[i] + a_comp * R_block[i];
                }
            }
            dbg_w_R[f] = impl_->R_freq[f];
            dbg_w_snaps[f].assign(K);
            for (std::size_t k = 0; k < K; ++k) {
                dbg_w_snaps[f][k].assign(
                    snap_flat + k * snap_stride,
                    snap_flat + k * snap_stride + snap_stride);
            }
        }
        impl_->R_initialised = true;
        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
            const std::vector<Cfloat>& R = impl_->R_freq[f];
            float tr = 0.0F;
            for (std::size_t i = 0; i < M_eff; ++i) tr += R[i * M_eff + i].real();
            impl_->R_trace[f] = tr;
        }
        if (capon) {
            for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                const std::vector<Cfloat>& R = impl_->R_freq[f];
                Cfloat* __restrict Lf = impl_->L_factors.data() + f * cov_slot;
                std::copy(R.data(), R.data() + cov_slot, Lf);
                impl_->capon_factor_ok[f] =
                    cholesky_factorize(Lf, M_eff) ? 1 : 0;
            }
        }
#else
        // ================================================================
        // Parallel covariance formation + RLS fold (fix-plan defect #1/#2):
        // the N_freq channels are independent, so this — the dominant per-
        // frame cost (~390 ms / frame single-threaded) — is parallelized
        // across frequencies. Each thread decodes its K snapshots into a
        // private `snapshot_buffer` slice, builds the Hermitian-symmetric
        // smoothed covariance into its `R_block_freq` slot, applies the Capon
        // diagonal load (Capon only), then folds with the previous window's
        // R_freq[f] (RLS) and caches trace(R_freq[f]) for the Bartlett scan.
        // All indexed scratch (`snapshot_buffer`, `R_block_freq`, `R_freq`,
        // `R_trace`) is per-frequency-distinct → no reduction needed; the
        // per-sub smoothing slab is keyed by `omp_get_thread_num()` so two
        // threads never touch the same sub-array accumulators.
        // ================================================================
        const std::size_t n_freq = dims_.n_freq;
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t fi = 0;
             fi < static_cast<std::ptrdiff_t>(n_freq); ++fi) {
            const std::size_t f = static_cast<std::size_t>(fi);
            Cfloat* __restrict snap_flat = snap_buf_base + f * snap_slot;
            for (std::size_t t = first_time; t < last_time; ++t) {
                decode_snapshot_into(packed, dims_, t, f,
                    snap_flat + (t - first_time) * snap_stride);
            }
            Cfloat* __restrict R_block = Rf_block_base + f * cov_slot;
            // Per-thread smoothing slab (only the smoothing path touches it;
            // the common M_eff == n_ant path accumulates straight into
            // R_block and ignores the pointer).
#if BEAMFORMER_TRACKER_OMP_ENABLED
            Cfloat* __restrict sub_slab =
                per_sub_base + static_cast<std::size_t>(omp_get_thread_num())
                                   * per_folder;
#else
            Cfloat* __restrict sub_slab = per_sub_base;
#endif
            spatial_smoothed_covariance_into(
                snap_flat, K, dims_.n_ant, snap_stride,
                config_.spatial_smoothing_subarray_size, M_eff,
                R_block, sub_slab, per_folder);
            if (capon) {
                apply_diagonal_load_into(R_block, M_eff, config_.diagonal_load);
            }
            const Cfloat* __restrict R_block_ro = R_block;
            std::vector<Cfloat>& R = impl_->R_freq[f];
            if (first_window || lambda_one) {
                for (std::size_t i = 0; i < cov_slot; ++i) {
                    R[i] = R_block_ro[i];
                }
            } else {
                for (std::size_t i = 0; i < cov_slot; ++i) {
                    R[i] = lambda * R[i] + a_comp * R_block_ro[i];
                }
            }
            // trace(R_freq[f]) — the diagonal sum half of a^H R a Bartlett
            // power, constant across all search cells of this frequency.
            float tr = 0.0F;
            for (std::size_t i = 0; i < M_eff; ++i) {
                tr += R[i * M_eff + i].real();
            }
            impl_->R_trace[f] = tr;
        }
        impl_->R_initialised = true;

        // ---- Phase 2 (FOSM): factorize each per-frequency Capon covariance
        // exactly ONCE for this window, parallel across frequencies. The lower-
        // triangular L stays in the per-frequency `L_factors[f]` slot (distinct
        // per f — safe under the reduction-free parallel loop). Bartlett never
        // factorizes (no inverse to amortize).
        if (capon) {
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t fi = 0;
                 fi < static_cast<std::ptrdiff_t>(n_freq); ++fi) {
                const std::size_t f = static_cast<std::size_t>(fi);
                const std::vector<Cfloat>& R = impl_->R_freq[f];
                Cfloat* __restrict Lf = impl_->L_factors.data() + f * cov_slot;
                std::copy(R.data(), R.data() + cov_slot, Lf);
                impl_->capon_factor_ok[f] =
                    cholesky_factorize(Lf, M_eff) ? 1 : 0;
            }
        }
#endif  // !BEAMFORMER_TRACKER_DEBUG

        // ---- O3 + O6: hierarchical coarse-to-fine search.
        //   Skolnik, Radar Handbook §7.11 (two-stage search);
        //   Haykin & Reilly 1991 (hierarchical ML lattice search).
        // Level 0: build the coarse grid centred on prev_estimate (the prior).
        std::vector<Vec3> coarse = build_centred_grid(prev_estimate, G, fov_l, fov_m);
        // Per-frequency coarse power summed across frequency for the
        // per-window direction decision (spec O1: "integrated across
        // frequency for the final per-window direction decision").
        // Reuse the preallocated `power_grid` slice (G*G); zero it once here.
        float* __restrict P_coarse = impl_->power_grid.data();
        for (std::size_t i = 0; i < G * G; ++i) P_coarse[i] = 0.0F;
        // Per-thread steering + Capon-RHS scratch: the parallel coarse scan
        // indexes thread-private slabs of the contiguous `steer_scratch` /
        // `capon_b_scratch` buffers by `omp_get_thread_num()`. None allocated
        // per cell.
        Cfloat* __restrict steer_buf_base = impl_->steer_scratch.data();
        Cfloat* __restrict capon_b_base = impl_->capon_b_scratch.data();
        // Per-thread coarse-power accumulator. OpenMP array-section reductions
        // are not portable C, so each thread accumulates into its own private
        // `P_local` (sized G*G); the reduction across threads is performed
        // serially after the parallel loop. G*G is small (≤144) so the serial
        // sum is negligible.
        const std::size_t nt_scan =
            static_cast<std::size_t>(std::max<int>(impl_->n_threads, 1));
        std::vector<std::vector<float>> P_local_per_thread(
            nt_scan, std::vector<float>(G * G, 0.0F));

        // ---- Phase 3 steering-vector preload (trig kill for coplanar arrays).
        // The relative phasor table `rel_coarse` was built in the Impl ctor
        // around the grid's TRUE geometric centre index `mid = (G-1)/2` (kept
        // fractional so it works for even G, where no integer cell sits at the
        // prior). The natural anchor for the additive decomposition is therefore
        // the prior `prev_estimate` itself (its (l,m) sit exactly at the grid
        // range centre by construction of build_centred_grid), NOT any grid
        // cell — for even G there is no centre cell. So:
        //   a(cell)[a] = centre_phasor(f,a) * rel_coarse(cell,f,a)
        // with `centre_phasor(f,a) = exp(-j*k(f)*(px*prev_l + py*prev_m))`,
        // computed ONCE per window per frequency (n_freq*M_eff trig). Each
        // candidate cell drops from 2 trig/antenna to 1 complex multiply/antenna.
        // `cell_clamped` flags cells whose nominal lattice point lay outside
        // the unit disk (so build_centred_grid horizon-clamped it — the actual
        // coarse[cell] then no longer matches the precomputed lattice phasor,
        // and the cell uses the exact steer_one fallback; the tested FoV never
        // clamps so this branch is dead in practice).
        const bool phasor_fast = impl_->phasor_fast_path;
        const float centre_l = phasor_fast ? prev_estimate[0] : 0.0F;
        const float centre_m = phasor_fast ? prev_estimate[1] : 0.0F;
        std::vector<char> cell_clamped;
        const double step_l =
            phasor_fast ? ((G == 1) ? 0.0
                                    : (2.0 * static_cast<double>(fov_l))
                                      / static_cast<double>(G - 1))
                        : 0.0;
        const double step_m =
            phasor_fast ? ((G == 1) ? 0.0
                                    : (2.0 * static_cast<double>(fov_m))
                                      / static_cast<double>(G - 1))
                        : 0.0;
        const double mid_idx = static_cast<double>(G - 1) * 0.5;
        if (phasor_fast) {
            cell_clamped.assign(G * G, 0);
            for (std::size_t v = 0; v < G; ++v) {
                for (std::size_t u = 0; u < G; ++u) {
                    const double nl =
                        centre_l + (static_cast<double>(u) - mid_idx) * step_l;
                    const double nm =
                        centre_m + (static_cast<double>(v) - mid_idx) * step_m;
                    cell_clamped[v * G + u] =
                        (nl * nl + nm * nm > 0.999) ? 1 : 0;
                }
            }
            // Compute the centre phasor once per (freq, antenna) — the only
            // per-window trig on this path.
            const double two_pi_over_c = two_pi / speed_of_light_m_per_s;
            const float* __restrict px = impl_->pos_x.data();
            const float* __restrict py = impl_->pos_y.data();
            Cfloat* __restrict cp = impl_->centre_phasor.data();
            for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                const double k =
                    two_pi_over_c * static_cast<double>(frequencies_hz_[f]);
                Cfloat* __restrict cpf = cp + f * M_eff;
                for (std::size_t a = 0; a < M_eff; ++a) {
                    const double phi =
                        k * (static_cast<double>(px[a]) * centre_l
                             + static_cast<double>(py[a]) * centre_m);
                    cpf[a] = Cfloat{static_cast<float>(std::cos(phi)),
                                    static_cast<float>(-std::sin(phi))};
                }
            }
        }
#if defined(BEAMFORMER_TRACKER_DEBUG)
        std::vector<float> P_coarse_dbg(G * G, 0.0F);
        impl_->dbg_coarse_centres.push_back(coarse);
        impl_->dbg_coarse_power.push_back(P_coarse_dbg);  // will fill below
        std::vector<float>& dbg_P_coarse = impl_->dbg_coarse_power.back();
        impl_->dbg_refine_cands.emplace_back();
        impl_->dbg_refine_power.emplace_back();
#endif
        // ----------------------------------------------------------------
        // Coarse grid scan, parallel across frequencies (fix-plan defect #6):
        // the N_freq channels are independent and each thread accumulates its
        // G*G coarse powers into a private `P_local`, summed into `P_coarse`
        // after the parallel region. Each thread builds its steering vector
        // into a private `steer_buf` slab and reuses a private `capon_b` slot,
        // so no cross-thread aliasing occurs. The Bartlett branch uses the
        // trace-precomputed Hermitian quadratic form (halves the per-cell FMAs;
        // the diagonal `trace(R_freq[f])` is reused across all G*G cells of a
        // frequency instead of recomputed per cell).
        // ----------------------------------------------------------------
#if !defined(BEAMFORMER_TRACKER_DEBUG)
        {
            const std::size_t coarse_cells = G * G;
            const std::size_t cell_freq_stride = dims_.n_freq * M_eff;
            const Cfloat* __restrict rel_coarse_base =
                phasor_fast ? impl_->rel_coarse.data() : nullptr;
            const Cfloat* __restrict centre_phasor_base =
                phasor_fast ? impl_->centre_phasor.data() : nullptr;
            const Cfloat* __restrict Lf_base = impl_->L_factors.data();
            const float* __restrict R_trace_base = impl_->R_trace.data();
            const char* __restrict factor_ok_base = impl_->capon_factor_ok.data();
            const std::vector<Vec3>& positions = positions_m_;
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
            for (std::ptrdiff_t fi = 0;
                 fi < static_cast<std::ptrdiff_t>(dims_.n_freq); ++fi) {
                const std::size_t f = static_cast<std::size_t>(fi);
#if BEAMFORMER_TRACKER_OMP_ENABLED
                const std::size_t tid =
                    static_cast<std::size_t>(omp_get_thread_num());
#else
                const std::size_t tid = 0;
#endif
                Cfloat* __restrict steer_buf = steer_buf_base + tid * M_eff;
                Cfloat* __restrict capon_b = capon_b_base + tid * M_eff;
                float* __restrict P_local =
                    P_local_per_thread[tid].data();
                const Cfloat* __restrict R_f = impl_->R_freq[f].data();
                const float trace_R = R_trace_base[f];
                const double wave_number =
                    two_pi * static_cast<double>(frequencies_hz_[f])
                    / speed_of_light_m_per_s;
                const Cfloat* __restrict Lf = Lf_base + f * M_eff * M_eff;
                Cfloat* __restrict work = const_cast<Cfloat*>(Lf);
                const bool factor_ok = factor_ok_base[f] != 0;
                const Cfloat* __restrict cpf =
                    phasor_fast ? (centre_phasor_base + f * M_eff) : nullptr;
                const std::size_t f_M = f * M_eff;
                for (std::size_t cell = 0; cell < coarse_cells; ++cell) {
                    if (phasor_fast && !cell_clamped[cell]) {
                        const Cfloat* __restrict rel =
                            rel_coarse_base + cell * cell_freq_stride + f_M;
                        for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                            steer_buf[a_idx] = cpf[a_idx] * rel[a_idx];
                        }
                    } else {
                        for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                            steer_buf[a_idx] = steer_one(positions, wave_number,
                                                         coarse[cell], a_idx);
                        }
                    }
                    float p;
                    if (!capon) {
                        // Hermitian + trace-precomputed Bartlett quadratic form
                        // (fix-plan defect #3): a^H R a = trace(R) +
                        // 2·Re(Σ_{r<c} conj(a_r)·R_{rc}·a_c).
                        p = bartlett_power_hermitian_trace(R_f, steer_buf,
                                                            M_eff, trace_R);
                    } else if (factor_ok) {
                        p = capon_power_from_factor(Lf, steer_buf, M_eff,
                                                     capon_b);
                    } else {
                        p = capon_power_into(R_f, steer_buf, M_eff, work,
                                             capon_b);
                    }
                    if (!std::isfinite(p)) p = 0.0F;
                    P_local[cell] += p;
                }
            }
            // Serial reduction: sum the per-thread P_local slabs into P_coarse.
            for (std::size_t t = 0; t < P_local_per_thread.size(); ++t) {
                const std::vector<float>& P_local = P_local_per_thread[t];
                for (std::size_t cell = 0; cell < coarse_cells; ++cell) {
                    P_coarse[cell] += P_local[cell];
                }
            }
        }
#else  // BEAMFORMER_TRACKER_DEBUG
        // Serial coarse scan in debug builds (preserves the captured
        // per-cell spectrum order for the dump writer).
        for (std::size_t f = 0; f < dims_.n_freq; ++f) {
            const Cfloat* __restrict R_f = impl_->R_freq[f].data();
            const float trace_R = impl_->R_trace[f];
            const double wave_number =
                two_pi * static_cast<double>(frequencies_hz_[f])
                / speed_of_light_m_per_s;
            const Cfloat* __restrict Lf = impl_->L_factors.data()
                + f * M_eff * M_eff;
            Cfloat* __restrict work = impl_->L_factors.data()
                + f * M_eff * M_eff;
            const bool factor_ok = impl_->capon_factor_ok[f] != 0;
            const Cfloat* __restrict cpf =
                phasor_fast ? (impl_->centre_phasor.data() + f * M_eff)
                            : nullptr;
            const Cfloat* __restrict rcf =
                phasor_fast ? impl_->rel_coarse.data() : nullptr;
            const std::size_t cell_freq_stride = dims_.n_freq * M_eff;
            Cfloat* __restrict steer_buf = steer_buf_base;
            Cfloat* __restrict capon_b = capon_b_base;
            for (std::size_t cell = 0; cell < G * G; ++cell) {
                if (phasor_fast && !cell_clamped[cell]) {
                    const Cfloat* __restrict rel =
                        rcf + cell * cell_freq_stride + f * M_eff;
                    for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                        steer_buf[a_idx] = cpf[a_idx] * rel[a_idx];
                    }
                } else {
                    for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                        steer_buf[a_idx] = steer_one(positions_m_, wave_number,
                                                     coarse[cell], a_idx);
                    }
                }
                float p;
                if (!capon) {
                    p = bartlett_power_hermitian_trace(R_f, steer_buf,
                                                       M_eff, trace_R);
                } else if (factor_ok) {
                    p = capon_power_from_factor(Lf, steer_buf, M_eff, capon_b);
                } else {
                    p = capon_power_into(R_f, steer_buf, M_eff, work, capon_b);
                }
                if (!std::isfinite(p)) p = 0.0F;
                P_coarse[cell] += p;
            }
        }
#endif  // BEAMFORMER_TRACKER_DEBUG
#if defined(BEAMFORMER_TRACKER_DEBUG)
        // Copy the fully-accumulated coarse power into the debug slot captured
        // above (the slot was pushed as a zero vector; copy the final values).
        for (std::size_t i = 0; i < G * G; ++i) dbg_P_coarse[i] = P_coarse[i];
#endif
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
            // 3×3 neighbourhood candidates around `centre`. Fixed-size stack
            // storage (9 cells) — no per-level heap allocation.
            Vec3 cand[9];
            float Pcand[9] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
            for (std::int64_t dv = -1; dv <= 1; ++dv) {
                for (std::int64_t du = -1; du <= 1; ++du) {
                    const std::size_t idx =
                        static_cast<std::size_t>((dv + 1) * 3 + (du + 1));
                    float l = centre[0] + static_cast<float>(du) * cell_half_l;
                    float m = centre[1] + static_cast<float>(dv) * cell_half_m;
                    const float r2 = l * l + m * m;
                    if (r2 > 0.999F) {
                        const float s = std::sqrt(0.999F / r2);
                        l *= s;
                        m *= s;
                    }
                    cand[idx] = direction_from_lm(l, m);
                }
            }
#if defined(BEAMFORMER_TRACKER_DEBUG)
            // Debug capture copies the stack arrays into the per-window debug
            // vectors (allocates, but only in diagnostic builds).
            impl_->dbg_refine_cands[window].emplace_back(cand, cand + 9);
            impl_->dbg_refine_power[window].emplace_back(Pcand, Pcand + 9);
#endif
            // Phase 3: precompute a per-level parent phasor once per frequency
            // for the running `centre` (the linear decomposition's anchor). The
            // 9 candidates then reuse `parent_phasor * rel_refine[L][o]` — one
            // complex multiply per antenna — instead of 9 trig-constructions.
            // A candidate whose nominal offset left the unit disk (clamped by
            // `direction_from_lm_one` below) falls back to `steer_one` so the
            // result stays bit-identical. The centre's recomputed phasor is the
            // running anchor even when O4 moved `centre` off the lattice.
            const Cfloat* __restrict rel_refine_L = nullptr;
            char cand_clamped[9];
            if (phasor_fast && level - 1 < impl_->rel_refine.size()) {
                rel_refine_L = impl_->rel_refine[level - 1].data();
                const float hl = impl_->refine_half_l[level - 1];
                const float hm = impl_->refine_half_m[level - 1];
                const float cl = centre[0];
                const float cm = centre[1];
                for (std::int64_t dv = -1; dv <= 1; ++dv) {
                    for (std::int64_t du = -1; du <= 1; ++du) {
                        const std::size_t idx =
                            static_cast<std::size_t>((dv + 1) * 3 + (du + 1));
                        const float nl = cl + static_cast<float>(du) * hl;
                        const float nm = cm + static_cast<float>(dv) * hm;
                        cand_clamped[idx] = (nl * nl + nm * nm > 0.999F) ? 1 : 0;
                    }
                }
            } else {
                for (std::size_t i = 0; i < 9; ++i) cand_clamped[i] = 1;
            }
            // Parent phasor = steering vector at the running centre (computed
            // with trig, once per level per freq — uses the centre's actual
            // (l,m) which may be an O4-interpolated off-lattice point). Stored
            // in the reused centre_phasor scratch.
            if (phasor_fast) {
                const double two_pi_over_c = two_pi / speed_of_light_m_per_s;
                const float cl = centre[0];
                const float cm = centre[1];
                const float* __restrict px = impl_->pos_x.data();
                const float* __restrict py = impl_->pos_y.data();
                Cfloat* __restrict cp = impl_->centre_phasor.data();
                for (std::size_t f = 0; f < dims_.n_freq; ++f) {
                    const double k =
                        two_pi_over_c * static_cast<double>(frequencies_hz_[f]);
                    Cfloat* __restrict cpf = cp + f * M_eff;
                    for (std::size_t a = 0; a < M_eff; ++a) {
                        const double phi =
                            k * (static_cast<double>(px[a]) * cl
                                 + static_cast<double>(py[a]) * cm);
                        cpf[a] = Cfloat{static_cast<float>(std::cos(phi)),
                                        static_cast<float>(-std::sin(phi))};
                    }
                }
            }
            // ----------------------------------------------------------------
            // Refinement candidate power, parallel across frequencies (fix-plan
            // defect #7): the N_freq channels are independent, each thread
            // accumulates its 9 candidate powers into a private `Pcand_local`,
            // then reduced serially into `Pcand`. Thread-private steer/capon-b
            // slabs; Bartlett uses the trace-precomputed Hermitian form.
            // ----------------------------------------------------------------
            {
                const std::size_t o_freq_stride = dims_.n_freq * M_eff;
                const Cfloat* __restrict Lf_base = impl_->L_factors.data();
                const Cfloat* __restrict centre_phasor_base =
                    phasor_fast ? impl_->centre_phasor.data() : nullptr;
                const float* __restrict R_trace_base = impl_->R_trace.data();
                const char* __restrict factor_ok_base = impl_->capon_factor_ok.data();
                const std::vector<Vec3>& positions = positions_m_;
                const std::size_t nt_ref =
                    P_local_per_thread.size();
                std::vector<std::array<float, 9>> Pcand_local_per_thread(
                    nt_ref, std::array<float, 9>{});
                for (auto& a : Pcand_local_per_thread) a.fill(0.0F);
#if BEAMFORMER_TRACKER_OMP_ENABLED
#pragma omp parallel for schedule(static)
#endif
                for (std::ptrdiff_t fi = 0;
                     fi < static_cast<std::ptrdiff_t>(dims_.n_freq); ++fi) {
                    const std::size_t f = static_cast<std::size_t>(fi);
#if BEAMFORMER_TRACKER_OMP_ENABLED
                    const std::size_t tid =
                        static_cast<std::size_t>(omp_get_thread_num());
#else
                    const std::size_t tid = 0;
#endif
                    Cfloat* __restrict steer_buf =
                        steer_buf_base + tid * M_eff;
                    Cfloat* __restrict capon_b =
                        capon_b_base + tid * M_eff;
                    float* __restrict Pcand_local =
                        Pcand_local_per_thread[tid].data();
                    const Cfloat* __restrict R_f = impl_->R_freq[f].data();
                    const float trace_R = R_trace_base[f];
                    const double wave_number =
                        two_pi * static_cast<double>(frequencies_hz_[f])
                        / speed_of_light_m_per_s;
                    const Cfloat* __restrict Lf = Lf_base + f * M_eff * M_eff;
                    Cfloat* __restrict work = const_cast<Cfloat*>(Lf);
                    const bool factor_ok = factor_ok_base[f] != 0;
                    const Cfloat* __restrict ppf =
                        phasor_fast ? (centre_phasor_base + f * M_eff)
                                    : nullptr;
                    const std::size_t f_M = f * M_eff;
                    for (std::size_t i = 0; i < 9; ++i) {
                        if (phasor_fast && !cand_clamped[i]) {
                            const Cfloat* __restrict rel =
                                rel_refine_L + i * o_freq_stride + f_M;
                            for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                                steer_buf[a_idx] = ppf[a_idx] * rel[a_idx];
                            }
                        } else {
                            for (std::size_t a_idx = 0; a_idx < M_eff; ++a_idx) {
                                steer_buf[a_idx] =
                                    steer_one(positions, wave_number,
                                              cand[i], a_idx);
                            }
                        }
                        float p;
                        if (!capon) {
                            p = bartlett_power_hermitian_trace(R_f, steer_buf,
                                                               M_eff, trace_R);
                        } else if (factor_ok) {
                            p = capon_power_from_factor(Lf, steer_buf, M_eff,
                                                        capon_b);
                        } else {
                            p = capon_power_into(R_f, steer_buf, M_eff, work,
                                                 capon_b);
                        }
                        if (!std::isfinite(p)) p = 0.0F;
                        Pcand_local[i] += p;
                    }
                }
                // Reduction into the shared stack Pcand[9].
                for (std::size_t t = 0; t < Pcand_local_per_thread.size(); ++t) {
                    const float* Pcand_local = Pcand_local_per_thread[t].data();
                    for (std::size_t i = 0; i < 9; ++i) {
                        Pcand[i] += Pcand_local[i];
                    }
                }
            }
            std::size_t best = 0;
            for (std::size_t i = 1; i < 9; ++i) {
                if (Pcand[i] > Pcand[best]) best = i;
            }
#if defined(BEAMFORMER_TRACKER_DEBUG)
            for (std::size_t i = 0; i < 9; ++i)
                impl_->dbg_refine_power[window].back()[i] = Pcand[i];
#endif
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
#if defined(BEAMFORMER_TRACKER_DEBUG)
        impl_->dbg_final_estimate[window] = centre;
#endif

        // ---- Final emission pass (Phase 4: fused, zero re-decode).
        // The scanning-path emission does NOT carry the byte-equal contract
        // (that contract belongs to the scan-disabled back-compat path, which
        // still uses `emit_window_das`). Here we reuse the snapshot buffer
        // already decoded during covariance formation (snap_buf_base) plus a
        // single beam-0 weight set for `centre` computed once under
        // #pragma omp single on the persistent team. This eliminates the
        // redundant int4 re-decode of the whole time×freq×antenna block and
        // the redundant `generate_weights` trig pass that
        // `emit_window_das_parallel` performed, without changing the per-cell
        // Bartlett power arithmetic (each cell's inner sum is the same serial
        // accumulation over the same float values as `emit_window_das`).
        {
            const auto w_c =
                generate_weights(dims_, positions_m_, frequencies_hz_,
                                 std::vector<Vec3>{centre});
            // generate_weights returns Weights (POD ComplexFloat); view it as
            // Cfloat (same two-float layout) for the fused emitter.
            emit_window_das_fused(
                snap_buf_base, K, snap_stride, snap_slot,
                dims_, reinterpret_cast<const Cfloat*>(w_c.data()),
                first_time, intensity);
        }
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