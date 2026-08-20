# Optimized CPU Beam Tracker — Algorithmic Specification

> **Scope of this document.** This is the single source-of-truth spec that the
> code-mode task will implement as `include/beamformer/cpu_opt_beam_tracker.hpp` +
> `src/cpu_opt_beam_tracker.cpp`. It records (a) what the *naive* tracker in
> `beam_tracker.cpp` actually computes and why it is limited, (b) a set of
> **algorithmic** optimizations chosen against the real code, with the math,
> the array-signal-processing origin and a citation for each, and (c) the exact
> public API the optimized header must expose so it is a **drop-in alternative**
> to `beamformer::BeamTracker` and reuses the existing packed-int4 shard data
> and CSV test fixtures byte-for-byte.
>
> **Explicitly out of scope here** (deferred to a later phase): SIMD / AVX
> intrinsics, memory packing / SoA layouts, thread pools, cache blocking. Those
> are micro-architectural optimizations layered on top of the *algorithm* fixed
> by this document. See the "Data layout & search plan" section.

---

## Naive baseline (current)

### What it computes

The current tracker lives in
[`beam_tracker_cpu_packed_intensity_into`](src/beam_tracker.cpp:90). It is a
**direct time-domain delay-and-sum (DAS), narrowband, per-spectrum Bartlett-style
coherent power** estimator, with a **trajectory-supplied (non-estimated) steering
direction** and a per-window weight recompute. Concretely, for one integration
window `w` with steering direction `d_w`:

```
w[f][a] = exp(-j * 2*pi * f * dot(pos[a], d_w) / c)            (geometric steering)
y[t][f] = Σ_a w[f][a] * x[t][f][a]                              (coherent sum)
P[t][f] = | y[t][f] |^2  =  y[t][f] * conj(y[t][f])             (Bartlett power)
```

where `x[t][f][a]` is the decoded signed-int4 voltage sample (real in low nibble,
imag in high nibble, range `-8..7`). `d_w` is **not estimated from the data**;
it is read from the linear placeholder trajectory `dir(t) = direction_from_lm(
l0 + t*dl, m0 + t*dm)` evaluated at the window's first sample
([`tracker_window_direction`](src/beam_tracker.cpp:67)).

So the naive code is:
1. a **steered beamformer** (not a snapshot-covariance estimator),
2. with a **fixed, parametrically-supplied** direction (not a direction-of-
   arrival estimator),
3. producing a **[`[time][freq][1]`](include/beamformer/formats.hpp:67)**
   intensity stream by naive triple-nested per-pixel MAC.

### Cost as actually present in the code

With `W = window_count = ceil(n_time / integration_spectra)` windows, the
hot loop in [`beam_tracker_cpu_packed_intensity_into`](src/beam_tracker.cpp:125)
is:

```
for window in [0,W):                       // W ≈ n_time / integration_spectra
  generate_weights(1 beam)                 //  O(n_freq * n_ant) per window
  for time   in [first,last):              //  integration_spectra per window
    for freq  in [0,n_freq):               //  n_freq = 336
      for ant  in [0,n_ant):               //  n_ant ∈ {32,64}
        decode int4, complex MAC
      write |sum|^2
```

Total FLOPs (complex MACs): `W * integration_spectra * n_freq * n_ant`
= `n_time * n_freq * n_ant`. For the default `n_time=15360, n_freq=336,
n_ant=64` that is **~3.3 × 10^8** complex MACs per run, plus `W` weight-set
generations each doing `n_freq * n_ant` `sincos`/`exp` evaluations (`48 * 336 * 64
≈ 1.0 × 10^6` transcendental-heavy ops). Big-O:

```
T_naive = Θ( n_time * n_freq * n_ant )      per run (memory-bound on int4 decode)
W_naive = Θ( W * n_freq * n_ant )            extra work for weight regeneration
S_naive = Θ( n_freq * n_ant )               per-window weight storage
```

### The concrete inefficiencies (algorithmic, not micro-architectural)

1. **The direction is never estimated from the data.** The whole point of a
   *tracker* — closing the loop between the receiver and the source's actual
   direction — is absent. `dir(t)` is an open-loop parametric guess. If the
   trajectory model is wrong, the coherent gain collapses silently (the
   "misaligned dims" exactly as the `aligned_total > misaligned_total` test in
    [`tests/cpu/test_beam_tracker.cpp:248`](tests/cpu/test_beam_tracker.cpp) shows). No
   spectrum of the data is ever formed over direction, so there is nothing to
   *track* adaptively.
2. **No spatial covariance is ever built.** Each `(t,f)` sample is consumed
   once and discarded. The natural statistic for direction estimation — the
   per-frequency sample covariance matrix `R[f] = (1/K) Σ_t x[t][f] x[t][f]^H`
   over the `K = integration_spectra` snapshots in a window — is never
   computed, so the rich family of covariance-based estimators (Bartlett,
   Capon/MVDR, MUSIC) that operate on `R` is unreachable. Resolution stays at
   the Rayleigh limit of the plain DAS.
3. **Per-window weight regeneration recomputes `exp`/`sincos` for the full
   `[freq][ant]` table.** [`generate_weights`](include/beamformer/weights.hpp:11)
   is called once per window with a one-element direction vector and independently
   re-evaluates `exp(-j*k*delay)` for `n_freq*n_ant` entries. For a slowly
   moving source the `delay` changes by a fraction of a channel between
   windows, yet the entire table is rebuilt from scratch.
4. **No coarse-to-fine search — in fact, no search at all.** Because there is
   no direction scan, there is also no opportunity to *prune* one. A full-grid
   scan over a `(n_u × n_v)` direction lattice would cost `Θ(n_freq * n_ant *
   n_u * n_v)` per spectrum if done naively per-pixel (exactly the pathology
   the fixed-grid [`cpu_beamformer`](include/beamformer/cpu_beamformer.hpp)
   already exhibits, scaled by grid size).
5. **Grid-only resolution; no sub-grid refinement.** Even if a direction scan
   were added, picking the argmax grid cell quantizes the estimated direction
   to the grid pitch (`l_step` default `0.02`). The tracked error floor is the
   grid pitch, not the Cramér-Rao bound.
6. **No time recursion across windows.** Each window's covariance (if it
   existed) and direction estimate would be recomputed from scratch even
   though consecutive windows share `(K-1)/K` of their snapshots for slow
   sources. An RLS-style recursion `R_w = λ R_{w-1} + x_w x_w^H` would let the
   estimate carry forward continuously.

These six points are the algorithmic gaps the optimizations below close. Each is
attributed to a canonical array-signal-processing result so the implementer can
cite the exact basis in the code comments.

---

## Algorithmic optimizations

Six optimizations are adopted. Each is stated as **name → math → origin/citation →
why it fixes this naive code → trade-offs / assumptions**. SIMD and cache layout
are explicitly *not* part of this set.

### O1. Per-frequency sample covariance + Bartlett / Capon (MVDR) spectrum

**Math.** Accumulate, per frequency `f` and window `w`, the sample covariance
matrix over the `K` snapshots in the window:

```
R_w[f] = (1/K) Σ_{t in window w} x[t][f] x[t][f]^H      ∈ C^{M×M},  M = n_ant
```

Direction scan over a steering vector `a(θ) = [e^{-j 2π f·dot(pos_a, θ)/c}]_a`:

```
Bartlett:   P_B(θ; f) = a(θ)^H R_w[f] a(θ)               (conventional beamformer)
Capon/MVDR: P_C(θ; f) = 1 / (a(θ)^H R_w[f]^{-1} a(θ))    (min. var., unity gain)
```

The scan power is integrated (averaged) across frequency for the final per-window
direction decision. Reconstruction of the *emitted* `[time][freq][1]` intensity
cube then uses the estimated `θ̂_w` with a final pass of the standard DAS pass.
**Default estimator: Bartlett** (no matrix inverse); **Capon is selectable** for
higher resolution once `R` is well-conditioned.

**Origin / citation.**
- Bartlett (conventional) beamformer: **Van Veen & Buckley, "Beamforming: a
  versatile approach to spatial filtering," IEEE ASSP Magazine, vol. 5, no. 2,
  pp. 4–24, Apr. 1988** — the canonical survey that names this the "conventional
  beamformer" `P = w^H R w` with `w = a`.
- Capon / MVDR: **J. Capon, "High-resolution frequency-wavenumber spectrum
  analysis," Proc. IEEE, vol. 57, no. 8, pp. 1408–1418, Aug. 1969** — the
  minimum-variance distortionless-response beamformer, `P = 1/(a^H R^{-1} a)`,
  with the unity-gain constraint `w^H a = 1`.

**Why it fixes *this* naive code.**
- Inefficiency **#2** directly: the naive code never builds `R`, so it can do
  *nothing* but the supplied-direction DAS. Forming `R` once per window per
  frequency (`Θ(K · M²)`) lets every direction in the scan reuse the *same*
  `R` via two matrix-vector products (`a^H R a`) instead of a fresh `Θ(M)`
  DAS per candidate direction. The scan share amortizes the covariance cost
  across the whole grid.
- Capon additionally raises angular resolution above the Rayleigh limit of the
  plain DAS, attacking inefficiency **#5** at the *estimator* level (the
  quadratic interpolation in O4 attacks it at the *grid* level).

**Trade-offs / assumptions.** Requires `K ≥ M` snapshots per window for `R` to
be full rank (with the default `K=320, M∈{32,64}` this holds with large margin),
and the **narrowband** assumption within a `300 kHz` channel (channel bandwidth
`≪` carrier `~336 MHz`, satisfied here). Capon needs `R` invertible; we add a
diagonal load `R + ε I` (`ε = λ_diag · trace(R)/M`) for numerical safety and to
model sensor noise, following the standard Tikhonov-loaded MVDR.

### O2. Spatial smoothing for rank / decorrelation (forward-backward)

**Math.** When the source environment is *not* a single point (multipath, RFI,
or two near sources), `R_w[f]` is rank-deficient or near-singular and Capon
degrades. For a **uniform linear / regular rectangular** sub-array (the project
geometry is a regular grid with `0.6 m` spacing, see
[`default_positions`](include/beamformer/geometry.hpp:25)), split the array
into `L` overlapping sub-arrays of length `P` (`M = L + P - 1`), form each
sub-array covariance `R_l`, and average:

```
R̃_w[f] = (1/L) Σ_{l=1..L} R_{l,w}[f]      (forward)
R̂_w[f] = ½ ( R̃_w[f] + J R̃_w[f]* J )      (forward-backward, J = exchange matrix)
```

`R̂` is full-rank (rank `P`) even for fully coherent sources, restoring Capon
resolution. The scan then uses the sub-array steering vector `a_P(θ)`.

**Origin / citation.**
- Spatial smoothing: **T.-J. Shan, M. Wax, T. Kailath, "On spatial smoothing for
  direction-of-arrival estimation of coherent signals," IEEE Trans. ASSP, vol.
  33, no. 4, pp. 806–811, Aug. 1985**.
- Forward-backward averaging: **A. J. Weiss, B. Friedlander, "Forward-backward
  smoothing of covariance matrices," IEEE Trans. Signal Process., vol. 45, no.
  7, pp. 1842–1849, Jul. 1997** (and the classic **Ulrych & Clayton 1976**
  treatment of FB averaging for spectral estimation).

**Why it fixes *this* naive code.** Without rank restoration, Capon's `R^{-1}`
is meaningless the moment the synthetic moving source plus any decorrelated noise
produces a near-rank-1 `R` (which is exactly the rank-1 case
[`beam_tracker_make_moving_point_source`](src/beam_tracker.cpp:148) generates:
a single coherent plane wave). Spatial smoothing + FB averaging is the
textbook fix that makes the covariance-based pipeline robust to the very inputs
the existing test fixtures produce. The regular grid geometry
([`regular_array`](include/beamformer/geometry.hpp:23)) is precisely the
assumption spatial smoothing requires.

**Trade-offs / assumptions.** Reduces effective aperture from `M` to `P`
(`P ≈ 2M/3` is a common choice), which widens the Bartlett lobe. This is a
deliberate **resolution-for-robustness** swap, selectable per run via a config
flag (`spatial_smoothing_subarray_size`); with `0` it is disabled and the code
behaves as plain O1.

### O3. Coarse-to-fine (multi-resolution / hierarchical) direction search

**Math.** Define a hierarchy of direction lattices `Λ_0 ⊃ Λ_1 ⊃ ⋯ ⊃ Λ_L`
with pitch `Δ_k = Δ_0 / 2^k` (`Δ_0` coarse, `Δ_L` fine — labels reversed from
the obvious reading; the idea is *coarse first, refine*). At each level:

```
θ_k* = argmax_{θ ∈ Λ_k ∩ N(θ_{k-1}*)}  P(θ)
```

where `N(·)` is a small neighbourhood (e.g. the `3×3` cells around the previous
peak). Starting from a coarse full-sky grid of `n_u × n_v = G_0²` cells, the
-cost is

```
T_scan = Θ( n_freq * G_0² )       level 0:  full coarse scan, one a^H R a per cell
        + Σ_{k=1..L} Θ( n_freq * 9 )    refinement: 3×3 neighbourhood per level
```

versus `Θ( n_freq * G_L² )` for a single fine-grid scan. The hierarchy reduces
the asymptotic scan cost from `Θ(n_freq * G_L²)` to `Θ(n_freq * G_0²)`, i.e. a
factor `~(G_L / G_0)^2` at the price of an `L`-step refinement whose per-step
work is constant (`3×3`). Choosing `Δ_0` covers the full FoV and `Δ_L =
Δ_0/2^L` reaches sub-`l_step=0.02` resolution, the search bound is provably tight
provided the response is unimodal in each cell neighbourhood (true for the
Bartlett mainlobe away from grating lobes at the design frequency).

**Origin / citation.**
- Multi-resolution / hierarchical search is the standard DOA coarse acquisition,
  treated generally in **Van Veen & Buckley, IEEE ASSP Mag. 1988 (op. cit.)**
  under "beam-space processing"; the coarse-to-fine acceleration specifically
  mirrors the **two-stage search in adaptive radar** described in
  **Skolnik, *Radar Handbook*, 3rd ed., McGraw-Hill, 2008, §7.11 (monopulse /
  sequential-lobing search)**.
- The unimodality-in-cell assumption and the resulting search bound are the
  content of the lattice search argument in **Haykin & Reilly, "Maximum-likelihood
  estimation for array signal processing," in *Advances in Spectrum Analysis and
  Array Processing*, Prentice-Hall, 1991** (hierarchical ML search).

**Why it fixes *this* naive code.** Inefficiencies **#1** and **#4**: the naive
code does *no* scan at all (the direction is supplied), so adding *any* scan
would already break the open-loop nature, and doing it *hierarchically* keeps
the scan affordable — full fine-grid Bartlett over a FoV of useful extent would
be the dominant cost otherwise. The hierarchical path makes the per-window
direction-decision cost comparable to a handful of windows of plain DAS.

**Trade-offs / assumptions.** Requires the response to be unimodal in the
`3×3` refinement neighbourhoods (true for a single dominant source in the
mainlobe; false near grating lobes or with two equal-power sources — the
fallback is to run the coarse level on a denser `Λ_0` for those cases, exposed
as a config knob `coarse_grid_resolution`).

### O4. Quadratic 3-point peak interpolation for sub-grid direction

**Math.** After the fine level `L` identifies grid cell `(i*, j*)` with power
`P* = P(i*,j*)`, sample its 4-neighbourhood and use the **closed-form parabolic
peak fit** in each axis independently (standard bivariate separable refinement):

```
δ_u = ½ (P_{+1,0} - P_{-1,0}) / (P_{-1,0} - 2 P_{0,0} + P_{+1,0})
δ_v = ½ (P_{0,+1} - P_{0,-1}) / (P_{0,-1} - 2 P_{0,0} + P_{0,+1})
θ̂ = θ(i* + δ_u,  j* + δ_v)       (interpolated direction cosines, |δ| ≤ ½)
```

The denominator is `(P_- - 2 P_0 + P_+)`; if it is non-positive (a non-concave
point) refinement is skipped and the grid argmax is returned.

**Origin / citation.**
- Quadratic interpolation of the DFT/periodogram peak: **D. J. Thomson, "Spectrum
  estimation and harmonic analysis," Proc. IEEE, vol. 70, no. 9, pp. 1055–1096,
  Sep. 1982** (sec. on peak interpolation in multi-taper analysis); the same
  parabolic fit is the standard sub-bin estimator in **Priestley, *Spectral
  Analysis and Time Series*, Academic Press, 1981, §6.1**.
- Application to the beamformer spatial spectrum is the textbook sub-grid DOA
  refinement in **Van Trees, *Optimum Array Processing* (Detection, Estimation,
  and Modulation Theory, Part IV), Wiley, 2002, §5.4 ("practical DOA
  estimation")**.

**Why it fixes *this* naive code.** Inefficiency **#5**: the grid pitch
(`l_step = 0.02`) sets the *floor* of the naive tracker's direction error even
if a scan existed. The parabolic fit gives ~1/10-cell sub-grid accuracy on a
smooth mainlobe for ~6 extra power samples per window — essentially free given
the covariance is already formed.

**Trade-offs / assumptions.** Assumes the response is locally parabolic around
the true peak (good in the mainlobe of a well-filled array; poor on the
shoulders of a grating lobe — the concavity check rejects those cases). Accuracy
is second-order in the grid pitch.

### O5. Recursive covariance update with exponential forgetting (RLS-style)

**Math.** Instead of recomputing `R_w[f]` from scratch each window over its `K`
snapshots, carry `R` across windows with an exponential forgetting factor
`0 < λ ≤ 1` (per-frequency, dropping the explicit `[f]` index):

```
R_w = λ R_{w-1} + (1-λ) x_w x_w^H       (rank-1 update, x_w = latest snapshot)
```

or, in windowed-block form with effective snapshot count `K_eff = 1/(1-λ)`:

```
R_w = λ^K R_0 + (1-λ) Σ_{t≤window} λ^{...} x_t x_t^H
```

The direction estimate at window `w` is then the argmax of `a^H R_w^{-1} a`
(or `a^H R_w a` for Bartlett) over the hierarchical search centred on the
*previous* estimate `θ̂_{w-1}` (prediction step; the linear trajectory model of
the naive code is used only as a **prior / initial guess** for the very first
window and the centre of the level-0 coarse grid thereafter).

**Origin / citation.**
- RLS / recursive covariance update: **S. Haykin, *Adaptive Filter Theory*,
  5th ed., Pearson, 2014, Chapter 10 ("Recursive Least-Squares Estimator (RLS)")** —
  the rank-1 update `R_w = λ R_{w-1} + (1-λ) x_w x_w^H` is the RLS covariance
  recursion; exponential forgetting for non-stationary signals is treated in
  §10.6 ("RLS with forgetting factor").
- The tracking-DOA formulation specifically is
  **B. Widrow and S. D. Stearns, *Adaptive Signal Processing*, Prentice-Hall,
  1985, Chapter 12 (adaptive arrays with the LMS / RLS family)** and
  **R. A. Monzingo and T. W. Miller, *Introduction to Adaptive Arrays*,
  Wiley, 1980** (the array-context recursion).

**Why it fixes *this* naive code.** Inefficiency **#6**: the naive code treats
every window as independent and recomputes the whole weight set from scratch.
The recursion lets the estimate *carry* smoothly across windows (true adaptive
tracking, closing the loop the naive code never closes — inefficiency **#1**),
and reduces the per-window covariance work from `Θ(K · M²)` to `Θ(M²)` plus a
single rank-1 update when only the newest spectrum arrives incrementally
(streaming variant), or `Θ(K · M²)` only when a window is processed wholesale
with carry-over via `λ`. The forgetting factor lets the tracker follow a
moving source (the moving point source
[`beam_tracker_make_moving_point_source`](src/beam_tracker.cpp:148) is exactly
the non-stationary signal RLS-with-forgetting was designed for) while rejecting
stale statistics.

**Trade-offs / assumptions.** `λ` trades **tracking agility vs. estimation
variance**: `λ → 1` gives a long effective window (low variance, slow tracking);
small `λ` tracks fast but is noisier. With `K = 320`-snapshot windows and the
~3.33 µs spectrum period, a default `λ` corresponding to an effective memory of
~1–2 windows is recommended and exposed as config `forgetting_factor`. Requires
the recursion to be numerically stable; the same diagonal load as O1 is applied
each step.

### O6. Precomputed steering phase tables + incremental power response

**Math.** The steering vector is `a(θ) = exp(-j φ(θ))`, `φ_a(θ; f) =
2π f · dot(pos_a, θ) / c`. For the *coarse* search lattice `Λ_0` (whose
cell count `G_0²` is small), precompute once per run, as `ComplexFloat` tables:

```
A[f][cell][a]  = exp(-j 2π f · dot(pos[a], θ_cell) / c)
```

(precomputed once at construction, O(n_freq * G_0² * n_ant) `sincos` calls, paid
*once* per run, not per window). The Bartlett power per cell is then

```
P_B(cell; f) = a_cell^H R_w[f] a_cell = Σ_{a,b} conj(A[f][cell][a]) R_w[f][a,b] A[f][cell][b]
```

which, given `R_w[f]` already formed, is a Hermitian quadratic form
reusable across **all** windows that scan the coarse grid (the steering vectors
do not change with `t`; only `R` does). The hierarchical refinement levels use
the **same trick recursively**: for each fine-cell visited, the steering vector
is computed once and cached for the lifetime of the run so subsequent windows
that re-enter that cell reuse it. Where the source moves slowly relative to the
window cadence (the linear-track case used in the existing test fixtures), an
**incremental** covariance update plus a cached steering table means each window
beyond the first pays only `Θ(n_freq * G_0² * M_eff²)` for the coarse scan plus
the `Θ(n_freq * 9 * M_eff²)` refinements, with no per-window `sincos` calls at
all.

**Origin / citation.**
- Steered-covariance (beam-space from element-space) preprocessing: the
  **"beam-space preprocessing"** section of **Van Veen & Buckley, IEEE ASSP
  Mag. 1988 (op. cit.)** and **Van Trees, *Optimum Array Processing*, 2002,
  §6.8 ("beam-space beamformers")** — pre-multiplying the element-space data (or
  covariance) by a fixed beamforming matrix `B` to reduce dimensionality and
  amortize steering cost.
- The precomputed-phasor FFT-beamforming lineage (steering-by-DFT of the aperture
  for a regular array, where the steering vector over direction *is* a DFT row
  so the whole direction scan reduces to a batched FFT) is in
  **A. V. Oppenheim & R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
  Pearson, 2010, Chapter 8 ("DFT/DFT-based filtering")** and the array
  formulation in **Van Trees, op. cit., §6.6 ("FFT beamforming")** — enabling a
  later phase to fold the coarse direction scan into a single per-frequency
  2-D FFT over the aperture (noted but **not** mandated here, see the deferred
  items in the "Data layout & search plan" section).

**Why it fixes *this* naive code.** Inefficiency **#3**: the naive code
re-invokes [`generate_weights`](include/beamformer/weights.hpp:11) — `exp` for
every `[f][a]` — once per window even though the *grid* of candidate directions
is static. Caching `A[f][cell][a]` removes all per-window steering-vector
transcendentals; the per-window work becomes pure floating-point quadratic forms
on already-formed `R_w`. The incremental covariance recursion from O5 makes this
strictly cheaper than the naive per-pixel DAS once more than a handful of
candidate directions are scanned, which is precisely the regime the hierarchical
scan of O3 puts us in.

**Trade-offs / assumptions.** Memory for the steering table is
`Θ(n_freq * G_0² * M)` `ComplexFloat`s plus the refinement cells actually
visited — bounded by `G_0²` (small) and the cells touched so far. With
`G_0² = 12² = 144` coarse cells, `n_freq = 336`, `n_ant = 64` this is
`336 * 144 * 64 * 8 ≈ 25 MB` — well within host RAM, and the table can be dropped
to float16/half the storage in a later phase (deferred). Assumes the array
ge `[pos[a]]` is fixed for the run (true: [`default_positions`](include/beamformer/geometry.hpp:25)
is constant per `n_ant`), so the cache is built once.

---

## Proposed public API

The optimized header will expose a **drop-in alternative** to the naive
`beamformer::BeamTracker` family. It reuses the exact input/output byte
contracts ([`PackedVoltage`](include/beamformer/formats.hpp:13),
[`Intensities`](include/beamformer/formats.hpp:19),
[`Dimensions`](include/beamformer/config.hpp:74),
[`Vec3`](include/beamformer/geometry.hpp:12)) and the existing
[`read_packed_voltage`](src/io.cpp)/[`write_intensities`](src/io.cpp) IO,
so the CLI in [`tools/beam_tracker_cpu.cpp`](tools/beam_tracker_cpu.cpp), the
test fixtures in [`tests/cpu/test_beam_tracker.cpp`](tests/cpu/test_beam_tracker.cpp),
and the CSV metrics format all carry over unchanged. The namespace and the
free-function mirror of the naive API are preserved; a stateful class is added
to hold the cached steering tables and the recursive covariance state.

```cpp
#pragma once

#include "beamformer/beam_tracker.hpp"   // reuses TrackerTrajectoryConfig, tracker_beam_count, tracker_window_count, tracker_window_direction
#include "beamformer/complex.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#include <cstddef>
#include <vector>
#include <optional>

namespace beamformer {

// Estimator family for the per-frequency spatial spectrum scan.
enum class TrackerEstimator {
    Bartlett,   // P(θ) = a(θ)^H R a(θ)            (no matrix inverse; default)
    Capon       // P(θ) = 1 / (a(θ)^H (R+εI)^{-1} a(θ))   (MVDR; higher resolution)
};

// Configuration of the optimized search/estimation. All knobs carry sensible
// defaults so a default-constructed struct reproduces the naive open-loop
// output exactly (zero search grid → direction supplied by the trajectory,
// same as the existing behavior; this is the back-compat / equivalence test
// anchor in tests/cpu/test_beam_tracker.cpp).
struct CpuOptTrackerConfig {
    // Steering / estimation.
    TrackerEstimator estimator = TrackerEstimator::Bartlett;

    // Coarse-to-fine search hierarchy (O3).
    //   coarse_grid_resolution: number of cells per FoV axis at level 0 (e.g. 12).
    //   refinement_levels:      number of 3x3 refinement halvings after level 0.
    //   search_fov_l, search_fov_m: half-extent of the scanned FoV in (l, m).
    // Setting coarse_grid_resolution <= 1 disables the scan entirely and the
    // tracker falls back to the supplied trajectory direction (naive behaviour).
    std :: size_t coarse_grid_resolution = 1;
    std :: size_t refinement_levels      = 0;
    float search_fov_l = 0.2F;
    float search_fov_m = 0.2F;

    // Spatial smoothing sub-array size (O2); 0 disables smoothing.
    // Must satisfy 1 <= subarray <= n_ant; typically ~2/3 * n_ant.
    std :: size_t spatial_smoothing_subarray_size = 0;

    // Recursive covariance forgetting factor (O5), 0 < λ <= 1.
    // λ = 1.0  => stationary covariance (no forgetting; equals block estimate).
    // λ < 1    => adaptive tracking with effective memory 1/(1-λ) snapshots.
    float forgetting_factor = 1.0F;

    // Diagonal loading of R for Capon numerical stability (O1): R + ε I, with
    // ε = diagonal_load * trace(R)/M. 0 disables loading (not recommended for Capon).
    float diagonal_load = 1.0e-3F;

    // Sub-grid quadratic interpolation switch (O4).
    bool enable_quadratic_peak_interp = true;

    // Per-window snapshot count reused from the naive API (one direction
    // decision per window). Mirrors TrackerConfig.integration_spectra.
    std :: size_t integration_spectra = integration_direct.integration_spectra;
};

// Stateful optimized tracker: caches the precomputed steering phase tables
// (O6) and holds the recursive covariance state across windows (O5).
class CpuOptBeamTracker {
public:
    // Construct with the array geometry + frequency plan reused from the naive
    // path; the tracker precomputes the coarse steering table here (O6).
    CpuOptBeamTracker(std::vector<Vec3> positions_m,
                      std::vector<float> frequencies_hz,
                      Dimensions dims,
                      CpuOptTrackerConfig config);

    // Run the optimized pipeline over a packed-int4 shard and write the standard
    // [time][freq][beam=1] float32 intensity cube. Byte-compatible output with
    // the naive beam_tracker_cpu_packed_intensity_into.
    void run_into(const PackedVoltage& packed, Intensities& intensity);

    // Direction estimated for window w (post-search + O4 interpolation).
    // Returns the trajectory-supplied direction when scanning is disabled
    // (coarse_grid_resolution <= 1), giving the naive back-compat path.
    Vec3 window_direction(std::size_t window) const;

    const Dimensions& dimensions() const noexcept;
    const CpuOptTrackerConfig& config() const noexcept;

private:
    std::vector<Vec3> positions_m_;
    std::vector<float> frequencies_hz_;
    Dimensions dims_;
    CpuOptTrackerConfig config_;
    // Cached coarse steering vectors A[f][cell][a]  (O6),
    // per-window recursive covariance R_w[f]  (O5),
    // per-window estimated direction vector.
    // (Layout / container types are an implementation detail of the .cpp.)
};

// ---- Free-function drop-in mirrors of the naive API -----------------------

// Returns a freshly-constructed intensity cube; identical byte layout to
// beam_tracker_cpu_packed_intensity. Uses default geometry + frequencies from
// default_positions / channelized_frequencies, matching the naive CLI path.
Intensities cpu_opt_beam_tracker_packed_intensity(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& trajectory,                  // initial guess / prior
    const CpuOptTrackerConfig& opt = CpuOptTrackerConfig{});

// Into-variant for reusable output buffers.
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& trajectory,
    Intensities& intensity,
    const CpuOptTrackerConfig& opt = CpuOptTrackerConfig{});

// Stateful variant: caller owns the tracker (preferred for streaming, since
// the covariance recursion and steering cache persist across calls).
void cpu_opt_beam_tracker_packed_intensity_into(
    const PackedVoltage& packed,
    const TrackerConfig& trajectory,
    CpuOptBeamTracker& tracker,
    Intensities& intensity);

} // namespace beamformer
```

### Behavioural contract / back-compatibility guarantees

- **Output byte layout** is exactly [`intensity[time][freq][beam=0]`](include/beamformer/formats.hpp:67)
  `float32`, written via the existing [`write_intensities`](src/io.cpp) —
  unchanged on disk.
- **Default-equivalence anchor.** With `CpuOptTrackerConfig` default-constructed
  (`coarse_grid_resolution = 1`, `refinement_levels = 0`,
  `forgetting_factor = 1.0`, `estimator = Bartlett`, smoothing off), the output
  must equal the naive [`beam_tracker_cpu_packed_intensity`](src/beam_tracker.cpp:78)
  to within float-rounding — this is the **exact** port of the
  `grid_intensity == tracker_intensity` assertion in
  [`tests/cpu/test_beam_tracker.cpp:139`](tests/cpu/test_beam_tracker.cpp) and the
  per-spectrum `close(actual, expected, 1e-4)` check at
  [`tests/cpu/test_beam_tracker.cpp:163`](tests/cpu/test_beam_tracker.cpp). This is the
  primary regression test the code-mode task must preserve.
- **Feature parity with the naive test fixtures.** The moving-source recovery
  test ([`tests/cpu/test_beam_tracker.cpp:248`](tests/cpu/test_beam_tracker.cpp)) must
  still pass (`aligned_total > misaligned_total`), and additionally, with the
  search enabled, the *estimated* `window_direction(w)` must lie closer to the
  true source direction than the open-loop trajectory guess (a new assertion the
  code-mode test is expected to add).
- `n_beams == 1` is enforced the same way
  ([`validate_dimensions`](include/beamformer/config.hpp:81) +
  tracker-specific check), `n_freq == 336` and `n_ant ∈ {32,64}` are inherited
  from [`Dimensions`](include/beamformer/config.hpp:74).

---

## Data layout & search plan

This section fixes the *algorithm* and explicitly defers the micro-architecture.

### What is designed here (algorithm)
- Element-space sample covariance `R_w[f]` per frequency, per window, stored as
  a row-major `M_eff × M_eff` Hermitian `ComplexFloat` matrix per `f` (lower
  triangle only, since `R = R^H`). `M_eff = spatial_smoothing_subarray_size`
  when smoothing is enabled, else `n_ant`.
- Precomputed coarse steering table `A[f][cell][a]` (`ComplexFloat`) built once
  in the `CpuOptBeamTracker` constructor, reused for every window via the cache
  in O6.
- Direction search plan, per window:
  1. Update `R_w` via the O5 recursion (`R_w = λ R_{w-1} + sample` block or
     rank-1).
  2. Level-0 coarse scan over `Λ_0` using the cached `A`: `P_B(cell) = a^H R a`
     (Bartlett) or `1 / (a^H R^{-1} a)` (Capon, with diagonal load).
  3. Refine 3×3 neighbourhoods through levels `1..L` (O3), recomputing the
     steering vector *once* per visited fine cell and caching it.
  4. O4 quadratic interpolation around the level-`L` argmax.
  5. Final DAS pass emitting the `[time][freq][1]` intensity cube using the
     estimated `θ̂_w` (the emitted product remains the plain Bartlett power, so
     the on-disk output is byte-identical to the naive path for the same
     direction — only the *direction itself* changes).
- Time-recursion: `R_0` initialised from the first window's snapshots (with the
  trajectory-provided direction used as the centre of the level-0 grid until the
  recursion has enough samples to dominate). The linear trajectory in
  [`TrackerTrajectoryConfig`](include/beamformer/beam_tracker.hpp:42) is
  repurposed as a **prior** (initial guess / level-0 grid centre), not as an
  oracle.

### Explicitly deferred to a later phase (NOT designed here)
- **SIMD / AVX / intrinsics** for the int4 decode, complex MAC, and Hermitian
  quadratic form `a^H R a`. The algorithm is written in scalar C++ in the first
  implementation; the math is laid out so that the quadratic forms vectorize
  trivially later.
- **SoA / memory-packing** of `R_w` (e.g. real/imag split, padded to cache
  lines) and of the cached steering table. The naive `ComplexFloat` (two adjacent
  `float`s, see [`complex.hpp`](include/beamformer/complex.hpp:7)) layout is
  kept.
- **Threading / parallelism over frequency** (the `n_freq` axis is embarrassingly
  parallel and is the natural partition; left for a later thread-pool phase).
- **FFT-based coarse direction scan (FFT beamforming).** For the *regular
  rectangular* array geometry, the coarse steering table `A` over the coarse
  direction lattice is a set of DFT rows; the entire coarse Bartlett scan
  `P_B(θ; f) = a_θ^H R_f a_θ` collapses to a 2-D FFT over the aperture for the
  regular-grid case (the lineage cited in O6). This is a high-value algorithmic
  optimization *but* one that the existing *rectangular* geometry (4×8 / 8×8 grid,
  [`default_positions`](include/beamformer/geometry.hpp:25)) only partially
  supports (DFT-of-aperture steering requires the array to be a *uniform
  rectangular* planar array, which it is), so it is *noted as a candidate next
  optimization* for the post-algorithmic phase, not committed to in v1 of the
  optimized code.
- **Caching of `R_w^{-1}` via the Woodbury / Sherman-Morrison rank-1 update**
  (which would let the Capon inverse be updated incrementally alongside the
  O5 covariance recursion) — an algorithmic refinement left for after the base
  Bartlett + Capon pipeline is validated against the naive tests.

### Reuse of existing infrastructure (no new contracts)
- Geometry: [`default_positions`](include/beamformer/geometry.hpp:25),
  [`channelized_frequencies`](include/beamformer/geometry.hpp:30),
  [`direction_from_lm`](include/beamformer/geometry.hpp:34).
- Formats / indexing: [`PackedVoltage`](include/beamformer/formats.hpp:13),
  [`Intensities`](include/beamformer/formats.hpp:19),
  [`voltage_index`](include/beamformer/indexing.hpp),
  [`intensity_index`](include/beamformer/indexing.hpp).
- int4 decode: [`unpack_complex_int4`](include/beamformer/int4.hpp) (reused
  verbatim — the optimized path decodes samples identically, just feeds them
  into the covariance accumulator instead of a per-pixel MAC).
- IO: [`read_packed_voltage`](src/io.cpp), [`write_intensities`](src/io.cpp),
  unchanged.
- Math constants: [`speed_of_light_m_per_s`](include/beamformer/physics.hpp),
  [`two_pi`](include/beamformer/physics.hpp).

---

## References

- **J. Capon**, "High-resolution frequency-wavenumber spectrum analysis,"
  *Proc. IEEE*, vol. 57, no. 8, pp. 1408–1418, Aug. 1969. — MVDR / Capon
  beamformer (O1).
- **B. D. Van Veen and K. M. Buckley**, "Beamforming: a versatile approach to
  spatial filtering," *IEEE ASSP Magazine*, vol. 5, no. 2, pp. 4–24,
  Apr. 1988. — Conventional (Bartlett) beamformer, beam-space preprocessing
  (O1, O6).
- **H. L. Van Trees**, *Optimum Array Processing* (Detection, Estimation, and
  Modulation Theory, Part IV), Wiley-Interscience, 2002. — Practical DOA
  estimation, beam-space / FFT beamforming, sub-grid refinement (O4, O6).
- **T.-J. Shan, M. Wax, T. Kailath**, "On spatial smoothing for
  direction-of-arrival estimation of coherent signals," *IEEE Trans. Acoust.,
  Speech, Signal Process.*, vol. 33, no. 4, pp. 806–811, Aug. 1985. — Spatial
  smoothing (O2).
- **A. J. Weiss, B. Friedlander**, "Forward-backward smoothing of covariance
  matrices," *IEEE Trans. Signal Process.*, vol. 45, no. 7, pp. 1842–1849,
  Jul. 1997. — FB averaging for rank restoration (O2).
- **S. Haykin**, *Adaptive Filter Theory*, 5th ed., Pearson/Prentice-Hall,
  2014, Chapter 10 (RLS) and §10.6 (forgetting factor). — Recursive covariance
  update / RLS tracking (O5).
- **B. Widrow and S. D. Stearns**, *Adaptive Signal Processing*,
  Prentice-Hall, 1985, Chapter 12 (adaptive arrays). — Array-context recursive
  tracking (O5).
- **R. A. Monzingo and T. W. Miller**, *Introduction to Adaptive Arrays*,
  Wiley, 1980. — Recursive adaptive-array estimation (O5).
- **M. I. Skolnik**, *Radar Handbook*, 3rd ed., McGraw-Hill, 2008,
  §7.11 (monopulse / sequential-lobing search). — Coarse-to-fine / two-stage
  search lineage (O3).
- **S. Haykin, J. P. Reilly et al.**, "Maximum-likelihood estimation for array
  signal processing," in *Advances in Spectrum Analysis and Array Processing*,
  Prentice-Hall, 1991. — Hierarchical ML lattice search and unimodality bound
  (O3).
- **D. J. Thomson**, "Spectrum estimation and harmonic analysis," *Proc. IEEE*,
  vol. 70, no. 9, pp. 1055–1096, Sep. 1982. — Peak interpolation in spectral
  estimation (O4).
- **M. B. Priestley**, *Spectral Analysis and Time Series*, Academic Press,
  1981, §6.1. — Parabolic (quadratic) peak-fit closed form (O4).
- **A. V. Oppenheim and R. W. Schafer**, *Discrete-Time Signal Processing*,
  3rd ed., Pearson/Prentice-Hall, 2010, Chapter 8 (DFT/DFT-based filtering). —
  FFT-beamforming lineage for the deferred coarse-scan folding (O6 note).
