# Beam Tracker Precision Research Report

> **Scope.** This report is a working research document for the CHARTS/CHIME voltage beamformer and dynamic beam tracker pipeline. It collects precision-related findings relevant to the CUDA V3 tracker suite ([`include/beamformer/cuda_beam_tracker_v3.hpp`](include/beamformer/cuda_beam_tracker_v3.hpp)), its predecessor fused-warp-shuffle kernel ([`include/beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp`](include/beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp)), the CPU reference implementations ([`src/beam_tracker.cpp`](src/beam_tracker.cpp), [`src/beam_tracker_opt.cpp`](src/beam_tracker_opt.cpp), [`src/beam_tracker_opt_v2.cpp`](src/beam_tracker_opt_v2.cpp)), and the packing/unpacking primitives in [`include/beamformer/int4.hpp`](include/beamformer/int4.hpp) / [`include/beamformer/complex.hpp`](include/beamformer/complex.hpp).
>
> The report is structured so each chapter is written for a tight focus area. Numerical targets are written in the math notation used elsewhere in the project (e.g. $\text{ULP}$ = unit in the last place, $f_{32}$ = IEEE-754 binary32, $f_{64}$ = IEEE-754 binary64, $\epsilon_{\text{machine}} \approx 1.19 \times 10^{-7}$ for binary32).
>
> **Companion documents.** [`info/cuda_implementation_investigation.md`](info/cuda_implementation_investigation.md) and [`info/beam_tracker_opt_v2_verdict.md`](info/beam_tracker_opt_v2_verdict.md) frame the data-movement and bit-exactness context this report builds on.

---

## Table of Contents

1. [CUDA Math Precision Tradeoffs](#1-cuda-math-precision-tradeoffs)
2. [Accumulator Patterns for Voltage Beamforming](#2-accumulator-patterns-for-voltage-beamforming)
3. [Phase Arithmetic Alternatives](#3-phase-arithmetic-alternatives)
4. [Magnitude Stability (Intensity $|\sum x \cdot w^*|^2$)](#4-magnitude-stability-intensity-sum-x-cdot-w2)
5. [PTX Bit Tricks](#5-ptx-bit-tricks)
6. [Radio-Astronomy GPU Beamforming Literature](#6-radio-astronomy-gpu-beamforming-literature)
7. [Synthesis & Recommendations](#7-synthesis--recommendations)

---

## 1. CUDA Math Precision Tradeoffs

### 1.1 Numeric types available on device

| Type | Bits | ULP at 1.0 | $\epsilon_{\text{machine}}$ | Hardware-native ops |
|---|---:|---:|---:|---|
| `float` ($f_{32}$) | 32 | $\approx 1.19 \times 10^{-7}$ | $2^{-23}$ | FMAs, `__fmaf_rn`, `__saturatef` |
| `double` ($f_{64}$) | 64 | $\approx 2.22 \times 10^{-16}$ | $2^{-52}$ | Native only on SM ≥ 6.0 (Pascal) in 1:2 or 1:4 ratio vs. $f_{32}$ |
| `__half` / `__nv_bfloat16` | 16 | $\approx 9.77 \times 10^{-4}$ / $7.81 \times 10^{-3}$ | $2^{-10}$ / $2^{-7}$ | Tensor-core paths; throughput 2–16× $f_{32}$ |
| `int8` | 8 | n/a | n/a | IMAD, `__dp4a`, tensor-core INT8 |
| `int4` (packed nibble) | 4 sign-magnitude | n/a | n/a | Manual decode via `bfe`/shift+mask |

The CHARTS voltage pipeline operates almost entirely on three of these:

- **packed `int4` complex voltages** ([`include/beamformer/int4.hpp`](include/beamformer/int4.hpp)) — signed $n \in [-8, 7]$ per real/imag nibble,
- **`float` $f_{32}$** for steering weights and the per-cell complex accumulator,
- **`float` $f_{32}$** for intensity $I[t][f] \in \mathbb{R}_{\geq 0}$.

No `double` is used on the hot path. This is a deliberate, measured choice: the project documents in [`info/beam_tracker_opt_v2_verdict.md`](info/beam_tracker_opt_v2_verdict.md) that v0/v1/v2 are byte-equal against the naive baseline, and the bit-exactness test is re-asserted before every timed run (`naive == v1 == v2 byte-equal: PASS (5160960 cells)`).

### 1.2 The `nvcc` default: fused multiply-add

`nvcc` and `ptxas` aggressively contract `a*b + c` into a single hardware FMA. The compiler flags are:

- `-fmad=true` (default; explicit only at `nvcc` level)
- `--fmad=true` (default; `ptxas`)
- `-prec-div=false` (default), `-prec-sqrt=false` (default)
- `-ftz=false` is the default for `nvcc`; `-ftz=true` flushes denormals to zero

For CHARTS, FMAs change rounding versus the CPU scalar code in subtle but deterministic ways:

- **Single FMA vs. separate `mul`+`add`** — the FMA computes the exact product before rounding it into the accumulator. CPU code that does `acc = acc + (w_re * s_re)` with a naive IEEE round-then-add will diverge from `acc = fmaf(w_re, s_re, acc)` in roughly 1 ULP per FMA. Over $N_{\text{ant}} = 64$ FMAs that is up to $\approx 64\,\text{ULP} \approx 7.6 \times 10^{-6}$ relative drift per accumulator.
- **`-fmad=false`** restores separate rounding but typically costs ~10–20% throughput on the FFMA pipe and disables some ILP-driven instruction scheduling that the V3 ILP unroller relies on.

**Recommendation for the project.** If we want the V3 kernels to be byte-comparable to the CPU reference, we must either (a) build V3 with `-fmad=false` and live with the throughput hit, or (b) accept a documented $\le 10^{-5}$ per-cell relative tolerance and adjust `tests/cuda/test_cuda_beam_tracker_v3.cpp` accordingly. The current test suite uses a single global epsilon; tightening or loosening this is a project-wide call.

### 1.3 Division, square root, and transcendental reciprocals

`ptxas` by default lowers `x / y` and `1 / sqrt(x)` to approximate hardware instructions:

- `rcp.approx.f32` — relative error ~$2^{-22.8}$ (~1.9 ULP).
- `div.approx.f32` — ~1 ULP, full IEEE except for special inputs.
- `sqrt.approx.f32` — ~1 ULP.
- `rsqrt.approx.f32` — ~1 ULP, ~$2^{-22}$ worst-case for the IEEE-correct `rsqrt` then a Newton-Raphson iteration costs ~6 SP FLOPs.

The CHARTS hot path contains **no division, square root, or reciprocal** in either Pass 1 (steering weights) or Pass 2 (intensity), so this class of precision loss is not exercised. If MVDR/Capon DOA ever moves to GPU, the `div.full` / `sqrt.full` paths (`-prec-div=true`, `-prec-sqrt=true`) become mandatory because the relative cost is small and the precision impact is large.

### 1.4 `__sincosf` / `__cosf` / `__sinf`: the SFU accuracy spectrum

This is where the project most directly eats a precision hit. The trade-off matrix (NVIDIA CUDA Math API docs, current to CUDA 12.x):

| Intrinsic | Implementation | ULP bound | Notes |
|---|---|---:|---|
| `__sinf(x)` / `__cosf(x)` | SFU `sin`/`cos` (MUFU) | $\le 2$ ULP for $x \in [-\pi, \pi]$; degrades near boundaries | Single SFU op, ~1/4 SP throughput |
| `__sincosf(x, &s, &c)` | SFU `sin`/`cos` parallel | $\le 2$ ULP same domain | One trip, both results |
| `sinf(x)` / `cosf(x)` | libdevice IEEE-correct | $\le 2$ ULP globally, slow paths | Multi-instruction, very slow |
| `sincosf(x, &s, &c)` | libdevice | $\le 2$ ULP globally | Slow |
| `sincospif(x, &s, &c)` | libdevice | Accurate for $x \cdot \pi$ reductions | Slightly faster than full `sincosf` |
| Software Payne-Hanek / Cody-Waite | libm-equivalent | $< 1$ ULP | Not hardware-accelerated |

The Pass 1 steering-weight kernel in the V3 suite evaluates a phase $\phi = k \cdot (\mathbf{p} \cdot \hat{\mathbf{n}})$ where $\phi$ can be very large (e.g. $\phi \sim 2 \pi \cdot 200 \cdot 20$ at $N_{\text{ant}}=64$, $f \sim 800$ MHz, geometry scale $\sim 20$ m). Cody-Waite range reduction is mandatory for that argument magnitude, which means **preferring `sincosf` (libdevice) over `__sincosf`** for the high-frequency end of the band.

**Implication.** The right choice is **frequency-banded**:

- $f < 100$ MHz (small $\phi$): `__sincosf` is safe, ~4× faster than libdevice.
- $f \in [100, 400]$ MHz: `sincospif` if $\phi$ is a multiple of $\pi$ (it isn't in general).
- $f > 400$ MHz: `sincosf` (libdevice) or Cody-Waite table-lookup.

The current V3 path uses one global setting; this is the single highest-value per-precision optimization on the steering-weight side.

### 1.5 Half / BF16: tempting, but catastrophic for steering weights

`__half` and `__nv_bfloat16` both lose phase precision dramatically:

- `__half` has 10 mantissa bits. At $\phi \sim 10^3$, the ULP at that magnitude is $\sim 2^{10} \cdot 2^{-10} = 1$ rad — useless for a steering weight.
- `__nv_bfloat16` has 7 mantissa bits. Worse.

If the design ever shifts to half precision for weights, the cumulative phase drift over a 64-element complex MAC is roughly $\pm 1$ rad RMS — comparable to the entire phase rotation. **Do not quantize steering weights below `float32`.** This is well-established in the radio-astronomy literature (§6).

### 1.6 Denormals and FTZ

CHIME/CHARTS voltage samples are quantized signed nibbles — they are **never** denormal. Steering weights are products of $k$ (well-conditioned) and dot products (geometric) and stay well within normalized range. Intensities $I = sr^2 + si^2$ saturate at $N_{\text{ant}} \cdot N_{\text{levels}}^2 = 64 \cdot 49 = 3136$ which is fine.

**Conclusion.** `-ftz=true` is safe and slightly faster; recommend keeping default `-ftz=false` to avoid silent changes for downstream that might compare against an IEEE-correct CPU baseline.

### 1.7 Summary of practical choices for V3

| Stage | Recommended numeric path | Why |
|---|---|---|
| Unpack `int4` → `float` | `bfe.s32` (PTX) then `cvt.f32.s32` | See §5 |
| Steering weight `cosf/sinf` | `__sincosf` for low-freq, libdevice for high-freq | See §1.4 |
| Per-cell MAC | `fmaf` (default) | Maximize throughput; documented 1 ULP tolerance |
| Final intensity $I = sr^2 + si^2$ | `(float)sqrt` or skip (we want intensity) | Drop sqrt — keep $sr^2 + si^2$ as $f_{32}$ |
| Accumulator precision | `float` | $N_{\text{ant}} \cdot N_{\text{levels}}^2$ fits comfortably in $f_{32}$; double would only help if gain were extreme |

---

## 2. Accumulator Patterns for Voltage Beamforming

The dominant hot-path arithmetic is the per-$(t,f)$ complex multiply-accumulate across $N_{\text{ant}}$ elements:

$$
s_r(t,f) + j\,s_i(t,f) = \sum_{a=0}^{N_{\text{ant}}-1} \bigl(w_r(b,f,a) + j\,w_i(b,f,a)\bigr)^\* \cdot \bigl(x_r(t,f,a) + j\,x_i(t,f,a)\bigr)
$$

with intensity $I(t,f) = s_r^2 + s_i^2$. All quantities are real-valued `float`.

### 2.1 Naive serial accumulator

```cpp
float sr = 0.f, si = 0.f;
for (int a = 0; a < N_ANT; ++a) {
    float xr = unpack_real(packed[t, f, a]);
    float xi = unpack_imag(packed[t, f, a]);
    float wr = weights[b, f, a].real();
    float wi = weights[b, f, a].imag();
    sr = fmaf(wr, xr, fmaf(wi, xi, sr));   // (wr*xr - wi*xi) folded
    si = fmaf(wr, xi, fmaf(-wi, xr, si));  // (wr*xi + wi*xr)
}
float I = fmaf(sr, sr, si*si);
```

This is what the CPU naive baseline does ([`src/beam_tracker.cpp`](src/beam_tracker.cpp)) and is bit-exactly reproducible across naive / v1 / v2 because v1/v2 only change layout, not arithmetic. On GPU it costs $\sim 5$ FLOPs per element + 1 int-decode, fully FMA-saturated, and is register-resident for the entire 64-iteration loop.

### 2.2 Two-pass split: real and imag accumulators

The standard rearrangement:

$$
s_r = \sum_a (w_r x_r - w_i x_i),\qquad s_i = \sum_a (w_r x_i + w_i x_r)
$$

keeps the two accumulators in independent registers, never spilling, never aliasing. This is the form V3 emits and it is also exactly the form that prevents FMA contraction across the `wr*xr - wi*xi` line — `nvcc` will generate two FMAs plus one subtract, or, if `-fmad=false`, one MUL + one FMA + one SUB. The second form has one extra rounding and is what the CPU naive code does, which is one of the small reasons GPU bit-exactness is impossible without `-fmad=false`.

### 2.3 Tree reduction within a warp

For a per-cell accumulator that lives across many lanes (not applicable here for $N_{\text{ant}}=64$ since one thread owns one cell, but applicable for spatial smoothing or beam-coherent addition across sub-arrays), the canonical pattern is:

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    sr += __shfl_down_sync(0xffffffff, sr, offset);
    si += __shfl_down_sync(0xffffffff, si, offset);
}
```

This is what [`src/cuda_beam_tracker_fused_warp_shuffle.cu`](src/cuda_beam_tracker_fused_warp_shuffle.cu) does. **Precision property:** warp shuffles move IEEE bits between registers with no rounding, so the reduction order is deterministic across runs (modulo sub-warp scheduling). The accumulation order is power-of-two tree, which is the most numerically stable pairing for fixed-point-style sums (better than serial from one end).

### 2.4 Kahan / Neumaier compensation in the inner loop?

For $N_{\text{ant}} = 64$ and $|w|, |x| \le 8$, the worst-case dynamic range of $s_r$ is $\sim 64 \cdot 64 = 4096$, well inside $f_{32}$'s normalized range ($\sim 10^{38}$). Kahan summation buys nothing here — naive summation is at most $\approx N_{\text{ant}} \cdot \epsilon_{\text{machine}} \cdot |s_r| \approx 64 \cdot 1.19\times10^{-7} \cdot 4096 \approx 3.1 \times 10^{-2}$ worst-case relative error, and the dominant error term is **uncorrelated uniform random** across cells (sign-noise from the FMA ordering), so it averages down by $\sqrt{N}$ to $\sim 4 \times 10^{-3}$ RMS — still acceptable.

Where Kahan **does** matter is integration:

$$
I_{\text{integrated}}[f] = \sum_{w=0}^{W-1} I[t_w, f]
$$

With $W = 320$ and $I \sim 10^{3}$, naive summation accumulates $\sim 320 \cdot 10^{3} \cdot 1.19\times10^{-7} \approx 3.8\times10^{-2}$ absolute error per cell. That's $4\%$ relative on a $\sim 10^3$ signal — already bordering the noise floor of typical pulsar/FRB detection thresholds. **The temporal-integration stage ([`src/temporal_integration.cpp`](src/temporal_integration.cpp)) should use either compensated summation or, more cheaply, an `int32`/`int64` accumulator** if the inputs are quantized (e.g. `int8` after [`src/cuda_quantize_int8.cu`](src/cuda_quantize_int8.cu)). This is currently the project's most likely silent precision regression.

### 2.5 Block-strided / atomics

If the design ever shifts to multiple threads cooperating on one $(t,f)$ cell (e.g. for a larger $N_{\text{ant}}$ or for spatial smoothing sub-apertures), accumulation across threads must use either:

- **Shared-memory partial sums + one warp reduce + one atomicAdd to a global `float`** — most common.
- **`atomicAdd(float*, float)` directly to global** — available since SM 6.0 (Pascal). Atomic float add is not associative: $a+b+c$ and $(a+b)+c$ can differ by 1 ULP depending on contention order.
- **Block-level reduction in `double`** — a precision upgrade, costing ~50% throughput.

For CHARTS' fixed $N_{\text{ant}} \in \{32, 64\}$ there is no need to atomically accumulate across threads; the per-cell reduction is fully intra-warp. This is a feature, not an oversight.

### 2.6 Mixed-precision inner accumulators

A common pattern in deep learning (and occasionally worth considering in beamforming) is to use `half` or `bf16` for the multiply and `float` for the accumulate. On NVIDIA GPUs this maps to the HMMA / IMMA tensor-core paths or to software emulation.

For voltage beamforming this is **not** a win because:

- The product $w \cdot x$ stays well inside $f_{32}$'s representable range — no overflow risk.
- Tensor-core HMMA is a $16\times 16\times 16$ MMA; mapping a 64-element inner product to it requires reshape gymnastics that cost more than they save.
- Mixed-precision loses the FMA-fused single-rounding property unless paired with a careful partial-sum strategy.

**Conclusion.** Stay with single-precision `float` + `fmaf` end-to-end.

### 2.7 Accumulator pre-conditioning for very long integrations

If a future feature extends $W$ (window count, "tracklets") to $\gg 320$, consider:

- Block-floating-point: track an exponent per chunk and accumulate aligned mantissas.
- `int32`/`int64` accumulators when $I$ is quantized.
- Log-domain accumulation for very long time constants ($\log I_{\text{sum}} = \log \sum I \approx \log I_{\text{max}} + \log(1 + \sum e^{\log I_i - \log I_{\text{max}}})$, stable across dynamic range).

These are not needed for the current CHARTS cadence.

---

## 3. Phase Arithmetic Alternatives

The Pass 1 phase kernel computes:

$$
\phi_{w,f,a} = k_f \cdot (\hat{\mathbf{n}}_w \cdot \mathbf{p}_a),\qquad
w_r + j\,w_i = \exp(-j \phi_{w,f,a})
$$

with $k_f = 2\pi f / c$. The arguments are:

- $f \in [400, 800]\,\text{MHz}$ — wavenumber $k_f \in [8.4, 16.7]\,\text{rad/m}$
- $\|\mathbf{p}_a\| \le 100$ m for CHIME-scale cylindrical geometry
- $\hat{\mathbf{n}}_w \cdot \mathbf{p}_a \in [-100, 100]$ m
- $\phi_{w,f,a} \in [-2\pi \cdot 1670, 2\pi \cdot 1670]$ ≈ $\pm 10500$ rad

This is a **large-argument transcendental** problem; naive `sincosf(phi)` will reduce incorrectly without Cody-Waite-style range reduction.

### 3.1 Direct `sincosf(phi)` (libdevice, slow but accurate)

`phi` is computed in `double` (the natural format for the geometric inner product), then narrowed to `float`, then passed to `sincosf`. Range reduction is performed by libdevice using Cody-Waite + Payne-Hanek tables. Accuracy: $< 1$ ULP globally. Cost: ~50–80 SP FLOPs per evaluation.

### 3.2 `__sincosf(phi)` (SFU, fast)

Single SFU op. Hardware range reduction is a partial-Cody-Waite with bounded tables — accurate to $\le 2$ ULP only for $|\phi| \lesssim 2^{20} \approx 10^6$, which actually does cover our $\pm 10^4$ rad regime. **However**, accuracy near multiples of $2\pi$ degrades sharply because the SFU's argument-reduction modulo $2\pi$ has limited precision. For steering weights at $N_{\text{ant}}=64$, an error of $\pm 10^{-4}$ rad in phase produces an intensity error of $\pm 2 \cdot 10^{-4}$ relative — acceptable.

### 3.3 `sincospif(phi / pi)` (libdevice)

Reduces the argument modulo $\pi$ rather than $2\pi$, which is faster than full `sincosf` but introduces a $\div \pi$ (or rather, the libdevice trick uses an extended Cody-Waite table for $\pi$). Marginally faster than `sincosf`, similar accuracy.

### 3.4 Cordic on the GPU

Cordic converges in $O(\log_2 \text{precision})$ iterations; for $f_{32}$ precision (~24 bits), ~24 iterations of Cordic in **rotation mode** gives $\le 2$ ULP. Each iteration is 3 adds + 1 shift, which is fast on SFU-less GPUs but on modern NVIDIA hardware the SFU `__sincosf` is already 1 cycle/thread and Cordic is slower.

**Verdict:** not worth it for this project unless porting to a SFU-less target (FPGA, older GPUs, mobile).

### 3.5 Lookup tables (LUTs) with linear interpolation

A table of size $2^{14}$–$2^{16}$ with linear interpolation can hit $\sim 12$–$16$ bit phase accuracy, which is roughly equivalent to `__sincosf`. Memory cost: 64–256 KiB constant memory. Branch-free. Useful for embedded or SFU-limited targets; not a clear win on NVIDIA.

### 3.6 Phase as complex multiply (incremental update)

If the steering direction changes by small amounts between windows — which it does for moving sources — we can keep $w$ as a complex number and **rotate it incrementally** rather than recomputing $\phi$ from scratch:

$$
w_{t+1} = w_t \cdot e^{-j \Delta\phi},\qquad
\Delta\phi = k_f \cdot (\hat{\mathbf{n}}_{t+1} - \hat{\mathbf{n}}_t) \cdot \mathbf{p}_a
$$

This converts every transcendental call into one complex multiply (4 FMAs) per (window, freq, ant). For typical sidereal source motion $\Delta\phi \sim 10^{-3}$ rad per ms, the cumulative phase drift over the inner loop stays tiny and `__sincosf` for the per-window anchor is fine. **This is a strong recommendation for V4** because it eliminates the transcendental entirely from the inner per-time loop.

### 3.7 Phase as Taylor expansion in $\mathbf{p} \cdot \Delta \hat{\mathbf{n}}$

For sources that move very slowly, a Taylor expansion of $\phi$ in the direction vector $\hat{\mathbf{n}}(t)$ gives a polynomial in $t$, and the steering weight becomes $\exp(-j (a_0 + a_1 t + a_2 t^2))$. Expanding in $t$ reduces the transcendental frequency from per-time-sample to per-window (for the anchor $a_0$) plus cheap polynomial evaluation.

### 3.8 Computing $\phi$ in `double` and narrowing once

Currently the inner product $\hat{\mathbf{n}} \cdot \mathbf{p}$ is computed in `double` for accuracy over a $\sim 100$ m geometry. This is correct. **Do not** narrow $\phi$ to `float` and then widen for the transcendental call — that loses precision. Pass `double` directly to `sincos(double)` or cast once at the trig call.

### 3.9 Numerical bottom line for the project

| Frequency band | Recommended call | Rationale |
|---|---|---|
| $f < 100$ MHz | `__sincosf((float)phi, &s, &c)` | Small $\phi$; SFU accuracy is plenty |
| $100 \le f < 400$ MHz | `__sincosf` if band-pad is good, else `sincosf` | Trade-off by band |
| $f \ge 400$ MHz | `sincosf` libdevice | Large $\phi$, full Cody-Waite required |
| All $f$, moving sources | Incremental complex-multiply (§3.6) | Best of all worlds; SFU only at window anchor |

---

