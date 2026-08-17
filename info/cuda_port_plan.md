# CUDA Port Implementation Plan: Beam Tracker Opt v2

## 1. Context and Objective

This document outlines the strategy for creating a CUDA implementation of the
beam tracking algorithm currently defined in `beam_tracker_opt_v2.cpp`. The CPU
implementation uses OpenMP and executes in two passes:

1. **Pass 1:** Precomputes a massive steering-weight table per
   `(window, freq, antenna)` from the trajectory-derived direction.
2. **Pass 2:** Performs the actual beamforming by computing the dot product of
   the 64-antenna weight vector against unpacked `int4` voltage samples, yielding
   `|sum|^2` for each `(window, time, freq)` cell.

The goal is to translate this to CUDA, starting with a 1:1 naive port to
guarantee bit-for-bit correctness, followed by structured optimizations.

## 2. Key Findings & Architectural Insights

* **Thread Scaling (CPU vs. GPU):** The CPU implementation peaked at 64 threads.
  GPUs require thousands of resident threads to hide memory latency. The
  algorithm parallelizes perfectly per output cell with zero cross-cell state.
* **The 64-Antenna Advantage:** `n_ant = 64` aligns with exactly 2 CUDA warps.
  Future optimizations should map the antenna axis to warp-shuffle reductions
  (`__shfl_down_sync()`).
* **Memory Movement Reality:** H2D transfer of packed voltage is an unavoidable
  production reality. Treat kernel correctness and pipeline memory movement as
  separate concerns; do not synthesize data on-device during kernel development.
* **Elimination of `all_weights` Phase 2:** On the GPU, recomputing `cos`/`sin`
  on the fly in registers bypasses global memory latency.

## 3. Resolved Formulas (bit-for-bit reproducibility)

Reproduced here so the kernels match the CPU path exactly:

* Unpack (`int4.hpp`): real = low nibble, imag = high nibble, via
  `decode_signed_nibble`.
* `intensity_index(t,f,b,dims) = (t*n_freq + f)*n_beams + b`.
* `speed_of_light_m_per_s = 299792458.0`; `two_pi = 6.283185307179586...`.
* `default_positions`: `n_ant==32` -> 4x8, `n_ant==64` -> 8x8, spacing 0.6m,
  `{col*spacing, row*spacing, 0}`.
* `channelized_frequencies`: start 300e6, width 300e3: `f[c] = 300e6 + c*300e3`.
* `tracker_window_count(n_time, I) = (n_time + I - 1)/I`.
* `tracker_window_direction(traj, w, I) = direction_from_lm(start.l + (w*I)*rate.l, start.m + (w*I)*rate.m)`.
* `direction_from_lm(l,m) = {l, m, sqrtf(1 - l*l - m*m)}` (throws if not finite or
  l*l+m*m > 1; host pre-scan must run this serial before kernel launch).
* Phase: `phase = (two_pi * f / c) * (pos.x*dir.x + pos.y*dir.y + pos.z*dir.z)`
  computed in `double`, then `cosf/sinf` cast to `float` for storage; MAC in
  `float`.

## 4. Steps

### Step 1: Read Dependency Headers  (DONE)
### Step 2: Implement `__device__` Helpers
### Step 3: Phase 1 — Literal 1:1 Port (correctness)
### Step 4: Phase 2 — Kernel Fusion (performance)
### Step 5: Phase 3 — Warp Reductions (advanced)

See `src/cuda_beamformer.cu` for established CUDA conventions in this repo.
