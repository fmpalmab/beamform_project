Short version: the *algorithm* is about as GPU-friendly as it gets — but whether porting it actually pays off comes down to a data-movement calculation that has nothing to do with FLOPs, and the numbers aren't obviously in CUDA's favor here.

## Why the algorithm maps well to a GPU

Both passes in `beam_tracker_opt_v2_cpu_packed_intensity_into` are exactly the shape GPUs like:

- **Pass 1** writes one disjoint `ComplexFloat` slot per `(window, freq, element)` — no cross-iteration dependency, no atomics, no reduction. It's a pure "one thread → one output cell" trig kernel.
- **Pass 2** writes one disjoint `float` per `(time, freq)` cell, and each thread's work is a tiny, fixed-length (`n_ant=64`) local reduction over registers. Also disjoint writes, no atomics.

This is structurally identical to the delay-and-sum beamformer kernels used in real radio-interferometer backends (MWA/CHIME/ASKAP-style GPU beamforming) — packed low-bit-width voltage samples in, per-(window,freq) steering weights, per-cell complex dot product out. It's a well-trodden CUDA pattern, not a research problem.

**Kernel sketch:**

```cpp
// Pass 1: grid = (window_count, n_freq), block = n_ant threads
__global__ void steering_weights(const Vec3* window_dirs, const double* wavenumbers,
                                  const Vec3* positions, float2* all_weights, ...) {
    int w = blockIdx.x, f = blockIdx.y, e = threadIdx.x;
    Vec3 dir = window_dirs[w];
    double delay = dot(positions[e], dir);
    double phase = wavenumbers[f] * delay;
    float s, c;
    __sincosf((float)phase, &s, &c);          // single SFU op for both
    all_weights[(w*n_freq+f)*n_ant + e] = {c, s};
}

// Pass 2: one thread per (time, freq) output cell
__global__ void accumulate(const float2* weights, const uint8_t* packed,
                            float* intensity, ...) {
    int cell = blockIdx.x*blockDim.x + threadIdx.x;   // maps to (t,f)
    float sr=0, si=0;
    #pragma unroll
    for (int e = 0; e < n_ant; ++e) {
        auto samp = unpack_complex_int4(packed[voltage_base+e]);
        float2 wt = weights[weight_base+e];
        sr += wt.x*samp.real - wt.y*samp.imag;
        si += wt.x*samp.imag + wt.y*samp.real;
    }
    intensity[cell] = sr*sr + si*si;
}
```

Straightforward, coalesced, high occupancy. **This part is not the hard problem.**

## The bit-exactness constraint breaks first

Your whole v2 verdict is built on `naive == v1 == v2` byte-equality. A GPU port almost certainly **cannot** preserve that:
- `__sincosf`/`__cosf`/`__sinf` are lower-precision hardware approximations, not libm-equivalent — even the accurate `cosf`/`sinf` on GPU won't match glibc's `cos`/`sin` to the last bit.
- `nvcc` fuses `a*b+c` into a single FMA by default, which changes rounding vs. the CPU's separate mul/add — you'd need `-fmad=false` and still not be guaranteed a match.

So the first decision is: does downstream tolerate an epsilon-based comparison (~1e-6 relative) instead of byte-equality? If bit-exact CPU/GPU parity is a hard requirement, this stops being a performance question.

## The actual performance argument: data movement, not FLOPs

Compute is trivial for a GPU. Pass 2 is ~330M complex MACs (~2.6 GFLOP) reading ~330 MB of packed voltage; Pass 1 is ~1M cheap trig evaluations. On any current data-center GPU, once the data is *sitting in HBM*, both passes together run in well under 0.5 ms — compute was never going to be the bottleneck.

The bottleneck is getting `packed` (n_time×n_freq×n_ant = 5,160,960 × 64 bytes ≈ **315 MiB**) onto the GPU and `intensity` (≈**19.7 MiB**) back off it, once per frame:

| Link | Realistic sustained BW | H2D time for 315 MiB | vs. current CPU best |
|---|---:|---:|---|
| PCIe3 x16 | ~12 GB/s | ~27.5 ms | 4.6–7.3x slower than 5.97/3.78 ms |
| PCIe4 x16 | ~25 GB/s | ~13.2 ms | 2.2–3.5x slower |
| PCIe5 x16 | ~50 GB/s | ~6.6 ms | roughly break-even |
| NVLink-C2C (Grace-Hopper class) | ~450 GB/s | ~0.7 ms | actually wins |

That's H2D alone, before adding the D2H copy of `intensity` or any kernel-launch/stream overhead. Your CPU already does the *entire computation* — Pass 1 + Pass 2 + everything — in 5.97 ms (or 3.78 ms under `numactl --interleave=all`). A commodity-GPU port over PCIe3/4 would spend more time just moving bytes across the bus than v2 spends computing the whole answer today.

Overlapping transfer with compute across frames (double-buffered streams) doesn't rescue this: with transfer (~13 ms) an order of magnitude larger than GPU compute (<0.5 ms), steady-state throughput becomes transfer-bound at ~76 frames/s (PCIe4) — worse than the CPU's current ~168–265 frames/s.

## When it would actually win

Only if the host round-trip is removed from the critical path:
- **Ingest directly into GPU memory** (GPUDirect RDMA from whatever digitizer/NIC produces `packed`, bypassing host RAM entirely) — this is exactly how production radio-beamforming pipelines avoid this problem, and it's the only regime where your stated 0.5 ms/frame target becomes physically plausible.
- **Downstream also GPU-resident**, so `intensity` never comes back to the host either.
- **NVLink-class interconnect** (Grace-Hopper/Grace-Blackwell unified memory) instead of PCIe.
- Or the batch shape changes fundamentally — e.g., if many frames could be transferred and processed as one large batch, amortizing a fixed transfer-setup cost — but your data volume *per frame* is the whole bottleneck, not per-transfer overhead, so batching doesn't help here the way it would for small-message latency problems.

## What I'd do before writing a single line of CUDA

Run the same kind of cheap, isolating diagnostic you did with `numactl`: allocate 315 MiB pinned host memory, `cudaMemcpy` it to device and back on your actual target GPU/PCIe generation, and time it. That one number tells you immediately whether this is worth pursuing — if it's anywhere near or above 5.97 ms, the CUDA port is a net loss for a host-originated, single-frame workload, full stop, and the honest recommendation is to stay on CPU and pursue the AVX-512/vectorization route your `cpu_opt_beam_tracker` notes already flagged, unless the data-ingest architecture changes.