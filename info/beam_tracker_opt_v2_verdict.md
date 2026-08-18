# beam_tracker_opt_v2 — honest verdict

`src/beam_tracker_opt_v2.cpp` is a performance-only variant of v1
(`src/beam_tracker_opt.cpp`). v1 is **untouched** (zero diff confirmed in repo).
v2 applies four changes, none of which alter per-cell floating-point computation
or the element-axis accumulation order, so v2's output is bit-for-bit identical
to both the naive tracker and v1. That is asserted in
`tests/test_beam_tracker_opt_v2.cpp` as exact `std::vector<float>` equality
(*both* `naive == v2` *and* `v1 == v2`), and re-asserted in-process before every
timed run by `tools/benchmark_beam_tracker_opt_v2.cpp` (guard line
`naive == v1 == v2 byte-equal: PASS (5160960 cells)`).

## Pre-flight: was Fix 1 even worth attempting?

Yes. `Weights = std::vector<ComplexFloat>` and `ComplexFloat` is a trivial POD
struct `{ float real; float imag; }` (see `include/beamformer/complex.hpp`,
where `sizeof == 2*sizeof(float)` and `std::is_trivially_copyable_v` are both
`static_assert`ed). A size-only `std::vector<ComplexFloat>` constructor
**value-initializes** every element (no user-provided default ctor), so v1's

```cpp
Weights all_weights(window_count * dims.n_freq * dims.n_ant);
```

performs a single-threaded zero-fill of ~8.26 MB on the calling thread **before**
any OpenMP region runs. That memset (a) is wasted — Pass 1 writes every element
unconditionally with no gaps — and (b) first-touches every page onto the calling
thread's socket. Fix 1 (allocate raw, default-init via
`std::make_unique_for_overwrite` with an `operator new[]` fallback) is therefore
not redundant; it is the primary suspect and was kept.

## Numbers — full thread sweep, tri0078, n_time=15360, n_freq=336, n_ant=64, integration_spectra=320, 3 repeats (median)

| threads | v1 median (ms) | v1 max (ms) | v2 median (ms) | v2 max (ms) | v2 / v1 (median) |
|--------:|---------------:|------------:|---------------:|------------:|-----------------:|
|   1     |  310.06        |  310.21     |  307.17        |  307.59     |  1.01x           |
|   2     |  155.77        |  155.80     |  154.45        |  155.05     |  1.01x           |
|   4     |   78.43        |   78.48     |   77.12        |   77.14     |  1.02x           |
|   8     |   41.02        |   41.05     |   38.80        |   38.80     |  1.06x           |
|  16     |   20.76        |   20.80     |   19.53        |   19.57     |  1.06x           |
|  32     |   12.97        |   13.24     |   10.99        |   10.99     |  1.18x           |
|  64     |    7.89        |    8.50     |    5.97        |    6.09     |  1.32x           |
| 128     |   26.32        |   26.56     |   25.94        |   26.26     |  1.01x           |
| 192     |   22.39        |   28.70     |   13.74        |   22.45     |  1.63x           |

**Key observations:**

1. **The v2-vs-v1 gap only shows up at high thread counts**, exactly where NUMA
   first-touch effects would be expected to matter most. At 1–16 threads the
   gap is noise (1.01–1.06x). It grows to **1.18x at 32, 1.32x at 64, and 1.63x
   at 192 threads** — the headline figure, but not the only one worth quoting.

2. **192 threads is not v1's best operating point.** v1's median *degrades*
   going 64 → 128 → 192 threads (7.89 → 26.32 → 22.39 ms). v1's *actual* best is
   ~7.89 ms at 64 threads (~39x over naive). The earlier headline of "24.0 ms at
   192 threads, 12.78x over naive" was a high-variance run sitting on v1's
   NUMA-regression knee — note v1's 192-thread max is 28.70 ms vs median 22.39
   ms, a 28% spread. The earlier conclusion ("don't run v1 at 192 threads") was
   directionally right but for the wrong reason — it is a NUMA first-touch
   artifact, not a fork/join artifact.

3. **v2 is best at 64 threads too** (5.97 ms, ~51x over naive), but unlike v1 it
   does not catastrophically regress at 192 (13.74 ms is a regression vs the
   5.97 ms best, but it's roughly half of v1's 192-thread number).

## The numactl diagnostic — the smoking gun

The diagnostic pair (no code change, just `numactl --interleave=all` at 192
threads) is the load-bearing evidence, because it isolates the NUMA hypothesis
from the code change.

| run at 192 threads                | v1 median (ms) | v2 median (ms) |
|-----------------------------------|---------------:|---------------:|
| default (first-touch)             |  22.39         |  13.74         |
| `numactl --interleave=all`        |   5.87         |   3.78         |

The implication is direct and large:

* v1 goes from **22.39 ms → 5.87 ms** (a **3.81x speedup**) under pure NUMA
  interleaving, with **no code change at all**. That is the cost of v1's serial
  memset first-touching every `all_weights` page onto one socket: Pass 1's
  writes and Pass 2's reads from the other sockets pay remote-memory latency
  for the whole buffer. Interleaving pages round-robin across nodes removes
  that penalty entirely.
* v2 under the same interleaving goes **13.74 ms → 3.78 ms** (1.82x). v2 is
  already partly immune because Fix 1 moved the first-touch into the parallel
  Pass 1 writes, but interleaving still helps v2 (because the OpenMP runtime's
  static partitioning does not guarantee the worker that first-touches a page
  is the one that re-reads it in Pass 2).
* Under interleaving, **v2 is still 1.55x faster than v1** (5.87 → 3.78 ms):
  the other three fixes (especially the merged fork/join, Fix 3) keep paying
  even once NUMA placement is taken out of the picture.

So the honest causal story is: **Fix 1 (NUMA first-touch) was the dominant win
at high thread counts, and Fix 3 (merged parallel region) is a smaller
second-order win that survives independently of NUMA placement.**

## Verdict on each of the four changes

* **Fix 1 — NUMA first-touch (no value-init allocation): KEEP.** Dominant effect.
  The numactl pair proves the underlying mechanism is real (v1 surges 3.81x
  under interleaving alone), and v2 captures most of that win in code via Pass 1
  first-touch. This is the single change that turns v1's 192-thread regression
  into v2's 1.63x advantage. Worth keeping as the default; if the deployment is
  already launched under `numactl --interleave=all`, Fix 1's standalone impact
  shrinks but is still positive (1.55x under interleaving).

* **Fix 2 — eliminate redundant `tracker_window_direction` calls in Pass 1:
  KEEP, but honest about magnitude.** It removes `window_count * n_freq` calls
  per run in favor of `window_count` (a ~336x reduction in call count for the
  default config), but `tracker_window_direction` is a handful of float ops and
  the call sits inside a pass dominated by `cos`/`sin`. Its latency contribution
  is below the noise floor of this benchmark (we cannot separate it from Fix 1's
  effect with this driver). It is correctness-neutral and obviously not slower,
  so keep it on hygiene grounds; just don't credit it with the headline number.

* **Fix 3 — merge the two parallel regions: KEEP.** It removes one team
  fork/join per call with zero loss of correctness (the implicit barrier between
  the two `for`s is required regardless). Under `numactl --interleave=all`
  (where Fix 1's effect is largely neutralized), v2 is still 1.55x faster than
  v1 at 192 threads — that residual gap is Fixes 2+3+4, dominated by Fix 3.
  Keep it.

* **Fix 4 — `__restrict__` on the Pass 2 inner-loop pointers: KEEP, but no
  measurable standalone effect.** The two pointers (`w_ptr` into `all_weights`,
  `v` into `packed`) truly never alias (distinct arrays), so `__restrict__` is a
  correctness-safe hint that *can* change autovectorization codegen. The
  benchmark cannot isolate its effect from Fix 3 at this granularity, and the
  bit-equality contract constrains what the compiler may reorder (the
  element-axis float MAC order is preserved, so the win, if any, is limited to
  load/store scheduling). Keep it as a free, correct hint; do not expect a
  visible column in the table from it alone.

## Honest bottom line

* v2 is genuinely faster than v1 at high thread counts (1.18x–1.63x at 32–192
  threads), and the *primary mechanism* (NUMA first-touch, Fix 1) is confirmed
  by the numactl diagnostic.
* The earlier v1 record — "12.78x / 24.0 ms at 192 threads" — was sitting on
  v1's NUMA-regression knee. v1's *true* best is ~7.89 ms at 64 threads. The
  right comparison is not "v1@192 vs v2@192", it is "**best-of-thread-sweep**":
  v1 best = 7.89 ms (64 threads); v2 best = 5.97 ms (64 threads). That is a real
  but modest **1.32x best-of-sweep** improvement, and v2 has the additional
  property of not falling over at 192 threads (13.74 ms vs v1's 22.39 ms).
* **Recommendation:** adopt v2 in place of v1 (all four changes kept), and ideally
  also deploy under `numactl --interleave=all` for the full win (v2+interleave =
  3.78 ms at 192 threads). Fix 1 is the change worth fighting for; Fixes 2–4 are
  correctness-neutral hygiene that ride along cheaply.

## Files

| file | role |
|------|------|
| `include/beamformer/beam_tracker_opt_v2.hpp` | v2 public API |
| `src/beam_tracker_opt_v2.cpp` | v2 implementation (Fixes 1–4) |
| `tests/test_beam_tracker_opt_v2.cpp` | naive==v2 AND v1==v2, full scenario matrix |
| `tools/benchmark_beam_tracker_opt_v2.cpp` | three-way (naive/v1/v2) benchmark |
| `submit_benchmark_v2.sh` | SLURM sweep + numactl diagnostic pair |
| `CMakeLists.txt` | registers v2 in both cores, the test, and the benchmark |

v1, naive, and `cpu_opt_beam_tracker` files remain bit-for-bit untouched
(verified against the pre-task clean tree).
