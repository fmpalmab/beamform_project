// tools/benchmark_cuda_tracker_v2.cpp
//
// Benchmark: CPU vs CUDA tracker implementations.
// Sweeps through CPU Naive, CPU Opt v1, CPU Opt v2, and the three CUDA v2
// kernels (TwoPass, Fused, WarpReduction).
//
// Re-asserts the tolerance equality contract in-process before timing,
// ensuring we do not benchmark fast-but-wrong kernels.
//
// CSV OUTPUTS (written under --outdir, prefix "benchmark_cuda_tracker_v2")
// ---------------------------------------------------------------------------
//   *_summary.csv           One row per invocation, APPENDED across runs, so
//                            a thread-count / n_ant sweep (see
//                            scripts/run_tracker_comparison_benchmarks.sh)
//                            accumulates into a single comparison table.
//                            Columns: n_time,n_ant,n_freq,integration_spectra,
//                            threads,naive_ms,cpu_v1_ms,cpu_v2_ms,
//                            cuda_twopass_ms,cuda_fused_ms,cuda_warp_ms,
//                            speedup_v2_vs_naive,speedup_gpu_warp_vs_cpu_v2
//   *_frame_latencies.csv   Per-integration-window latency for every engine,
//                            for THIS invocation only (TRUNCATED, not
//                            appended). A window-index axis is only
//                            meaningful for a single (n_ant, threads)
//                            configuration -- mixing sweep points into one
//                            file would make window_index ambiguous across
//                            rows. In a sweep, this file reflects whichever
//                            run completed most recently.
//                            Columns: window_index,time_start,engine,kernel,
//                            latency_ms
//   *_validation.csv        Numerical parity of every non-naive engine
//                            against the CPU Naive reference over the FULL
//                            sequence, for THIS invocation (TRUNCATED, same
//                            reasoning as frame_latencies).
//                            Columns: engine_name,max_abs_diff,rms_diff,
//                            rel_error,max_rel_error,passed_tolerance
//   *_window_validation.csv EXTENSION beyond the three files above. The
//                            plotting dashboard's error/drift-over-time panel
//                            and its power-profile-parity panel both need a
//                            per-window (not per-run-aggregate, not
//                            per-sample) breakdown, which *_validation.csv's
//                            single aggregate row per engine cannot supply.
//                            Reuses the full-sequence result buffers already
//                            held in memory from the guard check, so it costs
//                            one extra pass over already-computed data (no
//                            extra tracker calls). Includes ALL SIX engines
//                            (naive included, trivially 0 diff against
//                            itself) so the power-profile panel can overlay
//                            every implementation on one axis.
//                            Columns: window_index,time_start,engine,kernel,
//                            mean_power,max_abs_diff,rms_diff
//
// Per-window LATENCY is measured with FRESH per-window calls (a
// single-window Dimensions/TrackerConfig slice, sized integration_spectra
// samples, fed the corresponding slice of `packed`) -- NOT a
// total-time-divided-by-window-count estimate. This intentionally includes
// each engine's per-call overhead (OpenMP fork/join for the CPU paths;
// cudaMalloc/H2D/launch/D2H/cudaFree for the CUDA paths, since
// cuda_tracker_v2_packed_intensity_into owns that full round trip per call),
// i.e. it measures the latency a STREAMING pipeline would see if it handed
// one window to the tracker as soon as it arrived -- the quantity the
// 0.5 ms/frame real-time budget in this project's notes is actually about.
// Per-window ERROR/POWER stats, by contrast, are read directly out of the
// batched full-sequence buffers (see *_window_validation.csv above) since
// re-deriving them from isolated per-window calls would just reproduce the
// same per-cell floats at extra cost.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_beamformer.hpp"  // beamformer::CudaDeviceInfo, reused from the
                                            // direct-beamformer benchmark (tools/benchmark_cpu_cuda.cpp)
                                            // for the metadata.json's hardware-info block.
#include "beamformer/cuda_tracker_v2.hpp"
#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using beamformer::CudaTrackerKernelV2;
using beamformer::Dimensions;
using beamformer::Intensities;
using beamformer::PackedVoltage;
using beamformer::TrackerConfig;
using beamformer::Vec3;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 1;
    std::size_t repeat = 5;
    std::size_t window_repeats = 3;
    std::size_t threads = 0;
    float source_l0 = 0.0F;
    float source_m0 = 0.0F;
    float source_dl = 1.0e-5F;
    float source_dm = 0.0F;
    float prior_l0 = 0.0F;
    float prior_m0 = 0.0F;
    float prior_dl = 1.0e-5F;
    float prior_dm = 0.0F;
    std::optional<std::filesystem::path> outdir;
};

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* require_value(const int argc, char** argv, int& i) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    }
    return argv[++i];
}

std::size_t parse_size(const char* v, const char* opt) {
    std::size_t used = 0;
    const auto parsed = std::stoull(std::string(v), &used);
    if (used != std::string(v).size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + opt);
    }
    return static_cast<std::size_t>(parsed);
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  Benchmark CPU vs CUDA beam trackers; writes summary, per-window\n"
        << "  latency, and validation CSVs under --outdir.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                   32 or 64; default 64\n"
        << "  --integration-spectra N     default 320\n"
        << "  --warmup-runs N             default 1\n"
        << "  --repeat N                  median over repeats for the full-sequence\n"
        << "                              timing; default 5\n"
        << "  --window-repeats N          median over repeats for EACH per-window\n"
        << "                              latency measurement; default 3\n"
        << "  --threads N                 OpenMP threads for CPU opt; 0 = default\n"
        << "  --outdir DIR                write summary/frame/validation CSVs there\n";
}

Options parse_options(const int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--n-time") {
            o.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (a == "--n-ant") {
            o.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (a == "--integration-spectra") {
            o.integration_spectra = parse_size(require_value(argc, argv, i), "--integration-spectra");
        } else if (a == "--warmup-runs") {
            o.warmup_runs = parse_size(require_value(argc, argv, i), "--warmup-runs");
        } else if (a == "--repeat") {
            o.repeat = parse_size(require_value(argc, argv, i), "--repeat");
        } else if (a == "--window-repeats") {
            o.window_repeats = parse_size(require_value(argc, argv, i), "--window-repeats");
        } else if (a == "--threads") {
            o.threads = parse_size(require_value(argc, argv, i), "--threads");
        } else if (a == "--outdir") {
            o.outdir = std::filesystem::path(require_value(argc, argv, i));
        }
    }
    return o;
}

double median_inplace(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

template <typename F>
std::vector<double> time_runs(std::size_t warmup, std::size_t repeat, F&& work) {
    for (std::size_t i = 0; i < warmup; ++i) work();
    std::vector<double> ms;
    ms.reserve(repeat);
    for (std::size_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        work();
        const auto t1 = Clock::now();
        ms.push_back(elapsed_ms(t0, t1));
    }
    return ms;
}

// -----------------------------------------------------------------------
// Numerical diff stats, computed in one pass. Folds in what the original
// tool's check_tolerance() did (same abs_tol/rel_tol dual criterion, same
// pass/fail definition) so there is a single source of truth for "did this
// engine match the naive reference" instead of two loops that could drift
// apart from each other.
// -----------------------------------------------------------------------
struct DiffStats {
    double max_abs_diff = 0.0;
    double rms_diff = 0.0;
    double rel_error = 0.0;      // L2-relative error: rms_diff / rms(ref)
    double max_rel_error = 0.0;  // max_i |diff_i| / max(|ref_i|, abs_tol)
    bool passed_tolerance = true;
};

DiffStats compute_diff_stats(const Intensities& ref, const Intensities& test,
                             const float rel_tol = 1e-4F, const float abs_tol = 1e-5F) {
    DiffStats stats;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double r = static_cast<double>(ref[i]);
        const double t = static_cast<double>(test[i]);
        const double diff = std::abs(r - t);
        stats.max_abs_diff = std::max(stats.max_abs_diff, diff);
        sum_sq_diff += diff * diff;
        sum_sq_ref += r * r;
        const double denom = std::max(std::abs(r), static_cast<double>(abs_tol));
        stats.max_rel_error = std::max(stats.max_rel_error, diff / denom);
        if (diff > static_cast<double>(abs_tol) && diff > static_cast<double>(rel_tol) * r) {
            stats.passed_tolerance = false;
        }
    }
    const auto n = static_cast<double>(ref.size());
    stats.rms_diff = ref.empty() ? 0.0 : std::sqrt(sum_sq_diff / n);
    const double ref_rms = ref.empty() ? 0.0 : std::sqrt(sum_sq_ref / n);
    stats.rel_error = ref_rms > 0.0 ? stats.rms_diff / ref_rms : 0.0;
    return stats;
}

// -----------------------------------------------------------------------
// Engine table. Every CPU/GPU implementation exposes the same
// (packed, dims, tracker, intensity&) signature, so a single table drives
// the guard check, the full-sequence timing, the per-window timing, and the
// per-window validation/power pass -- replacing six hand-unrolled call sites
// with one loop each, and making it a one-line change to add a future
// kernel.
// -----------------------------------------------------------------------
using EngineRunner = std::function<void(const PackedVoltage&, const Dimensions&,
                                        const TrackerConfig&, Intensities&)>;

struct Engine {
    std::string engine_name;  // "cpu" or "gpu"
    std::string kernel_name;  // "naive","opt_v1","opt_v2","twopass","fused","warp_reduction"
    EngineRunner run;
};

std::vector<Engine> make_engines() {
    using beamformer::beam_tracker_cpu_packed_intensity_into;
    using beamformer::beam_tracker_opt_cpu_packed_intensity_into;
    using beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into;
    using beamformer::cuda_tracker_v2_packed_intensity_into;
    using beamformer::cuda_beam_tracker_fused_warp_shuffle_into;

    return {
        {"cpu", "naive",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             beam_tracker_cpu_packed_intensity_into(p, d, t, out);
         }},
        {"cpu", "opt_v1",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             beam_tracker_opt_cpu_packed_intensity_into(p, d, t, out);
         }},
        {"cpu", "opt_v2",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             beam_tracker_opt_v2_cpu_packed_intensity_into(p, d, t, out);
         }},
        {"gpu", "twopass",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             cuda_tracker_v2_packed_intensity_into(p, d, t, out, CudaTrackerKernelV2::TwoPass);
         }},
        {"gpu", "fused",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             cuda_tracker_v2_packed_intensity_into(p, d, t, out, CudaTrackerKernelV2::Fused);
         }},
        {"gpu", "warp_reduction",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             cuda_tracker_v2_packed_intensity_into(p, d, t, out, CudaTrackerKernelV2::WarpReduction);
         }},
        {"gpu", "fused_warp_shuffle",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             cuda_beam_tracker_fused_warp_shuffle_into(p, d, t, out);
         }},
    };
}

// -----------------------------------------------------------------------
// A single-window slice of the full sequence: one integration window's
// worth of packed voltage, re-dimensioned so any engine can be called on it
// in isolation (see file header for why this is timed separately from the
// batched run).
// -----------------------------------------------------------------------
struct WindowSlice {
    Dimensions dims;
    PackedVoltage packed;
    TrackerConfig tracker;
};

WindowSlice make_window_slice(const PackedVoltage& full_packed, const Dimensions& full_dims,
                              const TrackerConfig& full_tracker, const std::size_t window,
                              const std::size_t integration_spectra) {
    const std::size_t first_time = window * integration_spectra;
    const std::size_t window_n_time =
        std::min(integration_spectra, full_dims.n_time - first_time);
    const std::size_t element_count = window_n_time * full_dims.n_freq * full_dims.n_ant;
    const std::size_t offset = first_time * full_dims.n_freq * full_dims.n_ant;

    PackedVoltage window_packed(
        full_packed.begin() + static_cast<std::ptrdiff_t>(offset),
        full_packed.begin() + static_cast<std::ptrdiff_t>(offset + element_count));

    // Direction comes from the SAME trajectory model the batched engines use,
    // evaluated at this window's absolute index -- i.e. exactly the direction
    // a batched call's own per-window loop computes for this same window.
    // The slice's rate is zeroed because it now covers exactly one window:
    // tracker_window_direction(trajectory, 0, window_n_time) with a
    // zero rate reduces to direction_start, reproducing the same steering
    // vector without re-deriving the multi-window rate math here.
    const Vec3 direction = beamformer::tracker_window_direction(
        full_tracker.trajectory, window, integration_spectra);

    TrackerConfig window_tracker = full_tracker;
    window_tracker.trajectory.direction_start = direction;
    window_tracker.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
    window_tracker.integration_spectra = window_n_time;

    return WindowSlice{
        Dimensions{window_n_time, full_dims.n_freq, full_dims.n_ant, full_dims.n_beams},
        std::move(window_packed), window_tracker};
}

// -----------------------------------------------------------------------
// CSV plumbing.
// -----------------------------------------------------------------------
std::filesystem::path with_suffix(const std::filesystem::path& prefix,
                                  const std::string& suffix) {
    return prefix.parent_path() / (prefix.filename().string() + suffix);
}

bool has_content(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::exists(path, error)
           && std::filesystem::file_size(path, error) != 0;
}

void write_summary_csv(const std::filesystem::path& path, const Dimensions& dims,
                       const Options& opts, const int threads, const double n_med,
                       const double v1_med, const double v2_med, const double c2p_med,
                       const double cfus_med, const double cwrp_med, const double cfws_med) {
    const bool exists = has_content(path);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open summary CSV: " + path.string());
    }
    if (!exists) {
        out << "n_time,n_ant,n_freq,integration_spectra,threads,naive_ms,cpu_v1_ms,cpu_v2_ms,"
               "cuda_twopass_ms,cuda_fused_ms,cuda_warp_ms,cuda_fused_warp_shuffle_ms,"
               "speedup_v2_vs_naive,speedup_gpu_warp_vs_cpu_v2,speedup_gpu_fws_vs_cpu_v2\n";
    }
    out << std::fixed << std::setprecision(6)
        << dims.n_time << ',' << dims.n_ant << ',' << dims.n_freq << ','
        << opts.integration_spectra << ',' << threads << ','
        << n_med << ',' << v1_med << ',' << v2_med << ','
        << c2p_med << ',' << cfus_med << ',' << cwrp_med << ',' << cfws_med << ','
        << (n_med / v2_med) << ',' << (v2_med / cwrp_med) << ',' << (v2_med / cfws_med) << '\n';
}

struct FrameLatencyRow {
    std::size_t window_index;
    std::size_t time_start;
    std::string engine;
    std::string kernel;
    double latency_ms;
};

void write_frame_latencies_csv(const std::filesystem::path& path,
                               const std::vector<FrameLatencyRow>& rows) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open frame-latency CSV: " + path.string());
    }
    out << "window_index,time_start,engine,kernel,latency_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.window_index << ',' << row.time_start << ',' << row.engine << ','
            << row.kernel << ',' << row.latency_ms << '\n';
    }
}

struct ValidationRow {
    std::string engine_name;
    DiffStats stats;
};

void write_validation_csv(const std::filesystem::path& path,
                          const std::vector<ValidationRow>& rows) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open validation CSV: " + path.string());
    }
    out << "engine_name,max_abs_diff,rms_diff,rel_error,max_rel_error,passed_tolerance\n";
    out << std::fixed << std::setprecision(9);
    for (const auto& row : rows) {
        out << row.engine_name << ',' << row.stats.max_abs_diff << ','
            << row.stats.rms_diff << ',' << row.stats.rel_error << ','
            << row.stats.max_rel_error << ',' << (row.stats.passed_tolerance ? 1 : 0) << '\n';
    }
}

struct WindowStatsRow {
    std::size_t window_index;
    std::size_t time_start;
    std::string engine;
    std::string kernel;
    double mean_power;
    double max_abs_diff;
    double rms_diff;
};

void write_window_stats_csv(const std::filesystem::path& path,
                            const std::vector<WindowStatsRow>& rows) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open window-validation CSV: " + path.string());
    }
    out << "window_index,time_start,engine,kernel,mean_power,max_abs_diff,rms_diff\n";
    out << std::fixed << std::setprecision(9);
    for (const auto& row : rows) {
        out << row.window_index << ',' << row.time_start << ',' << row.engine << ','
            << row.kernel << ',' << row.mean_power << ',' << row.max_abs_diff << ','
            << row.rms_diff << '\n';
    }
}

// Companion to the four CSVs: array layout, trajectory parameters, and GPU
// info that the plotting dashboard's metadata-legend panel needs and no CSV
// row is a natural place for (they are per-invocation constants, not
// per-window or per-engine data). Mirrors the _metadata.json convention
// tools/benchmark_cpu_cuda.cpp already uses for the direct-beamformer
// benchmark, including reusing beamformer::CudaDeviceInfo for the hardware
// block, so both tools' metadata files share a shape.
void write_metadata_json(const std::filesystem::path& path, const Dimensions& dims,
                         const Options& opts, const beamformer::CudaDeviceInfo& device,
                         const int threads) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open metadata file: " + path.string());
    }
    out << std::fixed << std::setprecision(9);
    out << "{\n"
        << "  \"gpu_name\": \"" << device.name << "\",\n"
        << "  \"compute_capability\": \"" << device.compute_major << '.'
        << device.compute_minor << "\",\n"
        << "  \"gpu_global_memory_bytes\": " << device.global_memory_bytes << ",\n"
        << "  \"cuda_driver_version\": " << device.driver_version << ",\n"
        << "  \"cuda_runtime_version\": " << device.runtime_version << ",\n"
        << "  \"n_time\": " << dims.n_time << ",\n"
        << "  \"n_freq\": " << dims.n_freq << ",\n"
        << "  \"n_ant\": " << dims.n_ant << ",\n"
        << "  \"n_beams\": " << dims.n_beams << ",\n"
        << "  \"integration_spectra\": " << opts.integration_spectra << ",\n"
        << "  \"omp_threads\": " << threads << ",\n"
        << "  \"source_trajectory\": {\"l0\": " << opts.source_l0 << ", \"m0\": " << opts.source_m0
        << ", \"dl_per_sample\": " << opts.source_dl << ", \"dm_per_sample\": " << opts.source_dm
        << "},\n"
        << "  \"prior_trajectory\": {\"l0\": " << opts.prior_l0 << ", \"m0\": " << opts.prior_m0
        << ", \"dl_per_sample\": " << opts.prior_dl << ", \"dm_per_sample\": " << opts.prior_dm
        << "}\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    using namespace beamformer;

    Options opts;
    try {
        opts = parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    const Dimensions dims{opts.n_time, default_frequency_channels, opts.n_ant, tracker_beam_count};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

    TrackerTrajectoryConfig source_traj{direction_from_lm(opts.source_l0, opts.source_m0),
                                        {opts.source_dl, opts.source_dm}};
    const auto packed = beam_tracker_make_moving_point_source(dims, positions, frequencies,
                                                              source_traj, 4.0F);

    TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start = direction_from_lm(opts.prior_l0, opts.prior_m0);
    tracker_cfg.trajectory.direction_rate_per_sample = {opts.prior_dl, opts.prior_dm};
    tracker_cfg.integration_spectra = opts.integration_spectra;

    const std::size_t total_cells = dims.n_time * dims.n_freq * dims.n_beams;
    const auto engines = make_engines();
    std::vector<Intensities> results(engines.size(), Intensities(total_cells));

    std::printf("[guard] Pre-computing and verifying implementations...\n");
    for (std::size_t i = 0; i < engines.size(); ++i) {
        engines[i].run(packed, dims, tracker_cfg, results[i]);
    }

    // Validate every engine after index 0 (naive) against the naive
    // reference BEFORE deciding whether to abort, so a failing run still
    // leaves *_validation.csv on disk for debugging (see file header).
    std::vector<ValidationRow> validation_rows;
    validation_rows.reserve(engines.size() - 1);
    bool any_failed = false;
    for (std::size_t i = 1; i < engines.size(); ++i) {
        const auto stats = compute_diff_stats(results[0], results[i]);
        any_failed = any_failed || !stats.passed_tolerance;
        validation_rows.push_back(ValidationRow{
            engines[i].engine_name + "_" + engines[i].kernel_name, stats});
    }

    std::optional<std::filesystem::path> prefix;
    if (opts.outdir) {
        std::error_code ec;
        std::filesystem::create_directories(*opts.outdir, ec);
        prefix = *opts.outdir / "benchmark_cuda_tracker_v2";
        write_validation_csv(with_suffix(*prefix, "_validation.csv"), validation_rows);
    }

    if (any_failed) {
        std::fprintf(stderr,
                     "FATAL: Implementations do not match the naive reference. Aborting.\n");
        return 1;
    }
    std::printf("[guard] Tolerance equality verified for all kernels.\n\n");

#ifdef _OPENMP
    if (opts.threads > 0) omp_set_num_threads(static_cast<int>(opts.threads));
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif

    std::printf("================ CPU vs CUDA Tracker Benchmark ================\n");
    std::printf("config: n_time=%zu n_ant=%zu spectra=%zu OMP_threads=%d runs=%zu\n",
                dims.n_time, dims.n_ant, opts.integration_spectra, max_threads, opts.repeat);

    std::vector<double> medians(engines.size(), 0.0);
    for (std::size_t i = 0; i < engines.size(); ++i) {
        auto ms = time_runs(opts.warmup_runs, opts.repeat, [&, i] {
            engines[i].run(packed, dims, tracker_cfg, results[i]);
        });
        medians[i] = median_inplace(ms);
    }
    const double n_med = medians[0];
    const double v1_med = medians[1];
    const double v2_med = medians[2];
    const double c2p_med = medians[3];
    const double cfus_med = medians[4];
    const double cwrp_med = medians[5];

    std::printf("  CPU Naive:          %9.3f ms\n", n_med);
    std::printf("  CPU Opt v1:         %9.3f ms  (%.2fx vs Naive)\n", v1_med, n_med / v1_med);
    std::printf("  CPU Opt v2:         %9.3f ms  (%.2fx vs Naive)\n", v2_med, n_med / v2_med);
    std::printf("  CUDA TwoPass:       %9.3f ms  (%.2fx vs CPU v2)\n", c2p_med, v2_med / c2p_med);
    std::printf("  CUDA Fused:         %9.3f ms  (%.2fx vs CPU v2)\n", cfus_med, v2_med / cfus_med);
    std::printf("  CUDA WarpReduction: %9.3f ms  (%.2fx vs CPU v2)\n", cwrp_med, v2_med / cwrp_med);
    std::printf("  CUDA FusedWarpShuffle: %9.3f ms  (%.2fx vs CPU v2)\n", cfws_med, v2_med / cfws_med);
    std::printf("===============================================================\n");

    // Per-window pass: fresh isolated calls for latency (see file header),
    // plus a read-only scan of the already-computed full-sequence buffers
    // for per-window error/power stats (results[] from the guard check
    // above -- no extra tracker calls needed for that half).
    const std::size_t window_count = tracker_window_count(dims.n_time, opts.integration_spectra);
    std::vector<FrameLatencyRow> frame_rows;
    frame_rows.reserve(window_count * engines.size());
    std::vector<WindowStatsRow> window_stats_rows;
    window_stats_rows.reserve(window_count * engines.size());
    Intensities window_intensity(opts.integration_spectra * dims.n_freq * dims.n_beams);

    for (std::size_t w = 0; w < window_count; ++w) {
        const auto slice = make_window_slice(packed, dims, tracker_cfg, w, opts.integration_spectra);
        const std::size_t first_time = w * opts.integration_spectra;
        const std::size_t cell_offset = first_time * dims.n_freq * dims.n_beams;
        const std::size_t cell_count = slice.dims.n_time * dims.n_freq * dims.n_beams;

        for (std::size_t i = 0; i < engines.size(); ++i) {
            const auto& engine = engines[i];

            auto ms = time_runs(1, opts.window_repeats, [&] {
                engine.run(slice.packed, slice.dims, slice.tracker, window_intensity);
            });
            frame_rows.push_back(FrameLatencyRow{
                w, first_time, engine.engine_name, engine.kernel_name, median_inplace(ms)});

            double sum_power = 0.0;
            double max_abs = 0.0;
            double sum_sq_diff = 0.0;
            for (std::size_t c = 0; c < cell_count; ++c) {
                const double value = static_cast<double>(results[i][cell_offset + c]);
                const double ref = static_cast<double>(results[0][cell_offset + c]);
                sum_power += value;
                const double diff = std::abs(value - ref);
                max_abs = std::max(max_abs, diff);
                sum_sq_diff += diff * diff;
            }
            const double mean_power = cell_count > 0 ? sum_power / static_cast<double>(cell_count) : 0.0;
            const double rms_diff =
                cell_count > 0 ? std::sqrt(sum_sq_diff / static_cast<double>(cell_count)) : 0.0;
            window_stats_rows.push_back(WindowStatsRow{
                w, first_time, engine.engine_name, engine.kernel_name, mean_power, max_abs, rms_diff});
        }
    }
    std::printf("[frames] Measured %zu windows x %zu engines.\n", window_count, engines.size());

    if (prefix) {
        const beamformer::CudaDeviceInfo device_info;
        write_summary_csv(with_suffix(*prefix, "_summary.csv"), dims, opts, max_threads, n_med,
                          v1_med, v2_med, c2p_med, cfus_med, cwrp_med);
        write_frame_latencies_csv(with_suffix(*prefix, "_frame_latencies.csv"), frame_rows);
        write_window_stats_csv(with_suffix(*prefix, "_window_validation.csv"), window_stats_rows);
        write_metadata_json(with_suffix(*prefix, "_metadata.json"), dims, opts, device_info,
                            max_threads);
        std::printf("[csv] summary/frame-latency/validation/metadata written under %s\n",
                    opts.outdir->string().c_str());
    }

    return 0;
}
