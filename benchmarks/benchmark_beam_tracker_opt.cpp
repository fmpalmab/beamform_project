// Benchmark: optimized naive tracker (src/beam_tracker_opt.cpp) vs. the
// reference naive tracker (src/beam_tracker.cpp).
//
// Runs both trackers on identical synthetic packed-int4 input (a slowly moving
// point source) and reports the ACTUAL end-to-end wall-clock latency for each
// path plus a speedup ratio. The naive==opt byte-equality contract is
// re-asserted in-process before timing, so a fast-but-wrong opt path is
// caught here too.
//
// Usage:
//   benchmark_beam_tracker_opt [--n-time N] [--n-ant 32|64]
//                             [--integration-spectra N] [--warmup-runs N]
//                             [--repeat N] [--outdir DIR]
//
// Defaults match the production shard shape (n_time=15360, n_ant=64,
// integration_spectra=320). Each path is warmed up `warmup_runs` times and then
// timed `repeat` times; the reported number is the MEDIAN across repeats to
// reduce timing noise. The worst (max) run time is also printed.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 1;
    std::size_t repeat = 5;
    // Explicit thread count for the opt path's OpenMP region. 0 = honour the
    // process-wide OMP_NUM_THREADS (or the implementation default). A positive
    // value calls omp_set_num_threads() before the timed runs so a single SLURM
    // job can sweep thread counts without re-launching.
    std::size_t threads = 0;
    // Source trajectory: slow drift within the FoV so the per-window direction
    // actually changes (exercising the per-window weight recompute hot path).
    float source_l0 = 0.0F;
    float source_m0 = 0.0F;
    float source_dl = 1.0e-5F;
    float source_dm = 0.0F;
    // Prior/trajectory for the trackers (drives the steering direction).
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

float parse_float(const char* v, const char* opt) {
    std::size_t used = 0;
    const auto parsed = std::stof(std::string(v), &used);
    if (used != std::string(v).size()) {
        throw std::invalid_argument(std::string("invalid float for ") + opt);
    }
    return parsed;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  Benchmark src/beam_tracker_opt.cpp vs src/beam_tracker.cpp.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                   32 or 64; default 64\n"
        << "  --integration-spectra N     default 320\n"
        << "  --warmup-runs N             default 1\n"
        << "  --repeat N                  median over repeats; default 5\n"
        << "  --threads N                 opt-path OpenMP threads; 0 = honour\n"
        << "                              OMP_NUM_THREADS / default. default 0\n"
        << "  --source-l0/m0/dl/dm F      default 0/0/1e-5/0\n"
        << "  --prior-l0/m0/dl/dm F       default 0/0/1e-5/0\n"
        << "  --outdir DIR                write summary CSV there\n";
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
            o.integration_spectra =
                parse_size(require_value(argc, argv, i), "--integration-spectra");
        } else if (a == "--warmup-runs") {
            o.warmup_runs =
                parse_size(require_value(argc, argv, i), "--warmup-runs");
       } else if (a == "--repeat") {
            o.repeat = parse_size(require_value(argc, argv, i), "--repeat");
        } else if (a == "--threads") {
            o.threads = parse_size(require_value(argc, argv, i), "--threads");
        } else if (a == "--source-l0") {
            o.source_l0 = parse_float(require_value(argc, argv, i), "--source-l0");
        } else if (a == "--source-m0") {
            o.source_m0 = parse_float(require_value(argc, argv, i), "--source-m0");
        } else if (a == "--source-dl") {
            o.source_dl = parse_float(require_value(argc, argv, i), "--source-dl");
        } else if (a == "--source-dm") {
            o.source_dm = parse_float(require_value(argc, argv, i), "--source-dm");
        } else if (a == "--prior-l0") {
            o.prior_l0 = parse_float(require_value(argc, argv, i), "--prior-l0");
        } else if (a == "--prior-m0") {
            o.prior_m0 = parse_float(require_value(argc, argv, i), "--prior-m0");
        } else if (a == "--prior-dl") {
            o.prior_dl = parse_float(require_value(argc, argv, i), "--prior-dl");
        } else if (a == "--prior-dm") {
            o.prior_dm = parse_float(require_value(argc, argv, i), "--prior-dm");
        } else if (a == "--outdir") {
            o.outdir = std::filesystem::path(require_value(argc, argv, i));
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + a);
        }
    }
    return o;
}

// Compute and print summary stats + return the median of a sorted sample set.
double median_inplace(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? samples[n / 2]
                        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
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

} // namespace

int main(int argc, char** argv) {
    using namespace beamformer;

    Options opts;
    try {
        opts = parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage(argv[0]);
        return 2;
    }

    const Dimensions dims{opts.n_time, default_frequency_channels, opts.n_ant,
                          tracker_beam_count};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);

    // Source trajectory for the synthetic moving-point-source data.
    TrackerTrajectoryConfig source_traj{
        direction_from_lm(opts.source_l0, opts.source_m0),
        {opts.source_dl, opts.source_dm}};
    const auto packed = beam_tracker_make_moving_point_source(
        dims, positions, frequencies, source_traj, 4.0F);

    // Trajectory the trackers steer along (the INPUT direction model — NOT an
    // estimated output). Using a slightly different prior than the true source
    // trajectory makes the run load-bearing (per-window weight sets differ)
    // without changing what this benchmark measures (raw tracker speed, not
    // DOA accuracy — there is no DOA estimation here).
    TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start =
        direction_from_lm(opts.prior_l0, opts.prior_m0);
    tracker_cfg.trajectory.direction_rate_per_sample = {opts.prior_dl,
                                                         opts.prior_dm};
    tracker_cfg.integration_spectra = opts.integration_spectra;

    // --- Byte-equality guard ------------------------------------------------
    // The optimization is allowed to be faster only because it does the SAME
    // math as the naive path. Verify that up front so a timing win is never
    // reported for a numerically-divergent kernel.
    Intensities naive(dims.n_time * dims.n_freq * dims.n_beams);
    Intensities opt(dims.n_time * dims.n_freq * dims.n_beams);
    beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, naive);
    beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, opt);
    if (naive != opt) {
        std::fprintf(stderr,
                     "FATAL: naive != opt (byte-equality contract broken). "
                     "Aborting benchmark; perf of a wrong kernel is meaningless.\n");
        return 1;
    }
    std::printf("[guard] naive == opt byte-equal: PASS (%zu cells)\n", naive.size());

    // --- Thread count for the opt path ------------------------------------
    // The naive tracker is serial; only the opt path uses OpenMP. An explicit
    // positive --threads value overrides the process default here so a single
    // SLURM job can sweep thread counts without re-exporting OMP_NUM_THREADS
    // per launch.
#ifdef _OPENMP
    if (opts.threads > 0) {
        omp_set_num_threads(static_cast<int>(opts.threads));
    }
#endif

    // --- Timing --------------------------------------------------------------
    // Time the naive path.
    auto naive_ms = time_runs(opts.warmup_runs, opts.repeat, [&] {
        beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, naive);
    });

    // Time the optimized path. The into-variant is reused so we are not
    // measuring allocation churn.
    auto opt_ms = time_runs(opts.warmup_runs, opts.repeat, [&] {
        beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, opt);
    });

    const double naive_med = median_inplace(naive_ms);
    const double opt_med = median_inplace(opt_ms);
    const double naive_max = naive_ms.back();
    const double opt_max = opt_ms.back();
    const double speedup = (opt_med > 0.0) ? naive_med / opt_med : 0.0;

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif

    std::printf("\n================================ beam_tracker_naive_vs_opt "
                "================================\n");
    std::printf("config: n_time=%zu n_freq=%zu n_ant=%zu n_beams=%zu "
                "integration_spectra=%zu OMP_threads=%d repeat=%zu warmup=%zu\n",
                dims.n_time, dims.n_freq, dims.n_ant, dims.n_beams,
                opts.integration_spectra, max_threads, opts.repeat, opts.warmup_runs);
    std::printf("naive:  median=%9.3f ms   max=%9.3f ms\n", naive_med, naive_max);
    std::printf("  opt:  median=%9.3f ms   max=%9.3f ms\n", opt_med, opt_max);
    std::printf("speedup (median):  %.2fx   (faster means smaller-is-better for "
                "opt)\n", speedup);
    std::printf("byte-equality:     naive == opt  (PASS)\n");
    std::printf("===================================================================\n");

    if (opts.outdir) {
        std::error_code ec;
        std::filesystem::create_directories(*opts.outdir, ec);
        const auto path = *opts.outdir / "benchmark_beam_tracker_opt.csv";
        const bool exists = std::filesystem::exists(path);
        std::ofstream out(path, std::ios::app);
        if (!exists) {
            out << "n_time,n_freq,n_ant,integration_spectra,omp_threads,repeat,"
                   "naive_ms_median,opt_ms_median,speedup,naive_ms_max,opt_ms_max,"
                   "byte_equal\n";
        }
        out << dims.n_time << "," << dims.n_freq << "," << dims.n_ant << ","
            << opts.integration_spectra << "," << max_threads << "," << opts.repeat
            << "," << std::setprecision(6) << naive_med << "," << opt_med << ","
            << speedup << "," << naive_max << "," << opt_max << ",PASS\n";
        std::printf("[csv] summary written to %s\n", path.string().c_str());
    }

    return 0;
}
