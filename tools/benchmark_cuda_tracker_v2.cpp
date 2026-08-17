// tools/benchmark_cuda_tracker_v2.cpp
//
// Benchmark: CPU vs CUDA tracker implementations.
// Sweeps through CPU Naive, CPU Opt v1, CPU Opt v2, and the three CUDA v2
// kernels (TwoPass, Fused, WarpReduction).
//
// Re-asserts the tolerance equality contract in-process before timing, 
// ensuring we do not benchmark fast-but-wrong kernels.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_tracker_v2.hpp"
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

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 1;
    std::size_t repeat = 5;
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
        << "  Benchmark CPU vs CUDA beam trackers.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                   32 or 64; default 64\n"
        << "  --integration-spectra N     default 320\n"
        << "  --warmup-runs N             default 1\n"
        << "  --repeat N                  median over repeats; default 5\n"
        << "  --threads N                 OpenMP threads for CPU opt; 0 = default\n"
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
            o.integration_spectra = parse_size(require_value(argc, argv, i), "--integration-spectra");
        } else if (a == "--warmup-runs") {
            o.warmup_runs = parse_size(require_value(argc, argv, i), "--warmup-runs");
        } else if (a == "--repeat") {
            o.repeat = parse_size(require_value(argc, argv, i), "--repeat");
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

bool check_tolerance(const beamformer::Intensities& ref, const beamformer::Intensities& test, 
                     const float rel_tol = 1e-4F, const float abs_tol = 1e-5F) {
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float r = ref[i];
        const float t = test[i];
        const float diff = std::abs(r - t);
        if (diff > abs_tol && diff > rel_tol * r) return false;
    }
    return true;
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

    TrackerTrajectoryConfig source_traj{direction_from_lm(opts.source_l0, opts.source_m0), {opts.source_dl, opts.source_dm}};
    const auto packed = beam_tracker_make_moving_point_source(dims, positions, frequencies, source_traj, 4.0F);

    TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start = direction_from_lm(opts.prior_l0, opts.prior_m0);
    tracker_cfg.trajectory.direction_rate_per_sample = {opts.prior_dl, opts.prior_dm};
    tracker_cfg.integration_spectra = opts.integration_spectra;

    const std::size_t total_cells = dims.n_time * dims.n_freq * dims.n_beams;

    Intensities naive(total_cells), cpu_v1(total_cells), cpu_v2(total_cells);
    Intensities cuda_twopass(total_cells), cuda_fused(total_cells), cuda_warp(total_cells);

    std::printf("[guard] Pre-computing and verifying implementations...\n");
    beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, naive);
    beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_v1);
    beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_v2);
    cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_twopass, CudaTrackerKernelV2::TwoPass);
    cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_fused, CudaTrackerKernelV2::Fused);
    cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_warp, CudaTrackerKernelV2::WarpReduction);

    if (!check_tolerance(naive, cpu_v1) || !check_tolerance(naive, cpu_v2) || 
        !check_tolerance(naive, cuda_twopass) || !check_tolerance(naive, cuda_fused) || !check_tolerance(naive, cuda_warp)) {
        std::fprintf(stderr, "FATAL: Implementations do not match the naive reference. Aborting.\n");
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

    auto naive_ms = time_runs(opts.warmup_runs, opts.repeat, [&] { beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, naive); });
    auto v1_ms    = time_runs(opts.warmup_runs, opts.repeat, [&] { beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_v1); });
    auto v2_ms    = time_runs(opts.warmup_runs, opts.repeat, [&] { beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, cpu_v2); });
    auto c2p_ms   = time_runs(opts.warmup_runs, opts.repeat, [&] { cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_twopass, CudaTrackerKernelV2::TwoPass); });
    auto cfus_ms  = time_runs(opts.warmup_runs, opts.repeat, [&] { cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_fused, CudaTrackerKernelV2::Fused); });
    auto cwrp_ms  = time_runs(opts.warmup_runs, opts.repeat, [&] { cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, cuda_warp, CudaTrackerKernelV2::WarpReduction); });

    double n_med = median_inplace(naive_ms), v1_med = median_inplace(v1_ms), v2_med = median_inplace(v2_ms);
    double c2p_med = median_inplace(c2p_ms), cfus_med = median_inplace(cfus_ms), cwrp_med = median_inplace(cwrp_ms);

    std::printf("  CPU Naive:          %9.3f ms\n", n_med);
    std::printf("  CPU Opt v1:         %9.3f ms  (%.2fx vs Naive)\n", v1_med, n_med/v1_med);
    std::printf("  CPU Opt v2:         %9.3f ms  (%.2fx vs Naive)\n", v2_med, n_med/v2_med);
    std::printf("  CUDA TwoPass:       %9.3f ms  (%.2fx vs CPU v2)\n", c2p_med, v2_med/c2p_med);
    std::printf("  CUDA Fused:         %9.3f ms  (%.2fx vs CPU v2)\n", cfus_med, v2_med/cfus_med);
    std::printf("  CUDA WarpReduction: %9.3f ms  (%.2fx vs CPU v2)\n", cwrp_med, v2_med/cwrp_med);
    std::printf("===============================================================\n");

    if (opts.outdir) {
        std::error_code ec;
        std::filesystem::create_directories(*opts.outdir, ec);
        const auto path = *opts.outdir / "benchmark_cuda_tracker_v2.csv";
        const bool exists = std::filesystem::exists(path);
        std::ofstream out(path, std::ios::app);
        if (!exists) {
            out << "n_time,n_ant,integration_spectra,omp_threads,repeat,"
                   "naive_ms,cpu_v1_ms,cpu_v2_ms,cuda_twopass_ms,cuda_fused_ms,cuda_warp_ms\n";
        }
        out << dims.n_time << "," << dims.n_ant << "," << opts.integration_spectra << "," 
            << max_threads << "," << opts.repeat << "," 
            << n_med << "," << v1_med << "," << v2_med << "," 
            << c2p_med << "," << cfus_med << "," << cwrp_med << "\n";
        std::printf("[csv] summary written to %s\n", path.string().c_str());
    }

    return 0;
}