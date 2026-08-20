// tools/benchmark_cuda_tracker_v3.cpp
//
// Comprehensive Benchmark Suite: CPU vs CUDA Legacy vs Phase 4 (FWS) vs V3.
//
// Evaluates:
// 1. CPU Naive, CPU Opt v1, CPU Opt v2
// 2. CUDA TwoPass, CUDA Fused, CUDA WarpReduction
// 3. CUDA Phase 4 Fused Warp Shuffle (baseline)
// 4. CUDA V3 Direct (ILP + PTX bfe)
// 5. CUDA V3 Stream (Pipelined multi-stream)
// 6. CUDA V3 Batched Stream (Persistent buffers)
// 7. CUDA V3 Batched Graph (CUDA Graph accelerated)
// 8. CUDA V3 Batched Kernel Only (Compute bound)
// 9. CUDA V3 Device Resident (Zero-copy in-place)

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_tracker_v2.hpp"
#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"
#include "beamformer/cuda_beam_tracker_v3.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

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
#include <memory>
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
using beamformer::V3ExecutionConfig;
using beamformer::Vec3;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 2;
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
        << "  Benchmark CPU vs CUDA (Legacy, Phase 4 FWS, V3) beam trackers.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                   32 or 64; default 64\n"
        << "  --integration-spectra N     default 320\n"
        << "  --warmup-runs N             default 2\n"
        << "  --repeat N                  median over repeats for full sequence; default 5\n"
        << "  --window-repeats N          median over repeats for per-window; default 3\n"
        << "  --threads N                 OpenMP threads for CPU opt; 0 = default\n"
        << "  --outdir DIR                write CSVs and metadata there\n";
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

struct DiffStats {
    double max_abs_diff = 0.0;
    double rms_diff = 0.0;
    double rel_error = 0.0;
    double max_rel_error = 0.0;
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

using EngineRunner = std::function<void(const PackedVoltage&, const Dimensions&,
                                        const TrackerConfig&, Intensities&)>;

struct Engine {
    std::string engine_name;
    std::string kernel_name;
    EngineRunner run;
};

std::vector<Engine> make_all_engines() {
    using beamformer::beam_tracker_cpu_packed_intensity_into;
    using beamformer::beam_tracker_opt_cpu_packed_intensity_into;
    using beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into;
    using beamformer::cuda_tracker_v2_packed_intensity_into;
    using beamformer::cuda_beam_tracker_fused_warp_shuffle_stream;
    using beamformer::cuda_beam_tracker_v3_into;
    using beamformer::cuda_beam_tracker_v3_stream;
    using beamformer::cuda_beam_tracker_v3_device_resident;
    using beamformer::BatchedTrackerStreamV3;

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
        {"gpu", "fused_warp_shuffle_p4",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             cuda_beam_tracker_fused_warp_shuffle_stream(p, d, t, out, 3);
         }},
        {"gpu", "v3_direct",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             V3ExecutionConfig cfg;
             cfg.time_chunk_size = 80;
             cfg.time_unroll = 2;
             cuda_beam_tracker_v3_into(p, d, t, out, cfg);
         }},
        {"gpu", "v3_stream",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             V3ExecutionConfig cfg;
             cfg.time_chunk_size = 80;
             cfg.time_unroll = 2;
             cuda_beam_tracker_v3_stream(p, d, t, out, 3, cfg);
         }},
        {"gpu", "v3_batched_stream",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             static thread_local std::unique_ptr<BatchedTrackerStreamV3> cached_stream;
             static thread_local std::size_t cached_n_time = 0;
             static thread_local std::size_t cached_n_ant = 0;
             const std::size_t batch_size = beamformer::tracker_window_count(d.n_time, t.integration_spectra);
             if (!cached_stream || cached_n_time != d.n_time || cached_n_ant != d.n_ant) {
                 Dimensions win_dims{t.integration_spectra, d.n_freq, d.n_ant, d.n_beams};
                 V3ExecutionConfig cfg;
                 cfg.time_chunk_size = 80;
                 cfg.time_unroll = 2;
                 cfg.enable_cuda_graph = false;
                 cached_stream = std::make_unique<BatchedTrackerStreamV3>(win_dims, t, batch_size, cfg);
                 cached_n_time = d.n_time;
                 cached_n_ant = d.n_ant;
             }
             cached_stream->process_batch(0, p.data(), out.data());
         }},
        {"gpu", "v3_batched_graph",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             static thread_local std::unique_ptr<BatchedTrackerStreamV3> cached_stream;
             static thread_local std::size_t cached_n_time = 0;
             static thread_local std::size_t cached_n_ant = 0;
             const std::size_t batch_size = beamformer::tracker_window_count(d.n_time, t.integration_spectra);
             if (!cached_stream || cached_n_time != d.n_time || cached_n_ant != d.n_ant) {
                 Dimensions win_dims{t.integration_spectra, d.n_freq, d.n_ant, d.n_beams};
                 V3ExecutionConfig cfg;
                 cfg.time_chunk_size = 80;
                 cfg.time_unroll = 2;
                 cfg.enable_cuda_graph = true;
                 cached_stream = std::make_unique<BatchedTrackerStreamV3>(win_dims, t, batch_size, cfg);
                 cached_n_time = d.n_time;
                 cached_n_ant = d.n_ant;
             }
             cached_stream->process_batch(0, p.data(), out.data());
         }},
        {"gpu", "v3_batched_kernel_only",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             static thread_local std::unique_ptr<BatchedTrackerStreamV3> cached_stream;
             static thread_local std::size_t cached_n_time = 0;
             static thread_local std::size_t cached_n_ant = 0;
             const std::size_t batch_size = beamformer::tracker_window_count(d.n_time, t.integration_spectra);
             if (!cached_stream || cached_n_time != d.n_time || cached_n_ant != d.n_ant) {
                 Dimensions win_dims{t.integration_spectra, d.n_freq, d.n_ant, d.n_beams};
                 V3ExecutionConfig cfg;
                 cfg.time_chunk_size = 80;
                 cfg.time_unroll = 2;
                 cfg.enable_cuda_graph = true;
                 cached_stream = std::make_unique<BatchedTrackerStreamV3>(win_dims, t, batch_size, cfg);
                 cached_n_time = d.n_time;
                 cached_n_ant = d.n_ant;
                 cached_stream->process_batch(0, p.data(), out.data());
             }
             cached_stream->process_batch_kernel_only(0);
         }},
        {"gpu", "v3_device_resident",
         [](const PackedVoltage& p, const Dimensions& d, const TrackerConfig& t, Intensities& out) {
             static thread_local std::uint8_t* d_packed = nullptr;
             static thread_local float* d_intensity = nullptr;
             static thread_local std::size_t cached_size = 0;
             const std::size_t v_bytes = beamformer::voltage_sample_count(d) * sizeof(std::uint8_t);
             const std::size_t out_bytes = d.n_time * d.n_freq * d.n_beams * sizeof(float);
             if (cached_size != v_bytes) {
                 if (d_packed) cudaFree(d_packed);
                 if (d_intensity) cudaFree(d_intensity);
                 cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
                 cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);
                 cudaMemcpy(d_packed, p.data(), v_bytes, cudaMemcpyHostToDevice);
                 cached_size = v_bytes;
             }
             V3ExecutionConfig cfg;
             cuda_beam_tracker_v3_device_resident(d_packed, d_intensity, d, t, cfg);
             cudaMemcpy(out.data(), d_intensity, out_bytes, cudaMemcpyDeviceToHost);
         }},
    };
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
    const auto engines = make_all_engines();
    std::vector<Intensities> results(engines.size(), Intensities(total_cells));

    std::printf("[guard] Pre-computing and verifying implementations...\n");
    for (std::size_t i = 0; i < engines.size(); ++i) {
        engines[i].run(packed, dims, tracker_cfg, results[i]);
    }

    bool any_failed = false;
    for (std::size_t i = 1; i < engines.size(); ++i) {
        const auto stats = compute_diff_stats(results[0], results[i]);
        if (!stats.passed_tolerance) {
            std::fprintf(stderr, "FAIL: Engine [%s:%s] failed tolerance check (max_diff=%f, rms=%f)\n",
                         engines[i].engine_name.c_str(), engines[i].kernel_name.c_str(),
                         stats.max_abs_diff, stats.rms_diff);
            any_failed = true;
        }
    }

    if (any_failed) {
        std::fprintf(stderr, "FATAL: Implementations do not match CPU reference. Aborting.\n");
        return 1;
    }
    std::printf("[guard] Tolerance equality verified for all %zu engines.\n\n", engines.size());

#ifdef _OPENMP
    if (opts.threads > 0) omp_set_num_threads(static_cast<int>(opts.threads));
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif

    std::printf("================ CPU vs CUDA Legacy vs Phase 4 vs V3 Benchmark ================\n");
    std::printf("Configuration: n_time=%zu, n_ant=%zu, spectra=%zu, OMP_threads=%d, repeats=%zu\n",
                dims.n_time, dims.n_ant, opts.integration_spectra, max_threads, opts.repeat);
    std::printf("-------------------------------------------------------------------------------\n");

    std::vector<double> medians(engines.size(), 0.0);
    for (std::size_t i = 0; i < engines.size(); ++i) {
        auto ms = time_runs(opts.warmup_runs, opts.repeat, [&, i] {
            engines[i].run(packed, dims, tracker_cfg, results[i]);
        });
        medians[i] = median_inplace(ms);
        const double speedup_vs_cpu_v2 = (medians[2] > 0.0) ? medians[2] / medians[i] : 1.0;
        const double speedup_vs_p4 = (medians[6] > 0.0) ? medians[6] / medians[i] : 1.0;

        std::printf("  %-12s %-24s: %9.3f ms  (%6.2fx vs CPU v2, %6.2fx vs Phase 4)\n",
                    engines[i].engine_name.c_str(), engines[i].kernel_name.c_str(),
                    medians[i], speedup_vs_cpu_v2, speedup_vs_p4);
    }
    std::printf("===============================================================================\n");

    return 0;
}
