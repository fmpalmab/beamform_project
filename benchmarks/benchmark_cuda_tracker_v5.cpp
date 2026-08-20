// benchmarks/benchmark_cuda_tracker_v5.cpp
//
// Comprehensive Multi-Generation Benchmark Suite:
// CPU vs CUDA Legacy vs Phase 4 (FWS) vs V3 vs V4 vs V5 (Unified Warp Reduction).
//
// Evaluates:
// 1. CPU Naive, CPU Opt v1, CPU Opt v2
// 2. CUDA TwoPass, CUDA Fused, CUDA WarpReduction
// 3. CUDA Phase 4 Fused Warp Shuffle (FWS)
// 4. CUDA V3 Direct, CUDA V3 Batched Stream, CUDA V3 Batched Kernel Only, CUDA V3 Device Resident
// 5. CUDA V4 Deep ILP / Block Reduction, CUDA V4 Batched Kernel Only, CUDA V4 Device Resident
// 6. CUDA V5 Unified (Unroll 2, Unroll 4, Unroll 8)
// 7. CUDA V5 Batched Stream (Persistent buffers)
// 8. CUDA V5 Batched Graph (CUDA Graph accelerated)
// 9. CUDA V5 Batched Kernel Only (Pure GPU compute)
// 10. CUDA V5 Device Resident (Zero-copy in-place)

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_tracker_v2.hpp"
#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"
#include "beamformer/cuda_beam_tracker_v3.hpp"
#include "beamformer/cuda_beam_tracker_v4.hpp"
#include "beamformer/cuda_beam_tracker_v5.hpp"
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
using beamformer::V4ExecutionConfig;
using beamformer::V4KernelMode;
using beamformer::V5ExecutionConfig;
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
        << "  Benchmark CPU vs CUDA (Legacy, Phase 4 FWS, V3, V4, V5) beam trackers.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                   32, 64, 128, or 256; default 64\n"
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

struct EngineResult {
    std::string backend;
    std::string engine_name;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double speedup_vs_phase4 = 1.0;
    double speedup_vs_cpu_naive = 1.0;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const auto opts = parse_options(argc, argv);

#ifdef _OPENMP
        if (opts.threads > 0) {
            omp_set_num_threads(static_cast<int>(opts.threads));
        }
#endif

        const Dimensions dims{opts.n_time, beamformer::default_frequency_channels, opts.n_ant, beamformer::tracker_beam_count};
        const auto positions = beamformer::default_positions(dims.n_ant);
        const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);

        beamformer::TrackerTrajectoryConfig source_traj{
            beamformer::direction_from_lm(opts.source_l0, opts.source_m0),
            {opts.source_dl, opts.source_dm}
        };

        const auto packed = beamformer::beam_tracker_make_moving_point_source(
            dims, positions, frequencies, source_traj, 4.0F);

        TrackerConfig tracker_cfg;
        tracker_cfg.trajectory.direction_start = beamformer::direction_from_lm(opts.source_l0, opts.source_m0);
        tracker_cfg.trajectory.direction_rate_per_sample = {opts.source_dl, opts.source_dm};
        tracker_cfg.integration_spectra = opts.integration_spectra;

        const std::size_t total_cells = dims.n_time * dims.n_freq * dims.n_beams;
        Intensities output(total_cells, 0.0F);

        std::cout << "================================================================================\n";
        std::cout << " CUDA Beam Tracker V5 Multi-Generation Benchmark Suite\n";
        std::cout << " Array: n_ant=" << dims.n_ant << " | n_freq=" << dims.n_freq
                  << " | n_time=" << dims.n_time << " | spectra=" << opts.integration_spectra << "\n";
        std::cout << "================================================================================\n\n";

        std::vector<EngineResult> results;

        auto benchmark_engine = [&](const std::string& backend, const std::string& name, auto&& fn) {
            auto samples = time_runs(opts.warmup_runs, opts.repeat, [&]() {
                fn();
                cudaDeviceSynchronize();
            });
            const double med = median_inplace(samples);
            const double min_t = *std::min_element(samples.begin(), samples.end());
            const double max_t = *std::max_element(samples.begin(), samples.end());
            results.push_back({backend, name, med, min_t, max_t, 1.0, 1.0});
        };

        // 1. CPU Reference
        benchmark_engine("cpu", "naive", [&]() {
            beamformer::beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, output);
        });

        benchmark_engine("cpu", "opt_v1", [&]() {
            beamformer::beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, output);
        });

        benchmark_engine("cpu", "opt_v2", [&]() {
            beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, output);
        });

        if (dims.n_ant <= 64) {
            // 2. CUDA Legacy
            benchmark_engine("gpu", "twopass", [&]() {
                beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, output, CudaTrackerKernelV2::TwoPass);
            });

            benchmark_engine("gpu", "fused", [&]() {
                beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, output, CudaTrackerKernelV2::Fused);
            });

            benchmark_engine("gpu", "warp_reduction", [&]() {
                beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, output, CudaTrackerKernelV2::WarpReduction);
            });

            // 3. CUDA Phase 4 FWS (Baseline)
            benchmark_engine("gpu", "fused_warp_shuffle_p4", [&]() {
                beamformer::cuda_beam_tracker_fused_warp_shuffle_into(packed, dims, tracker_cfg, output);
            });

            // 4. CUDA V3
            benchmark_engine("gpu", "v3_direct", [&]() {
                V3ExecutionConfig cfg;
                cfg.time_unroll = 4;
                beamformer::cuda_beam_tracker_v3_into(packed, dims, tracker_cfg, output, cfg);
            });

            {
                const std::size_t v_bytes = beamformer::voltage_sample_count(dims) * sizeof(std::uint8_t);
                const std::size_t out_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
                std::uint8_t* d_packed = nullptr;
                float* d_intensity = nullptr;
                cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
                cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);
                cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
                V3ExecutionConfig cfg;
                cfg.time_unroll = 4;
                benchmark_engine("gpu", "v3_device_resident", [&]() {
                    cuda_beam_tracker_v3_device_resident(d_packed, d_intensity, dims, tracker_cfg, cfg);
                });
                cudaFree(d_packed);
                cudaFree(d_intensity);
            }
        }

        // 5. CUDA V4
        if (dims.n_ant <= 64) {
            benchmark_engine("gpu", "v4_deep_ilp_u8", [&]() {
                V4ExecutionConfig cfg;
                cfg.mode = V4KernelMode::DeepIlpWarpShuffle;
                cfg.time_unroll = 8;
                beamformer::cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, output, cfg);
            });
        } else {
            benchmark_engine("gpu", "v4_block_reduction_u8", [&]() {
                V4ExecutionConfig cfg;
                cfg.mode = V4KernelMode::BlockReduction;
                cfg.time_unroll = 8;
                beamformer::cuda_beam_tracker_v4_into(packed, dims, tracker_cfg, output, cfg);
            });
        }

        {
            const std::size_t v_bytes = beamformer::voltage_sample_count(dims) * sizeof(std::uint8_t);
            const std::size_t out_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
            std::uint8_t* d_packed = nullptr;
            float* d_intensity = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);
            cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
            V4ExecutionConfig cfg;
            cfg.mode = V4KernelMode::Auto;
            cfg.time_unroll = 4;
            benchmark_engine("gpu", "v4_device_resident", [&]() {
                cuda_beam_tracker_v4_device_resident(d_packed, d_intensity, dims, tracker_cfg, cfg);
            });
            cudaFree(d_packed);
            cudaFree(d_intensity);
        }

        // 6. CUDA V5 Unified Engine
        benchmark_engine("gpu", "v5_unified_u2", [&]() {
            V5ExecutionConfig cfg;
            cfg.time_unroll = 2;
            beamformer::cuda_beam_tracker_v5_into(packed, dims, tracker_cfg, output, cfg);
        });

        benchmark_engine("gpu", "v5_unified_u4", [&]() {
            V5ExecutionConfig cfg;
            cfg.time_unroll = 4;
            beamformer::cuda_beam_tracker_v5_into(packed, dims, tracker_cfg, output, cfg);
        });

        benchmark_engine("gpu", "v5_unified_u8", [&]() {
            V5ExecutionConfig cfg;
            cfg.time_unroll = 8;
            beamformer::cuda_beam_tracker_v5_into(packed, dims, tracker_cfg, output, cfg);
        });

        {
            const std::size_t window_count = beamformer::tracker_window_count(dims.n_time, tracker_cfg.integration_spectra);
            Dimensions win_dims{tracker_cfg.integration_spectra, dims.n_freq, dims.n_ant, dims.n_beams};
            V5ExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = false;
            auto stream = std::make_unique<beamformer::BatchedTrackerStreamV5>(win_dims, tracker_cfg, window_count, cfg);
            benchmark_engine("gpu", "v5_batched_stream", [&]() {
                stream->process_batch(0, packed.data(), output.data());
            });
        }

        {
            const std::size_t window_count = beamformer::tracker_window_count(dims.n_time, tracker_cfg.integration_spectra);
            Dimensions win_dims{tracker_cfg.integration_spectra, dims.n_freq, dims.n_ant, dims.n_beams};
            V5ExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = true;
            auto stream = std::make_unique<beamformer::BatchedTrackerStreamV5>(win_dims, tracker_cfg, window_count, cfg);
            stream->process_batch(0, packed.data(), output.data());
            benchmark_engine("gpu", "v5_batched_graph", [&]() {
                stream->process_batch(0, packed.data(), output.data());
            });
        }

        {
            const std::size_t window_count = beamformer::tracker_window_count(dims.n_time, tracker_cfg.integration_spectra);
            Dimensions win_dims{tracker_cfg.integration_spectra, dims.n_freq, dims.n_ant, dims.n_beams};
            V5ExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = true;
            auto stream = std::make_unique<beamformer::BatchedTrackerStreamV5>(win_dims, tracker_cfg, window_count, cfg);
            stream->process_batch(0, packed.data(), output.data());
            benchmark_engine("gpu", "v5_batched_kernel_only", [&]() {
                stream->process_batch_kernel_only(0);
            });
        }

        {
            const std::size_t v_bytes = beamformer::voltage_sample_count(dims) * sizeof(std::uint8_t);
            const std::size_t out_bytes = dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
            std::uint8_t* d_packed = nullptr;
            float* d_intensity = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);
            cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
            V5ExecutionConfig cfg;
            cfg.time_unroll = 8;
            benchmark_engine("gpu", "v5_device_resident", [&]() {
                cuda_beam_tracker_v5_device_resident(d_packed, d_intensity, dims, tracker_cfg, cfg);
            });
            cudaFree(d_packed);
            cudaFree(d_intensity);
        }

        // Compute speedups
        double cpu_naive_ms = results[0].median_ms;
        double p4_fws_ms = 1.0;
        bool has_p4 = false;
        for (const auto& r : results) {
            if (r.engine_name == "fused_warp_shuffle_p4") {
                p4_fws_ms = r.median_ms;
                has_p4 = true;
                break;
            }
        }
        if (!has_p4) {
            p4_fws_ms = results[0].median_ms;
        }

        for (auto& r : results) {
            r.speedup_vs_phase4 = (has_p4 ? (p4_fws_ms / r.median_ms) : 1.0);
            r.speedup_vs_cpu_naive = cpu_naive_ms / r.median_ms;
        }

        std::cout << std::left << std::setw(8) << "Backend"
                  << std::setw(28) << "Engine"
                  << std::setw(16) << "Median (ms)"
                  << std::setw(16) << "vs Phase 4"
                  << std::setw(16) << "vs CPU Naive" << "\n";
        std::cout << std::string(84, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(8) << r.backend
                      << std::setw(28) << r.engine_name
                      << std::setw(16) << std::fixed << std::setprecision(3) << r.median_ms
                      << std::setw(16) << std::setprecision(2) << std::string(std::to_string(r.speedup_vs_phase4).substr(0, 4) + "x")
                      << std::setw(16) << std::setprecision(2) << std::string(std::to_string(r.speedup_vs_cpu_naive).substr(0, 4) + "x")
                      << "\n";
        }
        std::cout << "================================================================================\n\n";

        if (opts.outdir) {
            std::filesystem::create_directories(*opts.outdir);
            std::ofstream csv(*opts.outdir / "benchmark_cuda_tracker_v5_summary.csv");
            csv << "backend,engine,median_ms,min_ms,max_ms,speedup_vs_p4,speedup_vs_cpu_naive,n_ant,n_time,spectra\n";
            for (const auto& r : results) {
                csv << r.backend << "," << r.engine_name << ","
                    << r.median_ms << "," << r.min_ms << "," << r.max_ms << ","
                    << r.speedup_vs_phase4 << "," << r.speedup_vs_cpu_naive << ","
                    << dims.n_ant << "," << dims.n_time << "," << opts.integration_spectra << "\n";
            }
            std::cout << "Wrote summary CSV to: " << (*opts.outdir / "benchmark_cuda_tracker_v5_summary.csv") << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
