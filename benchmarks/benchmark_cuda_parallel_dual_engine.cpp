// benchmarks/benchmark_cuda_parallel_dual_engine.cpp
//
// Performance benchmark demonstrating speedup, latency reduction, and memory
// savings when running Fixed-Grid Beamformer (V3) and Dynamic Beam Tracker (V5)
// simultaneously in parallel vs sequential execution.

#include "beamformer/cuda_dual_parallel_pipeline.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/cuda_beam_tracker_v5.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n_time = 3200;
    std::size_t n_freq = 336;
    std::size_t n_ant = 64;
    std::size_t n_bf_beams = 64;
    std::size_t n_tr_beams = 1;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 2;
    std::size_t repeat = 5;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --n-time <N>               (default: 3200)\n"
              << "  --n-freq <N>               (default: 336)\n"
              << "  --n-ant <N>                (default: 64)\n"
              << "  --n-bf-beams <N>           (default: 64)\n"
              << "  --n-tr-beams <N>           (default: 1)\n"
              << "  --integration-spectra <N>  (default: 320)\n"
              << "  --warmup-runs <N>          (default: 2)\n"
              << "  --repeat <N>               (default: 5)\n"
              << "  --help                     (print this message)\n";
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--n-time" && i + 1 < argc) {
            o.n_time = std::stoull(argv[++i]);
        } else if (a == "--n-freq" && i + 1 < argc) {
            o.n_freq = std::stoull(argv[++i]);
        } else if (a == "--n-ant" && i + 1 < argc) {
            o.n_ant = std::stoull(argv[++i]);
        } else if (a == "--n-bf-beams" && i + 1 < argc) {
            o.n_bf_beams = std::stoull(argv[++i]);
        } else if (a == "--n-tr-beams" && i + 1 < argc) {
            o.n_tr_beams = std::stoull(argv[++i]);
        } else if (a == "--integration-spectra" && i + 1 < argc) {
            o.integration_spectra = std::stoull(argv[++i]);
        } else if (a == "--warmup-runs" && i + 1 < argc) {
            o.warmup_runs = std::stoull(argv[++i]);
        } else if (a == "--repeat" && i + 1 < argc) {
            o.repeat = std::stoull(argv[++i]);
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
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return ms;
}

beamformer::Weights make_test_weights(const beamformer::Dimensions& dims) {
    const std::size_t count = dims.n_beams * dims.n_freq * dims.n_ant;
    beamformer::Weights weights(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float angle = static_cast<float>(i % 360) * 3.14159265358979323846F / 180.0F;
        weights[i] = {std::cos(angle) * 0.125F, std::sin(angle) * 0.125F};
    }
    return weights;
}

struct EngineResult {
    std::string mode;
    double total_wall_ms = 0.0;
    double gpu_compute_ms = 0.0;
    double speedup = 1.0;
    std::string memory_note;
};

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_options(argc, argv);

    const beamformer::Dimensions bf_dims{opts.n_time, opts.n_freq, opts.n_ant, opts.n_bf_beams};
    const beamformer::Dimensions tr_dims{opts.n_time, opts.n_freq, opts.n_ant, opts.n_tr_beams};
    const beamformer::TemporalIntegrationConfig tint{opts.integration_spectra};

    const auto bf_weights = make_test_weights(bf_dims);
    const std::vector<beamformer::TrackerTrajectoryConfig> trajs{
        {beamformer::direction_from_lm(0.02F, 0.03F), {0.00001F, 0.00001F}}
    };

    const auto packed = beamformer::make_noise(bf_dims, 7777);
    const std::size_t bf_out_count = beamformer::integrated_intensity_count(bf_dims, tint);
    const std::size_t tr_out_count = tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams;

    std::vector<float> bf_output(bf_out_count, 0.0F);
    std::vector<float> tr_output(tr_out_count, 0.0F);

    std::cout << "================================================================================\n";
    std::cout << " CUDA Dual Engine Parallel Execution Benchmark\n";
    std::cout << " Array: n_ant=" << opts.n_ant << " | n_freq=" << opts.n_freq << " | n_time=" << opts.n_time << "\n";
    std::cout << " Fixed-Grid Beamformer: " << opts.n_bf_beams << " beams | Dynamic Tracker: " << opts.n_tr_beams << " beams\n";
    std::cout << " Integration: " << opts.integration_spectra << " spectra\n";
    std::cout << "================================================================================\n\n";

    std::vector<EngineResult> results;

    // 1. Baseline: Sequential Independent (Two separate uploads, separate sequential kernels)
    {
        auto samples = time_runs(opts.warmup_runs, opts.repeat, [&]() {
            // Upload & run beamformer
            auto bf_res = beamformer::cuda_beamform_v3_packed_integrated_intensity(
                packed, bf_weights, bf_dims, tint);
            // Upload & run tracker separately
            beamformer::TrackerConfig tr_cfg;
            tr_cfg.trajectory = trajs[0];
            tr_cfg.integration_spectra = tint.integration_spectra;
            auto tr_res = beamformer::cuda_beam_tracker_v5(
                packed, tr_dims, tr_cfg);
            cudaDeviceSynchronize();
        });
        const double med = median_inplace(samples);
        results.push_back({"1. Naive Sequential (2x H2D, 2x VRAM)", med, med, 1.0, "200% PCIe & VRAM"});
    }

    // 2. Parallel Dual Pipeline (1x Shared H2D, Concurrent Streams)
    {
        beamformer::DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);
        beamformer::DualParallelTimings last_timings;

        auto samples = time_runs(opts.warmup_runs, opts.repeat, [&]() {
            last_timings = pipeline.process_frame(packed.data(), bf_output.data(), tr_output.data());
        });
        const double med = median_inplace(samples);
        results.push_back({"2. Concurrent Multi-Stream (1x Shared H2D)", med, med, 1.0, "Saved 50% Input VRAM"});
        
        std::cout << " [Pipeline Breakdown] H2D=" << last_timings.host_to_device_ms
                  << " ms, BF=" << last_timings.beamformer_kernel_ms
                  << " ms, TR=" << last_timings.tracker_kernel_ms
                  << " ms, Concurrent GPU=" << last_timings.concurrent_gpu_ms
                  << " ms, D2H=" << last_timings.device_to_host_ms
                  << " ms (Efficiency: " << last_timings.parallel_efficiency << "x)\n\n";
    }

    // 3. Device-Resident Parallel Dual Pipeline (Direct GPU-to-GPU)
    {
        beamformer::DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint);

        std::uint8_t* d_packed = pipeline.device_shared_packed_voltage();
        float* d_bf = pipeline.device_beamformer_output();
        float* d_tr = pipeline.device_tracker_output();

        // One-time upload
        cudaMemcpy(d_packed, packed.data(), pipeline.shared_input_bytes(), cudaMemcpyHostToDevice);

        auto samples = time_runs(opts.warmup_runs, opts.repeat, [&]() {
            pipeline.process_device_resident(d_packed, d_bf, d_tr);
        });
        const double med = median_inplace(samples);
        results.push_back({"3. Device-Resident Concurrent (Zero PCIe)", med, med, 1.0, "Zero-Copy In-VRAM"});
    }

    // Calculate speedup vs Naive Sequential
    const double baseline_med = results.front().total_wall_ms;
    for (auto& r : results) {
        r.speedup = baseline_med / r.total_wall_ms;
    }

    std::cout << std::left
              << std::setw(44) << "Execution Mode"
              << std::right
              << std::setw(16) << "Latency (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(24) << "Resource Efficiency"
              << "\n";
    std::cout << std::string(98, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(44) << r.mode
                  << std::right
                  << std::fixed
                  << std::setprecision(3) << std::setw(16) << r.total_wall_ms
                  << std::setprecision(2) << std::setw(13) << r.speedup << "x"
                  << std::setw(24) << r.memory_note
                  << "\n";
    }
    std::cout << "================================================================================\n\n";

    return 0;
}
