// benchmarks/benchmark_cuda_beamformer_v3.cpp
//
// Multi-Generation Comparative Benchmark: Beamformer V1 vs V2 vs V3

#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_beamformer_v2.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/cpu_beamformer.hpp"
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
    std::size_t n_beams = 64;
    std::size_t warmup_runs = 1;
    std::size_t repeat = 3;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --n-time <N>       (default: 3200)\n"
              << "  --n-freq <N>       (default: 336)\n"
              << "  --n-ant <N>        (default: 64)\n"
              << "  --n-beams <N>      (default: 64)\n"
              << "  --warmup-runs <N>  (default: 1)\n"
              << "  --repeat <N>       (default: 3)\n"
              << "  --help             (print this message)\n";
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
        } else if (a == "--n-beams" && i + 1 < argc) {
            o.n_beams = std::stoull(argv[++i]);
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

struct EngineResult {
    std::string backend;
    std::string engine_name;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double speedup_vs_v1 = 1.0;
    double gmac_s = 0.0;
    double tflops_s = 0.0;
};

beamformer::Weights make_test_weights(const beamformer::Dimensions& dims) {
    const std::size_t count = dims.n_beams * dims.n_freq * dims.n_ant;
    beamformer::Weights weights(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float angle = static_cast<float>(i % 360) * 3.14159265358979323846F / 180.0F;
        weights[i] = {std::cos(angle) * 0.125F, std::sin(angle) * 0.125F};
    }
    return weights;
}

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_options(argc, argv);
    const beamformer::Dimensions dims{opts.n_time, opts.n_freq, opts.n_ant, opts.n_beams};
    const auto weights = make_test_weights(dims);
    const auto packed = beamformer::make_noise(dims, 42);

    const std::size_t total_outputs = dims.n_time * dims.n_freq * dims.n_beams;
    const std::size_t total_macs = total_outputs * dims.n_ant;
    const double total_flops = static_cast<double>(total_macs) * 8.0 + static_cast<double>(total_outputs) * 3.0;

    beamformer::Intensities output(total_outputs, 0.0F);

    std::cout << "================================================================================\n";
    std::cout << " CUDA Beamformer Multi-Generation Benchmark (V1 vs V2 vs V3)\n";
    std::cout << " Array: n_ant=" << dims.n_ant << " | n_freq=" << dims.n_freq
              << " | n_beams=" << dims.n_beams << " | n_time=" << dims.n_time
              << " | total_cmac=" << (total_macs / 1e9) << " G-MAC\n";
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
        const double gmac_s = (static_cast<double>(total_macs) / (med * 1e6));
        const double tflops_s = (total_flops / (med * 1e9));
        results.push_back({backend, name, med, min_t, max_t, 1.0, gmac_s, tflops_s});
    };

    // 1. GPU V1 Tiled (if n_ant <= 64)
    if (dims.n_ant <= 64) {
        benchmark_engine("gpu_v1", "v1_tiled", [&]() {
            beamformer::cuda_beamform_packed_intensity(
                packed, weights, dims, nullptr, beamformer::CudaBeamformerKernel::Tiled);
        });
    }

    // 2. GPU V2 Unified (u8)
    benchmark_engine("gpu_v2", "v2_unified_u8", [&]() {
        beamformer::V2BeamformerExecutionConfig cfg;
        cfg.time_unroll = 8;
        beamformer::cuda_beamform_v2_packed_intensity_into(
            packed, weights, dims, output, nullptr, cfg);
    });

    // 3. GPU V3 Dual-Beam Warp Co-Execution (u8)
    benchmark_engine("gpu_v3", "v3_dualbeam_u8", [&]() {
        beamformer::V3BeamformerExecutionConfig cfg;
        cfg.time_unroll = 8;
        cfg.beams_per_warp = 2;
        beamformer::cuda_beamform_v3_packed_intensity_into(
            packed, weights, dims, output, nullptr, cfg);
    });

    // 4. GPU V3 Batched Graph
    {
        beamformer::V3BeamformerExecutionConfig cfg;
        cfg.enable_cuda_graph = true;
        beamformer::BatchedBeamformerStreamV3 stream(dims, weights, cfg);
        benchmark_engine("gpu_v3", "v3_batched_graph", [&]() {
            stream.process_batch(packed.data(), output.data());
        });
        benchmark_engine("gpu_v3", "v3_device_resident", [&]() {
            stream.process_batch_kernel_only();
        });
    }

    // Calculate speedup vs V1 baseline
    const double v1_med = (!results.empty()) ? results.front().median_ms : 1.0;
    for (auto& r : results) {
        r.speedup_vs_v1 = v1_med / r.median_ms;
    }

    std::cout << std::left
              << std::setw(10) << "Backend"
              << std::setw(24) << "Engine"
              << std::right
              << std::setw(14) << "Median (ms)"
              << std::setw(14) << "vs V1 Tiled"
              << std::setw(14) << "GMAC/s"
              << std::setw(14) << "TFLOP/s"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(10) << r.backend
                  << std::setw(24) << r.engine_name
                  << std::right
                  << std::fixed
                  << std::setprecision(3) << std::setw(14) << r.median_ms
                  << std::setprecision(2) << std::setw(13) << r.speedup_vs_v1 << "x"
                  << std::setprecision(1) << std::setw(14) << r.gmac_s
                  << std::setprecision(2) << std::setw(14) << r.tflops_s
                  << "\n";
    }
    std::cout << "================================================================================\n\n";

    return 0;
}
