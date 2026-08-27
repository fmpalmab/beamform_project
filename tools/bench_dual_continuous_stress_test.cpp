// tools/bench_dual_continuous_stress_test.cpp
//
// Continuous Endurance & Real-Time Stress Testing Harness for Dual Engine
// Concurrent Execution (Fixed-Grid Beamformer V3 + Dynamic Beam Tracker V5).

#include "beamformer/cuda_dual_parallel_pipeline.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
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
    std::size_t frame_count = 100;
    bool enable_graph = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --n-time <N>               (default: 3200)\n"
              << "  --n-freq <N>               (default: 336)\n"
              << "  --n-ant <N>                (default: 64)\n"
              << "  --n-bf-beams <N>           (default: 64)\n"
              << "  --n-tr-beams <N>           (default: 1)\n"
              << "  --integration-spectra <N>  (default: 320)\n"
              << "  --frame-count <N>          (default: 100)\n"
              << "  --cuda-graph               (enable CUDA graph capture)\n"
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
        } else if (a == "--frame-count" && i + 1 < argc) {
            o.frame_count = std::stoull(argv[++i]);
        } else if (a == "--cuda-graph") {
            o.enable_graph = true;
        }
    }
    return o;
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

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_options(argc, argv);

    const beamformer::Dimensions bf_dims{opts.n_time, opts.n_freq, opts.n_ant, opts.n_bf_beams};
    const beamformer::Dimensions tr_dims{opts.n_time, opts.n_freq, opts.n_ant, opts.n_tr_beams};
    const beamformer::TemporalIntegrationConfig tint{opts.integration_spectra};

    std::cout << "================================================================================\n";
    std::cout << " CUDA Dual Engine Concurrent Real-Time Continuous Stress Test\n";
    std::cout << " Array: n_ant=" << opts.n_ant << " | n_freq=" << opts.n_freq << " | n_time=" << opts.n_time << "\n";
    std::cout << " Fixed-Grid Beamformer: " << opts.n_bf_beams << " beams | Dynamic Tracker: " << opts.n_tr_beams << " beams\n";
    std::cout << " Total Frames: " << opts.frame_count << " | Integration: " << opts.integration_spectra << " spectra\n";
    std::cout << "================================================================================\n\n";

    // Measure initial GPU VRAM
    std::size_t free_mem_start = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem_start, &total_mem);

    const auto bf_weights = make_test_weights(bf_dims);
    const std::vector<beamformer::TrackerTrajectoryConfig> trajs{
        {beamformer::direction_from_lm(0.02F, 0.03F), {0.00001F, 0.00001F}}
    };

    beamformer::DualParallelConfig cfg;
    cfg.enable_cuda_graph = opts.enable_graph;

    beamformer::DualEngineParallelPipeline pipeline(bf_dims, bf_weights, tr_dims, trajs, tint, cfg);

    std::size_t free_mem_after_alloc = 0;
    cudaMemGetInfo(&free_mem_after_alloc, &total_mem);

    std::cout << " GPU Memory Footprint:\n";
    std::cout << "  - Initial Free VRAM:       " << (free_mem_start / (1024 * 1024)) << " MB\n";
    std::cout << "  - After Pipeline Setup:    " << (free_mem_after_alloc / (1024 * 1024)) << " MB\n";
    std::cout << "  - VRAM Allocated:          " << ((free_mem_start - free_mem_after_alloc) / (1024 * 1024)) << " MB\n";
    std::cout << "  - Shared Input Buffer:     " << (pipeline.shared_input_bytes() / (1024 * 1024)) << " MB\n";
    std::cout << "  - Input VRAM Saved (50%):  " << (pipeline.memory_saved_bytes() / (1024 * 1024)) << " MB\n\n";

    const auto packed = beamformer::make_noise(bf_dims, 9999);
    const std::size_t bf_out_count = beamformer::integrated_intensity_count(bf_dims, tint);
    const std::size_t tr_out_count = tr_dims.n_time * tr_dims.n_freq * tr_dims.n_beams;

    std::vector<float> bf_output(bf_out_count, 0.0F);
    std::vector<float> tr_output(tr_out_count, 0.0F);

    std::vector<double> frame_times;
    std::vector<double> concurrent_times;
    std::vector<double> h2d_times;
    frame_times.reserve(opts.frame_count);
    concurrent_times.reserve(opts.frame_count);
    h2d_times.reserve(opts.frame_count);

    std::size_t nan_inf_count = 0;

    // 1 frame warmup to initialize CUDA stream pools & driver contexts
    pipeline.process_frame(packed.data(), bf_output.data(), tr_output.data());
    cudaDeviceSynchronize();

    std::size_t free_mem_baseline = 0;
    cudaMemGetInfo(&free_mem_baseline, &total_mem);

    std::cout << " Running continuous streaming stress test (100 frames)...\n";
    const auto t_stress_start = Clock::now();

    for (std::size_t frame = 0; frame < opts.frame_count; ++frame) {
        const auto t = pipeline.process_frame(packed.data(), bf_output.data(), tr_output.data());
        frame_times.push_back(t.total_wall_ms);
        concurrent_times.push_back(t.concurrent_gpu_ms);
        h2d_times.push_back(t.host_to_device_ms);

        // Sanity check periodically (every 10 frames)
        if (frame % 10 == 0) {
            for (std::size_t i = 0; i < bf_out_count; i += 100) {
                if (std::isnan(bf_output[i]) || std::isinf(bf_output[i])) ++nan_inf_count;
            }
            for (std::size_t i = 0; i < tr_out_count; ++i) {
                if (std::isnan(tr_output[i]) || std::isinf(tr_output[i])) ++nan_inf_count;
            }
        }
    }

    const auto t_stress_end = Clock::now();
    const double total_stress_s = std::chrono::duration<double>(t_stress_end - t_stress_start).count();

    cudaDeviceSynchronize();
    // Verify GPU memory leak after stress run
    std::size_t free_mem_end = 0;
    cudaMemGetInfo(&free_mem_end, &total_mem);

    std::sort(frame_times.begin(), frame_times.end());
    std::sort(concurrent_times.begin(), concurrent_times.end());

    const double med_total = frame_times[frame_times.size() / 2];
    const double min_total = frame_times.front();
    const double max_total = frame_times.back();
    const double p95_total = frame_times[static_cast<std::size_t>(frame_times.size() * 0.95)];
    const double p99_total = frame_times[static_cast<std::size_t>(frame_times.size() * 0.99)];

    const double med_concurrent = concurrent_times[concurrent_times.size() / 2];
    const double min_concurrent = concurrent_times.front();
    const double max_concurrent = concurrent_times.back();

    std::cout << "\n================================================================================\n";
    std::cout << " Stress Test Results:\n";
    std::cout << "================================================================================\n";
    std::cout << "  - Total Execution Time:    " << std::fixed << std::setprecision(2) << total_stress_s << " s\n";
    std::cout << "  - Processed Frames:        " << opts.frame_count << "\n";
    std::cout << "  - Streaming Frame Rate:    " << (opts.frame_count / total_stress_s) << " fps\n";
    std::cout << "  - Numerical Anomalies:     " << nan_inf_count << " (NaN/Inf)\n";
    std::cout << "  - Memory Leak Detected:    " << ((free_mem_baseline == free_mem_end) ? "NO (0 bytes leaked)" : "YES") << "\n\n";

    std::cout << " End-to-End Frame Latency (ms):\n";
    std::cout << "  - Min:                     " << min_total << " ms\n";
    std::cout << "  - Median:                  " << med_total << " ms\n";
    std::cout << "  - P95:                     " << p95_total << " ms\n";
    std::cout << "  - P99:                     " << p99_total << " ms\n";
    std::cout << "  - Max:                     " << max_total << " ms\n\n";

    std::cout << " Concurrent GPU Compute Kernel Latency (ms):\n";
    std::cout << "  - Min:                     " << min_concurrent << " ms\n";
    std::cout << "  - Median:                  " << med_concurrent << " ms\n";
    std::cout << "  - Max:                     " << max_concurrent << " ms\n";
    std::cout << "================================================================================\n";

    return (nan_inf_count == 0 && free_mem_baseline == free_mem_end) ? 0 : 1;
}
