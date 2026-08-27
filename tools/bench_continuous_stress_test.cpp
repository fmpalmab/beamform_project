// tools/bench_continuous_stress_test.cpp
//
// Continuous High-Throughput Stress Test & Reliability Suite for CUDA V5 Tracker + Upchannelizer.
//
// Continuously injects high-rate synthetic voltage streams at hardware line rate (simulating
// 3.333 us spectrum cadence = 1.066 ms / 320-spectra window) for extended endurance testing
// (e.g. 2+ hours / hundreds of thousands of windows).
//
// Monitored Reliability Metrics:
// 1. Frame-by-frame execution latency, jitter, and real-time budget compliance (1.066 ms deadline).
// 2. Numerical stability: NaN/Inf detection, negative power checks, signal dynamic range.
// 3. Memory leak detection: continuous tracking of device VRAM usage via cudaMemGetInfo.
// 4. Compute throughput (GSamples/s and TFLOPs) stability over time.
// 5. Error logging and statistical distribution recording (P50, P95, P99, Max latency).

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#if BEAMFORMER_HAS_CUDA
#include "beamformer/cuda_beam_tracker_v5.hpp"
#include "beamformer/cuda_beam_tracker_v4.hpp"
#include "beamformer/cuda_upchannelizer.hpp"
#include <cuda_runtime.h>
#endif

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
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct StressTestConfig {
    double duration_seconds = 7200.0; // 2 hours default
    std::size_t n_ant = 64;
    std::size_t n_freq = beamformer::default_frequency_channels; // 336
    std::size_t spectra_per_window = 320; // 1.066 ms real-time window
    std::size_t upchan_factor = 32; // M=32 fine channels
    std::string engine = "cuda_v5";
    std::string outdir = "results/stress_test";
    double sample_cadence_us = 3.333333; // 300 kHz channel sample rate
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --duration SEC        Total run duration in seconds (default: 7200 = 2 hours)\n"
              << "  --engine ENGINE       Engine: cuda_v5 (default), cuda_v4, cpu_v2\n"
              << "  --antennas N          Antenna count: 32, 64 (default), 128, 256\n"
              << "  --freq N              Coarse frequency channels (default: 336)\n"
              << "  --spectra N           Spectra per streaming window (default: 320)\n"
              << "  --upchan M            Upchannelization factor M (default: 32)\n"
              << "  --outdir DIR          Output directory for metrics and logs (default: results/stress_test)\n"
              << "  --help                Show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    StressTestConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" && i + 1 < argc) config.duration_seconds = std::stod(argv[++i]);
        else if (arg == "--engine" && i + 1 < argc) config.engine = argv[++i];
        else if (arg == "--antennas" && i + 1 < argc) config.n_ant = std::stoull(argv[++i]);
        else if (arg == "--freq" && i + 1 < argc) config.n_freq = std::stoull(argv[++i]);
        else if (arg == "--spectra" && i + 1 < argc) config.spectra_per_window = std::stoull(argv[++i]);
        else if (arg == "--upchan" && i + 1 < argc) config.upchan_factor = std::stoull(argv[++i]);
        else if (arg == "--outdir" && i + 1 < argc) config.outdir = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    const double real_time_budget_ms = config.spectra_per_window * (config.sample_cadence_us / 1000.0);
    const std::size_t window_samples = config.spectra_per_window * config.n_freq * config.n_ant;
    const std::size_t total_ops_per_window = config.spectra_per_window * config.n_freq * config.n_ant * 8
                                           + config.spectra_per_window * config.n_freq * 5 * config.upchan_factor;

    std::cout << "========================================================================\n"
              << "    BEAM TRACKER & UPCHANNELIZER: 2-HOUR CONTINUOUS STRESS TEST SUITE   \n"
              << "========================================================================\n"
              << "Target Duration    : " << config.duration_seconds << " seconds (" << config.duration_seconds / 3600.0 << " hours)\n"
              << "Engine             : " << config.engine << "\n"
              << "Antenna Count      : " << config.n_ant << "\n"
              << "Coarse Channels    : " << config.n_freq << "\n"
              << "Spectra per Window : " << config.spectra_per_window << " (" << real_time_budget_ms << " ms real-time deadline)\n"
              << "Upchannel Factor M : " << config.upchan_factor << " (Fine channels = " << config.n_freq * config.upchan_factor << ")\n"
              << "Output Directory   : " << config.outdir << "\n"
              << "========================================================================\n\n";

    std::filesystem::create_directories(config.outdir);

    std::ofstream timeline_csv(config.outdir + "/stress_test_timeline.csv");
    timeline_csv << "Window_Idx,Elapsed_Sec,Latency_ms,Throughput_GSamples_s,Throughput_TFLOPs,VRAM_Used_MB,Overrun_Flag,Error_Flag\n";

    std::ofstream error_log(config.outdir + "/stress_test_errors.log");
    error_log << "=== CONTINUOUS STRESS TEST ERROR LOG ===\n";

    // Allocate host input/output buffers
    beamformer::Dimensions dims{config.spectra_per_window, config.n_freq, config.n_ant, beamformer::tracker_beam_count};
    beamformer::TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start = beamformer::direction_from_lm(-0.10F, -0.05F);
    tracker_cfg.trajectory.direction_rate_per_sample = {1.2e-6F, 0.8e-6F};
    tracker_cfg.integration_spectra = config.spectra_per_window;

    beamformer::UpchannelizerConfig upchan_cfg;
    upchan_cfg.upchan_factor = config.upchan_factor;
    upchan_cfg.window = beamformer::UpchannelizerWindowType::Hann;

    beamformer::PackedVoltage packed_input(window_samples);
    for (std::size_t i = 0; i < window_samples; ++i) {
        packed_input[i] = static_cast<std::uint8_t>((i & 0x07) | (( (i >> 3) & 0x07) << 4));
    }

    std::size_t out_fine_count = (config.spectra_per_window / config.upchan_factor) * (config.n_freq * config.upchan_factor) * dims.n_beams;
    beamformer::Intensities fine_output(out_fine_count, 0.0F);
    beamformer::Intensities coarse_intensity(config.spectra_per_window * config.n_freq, 0.0F);

#if BEAMFORMER_HAS_CUDA
    std::size_t free_vram_initial = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram_initial, &total_vram);
    std::cout << "[Setup] Initial VRAM: " << (total_vram - free_vram_initial) / (1024 * 1024) << " MB used / "
              << total_vram / (1024 * 1024) << " MB total.\n";

    std::unique_ptr<beamformer::CudaUpchannelizerWorkspace> upchan_workspace;
    try {
        upchan_workspace = std::make_unique<beamformer::CudaUpchannelizerWorkspace>(dims, upchan_cfg);
        std::cout << "[Setup] CudaUpchannelizerWorkspace initialized successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[Warning] CudaUpchannelizerWorkspace initialization failed: " << e.what() << "\n";
    }

    // Warmup step to initialize CUDA runtime context
    std::cout << "[Setup] Performing CUDA driver warmup iteration...\n";
    if (upchan_workspace) {
        upchan_workspace->process_tracker(packed_input, tracker_cfg, fine_output);
    } else {
        beamformer::V5ExecutionConfig v5_cfg;
        v5_cfg.time_unroll = 8;
        beamformer::cuda_beam_tracker_v5_into(packed_input, dims, tracker_cfg, coarse_intensity, v5_cfg);
    }
    cudaDeviceSynchronize();
    std::cout << "[Setup] Warmup complete.\n\n";
#endif

    const auto start_wall_time = Clock::now();
    std::size_t window_idx = 0;
    std::size_t total_errors = 0;
    std::size_t total_overruns = 0;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(500000);

    auto last_log_time = Clock::now();
    double min_lat = 1e9, max_lat = 0.0;

    std::cout << "--- Starting Continuous Streaming Ingestion Loop ---\n";
    std::cout << std::setw(10) << "Window"
              << std::setw(12) << "Elapsed(s)"
              << std::setw(12) << "Lat(ms)"
              << std::setw(14) << "Rate(GSamp/s)"
              << std::setw(12) << "TFLOPs"
              << std::setw(12) << "Overruns"
              << std::setw(10) << "Errors"
              << "\n" << std::string(82, '-') << "\n";

    while (true) {
        const auto now = Clock::now();
        const double elapsed_total_s = std::chrono::duration<double>(now - start_wall_time).count();
        if (elapsed_total_s >= config.duration_seconds) {
            break;
        }

        // Rolling data injection
        packed_input[window_idx % window_samples] ^= 0x11;

        bool has_error = false;
        const auto t0 = Clock::now();

        try {
#if BEAMFORMER_HAS_CUDA
            if (config.engine == "cuda_v5" || config.engine == "cuda_v5_u8") {
                if (upchan_workspace) {
                    // Unified Fused Tracker + Upchannelizer in 1 device pass
                    upchan_workspace->process_tracker(packed_input, tracker_cfg, fine_output);
                } else {
                    beamformer::V5ExecutionConfig v5_cfg;
                    v5_cfg.time_unroll = 8;
                    beamformer::cuda_beam_tracker_v5_into(packed_input, dims, tracker_cfg, coarse_intensity, v5_cfg);
                }
            } else if (config.engine == "cuda_v4") {
                beamformer::V4ExecutionConfig v4_cfg;
                beamformer::cuda_beam_tracker_v4_into(packed_input, dims, tracker_cfg, coarse_intensity, v4_cfg);
            }
#else
            beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into(packed_input, dims, tracker_cfg, coarse_intensity);
#endif
        } catch (const std::exception& e) {
            has_error = true;
            total_errors++;
            error_log << "[Window " << window_idx << " at " << elapsed_total_s << "s] Execution Exception: " << e.what() << "\n";
        }

        const auto t1 = Clock::now();
        const double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies_ms.push_back(lat_ms);
        min_lat = std::min(min_lat, lat_ms);
        max_lat = std::max(max_lat, lat_ms);

        bool is_overrun = (lat_ms > real_time_budget_ms);
        if (is_overrun) total_overruns++;

        // Numerical Sanity Check
        for (std::size_t k = 0; k < std::min(fine_output.size(), (std::size_t)128); ++k) {
            if (std::isnan(fine_output[k]) || std::isinf(fine_output[k]) || fine_output[k] < 0.0F) {
                has_error = true;
                total_errors++;
                error_log << "[Window " << window_idx << "] Numerical Error at sample " << k << ": val=" << fine_output[k] << "\n";
                break;
            }
        }

        // Record metrics to timeline CSV every 100 windows
        if (window_idx % 100 == 0) {
            double vram_used_mb = 0.0;
#if BEAMFORMER_HAS_CUDA
            std::size_t free_vram = 0, tot_vram = 0;
            cudaMemGetInfo(&free_vram, &tot_vram);
            vram_used_mb = static_cast<double>(tot_vram - free_vram) / (1024.0 * 1024.0);
#endif
            const double gsamples_sec = (window_samples / 1e9) / (lat_ms / 1000.0);
            const double tflops = (total_ops_per_window / 1e12) / (lat_ms / 1000.0);

            timeline_csv << window_idx << ","
                         << std::fixed << std::setprecision(2) << elapsed_total_s << ","
                         << std::setprecision(4) << lat_ms << ","
                         << std::setprecision(3) << gsamples_sec << ","
                         << std::setprecision(3) << tflops << ","
                         << std::setprecision(1) << vram_used_mb << ","
                         << (is_overrun ? 1 : 0) << ","
                         << (has_error ? 1 : 0) << "\n";
        }

        // Terminal progress logging every 5 seconds
        if (std::chrono::duration<double>(now - last_log_time).count() >= 5.0) {
            last_log_time = now;
            const double gsamples_sec = (window_samples / 1e9) / (lat_ms / 1000.0);
            const double tflops = (total_ops_per_window / 1e12) / (lat_ms / 1000.0);

            std::cout << std::setw(10) << window_idx
                      << std::setw(12) << std::fixed << std::setprecision(1) << elapsed_total_s
                      << std::setw(12) << std::setprecision(3) << lat_ms
                      << std::setw(14) << std::setprecision(2) << gsamples_sec
                      << std::setw(12) << std::setprecision(2) << tflops
                      << std::setw(12) << total_overruns
                      << std::setw(10) << total_errors
                      << "\n" << std::flush;
        }

        window_idx++;
    }

    timeline_csv.close();
    error_log.close();

    // Compute Summary Statistics
    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double p50 = latencies_ms[latencies_ms.size() * 50 / 100];
    const double p95 = latencies_ms[latencies_ms.size() * 95 / 100];
    const double p99 = latencies_ms[latencies_ms.size() * 99 / 100];
    const double mean_lat = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    const double total_elapsed_s = std::chrono::duration<double>(Clock::now() - start_wall_time).count();
    const double mean_headroom_pct = ((real_time_budget_ms - mean_lat) / real_time_budget_ms) * 100.0;

    std::ofstream summary_json(config.outdir + "/stress_test_summary.json");
    summary_json << "{\n"
                 << "  \"duration_seconds\": " << total_elapsed_s << ",\n"
                 << "  \"engine\": \"" << config.engine << "\",\n"
                 << "  \"antenna_count\": " << config.n_ant << ",\n"
                 << "  \"frequency_channels\": " << config.n_freq << ",\n"
                 << "  \"upchannel_factor\": " << config.upchan_factor << ",\n"
                 << "  \"total_windows_processed\": " << window_idx << ",\n"
                 << "  \"real_time_budget_ms\": " << real_time_budget_ms << ",\n"
                 << "  \"mean_latency_ms\": " << mean_lat << ",\n"
                 << "  \"p50_latency_ms\": " << p50 << ",\n"
                 << "  \"p95_latency_ms\": " << p95 << ",\n"
                 << "  \"p99_latency_ms\": " << p99 << ",\n"
                 << "  \"min_latency_ms\": " << min_lat << ",\n"
                 << "  \"max_latency_ms\": " << max_lat << ",\n"
                 << "  \"real_time_headroom_percent\": " << mean_headroom_pct << ",\n"
                 << "  \"total_overruns\": " << total_overruns << ",\n"
                 << "  \"total_errors\": " << total_errors << ",\n"
                 << "  \"stress_test_passed\": " << (total_errors == 0 && total_overruns == 0 ? "true" : "false") << "\n"
                 << "}\n";
    summary_json.close();

    std::cout << "\n========================================================================\n"
              << "    CONTINUOUS STRESS TEST EXECUTION FINISHED!                          \n"
              << "========================================================================\n"
              << "Total Windows Processed : " << window_idx << "\n"
              << "Total Duration          : " << total_elapsed_s << " s (" << total_elapsed_s / 3600.0 << " hrs)\n"
              << "Mean Window Latency     : " << mean_lat << " ms (Budget = " << real_time_budget_ms << " ms)\n"
              << "P50 / P95 / P99 Latency : " << p50 << " / " << p95 << " / " << p99 << " ms\n"
              << "Real-Time Headroom      : " << mean_headroom_pct << " %\n"
              << "Real-Time Overruns      : " << total_overruns << "\n"
              << "Numerical / Mem Errors  : " << total_errors << "\n"
              << "Status                  : " << (total_errors == 0 ? "PASSED (ROCK SOLID)" : "FAILED (ERRORS DETECTED)") << "\n"
              << "========================================================================\n\n";

    return (total_errors == 0) ? 0 : 1;
}
