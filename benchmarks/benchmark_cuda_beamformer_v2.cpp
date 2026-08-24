// benchmarks/benchmark_cuda_beamformer_v2.cpp
//
// Comprehensive Performance Benchmark Suite for Fixed-Grid Voltage Beamformers:
// CPU Reference vs CUDA V1 (Direct, Tiled) vs CUDA V2 (Unified Warp Reduction U2, U4, U8,
// Batched Stream, CUDA Graph, Device Resident, Fused Temporal Integration).

#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/cuda_beamformer.hpp"
#include "beamformer/cuda_beamformer_v2.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

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
using beamformer::ComplexFloat;
using beamformer::Dimensions;
using beamformer::Intensities;
using beamformer::IntegratedIntensities;
using beamformer::PackedVoltage;
using beamformer::TemporalIntegrationConfig;
using beamformer::V2BeamformerExecutionConfig;
using beamformer::Weights;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t n_beams = 64;
    std::size_t integration_spectra = 320;
    std::size_t warmup_runs = 1;
    std::size_t repeat = 3;
    std::size_t threads = 0;
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
        << "  Benchmark CPU vs CUDA V1 (Direct, Tiled) vs CUDA V2 (Unified Warp Reduction) beamformers.\n\n"
        << "  --n-time N                 default 15360\n"
        << "  --n-ant N                  32, 64, 128, or 256; default 64\n"
        << "  --n-beams N                default 64\n"
        << "  --integration-spectra N    default 320\n"
        << "  --warmup-runs N            default 2\n"
        << "  --repeat N                 median over repeats; default 5\n"
        << "  --threads N                OpenMP threads for CPU; 0 = default\n"
        << "  --outdir DIR               write CSVs and metadata there\n";
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
        } else if (a == "--n-beams") {
            o.n_beams = parse_size(require_value(argc, argv, i), "--n-beams");
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

struct EngineResult {
    std::string backend;
    std::string engine_name;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double speedup_vs_v1_direct = 1.0;
    double speedup_vs_cpu = 1.0;
    double gmac_per_sec = 0.0;
    double gflops_per_sec = 0.0;
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

        const Dimensions dims{opts.n_time, beamformer::default_frequency_channels, opts.n_ant, opts.n_beams};
        const auto positions = beamformer::default_positions(dims.n_ant);
        const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);
        const auto directions = (dims.n_ant <= 64)
                                    ? beamformer::fft_beam_grid(dims.n_ant, dims.n_beams)
                                    : beamformer::default_beam_grid(dims.n_beams);

        const auto direct_weights = beamformer::generate_weights(dims, positions, frequencies, directions);
        const auto tiled_weights = (dims.n_ant <= 64)
                                       ? beamformer::generate_tiled_weights(dims, positions, frequencies, directions)
                                       : direct_weights;

        const auto packed = beamformer::make_noise(dims, 42);

        const std::size_t total_outputs = dims.n_time * dims.n_freq * dims.n_beams;
        const std::size_t total_macs = total_outputs * dims.n_ant;
        const double total_flops = static_cast<double>(total_macs) * 8.0 + static_cast<double>(total_outputs) * 3.0;

        Intensities output(total_outputs, 0.0F);

        std::cout << "================================================================================\n";
        std::cout << " CUDA Beamformer V2 Multi-Generation Benchmark Suite\n";
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
            const double gflops_s = (total_flops / (med * 1e6));
            results.push_back({backend, name, med, min_t, max_t, 1.0, 1.0, gmac_s, gflops_s});
        };

        // 1. CPU Reference
        benchmark_engine("cpu", "reference_cpu", [&]() {
            beamformer::cpu_beamform_packed_intensity_into(packed, direct_weights, dims, output);
        });

        // 2. CUDA V1 Direct
        benchmark_engine("gpu_v1", "v1_direct", [&]() {
            beamformer::cuda_beamform_packed_intensity(packed, direct_weights, dims, nullptr, beamformer::CudaBeamformerKernel::Direct);
        });

        // 3. CUDA V1 Tiled (if n_ant <= 64)
        if (dims.n_ant <= 64) {
            benchmark_engine("gpu_v1", "v1_tiled", [&]() {
                beamformer::cuda_beamform_packed_intensity(packed, tiled_weights, dims, nullptr, beamformer::CudaBeamformerKernel::Tiled);
            });
        }

        // 4. CUDA V2 Unified (Unroll 2, 4, 8)
        benchmark_engine("gpu_v2", "v2_unified_u2", [&]() {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 2;
            beamformer::cuda_beamform_v2_packed_intensity_into(packed, direct_weights, dims, output, nullptr, cfg);
        });

        benchmark_engine("gpu_v2", "v2_unified_u4", [&]() {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 4;
            beamformer::cuda_beamform_v2_packed_intensity_into(packed, direct_weights, dims, output, nullptr, cfg);
        });

        benchmark_engine("gpu_v2", "v2_unified_u8", [&]() {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 8;
            beamformer::cuda_beamform_v2_packed_intensity_into(packed, direct_weights, dims, output, nullptr, cfg);
        });

        // 5. CUDA V2 Batched Stream
        {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = false;
            auto stream = std::make_unique<beamformer::BatchedBeamformerStreamV2>(dims, direct_weights, cfg);
            benchmark_engine("gpu_v2", "v2_batched_stream", [&]() {
                stream->process_batch(packed.data(), output.data());
            });
        }

        // 6. CUDA V2 Batched CUDA Graph
        {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = true;
            auto stream = std::make_unique<beamformer::BatchedBeamformerStreamV2>(dims, direct_weights, cfg);
            stream->process_batch(packed.data(), output.data());
            benchmark_engine("gpu_v2", "v2_batched_graph", [&]() {
                stream->process_batch(packed.data(), output.data());
            });
        }

        // 7. CUDA V2 Batched Kernel Only
        {
            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 8;
            cfg.time_chunk_size = 80;
            cfg.enable_cuda_graph = true;
            auto stream = std::make_unique<beamformer::BatchedBeamformerStreamV2>(dims, direct_weights, cfg);
            stream->process_batch(packed.data(), output.data());
            benchmark_engine("gpu_v2", "v2_batched_kernel_only", [&]() {
                stream->process_batch_kernel_only();
            });
        }

        // 8. CUDA V2 Device Resident
        {
            const std::size_t v_bytes = beamformer::voltage_sample_count(dims) * sizeof(std::uint8_t);
            const std::size_t w_bytes = dims.n_beams * dims.n_freq * dims.n_ant * sizeof(ComplexFloat);
            const std::size_t out_bytes = total_outputs * sizeof(float);

            std::uint8_t* d_packed = nullptr;
            ComplexFloat* d_weights = nullptr;
            float* d_intensity = nullptr;

            cudaMalloc(reinterpret_cast<void**>(&d_packed), v_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_weights), w_bytes);
            cudaMalloc(reinterpret_cast<void**>(&d_intensity), out_bytes);

            cudaMemcpy(d_packed, packed.data(), v_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, direct_weights.data(), w_bytes, cudaMemcpyHostToDevice);

            V2BeamformerExecutionConfig cfg;
            cfg.time_unroll = 8;
            benchmark_engine("gpu_v2", "v2_device_resident", [&]() {
                beamformer::cuda_beamform_v2_device_resident(d_packed, d_weights, d_intensity, dims, cfg);
            });

            cudaFree(d_packed);
            cudaFree(d_weights);
            cudaFree(d_intensity);
        }

        // 9. CUDA V2 Fused Temporal Integration (320 spectra)
        {
            TemporalIntegrationConfig temporal_cfg{opts.integration_spectra};
            IntegratedIntensities int_out(beamformer::integrated_intensity_count(dims, temporal_cfg));
            benchmark_engine("gpu_v2", "v2_fused_int320", [&]() {
                beamformer::cuda_beamform_v2_packed_integrated_intensity_into(
                    packed, direct_weights, dims, temporal_cfg, int_out);
            });
        }

        // 10. CUDA V2 Fused Temporal Integration (10 spectra)
        {
            TemporalIntegrationConfig temporal_cfg{10};
            IntegratedIntensities int_out(beamformer::integrated_intensity_count(dims, temporal_cfg));
            benchmark_engine("gpu_v2", "v2_fused_int10", [&]() {
                beamformer::cuda_beamform_v2_packed_integrated_intensity_into(
                    packed, direct_weights, dims, temporal_cfg, int_out);
            });
        }

        // Compute speedups
        double cpu_ms = results[0].median_ms;
        double v1_direct_ms = results.size() > 1 ? results[1].median_ms : cpu_ms;

        for (auto& r : results) {
            r.speedup_vs_v1_direct = v1_direct_ms / r.median_ms;
            r.speedup_vs_cpu = cpu_ms / r.median_ms;
        }

        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(26) << "Engine"
                  << std::setw(14) << "Median (ms)"
                  << std::setw(14) << "vs V1 Direct"
                  << std::setw(14) << "vs CPU"
                  << std::setw(14) << "GMAC/s"
                  << std::setw(14) << "TFLOP/s" << "\n";
        std::cout << std::string(106, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(10) << r.backend
                      << std::setw(26) << r.engine_name
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.median_ms
                      << std::setw(14) << std::setprecision(2) << std::string(std::to_string(r.speedup_vs_v1_direct).substr(0, 4) + "x")
                      << std::setw(14) << std::setprecision(2) << std::string(std::to_string(r.speedup_vs_cpu).substr(0, 4) + "x")
                      << std::setw(14) << std::setprecision(1) << r.gmac_per_sec
                      << std::setw(14) << std::setprecision(2) << (r.gflops_per_sec / 1e3)
                      << "\n";
        }
        std::cout << "================================================================================\n\n";

        if (opts.outdir) {
            std::filesystem::create_directories(*opts.outdir);
            std::ofstream csv(*opts.outdir / "benchmark_cuda_beamformer_v2_summary.csv");
            csv << "backend,engine,median_ms,min_ms,max_ms,speedup_vs_v1,speedup_vs_cpu,gmac_per_s,gflops_per_s,n_ant,n_beams,n_time\n";
            for (const auto& r : results) {
                csv << r.backend << "," << r.engine_name << ","
                    << r.median_ms << "," << r.min_ms << "," << r.max_ms << ","
                    << r.speedup_vs_v1_direct << "," << r.speedup_vs_cpu << ","
                    << r.gmac_per_sec << "," << r.gflops_per_sec << ","
                    << dims.n_ant << "," << dims.n_beams << "," << dims.n_time << "\n";
            }
            std::cout << "Wrote summary CSV to: " << (*opts.outdir / "benchmark_cuda_beamformer_v2_summary.csv") << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
