// benchmarks/benchmark_frb_classifier.cpp
//
// Performance benchmark for the real-time CUDA FRB dedispersion, boxcar matched filtering,
// and candidate extraction pipeline (FRBClassifierStreamV5).

#include "beamformer/frb_classifier.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace beamformer;

namespace {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

void benchmark_configuration(std::size_t n_time, std::size_t n_dm, int warmup_runs, int bench_runs) {
    Dimensions dims;
    dims.n_time = n_time;
    dims.n_freq = default_frequency_channels; // 336
    dims.n_ant = 64;
    dims.n_beams = 1;

    FRBClassifierConfig cfg;
    cfg.n_dm = n_dm;
    cfg.dm_min = 0.0F;
    cfg.dm_max = 1500.0F;
    cfg.snr_threshold = 6.0F;

    const std::size_t n_floats = dims.n_time * dims.n_freq * dims.n_beams;
    const std::size_t bytes = n_floats * sizeof(float);

    float* d_intensity = nullptr;
    check_cuda(cudaMalloc(&d_intensity, bytes), "cudaMalloc d_intensity");
    check_cuda(cudaMemset(d_intensity, 0, bytes), "cudaMemset d_intensity");

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    FRBClassifierStreamV5 classifier(d_intensity, static_cast<void*>(stream), dims, cfg);

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        classifier.run(0);
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup sync");

    // Timed runs
    float total_kernel_ms = 0.0F;
    float total_ms = 0.0F;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_runs; ++i) {
        classifier.run(static_cast<std::size_t>(i));
        total_kernel_ms += classifier.last_kernel_time_ms();
        total_ms += classifier.last_total_time_ms();
    }
    check_cuda(cudaStreamSynchronize(stream), "bench sync");
    auto t1 = std::chrono::high_resolution_clock::now();

    double wall_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_runs;
    double avg_kernel_ms = total_kernel_ms / bench_runs;
    double avg_total_ms = total_ms / bench_runs;

    // Cadence calculation
    double physical_window_ms = dims.n_time * (10.0 / 3.0) * 1e-3; // ms
    double real_time_factor = physical_window_ms / avg_total_ms;
    double samples_per_sec = (dims.n_time * dims.n_freq) / (avg_total_ms * 1e-3) / 1e6; // MSamp/s

    std::printf("%7zu | %6zu | %11.2f | %10.2f | %11.2f | %11.2f | %10.2fx | %9.1f MS/s\n",
                n_time, n_dm, physical_window_ms, avg_kernel_ms, avg_total_ms, wall_time_ms,
                real_time_factor, samples_per_sec);

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(d_intensity), "cudaFree");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("No CUDA device found.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("=================================================================================================\n");
    std::printf("                REAL-TIME CUDA FRB CLASSIFIER & SEARCH PIPELINE BENCHMARK                         \n");
    std::printf("Device: %s (SM %d.%d, %d SMs, %zu MB Global Mem)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / (1024 * 1024));
    std::printf("=================================================================================================\n");
    std::printf(" N_Time |   N_DM | Phys Win ms | Kernel ms  | Pipeline ms | Wall Time ms| RT Factor   | Throughput\n");
    std::printf("-------------------------------------------------------------------------------------------------\n");

    const std::size_t times[] = {4096, 8192, 16384};
    const std::size_t dms[] = {256, 512, 1024};

    for (std::size_t t : times) {
        for (std::size_t dm : dms) {
            benchmark_configuration(t, dm, /*warmup*/ 1, /*bench*/ 2);
        }
        std::printf("-------------------------------------------------------------------------------------------------\n");
    }

    std::printf("=================================================================================================\n");
    return 0;
}
