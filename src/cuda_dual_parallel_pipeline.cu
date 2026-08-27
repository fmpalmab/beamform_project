// src/cuda_dual_parallel_pipeline.cu
//
// Implementation of DualEngineParallelPipeline: High-Performance Concurrent
// Fixed-Grid Beamformer (V3) and Multi-Beam Dynamic Tracker (V5).

#include "beamformer/cuda_dual_parallel_pipeline.hpp"
#include "beamformer/cuda_beamformer_v3.hpp"
#include "beamformer/cuda_beam_tracker_v5.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

float event_elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, end), "cudaEventElapsedTime");
    return ms;
}

} // namespace

struct DualEngineParallelPipeline::Impl {
    Dimensions bf_dims;
    Weights bf_weights;
    Dimensions tracker_dims;
    std::vector<TrackerTrajectoryConfig> tracker_trajectories;
    TemporalIntegrationConfig temporal_integration;
    DualParallelConfig config;

    // Concurrency streams
    cudaStream_t stream_upload = nullptr;
    cudaStream_t stream_bf = nullptr;
    cudaStream_t stream_tracker = nullptr;
    cudaStream_t stream_download = nullptr;

    // Synchronization events
    cudaEvent_t event_upload_start = nullptr;
    cudaEvent_t event_upload_ready = nullptr;
    cudaEvent_t event_gpu_start = nullptr;
    cudaEvent_t event_bf_start = nullptr;
    cudaEvent_t event_bf_done = nullptr;
    cudaEvent_t event_tracker_start = nullptr;
    cudaEvent_t event_tracker_done = nullptr;
    cudaEvent_t event_gpu_end = nullptr;
    cudaEvent_t event_download_done = nullptr;

    // Shared & Dedicated Device Memory
    std::uint8_t* d_shared_packed = nullptr;
    ComplexFloat* d_bf_weights = nullptr;
    float* d_bf_output = nullptr;
    float* d_tracker_output = nullptr;

    std::size_t active_tracker_beams = 1;

    // CUDA Graph state
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;

    Impl(const Dimensions& bfd, const Weights& bfw,
         const Dimensions& trd, const std::vector<TrackerTrajectoryConfig>& trajs,
         const TemporalIntegrationConfig& tint, const DualParallelConfig& cfg)
        : bf_dims(bfd), bf_weights(bfw), tracker_dims(trd),
          tracker_trajectories(trajs), temporal_integration(tint), config(cfg) {

        if (bf_dims.n_time != tracker_dims.n_time ||
            bf_dims.n_freq != tracker_dims.n_freq ||
            bf_dims.n_ant != tracker_dims.n_ant) {
            throw std::invalid_argument("Beamformer and Tracker dimensions (time, freq, ant) must match for shared input");
        }

        active_tracker_beams = std::min(trajs.size(), static_cast<std::size_t>(MAX_TRACKER_BEAMS));

        // Create independent non-blocking streams
        check_cuda(cudaStreamCreateWithFlags(&stream_upload, cudaStreamNonBlocking), "stream_upload");
        check_cuda(cudaStreamCreateWithFlags(&stream_bf, cudaStreamNonBlocking), "stream_bf");
        check_cuda(cudaStreamCreateWithFlags(&stream_tracker, cudaStreamNonBlocking), "stream_tracker");
        check_cuda(cudaStreamCreateWithFlags(&stream_download, cudaStreamNonBlocking), "stream_download");

        // Create events
        check_cuda(cudaEventCreate(&event_upload_start), "event_upload_start");
        check_cuda(cudaEventCreate(&event_upload_ready), "event_upload_ready");
        check_cuda(cudaEventCreate(&event_gpu_start), "event_gpu_start");
        check_cuda(cudaEventCreate(&event_bf_start), "event_bf_start");
        check_cuda(cudaEventCreate(&event_bf_done), "event_bf_done");
        check_cuda(cudaEventCreate(&event_tracker_start), "event_tracker_start");
        check_cuda(cudaEventCreate(&event_tracker_done), "event_tracker_done");
        check_cuda(cudaEventCreate(&event_gpu_end), "event_gpu_end");
        check_cuda(cudaEventCreate(&event_download_done), "event_download_done");

        // Shared device packed buffer (allocated once for both engines!)
        const std::size_t shared_bytes = voltage_sample_count(bf_dims) * sizeof(std::uint8_t);
        const std::size_t bf_w_bytes = bf_dims.n_beams * bf_dims.n_freq * bf_dims.n_ant * sizeof(ComplexFloat);
        const std::size_t bf_out_bytes = integrated_intensity_count(bf_dims, temporal_integration) * sizeof(float);
        const std::size_t tr_out_bytes = tracker_dims.n_time * tracker_dims.n_freq * tracker_dims.n_beams * sizeof(float);

        check_cuda(cudaMalloc(&d_shared_packed, shared_bytes), "cudaMalloc d_shared_packed");
        check_cuda(cudaMalloc(&d_bf_weights, bf_w_bytes), "cudaMalloc d_bf_weights");
        check_cuda(cudaMalloc(&d_bf_output, bf_out_bytes), "cudaMalloc d_bf_output");
        check_cuda(cudaMalloc(&d_tracker_output, tr_out_bytes), "cudaMalloc d_tracker_output");

        // Upload static beamformer weights
        check_cuda(cudaMemcpyAsync(d_bf_weights, bf_weights.data(), bf_w_bytes,
                                   cudaMemcpyHostToDevice, stream_upload), "H2D bf_weights");
        check_cuda(cudaStreamSynchronize(stream_upload), "cudaStreamSynchronize init");
    }

    ~Impl() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (d_shared_packed) cudaFree(d_shared_packed);
        if (d_bf_weights) cudaFree(d_bf_weights);
        if (d_bf_output) cudaFree(d_bf_output);
        if (d_tracker_output) cudaFree(d_tracker_output);

        if (event_upload_start) cudaEventDestroy(event_upload_start);
        if (event_upload_ready) cudaEventDestroy(event_upload_ready);
        if (event_gpu_start) cudaEventDestroy(event_gpu_start);
        if (event_bf_start) cudaEventDestroy(event_bf_start);
        if (event_bf_done) cudaEventDestroy(event_bf_done);
        if (event_tracker_start) cudaEventDestroy(event_tracker_start);
        if (event_tracker_done) cudaEventDestroy(event_tracker_done);
        if (event_gpu_end) cudaEventDestroy(event_gpu_end);
        if (event_download_done) cudaEventDestroy(event_download_done);

        if (stream_upload) cudaStreamDestroy(stream_upload);
        if (stream_bf) cudaStreamDestroy(stream_bf);
        if (stream_tracker) cudaStreamDestroy(stream_tracker);
        if (stream_download) cudaStreamDestroy(stream_download);
    }
};

DualEngineParallelPipeline::DualEngineParallelPipeline(
    const Dimensions& bf_dims,
    const Weights& bf_weights,
    const Dimensions& tracker_dims,
    const std::vector<TrackerTrajectoryConfig>& tracker_trajectories,
    const TemporalIntegrationConfig& temporal_integration,
    const DualParallelConfig& config)
    : impl_(std::make_unique<Impl>(
          bf_dims, bf_weights, tracker_dims, tracker_trajectories, temporal_integration, config)) {}

DualEngineParallelPipeline::~DualEngineParallelPipeline() = default;

std::size_t DualEngineParallelPipeline::shared_input_bytes() const {
    return voltage_sample_count(impl_->bf_dims) * sizeof(std::uint8_t);
}

std::size_t DualEngineParallelPipeline::memory_saved_bytes() const {
    // Memory saved is the entire duplicate input buffer that we avoided allocating
    return shared_input_bytes();
}

const Dimensions& DualEngineParallelPipeline::beamformer_dimensions() const {
    return impl_->bf_dims;
}

const Dimensions& DualEngineParallelPipeline::tracker_dimensions() const {
    return impl_->tracker_dims;
}

std::uint8_t* DualEngineParallelPipeline::device_shared_packed_voltage() {
    return impl_->d_shared_packed;
}

float* DualEngineParallelPipeline::device_beamformer_output() {
    return impl_->d_bf_output;
}

float* DualEngineParallelPipeline::device_tracker_output() {
    return impl_->d_tracker_output;
}

void DualEngineParallelPipeline::update_tracker_trajectory(
    std::size_t beam_index, const TrackerTrajectoryConfig& new_traj) {
    if (beam_index < impl_->tracker_trajectories.size()) {
        impl_->tracker_trajectories[beam_index] = new_traj;
        if (impl_->graph_captured) {
            cudaGraphExecDestroy(impl_->graph_exec);
            cudaGraphDestroy(impl_->graph);
            impl_->graph_exec = nullptr;
            impl_->graph = nullptr;
            impl_->graph_captured = false;
        }
    }
}

void DualEngineParallelPipeline::set_tracker_active_beams(std::size_t active_count) {
    impl_->active_tracker_beams = std::min(
        active_count, static_cast<std::size_t>(MAX_TRACKER_BEAMS));
}

DualParallelTimings DualEngineParallelPipeline::process_frame(
    const std::uint8_t* host_packed_voltage,
    float* host_bf_integrated_output,
    float* host_tracker_integrated_output) {

    const auto t_wall_start = Clock::now();
    const std::size_t shared_bytes = shared_input_bytes();
    const std::size_t bf_out_bytes = integrated_intensity_count(impl_->bf_dims, impl_->temporal_integration) * sizeof(float);
    const std::size_t tr_out_bytes = impl_->tracker_dims.n_time * impl_->tracker_dims.n_freq * impl_->tracker_dims.n_beams * sizeof(float);

    // 1. Single PCIe H2D Upload
    check_cuda(cudaEventRecord(impl_->event_upload_start, impl_->stream_upload), "event_upload_start");
    check_cuda(cudaMemcpyAsync(impl_->d_shared_packed, host_packed_voltage, shared_bytes,
                               cudaMemcpyHostToDevice, impl_->stream_upload), "H2D shared packed");
    check_cuda(cudaEventRecord(impl_->event_upload_ready, impl_->stream_upload), "event_upload_ready");

    // 2. Fork into Parallel Execution Streams
    check_cuda(cudaStreamWaitEvent(impl_->stream_bf, impl_->event_upload_ready, 0), "wait event upload on bf stream");
    check_cuda(cudaStreamWaitEvent(impl_->stream_tracker, impl_->event_upload_ready, 0), "wait event upload on tr stream");

    check_cuda(cudaEventRecord(impl_->event_gpu_start, impl_->stream_bf), "event_gpu_start");
    check_cuda(cudaEventRecord(impl_->event_bf_start, impl_->stream_bf), "event_bf_start");
    check_cuda(cudaEventRecord(impl_->event_tracker_start, impl_->stream_tracker), "event_tracker_start");

    // Launch Branch A: Beamformer V3 Fused Integrated Kernel
    cuda_beamform_v3_integrated_device_resident(
        impl_->d_shared_packed,
        impl_->d_bf_weights,
        impl_->d_bf_output,
        impl_->bf_dims,
        impl_->temporal_integration,
        impl_->config.beamformer_config,
        impl_->stream_bf);
    check_cuda(cudaEventRecord(impl_->event_bf_done, impl_->stream_bf), "event_bf_done");

    // Launch Branch B: Beam Tracker V5 Multi-Beam Kernel
    TrackerConfig tr_cfg;
    if (!impl_->tracker_trajectories.empty()) {
        tr_cfg.trajectory = impl_->tracker_trajectories[0];
    }
    tr_cfg.integration_spectra = impl_->temporal_integration.integration_spectra;

    cuda_beam_tracker_v5_device_resident(
        impl_->d_shared_packed,
        impl_->d_tracker_output,
        impl_->tracker_dims,
        tr_cfg,
        impl_->config.tracker_config,
        impl_->stream_tracker);
    check_cuda(cudaEventRecord(impl_->event_tracker_done, impl_->stream_tracker), "event_tracker_done");

    // 3. Join on Download Stream
    check_cuda(cudaStreamWaitEvent(impl_->stream_download, impl_->event_bf_done, 0), "wait bf_done");
    check_cuda(cudaStreamWaitEvent(impl_->stream_download, impl_->event_tracker_done, 0), "wait tracker_done");
    check_cuda(cudaEventRecord(impl_->event_gpu_end, impl_->stream_download), "event_gpu_end");

    // Issue asynchronous D2H transfers
    if (host_bf_integrated_output != nullptr) {
        check_cuda(cudaMemcpyAsync(host_bf_integrated_output, impl_->d_bf_output, bf_out_bytes,
                                   cudaMemcpyDeviceToHost, impl_->stream_download), "D2H bf");
    }
    if (host_tracker_integrated_output != nullptr) {
        check_cuda(cudaMemcpyAsync(host_tracker_integrated_output, impl_->d_tracker_output, tr_out_bytes,
                                   cudaMemcpyDeviceToHost, impl_->stream_download), "D2H tracker");
    }
    check_cuda(cudaEventRecord(impl_->event_download_done, impl_->stream_download), "event_download_done");

    check_cuda(cudaStreamSynchronize(impl_->stream_download), "stream_download sync");

    const auto t_wall_end = Clock::now();

    DualParallelTimings timings;
    timings.host_to_device_ms = event_elapsed_ms(impl_->event_upload_start, impl_->event_upload_ready);
    timings.beamformer_kernel_ms = event_elapsed_ms(impl_->event_bf_start, impl_->event_bf_done);
    timings.tracker_kernel_ms = event_elapsed_ms(impl_->event_tracker_start, impl_->event_tracker_done);
    timings.concurrent_gpu_ms = event_elapsed_ms(impl_->event_gpu_start, impl_->event_gpu_end);
    timings.sequential_sum_ms = timings.beamformer_kernel_ms + timings.tracker_kernel_ms;
    timings.device_to_host_ms = event_elapsed_ms(impl_->event_gpu_end, impl_->event_download_done);
    timings.total_wall_ms = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();
    timings.parallel_efficiency = (timings.concurrent_gpu_ms > 0.0)
                                      ? (timings.sequential_sum_ms / timings.concurrent_gpu_ms)
                                      : 1.0;
    return timings;
}

DualParallelTimings DualEngineParallelPipeline::process_device_resident(
    const std::uint8_t* d_packed_voltage,
    float* d_bf_integrated_output,
    float* d_tracker_integrated_output) {

    const auto t_wall_start = Clock::now();

    check_cuda(cudaEventRecord(impl_->event_gpu_start, impl_->stream_bf), "event_gpu_start");
    check_cuda(cudaEventRecord(impl_->event_bf_start, impl_->stream_bf), "event_bf_start");
    check_cuda(cudaEventRecord(impl_->event_tracker_start, impl_->stream_tracker), "event_tracker_start");

    // Launch Branch A
    cuda_beamform_v3_integrated_device_resident(
        d_packed_voltage,
        impl_->d_bf_weights,
        d_bf_integrated_output,
        impl_->bf_dims,
        impl_->temporal_integration,
        impl_->config.beamformer_config,
        impl_->stream_bf);
    check_cuda(cudaEventRecord(impl_->event_bf_done, impl_->stream_bf), "event_bf_done");

    // Launch Branch B
    TrackerConfig tr_cfg;
    if (!impl_->tracker_trajectories.empty()) {
        tr_cfg.trajectory = impl_->tracker_trajectories[0];
    }
    tr_cfg.integration_spectra = impl_->temporal_integration.integration_spectra;

    cuda_beam_tracker_v5_device_resident(
        d_packed_voltage,
        d_tracker_integrated_output,
        impl_->tracker_dims,
        tr_cfg,
        impl_->config.tracker_config,
        impl_->stream_tracker);
    check_cuda(cudaEventRecord(impl_->event_tracker_done, impl_->stream_tracker), "event_tracker_done");

    // Join
    check_cuda(cudaStreamWaitEvent(impl_->stream_download, impl_->event_bf_done, 0), "wait bf_done");
    check_cuda(cudaStreamWaitEvent(impl_->stream_download, impl_->event_tracker_done, 0), "wait tracker_done");
    check_cuda(cudaEventRecord(impl_->event_gpu_end, impl_->stream_download), "event_gpu_end");
    check_cuda(cudaStreamSynchronize(impl_->stream_download), "sync");

    const auto t_wall_end = Clock::now();

    DualParallelTimings timings;
    timings.host_to_device_ms = 0.0;
    timings.beamformer_kernel_ms = event_elapsed_ms(impl_->event_bf_start, impl_->event_bf_done);
    timings.tracker_kernel_ms = event_elapsed_ms(impl_->event_tracker_start, impl_->event_tracker_done);
    timings.concurrent_gpu_ms = event_elapsed_ms(impl_->event_gpu_start, impl_->event_gpu_end);
    timings.sequential_sum_ms = timings.beamformer_kernel_ms + timings.tracker_kernel_ms;
    timings.device_to_host_ms = 0.0;
    timings.total_wall_ms = std::chrono::duration<double, std::milli>(t_wall_end - t_wall_start).count();
    timings.parallel_efficiency = (timings.concurrent_gpu_ms > 0.0)
                                      ? (timings.sequential_sum_ms / timings.concurrent_gpu_ms)
                                      : 1.0;
    return timings;
}

} // namespace beamformer
