#pragma once

// CUDA Upchannelizer for Beam Tracker and Voltage Beamformer Outputs.
//
// In the CHARTS / CHIME radio astronomy signal chain, the coarse channelizer
// delivers 336 frequency channels of 300 kHz width at 3.333 us sample time.
// The upchannelizer takes the beamformed / tracked complex voltage stream
// and subdivides each coarse channel into M fine frequency channels (e.g. M=32, 16, 8, 4),
// performing an M-point windowed Polyphase/FFT channelization along the time axis:
//
// 1. Warp-Level Register FFT (Zero Shared Memory):
//    For M = 32 (the standard CHARTS factor), each warp (32 threads) processes
//    one (time_frame, coarse_channel) block. The 32 time samples are channelized
//    in registers across 5 radix-2 butterfly stages using `__shfl_xor_sync`.
//
// 2. Coalesced Global Memory Transactions:
//    Lane k writes fine frequency channel `f_coarse * M + k` in a single
//    128-byte aligned, perfectly coalesced global memory write.
//
// 3. Fused Temporal Integration Post-Upchannelization:
//    Directly integrates S_post fine-time spectra (e.g. 10 spectra) into
//    integrated fine filterbank profiles [windows][n_freq * M][n_beams].
//
// 4. Fused Tracker + Upchannelizer:
//    Directly takes packed RFSoC voltage and time-varying trajectory,
//    beamforms across antennas, upchannelizes in registers, and outputs
//    fine spectral intensities in a single unified GPU kernel.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/temporal_integration.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace beamformer {

enum class UpchannelizerWindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
};

struct UpchannelizerConfig {
    std::size_t upchan_factor = 32; // M: typically 32, 16, 8, or 4
    UpchannelizerWindowType window = UpchannelizerWindowType::Hann;
    std::size_t post_integration_spectra = 10; // 0 or 1 for no post-integration
};

struct UpchannelizerDimensions {
    std::size_t n_time = 0;           // Input coarse time samples
    std::size_t n_freq = 0;           // Input coarse frequency channels
    std::size_t n_beams = 1;          // Beams (default 1 for tracked beam)
    std::size_t upchan_factor = 32;   // M

    std::size_t fine_time_count() const {
        return (upchan_factor > 0) ? (n_time / upchan_factor) : 0;
    }
    std::size_t fine_freq_count() const {
        return n_freq * upchan_factor;
    }
    std::size_t fine_intensity_count() const {
        return fine_time_count() * fine_freq_count() * n_beams;
    }
    std::size_t integrated_time_count(std::size_t post_int) const {
        const std::size_t ft = fine_time_count();
        return (post_int > 0) ? (ft / post_int) : ft;
    }
    std::size_t integrated_intensity_count(std::size_t post_int) const {
        return integrated_time_count(post_int) * fine_freq_count() * n_beams;
    }
};

// ---------------------------------------------------------------------------
// CPU Reference Implementations
// ---------------------------------------------------------------------------

// Upchannelize complex voltage stream [time][freq][beam] -> fine intensities [fine_time][fine_freq][beam]
Intensities cpu_upchannelize_voltage(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config = UpchannelizerConfig{});

void cpu_upchannelize_voltage_into(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config,
    Intensities& fine_intensity);

// End-to-end CPU Reference: Packed Voltage + Tracker -> Upchannelized Fine Intensities
Intensities cpu_tracker_upchannelize(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg = UpchannelizerConfig{});

// ---------------------------------------------------------------------------
// CUDA Device Kernels & Functional APIs
// ---------------------------------------------------------------------------

// Standalone CUDA Upchannelizer for complex voltage stream
Intensities cuda_upchannelize_voltage(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config = UpchannelizerConfig{});

void cuda_upchannelize_voltage_into(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config,
    Intensities& fine_intensity);

// Fused Post-Integrated CUDA Upchannelizer
Intensities cuda_upchannelize_voltage_integrated(
    const ComplexVoltage& voltage,
    const Dimensions& dims,
    const UpchannelizerConfig& config = UpchannelizerConfig{});

// Device-Resident Upchannelizer (d_voltage [n_time][n_freq][n_beams] -> d_fine_intensity [n_fine_time][n_fine_freq][n_beams])
void cuda_upchannelize_voltage_device_resident(
    const ComplexFloat* d_voltage,
    float* d_fine_intensity,
    const Dimensions& dims,
    const UpchannelizerConfig& config = UpchannelizerConfig{},
    void* stream = nullptr);

// Fused CUDA Tracker + Upchannelizer (d_packed -> d_fine_intensity in 1 kernel pass)
void cuda_tracker_upchannelize_device_resident(
    const std::uint8_t* d_packed,
    float* d_fine_intensity,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg = UpchannelizerConfig{},
    void* stream = nullptr);

Intensities cuda_tracker_upchannelize(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg = UpchannelizerConfig{});

void cuda_tracker_upchannelize_into(
    const PackedVoltage& packed,
    const Dimensions& dims,
    const TrackerConfig& tracker,
    const UpchannelizerConfig& upchan_cfg,
    Intensities& fine_intensity);

// ---------------------------------------------------------------------------
// Reusable CUDA Upchannelizer Workspace
// ---------------------------------------------------------------------------
class CudaUpchannelizerWorkspace {
public:
    explicit CudaUpchannelizerWorkspace(
        const Dimensions& capacity,
        const UpchannelizerConfig& config = UpchannelizerConfig{});
    ~CudaUpchannelizerWorkspace();

    CudaUpchannelizerWorkspace(const CudaUpchannelizerWorkspace&) = delete;
    CudaUpchannelizerWorkspace& operator=(const CudaUpchannelizerWorkspace&) = delete;

    double setup_ms() const;
    const UpchannelizerConfig& config() const;
    const UpchannelizerDimensions& dimensions() const;

    void process(const ComplexVoltage& voltage, Intensities& fine_intensity);
    void process_tracker(const PackedVoltage& packed, const TrackerConfig& tracker, Intensities& fine_intensity);

    ComplexFloat* device_voltage();
    std::uint8_t* device_packed();
    float* device_fine_intensity();
    void* device_stream();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
