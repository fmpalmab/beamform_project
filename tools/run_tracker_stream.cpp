// tools/run_tracker_stream.cpp
//
// CLI bridge to execute any CPU or GPU beam tracker implementation on
// packed voltage streams produced by the Python astronomical signal injector,
// and write out the resulting dynamic spectrum float32 intensities.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/beam_tracker_opt.hpp"
#include "beamformer/beam_tracker_opt_v2.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#if BEAMFORMER_HAS_CUDA
#include "beamformer/cuda_beam_tracker_fused_warp_shuffle.hpp"
#include "beamformer/cuda_tracker_v2.hpp"
#endif

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using beamformer::Dimensions;
using beamformer::Intensities;
using beamformer::PackedVoltage;
using beamformer::TrackerConfig;
using beamformer::Vec3;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --engine ENGINE       cpu_naive, cpu_v1, cpu_v2, cuda_twopass, cuda_fused, cuda_warp, cuda_fws\n"
              << "  --n-time N            number of time samples\n"
              << "  --n-freq N            number of frequency channels (default 336)\n"
              << "  --n-ant N             number of antenna elements (32 or 64)\n"
              << "  --spectra N           integration spectra per window (default 320)\n"
              << "  --source-l0 L0        steering start l0\n"
              << "  --source-m0 M0        steering start m0\n"
              << "  --source-dl DL        steering rate dl/sample\n"
              << "  --source-dm DM        steering rate dm/sample\n"
              << "  --input FILE          input packed voltage binary file\n"
              << "  --output FILE         output float32 intensity binary file\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string engine = "cpu_v2";
    std::size_t n_time = 15360;
    std::size_t n_freq = beamformer::default_frequency_channels;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    float l0 = 0.0F, m0 = 0.0F, dl = 0.0F, dm = 0.0F;
    std::string input_file;
    std::string output_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) engine = argv[++i];
        else if (arg == "--n-time" && i + 1 < argc) n_time = std::stoull(argv[++i]);
        else if (arg == "--n-freq" && i + 1 < argc) n_freq = std::stoull(argv[++i]);
        else if (arg == "--n-ant" && i + 1 < argc) n_ant = std::stoull(argv[++i]);
        else if (arg == "--spectra" && i + 1 < argc) integration_spectra = std::stoull(argv[++i]);
        else if (arg == "--source-l0" && i + 1 < argc) l0 = std::stof(argv[++i]);
        else if (arg == "--source-m0" && i + 1 < argc) m0 = std::stof(argv[++i]);
        else if (arg == "--source-dl" && i + 1 < argc) dl = std::stof(argv[++i]);
        else if (arg == "--source-dm" && i + 1 < argc) dm = std::stof(argv[++i]);
        else if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (input_file.empty() || output_file.empty()) {
        std::cerr << "Error: --input and --output are required.\n";
        print_usage(argv[0]);
        return 1;
    }

    Dimensions dims{n_time, n_freq, n_ant, beamformer::tracker_beam_count};
    TrackerConfig tracker_cfg;
    tracker_cfg.trajectory.direction_start = beamformer::direction_from_lm(l0, m0);
    tracker_cfg.trajectory.direction_rate_per_sample = {dl, dm};
    tracker_cfg.integration_spectra = integration_spectra;

    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Cannot open input file: " << input_file << "\n";
        return 1;
    }

    in.seekg(0, std::ios::end);
    std::size_t file_bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    std::size_t expected_bytes = n_time * n_freq * n_ant;
    if (file_bytes != expected_bytes) {
        std::cerr << "Error: Input file size (" << file_bytes 
                  << ") does not match expected (" << expected_bytes << ")\n";
        return 1;
    }

    PackedVoltage packed(expected_bytes);
    in.read(reinterpret_cast<char*>(packed.data()), expected_bytes);
    in.close();

    std::size_t out_floats = n_time * n_freq * dims.n_beams;
    Intensities intensity(out_floats, 0.0F);

    if (engine == "cpu_naive") {
        beamformer::beam_tracker_cpu_packed_intensity_into(packed, dims, tracker_cfg, intensity);
    } else if (engine == "cpu_v1") {
        beamformer::beam_tracker_opt_cpu_packed_intensity_into(packed, dims, tracker_cfg, intensity);
    } else if (engine == "cpu_v2") {
        beamformer::beam_tracker_opt_v2_cpu_packed_intensity_into(packed, dims, tracker_cfg, intensity);
    }
#if BEAMFORMER_HAS_CUDA
    else if (engine == "cuda_twopass") {
        beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, intensity, beamformer::CudaTrackerKernelV2::TwoPass);
    } else if (engine == "cuda_fused") {
        beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, intensity, beamformer::CudaTrackerKernelV2::Fused);
    } else if (engine == "cuda_warp") {
        beamformer::cuda_tracker_v2_packed_intensity_into(packed, dims, tracker_cfg, intensity, beamformer::CudaTrackerKernelV2::WarpReduction);
    } else if (engine == "cuda_fws") {
        beamformer::cuda_beam_tracker_fused_warp_shuffle_stream(packed, dims, tracker_cfg, intensity, 3);
    }
#endif
    else {
        std::cerr << "Error: Unknown engine or CUDA unavailable: " << engine << "\n";
        return 1;
    }

    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Cannot open output file: " << output_file << "\n";
        return 1;
    }

    out.write(reinterpret_cast<const char*>(intensity.data()), intensity.size() * sizeof(float));
    out.close();

    std::cout << "Successfully ran " << engine << " on " << input_file 
              << " -> wrote " << output_file << " (" << intensity.size() * sizeof(float) << " bytes)\n";
    return 0;
}
