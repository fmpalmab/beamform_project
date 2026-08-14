// Benchmark for the optimized CPU beam tracker.
//
// Builds a synthetic packed-int4 shard with a slowly moving point source, runs
// both the naive (`beam_tracker_cpu_packed_intensity`) and the optimized
// (`cpu_opt_beam_tracker_packed_intensity`) trackers, and reports:
//
//   * end-to-end wall time, naive vs optimized, with a speedup ratio;
//   * per-window ("per-frame") latency for the optimized path, read from the
//     `BEAMFORMER_TRACKER_PERF`-guarded `Impl::window_ms` vector populated
//     inside `CpuOptBeamTracker::run_into`. The headline objective is
//     **per-frame kernel time < 0.5 ms**;
//   * summary statistics (min / mean / median / 95th / max) over the per-frame
//     samples, plus a pass/fail flag against the 0.5 ms target;
//   * a DOA-recovery check (estimated vs true source direction distance,
//     averaged across windows) so a too-fast but wrong kernel is flagged.
//
// Two CSV files are written:
//   * --metrics FILE   : one summary row per (config) run, plus the headline
//                        pass/fail against the 0.5 ms objective.
//   * --frames FILE    : one row per integration window with its latency,
//                        suitable for the per-frame plot.
//
// Designed to be built with `-DBEAMFORMER_TRACKER_PERF` so the per-frame
// timings are available. The CMake target adds this flag (see CMakeLists.txt).
// If the macro is undefined the per-frame vector is empty and the benchmark
// falls back to end-to-end-only reporting with a clear warning.

#include "beamformer/beam_tracker.hpp"
#include "beamformer/cpu_opt_beam_tracker.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_ant = 64;
    std::size_t integration_spectra = 320;
    // Optimized-search knobs.
    std::size_t coarse_grid_resolution = 12;
    std::size_t refinement_levels = 2;
    float search_fov_l = 0.2F;
    float search_fov_m = 0.2F;
    float forgetting_factor = 1.0F;
    float diagonal_load = 1.0e-3F;
    bool capon = false;
    std::size_t spatial_smoothing_subarray_size = 0;
    // Source / prior geometry.
    float source_l0 = 0.03F;
    float source_m0 = 0.0F;
    float source_dl = 1.0e-5F;  // slow drift within FoV
    float source_dm = 0.0F;
    float prior_l0 = 0.0F;
    float prior_m0 = 0.0F;
    // Warm-up + repeat to reduce noise.
    std::size_t warmup_runs = 1;
    std::size_t repeat = 1;
    // Output.
    std::optional<std::filesystem::path> metrics;
    std::optional<std::filesystem::path> frames;
    std::optional<std::filesystem::path> intensity_out;
    // Target.
    double target_ms_per_frame = 0.5;
};

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

std::size_t parse_size(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stoull(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

float parse_float(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stof(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid float for ") + option);
    }
    return parsed;
}

double parse_double(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stod(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid double for ") + option);
    }
    return parsed;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  Run geometry / timing benchmark for the optimized CPU tracker.\n\n"
        << "  Grid:\n"
        << "    --n-time N                 default 15360\n"
        << "    --n-ant N                  32 or 64; default 64\n"
        << "    --integration-spectra N    default 320\n"
        << "  Optimized search:\n"
        << "    --coarse-grid-resolution N default 12\n"
        << "    --refinement-levels N      default 2\n"
        << "    --search-fov-l F           default 0.2\n"
        << "    --search-fov-m F           default 0.2\n"
        << "    --forgetting-factor F      default 1.0\n"
        << "    --diagonal-load F          default 1e-3\n"
        << "    --capon                    use Capon/MVDR estimator\n"
        << "    --smoothing-subarray N     spatial smoothing sub-array size; 0 disables\n"
        << "  Source / prior:\n"
        << "    --source-l0 F --source-m0 F --source-dl F --source-dm F\n"
        << "    --prior-l0 F  --prior-m0 F\n"
        << "  Benchmark:\n"
        << "    --warmup-runs N            default 1\n"
        << "    --repeat N                 default 1 (median over repeats)\n"
        << "    --target-ms-per-frame F    default 0.5 (pass/fail threshold)\n"
        << "  Output:\n"
        << "    --metrics FILE             summary CSV (append; header if new)\n"
        << "    --frames FILE              per-frame latency CSV (per repeat)\n"
        << "    --intensity-out FILE       write the optimized intensity cube\n";
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--integration-spectra") {
            options.integration_spectra = parse_size(
                require_value(argc, argv, i), "--integration-spectra");
        } else if (argument == "--coarse-grid-resolution") {
            options.coarse_grid_resolution = parse_size(
                require_value(argc, argv, i), "--coarse-grid-resolution");
        } else if (argument == "--refinement-levels") {
            options.refinement_levels = parse_size(
                require_value(argc, argv, i), "--refinement-levels");
        } else if (argument == "--search-fov-l") {
            options.search_fov_l = parse_float(require_value(argc, argv, i),
                                               "--search-fov-l");
        } else if (argument == "--search-fov-m") {
            options.search_fov_m = parse_float(require_value(argc, argv, i),
                                               "--search-fov-m");
        } else if (argument == "--forgetting-factor") {
            options.forgetting_factor = parse_float(
                require_value(argc, argv, i), "--forgetting-factor");
        } else if (argument == "--diagonal-load") {
            options.diagonal_load = parse_float(
                require_value(argc, argv, i), "--diagonal-load");
        } else if (argument == "--capon") {
            options.capon = true;
        } else if (argument == "--smoothing-subarray") {
            options.spatial_smoothing_subarray_size = parse_size(
                require_value(argc, argv, i), "--smoothing-subarray");
        } else if (argument == "--source-l0") {
            options.source_l0 = parse_float(require_value(argc, argv, i),
                                            "--source-l0");
        } else if (argument == "--source-m0") {
            options.source_m0 = parse_float(require_value(argc, argv, i),
                                            "--source-m0");
        } else if (argument == "--source-dl") {
            options.source_dl = parse_float(require_value(argc, argv, i),
                                            "--source-dl");
        } else if (argument == "--source-dm") {
            options.source_dm = parse_float(require_value(argc, argv, i),
                                            "--source-dm");
        } else if (argument == "--prior-l0") {
            options.prior_l0 = parse_float(require_value(argc, argv, i),
                                           "--prior-l0");
        } else if (argument == "--prior-m0") {
            options.prior_m0 = parse_float(require_value(argc, argv, i),
                                           "--prior-m0");
        } else if (argument == "--warmup-runs") {
            options.warmup_runs = parse_size(
                require_value(argc, argv, i), "--warmup-runs");
        } else if (argument == "--repeat") {
            options.repeat = parse_size(require_value(argc, argv, i), "--repeat");
        } else if (argument == "--target-ms-per-frame") {
            options.target_ms_per_frame = parse_double(
                require_value(argc, argv, i), "--target-ms-per-frame");
        } else if (argument == "--metrics") {
            options.metrics = require_value(argc, argv, i);
        } else if (argument == "--frames") {
            options.frames = require_value(argc, argv, i);
        } else if (argument == "--intensity-out") {
            options.intensity_out = require_value(argc, argv, i);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

// Summary statistics of a sample vector (millisecond latencies).
struct LatencyStats {
    double min = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    std::size_t count = 0;
};

LatencyStats summarise(std::vector<double> samples) {
    LatencyStats s;
    s.count = samples.size();
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.min = samples.front();
    s.max = samples.back();
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0)
             / static_cast<double>(s.count);
    const std::size_t mid = s.count / 2;
    s.median = (s.count % 2 == 0)
                   ? 0.5 * (samples[mid - 1] + samples[mid])
                   : samples[mid];
    const std::size_t p95_idx = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(s.count)) - 1.0);
    s.p95 = samples[std::min(p95_idx, s.count - 1)];
    return s;
}

float lm_distance(const beamformer::Vec3& a, const beamformer::Vec3& b) {
    const float dl = a[0] - b[0];
    const float dm = a[1] - b[1];
    return std::sqrt(dl * dl + dm * dm);
}

void write_intensity_cube(const std::filesystem::path& path,
                          const beamformer::Intensities& intensity) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open intensity output: " + path.string());
    out.write(reinterpret_cast<const char*>(intensity.data()),
              static_cast<std::streamsize>(intensity.size() * sizeof(float)));
    if (!out) throw std::runtime_error("failed writing intensity: " + path.string());
}

void append_metrics(const std::filesystem::path& path,
                    const Options& opt, const beamformer::Dimensions& dims,
                    double naive_ms, double opt_ms,
                    const LatencyStats& ls,
                    double doa_err_prior_mean, double doa_err_est_mean,
                    bool target_pass) {
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    const auto file_size = exists ? std::filesystem::file_size(path, ec) : 0;
    const bool needs_header = !exists || file_size == 0;
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open metrics file: " + path.string());
    if (needs_header) {
        out << "backend,n_time,n_freq,n_ant,integration_spectra,"
               "coarse_grid_resolution,refinement_levels,estimator,"
               "forgetting_factor,smoothing_subarray,repeat,"
               "naive_ms,opt_ms,speedup,"
               "frame_min_ms,frame_mean_ms,frame_median_ms,frame_p95_ms,"
               "frame_max_ms,frame_count,"
               "doa_err_prior_mean,doa_err_est_mean,target_ms,target_pass\n";
    }
    const double speedup = (opt_ms > 0.0) ? naive_ms / opt_ms : 0.0;
    out << std::fixed << std::setprecision(6)
        << "cpu_opt_beam_tracker," << dims.n_time << ',' << dims.n_freq << ','
        << dims.n_ant << ',' << opt.integration_spectra << ','
        << opt.coarse_grid_resolution << ',' << opt.refinement_levels << ','
        << (opt.capon ? "capon" : "bartlett") << ','
        << opt.forgetting_factor << ','
        << opt.spatial_smoothing_subarray_size << ','
        << opt.repeat << ','
        << naive_ms << ',' << opt_ms << ',' << speedup << ','
        << ls.min << ',' << ls.mean << ',' << ls.median << ',' << ls.p95 << ','
        << ls.max << ',' << ls.count << ','
        << doa_err_prior_mean << ',' << doa_err_est_mean << ','
        << opt.target_ms_per_frame << ',' << (target_pass ? 1 : 0) << '\n';
    if (!out) throw std::runtime_error("failed writing metrics: " + path.string());
}

void write_frames_csv(const std::filesystem::path& path, const Options& opt,
                      const std::vector<double>& frame_ms) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open frames file: " + path.string());
    out << "window,ms,n_time,n_ant,integration_spectra,coarse_grid_resolution,"
           "refinement_levels,estimator\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < frame_ms.size(); ++i) {
        out << i << ',' << frame_ms[i] << ',' << opt.n_time << ',' << opt.n_ant
            << ',' << opt.integration_spectra << ','
            << opt.coarse_grid_resolution << ',' << opt.refinement_levels
            << ',' << (opt.capon ? "capon" : "bartlett") << '\n';
    }
    if (!out) throw std::runtime_error("failed writing frames: " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_options(argc, argv);

        const beamformer::Dimensions dims{
            opt.n_time,
            beamformer::default_frequency_channels,
            opt.n_ant,
            beamformer::tracker_beam_count,
        };
        beamformer::validate_dimensions(dims);

        const auto positions = beamformer::default_positions(dims.n_ant);
        const auto frequencies = beamformer::channelized_frequencies(dims.n_freq);

        // Prior trajectory (open-loop guess supplied to both paths).
        beamformer::TrackerConfig traj;
        traj.trajectory.direction_start =
            beamformer::direction_from_lm(opt.prior_l0, opt.prior_m0);
        traj.trajectory.direction_rate_per_sample = {0.0F, 0.0F};
        traj.integration_spectra = opt.integration_spectra;

        // True source trajectory (slowly moving point source within the FoV).
        beamformer::TrackerTrajectoryConfig source_traj{
            beamformer::direction_from_lm(opt.source_l0, opt.source_m0),
            {opt.source_dl, opt.source_dm}};

        // Build the synthetic voltage shard ONCE; both paths consume it.
        const beamformer::PackedVoltage packed =
            beamformer::beam_tracker_make_moving_point_source(
                dims, positions, frequencies, source_traj, 4.0F);

        // Naive tracker (baseline timing reference).
        double naive_ms_best = std::numeric_limits<double>::infinity();
        beamformer::Intensities naive_intensity;
        for (std::size_t run = 0; run < opt.warmup_runs + opt.repeat; ++run) {
            const auto t0 = Clock::now();
            naive_intensity =
                beamformer::beam_tracker_cpu_packed_intensity(packed, dims, traj);
            const auto t1 = Clock::now();
            if (run >= opt.warmup_runs) {
                naive_ms_best = std::min(naive_ms_best, elapsed_ms(t0, t1));
            }
        }
        if (opt.repeat == 0) naive_ms_best = 0.0;

        // Optimized tracker config (scan enabled unless caller disabled it).
        beamformer::CpuOptTrackerConfig ocfg;
        ocfg.coarse_grid_resolution = opt.coarse_grid_resolution;
        ocfg.refinement_levels = opt.refinement_levels;
        ocfg.search_fov_l = opt.search_fov_l;
        ocfg.search_fov_m = opt.search_fov_m;
        ocfg.forgetting_factor = opt.forgetting_factor;
        ocfg.diagonal_load = opt.diagonal_load;
        ocfg.estimator = opt.capon ? beamformer::TrackerEstimator::Capon
                                   : beamformer::TrackerEstimator::Bartlett;
        ocfg.spatial_smoothing_subarray_size =
            opt.spatial_smoothing_subarray_size;
        ocfg.integration_spectra = opt.integration_spectra;

        // Warm-up runs (discarded), then measurement runs. We keep per-frame
        // timings for the *last* measurement run (representative steady state).
        double opt_ms_best = std::numeric_limits<double>::infinity();
        beamformer::Intensities opt_intensity;
        std::vector<double> frame_ms;
        std::vector<beamformer::Vec3> estimated_dirs;
        for (std::size_t run = 0; run < opt.warmup_runs + opt.repeat; ++run) {
            beamformer::CpuOptBeamTracker tracker(positions, frequencies, dims,
                                                   ocfg);
            opt_intensity.assign(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
            tracker.seed_trajectory(traj.trajectory);
            const auto t0 = Clock::now();
            tracker.run_into(packed, opt_intensity);
            const auto t1 = Clock::now();
            if (run >= opt.warmup_runs) {
                opt_ms_best = std::min(opt_ms_best, elapsed_ms(t0, t1));
            }
            // Keep the last run's per-frame timings and estimated directions.
            if (run + 1 == opt.warmup_runs + opt.repeat) {
                // Public accessor: populated only when the perf macro compiled
                // the per-frame hook in (benchmark target always does); empty
                // otherwise (zero-overhead production builds).
                frame_ms = tracker.per_frame_ms();
                const std::size_t W = beamformer::tracker_window_count(
                    dims.n_time, ocfg.integration_spectra);
                estimated_dirs.clear();
                for (std::size_t w = 0; w < W; ++w) {
                    estimated_dirs.push_back(tracker.window_direction(w));
                }
            }
        }

        // DOA recovery: average distance between the estimated window direction
        // and the true source direction at the window's first sample, vs the
        // same distance for the (open-loop) prior.
        const std::size_t W = beamformer::tracker_window_count(
            dims.n_time, opt.integration_spectra);
        double doa_err_prior_sum = 0.0, doa_err_est_sum = 0.0;
        for (std::size_t w = 0; w < W; ++w) {
            const beamformer::Vec3 true_dir =
                beamformer::tracker_direction(source_traj,
                                              w * opt.integration_spectra);
            const beamformer::Vec3 prior_dir = beamformer::tracker_window_direction(
                traj.trajectory, w, opt.integration_spectra);
            const beamformer::Vec3 est_dir =
                (w < estimated_dirs.size()) ? estimated_dirs[w]
                                            : beamformer::Vec3{0, 0, 1};
            doa_err_prior_sum += lm_distance(true_dir, prior_dir);
            doa_err_est_sum += lm_distance(true_dir, est_dir);
        }
        const double doa_err_prior_mean = doa_err_prior_sum / static_cast<double>(W);
        const double doa_err_est_mean = doa_err_est_sum / static_cast<double>(W);

        const LatencyStats ls = summarise(frame_ms);
        const bool perf_available = !frame_ms.empty();
        const double worst = perf_available ? std::max(ls.p95, ls.max) : 0.0;
        const bool target_pass = perf_available && worst <= opt.target_ms_per_frame;
        const double speedup = (opt_ms_best > 0.0) ? naive_ms_best / opt_ms_best : 0.0;

        // ----- stdout report -----
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== CPU optimized beam tracker benchmark ===\n";
        std::cout << "grid: n_time=" << dims.n_time << " n_freq=" << dims.n_freq
                  << " n_ant=" << dims.n_ant
                  << " integration_spectra=" << opt.integration_spectra
                  << "  windows=" << W << '\n';
        std::cout << "search: coarse=" << opt.coarse_grid_resolution
                  << " refine=" << opt.refinement_levels
                  << " fov=(" << opt.search_fov_l << ',' << opt.search_fov_m
                  << ") estimator=" << (opt.capon ? "capon" : "bartlett")
                  << " lambda=" << opt.forgetting_factor
                  << " smoothing=" << opt.spatial_smoothing_subarray_size << '\n';
        std::cout << "naive  end-to-end: " << naive_ms_best << " ms\n";
        std::cout << "opt    end-to-end: " << opt_ms_best << " ms"
                  << "  (speedup " << speedup << "x vs naive)\n";
        if (perf_available) {
            std::cout << "per-frame kernel latencies (ms):\n";
            std::cout << "  min    = " << ls.min << '\n';
            std::cout << "  mean   = " << ls.mean << '\n';
            std::cout << "  median = " << ls.median << '\n';
            std::cout << "  p95    = " << ls.p95 << '\n';
            std::cout << "  max    = " << ls.max << '\n';
            std::cout << "  frames measured = " << ls.count << '\n';
            std::cout << "TARGET per-frame <= " << opt.target_ms_per_frame
                      << " ms : "
                      << (target_pass ? "PASS" : "FAIL")
                      << " (worst of p95/max = " << worst << " ms)\n";
        } else {
            std::cout << "WARNING: per-frame timings unavailable (build with "
                         "-DBEAMFORMER_TRACKER_PERF to enable).\n";
            std::cout << "TARGET per-frame <= " << opt.target_ms_per_frame
                      << " ms : SKIPPED (no per-frame data)\n";
        }
        std::cout << "DOA recovery (mean (l,m) error over windows):\n";
        std::cout << "  prior (open-loop) = " << doa_err_prior_mean << '\n';
        std::cout << "  optimized estimate= " << doa_err_est_mean << '\n';
        std::cout << "  improvement       = "
                  << (doa_err_est_mean < doa_err_prior_mean ? "yes" : "no")
                  << '\n';

        // ----- CSV outputs -----
        if (opt.metrics) {
            std::filesystem::create_directories(opt.metrics->parent_path());
            append_metrics(*opt.metrics, opt, dims, naive_ms_best, opt_ms_best,
                           ls, doa_err_prior_mean, doa_err_est_mean,
                           target_pass);
            std::cout << "Wrote metrics CSV to " << *opt.metrics << '\n';
        }
        if (opt.frames) {
            write_frames_csv(*opt.frames, opt, frame_ms);
            std::cout << "Wrote per-frame CSV (" << frame_ms.size()
                      << " rows) to " << *opt.frames << '\n';
        }
        if (opt.intensity_out) {
            write_intensity_cube(*opt.intensity_out, opt_intensity);
            std::cout << "Wrote optimized intensity cube to "
                      << *opt.intensity_out << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cpu_opt_beam_tracker: " << error.what() << '\n';
        return 1;
    }
}
