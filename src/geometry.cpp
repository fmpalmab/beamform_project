#include "beamformer/geometry.hpp"

#include "beamformer/physics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace beamformer {
namespace {

std::string data_part(std::string line) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
        line.erase(comment);
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    return line;
}

std::runtime_error parse_error(const std::filesystem::path& path, const std::size_t line_number,
                               const std::string& reason) {
    return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + reason);
}

} // namespace

std::vector<Vec3> regular_array(const std::size_t rows, const std::size_t columns,
                                const float spacing_m) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("array rows and columns must be positive");
    }
    if (spacing_m <= 0.0F) {
        throw std::invalid_argument("antenna spacing must be positive");
    }

    std::vector<Vec3> positions;
    positions.reserve(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            positions.push_back({static_cast<float>(column) * spacing_m,
                                 static_cast<float>(row) * spacing_m, 0.0F});
        }
    }
    return positions;
}

std::vector<Vec3> default_positions(const std::size_t n_ant, const float spacing_m) {
    if (n_ant == 32) {
        return regular_array(4, 8, spacing_m);
    }
    if (n_ant == 64) {
        return regular_array(8, 8, spacing_m);
    }
    throw std::invalid_argument("default geometry is available only for 32 or 64 antennas");
}

std::vector<float> constant_frequencies(const std::size_t n_freq, const float frequency_hz) {
    if (n_freq == 0) {
        throw std::invalid_argument("frequency count must be positive");
    }
    if (frequency_hz <= 0.0F) {
        throw std::invalid_argument("frequency must be positive");
    }
    return std::vector<float>(n_freq, frequency_hz);
}

std::vector<float> channelized_frequencies(const std::size_t n_freq,
                                           const float start_hz,
                                           const float channel_width_hz) {
    if (n_freq == 0) {
        throw std::invalid_argument("frequency count must be positive");
    }
    if (!std::isfinite(start_hz) || start_hz <= 0.0F
        || !std::isfinite(channel_width_hz) || channel_width_hz <= 0.0F) {
        throw std::invalid_argument("frequency start and channel width must be positive");
    }
    std::vector<float> frequencies(n_freq);
    for (std::size_t channel = 0; channel < n_freq; ++channel) {
        frequencies[channel] =
            start_hz + static_cast<float>(channel) * channel_width_hz;
    }
    return frequencies;
}

Vec3 direction_from_lm(const float l, const float m) {
    if (!std::isfinite(l) || !std::isfinite(m)) {
        throw std::invalid_argument("direction cosines l and m must be finite");
    }
    const float transverse_squared = l * l + m * m;
    if (transverse_squared > 1.0F) {
        throw std::invalid_argument("direction cosines must satisfy l*l + m*m <= 1");
    }
    return {l, m, std::sqrt(1.0F - transverse_squared)};
}

std::vector<Vec3> default_beam_grid(const std::size_t n_beams, const float l_step,
                                    const float m) {
    if (n_beams == 0 || n_beams > maximum_beams) {
        throw std::invalid_argument("beam grid size must be between 1 and 128");
    }
    if (!std::isfinite(l_step) || l_step <= 0.0F) {
        throw std::invalid_argument("beam l step must be positive and finite");
    }

    std::vector<Vec3> directions;
    directions.reserve(n_beams);
    const auto center = static_cast<long long>(n_beams / 2);
    for (std::size_t beam = 0; beam < n_beams; ++beam) {
        const auto offset = static_cast<long long>(beam) - center;
        directions.push_back(direction_from_lm(static_cast<float>(offset) * l_step, m));
    }
    return directions;
}

std::vector<long long> centered_integer_range(const std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("centered integer range count must be positive");
    }
    std::vector<long long> indices(count);
    const auto start = -static_cast<long long>(count / 2);
    for (std::size_t index = 0; index < count; ++index) {
        indices[index] = start + static_cast<long long>(index);
    }
    return indices;
}

FftBeamSelection select_fft_beam_centers_rectangular(
    const float du_fft_u, const float dv_fft_v, const std::size_t n_beams,
    const std::size_t m_side, const std::size_t n_side) {
    if (!std::isfinite(du_fft_u) || du_fft_u <= 0.0F
        || !std::isfinite(dv_fft_v) || dv_fft_v <= 0.0F) {
        throw std::invalid_argument("FFT beam spacing must be positive and finite");
    }
    if (m_side == 0 || n_side == 0) {
        throw std::invalid_argument("FFT array sides must be positive");
    }
    if (n_beams == 0) {
        throw std::invalid_argument("FFT beam count must be positive");
    }

    FftBeamSelection selection;
    selection.n_bank_u = 2 * m_side;
    selection.n_bank_v = 2 * n_side;
    const std::size_t bank_total = selection.n_bank_u * selection.n_bank_v;
    if (n_beams > bank_total) {
        throw std::invalid_argument(
            "n_beams exceeds the zero-padded rectangular FFT bin bank");
    }

    const auto ceil_div = [](const std::size_t numerator,
                             const std::size_t denominator) {
        return (numerator + denominator - 1) / denominator;
    };
    selection.n_u = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(n_beams))));
    selection.n_v = ceil_div(n_beams, selection.n_u);
    if (selection.n_u > selection.n_bank_u) {
        selection.n_u = selection.n_bank_u;
        selection.n_v = ceil_div(n_beams, selection.n_u);
    }
    if (selection.n_v > selection.n_bank_v) {
        selection.n_v = selection.n_bank_v;
        // Recompute n_u after clamping n_v. Without this step the 4x8,
        // 128-beam case would produce only 12x8=96 candidates.
        selection.n_u = ceil_div(n_beams, selection.n_v);
    }
    if (selection.n_u > selection.n_bank_u
        || selection.n_u * selection.n_v < n_beams) {
        throw std::logic_error("failed to select a sufficient FFT beam window");
    }

    struct RankedCenter {
        double radius_squared = 0.0;
        float u = 0.0F;
        float v = 0.0F;
    };
    std::vector<RankedCenter> ranked;
    ranked.reserve(selection.n_u * selection.n_v);
    const auto indices_u = centered_integer_range(selection.n_u);
    const auto indices_v = centered_integer_range(selection.n_v);
    for (const long long i : indices_u) {
        for (const long long j : indices_v) {
            const float u = static_cast<float>(i) * du_fft_u;
            const float v = static_cast<float>(j) * dv_fft_v;
            ranked.push_back({
                static_cast<double>(u) * u + static_cast<double>(v) * v,
                u,
                v,
            });
        }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedCenter& left, const RankedCenter& right) {
                  return std::tie(left.radius_squared, left.u, left.v)
                         < std::tie(right.radius_squared, right.u, right.v);
              });

    selection.centers.reserve(n_beams);
    for (std::size_t index = 0; index < n_beams; ++index) {
        selection.centers.push_back({ranked[index].u, ranked[index].v});
    }
    return selection;
}

std::vector<Vec3> fft_beam_grid(const std::size_t n_ant,
                                const std::size_t n_beams,
                                const float spacing_m,
                                const float design_frequency_hz) {
    if (!std::isfinite(spacing_m) || spacing_m <= 0.0F
        || !std::isfinite(design_frequency_hz)
        || design_frequency_hz <= 0.0F) {
        throw std::invalid_argument(
            "FFT beam-grid spacing and design frequency must be positive");
    }
    const std::size_t n_side = n_ant == 32 ? 4 : n_ant == 64 ? 8 : 0;
    const std::size_t m_side = n_ant == 32 || n_ant == 64 ? 8 : 0;
    if (m_side == 0) {
        throw std::invalid_argument(
            "FFT beam grid is available only for 32 or 64 antennas");
    }

    const float wavelength_m = static_cast<float>(
        speed_of_light_m_per_s / static_cast<double>(design_frequency_hz));
    const float du_fft_u =
        wavelength_m / (2.0F * static_cast<float>(m_side) * spacing_m);
    const float dv_fft_v =
        wavelength_m / (2.0F * static_cast<float>(n_side) * spacing_m);
    const auto selection = select_fft_beam_centers_rectangular(
        du_fft_u, dv_fft_v, n_beams, m_side, n_side);

    std::vector<Vec3> directions;
    directions.reserve(selection.centers.size());
    for (const auto& center : selection.centers) {
        directions.push_back(direction_from_lm(center[0], center[1]));
    }
    return directions;
}

std::size_t estimate_nbeams_hex_formula(
    const float spacing_m, const std::size_t m_side,
    const std::size_t n_side, const float bw_e_deg,
    const float bw_h_deg, const float wavelength_m) {
    if (!std::isfinite(spacing_m) || spacing_m <= 0.0F
        || m_side == 0 || n_side == 0
        || !std::isfinite(bw_e_deg) || bw_e_deg <= 0.0F
        || !std::isfinite(bw_h_deg) || bw_h_deg <= 0.0F
        || !std::isfinite(wavelength_m) || wavelength_m <= 0.0F) {
        throw std::invalid_argument("hex beam estimate inputs must be positive");
    }
    const double eta_hex = two_pi / (4.0 * std::sqrt(3.0));
    const double d_u = static_cast<double>(spacing_m) * (m_side - 1);
    const double d_v = static_cast<double>(spacing_m) * (n_side - 1);
    const double degrees_to_radians = two_pi / 360.0;
    const double estimate =
        eta_hex * 4.0 * d_u * d_v
        * std::sin(static_cast<double>(bw_e_deg) * degrees_to_radians / 2.0)
        * std::sin(static_cast<double>(bw_h_deg) * degrees_to_radians / 2.0)
        / (static_cast<double>(wavelength_m) * wavelength_m);
    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(estimate)));
}

std::vector<Vec2> generate_hex_targets_cropped_fov(
    const std::size_t n_beams, const float u_max, const float v_max) {
    if (n_beams == 0) {
        return {};
    }
    if (!std::isfinite(u_max) || u_max <= 0.0F
        || !std::isfinite(v_max) || v_max <= 0.0F) {
        throw std::invalid_argument("hex FoV axes must be positive and finite");
    }

    double unit_spacing =
        std::sqrt(two_pi / (std::sqrt(3.0) * static_cast<double>(n_beams)));
    std::vector<Vec2> points;
    for (std::size_t attempt = 0; attempt < 10; ++attempt) {
        points.clear();
        const double dy = std::sqrt(3.0) * 0.5 * unit_spacing;
        const auto j_max = static_cast<long long>(
            std::ceil(1.0 / std::max(dy, 1.0e-12))) + 2;
        const auto i_max = static_cast<long long>(
            std::ceil(1.0 / std::max(unit_spacing, 1.0e-12))) + 2;
        for (long long j = -j_max; j <= j_max; ++j) {
            const double y = static_cast<double>(j) * dy;
            const double x_shift = j % 2 != 0 ? 0.5 * unit_spacing : 0.0;
            for (long long i = -i_max; i <= i_max; ++i) {
                const double x = static_cast<double>(i) * unit_spacing + x_shift;
                if (x * x + y * y <= 1.0) {
                    points.push_back({
                        static_cast<float>(x * u_max),
                        static_cast<float>(y * v_max),
                    });
                }
            }
        }
        if (points.size() >= n_beams) {
            break;
        }
        unit_spacing *= 0.9;
    }
    if (points.empty()) {
        return {{0.0F, 0.0F}};
    }
    std::sort(points.begin(), points.end(), [](const Vec2& left, const Vec2& right) {
        const double left_radius =
            static_cast<double>(left[0]) * left[0]
            + static_cast<double>(left[1]) * left[1];
        const double right_radius =
            static_cast<double>(right[0]) * right[0]
            + static_cast<double>(right[1]) * right[1];
        return std::tie(left_radius, left[0], left[1])
               < std::tie(right_radius, right[0], right[1]);
    });
    if (points.size() > n_beams) {
        points.resize(n_beams);
    }
    return points;
}

std::vector<Vec3> rectangular_beam_grid(const std::size_t n_ant,
                                        const float spacing_m,
                                        const float design_frequency_hz) {
    if (spacing_m <= 0.0F || !std::isfinite(spacing_m)) {
        throw std::invalid_argument("antenna spacing must be positive and finite");
    }
    if (design_frequency_hz <= 0.0F || !std::isfinite(design_frequency_hz)) {
        throw std::invalid_argument("beam-grid design frequency must be positive and finite");
    }

    const std::size_t rows = n_ant == 32 ? 4 : n_ant == 64 ? 8 : 0;
    const std::size_t columns = n_ant == 32 || n_ant == 64 ? 8 : 0;
    if (rows == 0) {
        throw std::invalid_argument("rectangular beam grid is available only for 32 or 64 beams");
    }

    const double wavelength_m = speed_of_light_m_per_s
                                / static_cast<double>(design_frequency_hz);
    const double delta_l = wavelength_m / (static_cast<double>(columns) * spacing_m);
    const double delta_m = wavelength_m / (static_cast<double>(rows) * spacing_m);

    std::vector<Vec3> directions;
    directions.reserve(n_ant);
    for (std::size_t row = 0; row < rows; ++row) {
        const double m = (static_cast<double>(row)
                          - static_cast<double>(rows - 1) / 2.0) * delta_m;
        for (std::size_t column = 0; column < columns; ++column) {
            const double l = (static_cast<double>(column)
                              - static_cast<double>(columns - 1) / 2.0) * delta_l;
            directions.push_back(direction_from_lm(static_cast<float>(l),
                                                   static_cast<float>(m)));
        }
    }
    return directions;
}

std::vector<Vec3> load_positions(const std::filesystem::path& path,
                                 const std::size_t expected_count) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open positions file: " + path.string());
    }

    std::vector<Vec3> positions;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream values(data_part(std::move(line)));
        Vec3 position{};
        if (!(values >> position[0])) {
            continue;
        }
        if (!(values >> position[1] >> position[2])) {
            throw parse_error(path, line_number, "expected three coordinates");
        }
        std::string extra;
        if (values >> extra) {
            throw parse_error(path, line_number, "unexpected value after third coordinate");
        }
        positions.push_back(position);
    }

    if (positions.size() != expected_count) {
        throw std::runtime_error("positions file contains " + std::to_string(positions.size())
                                 + " rows; expected " + std::to_string(expected_count));
    }
    return positions;
}

std::vector<float> load_frequencies(const std::filesystem::path& path,
                                    const std::size_t expected_count) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open frequency file: " + path.string());
    }

    std::vector<float> frequencies;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream values(data_part(std::move(line)));
        float frequency = 0.0F;
        if (!(values >> frequency)) {
            continue;
        }
        if (frequency <= 0.0F) {
            throw parse_error(path, line_number, "frequency must be positive");
        }
        std::string extra;
        if (values >> extra) {
            throw parse_error(path, line_number, "expected one frequency per line");
        }
        frequencies.push_back(frequency);
    }

    if (frequencies.size() != expected_count) {
        throw std::runtime_error("frequency file contains " + std::to_string(frequencies.size())
                                 + " rows; expected " + std::to_string(expected_count));
    }
    return frequencies;
}

std::vector<Vec3> load_directions(const std::filesystem::path& path,
                                  const std::size_t expected_count) {
    const auto directions = load_positions(path, expected_count);
    for (const auto& direction : directions) {
        const float norm_squared = direction[0] * direction[0] + direction[1] * direction[1]
                                   + direction[2] * direction[2];
        if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0F) > 1.0e-3F) {
            throw std::runtime_error("beam directions must be finite unit vectors");
        }
    }
    return directions;
}

} // namespace beamformer
