#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/physics.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

bool close(const float actual, const float expected) {
    return std::abs(actual - expected) < 1.0e-6F;
}

} // namespace

int main() {
    using namespace beamformer;

    const auto positions_32 = default_positions(32);
    assert(positions_32.size() == 32);
    assert((positions_32[0] == Vec3{0.0F, 0.0F, 0.0F}));
    assert(close(positions_32[7][0], 4.2F));
    assert(close(positions_32[8][1], 0.6F));
    assert(close(positions_32[31][0], 4.2F));
    assert(close(positions_32[31][1], 1.8F));

    const auto positions_64 = default_positions(64, 0.5F);
    assert(positions_64.size() == 64);
    assert((positions_64[63] == Vec3{3.5F, 3.5F, 0.0F}));

    const auto frequencies = channelized_frequencies();
    assert(frequencies.size() == default_frequency_channels);
    assert(frequencies.front() == 300'000'000.0F);
    assert(frequencies[1] == 300'300'000.0F);
    assert(frequencies.back() == 501'300'000.0F);

    const auto beams_1 = default_beam_grid(1);
    assert(beams_1.size() == 1);
    assert(beams_1[0][0] == 0.0F);
    assert(beams_1[0][1] == 0.0F);
    assert(beams_1[0][2] == 1.0F);

    const auto beams_5 = default_beam_grid(5);
    assert(beams_5[0][0] == -0.04F);
    assert(beams_5[2][0] == 0.0F);
    assert(beams_5[4][0] == 0.04F);

    const auto beams_10 = default_beam_grid(10);
    assert(beams_10[7][0] == 0.04F);
    for (const auto& direction : beams_10) {
        const float norm = std::sqrt(direction[0] * direction[0]
                                     + direction[1] * direction[1]
                                     + direction[2] * direction[2]);
        assert(std::abs(norm - 1.0F) < 1.0e-6F);
    }
    assert(default_beam_grid(128, 0.005F).size() == 128);

    const auto beams_32 = rectangular_beam_grid(32);
    assert(beams_32.size() == 32);
    const float wavelength_400_m =
        static_cast<float>(speed_of_light_m_per_s / 400'000'000.0);
    const float expected_delta_l = wavelength_400_m / (8.0F * default_spacing_m);
    const float expected_delta_m = wavelength_400_m / (4.0F * default_spacing_m);
    assert(close(beams_32[1][0] - beams_32[0][0], expected_delta_l));
    assert(close(beams_32[8][1] - beams_32[0][1], expected_delta_m));
    assert(close(beams_32[12][0], 0.5F * expected_delta_l));
    assert(close(beams_32[12][1], -0.5F * expected_delta_m));
    assert(beams_32.front()[0] < 0.0F && beams_32.front()[1] < 0.0F);
    assert(beams_32.back()[0] > 0.0F && beams_32.back()[1] > 0.0F);
    for (const auto& direction : beams_32) {
        const float norm = std::sqrt(direction[0] * direction[0]
                                     + direction[1] * direction[1]
                                     + direction[2] * direction[2]);
        assert(std::abs(norm - 1.0F) < 1.0e-6F);
    }
    const auto beams_64 = rectangular_beam_grid(64);
    assert(beams_64.size() == 64);

    const auto centered_4 = centered_integer_range(4);
    assert((centered_4 == std::vector<long long>{-2, -1, 0, 1}));

    const float fft_delta_u = wavelength_400_m / (2.0F * 8.0F * default_spacing_m);
    const float fft_delta_v_32 =
        wavelength_400_m / (2.0F * 4.0F * default_spacing_m);
    const auto fft_selection_32 = select_fft_beam_centers_rectangular(
        fft_delta_u, fft_delta_v_32, 128, 8, 4);
    assert(fft_selection_32.centers.size() == 128);
    assert(fft_selection_32.n_u == 16);
    assert(fft_selection_32.n_v == 8);
    assert(fft_selection_32.n_bank_u == 16);
    assert(fft_selection_32.n_bank_v == 8);
    assert((fft_selection_32.centers.front() == Vec2{0.0F, 0.0F}));

    const auto fft_directions_32 = fft_beam_grid(32, 128);
    assert(fft_directions_32.size() == 128);
    for (const auto& direction : fft_directions_32) {
        const float norm = std::sqrt(direction[0] * direction[0]
                                     + direction[1] * direction[1]
                                     + direction[2] * direction[2]);
        assert(std::abs(norm - 1.0F) < 1.0e-6F);
    }

    const auto fft_selection_64 = select_fft_beam_centers_rectangular(
        fft_delta_u, fft_delta_u, 128, 8, 8);
    assert(fft_selection_64.centers.size() == 128);
    assert(fft_selection_64.n_u == 12);
    assert(fft_selection_64.n_v == 11);
    assert(fft_selection_64.n_bank_u == 16);
    assert(fft_selection_64.n_bank_v == 16);

    const auto hex_estimate = estimate_nbeams_hex_formula(
        default_spacing_m, 8, 8, 108.0F, 74.0F, wavelength_400_m);
    assert(hex_estimate > 0);
    const auto hex_targets = generate_hex_targets_cropped_fov(128, 0.8F, 0.6F);
    assert(hex_targets.size() == 128);
    for (const auto& target : hex_targets) {
        const float elliptical_radius =
            target[0] * target[0] / (0.8F * 0.8F)
            + target[1] * target[1] / (0.6F * 0.6F);
        assert(elliptical_radius <= 1.0F + 1.0e-6F);
    }

    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto positions_path = temp_dir / "beamformer_poc_positions_test.txt";
    const auto frequencies_path = temp_dir / "beamformer_poc_frequencies_test.txt";

    {
        std::ofstream output(positions_path);
        output << "# x, y, z in metres\n";
        output << "1.0, 2.0, 3.0\n";
        output << "4.0 5.0 6.0\n";
    }
    {
        std::ofstream output(frequencies_path);
        output << "# Hz\n";
        output << "400000000\n";
        output << "401000000\n";
    }

    const auto loaded_positions = load_positions(positions_path, 2);
    const auto loaded_frequencies = load_frequencies(frequencies_path, 2);
    assert((loaded_positions[0] == Vec3{1.0F, 2.0F, 3.0F}));
    assert((loaded_positions[1] == Vec3{4.0F, 5.0F, 6.0F}));
    assert(loaded_frequencies[0] == 400'000'000.0F);
    assert(loaded_frequencies[1] == 401'000'000.0F);

    std::filesystem::remove(positions_path);
    std::filesystem::remove(frequencies_path);

    bool invalid_geometry_rejected = false;
    try {
        static_cast<void>(default_positions(48));
    } catch (const std::invalid_argument&) {
        invalid_geometry_rejected = true;
    }
    assert(invalid_geometry_rejected);

    return 0;
}
