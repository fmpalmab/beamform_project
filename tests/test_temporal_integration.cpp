#include "beamformer/config.hpp"
#include "beamformer/temporal_integration.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

bool close(const float actual, const float expected) {
    return std::abs(actual - expected) <= 1.0e-5F;
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

std::size_t raw_index(const std::size_t time, const std::size_t frequency,
                      const std::size_t beam, const beamformer::Dimensions& dims) {
    return (time * dims.n_freq + frequency) * dims.n_beams + beam;
}

std::size_t integrated_index(const std::size_t window, const std::size_t frequency,
                             const std::size_t beam,
                             const beamformer::Dimensions& dims) {
    return (window * dims.n_freq + frequency) * dims.n_beams + beam;
}

} // namespace

int main() {
    using namespace beamformer;

    assert(default_integration_spectra == 320);
    assert(spectrum_period_us_numerator == 10);
    assert(spectrum_period_us_denominator == 3);
    assert(integration_after_upchan.integration_spectra == 10);
    assert(integration_direct.integration_spectra == 320);
    assert(integration_with_upchan.integration_spectra
           == integration_after_upchan.integration_spectra);
    assert(integration_without_upchan.integration_spectra
           == integration_direct.integration_spectra);
    assert(integrated_time_count(480, integration_after_upchan) == 48);
    assert(integrated_time_count(15360, integration_direct) == 48);
    assert(integrated_time_count(1) == 1);
    assert(integrated_time_count(299) == 1);
    assert(integrated_time_count(300) == 1);
    assert(integrated_time_count(301) == 1);
    assert(integrated_time_count(15360) == 48);
    assert(throws_invalid_argument([] { integrated_time_count(0); }));
    assert(throws_invalid_argument([] { integrated_time_count(10, 0); }));

    const Dimensions dims{7, default_frequency_channels, 32, 3};
    Intensities input(dims.n_time * dims.n_freq * dims.n_beams, 0.0F);
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                input[raw_index(time, frequency, beam, dims)] =
                    static_cast<float>(100 * time + 10 * frequency + beam);
            }
        }
    }

    const auto integrated = cpu_integrate_intensity(input, dims, 3);
    assert(integrated.size() == 3 * dims.n_freq * dims.n_beams);
    for (std::size_t window = 0; window < 3; ++window) {
        const std::size_t first_time = window * 3;
        const std::size_t last_time = std::min(first_time + 3, dims.n_time);
        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
                float expected = 0.0F;
                for (std::size_t time = first_time; time < last_time; ++time) {
                    expected += static_cast<float>(100 * time + 10 * frequency + beam);
                }
                assert(close(integrated[integrated_index(
                    window, frequency, beam, dims)], expected));
            }
        }
    }

    const auto default_integrated = cpu_integrate_intensity(input, dims);
    assert(default_integrated.size() == dims.n_freq * dims.n_beams);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
            float expected = 0.0F;
            for (std::size_t time = 0; time < dims.n_time; ++time) {
                expected += input[raw_index(time, frequency, beam, dims)];
            }
            assert(close(default_integrated[
                             integrated_index(0, frequency, beam, dims)], expected));
        }
    }

    const Dimensions partial_dims{321, default_frequency_channels, 32, 1};
    Intensities ones(partial_dims.n_time * partial_dims.n_freq, 1.0F);
    const auto partial = cpu_integrate_intensity(ones, partial_dims);
    assert(partial.size() == 2 * partial_dims.n_freq);
    for (std::size_t frequency = 0; frequency < partial_dims.n_freq; ++frequency) {
        assert(partial[integrated_index(0, frequency, 0, partial_dims)] == 320.0F);
        assert(partial[integrated_index(1, frequency, 0, partial_dims)] == 1.0F);
    }

    const Dimensions post_upchan_dims{480, default_frequency_channels, 32, 3};
    Intensities post_upchan_input(
        post_upchan_dims.n_time * post_upchan_dims.n_freq * post_upchan_dims.n_beams,
        1.0F);
    const auto after_upchan = cpu_integrate_intensity(
        post_upchan_input, post_upchan_dims, integration_after_upchan);
    assert(after_upchan.size() == 48 * post_upchan_dims.n_freq
           * post_upchan_dims.n_beams);
    for (std::size_t value = 0; value < after_upchan.size(); ++value) {
        assert(after_upchan[value] == 10.0F);
    }

    assert(throws_invalid_argument([] {
        validate_temporal_config({0});
    }));

    IntegratedIntensities reusable(3 * dims.n_freq * dims.n_beams + 1, -7.0F);
    cpu_integrate_intensity_into(input, dims, 3, reusable);
    assert(reusable.back() == -7.0F);
    assert(throws_invalid_argument([&] {
        cpu_integrate_intensity(input, dims, 0);
    }));
    assert(throws_invalid_argument([&] {
        cpu_integrate_intensity(Intensities(1), dims, 3);
    }));
    IntegratedIntensities too_small(1);
    assert(throws_invalid_argument([&] {
        cpu_integrate_intensity_into(input, dims, 3, too_small);
    }));
    return 0;
}
