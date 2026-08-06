#include "beamformer/weights.hpp"

#include "beamformer/indexing.hpp"
#include "beamformer/physics.hpp"

#include <cmath>
#include <stdexcept>

namespace beamformer {

Weights generate_weights(const Dimensions& dims,
                         const std::vector<Vec3>& positions_m,
                         const std::vector<float>& frequencies_hz,
                         const std::vector<Vec3>& beam_directions) {
    validate_dimensions(dims);
    if (positions_m.size() != dims.n_ant) {
        throw std::invalid_argument("position count must match n_ant");
    }
    if (frequencies_hz.size() != dims.n_freq) {
        throw std::invalid_argument("frequency count must match n_freq");
    }
    if (beam_directions.size() != dims.n_beams) {
        throw std::invalid_argument("beam direction count must match n_beams");
    }

    Weights weights(dims.n_beams * dims.n_freq * dims.n_ant);
    for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
        const auto& direction = beam_directions[beam];
        const double norm_squared =
            static_cast<double>(direction[0]) * direction[0]
            + static_cast<double>(direction[1]) * direction[1]
            + static_cast<double>(direction[2]) * direction[2];
        if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0) > 1.0e-3) {
            throw std::invalid_argument("beam directions must be finite unit vectors");
        }

        for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
            if (!std::isfinite(frequencies_hz[frequency])
                || frequencies_hz[frequency] <= 0.0F) {
                throw std::invalid_argument("frequencies must be positive and finite");
            }
            const double wave_number =
                two_pi * static_cast<double>(frequencies_hz[frequency])
                / speed_of_light_m_per_s;
            for (std::size_t element = 0; element < dims.n_ant; ++element) {
                const auto& position = positions_m[element];
                const double delay_m =
                    static_cast<double>(position[0]) * direction[0]
                    + static_cast<double>(position[1]) * direction[1]
                    + static_cast<double>(position[2]) * direction[2];
                const double phase = wave_number * delay_m;
                weights[weight_index(beam, frequency, element, dims)] = {
                    static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase)),
                };
            }
        }
    }
    return weights;
}


TiledWeights generate_tiled_weights(
    const Dimensions& dims, const std::vector<Vec3>& positions_m,
    const std::vector<float>& frequencies_hz,
    const std::vector<Vec3>& beam_directions) {
    validate_dimensions(dims);
    if (positions_m.size() != dims.n_ant) {
        throw std::invalid_argument("position count must match n_ant");
    }
    if (frequencies_hz.size() != dims.n_freq) {
        throw std::invalid_argument("frequency count must match n_freq");
    }
    if (beam_directions.size() != dims.n_beams) {
        throw std::invalid_argument("beam direction count must match n_beams");
    }

    TiledWeights weights(tiled_weight_count(dims), ComplexFloat{0.0F, 0.0F});
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        if (!std::isfinite(frequencies_hz[frequency])
            || frequencies_hz[frequency] <= 0.0F) {
            throw std::invalid_argument("frequencies must be positive and finite");
        }
        const double wave_number =
            two_pi * static_cast<double>(frequencies_hz[frequency])
            / speed_of_light_m_per_s;
        for (std::size_t beam_tile = 0;
             beam_tile < tiled_weight_beam_tiles(dims); ++beam_tile) {
            for (std::size_t element = 0; element < dims.n_ant; ++element) {
                for (std::size_t local_beam = 0;
                     local_beam < tiled_weight_beam_tile; ++local_beam) {
                    const std::size_t beam = beam_tile * tiled_weight_beam_tile
                                             + local_beam;
                    if (beam >= dims.n_beams) {
                        continue;
                    }
                    const auto& direction = beam_directions[beam];
                    const double norm_squared =
                        static_cast<double>(direction[0]) * direction[0]
                        + static_cast<double>(direction[1]) * direction[1]
                        + static_cast<double>(direction[2]) * direction[2];
                    if (!std::isfinite(norm_squared)
                        || std::abs(norm_squared - 1.0) > 1.0e-3) {
                        throw std::invalid_argument(
                            "beam directions must be finite unit vectors");
                    }
                    const auto& position = positions_m[element];
                    const double delay_m =
                        static_cast<double>(position[0]) * direction[0]
                        + static_cast<double>(position[1]) * direction[1]
                        + static_cast<double>(position[2]) * direction[2];
                    const double phase = wave_number * delay_m;
                    weights[tiled_weight_index(
                        frequency, beam_tile, element, local_beam, dims)] = {
                        static_cast<float>(std::cos(phase)),
                        static_cast<float>(std::sin(phase))};
                }
            }
        }
    }
    return weights;
}

} // namespace beamformer
