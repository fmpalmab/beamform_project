#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"

#include <vector>

namespace beamformer {

Weights generate_weights(const Dimensions& dims,
                         const std::vector<Vec3>& positions_m,
                         const std::vector<float>& frequencies_hz,
                         const std::vector<Vec3>& beam_directions);

TiledWeights generate_tiled_weights(
    const Dimensions& dims, const std::vector<Vec3>& positions_m,
    const std::vector<float>& frequencies_hz,
    const std::vector<Vec3>& beam_directions);

} // namespace beamformer
