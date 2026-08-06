#pragma once

#include <cstdint>
#include <stdexcept>

namespace beamformer {

#if defined(__CUDACC__)
#define BEAMFORMER_HOST_DEVICE __host__ __device__
#else
#define BEAMFORMER_HOST_DEVICE
#endif

struct ComplexInt4 {
    std::int8_t real;
    std::int8_t imag;
};

BEAMFORMER_HOST_DEVICE constexpr std::int8_t decode_signed_nibble(
    const std::uint8_t nibble) {
    const auto bits = static_cast<std::uint8_t>(nibble & 0x0F);
    return bits < 8 ? static_cast<std::int8_t>(bits)
                    : static_cast<std::int8_t>(static_cast<int>(bits) - 16);
}

BEAMFORMER_HOST_DEVICE constexpr ComplexInt4 unpack_complex_int4(
    const std::uint8_t packed) {
    return {decode_signed_nibble(packed),
            decode_signed_nibble(static_cast<std::uint8_t>(packed >> 4))};
}

inline std::uint8_t pack_complex_int4(const std::int8_t real, const std::int8_t imag) {
    if (real < -8 || real > 7 || imag < -8 || imag > 7) {
        throw std::out_of_range("int4 components must be in [-8, 7]");
    }
    const auto real_bits = static_cast<std::uint8_t>(real) & 0x0F;
    const auto imag_bits = static_cast<std::uint8_t>(imag) & 0x0F;
    return static_cast<std::uint8_t>(real_bits | (imag_bits << 4));
}

#undef BEAMFORMER_HOST_DEVICE

} // namespace beamformer
