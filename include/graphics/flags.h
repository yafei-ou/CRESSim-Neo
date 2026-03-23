#ifndef CRESSIM_NEO_GRAPHICS_GRAPHICS_FLAGS_H
#define CRESSIM_NEO_GRAPHICS_GRAPHICS_FLAGS_H

#include <cstdint>
#include <type_traits>

namespace cressim::neo::graphics
{

enum class MaterialFeatureFlags : std::uint32_t
{
    None        = 0u,
    AlphaTest   = 1u << 0u,
    NormalMap   = 1u << 1u,
    ClearCoat   = 1u << 2u,
    DoubleSided = 1u << 3u,
};

template <typename E>
struct EnableBitMaskOps : std::false_type
{
};

template <>
struct EnableBitMaskOps<MaterialFeatureFlags> : std::true_type
{
};

template <typename E>
constexpr bool EnableBitMaskOps_v = EnableBitMaskOps<E>::value;

template <typename E, std::enable_if_t<EnableBitMaskOps_v<E>, int> = 0>
constexpr E operator|(E lhs, E rhs) noexcept
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename E, std::enable_if_t<EnableBitMaskOps_v<E>, int> = 0>
constexpr E operator&(E lhs, E rhs) noexcept
{
    using U = std::underlying_type_t<E>;
    return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <typename E, std::enable_if_t<EnableBitMaskOps_v<E>, int> = 0>
constexpr E& operator|=(E& lhs, E rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

template <typename E>
constexpr bool hasFlag(E value, E flag) noexcept
{
    static_assert(EnableBitMaskOps_v<E>, "hasFlag used with non-flag enum");
    return (value & flag) != static_cast<E>(0);
}

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GRAPHICS_FLAGS_H