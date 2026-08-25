#ifndef CRESSIM_NEO_COMMON_FLAGS_H
#define CRESSIM_NEO_COMMON_FLAGS_H

#include <type_traits>

/// @file flags.h
/// @brief Type-safe bitmask enumeration operators and flag utilities.

/// @def CRESSIM_NEO_DEFINE_ENUM_FLAGS(EnumType)
/// @brief Generates bitwise operators (`|`, `&`, `|=`) and a `hasFlag` helper for a scoped enumeration.
/// @param EnumType Scoped enumeration type to enable as a bitmask.
#define CRESSIM_NEO_DEFINE_ENUM_FLAGS(EnumType)                                                    \
    constexpr EnumType operator|(EnumType lhs, EnumType rhs) noexcept                              \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(static_cast<U>(lhs) | static_cast<U>(rhs));                   \
    }                                                                                              \
                                                                                                   \
    constexpr EnumType operator&(EnumType lhs, EnumType rhs) noexcept                              \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(static_cast<U>(lhs) & static_cast<U>(rhs));                   \
    }                                                                                              \
                                                                                                   \
    constexpr EnumType &operator|=(EnumType &lhs, EnumType rhs) noexcept                           \
    {                                                                                              \
        lhs = lhs | rhs;                                                                           \
        return lhs;                                                                                \
    }                                                                                              \
                                                                                                   \
    constexpr bool hasFlag(EnumType value, EnumType flag) noexcept                                 \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return (static_cast<U>(value) & static_cast<U>(flag)) != 0;                                \
    }

#endif // CRESSIM_NEO_COMMON_FLAGS_H
