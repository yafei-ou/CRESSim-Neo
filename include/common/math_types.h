#ifndef CRESSIM_NEO_COMMON_MATH_TYPES_H
#define CRESSIM_NEO_COMMON_MATH_TYPES_H

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

/// @file math_types.h
/// @brief Core mathematical structures and 3D spatial transformation representations.

namespace cressim::neo::common
{

/// @brief Represents a 3D affine transformation composed of translation, orientation, and scale.
///
/// In Python, this type is exposed as `cressim_neo.Transform` with `position`, `rotation`, and `scale` attributes.
/// It is used across physics, graphics, and ECS transform components (`engine::TransformComponent`).
///
/// @see cressim_neo.Transform
struct Transform
{
    Diligent::float3 position{0.0f, 0.0f, 0.0f};           ///< World-space 3D translation (x, y, z).
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};///< World-space orientation quaternion (x, y, z, w).
    Diligent::float3 scale{1.0f, 1.0f, 1.0f};              ///< Per-axis 3D scale factors (sx, sy, sz).

    /// @brief Checks equality with another Transform.
    /// @param rhs Right-hand side Transform to compare against.
    /// @return True if position, rotation, and scale match exactly.
    bool operator==(const Transform &rhs) const noexcept
    {
        return position == rhs.position && rotation == rhs.rotation && scale == rhs.scale;
    }

    /// @brief Checks inequality with another Transform.
    /// @param rhs Right-hand side Transform to compare against.
    /// @return True if any component differs.
    bool operator!=(const Transform &rhs) const noexcept
    {
        return !(*this == rhs);
    }
};

} // namespace cressim::neo::common

#endif // CRESSIM_NEO_COMMON_MATH_TYPES_H
