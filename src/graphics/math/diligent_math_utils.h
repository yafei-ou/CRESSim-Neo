#ifndef CRESSIM_NEO_GRAPHICS_MATH_DILIGENT_MATH_UTILS_H
#define CRESSIM_NEO_GRAPHICS_MATH_DILIGENT_MATH_UTILS_H

#include "common/math_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

namespace cressim::neo::graphics::math
{

Diligent::float3 toFloat3(const common::Vec3f& value) noexcept;

Diligent::float4x4 transformMatrix(const common::Transform& transform) noexcept;
Diligent::float4x4 viewMatrixFromTransform(const common::Transform& cameraTransform) noexcept;
Diligent::float4x4 perspectiveMatrix(float verticalFovDegrees, float aspectRatio, float nearClip, float farClip) noexcept;

void copyMatrixRowMajor(float dst[16], const Diligent::float4x4& matrix) noexcept;

} // namespace cressim::neo::graphics::math

#endif // CRESSIM_NEO_GRAPHICS_MATH_DILIGENT_MATH_UTILS_H
