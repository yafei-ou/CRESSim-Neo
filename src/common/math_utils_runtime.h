#ifndef CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
#define CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H

#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cressim::neo::common::runtime_math
{

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-12f;

inline std::uint32_t clampExtent(std::uint32_t value)
{
    return std::max<std::uint32_t>(value, 1u);
}

inline float clamp01(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

inline float clampPositive(float value, float fallback)
{
    return value > 0.0f ? value : fallback;
}

inline float degreesToRadians(float value)
{
    return value * (kPi / 180.0f);
}

inline float radiansToDegrees(float value)
{
    return value * (180.0f / kPi);
}

inline Diligent::float3 safeNormalize(
    const Diligent::float3& value,
    const Diligent::float3& fallback = Diligent::float3{0.0f, 0.0f, 0.0f})
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= kEpsilon)
    {
        return fallback;
    }
    return value * (1.0f / std::sqrt(lengthSq));
}

inline Diligent::QuaternionF normalizeQuaternion(const Diligent::QuaternionF& value)
{
    const float lengthSq = Diligent::dot(value.q, value.q);
    if (lengthSq <= kEpsilon)
    {
        return Diligent::QuaternionF{0.0f, 0.0f, 0.0f, 1.0f};
    }
    return Diligent::normalize(value);
}

inline Diligent::QuaternionF quaternionFromEulerDegrees(float pitchDegrees, float yawDegrees, float rollDegrees)
{
    const float pitch = degreesToRadians(pitchDegrees) * 0.5f;
    const float yaw = degreesToRadians(yawDegrees) * 0.5f;
    const float roll = degreesToRadians(rollDegrees) * 0.5f;

    const float sinPitch = std::sin(pitch);
    const float cosPitch = std::cos(pitch);
    const float sinYaw = std::sin(yaw);
    const float cosYaw = std::cos(yaw);
    const float sinRoll = std::sin(roll);
    const float cosRoll = std::cos(roll);

    return Diligent::QuaternionF{
        sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
        cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
        cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
        cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw};
}

inline graphics::RenderViewport normalizeViewport(const graphics::RenderViewport& viewport)
{
    graphics::RenderViewport normalized{};
    normalized.x = clamp01(viewport.x);
    normalized.y = clamp01(viewport.y);
    normalized.width = clamp01(viewport.width);
    normalized.height = clamp01(viewport.height);

    const float maxWidth = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width = std::min(normalized.width, maxWidth);
    normalized.height = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y = 0.0f;
    }

    return normalized;
}

} // namespace cressim::neo::common::runtime_math

#endif // CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
