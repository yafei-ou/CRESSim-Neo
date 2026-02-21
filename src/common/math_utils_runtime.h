#ifndef CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
#define CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H

#include "gpu/gpu_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cressim::neo::common::runtime_math
{

constexpr float kPi        = 3.14159265358979323846f;
constexpr float kRadPerDeg = kPi / 180.0f;
constexpr float kDegPerRad = 180.0f / kPi;
constexpr float kEpsilon   = 1.0e-6f;

inline std::uint32_t clampExtent(std::uint32_t value)
{
    return std::max<std::uint32_t>(value, 1u);
}

inline float clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

inline float clampPositive(float value, float fallback)
{
    return value > 0.0f ? value : fallback;
}

inline float degreesToRadians(float value)
{
    return value * kRadPerDeg;
}

inline float radiansToDegrees(float value)
{
    return value * kDegPerRad;
}

inline Diligent::float3 safeNormalize(const Diligent::float3& value,
                                      const Diligent::float3& fallback = Diligent::float3{
                                          0.0f, 0.0f, 0.0f})
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

inline Diligent::QuaternionF quaternionFromEulerDegrees(float xDegrees, float yDegrees,
                                                        float zDegrees)
{
    const float x = degreesToRadians(xDegrees) * 0.5f;
    const float y = degreesToRadians(yDegrees) * 0.5f;
    const float z = degreesToRadians(zDegrees) * 0.5f;

    const float sinX = std::sin(x);
    const float cosX = std::cos(x);
    const float sinY = std::sin(y);
    const float cosY = std::cos(y);
    const float sinZ = std::sin(z);
    const float cosZ = std::cos(z);

    return Diligent::QuaternionF{
        sinZ * cosX * cosY - cosZ * sinX * sinY, cosZ * sinX * cosY + sinZ * cosX * sinY,
        cosZ * cosX * sinY - sinZ * sinX * cosY, cosZ * cosX * cosY + sinZ * sinX * sinY};
}

inline gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport& viewport)
{
    gpu::GpuRenderViewport normalized{};
    normalized.x      = clamp01(viewport.x);
    normalized.y      = clamp01(viewport.y);
    normalized.width  = clamp01(viewport.width);
    normalized.height = clamp01(viewport.height);

    const float maxWidth  = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width      = std::min(normalized.width, maxWidth);
    normalized.height     = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x     = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y      = 0.0f;
    }

    return normalized;
}

} // namespace cressim::neo::common::runtime_math

#endif // CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
