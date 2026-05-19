#ifndef CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
#define CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H

#include "common/math_types.h"
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

inline Diligent::float3 safeNormalize(const Diligent::float3 &value,
                                      const Diligent::float3 &fallback = Diligent::float3{
                                          0.0f, 0.0f, 0.0f})
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= kEpsilon)
    {
        return fallback;
    }
    return value * (1.0f / std::sqrt(lengthSq));
}

inline Diligent::QuaternionF normalizeQuaternion(const Diligent::QuaternionF &value)
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

inline Diligent::float3 applyTransform(const common::Transform &transform,
                                       const Diligent::float3 &objectSpacePosition)
{
    const Diligent::float3 scaled{objectSpacePosition.x * transform.scale.x,
                                  objectSpacePosition.y * transform.scale.y,
                                  objectSpacePosition.z * transform.scale.z};
    return transform.rotation.RotateVector(scaled) + transform.position;
}

inline gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport &viewport)
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

inline Diligent::float2 effectiveViewportSize(float outputWidth, float outputHeight,
                                              const gpu::GpuRenderViewport &viewport)
{
    const gpu::GpuRenderViewport normalized = normalizeViewport(viewport);
    return Diligent::float2{
        clampPositive(outputWidth, 1.0f) * clampPositive(normalized.width, 1.0f),
        clampPositive(outputHeight, 1.0f) * clampPositive(normalized.height, 1.0f)};
}

inline float effectiveViewportAspect(float outputWidth, float outputHeight,
                                     const gpu::GpuRenderViewport &viewport)
{
    const Diligent::float2 size = effectiveViewportSize(outputWidth, outputHeight, viewport);
    return clampPositive(size.x, 1.0f) / clampPositive(size.y, 1.0f);
}

inline gpu::GpuRenderViewport resolvedViewport(const gpu::GpuRenderViewport &viewport,
                                               bool useViewport)
{
    return useViewport ? normalizeViewport(viewport) : gpu::GpuRenderViewport{};
}

inline Diligent::float4 viewportRect(const gpu::GpuRenderViewport &viewport, bool useViewport)
{
    const gpu::GpuRenderViewport resolved = resolvedViewport(viewport, useViewport);
    return Diligent::float4{resolved.x, resolved.y, resolved.width, resolved.height};
}

inline Diligent::uint4 viewportPixelRect(std::uint32_t outputWidth, std::uint32_t outputHeight,
                                         const gpu::GpuRenderViewport &viewport, bool useViewport)
{
    const std::uint32_t clampedWidth      = clampExtent(outputWidth);
    const std::uint32_t clampedHeight     = clampExtent(outputHeight);
    const gpu::GpuRenderViewport resolved = resolvedViewport(viewport, useViewport);

    const std::uint32_t x = std::min(
        clampedWidth - 1u,
        static_cast<std::uint32_t>(std::floor(resolved.x * static_cast<float>(clampedWidth))));
    const std::uint32_t y = std::min(
        clampedHeight - 1u,
        static_cast<std::uint32_t>(std::floor(resolved.y * static_cast<float>(clampedHeight))));
    const std::uint32_t endX = std::max(
        x + 1u, std::min(clampedWidth,
                         static_cast<std::uint32_t>(std::ceil((resolved.x + resolved.width) *
                                                              static_cast<float>(clampedWidth)))));
    const std::uint32_t endY = std::max(
        y + 1u, std::min(clampedHeight,
                         static_cast<std::uint32_t>(std::ceil((resolved.y + resolved.height) *
                                                              static_cast<float>(clampedHeight)))));
    return Diligent::uint4{x, y, endX - x, endY - y};
}

} // namespace cressim::neo::common::runtime_math

#endif // CRESSIM_NEO_COMMON_MATH_UTILS_RUNTIME_H
