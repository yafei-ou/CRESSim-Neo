#ifndef CRESSIM_NEO_EXAMPLES_HELPERS_SHAPE_MESHES_H
#define CRESSIM_NEO_EXAMPLES_HELPERS_SHAPE_MESHES_H

#include "graphics/render_resource_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cressim::neo::examples::helpers
{

namespace detail
{

constexpr float kPi = 3.14159265358979323846f;

inline Diligent::float4 makeFaceTangent(const Diligent::float3 &v0, const Diligent::float3 &v1)
{
    const Diligent::float3 tangentDir{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    const float tangentLengthSq = tangentDir.x * tangentDir.x + tangentDir.y * tangentDir.y +
                                  tangentDir.z * tangentDir.z;
    const float tangentInvLength =
        tangentLengthSq > 1.0e-6f ? 1.0f / std::sqrt(tangentLengthSq) : 1.0f;
    return {tangentDir.x * tangentInvLength, tangentDir.y * tangentInvLength,
            tangentDir.z * tangentInvLength, 1.0f};
}

} // namespace detail

inline graphics::MeshResourceDesc makeBoxMesh(const Diligent::float3 &halfExtents,
                                              const std::string &debugName,
                                              float uvScaleU = 1.0f, float uvScaleV = 1.0f)
{
    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    const auto addFace = [&](const Diligent::float3 &normal, const Diligent::float3 &v0,
                             const Diligent::float3 &v1, const Diligent::float3 &v2,
                             const Diligent::float3 &v3) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        const Diligent::float4 tangent = detail::makeFaceTangent(v0, v1);
        mesh.vertices.push_back({v0, normal, 0.0f, 0.0f, tangent});
        mesh.vertices.push_back({v1, normal, uvScaleU, 0.0f, tangent});
        mesh.vertices.push_back({v2, normal, uvScaleU, uvScaleV, tangent});
        mesh.vertices.push_back({v3, normal, 0.0f, uvScaleV, tangent});

        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 3u);
        mesh.indices.push_back(base + 2u);
    };

    const float hx = halfExtents.x;
    const float hy = halfExtents.y;
    const float hz = halfExtents.z;
    addFace({0.0f, 0.0f, 1.0f}, {-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz});
    addFace({0.0f, 0.0f, -1.0f}, {hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx, hy, -hz},
            {hx, hy, -hz});
    addFace({-1.0f, 0.0f, 0.0f}, {-hx, -hy, -hz}, {-hx, -hy, hz}, {-hx, hy, hz},
            {-hx, hy, -hz});
    addFace({1.0f, 0.0f, 0.0f}, {hx, -hy, hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {hx, hy, hz});
    addFace({0.0f, 1.0f, 0.0f}, {-hx, hy, hz}, {hx, hy, hz}, {hx, hy, -hz}, {-hx, hy, -hz});
    addFace({0.0f, -1.0f, 0.0f}, {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz},
            {-hx, -hy, hz});
    return mesh;
}

inline graphics::MeshResourceDesc makeCubeMesh(float halfExtent, const std::string &debugName,
                                               float uvScale = 1.0f)
{
    return makeBoxMesh({halfExtent, halfExtent, halfExtent}, debugName, uvScale, uvScale);
}

inline graphics::MeshResourceDesc makePlaneMesh(float halfExtent, const std::string &debugName,
                                                float uvScale = 1.0f)
{
    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    const float h = halfExtent;
    constexpr Diligent::float4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    mesh.vertices = {
        {{-h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, 0.0f, 0.0f, tangent},
        {{h, 0.0f, -h}, {0.0f, 1.0f, 0.0f}, uvScale, 0.0f, tangent},
        {{h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, uvScale, uvScale, tangent},
        {{-h, 0.0f, h}, {0.0f, 1.0f, 0.0f}, 0.0f, uvScale, tangent}};
    mesh.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    return mesh;
}

inline graphics::MeshResourceDesc makePlaneMesh(float halfExtent, float uvScale,
                                                const std::string &debugName)
{
    return makePlaneMesh(halfExtent, debugName, uvScale);
}

inline graphics::MeshResourceDesc makeSphereMesh(float radius, std::uint32_t slices,
                                                 std::uint32_t stacks,
                                                 const std::string &debugName)
{
    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    mesh.vertices.reserve((stacks + 1u) * (slices + 1u));
    mesh.indices.reserve(stacks * slices * 6u);

    for (std::uint32_t stack = 0u; stack <= stacks; ++stack)
    {
        const float v = static_cast<float>(stack) / static_cast<float>(stacks);
        const float phi = v * detail::kPi;
        const float y = std::cos(phi);
        const float ringRadius = std::sin(phi);

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * detail::kPi);
            const float x = ringRadius * std::cos(theta);
            const float z = ringRadius * std::sin(theta);
            const Diligent::float3 normal{x, y, z};
            mesh.vertices.push_back({normal * radius, normal, u, v});
        }
    }

    const std::uint32_t ring = slices + 1u;
    for (std::uint32_t stack = 0u; stack < stacks; ++stack)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = stack * ring + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ring;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

inline graphics::MeshResourceDesc makeCapsuleMesh(float radius, float halfHeight,
                                                  std::uint32_t slices,
                                                  std::uint32_t hemisphereRings,
                                                  std::uint32_t bodyRings,
                                                  const std::string &debugName)
{
    struct Ring
    {
        float y = 0.0f;
        float r = 0.0f;
    };

    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;
    std::vector<Ring> rings;
    rings.reserve(2u * hemisphereRings + bodyRings + 2u);

    rings.push_back({halfHeight + radius, 0.0f});
    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (detail::kPi * 0.5f);
        rings.push_back(
            {halfHeight + radius * std::cos(angle), radius * std::sin(angle)});
    }

    for (std::uint32_t i = 1u; i <= bodyRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(bodyRings + 1u);
        rings.push_back({halfHeight * (1.0f - 2.0f * t), radius});
    }

    for (std::uint32_t i = 1u; i <= hemisphereRings; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(hemisphereRings);
        const float angle = t * (detail::kPi * 0.5f);
        rings.push_back(
            {-halfHeight - radius * std::sin(angle), radius * std::cos(angle)});
    }

    mesh.vertices.reserve(static_cast<std::size_t>(rings.size()) * (slices + 1u));
    mesh.indices.reserve((static_cast<std::uint32_t>(rings.size()) - 1u) * slices * 6u);

    for (std::size_t ringIndex = 0u; ringIndex < rings.size(); ++ringIndex)
    {
        const float y = rings[ringIndex].y;
        const float rr = rings[ringIndex].r;

        for (std::uint32_t slice = 0u; slice <= slices; ++slice)
        {
            const float u = static_cast<float>(slice) / static_cast<float>(slices);
            const float theta = u * (2.0f * detail::kPi);
            const float x = rr * std::cos(theta);
            const float z = rr * std::sin(theta);

            Diligent::float3 normal{};
            if (y > halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y - halfHeight, z});
            }
            else if (y < -halfHeight)
            {
                normal = Diligent::normalize(Diligent::float3{x, y + halfHeight, z});
            }
            else
            {
                normal = rr > 0.0f ? Diligent::normalize(Diligent::float3{x, 0.0f, z})
                                   : Diligent::float3{0.0f, 1.0f, 0.0f};
            }

            const float v = static_cast<float>(ringIndex) /
                            static_cast<float>(std::max<std::size_t>(1u, rings.size() - 1u));
            mesh.vertices.push_back({{x, y, z}, normal, u, v});
        }
    }

    const std::uint32_t ringStride = slices + 1u;
    for (std::uint32_t ringIndex = 0u;
         ringIndex + 1u < static_cast<std::uint32_t>(rings.size()); ++ringIndex)
    {
        for (std::uint32_t slice = 0u; slice < slices; ++slice)
        {
            const std::uint32_t i0 = ringIndex * ringStride + slice;
            const std::uint32_t i1 = i0 + 1u;
            const std::uint32_t i2 = i0 + ringStride;
            const std::uint32_t i3 = i2 + 1u;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

inline graphics::MeshResourceDesc makeCanonicalCurveTubeMesh(std::uint32_t sampleCount,
                                                             std::uint32_t radialResolution,
                                                             const std::string &debugName,
                                                             float uvScale = 1.0f)
{
    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;

    const std::uint32_t clampedSamples = std::max(sampleCount, 2u);
    const std::uint32_t clampedRadialResolution = std::max(radialResolution, 3u);
    mesh.vertices.reserve(static_cast<std::size_t>(clampedSamples) * clampedRadialResolution);
    mesh.indices.reserve(static_cast<std::size_t>(clampedSamples - 1u) * clampedRadialResolution *
                         6u);

    const float longitudinalScale = std::max(uvScale, 1.0e-4f);
    for (std::uint32_t sampleIndex = 0u; sampleIndex < clampedSamples; ++sampleIndex)
    {
        const float u = (clampedSamples > 1u)
                            ? (static_cast<float>(sampleIndex) /
                               static_cast<float>(clampedSamples - 1u)) *
                                  longitudinalScale
                            : 0.0f;
        const float x = static_cast<float>(sampleIndex);
        for (std::uint32_t radialIndex = 0u; radialIndex < clampedRadialResolution; ++radialIndex)
        {
            const float v = static_cast<float>(radialIndex) /
                            static_cast<float>(clampedRadialResolution);
            const float angle = v * (2.0f * detail::kPi);
            const Diligent::float3 normal{0.0f, std::cos(angle), std::sin(angle)};
            constexpr Diligent::float4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
            mesh.vertices.push_back({{x, normal.y, normal.z}, normal, u, v, tangent});
        }
    }

    for (std::uint32_t sampleIndex = 0u; sampleIndex + 1u < clampedSamples; ++sampleIndex)
    {
        const std::uint32_t ringBase = sampleIndex * clampedRadialResolution;
        const std::uint32_t nextRingBase = ringBase + clampedRadialResolution;
        for (std::uint32_t radialIndex = 0u; radialIndex < clampedRadialResolution; ++radialIndex)
        {
            const std::uint32_t nextRadialIndex = (radialIndex + 1u) % clampedRadialResolution;
            const std::uint32_t i0 = ringBase + radialIndex;
            const std::uint32_t i1 = ringBase + nextRadialIndex;
            const std::uint32_t i2 = nextRingBase + radialIndex;
            const std::uint32_t i3 = nextRingBase + nextRadialIndex;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

inline graphics::MeshResourceDesc makePolylineTubeMesh(const std::vector<Diligent::float3> &points,
                                                       std::uint32_t radialResolution,
                                                       float radius,
                                                       const std::string &debugName,
                                                       float uvScale = 1.0f,
                                                       float startRadiusScale = 1.0f,
                                                       float endRadiusScale = 1.0f)
{
    graphics::MeshResourceDesc mesh{};
    mesh.debugName = debugName;

    const std::uint32_t sampleCount = static_cast<std::uint32_t>(points.size());
    const std::uint32_t clampedRadialResolution = std::max(radialResolution, 3u);
    const float clampedRadius = std::max(radius, 1.0e-5f);
    const float clampedUvScale = std::max(uvScale, 1.0e-4f);
    if (sampleCount < 2u)
    {
        return mesh;
    }

    mesh.vertices.reserve(static_cast<std::size_t>(sampleCount) * clampedRadialResolution);
    mesh.indices.reserve(static_cast<std::size_t>(sampleCount - 1u) * clampedRadialResolution * 6u);

    auto normalizeOrFallback = [](const Diligent::float3 &value,
                                  const Diligent::float3 &fallback) -> Diligent::float3
    {
        const float lenSq = Diligent::dot(value, value);
        return lenSq > 1.0e-12f ? value / std::sqrt(lenSq) : fallback;
    };
    auto chooseReferenceAxis = [](const Diligent::float3 &tangent) -> Diligent::float3
    {
        return std::abs(tangent.y) < 0.95f ? Diligent::float3{0.0f, 1.0f, 0.0f}
                                           : Diligent::float3{1.0f, 0.0f, 0.0f};
    };
    auto rotateAroundAxis = [](const Diligent::float3 &value, const Diligent::float3 &axis,
                               float sinAngle, float cosAngle) -> Diligent::float3
    {
        return value * cosAngle + Diligent::cross(axis, value) * sinAngle +
               axis * Diligent::dot(axis, value) * (1.0f - cosAngle);
    };

    std::vector<float> arcLengths(sampleCount, 0.0f);
    for (std::uint32_t i = 1u; i < sampleCount; ++i)
    {
        arcLengths[i] =
            arcLengths[i - 1u] + Diligent::length(points[i] - points[i - 1u]) * clampedUvScale;
    }
    const float totalArcLength = std::max(arcLengths.back(), 1.0e-4f);

    Diligent::float3 prevTangent =
        normalizeOrFallback(points[1] - points[0], Diligent::float3{1.0f, 0.0f, 0.0f});
    Diligent::float3 normal = normalizeOrFallback(
        Diligent::cross(chooseReferenceAxis(prevTangent), prevTangent),
        Diligent::float3{0.0f, 0.0f, 1.0f});
    Diligent::float3 binormal =
        normalizeOrFallback(Diligent::cross(prevTangent, normal), Diligent::float3{0.0f, 1.0f, 0.0f});

    for (std::uint32_t sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex)
    {
        const Diligent::float3 center = points[sampleIndex];
        Diligent::float3 tangent = prevTangent;
        if (sampleIndex == 0u)
        {
            tangent = normalizeOrFallback(points[1] - center, prevTangent);
        }
        else if (sampleIndex + 1u >= sampleCount)
        {
            tangent = normalizeOrFallback(center - points[sampleIndex - 1u], prevTangent);
        }
        else
        {
            tangent = normalizeOrFallback(points[sampleIndex + 1u] - points[sampleIndex - 1u],
                                          prevTangent);
        }

        if (sampleIndex > 0u)
        {
            const Diligent::float3 axis = Diligent::cross(prevTangent, tangent);
            const float axisLenSq = Diligent::dot(axis, axis);
            if (axisLenSq > 1.0e-12f)
            {
                const float axisLen = std::sqrt(axisLenSq);
                const Diligent::float3 unitAxis = axis / axisLen;
                const float cosAngle =
                    std::clamp(Diligent::dot(prevTangent, tangent), -1.0f, 1.0f);
                const float sinAngle = std::min(axisLen, 1.0f);
                normal = rotateAroundAxis(normal, unitAxis, sinAngle, cosAngle);
            }
            normal =
                normalizeOrFallback(normal - tangent * Diligent::dot(normal, tangent),
                                    normalizeOrFallback(
                                        Diligent::cross(chooseReferenceAxis(tangent), tangent),
                                        Diligent::float3{0.0f, 0.0f, 1.0f}));
            binormal = normalizeOrFallback(Diligent::cross(tangent, normal), binormal);
        }

        const float t = sampleCount > 1u
                            ? static_cast<float>(sampleIndex) / static_cast<float>(sampleCount - 1u)
                            : 0.0f;
        const float radiusScale = (1.0f - t) * startRadiusScale + t * endRadiusScale;
        const float sampleRadius = clampedRadius * std::max(radiusScale, 0.0f);
        const float u = arcLengths[sampleIndex] / totalArcLength;

        for (std::uint32_t radialIndex = 0u; radialIndex < clampedRadialResolution; ++radialIndex)
        {
            const float v = static_cast<float>(radialIndex) /
                            static_cast<float>(clampedRadialResolution);
            const float angle = v * (2.0f * detail::kPi);
            const float s = std::sin(angle);
            const float c = std::cos(angle);
            const Diligent::float3 radialDirection = c * normal + s * binormal;
            mesh.vertices.push_back({center + radialDirection * sampleRadius,
                                     normalizeOrFallback(radialDirection, normal), u, v,
                                     Diligent::float4{tangent.x, tangent.y, tangent.z, 1.0f}});
        }

        prevTangent = tangent;
    }

    for (std::uint32_t sampleIndex = 0u; sampleIndex + 1u < sampleCount; ++sampleIndex)
    {
        const std::uint32_t ringBase = sampleIndex * clampedRadialResolution;
        const std::uint32_t nextRingBase = ringBase + clampedRadialResolution;
        for (std::uint32_t radialIndex = 0u; radialIndex < clampedRadialResolution; ++radialIndex)
        {
            const std::uint32_t nextRadialIndex = (radialIndex + 1u) % clampedRadialResolution;
            const std::uint32_t i0 = ringBase + radialIndex;
            const std::uint32_t i1 = ringBase + nextRadialIndex;
            const std::uint32_t i2 = nextRingBase + radialIndex;
            const std::uint32_t i3 = nextRingBase + nextRadialIndex;
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
            mesh.indices.push_back(i3);
        }
    }

    return mesh;
}

} // namespace cressim::neo::examples::helpers

#endif
