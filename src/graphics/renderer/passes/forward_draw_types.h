#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H

#include "common/id.h"

#include <cstdint>

namespace cressim::neo::graphics
{

enum class ForwardShadingModel
{
    Pbr,
    Phong,
    BlinnPhong,
};

struct ForwardMaterialData
{
    float baseColor[3] = {1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;
};

struct ForwardDirectionalLightData
{
    float direction[3] = {0.0f, -1.0f, 0.0f};
    float intensity = 1.0f;
    float color[3] = {1.0f, 1.0f, 1.0f};
    float _padding = 0.0f;
};

struct ForwardDrawCommand
{
    ForwardShadingModel shadingModel = ForwardShadingModel::Pbr;
    // Stable id/version pair used by pass-level mesh buffer caches.
    common::ResourceId meshId = common::kInvalidResourceId;
    std::uint64_t meshVersion = 0;

    const void* vertexData = nullptr;
    std::uint32_t vertexCount = 0;
    std::uint32_t vertexStrideBytes = 0;

    const std::uint32_t* indexData = nullptr;
    std::uint32_t indexCount = 0;

    // Row-major 4x4 matrices.
    float modelMatrix[16] = {0.0f};
    float viewProjectionMatrix[16] = {0.0f};
    float cameraPosition[3] = {0.0f, 0.0f, 0.0f};
    float _padding0 = 0.0f;

    ForwardMaterialData material{};
    ForwardDirectionalLightData light{};
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
