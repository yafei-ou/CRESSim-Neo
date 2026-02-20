#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H

#include "common/id.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>

namespace cressim::neo::graphics
{

constexpr std::uint32_t kShadowCascadeCount = 4;
constexpr std::uint32_t kShadowMapResolution = 2048;

struct ForwardMaterialData
{
    Diligent::float3 baseColor{1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float opacity = 1.0f;
    float alphaCutoff = 0.5f;
    float receivesShadows = 1.0f;
};

struct ForwardDirectionalLightData
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    float intensity = 1.0f;
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float shadowDistance = 120.0f;
    float shadowFadeDistance = 20.0f;
};

struct ForwardDrawCommand
{
    MaterialProgramFamily programFamily = MaterialProgramFamily::StandardLit;
    std::uint32_t materialFeatureFlags = MaterialFeature_None;
    // Stable id/version pair used by pass-level mesh buffer caches.
    common::ResourceId meshId = common::kInvalidResourceId;
    common::ResourceId materialId = common::kInvalidResourceId;
    std::uint64_t meshVersion = 0;

    const void* vertexData = nullptr;
    std::uint32_t vertexCount = 0;
    std::uint32_t vertexStrideBytes = 0;

    const std::uint32_t* indexData = nullptr;
    std::uint32_t indexCount = 0;

    Diligent::float4x4 modelMatrix = Diligent::float4x4::Identity();
    Diligent::float4x4 normalMatrix = Diligent::float4x4::Identity();

    ForwardMaterialData material{};
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
