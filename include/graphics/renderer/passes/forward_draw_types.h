#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H

#include "common/id.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>

namespace cressim::neo::graphics
{

constexpr std::uint32_t kShadowCascadeCount  = 4;
constexpr std::uint32_t kShadowMapResolution = 2048;

struct ForwardDirectionalLightData
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    float intensity = 1.0f;
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float shadowDistance     = 120.0f;
    float shadowFadeDistance = 20.0f;
};

struct ForwardDrawCommand
{
    std::uint32_t instanceIndex               = 0xffffffffu;
    std::uint32_t drawListOffset              = 0u;
    std::uint32_t useDrawListBuffer           = 0u;
    std::uint32_t reserved0                   = 0u;
    MaterialProgramFamily programFamily       = MaterialProgramFamily::StandardLit;
    MaterialFeatureFlags materialFeatureFlags = MaterialFeatureFlags::None;
    common::ResourceId meshId                 = common::kInvalidResourceId;
    common::ResourceId materialId             = common::kInvalidResourceId;
    std::uint64_t meshVersion                 = 0;
    std::uint32_t indexCount                  = 0u;
    std::uint32_t reserved1                   = 0u;
};

struct IndirectCommandRegistryEntry
{
    ForwardDrawCommand drawCommand{};
    std::uint32_t maxVisibleCount = 0u;
    std::uint32_t reserved0       = 0u;
    std::uint32_t reserved1       = 0u;
    std::uint32_t reserved2       = 0u;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_DRAW_TYPES_H
