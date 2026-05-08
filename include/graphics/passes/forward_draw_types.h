#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H

#include "common/id.h"
#include "graphics/render_resource_manager.h"

#include <cstdint>

namespace cressim::neo::graphics
{

constexpr std::uint32_t kShadowCascadeCount  = 4;
constexpr std::uint32_t kShadowMapResolution = 2048;

struct ForwardDrawCommand
{
    std::uint32_t instanceIndex               = 0xffffffffu;
    std::uint32_t drawListOffset              = 0u;
    std::uint32_t useDrawListBuffer           = 0u;
    MaterialProgramFamily programFamily       = MaterialProgramFamily::StandardLit;
    MaterialFeatureFlags materialFeatureFlags = MaterialFeatureFlags::None;
    common::ResourceId meshId                 = common::kInvalidResourceId;
    common::ResourceId materialId             = common::kInvalidResourceId;
    std::uint64_t meshVersion                 = 0;
    std::uint32_t indexCount                  = 0u;
};

struct IndirectCommandRegistryEntry
{
    ForwardDrawCommand drawCommand{};
    std::uint32_t maxVisibleCount = 0u;
};

struct TransparentDrawEntry
{
    ForwardDrawCommand drawCommand{};
    std::uint32_t objectIndex = 0xffffffffu;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H
