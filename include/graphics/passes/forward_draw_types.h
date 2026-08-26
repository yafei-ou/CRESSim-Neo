#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H

#include "common/id.h"
#include "graphics/render_resource_manager.h"

#include <cstdint>

/// @file forward_draw_types.h
/// @brief Forward rendering draw commands, indirect command registry entries, and transparent
/// sorting records.

namespace cressim::neo::graphics
{

/// @brief Number of directional shadow map cascades.
constexpr std::uint32_t kShadowCascadeCount  = 4;
/// @brief Pixel resolution per shadow map tile or cascade slice.
constexpr std::uint32_t kShadowMapResolution = 2048;

/// @brief Low-level forward rasterization draw command issued to the graphics hardware.
struct ForwardDrawCommand
{
    std::uint32_t instanceIndex  = 0xffffffffu; ///< Base object index or draw call instance offset.
    std::uint32_t drawListOffset = 0u;          ///< Offset into indirect draw arguments buffer.
    std::uint32_t useDrawListBuffer =
        0u; ///< Flag indicating whether indirect draw list indexing is used.
    MaterialProgramFamily programFamily =
        MaterialProgramFamily::StandardLit; ///< Material shader program specialization.
    MaterialFeatureFlags materialFeatureFlags =
        MaterialFeatureFlags::None;                         ///< Compiled feature flags.
    common::ResourceId meshId = common::kInvalidResourceId; ///< Bound mesh resource identifier.
    common::ResourceId materialId =
        common::kInvalidResourceId; ///< Bound material resource identifier.
    std::uint64_t meshVersion = 0;  ///< Mesh registration version used to validate cached GPU data.
    std::uint32_t indexCount  = 0u; ///< Number of indices to draw.
};

/// @brief Entry in the batched indirect draw command registry for opaque and shadow passes.
struct IndirectCommandRegistryEntry
{
    ForwardDrawCommand drawCommand{}; ///< Underlying forward draw command template.
    std::uint32_t maxVisibleCount =
        0u; ///< Maximum number of visible instances allocated in GPU draw buffers.
};

/// @brief Sorted draw entry representing a transparent surface rendered back-to-front.
struct TransparentDrawEntry
{
    ForwardDrawCommand drawCommand{}; ///< Underlying forward draw command.
    std::int32_t renderOrder  = 0;    ///< Explicit user sort key within the transparent pass.
    std::uint32_t objectIndex = 0xffffffffu; ///< Object slot index in the scene table.
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_DRAW_TYPES_H
