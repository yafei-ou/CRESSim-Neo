#ifndef CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_QUEUE_BUILDER_H
#define CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_QUEUE_BUILDER_H

#include "gpu/gpu_scene.h"
#include "graphics/host_scene.h"
#include "graphics/passes/forward_draw_types.h"

#include <vector>

namespace cressim::neo::graphics
{

class RenderResourceManager;

namespace detail
{

struct LocalShadowCommand
{
    ForwardDrawCommand drawCommand{};
    std::uint32_t shadowViewIndex = 0u;
    std::uint32_t drawListOffset  = 0u;
    std::uint32_t instanceCount   = 0u;
};

struct LocalShadowBuildResult
{
    std::vector<gpu::GpuLocalShadowView> shadowViews;
    std::vector<gpu::GpuLightShadowAssignment> lightAssignments;
    std::vector<gpu::GpuVisiblePairInstance> visiblePairs;
    std::vector<LocalShadowCommand> commands;
    std::vector<std::uint32_t> shadowViewCommandOffsets;
    std::vector<std::uint32_t> shadowViewCommandCounts;
    std::uint32_t local2DLayerCount = 0u;
    std::uint32_t pointLayerCount   = 0u;
};

LocalShadowBuildResult buildLocalShadowData(const HostSceneView &sceneView,
                                            RenderResourceManager &resourceManager);

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_LOCAL_SHADOW_QUEUE_BUILDER_H
