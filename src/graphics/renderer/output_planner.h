#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_OUTPUT_PLANNER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_OUTPUT_PLANNER_H

#include "gpu/gpu_render_target_system.h"
#include "graphics/renderer.h"
#include "graphics/renderer/passes/render_pass_types.h"
#include "graphics/renderer/render_target_cache_key.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cressim::neo::graphics::detail
{

struct CameraOutputPlanningResult
{
    std::vector<ResolvedCameraView> resolvedCameras{};
    std::optional<DisplayResolveRequest> displayResolve{};
    std::unordered_set<RenderTargetFamilyKey, RenderTargetFamilyKeyHasher> usedManagedFamilies{};
};

CameraOutputPlanningResult planCameraOutputs(
    const std::vector<CameraData> &cameras, const gpu::GpuEntitySceneView &gpuScene,
    gpu::GpuRenderTargetSystem &renderTargetSystem,
    const gpu::GpuRenderTargetDesc &defaultTargetDesc,
    const gpu::GpuRenderTargetBinding &defaultTargetBinding, bool hasDefaultTarget,
    const RenderFrameOptions &options,
    std::unordered_map<RenderTargetFamilyKey, gpu::GpuRenderTargetHandle,
                       RenderTargetFamilyKeyHasher> &managedPrimaryTargets,
    RenderStats &inOutStats);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_OUTPUT_PLANNER_H
