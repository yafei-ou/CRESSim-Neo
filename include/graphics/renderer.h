#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_H

#include "common/frame_context.h"
#include "common/id.h"
#include "gpu/gpu_device.h"
#include "graphics/export.h"
#include "graphics/host_scene.h"
#include "graphics/render_resource_manager.h"

#include <cstdint>
#include <memory>

namespace cressim::neo::graphics
{

namespace detail
{
class ForwardPipeline;
class DisplayResolvePass;
}

struct RendererDesc
{
};

struct RenderFrameOptions
{
    common::EntityId presentedCameraEntity = common::kInvalidEntityId;
};

struct RenderStats
{
    // Current counters are framework-level instrumentation, not GPU timestamps.
    std::uint32_t drawCalls                   = 0;
    std::uint32_t opaqueDrawCalls             = 0;
    std::uint32_t shadowDrawCalls             = 0;
    std::uint32_t renderableCount             = 0;
    std::uint32_t lightCount                  = 0;
    std::uint32_t cameraCount                 = 0;
    std::uint32_t renderTargetResizeRequests  = 0;
    std::uint32_t renderTargetResizeNoOps     = 0;
    std::uint32_t renderTargetRecreateCount   = 0;
    std::uint32_t renderTargetResizeConflicts = 0;
};

class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    Renderer(gpu::GpuDevice& device, RenderResourceManager& resourceManager,
             const RendererDesc& desc = RendererDesc{});
    ~Renderer();

    bool initialize();
    RenderStats render(const common::FrameContext& frameContext, const HostSceneView& sceneView,
                       const RenderFrameOptions& options = RenderFrameOptions{});

private:
    struct GpuScenePrepareState;
    bool ensureGpuScenePrepareState();
    bool prepareGpuScene(const gpu::GpuEntitySceneView& sceneView);

    gpu::GpuDevice& mDevice;
    RenderResourceManager& mResourceManager;
    RendererDesc mDesc{};
    std::unique_ptr<detail::ForwardPipeline> mForwardPipeline;
    std::unique_ptr<detail::DisplayResolvePass> mDisplayResolvePass;
    std::unique_ptr<GpuScenePrepareState> mGpuScenePrepare;
    std::unique_ptr<struct RendererOutputPlanningState> mOutputPlanningState;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_H
