#ifndef CRESSIM_NEO_GRAPHICS_H
#define CRESSIM_NEO_GRAPHICS_H

#include "common/frame_context.h"
#include "common/id.h"
#include "gpu/gpu_device.h"
#include "graphics/export.h"
#include "graphics/host_scene.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics
{

namespace detail
{
class ForwardPipeline;
class DisplayResolvePass;
} // namespace detail

enum class ToneMapper : std::uint32_t
{
    Disabled = 0u,
    Reinhard = 1u,
    Filmic   = 2u,
};

struct RendererDesc
{
    IblQualityTier iblQualityTier = IblQualityTier::Off;
};

struct RenderFrameOptions
{
    struct DebugParticleOptions
    {
        bool enabled                    = false;
        Diligent::float4 color          = {0.2f, 0.8f, 1.0f, 1.0f};
        bool useParticleRadii           = true;
        float fallbackRadius            = 0.15f;
    };

    common::EntityId presentedCameraEntity = common::kInvalidEntityId;
    std::optional<gpu::GpuPresentationTargetDesc> presentationTarget{};
    ToneMapper toneMapper = ToneMapper::Reinhard;
    float exposure        = 1.0f;
    DebugParticleOptions debugParticles{};
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
    Renderer(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
             const RendererDesc &desc = RendererDesc{});
    ~Renderer();

    bool initialize();
    RenderStats render(const common::FrameContext &frameContext, const HostSceneView &sceneView,
                       const physics::PhysicsGpuSceneView *physicsScene,
                       const RenderFrameOptions &options = RenderFrameOptions{});

private:
    struct GpuScenePrepareState;
    bool ensureGpuScenePrepareState();
    bool prepareGpuScene(const GpuEntitySceneView &sceneView);

    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    RendererDesc mDesc{};
    std::unique_ptr<detail::ForwardPipeline> mForwardPipeline;
    std::unique_ptr<detail::DisplayResolvePass> mDisplayResolvePass;
    std::unique_ptr<GpuScenePrepareState> mGpuScenePrepare;
    std::unique_ptr<struct RendererOutputPlanningState> mOutputPlanningState;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_H
