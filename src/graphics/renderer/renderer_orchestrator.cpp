#include "graphics/renderer.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/renderer_internal.h"

#include <unordered_map>

namespace cressim::neo::graphics
{

Renderer::Renderer(GraphicsDevice& device, RenderResourceManager& resourceManager,
                   const RendererDesc& desc)
    : mDevice(device), mResourceManager(resourceManager), mDesc(desc)
{
}

Renderer::~Renderer() = default;

bool Renderer::initialize()
{
    GraphicsDeviceImpl& deviceImpl = static_cast<GraphicsDeviceImpl&>(mDevice);
    mForwardPipeline               = std::make_unique<detail::ForwardPipeline>(deviceImpl);
    if (!mForwardPipeline->initialize())
    {
        mForwardPipeline.reset();
        return false;
    }

    mInitialized = true;
    return mInitialized;
}

RenderStats Renderer::render(const common::FrameContext& frameContext, const RenderWorld& world)
{
    RenderStats stats{};

    if (!mInitialized)
    {
        return stats;
    }

    mDevice.beginFrame(frameContext);

    const auto& renderables = world.renderables();
    const auto preparedRenderables =
        detail::buildPreparedRenderables(renderables, mResourceManager);
    const ForwardDirectionalLightData lightData = detail::buildMainLight(world.directionalLights());

    stats.renderableCount      = static_cast<std::uint32_t>(renderables.size());
    stats.validRenderableCount = static_cast<std::uint32_t>(preparedRenderables.size());
    stats.lightCount           = static_cast<std::uint32_t>(world.directionalLights().size());

    std::vector<CameraData> cameras = detail::sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(detail::defaultCamera());
    }

    struct RequestedExtent
    {
        std::uint32_t width  = 0;
        std::uint32_t height = 0;
    };
    std::unordered_map<common::ResourceId, RequestedExtent> requestedExtents;

    const auto renderCamera = [&](const CameraData& camera)
    {
        RenderTargetHandle target = camera.outputTarget;
        if (!mDevice.isValidRenderTarget(target))
        {
            target = mDevice.defaultRenderTarget();
        }
        if (!mDevice.isValidRenderTarget(target))
        {
            return;
        }

        RenderTargetDesc targetDesc{};
        if (!mDevice.tryGetRenderTargetDesc(target, targetDesc))
        {
            return;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            ++stats.renderTargetResizeRequests;

            RequestedExtent desired{};
            desired.width  = (camera.outputWidth == 0 ? targetDesc.width : camera.outputWidth);
            desired.height = (camera.outputHeight == 0 ? targetDesc.height : camera.outputHeight);

            const auto requestedIt = requestedExtents.find(target.id);
            if (requestedIt == requestedExtents.end())
            {
                requestedExtents.emplace(target.id, desired);
            }
            else
            {
                const bool conflict = requestedIt->second.width != desired.width ||
                                      requestedIt->second.height != desired.height;
                if (conflict)
                {
                    ++stats.renderTargetResizeConflicts;
                }
                desired = requestedIt->second;
            }

            if (targetDesc.width != desired.width || targetDesc.height != desired.height)
            {
                const RenderTargetUpdateResult updateResult =
                    mDevice.resizeRenderTarget(target, desired.width, desired.height);
                if (updateResult == RenderTargetUpdateResult::Unchanged)
                {
                    ++stats.renderTargetResizeNoOps;
                }
                else if (updateResult == RenderTargetUpdateResult::Recreated)
                {
                    ++stats.renderTargetRecreateCount;
                }
                if (updateResult != RenderTargetUpdateResult::Failed)
                {
                    if (!mDevice.tryGetRenderTargetDesc(target, targetDesc))
                    {
                        return;
                    }
                }
            }
            else
            {
                ++stats.renderTargetResizeNoOps;
            }
        }

        const RenderViewport viewport = detail::normalizeViewport(camera.viewport);
        const FrameViewData frameView =
            detail::buildFrameViewData(camera, targetDesc, target, viewport, lightData);
        const CameraRenderQueues queues = detail::buildCameraRenderQueues(
            preparedRenderables, frameView, mResourceManager, stats);

        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->execute(frameContext, frameView, queues, passStats);
        }

        stats.opaqueDrawCalls += passStats.opaqueDrawCalls;
        stats.shadowDrawCalls += passStats.shadowDrawCalls;
        stats.transparentDrawCalls += passStats.transparentDrawCalls;
        ++stats.cameraCount;
    };

    for (const CameraData& camera : cameras)
    {
        renderCamera(camera);
    }

    stats.drawCalls = stats.opaqueDrawCalls + stats.shadowDrawCalls + stats.transparentDrawCalls;

    mDevice.endFrame(frameContext);
    return stats;
}

} // namespace cressim::neo::graphics
