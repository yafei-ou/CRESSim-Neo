#include "graphics/renderer.h"

#include <algorithm>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{

float clamp01(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    RenderViewport normalized{};
    normalized.x = clamp01(viewport.x);
    normalized.y = clamp01(viewport.y);
    normalized.width = clamp01(viewport.width);
    normalized.height = clamp01(viewport.height);

    const float maxWidth = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width = std::min(normalized.width, maxWidth);
    normalized.height = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y = 0.0f;
    }

    return normalized;
}

std::uint32_t countValidRenderables(const std::vector<RenderableInstance>& renderables, const RenderResourceManager& resources)
{
    std::uint32_t validCount = 0;
    for (const RenderableInstance& renderable : renderables)
    {
        if (!resources.isValid(renderable.mesh))
        {
            continue;
        }
        if (!resources.isValid(renderable.material))
        {
            continue;
        }
        ++validCount;
    }
    return validCount;
}

} // namespace

Renderer::Renderer(IGraphicsDevice& device, RenderResourceManager& resourceManager) :
    mDevice(device),
    mResourceManager(resourceManager)
{
}

bool Renderer::initialize()
{
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
    stats.renderableCount = static_cast<std::uint32_t>(renderables.size());
    stats.validRenderableCount = countValidRenderables(renderables, mResourceManager);
    stats.lightCount = static_cast<std::uint32_t>(world.directionalLights().size());

    std::vector<CameraData> cameras = world.cameras();
    // Keep render order deterministic across runs/platforms.
    std::sort(cameras.begin(), cameras.end(), [](const CameraData& lhs, const CameraData& rhs) {
        if (lhs.renderOrder != rhs.renderOrder)
        {
            return lhs.renderOrder < rhs.renderOrder;
        }
        return lhs.entityId < rhs.entityId;
    });

    const auto renderCamera = [&](const CameraData& camera) {
        RenderTargetHandle target = camera.outputTarget;
        if (!mDevice.isValidRenderTarget(target))
        {
            target = mDevice.defaultRenderTarget();
        }
        if (!mDevice.isValidRenderTarget(target))
        {
            return;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            (void)mDevice.resizeRenderTarget(target, camera.outputWidth, camera.outputHeight);
        }

        mDevice.setRenderTargetViewport(target, normalizeViewport(camera.viewport));

        if (camera.requestReadback)
        {
            mDevice.requestReadback(target);
            ++stats.readbackRequests;
        }

        mDevice.beginRenderTarget(target, frameContext);
        // Temporary vertical slice: each valid renderable emits one debug triangle draw.
        for (std::uint32_t i = 0; i < stats.validRenderableCount; ++i)
        {
            if (mDevice.drawDebugTriangle(target))
            {
                ++stats.drawCalls;
            }
        }
        mDevice.endRenderTarget(target, frameContext);

        ++stats.cameraCount;
    };

    if (cameras.empty())
    {
        // Scene without explicit cameras still renders to the default target.
        renderCamera(CameraData{});
    }
    else
    {
        for (const CameraData& camera : cameras)
        {
            renderCamera(camera);
        }
    }

    mDevice.endFrame(frameContext);

    return stats;
}

} // namespace cressim::neo::graphics
