#include "graphics/renderer.h"

namespace cressim::neo::graphics
{

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

    stats.drawCalls = static_cast<std::uint32_t>(world.renderables().size());
    stats.renderableCount = static_cast<std::uint32_t>(world.renderables().size());
    stats.lightCount = static_cast<std::uint32_t>(world.directionalLights().size());

    (void)mResourceManager;

    mDevice.endFrame(frameContext);

    return stats;
}

} // namespace cressim::neo::graphics
