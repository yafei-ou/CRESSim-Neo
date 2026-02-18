#include "engine/runtime.h"
#include "engine/world_to_render_world_sync.h"

namespace cressim::neo::engine
{

bool Runtime::initialize(const RuntimeConfig& config)
{
    if (mInitialized)
    {
        return true;
    }

    mGraphicsDevice = graphics::createGraphicsDevice();
    if (!mGraphicsDevice)
    {
        return false;
    }

    if (!mGraphicsDevice->initialize(config.graphicsDeviceDesc))
    {
        mGraphicsDevice.reset();
        return false;
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGraphicsDevice, mScene.resources(), config.rendererDesc);
    if (!mRenderer->initialize())
    {
        mRenderer.reset();
        mGraphicsDevice->shutdown();
        mGraphicsDevice.reset();
        return false;
    }

    mInitialized = true;
    return true;
}

void Runtime::shutdown()
{
    if (!mInitialized)
    {
        return;
    }

    mRenderer.reset();

    if (mGraphicsDevice)
    {
        mGraphicsDevice->shutdown();
        mGraphicsDevice.reset();
    }

    mLastRenderStats = {};
    mInitialized = false;
}

void Runtime::tick(const common::FrameContext& frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    syncWorldToRenderWorld();
    mLastRenderStats = mRenderer->render(frameContext, mScene.world());
}

World& Runtime::getWorld() noexcept
{
    return mWorld;
}

const World& Runtime::getWorld() const noexcept
{
    return mWorld;
}

graphics::GraphicsDevice* Runtime::getGraphicsDevice() noexcept
{
    return mGraphicsDevice.get();
}

const graphics::GraphicsDevice* Runtime::getGraphicsDevice() const noexcept
{
    return mGraphicsDevice.get();
}

graphics::RenderTargetReadbackRequest Runtime::requestRenderTargetReadback(graphics::RenderTargetHandle target)
{
    if (!mGraphicsDevice)
    {
        return {};
    }
    return mGraphicsDevice->requestRenderTargetReadback(target);
}

bool Runtime::tryGetRenderTargetReadback(graphics::RenderTargetReadbackRequest request, graphics::RenderTargetReadbackEvent& outEvent)
{
    if (!mGraphicsDevice)
    {
        return false;
    }
    return mGraphicsDevice->tryGetRenderTargetReadback(request, outEvent);
}

const graphics::RenderStats& Runtime::lastRenderStats() const noexcept
{
    return mLastRenderStats;
}

graphics::Scene& Runtime::getScene() noexcept
{
    return mScene;
}

const graphics::Scene& Runtime::getScene() const noexcept
{
    return mScene;
}

void Runtime::syncWorldToRenderWorld()
{
    detail::syncWorldToRenderWorld(mWorld, mScene.world());
}

} // namespace cressim::neo::engine
