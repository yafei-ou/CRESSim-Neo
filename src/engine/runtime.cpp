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

    if (!mGraphicsDevice->initialize(config.graphics))
    {
        mGraphicsDevice.reset();
        return false;
    }

    mRenderer = std::make_unique<graphics::Renderer>(*mGraphicsDevice, mScene.resources());
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

    mInitialized = false;
}

void Runtime::tick(const common::FrameContext& frameContext)
{
    if (!mInitialized)
    {
        return;
    }

    syncWorldToRenderWorld();
    (void)mRenderer->render(frameContext, mScene.world());
}

World& Runtime::getWorld() noexcept
{
    return mWorld;
}

const World& Runtime::getWorld() const noexcept
{
    return mWorld;
}

graphics::IGraphicsDevice* Runtime::getGraphicsDevice() noexcept
{
    return mGraphicsDevice.get();
}

const graphics::IGraphicsDevice* Runtime::getGraphicsDevice() const noexcept
{
    return mGraphicsDevice.get();
}

bool Runtime::tryPopReadbackEvent(graphics::RenderTargetReadbackEvent& outEvent)
{
    if (!mGraphicsDevice)
    {
        return false;
    }
    return mGraphicsDevice->tryPopReadbackEvent(outEvent);
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
