#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/export.h"
#include "engine/world.h"
#include "graphics/graphics_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/render_world.h"
#include "graphics/renderer.h"

#include <memory>

namespace cressim::neo::engine
{

struct RuntimeConfig
{
    graphics::GraphicsDeviceDesc graphicsDeviceDesc{};
    graphics::RendererDesc rendererDesc{};
};

class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    bool initialize(const RuntimeConfig& config = RuntimeConfig{});
    void shutdown();

    void tick(const common::FrameContext& frameContext);

    World& getWorld() noexcept;
    const World& getWorld() const noexcept;

    // Direct access for explicit target creation and other low-level graphics control.
    graphics::GraphicsDevice* getGraphicsDevice() noexcept;
    const graphics::GraphicsDevice* getGraphicsDevice() const noexcept;
    // Queues a readback request for a target and returns a request handle.
    graphics::RenderTargetReadbackRequest requestRenderTargetReadback(graphics::RenderTargetHandle target);
    // Polls a specific readback request for completion metadata and optional payload.
    bool tryGetRenderTargetReadback(graphics::RenderTargetReadbackRequest request, graphics::RenderTargetReadbackEvent& outEvent);
    const graphics::RenderStats& lastRenderStats() const noexcept;

    graphics::RenderResourceManager& getResources() noexcept;
    const graphics::RenderResourceManager& getResources() const noexcept;

private:
    void syncWorldToRenderWorld();

    bool mInitialized = false;
    std::unique_ptr<graphics::GraphicsDevice> mGraphicsDevice;
    std::unique_ptr<graphics::Renderer> mRenderer;
    graphics::RenderStats mLastRenderStats{};
    World mWorld;
    graphics::RenderResourceManager mResources;
    graphics::RenderWorld mRenderWorld;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
