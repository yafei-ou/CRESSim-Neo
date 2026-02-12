#ifndef CRESSIM_NEO_ENGINE_RUNTIME_H
#define CRESSIM_NEO_ENGINE_RUNTIME_H

#include "common/frame_context.h"
#include "engine/export.h"
#include "engine/world.h"
#include "graphics/graphics_device.h"
#include "graphics/renderer.h"
#include "graphics/scene.h"

#include <memory>

namespace cressim::neo::engine
{

struct RuntimeConfig
{
    graphics::GraphicsDeviceDesc graphics{};
};

class CRESSIM_NEO_ENGINE_API Runtime
{
public:
    bool initialize(const RuntimeConfig& config = RuntimeConfig{});
    void shutdown();

    void tick(const common::FrameContext& frameContext);

    World& getWorld() noexcept;
    const World& getWorld() const noexcept;

    graphics::Scene& getScene() noexcept;
    const graphics::Scene& getScene() const noexcept;

private:
    void syncWorldToRenderWorld();

    bool mInitialized = false;
    std::unique_ptr<graphics::IGraphicsDevice> mGraphicsDevice;
    std::unique_ptr<graphics::Renderer> mRenderer;
    World mWorld;
    graphics::Scene mScene;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_RUNTIME_H
