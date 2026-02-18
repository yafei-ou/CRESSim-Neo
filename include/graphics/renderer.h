#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_H

#include "common/frame_context.h"
#include "graphics/export.h"
#include "graphics/graphics_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/render_world.h"

#include <cstdint>
#include <memory>

namespace cressim::neo::graphics
{

namespace detail
{
class ForwardPipeline;
}

struct RenderStats
{
    // Current counters are framework-level instrumentation, not GPU timestamps.
    std::uint32_t drawCalls = 0;
    std::uint32_t renderableCount = 0;
    std::uint32_t validRenderableCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t cameraCount = 0;
    std::uint32_t readbackRequests = 0;
};

class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    Renderer(GraphicsDevice& device, RenderResourceManager& resourceManager);
    ~Renderer();

    bool initialize();
    RenderStats render(const common::FrameContext& frameContext, const RenderWorld& world);

private:
    GraphicsDevice& mDevice;
    RenderResourceManager& mResourceManager;
    std::unique_ptr<detail::ForwardPipeline> mForwardPipeline;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_H
