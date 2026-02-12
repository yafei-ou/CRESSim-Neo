#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_H

#include "common/frame_context.h"
#include "graphics/export.h"
#include "graphics/graphics_device.h"
#include "graphics/render_resource_manager.h"
#include "graphics/render_world.h"

#include <cstdint>

namespace cressim::neo::graphics
{

struct RenderStats
{
    std::uint32_t drawCalls = 0;
    std::uint32_t renderableCount = 0;
    std::uint32_t lightCount = 0;
};

class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    Renderer(IGraphicsDevice& device, RenderResourceManager& resourceManager);

    bool initialize();
    RenderStats render(const common::FrameContext& frameContext, const RenderWorld& world);

private:
    IGraphicsDevice& mDevice;
    RenderResourceManager& mResourceManager;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_H
