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
class DebugViewPresenter;
class ForwardPipeline;
}

struct RendererDesc
{
    struct DebugViewerDesc
    {
        bool enabled = false;
        // Passed to ISwapChain::Present(). 1 = v-sync on, 0 = v-sync off.
        std::uint32_t syncInterval = 1;

        // Platform-native window handles.
        // Win32: nativeWindow = HWND
        // Linux/X11: nativeWindowId = Window, nativeDisplay = Display*
        // Linux/XCB: nativeWindowId = xcb_window_t, nativeConnection = xcb_connection_t*
        // macOS: nativeWindow = NSView*
        void* nativeWindow = nullptr;
        std::uint64_t nativeWindowId = 0;
        void* nativeDisplay = nullptr;
        void* nativeConnection = nullptr;
    };

    DebugViewerDesc debugViewer{};
};

struct RenderStats
{
    // Current counters are framework-level instrumentation, not GPU timestamps.
    std::uint32_t drawCalls = 0;
    std::uint32_t opaqueDrawCalls = 0;
    std::uint32_t shadowDrawCalls = 0;
    std::uint32_t transparentDrawCalls = 0;
    std::uint32_t renderableCount = 0;
    std::uint32_t validRenderableCount = 0;
    std::uint32_t culledRenderableCount = 0;
    std::uint32_t opaqueQueueCount = 0;
    std::uint32_t shadowCasterQueueCount = 0;
    std::uint32_t transparentQueueCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t cameraCount = 0;
};

class CRESSIM_NEO_GRAPHICS_API Renderer
{
public:
    Renderer(GraphicsDevice& device, RenderResourceManager& resourceManager, const RendererDesc& desc = RendererDesc{});
    ~Renderer();

    bool initialize();
    RenderStats render(const common::FrameContext& frameContext, const RenderWorld& world);

private:
    GraphicsDevice& mDevice;
    RenderResourceManager& mResourceManager;
    RendererDesc mDesc{};
    std::unique_ptr<detail::ForwardPipeline> mForwardPipeline;
    std::unique_ptr<detail::DebugViewPresenter> mDebugViewPresenter;
    bool mInitialized = false;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_H
