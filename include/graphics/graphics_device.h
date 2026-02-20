#ifndef CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
#define CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H

#include "common/frame_context.h"
#include "common/id.h"
#include "graphics/export.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cressim::neo::graphics
{

enum class GraphicsBackend
{
    Null,
    Vulkan,
};

enum class RenderTargetDepthFormat
{
    D32Float,
};

enum class RenderTargetUpdateResult
{
    Failed,
    Unchanged,
    Recreated,
};

struct RenderTargetHandle
{
    // Opaque per-device handle for an offscreen target.
    common::ResourceId id = common::kInvalidResourceId;
};

struct RenderTargetDesc
{
    // Zero means "use the current default target size".
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool color = true;
    bool depth = true;
    // TEX_FORMAT_UNKNOWN means "auto".
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    RenderTargetDepthFormat depthFormat = RenderTargetDepthFormat::D32Float;
    // Enables sampling this target in later shader passes.
    bool shaderReadable = true;
    // Enables readback request tracking for this target.
    bool cpuReadback = false;
    std::string debugName;
};

struct GraphicsDeviceDesc
{
    struct PresentationDesc
    {
        bool enabled = false;
        // Passed to ISwapChain::Present(). 1 = v-sync on, 0 = v-sync off.
        std::uint32_t syncInterval = 1;
        // TEX_FORMAT_UNKNOWN lets the backend choose.
        Diligent::TEXTURE_FORMAT preferredColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

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

    GraphicsBackend preferredBackend = GraphicsBackend::Vulkan;
    bool enableValidation = true;
    RenderTargetDesc defaultRenderTargetDesc{};
    PresentationDesc presentation{};
    // Optional override for runtime shader source directory.
    // If empty, the engine resolves its default search paths.
    std::string shaderDirectory;
};

struct RenderViewport
{
    // Normalized coordinates [0..1] relative to the bound target.
    float x = 0.0f;
    float y = 0.0f;
    float width = 1.0f;
    float height = 1.0f;
};

struct RenderPassBeginDesc
{
    bool clearColor = true;
    bool clearDepth = true;
    float clearColorValue[4] = {0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue = 1.0f;
};

struct RenderTargetReadbackEvent
{
    RenderTargetHandle target{};
    std::uint64_t frameIndex = 0;
    // Format of color payload when available.
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    // Optional 4-channel 8-bit payload copied from the target color buffer.
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t rowStrideBytes = 0;
    std::vector<std::uint8_t> colorBytes{};
};

struct RenderTargetReadbackRequest
{
    // Opaque per-device handle for a queued readback request.
    std::uint64_t id = 0;
};

class CRESSIM_NEO_GRAPHICS_API GraphicsDevice
{
public:
    virtual ~GraphicsDevice() = default;

    virtual bool initialize(const GraphicsDeviceDesc& desc) = 0;
    virtual void shutdown() = 0;

    // Per-target management API (for multi-camera and GPU-only processing chains).
    virtual RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) = 0;
    virtual RenderTargetUpdateResult resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) = 0;
    // Recreates target resources from an updated descriptor while preserving handle identity.
    virtual RenderTargetUpdateResult reconfigureRenderTarget(RenderTargetHandle target, const RenderTargetDesc& desc) = 0;
    virtual void destroyRenderTarget(RenderTargetHandle target) = 0;
    virtual bool isValidRenderTarget(RenderTargetHandle target) const = 0;
    virtual bool tryGetRenderTargetDesc(RenderTargetHandle target, RenderTargetDesc& outDesc) const = 0;
    // Built-in fallback target used when a camera has no explicit output target.
    virtual RenderTargetHandle defaultRenderTarget() const = 0;

    virtual void beginFrame(const common::FrameContext& frameContext) = 0;
    // Sets viewport state used for the next beginRenderTarget/endRenderTarget pair on this target.
    virtual void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) = 0;
    virtual void beginRenderTarget(
        RenderTargetHandle target,
        const common::FrameContext& frameContext,
        const RenderPassBeginDesc& beginDesc) = 0;
    virtual void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) = 0;
    // Queues a target readback request that completes after a subsequent render pass of that target.
    virtual RenderTargetReadbackRequest requestRenderTargetReadback(RenderTargetHandle target) = 0;
    // Polls completion for a specific request id, returning payload when available.
    virtual bool tryGetRenderTargetReadback(RenderTargetReadbackRequest request, RenderTargetReadbackEvent& outEvent) = 0;
    virtual void endFrame(const common::FrameContext& frameContext) = 0;

    virtual GraphicsBackend backend() const = 0;
};

CRESSIM_NEO_GRAPHICS_API std::unique_ptr<GraphicsDevice> createGraphicsDevice();

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
