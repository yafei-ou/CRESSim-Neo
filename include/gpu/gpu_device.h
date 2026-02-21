#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_H

#include "common/frame_context.h"
#include "gpu/export.h"
#include "gpu/gpu_types.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <memory>
#include <string>

namespace cressim::neo::gpu
{

struct GpuDeviceDesc
{
    struct PresentationDesc
    {
        bool enabled                                  = false;
        // Passed to ISwapChain::Present(). 1 = v-sync on, 0 = v-sync off.
        std::uint32_t syncInterval                    = 1;
        // TEX_FORMAT_UNKNOWN lets the backend choose.
        Diligent::TEXTURE_FORMAT preferredColorFormat = Diligent::TEX_FORMAT_UNKNOWN;

        // Platform-native window handles.
        // Win32: nativeWindow = HWND
        // Linux/X11: nativeWindowId = Window, nativeDisplay = Display*
        // Linux/XCB: nativeWindowId = xcb_window_t, nativeConnection = xcb_connection_t*
        // macOS: nativeWindow = NSView*
        void* nativeWindow           = nullptr;
        std::uint64_t nativeWindowId = 0;
        void* nativeDisplay          = nullptr;
        void* nativeConnection       = nullptr;
    };

    GpuBackend preferredBackend = GpuBackend::Vulkan;
    bool enableValidation       = true;
    GpuRenderTargetDesc defaultRenderTargetDesc{};
    PresentationDesc presentation{};
    // Optional override for runtime shader source directory.
    // If empty, the engine resolves its default search paths.
    std::string shaderDirectory;
};

class CRESSIM_NEO_GPU_API GpuDevice
{
public:
    virtual ~GpuDevice() = default;

    virtual bool initialize(const GpuDeviceDesc& desc) = 0;
    virtual void shutdown()                            = 0;

    // Per-target management API (for multi-camera and GPU-only processing chains).
    virtual GpuRenderTargetHandle createRenderTarget(const GpuRenderTargetDesc& desc) = 0;
    virtual GpuRenderTargetUpdateResult resizeRenderTarget(GpuRenderTargetHandle target,
                                                           std::uint32_t width,
                                                           std::uint32_t height)      = 0;
    // Recreates target resources from an updated descriptor while preserving handle identity.
    virtual GpuRenderTargetUpdateResult reconfigureRenderTarget(
        GpuRenderTargetHandle target, const GpuRenderTargetDesc& desc)      = 0;
    virtual void destroyRenderTarget(GpuRenderTargetHandle target)          = 0;
    virtual bool isValidRenderTarget(GpuRenderTargetHandle target) const    = 0;
    virtual bool tryGetRenderTargetDesc(GpuRenderTargetHandle target,
                                        GpuRenderTargetDesc& outDesc) const = 0;
    // Built-in fallback target used when a camera has no explicit output target.
    virtual GpuRenderTargetHandle defaultRenderTarget() const               = 0;

    virtual void beginFrame(const common::FrameContext& frameContext)       = 0;
    // Sets viewport state used for the next beginRenderTarget/endRenderTarget pair on this target.
    virtual void setRenderTargetViewport(GpuRenderTargetHandle target,
                                         const GpuRenderViewport& viewport) = 0;
    virtual void beginRenderTarget(GpuRenderTargetHandle target,
                                   const common::FrameContext& frameContext,
                                   const GpuRenderPassBeginDesc& beginDesc) = 0;
    virtual void endRenderTarget(GpuRenderTargetHandle target,
                                 const common::FrameContext& frameContext)  = 0;
    // Queues a target readback request that completes after a subsequent render pass of that
    // target.
    virtual GpuRenderTargetReadbackRequest requestRenderTargetReadback(
        GpuRenderTargetHandle target)                                               = 0;
    // Polls completion for a specific request id, returning payload when available.
    virtual bool tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                            GpuRenderTargetReadbackEvent& outEvent) = 0;
    virtual void endFrame(const common::FrameContext& frameContext)                 = 0;

    virtual GpuBackend backend() const                                           = 0;
    virtual bool tryGetBackendContext(GpuBackendContext& outContext)             = 0;
    virtual bool tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture*& outTexture) = 0;
    virtual bool tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture*& outTexture) = 0;
    virtual const std::string& shaderSourceDirectory() const                     = 0;
};

CRESSIM_NEO_GPU_API std::unique_ptr<GpuDevice> createGpuDevice();

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_H
