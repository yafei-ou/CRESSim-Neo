#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_H

#include "common/frame_context.h"
#include "gpu/export.h"
#include "gpu/gpu_render_target_system.h"
#include "gpu/gpu_types.h"

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

    virtual void beginFrame(const common::FrameContext& frameContext) = 0;
    virtual void endFrame(const common::FrameContext& frameContext)   = 0;
    virtual GpuRenderTargetSystem& renderTargetSystem()               = 0;

    virtual GpuBackend backend() const                               = 0;
    virtual bool tryGetBackendContext(GpuBackendContext& outContext) = 0;
    virtual const std::string& shaderSourceDirectory() const         = 0;
};

CRESSIM_NEO_GPU_API std::unique_ptr<GpuDevice> createGpuDevice();

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_H
