#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_H

#include "common/frame_context.h"
#include "gpu/export.h"
#include "gpu/gpu_render_target_system.h"
#include "gpu/gpu_types.h"
#include "gpu/shader_source_provider.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Shader.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cressim::neo::gpu
{

enum class VulkanShaderCompilerMode
{
    Auto,
    ForceDefault,
    ForceDXC,
};

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
        void *nativeWindow           = nullptr;
        std::uint64_t nativeWindowId = 0;
        void *nativeDisplay          = nullptr;
        void *nativeConnection       = nullptr;
    };

    GpuBackend preferredBackend = GpuBackend::Vulkan;
    bool enableValidation       = true;
    GpuRenderTargetDesc defaultRenderTargetDesc{};
    PresentationDesc presentation{};
    VulkanShaderCompilerMode vulkanShaderCompilerMode = VulkanShaderCompilerMode::Auto;
    // Optional override for the engine shader package root. If empty, the engine resolves its
    // build-time and runtime defaults.
    std::string shaderDirectory;
    // Additional, ordered include roots. Engine headers remain available from
    // <shaderDirectory>/include; these roots are intended for application-owned shader headers.
    std::vector<std::filesystem::path> shaderIncludeDirectories;
};

class CRESSIM_NEO_GPU_API GpuDevice
{
public:
    virtual ~GpuDevice() = default;

    virtual bool initialize(const GpuDeviceDesc &desc) = 0;
    virtual void shutdown()                            = 0;

    virtual void beginFrame(const common::FrameContext &frameContext) = 0;
    virtual void endFrame(const common::FrameContext &frameContext)   = 0;
    virtual GpuRenderTargetSystem &renderTargetSystem()               = 0;

    virtual GpuBackend backend() const                                               = 0;
    virtual bool tryGetGraphicsBackendContext(GpuGraphicsBackendContext &outContext) = 0;
    virtual bool tryGetPhysicsBackendContext(GpuComputeBackendContext &outContext)   = 0;
    virtual bool waitForPhysicsOnGraphics()                                          = 0;
    virtual bool waitForGraphicsOnPhysics()                                          = 0;
    virtual bool tryGetDefaultRenderTargetDesc(GpuRenderTargetDesc &outDesc) const   = 0;
    virtual bool tryGetPresentationTargetDesc(GpuPresentationTargetDesc &outDesc)    = 0;
    virtual GpuPresentationReadbackRequest requestPresentationReadback()             = 0;
    virtual bool tryGetPresentationReadback(GpuPresentationReadbackRequest request,
                                            GpuPresentationReadbackEvent &outEvent)  = 0;
    virtual bool supportsNativePhysicsFloatAtomics() const                           = 0;
    // Legacy single-root accessor retained for custom GpuDevice implementations.
    virtual const std::string &shaderSourceDirectory() const                         = 0;
    virtual ShaderSourceConfig shaderSourceConfig() const
    {
        ShaderSourceConfig config{};
        config.sourceDirectory = shaderSourceDirectory();
        return config;
    }
    virtual bool createShader(const Diligent::ShaderCreateInfo &createInfo,
                              Diligent::IShader **shader) = 0;
    virtual bool createGraphicsPipelineState(
        const Diligent::GraphicsPipelineStateCreateInfo &createInfo,
        Diligent::IPipelineState **pipelineState) = 0;
    virtual bool createComputePipelineState(
        const Diligent::ComputePipelineStateCreateInfo &createInfo,
        Diligent::IPipelineState **pipelineState) = 0;
};

CRESSIM_NEO_GPU_API std::unique_ptr<GpuDevice> createGpuDevice();

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_H
