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

/// @file gpu_device.h
/// @brief Hardware device abstraction interface, presentation configuration, and shader/pipeline state factories.

namespace cressim::neo::gpu
{

/// @brief Compiler front-end selection mode when compiling HLSL shaders targeting Vulkan SPIR-V.
enum class VulkanShaderCompilerMode
{
    Auto,         ///< Automatically select DXC if available, falling back to glslang.
    ForceDefault, ///< Force legacy default compiler (glslang).
    ForceDXC,     ///< Force DirectX Shader Compiler (DXC).
};

/// @brief Configuration descriptor for initializing the GPU device and window presentation system.
struct GpuDeviceDesc
{
    /// @brief Window swapchain and display presentation configuration.
    struct PresentationDesc
    {
        bool enabled                                  = false; ///< Whether to create and attach a presentation swapchain.
        std::uint32_t syncInterval                    = 1;     ///< Vertical synchronization interval (1 = V-Sync enabled, 0 = immediate presentation).
        Diligent::TEXTURE_FORMAT preferredColorFormat = Diligent::TEX_FORMAT_UNKNOWN; ///< Preferred swapchain surface color format (`TEX_FORMAT_UNKNOWN` chooses default).

        void *nativeWindow           = nullptr; ///< Platform native window pointer (`HWND` on Windows, `NSView*` on macOS).
        std::uint64_t nativeWindowId = 0;       ///< Platform native window ID (`Window` for X11, `xcb_window_t` for XCB).
        void *nativeDisplay          = nullptr; ///< Platform native display connection (`Display*` for X11).
        void *nativeConnection       = nullptr; ///< Platform native connection (`xcb_connection_t*` for XCB).
    };

    GpuBackend preferredBackend = GpuBackend::Vulkan; ///< Requested graphics API backend.
    bool enableValidation       = true;               ///< Enable graphics API debug and validation layers.
    GpuRenderTargetDesc defaultRenderTargetDesc{};    ///< Default descriptor applied to offscreen rendering passes.
    PresentationDesc presentation{};                  ///< Presentation window and swapchain settings.
    VulkanShaderCompilerMode vulkanShaderCompilerMode = VulkanShaderCompilerMode::Auto; ///< Shader compiler choice for Vulkan.
    std::string shaderDirectory;                      ///< Root directory for shader source assets (empty resolves build-time default).
    std::vector<std::filesystem::path> shaderIncludeDirectories; ///< Additional ordered search paths for HLSL `#include` headers.
};

/// @brief Hardware graphics and compute device interface managing resources, contexts, and pipelines.
class CRESSIM_NEO_GPU_API GpuDevice
{
public:
    /// @brief Virtual destructor.
    virtual ~GpuDevice() = default;

    /// @brief Initializes the hardware device, creates graphics contexts, and initializes swapchain if enabled.
    /// @param desc Initialization descriptor.
    /// @return True on success, false on failure.
    virtual bool initialize(const GpuDeviceDesc &desc) = 0;
    /// @brief Destroys all GPU resources and cleanly shuts down the graphics backend.
    virtual void shutdown()                            = 0;

    /// @brief Begins a new rendering and simulation frame.
    /// @param frameContext Temporal frame context state.
    virtual void beginFrame(const common::FrameContext &frameContext) = 0;
    /// @brief Ends the current frame, executing presentation and queuing asynchronous readbacks.
    /// @param frameContext Temporal frame context state.
    virtual void endFrame(const common::FrameContext &frameContext)   = 0;
    /// @brief Retrieves the offscreen render target management system.
    /// @return Reference to GpuRenderTargetSystem.
    virtual GpuRenderTargetSystem &renderTargetSystem()               = 0;

    /// @brief Retrieves the active graphics API backend enum.
    /// @return GpuBackend.
    virtual GpuBackend backend() const                                               = 0;
    /// @brief Retrieves the hardware graphics backend context and command queue bundle.
    /// @param outContext Output reference to populate.
    /// @return True if graphics context is available.
    virtual bool tryGetGraphicsBackendContext(GpuGraphicsBackendContext &outContext) = 0;
    /// @brief Retrieves the hardware compute backend context dedicated to physics simulation.
    /// @param outContext Output reference to populate.
    /// @return True if physics compute context is available.
    virtual bool tryGetPhysicsBackendContext(GpuComputeBackendContext &outContext)   = 0;
    /// @brief Inserts a GPU barrier or semaphore wait synchronizing graphics commands after physics compute commands.
    /// @return True on success.
    virtual bool waitForPhysicsOnGraphics()                                          = 0;
    /// @brief Inserts a GPU barrier or semaphore wait synchronizing physics compute commands after graphics commands.
    /// @return True on success.
    virtual bool waitForGraphicsOnPhysics()                                          = 0;
    /// @brief Queries the default render target descriptor configured on this device.
    /// @param outDesc Output descriptor to populate.
    /// @return True on success.
    virtual bool tryGetDefaultRenderTargetDesc(GpuRenderTargetDesc &outDesc) const   = 0;
    /// @brief Queries the geometry and format of the active presentation swapchain.
    /// @param outDesc Output descriptor to populate.
    /// @return True if presentation swapchain is active.
    virtual bool tryGetPresentationTargetDesc(GpuPresentationTargetDesc &outDesc)    = 0;
    /// @brief Enqueues an asynchronous readback request for the primary presentation frame.
    /// @return Tracking readback request handle.
    virtual GpuPresentationReadbackRequest requestPresentationReadback()             = 0;
    /// @brief Attempts to retrieve downloaded pixel data for a previously enqueued presentation readback request.
    /// @param request Tracking handle.
    /// @param outEvent Output readback event data to populate.
    /// @return True if readback is complete and data was copied.
    virtual bool tryGetPresentationReadback(GpuPresentationReadbackRequest request,
                                            GpuPresentationReadbackEvent &outEvent)  = 0;
    /// @brief Checks whether the hardware device supports native floating-point atomic operations in compute shaders.
    /// @return True if native float atomics supported.
    virtual bool supportsNativePhysicsFloatAtomics() const                           = 0;
    /// @brief Retrieves the root directory path containing entry-point shaders.
    /// @return String reference to shader directory path.
    virtual const std::string &shaderSourceDirectory() const                         = 0;
    /// @brief Retrieves the active shader source and include paths configuration.
    /// @return ShaderSourceConfig struct.
    virtual ShaderSourceConfig shaderSourceConfig() const
    {
        ShaderSourceConfig config{};
        config.sourceDirectory = shaderSourceDirectory();
        return config;
    }
    /// @brief Compiles and creates a shader object.
    /// @param createInfo Shader compilation descriptor.
    /// @param shader Output pointer to receive created shader.
    /// @return True on success.
    virtual bool createShader(const Diligent::ShaderCreateInfo &createInfo,
                              Diligent::IShader **shader) = 0;
    /// @brief Creates a graphics pipeline state object (PSO).
    /// @param createInfo Pipeline state creation descriptor.
    /// @param pipelineState Output pointer to receive created PSO.
    /// @return True on success.
    virtual bool createGraphicsPipelineState(
        const Diligent::GraphicsPipelineStateCreateInfo &createInfo,
        Diligent::IPipelineState **pipelineState) = 0;
    /// @brief Creates a compute pipeline state object (PSO).
    /// @param createInfo Compute pipeline creation descriptor.
    /// @param pipelineState Output pointer to receive created compute PSO.
    /// @return True on success.
    virtual bool createComputePipelineState(
        const Diligent::ComputePipelineStateCreateInfo &createInfo,
        Diligent::IPipelineState **pipelineState) = 0;
};

/// @brief Factory function creating an uninitialized GpuDevice instance.
/// @return Unique pointer to GpuDevice.
CRESSIM_NEO_GPU_API std::unique_ptr<GpuDevice> createGpuDevice();

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_H
