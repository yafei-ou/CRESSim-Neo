#ifndef CRESSIM_NEO_GPU_GPU_TYPES_H
#define CRESSIM_NEO_GPU_GPU_TYPES_H

#include "common/id.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

#include <cstdint>
#include <string>
#include <vector>

/// @file gpu_types.h
/// @brief Fundamental GPU types, backend enumerations, render target descriptors, and
/// graphics/compute context views.

namespace cressim::neo::gpu
{

/// @brief Generates a 64-bit bitmask corresponding to a specific device context identifier.
/// @param contextId Zero-based context index.
/// @pre @p contextId is less than 64.
/// @return 64-bit bitmask with bit `contextId` set.
constexpr Diligent::Uint64 contextMaskForId(std::uint32_t contextId) noexcept
{
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

/// @brief Underlying graphics API backend implemented by the GPU device.
enum class GpuBackend
{
    Null,   ///< Null / mock headless backend for CPU-only testing.
    D3D12,  ///< Direct3D 12 backend (Windows only).
    Vulkan, ///< Vulkan 1.2+ cross-platform backend (Linux / Windows).
};

/// @brief Result status returned after updating or recreating offscreen render targets.
enum class GpuRenderTargetUpdateResult
{
    Failed,    ///< Allocation or configuration failed.
    Unchanged, ///< Existing render target matched descriptor; no recreation occurred.
    Recreated, ///< Render target dimensions, formats, or flags changed and textures were recreated.
};

/// @brief Opaque per-device resource handle for an allocated offscreen render target.
struct GpuRenderTargetHandle
{
    common::ResourceId id = common::kInvalidResourceId; ///< Unique numeric resource identifier.
};

/// @brief Target attachment binding specifying an offscreen target and optional 2D texture array
/// layer range.
struct GpuRenderTargetBinding
{
    GpuRenderTargetHandle target{}; ///< Bound render target resource handle.
    std::uint32_t firstLayer = 0u;  ///< Zero-based index of the first texture array layer.
    std::uint32_t layerCount = 1u;  ///< Number of consecutive texture array layers to bind.

    /// @brief Checks if this binding references a valid render target and non-zero layer count.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return target.id != common::kInvalidResourceId && layerCount > 0u;
    }

    /// @brief Checks equality between two render target bindings.
    /// @param rhs Right-hand side binding to compare.
    /// @return True if both handles and layer ranges match.
    bool operator==(const GpuRenderTargetBinding &rhs) const noexcept
    {
        return target.id == rhs.target.id && firstLayer == rhs.firstLayer &&
               layerCount == rhs.layerCount;
    }
};

/// @brief Identifies whether a texture operation targets the color or depth plane of a render
/// target.
enum class GpuRenderTargetTexturePlane
{
    Color, ///< Color attachment texture.
    Depth, ///< Depth/stencil attachment texture.
};

/// @brief Output routing mode for render passes.
enum class RenderOutputMode
{
    ManagedPrimary,  ///< Render ColorDepth output to a renderer-managed offscreen target; a chosen
                     ///< camera may then be resolved to the presentation swapchain.
    ExplicitSurface, ///< Render to an explicitly bound offscreen target surface.
};

/// @brief Composite binding pairing an output mode with its associated target binding.
struct RenderOutputBinding
{
    RenderOutputMode mode = RenderOutputMode::ManagedPrimary; ///< Routing destination mode.
    GpuRenderTargetBinding
        binding{}; ///< Offscreen target binding (used when mode == ExplicitSurface).
};

/// @brief Configuration descriptor for allocating offscreen 2D or 2D-array render targets.
struct GpuRenderTargetDesc
{
    std::uint32_t width  = 0; ///< Pixel width (0 uses 1280 for direct target allocation; managed
                              ///< camera outputs resolve it from their destination size).
    std::uint32_t height = 0; ///< Pixel height (0 uses 720 for direct target allocation; managed
                              ///< camera outputs resolve it from their destination size).
    std::uint32_t arraySize =
        1; ///< Number of 2D texture array slices (e.g. for multi-camera batched rendering).
    bool color = true; ///< Whether to allocate a color attachment texture.
    bool depth = true; ///< Whether to allocate a depth/stencil attachment texture.
    bool layeredRendering =
        false; ///< Enable single-pass multi-layer rendering via geometry/mesh shaders.
    Diligent::TEXTURE_FORMAT colorFormat =
        Diligent::TEX_FORMAT_UNKNOWN; ///< Color texture format (`TEX_FORMAT_UNKNOWN` uses default
                                      ///< RGBA16_FLOAT).
    Diligent::TEXTURE_FORMAT depthFormat =
        Diligent::TEX_FORMAT_D32_FLOAT; ///< Depth texture format.
    bool shaderReadable =
        true; ///< Enables shader resource views (SRVs) for sampling in subsequent passes.
    bool unorderedAccess =
        false;             ///< Enables unordered access views (UAVs) for compute shader writeback.
    std::string debugName; ///< Optional debug name assigned to the GPU texture objects.
};

/// @brief Normalized viewport rectangle defining the active rendering subregion.
struct GpuRenderViewport
{
    float x      = 0.0f; ///< Normalized horizontal origin [0..1] relative to the target width.
    float y      = 0.0f; ///< Normalized vertical origin [0..1] relative to the target height.
    float width  = 1.0f; ///< Normalized viewport width [0..1].
    float height = 1.0f; ///< Normalized viewport height [0..1].
};

/// @brief Clear values and flags applied when beginning a render pass.
struct GpuRenderPassBeginDesc
{
    bool clearColor          = true; ///< Whether to clear the color attachment upon begin.
    bool clearDepth          = true; ///< Whether to clear the depth attachment upon begin.
    float clearColorValue[4] = {0.0f, 0.0f, 0.0f, 1.0f}; ///< RGBA clear color values.
    float clearDepthValue    = 1.0f; ///< Depth clear value (typically 1.0 for standard depth).
};

/// @brief Event payload containing downloaded pixel data copied from an offscreen render target.
struct GpuRenderTargetReadbackEvent
{
    GpuRenderTargetBinding binding{}; ///< Source render target binding.
    std::uint64_t frameIndex = 0;     ///< Frame counter at the time of readback capture.
    Diligent::TEXTURE_FORMAT colorFormat =
        Diligent::TEX_FORMAT_UNKNOWN;       ///< Format of color pixel buffer.
    std::uint32_t colorWidth          = 0;  ///< Width of color image in pixels.
    std::uint32_t colorHeight         = 0;  ///< Height of color image in pixels.
    std::uint32_t colorRowStrideBytes = 0;  ///< Stride between consecutive rows in bytes.
    std::vector<std::uint8_t> colorBytes{}; ///< Raw downloaded color byte buffer.
    std::uint32_t width          = 0;       ///< Legacy alias for color width.
    std::uint32_t height         = 0;       ///< Legacy alias for color height.
    std::uint32_t rowStrideBytes = 0;       ///< Legacy alias for color row stride.
    Diligent::TEXTURE_FORMAT depthFormat =
        Diligent::TEX_FORMAT_UNKNOWN;       ///< Format of depth pixel buffer.
    std::uint32_t depthWidth          = 0;  ///< Width of depth image in pixels.
    std::uint32_t depthHeight         = 0;  ///< Height of depth image in pixels.
    std::uint32_t depthRowStrideBytes = 0;  ///< Row stride of depth buffer in bytes.
    std::vector<std::uint8_t> depthBytes{}; ///< Raw downloaded depth byte buffer.
};

/// @brief Tracking handle for an asynchronous offscreen render target readback request.
struct GpuRenderTargetReadbackRequest
{
    std::uint64_t id = 0; ///< Monotonically assigned request identifier.
};

/// @brief Geometry and format description of the primary presentation swapchain.
struct GpuPresentationTargetDesc
{
    std::uint32_t width  = 0; ///< Swapchain width in pixels.
    std::uint32_t height = 0; ///< Swapchain height in pixels.
    Diligent::TEXTURE_FORMAT colorFormat =
        Diligent::TEX_FORMAT_UNKNOWN; ///< Swapchain color format.
    Diligent::TEXTURE_FORMAT depthFormat =
        Diligent::TEX_FORMAT_UNKNOWN; ///< Swapchain depth format.
    bool hasDepth = false;            ///< Whether a presentation depth buffer is attached.
};

/// @brief Event payload containing pixel data captured from the primary presentation window
/// swapchain.
struct GpuPresentationReadbackEvent
{
    std::uint64_t frameIndex             = 0; ///< Frame index of capture.
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN; ///< Color texture format.
    std::uint32_t width                  = 0; ///< Framebuffer width in pixels.
    std::uint32_t height                 = 0; ///< Framebuffer height in pixels.
    std::uint32_t rowStrideBytes         = 0; ///< Byte stride per scanline row.
    std::vector<std::uint8_t> colorBytes{};   ///< Raw RGB/RGBA byte buffer.
};

/// @brief Tracking handle for an asynchronous presentation swapchain readback request.
struct GpuPresentationReadbackRequest
{
    std::uint64_t id = 0; ///< Monotonically assigned request identifier.
};

/// @brief Hardware render context bundle passed into graphics pipeline execution passes.
struct GpuGraphicsBackendContext
{
    Diligent::IRenderDevice *renderDevice     = nullptr; ///< Underlying Diligent graphics device.
    Diligent::IDeviceContext *graphicsContext = nullptr; ///< Active graphics device context.
    Diligent::ISwapChain *primarySwapChain    = nullptr; ///< Primary window presentation swapchain.
    std::uint32_t contextId                   = 0u;      ///< Zero-based context index.
    Diligent::COMMAND_QUEUE_TYPE queueType =
        Diligent::COMMAND_QUEUE_TYPE_UNKNOWN; ///< Hardware queue category.
    GpuRenderTargetBinding
        activeRenderTargetBinding{};         ///< Currently bound offscreen target binding.
    bool hasActiveRenderTarget      = false; ///< True if an offscreen target is bound.
    bool activeRenderTargetHasDepth = false; ///< True if bound target has a depth attachment.
    Diligent::TEXTURE_FORMAT activeRenderTargetColorFormat =
        Diligent::TEX_FORMAT_UNKNOWN; ///< Bound color format.
};

/// @brief Hardware compute context bundle passed into asynchronous or immediate compute passes.
struct GpuComputeBackendContext
{
    Diligent::IRenderDevice *renderDevice    = nullptr; ///< Underlying Diligent graphics device.
    Diligent::IDeviceContext *computeContext = nullptr; ///< Active compute device context.
    std::uint32_t contextId                  = 0;       ///< Zero-based context index.
    Diligent::COMMAND_QUEUE_TYPE queueType =
        Diligent::COMMAND_QUEUE_TYPE_UNKNOWN; ///< Hardware queue category.
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_TYPES_H
