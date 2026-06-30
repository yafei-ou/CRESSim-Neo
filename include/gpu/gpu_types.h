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

namespace cressim::neo::gpu
{

constexpr Diligent::Uint64 contextMaskForId(std::uint32_t contextId) noexcept
{
    return static_cast<Diligent::Uint64>(1ull) << contextId;
}

enum class GpuBackend
{
    Null,
    D3D12,
    Vulkan,
};

enum class GpuRenderTargetUpdateResult
{
    Failed,
    Unchanged,
    Recreated,
};

struct GpuRenderTargetHandle
{
    // Opaque per-device handle for an offscreen target.
    common::ResourceId id = common::kInvalidResourceId;
};

struct GpuRenderTargetBinding
{
    GpuRenderTargetHandle target{};
    std::uint32_t firstLayer = 0u;
    std::uint32_t layerCount = 1u;

    bool isValid() const noexcept
    {
        return target.id != common::kInvalidResourceId && layerCount > 0u;
    }

    bool operator==(const GpuRenderTargetBinding &rhs) const noexcept
    {
        return target.id == rhs.target.id && firstLayer == rhs.firstLayer &&
               layerCount == rhs.layerCount;
    }
};

enum class GpuRenderTargetTexturePlane
{
    Color,
    Depth,
};

enum class RenderOutputMode
{
    ManagedPrimary,
    ExplicitSurface,
};

struct RenderOutputBinding
{
    RenderOutputMode mode = RenderOutputMode::ManagedPrimary;
    GpuRenderTargetBinding binding{};
};

struct GpuRenderTargetDesc
{
    // Zero means "use current presentation size if present,
    // otherwise fall back to 1280x720".
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    std::uint32_t arraySize              = 1;
    bool color                           = true;
    bool depth                           = true;
    bool layeredRendering                = false;
    // TEX_FORMAT_UNKNOWN means "auto".
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
    // Enables sampling this target in later shader passes.
    bool shaderReadable                  = true;
    // Enables unordered-access views for color textures used by compute passes.
    bool unorderedAccess                 = false;
    std::string debugName;
};

struct GpuRenderViewport
{
    // Normalized coordinates [0..1] relative to the bound target.
    float x      = 0.0f;
    float y      = 0.0f;
    float width  = 1.0f;
    float height = 1.0f;
};

struct GpuRenderPassBeginDesc
{
    bool clearColor          = true;
    bool clearDepth          = true;
    float clearColorValue[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float clearDepthValue    = 1.0f;
};

struct GpuRenderTargetReadbackEvent
{
    GpuRenderTargetBinding binding{};
    std::uint64_t frameIndex             = 0;
    // Color payload when the target exposes a color attachment.
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    std::uint32_t colorWidth             = 0;
    std::uint32_t colorHeight            = 0;
    std::uint32_t colorRowStrideBytes    = 0;
    std::vector<std::uint8_t> colorBytes{};
    // Legacy aliases for the color payload.
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    std::uint32_t rowStrideBytes         = 0;
    // Depth payload when the target exposes a depth attachment.
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
    std::uint32_t depthWidth             = 0;
    std::uint32_t depthHeight            = 0;
    std::uint32_t depthRowStrideBytes    = 0;
    std::vector<std::uint8_t> depthBytes{};
};

struct GpuRenderTargetReadbackRequest
{
    // Opaque per-device handle for a queued readback request.
    std::uint64_t id = 0;
};

struct GpuPresentationTargetDesc
{
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
    bool hasDepth                        = false;
};

struct GpuPresentationReadbackEvent
{
    std::uint64_t frameIndex             = 0;
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    std::uint32_t rowStrideBytes         = 0;
    std::vector<std::uint8_t> colorBytes{};
};

struct GpuPresentationReadbackRequest
{
    std::uint64_t id = 0;
};

struct GpuGraphicsBackendContext
{
    Diligent::IRenderDevice *renderDevice     = nullptr;
    Diligent::IDeviceContext *graphicsContext = nullptr;
    Diligent::ISwapChain *primarySwapChain    = nullptr;
    std::uint32_t contextId                   = 0u;
    Diligent::COMMAND_QUEUE_TYPE queueType    = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    GpuRenderTargetBinding activeRenderTargetBinding{};
    bool hasActiveRenderTarget                             = false;
    bool activeRenderTargetHasDepth                        = false;
    Diligent::TEXTURE_FORMAT activeRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
};

struct GpuComputeBackendContext
{
    Diligent::IRenderDevice *renderDevice    = nullptr;
    Diligent::IDeviceContext *computeContext = nullptr;
    std::uint32_t contextId                  = 0;
    Diligent::COMMAND_QUEUE_TYPE queueType   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_TYPES_H
