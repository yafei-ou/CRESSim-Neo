#ifndef CRESSIM_NEO_GPU_GPU_TYPES_H
#define CRESSIM_NEO_GPU_GPU_TYPES_H

#include "common/id.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/GraphicsTypes.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cressim::neo::gpu
{

enum class GpuBackend
{
    Null,
    Vulkan,
};

enum class GpuRenderTargetUpdateResult
{
    Failed,
    Unchanged,
    Recreated,
};

enum class GpuContextRole
{
    Graphics,
    Physics,
};

struct GpuRenderTargetHandle
{
    // Opaque per-device handle for an offscreen target.
    common::ResourceId id = common::kInvalidResourceId;
};

struct GpuRenderTargetDesc
{
    // Zero means "use the current default target size".
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    bool color                           = true;
    bool depth                           = true;
    // TEX_FORMAT_UNKNOWN means "auto".
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
    // Enables sampling this target in later shader passes.
    bool shaderReadable                  = true;
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
    float clearColorValue[4] = {0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue    = 1.0f;
};

struct GpuRenderTargetReadbackEvent
{
    GpuRenderTargetHandle target{};
    std::uint64_t frameIndex             = 0;
    // Format of color payload when available.
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    // Optional 4-channel 8-bit payload copied from the target color buffer.
    std::uint32_t width                  = 0;
    std::uint32_t height                 = 0;
    std::uint32_t rowStrideBytes         = 0;
    std::vector<std::uint8_t> colorBytes{};
};

struct GpuRenderTargetReadbackRequest
{
    // Opaque per-device handle for a queued readback request.
    std::uint64_t id = 0;
};

struct GpuBackendContext
{
    Diligent::IRenderDevice* renderDevice                  = nullptr;
    Diligent::IDeviceContext* immediateContext             = nullptr;
    common::ResourceId activeRenderTargetId                = common::kInvalidResourceId;
    bool hasActiveRenderTarget                             = false;
    bool activeRenderTargetHasDepth                        = false;
    Diligent::TEXTURE_FORMAT activeRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
};

struct GpuComputeBackendContext
{
    Diligent::IRenderDevice* renderDevice    = nullptr;
    Diligent::IDeviceContext* computeContext = nullptr;
    std::uint32_t contextId                  = 0;
    Diligent::COMMAND_QUEUE_TYPE queueType   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    GpuContextRole role                      = GpuContextRole::Physics;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_TYPES_H
