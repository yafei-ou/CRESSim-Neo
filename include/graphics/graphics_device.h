#ifndef CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
#define CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H

#include "common/frame_context.h"
#include "common/id.h"
#include "graphics/export.h"

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

struct GraphicsDeviceDesc
{
    GraphicsBackend preferredBackend = GraphicsBackend::Vulkan;
    bool enableValidation = true;
    std::uint32_t initialWidth = 1280;
    std::uint32_t initialHeight = 720;
    // Optional override for runtime shader source directory.
    // If empty, the engine resolves its default search paths.
    std::string shaderDirectory;
    // Allows using embedded fallback sources when shader files are unavailable.
    bool allowShaderFallback = true;
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
    // Enables sampling this target in later shader passes.
    bool shaderReadable = true;
    // Enables readback request tracking for this target.
    bool cpuReadback = false;
    std::string debugName;
};

struct RenderViewport
{
    // Normalized coordinates [0..1] relative to the bound target.
    float x = 0.0f;
    float y = 0.0f;
    float width = 1.0f;
    float height = 1.0f;
};

struct RenderTargetReadbackEvent
{
    RenderTargetHandle target{};
    std::uint64_t frameIndex = 0;
    // Optional RGBA8 payload copied from the target color buffer.
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t rowStrideBytes = 0;
    std::vector<std::uint8_t> colorRgba8{};
};

struct PbrMaterialData
{
    float baseColor[3] = {1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 0.5f;
};

struct PbrDirectionalLightData
{
    float direction[3] = {0.0f, -1.0f, 0.0f};
    float intensity = 1.0f;
    float color[3] = {1.0f, 1.0f, 1.0f};
    float _padding = 0.0f;
};

struct PbrDrawCommand
{
    // Stable id/version pair used by the backend for mesh buffer caches.
    common::ResourceId meshId = common::kInvalidResourceId;
    std::uint64_t meshVersion = 0;

    const void* vertexData = nullptr;
    std::uint32_t vertexCount = 0;
    std::uint32_t vertexStrideBytes = 0;

    const std::uint32_t* indexData = nullptr;
    std::uint32_t indexCount = 0;

    // Row-major 4x4 matrices.
    float modelMatrix[16] = {0.0f};
    float viewProjectionMatrix[16] = {0.0f};
    float cameraPosition[3] = {0.0f, 0.0f, 0.0f};
    float _padding0 = 0.0f;

    PbrMaterialData material{};
    PbrDirectionalLightData light{};
};

class CRESSIM_NEO_GRAPHICS_API IGraphicsDevice
{
public:
    virtual ~IGraphicsDevice() = default;

    virtual bool initialize(const GraphicsDeviceDesc& desc) = 0;
    virtual void shutdown() = 0;

    // Resizes the built-in fallback target used when a camera has no explicit output target.
    virtual void resizeDefaultRenderTarget(std::uint32_t width, std::uint32_t height) = 0;
    // Explicit per-target management API (for multi-camera and GPU-only processing chains).
    virtual RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) = 0;
    virtual bool resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) = 0;
    virtual void destroyRenderTarget(RenderTargetHandle target) = 0;
    virtual bool isValidRenderTarget(RenderTargetHandle target) const = 0;
    virtual RenderTargetHandle defaultRenderTarget() const = 0;

    virtual void beginFrame(const common::FrameContext& frameContext) = 0;
    // Sets viewport state used for the next beginRenderTarget/endRenderTarget pair on this target.
    virtual void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) = 0;
    virtual void beginRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) = 0;
    virtual bool drawPbr(RenderTargetHandle target, const PbrDrawCommand& drawCommand) = 0;
    virtual void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) = 0;
    // Requests a CPU-facing completion event for this target once rendering is done.
    virtual void requestReadback(RenderTargetHandle target) = 0;
    // Pops readback completion metadata and optional color payload.
    virtual bool tryPopReadbackEvent(RenderTargetReadbackEvent& outEvent) = 0;
    virtual void endFrame(const common::FrameContext& frameContext) = 0;

    virtual GraphicsBackend backend() const = 0;
};

CRESSIM_NEO_GRAPHICS_API std::unique_ptr<IGraphicsDevice> createGraphicsDevice();

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
