#ifndef CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
#define CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H

#include "common/frame_context.h"
#include "graphics/export.h"

#include <cstdint>
#include <memory>

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
};

class CRESSIM_NEO_GRAPHICS_API IGraphicsDevice
{
public:
    virtual ~IGraphicsDevice() = default;

    virtual bool initialize(const GraphicsDeviceDesc& desc) = 0;
    virtual void shutdown() = 0;

    virtual void resize(std::uint32_t width, std::uint32_t height) = 0;
    virtual void beginFrame(const common::FrameContext& frameContext) = 0;
    virtual void endFrame(const common::FrameContext& frameContext) = 0;

    virtual GraphicsBackend backend() const = 0;
};

CRESSIM_NEO_GRAPHICS_API std::unique_ptr<IGraphicsDevice> createGraphicsDevice();

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_GRAPHICS_DEVICE_H
