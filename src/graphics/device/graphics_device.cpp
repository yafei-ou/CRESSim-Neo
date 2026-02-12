#include "graphics/graphics_device.h"

#include <memory>

namespace cressim::neo::graphics
{

namespace
{

class NullGraphicsDevice final : public IGraphicsDevice
{
public:
    bool initialize(const GraphicsDeviceDesc& desc) override
    {
        mDesc = desc;
        mBackend = GraphicsBackend::Null;
        mInitialized = true;
        return true;
    }

    void shutdown() override
    {
        mInitialized = false;
    }

    void resize(std::uint32_t width, std::uint32_t height) override
    {
        mDesc.initialWidth = width;
        mDesc.initialHeight = height;
    }

    void beginFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;
    }

    void endFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;
    }

    GraphicsBackend backend() const override
    {
        return mBackend;
    }

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;
};

} // namespace

std::unique_ptr<IGraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<NullGraphicsDevice>();
}

} // namespace cressim::neo::graphics
