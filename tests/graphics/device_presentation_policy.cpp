#include "engine/runtime.h"

#include <iostream>

namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::graphics::GraphicsBackend;
using cressim::neo::graphics::GraphicsDevice;
using cressim::neo::graphics::RenderTargetDesc;

} // namespace

int main()
{
    {
        RuntimeConfig config{};
        config.graphicsDeviceDesc.preferredBackend = GraphicsBackend::Null;
        config.graphicsDeviceDesc.presentation.enabled = false;
        config.graphicsDeviceDesc.defaultRenderTargetDesc.colorFormat = Diligent::TEX_FORMAT_UNKNOWN;

        Runtime runtime;
        if (!runtime.initialize(config))
        {
            std::cerr << "Runtime initialization failed for headless null backend.\n";
            return 1;
        }

        GraphicsDevice* device = runtime.getGraphicsDevice();
        if (device == nullptr)
        {
            std::cerr << "Graphics device not available.\n";
            runtime.shutdown();
            return 1;
        }

        RenderTargetDesc defaultDesc{};
        if (!device->tryGetRenderTargetDesc(device->defaultRenderTarget(), defaultDesc))
        {
            std::cerr << "Failed to query default render target descriptor.\n";
            runtime.shutdown();
            return 1;
        }
        runtime.shutdown();

        if (defaultDesc.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
        {
            std::cerr << "Expected auto default color format to resolve in headless mode.\n";
            return 1;
        }
    }

    {
        RuntimeConfig config{};
        config.graphicsDeviceDesc.preferredBackend = GraphicsBackend::Null;
        config.graphicsDeviceDesc.presentation.enabled = true;

        Runtime runtime;
        if (runtime.initialize(config))
        {
            std::cerr << "Expected runtime initialization to fail when presentation is enabled on null backend.\n";
            runtime.shutdown();
            return 1;
        }
    }

    std::cout << "Device presentation policy checks passed.\n";
    return 0;
}
