#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_DEBUG_VIEW_PRESENTER_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_DEBUG_VIEW_PRESENTER_H

#include "graphics/device/graphics_device_impl.h"
#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

namespace cressim::neo::graphics::detail
{

class DebugViewPresenter
{
public:
    DebugViewPresenter(GraphicsDeviceImpl& device, const RendererDesc::DebugViewerDesc& desc);

    bool initialize();
    bool present(RenderTargetHandle sourceTarget);

private:
    bool createSwapChain(std::uint32_t width, std::uint32_t height, Diligent::TEXTURE_FORMAT colorFormat);

private:
    GraphicsDeviceImpl& mDevice;
    RendererDesc::DebugViewerDesc mDesc{};
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mSwapChain;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_SERVICES_DEBUG_VIEW_PRESENTER_H
