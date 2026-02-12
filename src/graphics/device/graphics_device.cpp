#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngineVulkan/interface/EngineFactoryVk.h"

#include <algorithm>
#include <memory>

namespace cressim::neo::graphics
{

namespace
{

std::uint32_t clampExtent(std::uint32_t value)
{
    return std::max<std::uint32_t>(value, 1u);
}

class DiligentGraphicsDevice final : public IGraphicsDevice
{
public:
    bool initialize(const GraphicsDeviceDesc& desc) override
    {
        shutdown();

        mDesc = desc;
        mDesc.initialWidth = clampExtent(mDesc.initialWidth);
        mDesc.initialHeight = clampExtent(mDesc.initialHeight);

        if (mDesc.preferredBackend == GraphicsBackend::Null)
        {
            mBackend = GraphicsBackend::Null;
            mInitialized = true;
            return true;
        }

        // Vulkan-only backend path.
        if (mDesc.preferredBackend != GraphicsBackend::Vulkan)
        {
            return false;
        }

        if (!initializeVulkan())
        {
            shutdown();
            return false;
        }

        mInitialized = true;
        return true;
    }

    void shutdown() override
    {
        mDepthTexture = nullptr;
        mColorTexture = nullptr;
        mImmediateContext = nullptr;
        mRenderDevice = nullptr;

        mBackend = GraphicsBackend::Null;
        mInitialized = false;
    }

    void resize(std::uint32_t width, std::uint32_t height) override
    {
        mDesc.initialWidth = clampExtent(width);
        mDesc.initialHeight = clampExtent(height);

        if (!mInitialized || mBackend != GraphicsBackend::Vulkan)
        {
            return;
        }

        (void)createHeadlessTargets();
    }

    void beginFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;

        if (!mInitialized || mBackend != GraphicsBackend::Vulkan)
        {
            return;
        }

        if (!mImmediateContext || !mColorTexture || !mDepthTexture)
        {
            return;
        }

        Diligent::ITextureView* colorRtv = mColorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET);
        Diligent::ITextureView* depthDsv = mDepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL);
        if (colorRtv == nullptr || depthDsv == nullptr)
        {
            return;
        }

        mImmediateContext->SetRenderTargets(1, &colorRtv, depthDsv, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

        constexpr float clearColor[4] = {0.02f, 0.02f, 0.03f, 1.0f};
        mImmediateContext->ClearRenderTarget(colorRtv, clearColor, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        mImmediateContext->ClearDepthStencil(depthDsv, Diligent::CLEAR_DEPTH_FLAG, 1.0f, 0, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    void endFrame(const common::FrameContext& frameContext) override
    {
        (void)frameContext;

        if (!mInitialized || mBackend != GraphicsBackend::Vulkan || !mImmediateContext)
        {
            return;
        }

        mImmediateContext->Flush();
        mImmediateContext->FinishFrame();
    }

    GraphicsBackend backend() const override
    {
        return mBackend;
    }

private:
    bool initializeVulkan()
    {
        Diligent::IEngineFactoryVk* factoryVk = Diligent::LoadAndGetEngineFactoryVk();
        if (factoryVk == nullptr)
        {
            return false;
        }

        Diligent::EngineVkCreateInfo engineCreateInfo{};
        engineCreateInfo.EnableValidation = static_cast<Diligent::Bool>(mDesc.enableValidation ? 1 : 0);
        factoryVk->CreateDeviceAndContextsVk(engineCreateInfo, &mRenderDevice, &mImmediateContext);
        if (!mRenderDevice || !mImmediateContext)
        {
            return false;
        }

        if (!createHeadlessTargets())
        {
            return false;
        }

        mBackend = GraphicsBackend::Vulkan;
        return true;
    }

    bool createHeadlessTargets()
    {
        if (!mRenderDevice)
        {
            return false;
        }

        Diligent::TextureDesc colorDesc{};
        colorDesc.Name = "CRESSimNeo.Headless.Color";
        colorDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        colorDesc.Width = mDesc.initialWidth;
        colorDesc.Height = mDesc.initialHeight;
        colorDesc.MipLevels = 1;
        colorDesc.ArraySize = 1;
        colorDesc.Format = Diligent::TEX_FORMAT_RGBA8_UNORM;
        colorDesc.BindFlags = Diligent::BIND_RENDER_TARGET;
        colorDesc.Usage = Diligent::USAGE_DEFAULT;

        Diligent::RefCntAutoPtr<Diligent::ITexture> newColorTexture;
        mRenderDevice->CreateTexture(colorDesc, nullptr, &newColorTexture);
        if (!newColorTexture || newColorTexture->GetDefaultView(Diligent::TEXTURE_VIEW_RENDER_TARGET) == nullptr)
        {
            return false;
        }

        Diligent::TextureDesc depthDesc{};
        depthDesc.Name = "CRESSimNeo.Headless.Depth";
        depthDesc.Type = Diligent::RESOURCE_DIM_TEX_2D;
        depthDesc.Width = mDesc.initialWidth;
        depthDesc.Height = mDesc.initialHeight;
        depthDesc.MipLevels = 1;
        depthDesc.ArraySize = 1;
        depthDesc.Format = Diligent::TEX_FORMAT_D32_FLOAT;
        depthDesc.BindFlags = Diligent::BIND_DEPTH_STENCIL;
        depthDesc.Usage = Diligent::USAGE_DEFAULT;

        Diligent::RefCntAutoPtr<Diligent::ITexture> newDepthTexture;
        mRenderDevice->CreateTexture(depthDesc, nullptr, &newDepthTexture);
        if (!newDepthTexture || newDepthTexture->GetDefaultView(Diligent::TEXTURE_VIEW_DEPTH_STENCIL) == nullptr)
        {
            return false;
        }

        mColorTexture = std::move(newColorTexture);
        mDepthTexture = std::move(newDepthTexture);
        return true;
    }

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::ITexture> mColorTexture;
    Diligent::RefCntAutoPtr<Diligent::ITexture> mDepthTexture;
};
} // namespace

std::unique_ptr<IGraphicsDevice> createGraphicsDevice()
{
    return std::make_unique<DiligentGraphicsDevice>();
}

} // namespace cressim::neo::graphics
