#ifndef CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H
#define CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H

#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl final : public GraphicsDevice
{
public:
    struct VulkanBackendContext
    {
        Diligent::IRenderDevice* renderDevice = nullptr;
        Diligent::IDeviceContext* immediateContext = nullptr;
        common::ResourceId activeRenderTargetId = common::kInvalidResourceId;
        bool hasActiveRenderTarget = false;
        bool activeRenderTargetHasDepth = false;
        Diligent::TEXTURE_FORMAT activeRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    };

    bool initialize(const GraphicsDeviceDesc& desc) override;
    void shutdown() override;

    void resizeDefaultRenderTarget(std::uint32_t width, std::uint32_t height) override;
    RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) override;
    bool resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) override;
    void destroyRenderTarget(RenderTargetHandle target) override;
    bool isValidRenderTarget(RenderTargetHandle target) const override;
    RenderTargetHandle defaultRenderTarget() const override;

    void beginFrame(const common::FrameContext& frameContext) override;
    void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) override;
    void beginRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override;
    void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override;
    void requestReadback(RenderTargetHandle target) override;
    bool tryPopReadbackEvent(RenderTargetReadbackEvent& outEvent) override;
    void endFrame(const common::FrameContext& frameContext) override;

    GraphicsBackend backend() const override;
    bool tryGetVulkanContext(VulkanBackendContext& outContext);
    const std::string& shaderSourceDirectory() const;
    bool allowShaderFallback() const;

private:
    struct RenderTargetResources
    {
        RenderTargetDesc desc{};
        RenderViewport viewport{};
        Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
        Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
    };

    struct PendingReadbackCopy
    {
        RenderTargetHandle target{};
        std::uint64_t frameIndex = 0;
        std::uint64_t fenceValue = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    };

    bool initializeVulkan();
    bool createDebugViewerSwapChain();
    bool createDefaultRenderTarget();
    RenderTargetDesc normalizeTargetDesc(const RenderTargetDesc& desc) const;
    bool createRenderTargetTextures(const RenderTargetDesc& desc, RenderTargetResources& resources);

    bool queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex);

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;
    bool mHasActiveRenderTarget = false;
    bool mActiveRenderTargetHasDepth = false;
    Diligent::TEXTURE_FORMAT mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    common::ResourceId mNextRenderTargetId = 1;
    RenderTargetHandle mDefaultRenderTarget{};
    RenderTargetHandle mActiveRenderTarget{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    // Targets that requested readback and are waiting for endRenderTarget().
    std::unordered_set<common::ResourceId> mPendingReadbacks;
    // GPU->CPU copy jobs collected during render target completion and consumed in endFrame().
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    // FIFO completion metadata consumed through tryPopReadbackEvent().
    std::deque<RenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    std::uint64_t mNextReadbackFenceValue = 1;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mSwapChain;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H
