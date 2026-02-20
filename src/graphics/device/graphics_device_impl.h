#ifndef CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H
#define CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H

#include "graphics/graphics_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <unordered_map>
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

    RenderTargetHandle createRenderTarget(const RenderTargetDesc& desc) override;
    RenderTargetUpdateResult resizeRenderTarget(RenderTargetHandle target, std::uint32_t width, std::uint32_t height) override;
    RenderTargetUpdateResult reconfigureRenderTarget(RenderTargetHandle target, const RenderTargetDesc& desc) override;
    void destroyRenderTarget(RenderTargetHandle target) override;
    bool isValidRenderTarget(RenderTargetHandle target) const override;
    bool tryGetRenderTargetDesc(RenderTargetHandle target, RenderTargetDesc& outDesc) const override;
    RenderTargetHandle defaultRenderTarget() const override;

    void beginFrame(const common::FrameContext& frameContext) override;
    void setRenderTargetViewport(RenderTargetHandle target, const RenderViewport& viewport) override;
    void beginRenderTarget(
        RenderTargetHandle target,
        const common::FrameContext& frameContext,
        const RenderPassBeginDesc& beginDesc) override;
    void endRenderTarget(RenderTargetHandle target, const common::FrameContext& frameContext) override;
    RenderTargetReadbackRequest requestRenderTargetReadback(RenderTargetHandle target) override;
    bool tryGetRenderTargetReadback(RenderTargetReadbackRequest request, RenderTargetReadbackEvent& outEvent) override;
    void endFrame(const common::FrameContext& frameContext) override;

    GraphicsBackend backend() const override;
    bool tryGetVulkanContext(VulkanBackendContext& outContext);
    bool tryGetRenderTargetColorTexture(RenderTargetHandle target, Diligent::ITexture*& outTexture);
    bool tryGetRenderTargetDepthTexture(RenderTargetHandle target, Diligent::ITexture*& outTexture);
    const std::string& shaderSourceDirectory() const;

private:
    struct RenderTargetResources
    {
        RenderTargetDesc desc{};
        RenderViewport viewport{};
        RenderTargetColorFormat colorFormat = RenderTargetColorFormat::Rgba8Unorm;
        RenderTargetDepthFormat depthFormat = RenderTargetDepthFormat::D32Float;
        Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
        Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
    };

    struct PendingReadbackCopy
    {
        std::vector<std::uint64_t> requestIds{};
        RenderTargetHandle target{};
        std::uint64_t frameIndex = 0;
        std::uint64_t fenceValue = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        RenderTargetColorFormat colorFormat = RenderTargetColorFormat::Rgba8Unorm;
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    };

    bool initializeVulkan();
    bool createDefaultRenderTarget();
    RenderTargetDesc normalizeDefaultRenderTargetDesc(const RenderTargetDesc& desc) const;
    RenderTargetDesc normalizeTargetDesc(const RenderTargetDesc& desc) const;
    bool createRenderTargetTextures(const RenderTargetDesc& desc, RenderTargetResources& resources);

    bool queueReadbackCopy(RenderTargetHandle target, std::uint64_t frameIndex, const std::vector<std::uint64_t>& requestIds);

private:
    GraphicsDeviceDesc mDesc{};
    GraphicsBackend mBackend = GraphicsBackend::Null;
    bool mInitialized = false;
    bool mHasActiveRenderTarget = false;
    bool mActiveRenderTargetHasDepth = false;
    Diligent::TEXTURE_FORMAT mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    common::ResourceId mNextRenderTargetId = 1;
    std::uint64_t mNextReadbackRequestId = 1;
    RenderTargetHandle mDefaultRenderTarget{};
    RenderTargetHandle mActiveRenderTarget{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    // Per-target readback request ids waiting for the next completed render pass of that target.
    std::unordered_map<common::ResourceId, std::vector<std::uint64_t>> mPendingReadbackRequests;
    // GPU->CPU copy jobs collected during render target completion and consumed in endFrame().
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    // Completed results consumed through tryGetRenderTargetReadback().
    std::unordered_map<std::uint64_t, RenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    std::uint64_t mNextReadbackFenceValue = 1;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_DEVICE_GRAPHICS_DEVICE_IMPL_H
