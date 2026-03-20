#ifndef CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_IMPL_H
#define CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_IMPL_H

#include "gpu/gpu_render_target_system.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cressim::neo::gpu
{

class GpuRenderTargetSystemImpl final : public GpuRenderTargetSystem
{
public:
    bool initialize(const GpuRenderTargetDesc& defaultDesc, bool isVulkanBackend,
                    Diligent::IRenderDevice* renderDevice,
                    Diligent::IDeviceContext* immediateContext);
    void shutdown();
    void endFrame(const common::FrameContext& frameContext);

    void fillBackendContextState(GpuBackendContext& outContext) const;

    GpuRenderTargetHandle createRenderTarget(const GpuRenderTargetDesc& desc) override;
    GpuRenderTargetUpdateResult resizeRenderTarget(GpuRenderTargetHandle target,
                                                   std::uint32_t width,
                                                   std::uint32_t height) override;
    GpuRenderTargetUpdateResult reconfigureRenderTarget(GpuRenderTargetHandle target,
                                                        const GpuRenderTargetDesc& desc) override;
    void destroyRenderTarget(GpuRenderTargetHandle target) override;
    bool isValidRenderTarget(GpuRenderTargetHandle target) const override;
    bool tryGetRenderTargetDesc(GpuRenderTargetHandle target,
                                GpuRenderTargetDesc& outDesc) const override;
    GpuRenderTargetHandle defaultRenderTarget() const override;
    GpuRenderTargetBinding defaultRenderTargetBinding() const override;

    void setRenderTargetViewport(const GpuRenderTargetBinding& binding,
                                 const GpuRenderViewport& viewport) override;
    void beginRenderTarget(const GpuRenderTargetBinding& binding,
                           const common::FrameContext& frameContext,
                           const GpuRenderPassBeginDesc& beginDesc) override;
    void endRenderTarget(const GpuRenderTargetBinding& binding,
                         const common::FrameContext& frameContext) override;

    GpuRenderTargetReadbackRequest requestRenderTargetReadback(
        const GpuRenderTargetBinding& binding) override;
    bool tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                    GpuRenderTargetReadbackEvent& outEvent) override;
    bool tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                        Diligent::ITexture*& outTexture) override;
    bool tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                        Diligent::ITexture*& outTexture) override;

private:
    struct RenderTargetResources
    {
        GpuRenderTargetDesc desc{};
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
        Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
        Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
        std::unordered_map<std::uint64_t, GpuRenderViewport> viewports;
        std::unordered_map<std::uint64_t, Diligent::RefCntAutoPtr<Diligent::ITextureView>>
            colorRenderTargetViews;
        std::unordered_map<std::uint64_t, Diligent::RefCntAutoPtr<Diligent::ITextureView>>
            depthStencilViews;
    };

    struct PendingReadbackCopy
    {
        std::uint64_t requestId                = 0;
        GpuRenderTargetBinding binding{};
        std::uint64_t frameIndex             = 0;
        std::uint64_t fenceValue             = 0;
        std::uint32_t width                  = 0;
        std::uint32_t height                 = 0;
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    };

    GpuRenderTargetDesc normalizeDefaultRenderTargetDesc(const GpuRenderTargetDesc& desc) const;
    GpuRenderTargetDesc normalizeTargetDesc(const GpuRenderTargetDesc& desc) const;
    bool createRenderTargetTextures(const GpuRenderTargetDesc& desc,
                                    RenderTargetResources& resources);
    static std::uint64_t bindingKey(const GpuRenderTargetBinding& binding) noexcept;
    GpuRenderTargetBinding normalizeBinding(const GpuRenderTargetBinding& binding,
                                            const RenderTargetResources& resources) const;
    Diligent::ITextureView* getOrCreateRenderTargetView(RenderTargetResources& resources,
                                                        const GpuRenderTargetBinding& binding);
    Diligent::ITextureView* getOrCreateDepthStencilView(RenderTargetResources& resources,
                                                        const GpuRenderTargetBinding& binding);
    bool queueReadbackCopy(const GpuRenderTargetBinding& binding, std::uint64_t frameIndex,
                           std::uint64_t requestId);

private:
    bool mInitialized                                       = false;
    bool mIsVulkanBackend                                   = false;
    bool mHasActiveRenderTarget                             = false;
    bool mActiveRenderTargetHasDepth                        = false;
    Diligent::TEXTURE_FORMAT mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    common::ResourceId mNextRenderTargetId                  = 1;
    std::uint64_t mNextReadbackRequestId                    = 1;
    std::uint64_t mNextReadbackFenceValue                   = 1;

    GpuRenderTargetDesc mDefaultRenderTargetDesc{};
    GpuRenderTargetHandle mDefaultRenderTarget{};
    GpuRenderTargetBinding mActiveRenderTargetBinding{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    std::unordered_map<std::uint64_t, GpuRenderTargetBinding> mPendingReadbackRequests;
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    std::unordered_map<std::uint64_t, GpuRenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_IMPL_H
