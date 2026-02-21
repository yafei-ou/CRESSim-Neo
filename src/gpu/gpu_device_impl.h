#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H

#include "gpu/gpu_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cressim::neo::gpu
{

class GpuDeviceImpl final : public GpuDevice
{
public:
    bool initialize(const GpuDeviceDesc& desc) override;
    void shutdown() override;

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

    void beginFrame(const common::FrameContext& frameContext) override;
    void setRenderTargetViewport(GpuRenderTargetHandle target,
                                 const GpuRenderViewport& viewport) override;
    void beginRenderTarget(GpuRenderTargetHandle target, const common::FrameContext& frameContext,
                           const GpuRenderPassBeginDesc& beginDesc) override;
    void endRenderTarget(GpuRenderTargetHandle target,
                         const common::FrameContext& frameContext) override;
    GpuRenderTargetReadbackRequest requestRenderTargetReadback(
        GpuRenderTargetHandle target) override;
    bool tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                    GpuRenderTargetReadbackEvent& outEvent) override;
    void endFrame(const common::FrameContext& frameContext) override;

    GpuBackend backend() const override;
    bool tryGetBackendContext(GpuBackendContext& outContext) override;
    bool tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                        Diligent::ITexture*& outTexture) override;
    bool tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                        Diligent::ITexture*& outTexture) override;
    const std::string& shaderSourceDirectory() const override;

private:
    struct RenderTargetResources
    {
        GpuRenderTargetDesc desc{};
        GpuRenderViewport viewport{};
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_D32_FLOAT;
        Diligent::RefCntAutoPtr<Diligent::ITexture> colorTexture;
        Diligent::RefCntAutoPtr<Diligent::ITexture> depthTexture;
    };

    struct PendingReadbackCopy
    {
        std::vector<std::uint64_t> requestIds{};
        GpuRenderTargetHandle target{};
        std::uint64_t frameIndex             = 0;
        std::uint64_t fenceValue             = 0;
        std::uint32_t width                  = 0;
        std::uint32_t height                 = 0;
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::RefCntAutoPtr<Diligent::ITexture> stagingTexture;
    };

    bool initializeVulkan();
    bool createPrimarySwapChain();
    bool presentPrimarySwapChain();
    bool createDefaultRenderTarget();
    GpuRenderTargetDesc normalizeDefaultRenderTargetDesc(const GpuRenderTargetDesc& desc) const;
    GpuRenderTargetDesc normalizeTargetDesc(const GpuRenderTargetDesc& desc) const;
    bool createRenderTargetTextures(const GpuRenderTargetDesc& desc,
                                    RenderTargetResources& resources);

    bool queueReadbackCopy(GpuRenderTargetHandle target, std::uint64_t frameIndex,
                           const std::vector<std::uint64_t>& requestIds);

private:
    GpuDeviceDesc mDesc{};
    GpuBackend mBackend                                     = GpuBackend::Null;
    bool mInitialized                                       = false;
    bool mHasActiveRenderTarget                             = false;
    bool mActiveRenderTargetHasDepth                        = false;
    Diligent::TEXTURE_FORMAT mActiveRenderTargetColorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    common::ResourceId mNextRenderTargetId                  = 1;
    std::uint64_t mNextReadbackRequestId                    = 1;
    GpuRenderTargetHandle mDefaultRenderTarget{};
    GpuRenderTargetHandle mActiveRenderTarget{};

    std::unordered_map<common::ResourceId, RenderTargetResources> mRenderTargets;
    // Per-target readback request ids waiting for the next completed render pass of that target.
    std::unordered_map<common::ResourceId, std::vector<std::uint64_t>> mPendingReadbackRequests;
    // GPU->CPU copy jobs collected during render target completion and consumed in endFrame().
    std::vector<PendingReadbackCopy> mPendingReadbackCopies;
    // Completed results consumed through tryGetRenderTargetReadback().
    std::unordered_map<std::uint64_t, GpuRenderTargetReadbackEvent> mCompletedReadbacks;

    Diligent::RefCntAutoPtr<Diligent::IFence> mReadbackFence;
    std::uint64_t mNextReadbackFenceValue = 1;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mPrimarySwapChain;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
