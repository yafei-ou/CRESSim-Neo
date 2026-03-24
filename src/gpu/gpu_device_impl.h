#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H

#include "gpu/gpu_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Fence.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace cressim::neo::gpu
{

class GpuRenderTargetSystemImpl;

class GpuDeviceImpl final : public GpuDevice
{
public:
    bool initialize(const GpuDeviceDesc &desc) override;
    void shutdown() override;

    void beginFrame(const common::FrameContext &frameContext) override;
    void endFrame(const common::FrameContext &frameContext) override;
    GpuRenderTargetSystem &renderTargetSystem() override;

    GpuBackend backend() const override;
    bool tryGetGraphicsBackendContext(GpuBackendContext &outContext) override;
    bool tryGetPhysicsBackendContext(GpuComputeBackendContext &outContext) override;
    bool tryGetDefaultRenderTargetDesc(GpuRenderTargetDesc &outDesc) const override;
    bool tryGetPresentationTargetDesc(GpuPresentationTargetDesc &outDesc) override;
    GpuPresentationReadbackRequest requestPresentationReadback() override;
    bool tryGetPresentationReadback(GpuPresentationReadbackRequest request,
                                    GpuPresentationReadbackEvent &outEvent) override;
    const std::string &shaderSourceDirectory() const override;

private:
    struct PendingPresentationReadback
    {
        std::uint64_t requestId              = 0;
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
    bool queuePresentationReadback(const common::FrameContext &frameContext);
    bool consumePresentationReadback(PendingPresentationReadback &copy,
                                     GpuPresentationReadbackEvent &outEvent);

private:
    GpuDeviceDesc mDesc{};
    GpuBackend mBackend = GpuBackend::Null;
    bool mInitialized   = false;
    std::unique_ptr<GpuRenderTargetSystemImpl> mRenderTargets;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mPhysicsContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mPrimarySwapChain;
    std::uint32_t mGraphicsContextId                  = 0;
    std::uint32_t mPhysicsContextId                   = 0;
    Diligent::COMMAND_QUEUE_TYPE mGraphicsQueueType   = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    Diligent::COMMAND_QUEUE_TYPE mPhysicsQueueType    = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    std::uint64_t mNextPresentationReadbackRequestId  = 1;
    std::uint64_t mNextPresentationReadbackFenceValue = 1;
    std::unordered_map<std::uint64_t, std::uint64_t> mPendingPresentationReadbackRequests;
    std::vector<PendingPresentationReadback> mPendingPresentationReadbackCopies;
    std::unordered_map<std::uint64_t, GpuPresentationReadbackEvent> mCompletedPresentationReadbacks;
    Diligent::RefCntAutoPtr<Diligent::IFence> mPresentationReadbackFence;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
