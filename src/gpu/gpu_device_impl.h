#ifndef CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
#define CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H

#include "gpu/gpu_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/SwapChain.h"

#include <memory>

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
    const std::string &shaderSourceDirectory() const override;

private:
    bool initializeVulkan();
    bool createPrimarySwapChain();
    bool presentPrimarySwapChain();

private:
    GpuDeviceDesc mDesc{};
    GpuBackend mBackend = GpuBackend::Null;
    bool mInitialized   = false;
    std::unique_ptr<GpuRenderTargetSystemImpl> mRenderTargets;

    Diligent::RefCntAutoPtr<Diligent::IRenderDevice> mRenderDevice;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mImmediateContext;
    Diligent::RefCntAutoPtr<Diligent::IDeviceContext> mPhysicsContext;
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mPrimarySwapChain;
    std::uint32_t mGraphicsContextId                = 0;
    std::uint32_t mPhysicsContextId                 = 0;
    Diligent::COMMAND_QUEUE_TYPE mGraphicsQueueType = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
    Diligent::COMMAND_QUEUE_TYPE mPhysicsQueueType  = Diligent::COMMAND_QUEUE_TYPE_UNKNOWN;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
