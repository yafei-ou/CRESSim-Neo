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
    bool initialize(const GpuDeviceDesc& desc) override;
    void shutdown() override;

    void beginFrame(const common::FrameContext& frameContext) override;
    void endFrame(const common::FrameContext& frameContext) override;
    GpuRenderTargetSystem& renderTargetSystem() override;

    GpuBackend backend() const override;
    bool tryGetBackendContext(GpuBackendContext& outContext) override;
    const std::string& shaderSourceDirectory() const override;

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
    Diligent::RefCntAutoPtr<Diligent::ISwapChain> mPrimarySwapChain;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_DEVICE_IMPL_H
