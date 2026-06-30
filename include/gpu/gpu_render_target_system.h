#ifndef CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H
#define CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H

#include "common/frame_context.h"
#include "gpu/export.h"
#include "gpu/gpu_types.h"

#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <cstdint>

namespace cressim::neo::gpu
{

class CRESSIM_NEO_GPU_API GpuRenderTargetSystem
{
public:
    virtual ~GpuRenderTargetSystem() = default;

    virtual GpuRenderTargetHandle createRenderTarget(const GpuRenderTargetDesc &desc) = 0;
    virtual GpuRenderTargetUpdateResult resizeRenderTarget(GpuRenderTargetHandle target,
                                                           std::uint32_t width,
                                                           std::uint32_t height)      = 0;
    virtual GpuRenderTargetUpdateResult reconfigureRenderTarget(
        GpuRenderTargetHandle target, const GpuRenderTargetDesc &desc)      = 0;
    virtual void destroyRenderTarget(GpuRenderTargetHandle target)          = 0;
    virtual bool isValidRenderTarget(GpuRenderTargetHandle target) const    = 0;
    virtual bool tryGetRenderTargetDesc(GpuRenderTargetHandle target,
                                        GpuRenderTargetDesc &outDesc) const = 0;

    virtual void setRenderTargetViewport(const GpuRenderTargetBinding &binding,
                                         const GpuRenderViewport &viewport) = 0;
    virtual void beginRenderTarget(const GpuRenderTargetBinding &binding,
                                   const common::FrameContext &frameContext,
                                   const GpuRenderPassBeginDesc &beginDesc) = 0;
    virtual void endRenderTarget(const GpuRenderTargetBinding &binding,
                                 const common::FrameContext &frameContext)  = 0;

    virtual GpuRenderTargetReadbackRequest requestRenderTargetReadback(
        const GpuRenderTargetBinding &binding)                                          = 0;
    virtual bool tryGetRenderTargetReadback(GpuRenderTargetReadbackRequest request,
                                            GpuRenderTargetReadbackEvent &outEvent)     = 0;
    virtual bool tryGetRenderTargetColorTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture *&outTexture)        = 0;
    virtual bool tryGetRenderTargetDepthTexture(GpuRenderTargetHandle target,
                                                Diligent::ITexture *&outTexture)        = 0;
    virtual bool tryGetRenderTargetShaderResourceView(const GpuRenderTargetBinding &binding,
                                                      GpuRenderTargetTexturePlane plane,
                                                      Diligent::ITextureView *&outView) = 0;
};

} // namespace cressim::neo::gpu

#endif // CRESSIM_NEO_GPU_GPU_RENDER_TARGET_SYSTEM_H
