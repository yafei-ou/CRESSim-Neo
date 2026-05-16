#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_FILTER_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_FILTER_PASS_H

#include "gpu/gpu_compute_pass.h"
#include "gpu/gpu_device.h"
#include "graphics/passes/render_pass_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

namespace cressim::neo::graphics::detail
{

class FluidDepthFilterPass
{
public:
    explicit FluidDepthFilterPass(gpu::GpuDevice &device);

    bool initialize();
    bool filter(Diligent::ITextureView *sourceSrv, Diligent::ITextureView *destUav,
                const ResolvedCameraView &camera, std::uint32_t sourceLayer,
                const EnvironmentFluidDesc &environmentFluid);

private:
    struct FilterConstants
    {
        std::uint32_t layer        = 0u;
        std::uint32_t outputWidth  = 0u;
        std::uint32_t outputHeight = 0u;
        std::uint32_t filterRadius = 3u;
        float depthEdgeThreshold   = 0.2f;
        float padding0             = 0.0f;
        float padding1             = 0.0f;
        float padding2             = 0.0f;
    };

    gpu::GpuDevice &mDevice;
    bool mInitialized = false;
    gpu::GpuComputePass mFilterPass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
};

} // namespace cressim::neo::graphics::detail

#endif
