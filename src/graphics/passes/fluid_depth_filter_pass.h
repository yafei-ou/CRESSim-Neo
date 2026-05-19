#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_FILTER_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_FILTER_PASS_H

#include "gpu/gpu_compute_pass.h"
#include "gpu/gpu_device.h"
#include "graphics/passes/render_pass_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"

namespace cressim::neo::graphics
{
struct GpuEntitySceneView;
}

namespace cressim::neo::graphics::detail
{

class FluidDepthFilterPass
{
public:
    explicit FluidDepthFilterPass(gpu::GpuDevice &device);

    bool initialize();
    bool filter(const cressim::neo::graphics::GpuEntitySceneView &gpuScene,
                Diligent::ITextureView *sourceSrv, Diligent::ITextureView *destUav,
                const ResolvedCameraView &camera, std::uint32_t sourceLayer,
                const EnvironmentFluidDesc &environmentFluid);

private:
    struct FilterConstants
    {
        // x: layer, y: cameraIndex, z: maxFilterRadius, w: reserved
        Diligent::uint4 dispatchParams{0u, 0u, 6u, 0u};
        // x: filterWorldRadius, y: filterDepthThreshold, z/w: reserved
        Diligent::float4 filterParams{0.18f, 0.12f, 0.0f, 0.0f};
        // x/y: viewport origin in pixels, z/w: viewport size in pixels
        Diligent::uint4 viewportRect{0u, 0u, 1u, 1u};
    };
    static_assert(sizeof(FilterConstants) == 48u);

    gpu::GpuDevice &mDevice;
    bool mInitialized = false;
    gpu::GpuComputePass mFilterPass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
};

} // namespace cressim::neo::graphics::detail

#endif
