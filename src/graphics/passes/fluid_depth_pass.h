#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FLUID_DEPTH_PASS_H

#include "gpu/gpu_device.h"
#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <cstdint>
#include <unordered_map>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics
{
struct ResolvedCameraView;
}

namespace cressim::neo::graphics::detail
{

class FluidDepthPass
{
public:
    explicit FluidDepthPass(gpu::GpuDevice &device);

    bool initialize();
    bool draw(const gpu::GpuRenderTargetBinding &targetBinding,
              const gpu::GpuRenderTargetDesc &targetDesc, const GpuEntitySceneView &gpuScene,
              const cressim::neo::physics::PhysicsGpuSceneView &physicsScene,
              const ResolvedCameraView &camera, std::uint32_t sceneDepthLayer,
              Diligent::ITextureView *sceneDepthSrv, const EnvironmentFluidDesc &environmentFluid);

private:
    struct DrawConstants
    {
        std::uint32_t cameraIndex     = 0u;
        std::uint32_t sceneDepthLayer = 0u;
        std::uint32_t envIndex        = 0u;
        float padding1                = 0.0f;
        float depthEdgeThreshold      = 0.2f;
        float padding0                = 0.0f;
        float padding2                = 0.0f;
    };

    struct PipelineKey
    {
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;

        bool operator==(const PipelineKey &rhs) const noexcept
        {
            return colorFormat == rhs.colorFormat;
        }
    };

    struct PipelineKeyHasher
    {
        std::size_t operator()(const PipelineKey &key) const noexcept;
    };

    bool ensureConstants(Diligent::IRenderDevice *renderDevice);
    Diligent::IPipelineState *getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                  const PipelineKey &key);
    Diligent::IShaderResourceBinding *getOrCreateBinding(Diligent::IPipelineState *pipeline);

    gpu::GpuDevice &mDevice;
    bool mInitialized = false;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mLinearClampSampler;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mPipelines;
    std::unordered_map<Diligent::IPipelineState *,
                       Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>>
        mBindings;
};

} // namespace cressim::neo::graphics::detail

#endif
