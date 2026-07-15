#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FLUID_COMPOSITE_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FLUID_COMPOSITE_PASS_H

#include "gpu/gpu_device.h"
#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <cstdint>
#include <unordered_map>

namespace cressim::neo::graphics
{
struct ResolvedCameraView;
}

namespace cressim::neo::graphics::detail
{

class FluidCompositePass
{
public:
    explicit FluidCompositePass(gpu::GpuDevice &device);

    bool initialize();
    bool composite(const gpu::GpuRenderTargetBinding &targetBinding,
                   const gpu::GpuRenderTargetDesc &targetDesc, const GpuEntitySceneView &gpuScene,
                   const ResolvedCameraView &camera, std::uint32_t fluidDepthLayer,
                   std::uint32_t sceneDepthLayer, Diligent::ITextureView *filteredDepthSrv,
                   Diligent::ITextureView *surfaceColorSrv, Diligent::ITextureView *sceneColorSrv,
                   Diligent::ITextureView *sceneDepthSrv,
                   const EnvironmentFluidDesc &environmentFluid);

private:
    struct CompositeConstants
    {
        Diligent::float4 specularSmoothness{0.35f, 0.4f, 0.45f, 0.92f};
        std::uint32_t cameraIndex                = 0u;
        std::uint32_t fluidDepthLayer            = 0u;
        std::uint32_t sceneDepthLayer            = 0u;
        std::uint32_t mainLightIndex             = 0u;
        float fresnel                            = 0.8f;
        float refractionIor                      = 1.33f;
        float refractionViewThickness            = 0.35f;
        float padding0                           = 0.0f;
        float normalReconstructionDepthThreshold = 0.2f;
        float padding1                           = 0.0f;
        float padding2                           = 0.0f;
        float padding3                           = 0.0f;
        Diligent::float4 viewportRect{0.0f, 0.0f, 1.0f, 1.0f};
    };

    struct PipelineKey
    {
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
        bool enableBackgroundRefraction      = false;
        bool sceneInputsAreArray             = true;

        bool operator==(const PipelineKey &rhs) const noexcept
        {
            return colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat &&
                   enableBackgroundRefraction == rhs.enableBackgroundRefraction &&
                   sceneInputsAreArray == rhs.sceneInputsAreArray;
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
