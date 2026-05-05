#ifndef CRESSIM_NEO_GRAPHICS_SKYBOX_PASS_H
#define CRESSIM_NEO_GRAPHICS_SKYBOX_PASS_H

#include "gpu/gpu_device.h"
#include "graphics/gpu_scene.h"
#include "graphics/host_scene.h"
#include "graphics/passes/render_pass_types.h"
#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cressim::neo::graphics::detail
{

class SkyboxPass
{
public:
    SkyboxPass(gpu::GpuDevice &device, RenderResourceManager &resourceManager);

    bool initialize();
    bool drawBatch(const gpu::GpuRenderTargetBinding &targetBinding,
                   const gpu::GpuRenderTargetDesc &targetDesc,
                   const GpuEntitySceneView &gpuScene, Diligent::IBuffer *batchCameraBuffer,
                   std::uint32_t batchCameraCount,
                   const std::vector<EnvironmentIblDesc> *environmentIbls,
                   std::uint32_t envCount);

private:
    struct EnvironmentBackgroundLookupEntry
    {
        std::uint32_t sliceIndex = 0u;
        std::uint32_t enabled    = 0u;
        float intensity          = 1.0f;
        float padding0           = 0.0f;
    };

    struct PipelineKey
    {
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;

        bool operator==(const PipelineKey &rhs) const noexcept
        {
            return colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat;
        }
    };

    struct PipelineKeyHasher
    {
        std::size_t operator()(const PipelineKey &key) const noexcept;
    };

    bool ensureResources(Diligent::IRenderDevice *renderDevice);
    bool ensureBackgroundResources(Diligent::IRenderDevice *renderDevice,
                                   Diligent::IDeviceContext *graphicsContext,
                                   Diligent::Uint64 graphicsContextMask,
                                   const std::vector<EnvironmentIblDesc> *environmentIbls,
                                   std::uint32_t envCount);
    Diligent::IPipelineState *getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                  const PipelineKey &key);
    Diligent::IShaderResourceBinding *getOrCreateBinding(Diligent::IPipelineState *pipeline);

    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    bool mInitialized = false;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mSampler;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mBackgroundArraySrv;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mBackgroundLookupBuffer;
    std::uint32_t mBackgroundLookupCapacity = 0u;
    std::size_t mBackgroundStateHash        = 0u;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mPipelines;
    std::unordered_map<Diligent::IPipelineState *,
                       Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>>
        mBindings;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_SKYBOX_PASS_H
