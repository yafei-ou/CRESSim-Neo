#ifndef CRESSIM_NEO_GRAPHICS_PASSES_RASTER_SENSOR_MATERIAL_H
#define CRESSIM_NEO_GRAPHICS_PASSES_RASTER_SENSOR_MATERIAL_H

#include "graphics/render_resource_manager.h"
#include "graphics/services/texture_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/ShaderResourceBinding.h"

namespace cressim::neo::graphics::detail
{

class RasterSensorMaterialHelper
{
public:
    RasterSensorMaterialHelper(const RenderResourceManager &resourceManager,
                               const char *debugPrefix);

    bool initialize(Diligent::IRenderDevice *renderDevice);
    bool bindStaticResources(Diligent::IPipelineState *pipeline);
    bool bindMaterialResources(Diligent::IRenderDevice *renderDevice,
                               Diligent::IDeviceContext *graphicsContext,
                               Diligent::IShaderResourceBinding *shaderBinding,
                               common::ResourceId materialId);
    bool updateMaterialConstants(Diligent::IDeviceContext *graphicsContext,
                                 common::ResourceId materialId);

private:
    struct ForwardPerMaterialConstants
    {
        Diligent::float4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 materialParams{0.0f, 0.5f, 0.5f, 0.0f};
    };

    bool ensureFallbackBaseColor(Diligent::IRenderDevice *renderDevice);

    const RenderResourceManager &mResourceManager;
    TextureGpuCache mTextureGpuCache;
    const char *mDebugPrefix = nullptr;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mMaterialSampler;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackBaseColorSrv;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerMaterialBuffer;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_RASTER_SENSOR_MATERIAL_H
