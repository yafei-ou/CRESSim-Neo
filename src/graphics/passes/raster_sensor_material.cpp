#include "graphics/passes/raster_sensor_material.h"

#include <array>
#include <cstring>

namespace cressim::neo::graphics::detail
{

namespace
{

bool createSolidTexture(Diligent::IRenderDevice *renderDevice, Diligent::ISampler *sampler,
                        const char *debugName, Diligent::TEXTURE_FORMAT format,
                        const std::array<std::uint8_t, 4> &rgba,
                        Diligent::RefCntAutoPtr<Diligent::ITextureView> &outSrv)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    Diligent::TextureDesc textureDesc{};
    textureDesc.Name      = debugName;
    textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_2D;
    textureDesc.Width     = 1u;
    textureDesc.Height    = 1u;
    textureDesc.MipLevels = 1u;
    textureDesc.ArraySize = 1u;
    textureDesc.Format    = format;
    textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    textureDesc.Usage     = Diligent::USAGE_IMMUTABLE;

    Diligent::TextureSubResData subresource{rgba.data(), 4u};
    Diligent::TextureData initialData{&subresource, 1u};
    Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
    renderDevice->CreateTexture(textureDesc, &initialData, &texture);
    if (texture == nullptr)
    {
        return false;
    }

    outSrv = texture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    if (outSrv != nullptr && sampler != nullptr)
    {
        outSrv->SetSampler(sampler);
    }
    return outSrv != nullptr;
}

} // namespace

RasterSensorMaterialHelper::RasterSensorMaterialHelper(const RenderResourceManager &resourceManager,
                                                       const char *debugPrefix)
    : mResourceManager(resourceManager),
      mTextureGpuCache(debugPrefix != nullptr ? std::string(debugPrefix) + ".TextureCache"
                                              : "CRESSimNeo.RasterSensorMaterial.TextureCache"),
      mDebugPrefix(debugPrefix)
{
}

bool RasterSensorMaterialHelper::ensureFallbackBaseColor(Diligent::IRenderDevice *renderDevice)
{
    if (mFallbackBaseColorSrv != nullptr)
    {
        return true;
    }

    return createSolidTexture(renderDevice, mMaterialSampler,
                              mDebugPrefix != nullptr ? mDebugPrefix : "CRESSimNeo.RasterSensor",
                              Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB, {255u, 255u, 255u, 255u},
                              mFallbackBaseColorSrv);
}

bool RasterSensorMaterialHelper::initialize(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }
    if (mPerMaterialBuffer != nullptr && mMaterialSampler != nullptr &&
        mFallbackBaseColorSrv != nullptr)
    {
        return true;
    }

    if (mMaterialSampler == nullptr)
    {
        Diligent::SamplerDesc samplerDesc{};
        samplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.AddressU  = Diligent::TEXTURE_ADDRESS_WRAP;
        samplerDesc.AddressV  = Diligent::TEXTURE_ADDRESS_WRAP;
        samplerDesc.AddressW  = Diligent::TEXTURE_ADDRESS_WRAP;
        renderDevice->CreateSampler(samplerDesc, &mMaterialSampler);
        if (mMaterialSampler == nullptr)
        {
            return false;
        }
    }

    if (mPerMaterialBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name =
            mDebugPrefix != nullptr ? mDebugPrefix : "CRESSimNeo.RasterSensorMaterial";
        constantBufferDesc.Size           = sizeof(ForwardPerMaterialConstants);
        constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mPerMaterialBuffer);
        if (mPerMaterialBuffer == nullptr)
        {
            return false;
        }
    }

    return ensureFallbackBaseColor(renderDevice);
}

bool RasterSensorMaterialHelper::bindStaticResources(Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr || mPerMaterialBuffer == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable *perMaterialVar = pipeline->GetStaticVariableByName(
        Diligent::SHADER_TYPE_PIXEL, "GraphicsForwardPerMaterial");
    if (perMaterialVar != nullptr)
    {
        perMaterialVar->Set(mPerMaterialBuffer);
    }
    return true;
}

bool RasterSensorMaterialHelper::bindMaterialResources(
    Diligent::IRenderDevice *renderDevice, Diligent::IDeviceContext *graphicsContext,
    Diligent::IShaderResourceBinding *shaderBinding, common::ResourceId materialId)
{
    if (shaderBinding == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable *baseColorVar =
        shaderBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL, "g_BaseColorTexture");
    if (baseColorVar == nullptr)
    {
        return true;
    }

    const MaterialResourceDesc *material =
        mResourceManager.tryGetMaterial(MaterialHandle{materialId});
    if (material == nullptr || renderDevice == nullptr || graphicsContext == nullptr)
    {
        return false;
    }
    if (!initialize(renderDevice))
    {
        return false;
    }

    Diligent::ITextureView *textureView = mFallbackBaseColorSrv;
    if (material->baseColorTexture.id != common::kInvalidResourceId)
    {
        TextureGpuCache::CachedTexture *cachedTexture =
            mTextureGpuCache.getOrCreate(mResourceManager, material->baseColorTexture, renderDevice,
                                         graphicsContext, mMaterialSampler);
        if (cachedTexture != nullptr && cachedTexture->shaderResourceView != nullptr)
        {
            textureView = cachedTexture->shaderResourceView;
        }
    }
    if (textureView == nullptr)
    {
        return false;
    }

    baseColorVar->Set(textureView);
    return true;
}

bool RasterSensorMaterialHelper::updateMaterialConstants(Diligent::IDeviceContext *graphicsContext,
                                                         common::ResourceId materialId)
{
    if (graphicsContext == nullptr || mPerMaterialBuffer == nullptr)
    {
        return false;
    }

    const MaterialResourceDesc *material =
        mResourceManager.tryGetMaterial(MaterialHandle{materialId});
    if (material == nullptr)
    {
        return false;
    }

    ForwardPerMaterialConstants materialConstants{};
    materialConstants.baseColorFactor = Diligent::float4{
        material->baseColor.x, material->baseColor.y, material->baseColor.z, material->opacity};
    materialConstants.emissiveFactor = Diligent::float4{
        material->emissiveFactor.x, material->emissiveFactor.y, material->emissiveFactor.z, 0.0f};
    materialConstants.materialParams =
        Diligent::float4{material->metallic, material->roughness, material->pipeline.alphaCutoff,
                         material->receivesShadows ? 1.0f : 0.0f};

    void *mapped = nullptr;
    graphicsContext->MapBuffer(mPerMaterialBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                               mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &materialConstants, sizeof(materialConstants));
    graphicsContext->UnmapBuffer(mPerMaterialBuffer, Diligent::MAP_WRITE);
    return true;
}

} // namespace cressim::neo::graphics::detail
