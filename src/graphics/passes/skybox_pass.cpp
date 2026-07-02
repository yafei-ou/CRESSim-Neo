#include "graphics/passes/skybox_pass.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/shader_library.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cressim::neo::graphics::detail
{

namespace
{

template <typename T>
void hashCombine(std::size_t &seed, const T &value)
{
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

Diligent::TEXTURE_FORMAT resolveTextureFormat(const TextureResourceDesc &desc) noexcept
{
    switch (desc.pixelFormat)
    {
    case TexturePixelFormat::RGBA8:
        return desc.colorSpace == TextureColorSpace::Srgb ? Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB
                                                          : Diligent::TEX_FORMAT_RGBA8_UNORM;
    case TexturePixelFormat::RGBA16F:
        return Diligent::TEX_FORMAT_RGBA16_FLOAT;
    default:
        return Diligent::TEX_FORMAT_UNKNOWN;
    }
}

std::uint32_t bytesPerPixel(TexturePixelFormat format) noexcept
{
    switch (format)
    {
    case TexturePixelFormat::RGBA8:
        return 4u;
    case TexturePixelFormat::RGBA16F:
        return 8u;
    default:
        return 0u;
    }
}

std::size_t subresourceIndex(std::uint32_t mipLevel, std::uint32_t arrayLayer,
                             std::uint32_t layersPerTexture) noexcept
{
    return static_cast<std::size_t>(mipLevel) * layersPerTexture + arrayLayer;
}

std::size_t diligentSubresourceIndex(std::uint32_t arrayLayer, std::uint32_t mipLevel,
                                     std::uint32_t mipCount) noexcept
{
    return static_cast<std::size_t>(arrayLayer) * mipCount + mipLevel;
}

bool isTextureArrayCompatible(const TextureResourceDesc &lhs,
                              const TextureResourceDesc &rhs) noexcept
{
    return lhs.width == rhs.width && lhs.height == rhs.height &&
           lhs.mipLevelCount == rhs.mipLevelCount && lhs.dimension == rhs.dimension &&
           lhs.pixelFormat == rhs.pixelFormat && lhs.colorSpace == rhs.colorSpace;
}

Diligent::RefCntAutoPtr<Diligent::ITextureView> createTextureSrv(Diligent::ITexture *texture,
                                                                 Diligent::ISampler *sampler)
{
    if (texture == nullptr)
    {
        return {};
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> view{
        texture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE)};
    if (view != nullptr && sampler != nullptr)
    {
        view->SetSampler(sampler);
    }
    return view;
}

} // namespace

SkyboxPass::SkyboxPass(gpu::GpuDevice &device, RenderResourceManager &resourceManager)
    : mDevice(device), mResourceManager(resourceManager)
{
}

std::size_t SkyboxPass::PipelineKeyHasher::operator()(const PipelineKey &key) const noexcept
{
    const std::size_t colorHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat));
    const std::size_t depthHash =
        std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat));
    return colorHash ^ (depthHash << 1u);
}

bool SkyboxPass::initialize()
{
    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }

    mInitialized = ensureResources(backendContext.renderDevice);
    return mInitialized;
}

bool SkyboxPass::ensureResources(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }
    if (mSampler == nullptr)
    {
        Diligent::SamplerDesc samplerDesc{};
        samplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
        samplerDesc.AddressU  = Diligent::TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressV  = Diligent::TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressW  = Diligent::TEXTURE_ADDRESS_CLAMP;
        renderDevice->CreateSampler(samplerDesc, &mSampler);
    }

    return mSampler != nullptr;
}

bool SkyboxPass::ensureBackgroundResources(Diligent::IRenderDevice *renderDevice,
                                           Diligent::IDeviceContext *graphicsContext,
                                           Diligent::Uint64 graphicsContextMask,
                                           const std::vector<EnvironmentIblDesc> *environmentIbls,
                                           std::uint32_t envCount)
{
    if (renderDevice == nullptr || graphicsContext == nullptr)
    {
        return false;
    }

    std::size_t stateHash = 0u;
    hashCombine(stateHash, envCount);
    if (environmentIbls != nullptr)
    {
        for (const EnvironmentIblDesc &ibl : *environmentIbls)
        {
            hashCombine(stateHash, ibl.backgroundCubemap.id);
            hashCombine(stateHash, ibl.backgroundIntensity);
        }
    }
    if (mBackgroundArraySrv != nullptr && mBackgroundLookupBuffer != nullptr &&
        stateHash == mBackgroundStateHash)
    {
        return true;
    }

    std::vector<EnvironmentBackgroundLookupEntry> lookupEntries(envCount);
    std::vector<const TextureResourceDesc *> backgroundTextures;
    const TextureResourceDesc *canonicalBackground = nullptr;
    std::unordered_map<common::ResourceId, std::uint32_t> uniqueSliceIndices;

    const auto tryGetTexture = [&](TextureHandle handle) -> const TextureResourceDesc *
    {
        if (handle.id == common::kInvalidResourceId)
        {
            return nullptr;
        }
        return mResourceManager.tryGetTexture(handle);
    };

    if (environmentIbls != nullptr)
    {
        for (std::uint32_t envIndex = 0u;
             envIndex <
             std::min<std::uint32_t>(envCount, static_cast<std::uint32_t>(environmentIbls->size()));
             ++envIndex)
        {
            const EnvironmentIblDesc &ibl         = (*environmentIbls)[envIndex];
            const TextureResourceDesc *background = tryGetTexture(ibl.backgroundCubemap);
            if (background == nullptr || background->dimension != TextureDimension::TextureCube)
            {
                continue;
            }
            if (canonicalBackground == nullptr)
            {
                canonicalBackground = background;
            }
            if (canonicalBackground == nullptr ||
                !isTextureArrayCompatible(*canonicalBackground, *background))
            {
                continue;
            }

            const auto [it, inserted] = uniqueSliceIndices.emplace(
                ibl.backgroundCubemap.id, static_cast<std::uint32_t>(backgroundTextures.size()));
            if (inserted)
            {
                backgroundTextures.push_back(background);
            }

            lookupEntries[envIndex].sliceIndex = it->second;
            lookupEntries[envIndex].enabled    = 1u;
            lookupEntries[envIndex].intensity  = std::max(ibl.backgroundIntensity, 0.0f);
        }
    }

    if (!gpu::detail::ensureStructuredBufferCapacity(
            renderDevice, "CRESSimNeo.SkyboxPass.BackgroundLookup",
            sizeof(EnvironmentBackgroundLookupEntry), std::max(envCount, 1u), 1u,
            Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
            graphicsContextMask, mBackgroundLookupBuffer, mBackgroundLookupCapacity))
    {
        return false;
    }
    if (!lookupEntries.empty())
    {
        graphicsContext->UpdateBuffer(
            mBackgroundLookupBuffer, 0u,
            static_cast<Diligent::Uint32>(lookupEntries.size() *
                                          sizeof(EnvironmentBackgroundLookupEntry)),
            lookupEntries.data(), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }

    if (canonicalBackground == nullptr || backgroundTextures.empty())
    {
        mBackgroundArraySrv.Release();
        mBackgroundStateHash = stateHash;
        return true;
    }

    const std::uint32_t layersPerTexture = 6u;
    const std::uint32_t arrayTextures    = static_cast<std::uint32_t>(backgroundTextures.size());
    const std::uint32_t totalArrayLayers = arrayTextures * layersPerTexture;
    const std::uint32_t bpp              = bytesPerPixel(canonicalBackground->pixelFormat);
    if (bpp == 0u)
    {
        return false;
    }

    Diligent::TextureDesc textureDesc{};
    textureDesc.Name      = "CRESSimNeo.SkyboxPass.BackgroundArray";
    textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
    textureDesc.Width     = canonicalBackground->width;
    textureDesc.Height    = canonicalBackground->height;
    textureDesc.MipLevels = canonicalBackground->mipLevelCount;
    textureDesc.ArraySize = totalArrayLayers;
    textureDesc.Format    = resolveTextureFormat(*canonicalBackground);
    textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    textureDesc.Usage     = Diligent::USAGE_IMMUTABLE;

    std::vector<Diligent::TextureSubResData> subresources(
        static_cast<std::size_t>(textureDesc.MipLevels) * totalArrayLayers);
    for (std::uint32_t textureIndex = 0u; textureIndex < arrayTextures; ++textureIndex)
    {
        const TextureResourceDesc &source = *backgroundTextures[textureIndex];
        for (std::uint32_t layer = 0u; layer < layersPerTexture; ++layer)
        {
            const std::uint32_t dstLayer = textureIndex * layersPerTexture + layer;
            for (std::uint32_t mipLevel = 0u; mipLevel < textureDesc.MipLevels; ++mipLevel)
            {
                const std::uint32_t mipWidth  = std::max(textureDesc.Width >> mipLevel, 1u);
                const std::uint32_t rowStride = mipWidth * bpp;
                const std::size_t dstIndex =
                    diligentSubresourceIndex(dstLayer, mipLevel, textureDesc.MipLevels);
                const auto &sourceSubresource =
                    source.subresources[subresourceIndex(mipLevel, layer, layersPerTexture)];
                subresources[dstIndex] =
                    Diligent::TextureSubResData{sourceSubresource.pixelData.data(), rowStride};
            }
        }
    }

    Diligent::TextureData initialData{subresources.data(),
                                      static_cast<Diligent::Uint32>(subresources.size())};
    Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
    renderDevice->CreateTexture(textureDesc, &initialData, &texture);
    if (texture == nullptr)
    {
        return false;
    }

    mBackgroundArraySrv  = createTextureSrv(texture, mSampler);
    mBackgroundStateHash = stateHash;
    return mBackgroundArraySrv != nullptr;
}

Diligent::IPipelineState *SkyboxPass::getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                          const PipelineKey &key)
{
    auto it = mPipelines.find(key);
    if (it != mPipelines.end())
    {
        return it->second;
    }
    if (renderDevice == nullptr || key.colorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return nullptr;
    }

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return nullptr;
    }

    Diligent::ShaderCreateInfo shaderCreateInfo{};
    shaderCreateInfo.SourceLanguage                  = Diligent::SHADER_SOURCE_LANGUAGE_HLSL;
    shaderCreateInfo.Desc.UseCombinedTextureSamplers = true;
    shaderCreateInfo.EntryPoint                      = "main";
    shaderCreateInfo.pShaderSourceStreamFactory      = streamFactory;

    Diligent::ShaderMacro layerMacros[] = {
        {"MANUAL_LAYER_EXPORT", "1"},
    };
    shaderCreateInfo.Macros =
        Diligent::ShaderMacroArray{layerMacros, static_cast<Diligent::Uint32>(1u)};

    Diligent::RefCntAutoPtr<Diligent::IShader> vertexShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_VERTEX;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.SkyboxPass.VS";
    shaderCreateInfo.FilePath        = "graphics/skybox.vs.hlsl";
    if (!mDevice.createShader(shaderCreateInfo, &vertexShader))
    {
        vertexShader = nullptr;
    }
    if (vertexShader == nullptr)
    {
        return nullptr;
    }

    Diligent::RefCntAutoPtr<Diligent::IShader> pixelShader;
    shaderCreateInfo.Desc.ShaderType = Diligent::SHADER_TYPE_PIXEL;
    shaderCreateInfo.Desc.Name       = "CRESSimNeo.SkyboxPass.PS";
    shaderCreateInfo.FilePath        = "graphics/skybox.ps.hlsl";
    shaderCreateInfo.Macros          = {};
    if (!mDevice.createShader(shaderCreateInfo, &pixelShader))
    {
        pixelShader = nullptr;
    }
    if (pixelShader == nullptr)
    {
        return nullptr;
    }

    Diligent::GraphicsPipelineStateCreateInfo psoCreateInfo{};
    psoCreateInfo.PSODesc.Name                       = "CRESSimNeo.SkyboxPass.PSO";
    psoCreateInfo.PSODesc.PipelineType               = Diligent::PIPELINE_TYPE_GRAPHICS;
    psoCreateInfo.GraphicsPipeline.NumRenderTargets  = 1;
    psoCreateInfo.GraphicsPipeline.RTVFormats[0]     = key.colorFormat;
    psoCreateInfo.GraphicsPipeline.DSVFormat         = key.depthFormat;
    psoCreateInfo.GraphicsPipeline.PrimitiveTopology = Diligent::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    psoCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode           = Diligent::CULL_MODE_NONE;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable      = Diligent::False;
    psoCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthWriteEnable = Diligent::False;
    psoCreateInfo.GraphicsPipeline.BlendDesc.RenderTargets[0].BlendEnable = Diligent::False;
    psoCreateInfo.PSODesc.ResourceLayout.DefaultVariableType =
        Diligent::SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

    constexpr Diligent::ShaderResourceVariableDesc kVars[] = {
        {Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_PreparedCameras",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentBackgroundLookup",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
        {Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentBackgroundArray",
         Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    };
    psoCreateInfo.PSODesc.ResourceLayout.Variables = kVars;
    psoCreateInfo.PSODesc.ResourceLayout.NumVariables =
        static_cast<Diligent::Uint32>(std::size(kVars));
    psoCreateInfo.pVS = vertexShader;
    psoCreateInfo.pPS = pixelShader;

    Diligent::RefCntAutoPtr<Diligent::IPipelineState> pipeline;
    if (!mDevice.createGraphicsPipelineState(psoCreateInfo, &pipeline))
    {
        pipeline = nullptr;
    }
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    auto insertResult = mPipelines.emplace(key, pipeline);
    return insertResult.first->second;
}

Diligent::IShaderResourceBinding *SkyboxPass::getOrCreateBinding(Diligent::IPipelineState *pipeline)
{
    if (pipeline == nullptr)
    {
        return nullptr;
    }

    auto it = mBindings.find(pipeline);
    if (it != mBindings.end())
    {
        return it->second;
    }

    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> srb;
    pipeline->CreateShaderResourceBinding(&srb, true);
    if (srb == nullptr)
    {
        return nullptr;
    }

    auto insertResult = mBindings.emplace(pipeline, srb);
    return insertResult.first->second;
}

bool SkyboxPass::drawBatch(const gpu::GpuRenderTargetBinding &targetBinding,
                           const gpu::GpuRenderTargetDesc &targetDesc,
                           const GpuEntitySceneView &gpuScene, Diligent::IBuffer *batchCameraBuffer,
                           std::uint32_t batchCameraCount,
                           const std::vector<EnvironmentIblDesc> *environmentIbls,
                           std::uint32_t envCount)
{
    if (batchCameraCount == 0u)
    {
        return true;
    }
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr)
    {
        return false;
    }
    if (!backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding) ||
        gpuScene.cameraInputsBuffer == nullptr || batchCameraBuffer == nullptr)
    {
        return false;
    }

    if (!ensureResources(backendContext.renderDevice) ||
        !ensureBackgroundResources(backendContext.renderDevice, backendContext.graphicsContext,
                                   gpu::contextMaskForId(backendContext.contextId), environmentIbls,
                                   envCount))
    {
        return false;
    }
    if (mBackgroundArraySrv == nullptr || mBackgroundLookupBuffer == nullptr)
    {
        return true;
    }

    const PipelineKey key{targetDesc.colorFormat,
                          targetDesc.depth ? targetDesc.depthFormat : Diligent::TEX_FORMAT_UNKNOWN};
    Diligent::IPipelineState *pipeline = getOrCreatePipeline(backendContext.renderDevice, key);
    Diligent::IShaderResourceBinding *binding = getOrCreateBinding(pipeline);
    if (pipeline == nullptr || binding == nullptr)
    {
        return false;
    }

    const auto setBufferView = [&](Diligent::SHADER_TYPE shaderType, const char *name,
                                   Diligent::IBuffer *buffer) -> bool
    {
        Diligent::IShaderResourceVariable *variable = binding->GetVariableByName(shaderType, name);
        if (variable == nullptr || buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv = buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
        return true;
    };
    const auto setTextureView = [&](Diligent::SHADER_TYPE shaderType, const char *name,
                                    Diligent::ITextureView *view) -> bool
    {
        Diligent::IShaderResourceVariable *variable = binding->GetVariableByName(shaderType, name);
        if (variable == nullptr || view == nullptr)
        {
            return false;
        }
        variable->Set(view);
        return true;
    };

    if (!setBufferView(Diligent::SHADER_TYPE_VERTEX, "g_BatchCameras", batchCameraBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_PIXEL, "g_CameraInputs",
                       gpuScene.cameraInputsBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_PIXEL, "g_PreparedCameras",
                       gpuScene.preparedCamerasBuffer) ||
        !setBufferView(Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentBackgroundLookup",
                       mBackgroundLookupBuffer) ||
        !setTextureView(Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentBackgroundArray",
                        mBackgroundArraySrv))
    {
        return false;
    }

    backendContext.graphicsContext->SetPipelineState(pipeline);
    backendContext.graphicsContext->CommitShaderResources(
        binding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawAttribs drawAttrs{};
    drawAttrs.NumVertices  = 3u;
    drawAttrs.NumInstances = batchCameraCount;
    drawAttrs.Flags        = Diligent::DRAW_FLAG_VERIFY_ALL;
    backendContext.graphicsContext->Draw(drawAttrs);
    return true;
}

} // namespace cressim::neo::graphics::detail
