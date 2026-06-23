#include "graphics/passes/forward_opaque_pass.h"

#include "common/math_utils_runtime.h"
#include "gpu/gpu_buffer_utils.h"
#include "physics/physics_gpu_scene_view.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cressim::neo::graphics::detail
{

namespace
{

std::uint16_t encodeFloat16(float value)
{
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent    = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa   = bits & 0x007fffffu;

    if (exponent <= 0)
    {
        if (exponent < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }

        mantissa = (mantissa | 0x00800000u) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }
    if (exponent >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) |
                                      ((mantissa + 0x00001000u) >> 13u));
}

Diligent::RefCntAutoPtr<Diligent::ITextureView> createArrayShadowSrv(Diligent::ITexture *texture,
                                                                     Diligent::ISampler *sampler)
{
    if (texture == nullptr)
    {
        return {};
    }

    const Diligent::TextureDesc &textureDesc = texture->GetDesc();
    if (textureDesc.Type != Diligent::RESOURCE_DIM_TEX_2D_ARRAY)
    {
        Diligent::RefCntAutoPtr<Diligent::ITextureView> view{
            texture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE)};
        if (view != nullptr && sampler != nullptr)
        {
            view->SetSampler(sampler);
        }
        return view;
    }

    Diligent::TextureViewDesc viewDesc{};
    viewDesc.ViewType        = Diligent::TEXTURE_VIEW_SHADER_RESOURCE;
    viewDesc.TextureDim      = textureDesc.Type;
    viewDesc.MostDetailedMip = 0u;
    viewDesc.NumMipLevels    = 1u;
    viewDesc.FirstArraySlice = 0u;
    viewDesc.NumArraySlices  = textureDesc.ArraySize;

    Diligent::RefCntAutoPtr<Diligent::ITextureView> arraySrv;
    texture->CreateView(viewDesc, &arraySrv);
    if (arraySrv != nullptr && sampler != nullptr)
    {
        arraySrv->SetSampler(sampler);
    }
    return arraySrv;
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

    outSrv = createTextureSrv(texture, sampler);
    return outSrv != nullptr;
}

bool createBrdfLutTexture(Diligent::IRenderDevice *renderDevice, Diligent::ISampler *sampler,
                          const char *debugName,
                          Diligent::RefCntAutoPtr<Diligent::ITextureView> &outSrv)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    constexpr std::uint32_t kBrdfLutSize = 128u;
    std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kBrdfLutSize) * kBrdfLutSize * 4u);
    for (std::uint32_t y = 0u; y < kBrdfLutSize; ++y)
    {
        const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(kBrdfLutSize);
        for (std::uint32_t x = 0u; x < kBrdfLutSize; ++x)
        {
            const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(kBrdfLutSize);
            const float rx    = roughness * -1.0f + 1.0f;
            const float ry    = roughness * -0.0275f + 0.0425f;
            const float rz    = roughness * -0.572f + 1.04f;
            const float rw    = roughness * 0.022f - 0.04f;
            const float a004  = std::min(rx * rx, std::exp2(-9.28f * nDotV)) * rx + ry;
            const float a     = common::runtime_math::clamp01((-1.04f) * a004 + rz);
            const float b     = common::runtime_math::clamp01((1.04f) * a004 + rw);

            const std::size_t texelIndex = (static_cast<std::size_t>(y) * kBrdfLutSize + x) * 4u;
            pixels[texelIndex + 0u]      = encodeFloat16(a);
            pixels[texelIndex + 1u]      = encodeFloat16(b);
            pixels[texelIndex + 2u]      = encodeFloat16(0.0f);
            pixels[texelIndex + 3u]      = encodeFloat16(1.0f);
        }
    }

    Diligent::TextureDesc textureDesc{};
    textureDesc.Name      = debugName;
    textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_2D;
    textureDesc.Width     = kBrdfLutSize;
    textureDesc.Height    = kBrdfLutSize;
    textureDesc.MipLevels = 1u;
    textureDesc.ArraySize = 1u;
    textureDesc.Format    = Diligent::TEX_FORMAT_RGBA16_FLOAT;
    textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    textureDesc.Usage     = Diligent::USAGE_IMMUTABLE;

    Diligent::TextureSubResData subresource{pixels.data(),
                                            kBrdfLutSize * sizeof(std::uint16_t) * 4u};
    Diligent::TextureData initialData{&subresource, 1u};
    Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
    renderDevice->CreateTexture(textureDesc, &initialData, &texture);
    if (texture == nullptr)
    {
        return false;
    }

    outSrv = createTextureSrv(texture, sampler);
    return outSrv != nullptr;
}

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

std::uint32_t arrayLayerCount(TextureDimension dimension) noexcept
{
    return dimension == TextureDimension::TextureCube ? 6u : 1u;
}

std::size_t subresourceIndex(std::uint32_t mipLevel, std::uint32_t layer, std::uint32_t layerCount)
{
    return static_cast<std::size_t>(mipLevel) * static_cast<std::size_t>(layerCount) +
           static_cast<std::size_t>(layer);
}

std::size_t diligentSubresourceIndex(std::uint32_t layer, std::uint32_t mipLevel,
                                     std::uint32_t mipLevelCount)
{
    return static_cast<std::size_t>(layer) * static_cast<std::size_t>(mipLevelCount) +
           static_cast<std::size_t>(mipLevel);
}

bool isTextureArrayCompatible(const TextureResourceDesc &lhs, const TextureResourceDesc &rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height &&
           lhs.mipLevelCount == rhs.mipLevelCount && lhs.dimension == rhs.dimension &&
           lhs.pixelFormat == rhs.pixelFormat && lhs.colorSpace == rhs.colorSpace;
}

bool createFallbackCubeArray(Diligent::IRenderDevice *renderDevice, Diligent::ISampler *sampler,
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
    textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_CUBE_ARRAY;
    textureDesc.Width     = 1u;
    textureDesc.Height    = 1u;
    textureDesc.MipLevels = 1u;
    textureDesc.ArraySize = 6u;
    textureDesc.Format    = format;
    textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
    textureDesc.Usage     = Diligent::USAGE_IMMUTABLE;

    std::array<Diligent::TextureSubResData, 6> subresources{};
    for (auto &subresource : subresources)
    {
        subresource = Diligent::TextureSubResData{rgba.data(), 4u};
    }

    Diligent::TextureData initialData{subresources.data(),
                                      static_cast<Diligent::Uint32>(subresources.size())};
    Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
    renderDevice->CreateTexture(textureDesc, &initialData, &texture);
    if (texture == nullptr)
    {
        return false;
    }

    outSrv = createTextureSrv(texture, sampler);
    return outSrv != nullptr;
}

struct UniqueIblKey
{
    common::ResourceId irradianceId  = common::kInvalidResourceId;
    common::ResourceId prefilteredId = common::kInvalidResourceId;

    bool operator==(const UniqueIblKey &rhs) const noexcept
    {
        return irradianceId == rhs.irradianceId && prefilteredId == rhs.prefilteredId;
    }
};

struct UniqueIblKeyHasher
{
    std::size_t operator()(const UniqueIblKey &key) const noexcept
    {
        std::size_t seed = 0u;
        hashCombine(seed, key.irradianceId);
        hashCombine(seed, key.prefilteredId);
        return seed;
    }
};

} // namespace

ForwardOpaquePass::ForwardOpaquePass(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                                     IblQualityTier iblQualityTier)
    : mDevice(device), mResourceManager(resourceManager), mIblQualityTier(iblQualityTier),
      mShaderLibrary(""), mMeshGpuCache("CRESSimNeo.ForwardOpaquePass"),
      mTextureGpuCache("CRESSimNeo.ForwardOpaquePass")
{
}

bool ForwardOpaquePass::initialize()
{
    mShaderLibrary   = gpu::ShaderLibrary(mDevice.shaderSourceDirectory());
    mProgramRegistry = std::make_unique<MaterialProgramRegistry>(mDevice, mShaderLibrary);

    gpu::GpuGraphicsBackendContext backendContext{};
    if (mDevice.tryGetGraphicsBackendContext(backendContext) &&
        backendContext.renderDevice != nullptr)
    {
        mGraphicsContextMask = gpu::contextMaskForId(backendContext.contextId);
        Diligent::SamplerDesc materialSamplerDesc{};
        materialSamplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
        materialSamplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
        materialSamplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
        materialSamplerDesc.AddressU  = Diligent::TEXTURE_ADDRESS_WRAP;
        materialSamplerDesc.AddressV  = Diligent::TEXTURE_ADDRESS_WRAP;
        materialSamplerDesc.AddressW  = Diligent::TEXTURE_ADDRESS_WRAP;
        backendContext.renderDevice->CreateSampler(materialSamplerDesc, &mMaterialSampler);

        Diligent::SamplerDesc shadowSamplerDesc{};
        shadowSamplerDesc.MinFilter      = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MagFilter      = Diligent::FILTER_TYPE_COMPARISON_LINEAR;
        shadowSamplerDesc.MipFilter      = Diligent::FILTER_TYPE_LINEAR;
        shadowSamplerDesc.AddressU       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressV       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.AddressW       = Diligent::TEXTURE_ADDRESS_CLAMP;
        shadowSamplerDesc.ComparisonFunc = Diligent::COMPARISON_FUNC_LESS_EQUAL;
        backendContext.renderDevice->CreateSampler(shadowSamplerDesc, &mShadowSampler);

        Diligent::TextureDesc textureDesc{};
        textureDesc.Name      = "CRESSimNeo.ForwardOpaquePass.FallbackShadow";
        textureDesc.Type      = Diligent::RESOURCE_DIM_TEX_2D_ARRAY;
        textureDesc.Width     = 1;
        textureDesc.Height    = 1;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format    = Diligent::TEX_FORMAT_D32_FLOAT;
        textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_DEPTH_STENCIL;
        textureDesc.Usage     = Diligent::USAGE_DEFAULT;

        Diligent::RefCntAutoPtr<Diligent::ITexture> fallbackTexture;
        backendContext.renderDevice->CreateTexture(textureDesc, nullptr, &fallbackTexture);
        if (fallbackTexture != nullptr)
        {
            mFallbackShadowMapSrv = createArrayShadowSrv(fallbackTexture, mShadowSampler);
        }

        if (!createSolidTexture(backendContext.renderDevice, mMaterialSampler,
                                "CRESSimNeo.ForwardOpaquePass.FallbackBaseColor",
                                Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB, {255u, 255u, 255u, 255u},
                                mFallbackBaseColorSrv) ||
            !createSolidTexture(backendContext.renderDevice, mMaterialSampler,
                                "CRESSimNeo.ForwardOpaquePass.FallbackNormal",
                                Diligent::TEX_FORMAT_RGBA8_UNORM, {128u, 128u, 255u, 255u},
                                mFallbackNormalSrv) ||
            !createSolidTexture(backendContext.renderDevice, mMaterialSampler,
                                "CRESSimNeo.ForwardOpaquePass.FallbackMetallicRoughness",
                                Diligent::TEX_FORMAT_RGBA8_UNORM, {0u, 255u, 255u, 255u},
                                mFallbackMetallicRoughnessSrv) ||
            !createSolidTexture(backendContext.renderDevice, mMaterialSampler,
                                "CRESSimNeo.ForwardOpaquePass.FallbackEmissive",
                                Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB, {255u, 255u, 255u, 255u},
                                mFallbackEmissiveSrv) ||
            !createSolidTexture(backendContext.renderDevice, mMaterialSampler,
                                "CRESSimNeo.ForwardOpaquePass.FallbackAO",
                                Diligent::TEX_FORMAT_RGBA8_UNORM, {255u, 255u, 255u, 255u},
                                mFallbackAoSrv))
        {
            return false;
        }
    }

    mInitialized = true;
    return true;
}

bool ForwardOpaquePass::beginBatchFrame(std::uint32_t currentCameraIndex)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext))
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr)
    {
        return false;
    }
    mGraphicsContextMask = gpu::contextMaskForId(backendContext.contextId);
    if (!ensureConstantBuffers(backendContext.renderDevice) ||
        !ensureEnvironmentIblResources(backendContext.renderDevice, backendContext.graphicsContext))
    {
        return false;
    }

    ForwardPerFrameConstants frameConstants{};
    frameConstants.currentCameraIndex      = currentCameraIndex;
    frameConstants.hasAnyShadowMap         = hasAnyShadowMap() ? 1.0f : 0.0f;
    frameConstants.shadowMinimumVisibility = 0.35f;
    frameConstants.iblSpecularParams       = Diligent::float4{
        mIblQualityTier == IblQualityTier::Full ? mEnvironmentIblPrefilteredMipCount : 0.0f, 0.0f,
        0.0f, 0.0f};

    void *mappedConstants = nullptr;
    backendContext.graphicsContext->MapBuffer(mForwardPerFrameBuffer, Diligent::MAP_WRITE,
                                              Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &frameConstants, sizeof(frameConstants));
    backendContext.graphicsContext->UnmapBuffer(mForwardPerFrameBuffer, Diligent::MAP_WRITE);
    return true;
}

void ForwardOpaquePass::setGpuSceneView(const GpuEntitySceneView &sceneView) noexcept
{
    mSceneView = sceneView;
}

void ForwardOpaquePass::setPhysicsSceneView(
    const physics::PhysicsGpuSceneView *physicsScene) noexcept
{
    mPhysicsScene = physicsScene;
}

void ForwardOpaquePass::setEnvironmentIbls(const std::vector<EnvironmentIblDesc> *ibls,
                                           std::uint32_t envCount) noexcept
{
    mEnvironmentIbls        = ibls;
    mEnvironmentIblEnvCount = envCount;
}

void ForwardOpaquePass::setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept
{
    mVisiblePairBuffer = buffer;
}

void ForwardOpaquePass::setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept
{
    mBatchCameraBuffer = buffer;
}

void ForwardOpaquePass::setShadowMapTargets(
    const std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> &shadowMapTargets,
    std::uint32_t shadowMapCount)
{
    mShadowMapTargets = shadowMapTargets;
    mShadowMapCount   = std::min<std::uint32_t>(shadowMapCount, kShadowCascadeCount);
}

void ForwardOpaquePass::setLocalShadowResources(gpu::GpuRenderTargetHandle localShadowMap2D,
                                                gpu::GpuRenderTargetHandle pointShadowMap,
                                                Diligent::IBuffer *localShadowViewBuffer,
                                                Diligent::IBuffer *lightShadowAssignmentBuffer,
                                                std::uint32_t localShadowViewCount) noexcept
{
    mLocalShadowMap2D            = localShadowMap2D;
    mPointShadowMap              = pointShadowMap;
    mLocalShadowViewBuffer       = localShadowViewBuffer;
    mLightShadowAssignmentBuffer = lightShadowAssignmentBuffer;
    mLocalShadowViewCount        = localShadowViewCount;
}

bool ForwardOpaquePass::prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                                    const ForwardDrawCommand &drawCommand, DrawSetup &outSetup)
{
    if (!mInitialized)
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext))
    {
        return false;
    }
    if (!backendContext.hasActiveRenderTarget ||
        !(backendContext.activeRenderTargetBinding == targetBinding))
    {
        return false;
    }
    if (backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr)
    {
        return false;
    }
    if (backendContext.activeRenderTargetColorFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return false;
    }

    if (drawCommand.meshId == common::kInvalidResourceId ||
        drawCommand.materialId == common::kInvalidResourceId)
    {
        return false;
    }
    if (drawCommand.indexCount < 3)
    {
        return false;
    }

    MeshGpuCache::CachedBuffers *meshBuffers =
        mMeshGpuCache.getOrCreate(mResourceManager, drawCommand, backendContext.renderDevice);
    if (meshBuffers == nullptr || meshBuffers->vertexBuffer == nullptr ||
        meshBuffers->indexBuffer == nullptr || meshBuffers->indexCount == 0)
    {
        return false;
    }

    if (!ensureConstantBuffers(backendContext.renderDevice) || mProgramRegistry == nullptr)
    {
        return false;
    }

    const MaterialResourceDesc *material =
        mResourceManager.tryGetMaterial(MaterialHandle{drawCommand.materialId});
    if (material == nullptr)
    {
        return false;
    }
    if (!backendContext.activeRenderTargetHasDepth)
    {
        return false;
    }

    const bool transparentMainPass = material->renderMode == MaterialRenderMode::Transparent;
    gpu::GpuRenderTargetDesc targetDesc{};
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
    if (mDevice.renderTargetSystem().tryGetRenderTargetDesc(
            backendContext.activeRenderTargetBinding.target, targetDesc) &&
        targetDesc.depth)
    {
        depthFormat = targetDesc.depthFormat;
    }
    if (depthFormat == Diligent::TEX_FORMAT_UNKNOWN)
    {
        return false;
    }
    const MaterialProgramRegistry::ProgramKey key = MaterialProgramRegistry::buildProgramKey(
        transparentMainPass ? MainPassClass::ForwardTransparent : MainPassClass::ForwardOpaque,
        drawCommand.programFamily, drawCommand.materialFeatureFlags, mIblQualityTier,
        backendContext.activeRenderTargetColorFormat, depthFormat, true, !transparentMainPass,
        transparentMainPass);
    MaterialProgramRegistry::ProgramResources *program = mProgramRegistry->getOrCreateProgram(key);
    if (program == nullptr || program->pipelineState == nullptr)
    {
        return false;
    }
    if (!bindProgramConstants(*program))
    {
        return false;
    }
    if (program->shaderResourceBinding == nullptr ||
        program->sceneBindingGeneration != mSceneView.bindingGeneration)
    {
        program->pipelineState->CreateShaderResourceBinding(&program->shaderResourceBinding, true);
        program->sceneBindingGeneration = mSceneView.bindingGeneration;
    }
    if (program->shaderResourceBinding == nullptr)
    {
        return false;
    }

    if (mSceneView.poses.positionsBuffer == nullptr ||
        mSceneView.poses.orientationsBuffer == nullptr ||
        mSceneView.poses.scalesBuffer == nullptr ||
        mSceneView.renderableMetadataBuffer == nullptr ||
        mSceneView.renderableVisibilityFlagsBuffer == nullptr ||
        mSceneView.preparedCamerasBuffer == nullptr)
    {
        return false;
    }

    outSetup.backendContext = backendContext;
    outSetup.meshBuffers    = meshBuffers;
    outSetup.program        = program;
    return true;
}

bool ForwardOpaquePass::bindShadowMaps(MaterialProgramRegistry::ProgramResources &program)
{
    std::array<Diligent::RefCntAutoPtr<Diligent::ITextureView>, kShadowCascadeCount>
        shadowMapSrvs{};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::RefCntAutoPtr<Diligent::ITextureView> shadowMapSrv = mFallbackShadowMapSrv;
        if (cascadeIdx < mShadowMapCount &&
            mShadowMapTargets[cascadeIdx].id != common::kInvalidResourceId)
        {
            Diligent::ITexture *depthTexture = nullptr;
            if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(
                    mShadowMapTargets[cascadeIdx], depthTexture) &&
                depthTexture != nullptr)
            {
                Diligent::RefCntAutoPtr<Diligent::ITextureView> depthSrv =
                    createArrayShadowSrv(depthTexture, mShadowSampler);
                if (depthSrv != nullptr)
                {
                    shadowMapSrv = depthSrv;
                }
            }
        }
        if (shadowMapSrv == nullptr)
        {
            return false;
        }
        shadowMapSrvs[cascadeIdx] = shadowMapSrv;
    }

    constexpr const char *kShadowMapVarNames[kShadowCascadeCount] = {
        "g_ShadowMap0", "g_ShadowMap1", "g_ShadowMap2", "g_ShadowMap3"};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        Diligent::IShaderResourceVariable *shadowMapVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                             kShadowMapVarNames[cascadeIdx]);
        if (shadowMapVar == nullptr)
        {
            return false;
        }
        shadowMapVar->Set(shadowMapSrvs[cascadeIdx]);
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> localShadowSrv = mFallbackShadowMapSrv;
    if (mLocalShadowMap2D.id != common::kInvalidResourceId)
    {
        Diligent::ITexture *depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(mLocalShadowMap2D,
                                                                        depthTexture) &&
            depthTexture != nullptr)
        {
            Diligent::RefCntAutoPtr<Diligent::ITextureView> depthSrv =
                createArrayShadowSrv(depthTexture, mShadowSampler);
            if (depthSrv != nullptr)
            {
                localShadowSrv = depthSrv;
            }
        }
    }

    Diligent::RefCntAutoPtr<Diligent::ITextureView> pointShadowSrv = mFallbackShadowMapSrv;
    if (mPointShadowMap.id != common::kInvalidResourceId)
    {
        Diligent::ITexture *depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(mPointShadowMap,
                                                                        depthTexture) &&
            depthTexture != nullptr)
        {
            Diligent::RefCntAutoPtr<Diligent::ITextureView> depthSrv =
                createArrayShadowSrv(depthTexture, mShadowSampler);
            if (depthSrv != nullptr)
            {
                pointShadowSrv = depthSrv;
            }
        }
    }

    Diligent::IShaderResourceVariable *localShadowVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_LocalShadowMap");
    Diligent::IShaderResourceVariable *pointShadowVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_PointShadowMap");
    if (localShadowVar == nullptr || pointShadowVar == nullptr || localShadowSrv == nullptr ||
        pointShadowSrv == nullptr)
    {
        return false;
    }
    localShadowVar->Set(localShadowSrv);
    pointShadowVar->Set(pointShadowSrv);
    return true;
}

bool ForwardOpaquePass::bindSceneBuffers(MaterialProgramRegistry::ProgramResources &program,
                                         MaterialProgramFamily programFamily) const
{
    if (program.shaderResourceBinding == nullptr)
    {
        return false;
    }
    struct VariableBinding
    {
        const char *name;
        Diligent::IBuffer *buffer;
    };
    const VariableBinding vertexBindings[] = {
        {"g_EntityPositions", mSceneView.poses.positionsBuffer},
        {"g_EntityOrientations", mSceneView.poses.orientationsBuffer},
        {"g_EntityScales", mSceneView.poses.scalesBuffer},
        {"g_RenderableMetadata", mSceneView.renderableMetadataBuffer},
        {"g_RenderableVisibilityFlags", mSceneView.renderableVisibilityFlagsBuffer},
    };
    for (const VariableBinding &binding : vertexBindings)
    {
        Diligent::IShaderResourceVariable *variable =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             binding.name);
        if (variable == nullptr || binding.buffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *srv =
            binding.buffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (srv == nullptr)
        {
            return false;
        }
        variable->Set(srv);
    }

    Diligent::IShaderResourceVariable *visiblePairsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                         "g_VisiblePairs");
    if (visiblePairsVar != nullptr)
    {
        Diligent::IBuffer *visiblePairBuffer = mVisiblePairBuffer != nullptr
                                                   ? mVisiblePairBuffer
                                                   : mSceneView.renderableVisibilityFlagsBuffer;
        if (visiblePairBuffer == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *visiblePairsSrv =
            visiblePairBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (visiblePairsSrv == nullptr)
        {
            return false;
        }
        visiblePairsVar->Set(visiblePairsSrv);
    }

    if (mSceneView.preparedCamerasBuffer == nullptr)
    {
        return false;
    }
    Diligent::IBufferView *preparedCameraSrv =
        mSceneView.preparedCamerasBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    if (preparedCameraSrv == nullptr)
    {
        return false;
    }
    for (const Diligent::SHADER_TYPE shaderType :
         {Diligent::SHADER_TYPE_VERTEX, Diligent::SHADER_TYPE_PIXEL})
    {
        Diligent::IShaderResourceVariable *preparedCameraVar =
            program.shaderResourceBinding->GetVariableByName(shaderType, "g_PreparedCameras");
        if (preparedCameraVar != nullptr)
        {
            preparedCameraVar->Set(preparedCameraSrv);
        }
    }

    if (mBatchCameraBuffer == nullptr || mSceneView.lightInputsBuffer == nullptr ||
        mSceneView.localLightSelectionBuffer == nullptr)
    {
        return false;
    }

    Diligent::IBufferView *batchCameraSrv =
        mBatchCameraBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    Diligent::IBufferView *lightInputsSrv =
        mSceneView.lightInputsBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    Diligent::IBufferView *localLightSelectionSrv =
        mSceneView.localLightSelectionBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    if (batchCameraSrv == nullptr || lightInputsSrv == nullptr || localLightSelectionSrv == nullptr)
    {
        return false;
    }

    Diligent::IShaderResourceVariable *batchCameraVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                         "g_BatchCameras");
    Diligent::IShaderResourceVariable *lightInputsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_LightInputs");
    Diligent::IShaderResourceVariable *localLightsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_LocalLightSelections");
    Diligent::IShaderResourceVariable *lightShadowAssignmentsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_LightShadowAssignments");
    Diligent::IShaderResourceVariable *localShadowViewsVar =
        program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                         "g_LocalShadowViews");
    if (batchCameraVar == nullptr || lightInputsVar == nullptr || localLightsVar == nullptr ||
        lightShadowAssignmentsVar == nullptr || localShadowViewsVar == nullptr ||
        mLightShadowAssignmentBuffer == nullptr || mLocalShadowViewBuffer == nullptr)
    {
        return false;
    }
    Diligent::IBufferView *lightShadowAssignmentsSrv =
        mLightShadowAssignmentBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    Diligent::IBufferView *localShadowViewsSrv =
        mLocalShadowViewBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
    if (lightShadowAssignmentsSrv == nullptr || localShadowViewsSrv == nullptr)
    {
        return false;
    }
    batchCameraVar->Set(batchCameraSrv);
    lightInputsVar->Set(lightInputsSrv);
    localLightsVar->Set(localLightSelectionSrv);
    lightShadowAssignmentsVar->Set(lightShadowAssignmentsSrv);
    localShadowViewsVar->Set(localShadowViewsSrv);

    if (programFamily == MaterialProgramFamily::SoftBodyLit)
    {
        if (mPhysicsScene == nullptr || mPhysicsScene->soft.renderPositionsBuffer == nullptr ||
            mPhysicsScene->soft.renderNormalsBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *softPositionVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             "g_SoftBodyRenderPositions");
        Diligent::IShaderResourceVariable *normalVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             "g_SoftBodyVertexNormals");
        if (softPositionVar == nullptr || normalVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *softPositionSrv =
            mPhysicsScene->soft.renderPositionsBuffer->GetDefaultView(
                Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        Diligent::IBufferView *normalSrv = mPhysicsScene->soft.renderNormalsBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (softPositionSrv == nullptr || normalSrv == nullptr)
        {
            return false;
        }
        softPositionVar->Set(softPositionSrv);
        normalVar->Set(normalSrv);
    }
    else if (programFamily == MaterialProgramFamily::CurveLit)
    {
        if (mPhysicsScene == nullptr || mPhysicsScene->curve.positionsBuffer == nullptr ||
            mPhysicsScene->curve.normalsBuffer == nullptr)
        {
            return false;
        }
        Diligent::IShaderResourceVariable *positionVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             "g_CurveRenderPositions");
        Diligent::IShaderResourceVariable *normalVar =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_VERTEX,
                                                             "g_CurveRenderNormals");
        if (positionVar == nullptr || normalVar == nullptr)
        {
            return false;
        }
        Diligent::IBufferView *positionSrv = mPhysicsScene->curve.positionsBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        Diligent::IBufferView *normalSrv = mPhysicsScene->curve.normalsBuffer->GetDefaultView(
            Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (positionSrv == nullptr || normalSrv == nullptr)
        {
            return false;
        }
        positionVar->Set(positionSrv);
        normalVar->Set(normalSrv);
    }
    return true;
}

bool ForwardOpaquePass::bindMaterialTextures(MaterialProgramRegistry::ProgramResources &program,
                                             Diligent::IRenderDevice *renderDevice,
                                             Diligent::IDeviceContext *graphicsContext,
                                             common::ResourceId materialId)
{
    if (program.shaderResourceBinding == nullptr)
    {
        return false;
    }

    const MaterialResourceDesc *material =
        mResourceManager.tryGetMaterial(MaterialHandle{materialId});
    if (material == nullptr)
    {
        return false;
    }

    struct TextureBinding
    {
        const char *name;
        TextureHandle handle;
        Diligent::ITextureView *fallbackView;
        bool required;
    };

    const TextureBinding bindings[] = {
        {"g_BaseColorTexture", material->baseColorTexture, mFallbackBaseColorSrv.RawPtr(), true},
        {"g_NormalTexture", material->normalTexture, mFallbackNormalSrv.RawPtr(), false},
        {"g_MetallicRoughnessTexture", material->metallicRoughnessTexture,
         mFallbackMetallicRoughnessSrv.RawPtr(), true},
        {"g_EmissiveTexture", material->emissiveTexture, mFallbackEmissiveSrv.RawPtr(), true},
        {"g_AoTexture", material->aoTexture, mFallbackAoSrv.RawPtr(), true},
    };

    for (const TextureBinding &binding : bindings)
    {
        Diligent::ITextureView *textureView = binding.fallbackView;
        if (binding.handle.id != common::kInvalidResourceId)
        {
            TextureGpuCache::CachedTexture *cachedTexture =
                mTextureGpuCache.getOrCreate(mResourceManager, binding.handle, renderDevice,
                                             graphicsContext, mMaterialSampler.RawPtr());
            if (cachedTexture != nullptr && cachedTexture->shaderResourceView != nullptr)
            {
                textureView = cachedTexture->shaderResourceView;
            }
        }

        if (textureView == nullptr)
        {
            return false;
        }

        Diligent::IShaderResourceVariable *variable =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                             binding.name);
        if (variable == nullptr)
        {
            if (binding.required)
            {
                return false;
            }
            continue;
        }
        variable->Set(textureView);
    }

    return true;
}

bool ForwardOpaquePass::ensureEnvironmentIblResources(Diligent::IRenderDevice *renderDevice,
                                                      Diligent::IDeviceContext *graphicsContext)
{
    if (renderDevice == nullptr || graphicsContext == nullptr)
    {
        return false;
    }

    if (mIblQualityTier == IblQualityTier::Off)
    {
        mIrradianceArraySrv.Release();
        mPrefilteredSpecularArraySrv.Release();
        mBrdfLutSrv.Release();
        mEnvironmentIblLookupBuffer.Release();
        mEnvironmentIblLookupCapacity      = 0u;
        mEnvironmentIblPrefilteredMipCount = 0.0f;
        mEnvironmentIblStateHash           = 0u;
        return true;
    }

    std::size_t stateHash = 0u;
    hashCombine(stateHash, static_cast<std::uint32_t>(mIblQualityTier));
    hashCombine(stateHash, mEnvironmentIblEnvCount);
    if (mEnvironmentIbls != nullptr)
    {
        for (const EnvironmentIblDesc &ibl : *mEnvironmentIbls)
        {
            hashCombine(stateHash, ibl.irradianceCubemap.id);
            if (mIblQualityTier == IblQualityTier::Full)
            {
                hashCombine(stateHash, ibl.prefilteredSpecularCubemap.id);
            }
            hashCombine(stateHash, ibl.intensity);
        }
    }
    const bool requiresPrefiltered = (mIblQualityTier == IblQualityTier::Full);
    const bool requiresBrdfLut     = (mIblQualityTier == IblQualityTier::Full);
    const bool hasCachedResources =
        mIrradianceArraySrv != nullptr && mEnvironmentIblLookupBuffer != nullptr &&
        (!requiresPrefiltered || mPrefilteredSpecularArraySrv != nullptr) &&
        (!requiresBrdfLut || mBrdfLutSrv != nullptr);
    if (hasCachedResources && stateHash == mEnvironmentIblStateHash)
    {
        return true;
    }

    std::vector<EnvironmentIblLookupEntry> lookupEntries(mEnvironmentIblEnvCount);
    std::vector<const TextureResourceDesc *> irradianceTextures;
    std::vector<const TextureResourceDesc *> prefilteredTextures;
    const TextureResourceDesc *canonicalIrradiance  = nullptr;
    const TextureResourceDesc *canonicalPrefiltered = nullptr;
    std::unordered_map<UniqueIblKey, std::uint32_t, UniqueIblKeyHasher> uniqueSliceIndices;

    const auto tryGetTexture = [&](TextureHandle handle) -> const TextureResourceDesc *
    {
        if (handle.id == common::kInvalidResourceId)
        {
            return nullptr;
        }
        return mResourceManager.tryGetTexture(handle);
    };

    if (mEnvironmentIbls != nullptr)
    {
        for (std::uint32_t envIndex = 0u;
             envIndex <
             std::min<std::uint32_t>(mEnvironmentIblEnvCount,
                                     static_cast<std::uint32_t>(mEnvironmentIbls->size()));
             ++envIndex)
        {
            const EnvironmentIblDesc &ibl         = (*mEnvironmentIbls)[envIndex];
            const TextureResourceDesc *irradiance = tryGetTexture(ibl.irradianceCubemap);
            const TextureResourceDesc *prefiltered =
                requiresPrefiltered ? tryGetTexture(ibl.prefilteredSpecularCubemap) : nullptr;
            if (irradiance == nullptr || irradiance->dimension != TextureDimension::TextureCube)
            {
                continue;
            }
            if (requiresPrefiltered &&
                (prefiltered == nullptr || prefiltered->dimension != TextureDimension::TextureCube))
            {
                continue;
            }

            if (canonicalIrradiance == nullptr)
            {
                canonicalIrradiance = irradiance;
            }
            if (requiresPrefiltered && canonicalPrefiltered == nullptr)
            {
                canonicalPrefiltered = prefiltered;
            }
            if (canonicalIrradiance == nullptr ||
                !isTextureArrayCompatible(*canonicalIrradiance, *irradiance))
            {
                continue;
            }
            if (requiresPrefiltered &&
                (canonicalPrefiltered == nullptr ||
                 !isTextureArrayCompatible(*canonicalPrefiltered, *prefiltered)))
            {
                continue;
            }

            UniqueIblKey key{};
            key.irradianceId          = ibl.irradianceCubemap.id;
            key.prefilteredId         = requiresPrefiltered ? ibl.prefilteredSpecularCubemap.id
                                                            : common::kInvalidResourceId;
            const auto [it, inserted] = uniqueSliceIndices.emplace(
                key, static_cast<std::uint32_t>(irradianceTextures.size()));
            if (inserted)
            {
                irradianceTextures.push_back(irradiance);
                if (requiresPrefiltered)
                {
                    prefilteredTextures.push_back(prefiltered);
                }
            }

            lookupEntries[envIndex].sliceIndex = it->second;
            lookupEntries[envIndex].enabled    = 1u;
            lookupEntries[envIndex].intensity  = std::max(ibl.intensity, 0.0f);
        }
    }

    const auto uploadLookupBuffer =
        [&](const std::vector<EnvironmentIblLookupEntry> &entries) -> bool
    {
        if (!gpu::detail::ensureStructuredBufferCapacity(
                renderDevice, "CRESSimNeo.ForwardOpaquePass.EnvironmentIblLookup",
                sizeof(EnvironmentIblLookupEntry), static_cast<std::uint32_t>(entries.size()), 1u,
                Diligent::BIND_SHADER_RESOURCE, Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE,
                mGraphicsContextMask, mEnvironmentIblLookupBuffer, mEnvironmentIblLookupCapacity))
        {
            return false;
        }
        if (entries.empty())
        {
            return true;
        }

        graphicsContext->UpdateBuffer(
            mEnvironmentIblLookupBuffer, 0u,
            static_cast<Diligent::Uint32>(entries.size() * sizeof(EnvironmentIblLookupEntry)),
            entries.data(), Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
        return true;
    };

    const auto buildArrayTexture =
        [&](const std::vector<const TextureResourceDesc *> &textures,
            const TextureResourceDesc &prototype, Diligent::RESOURCE_DIMENSION type,
            std::uint32_t layersPerTexture, const char *debugName,
            Diligent::RefCntAutoPtr<Diligent::ITextureView> &outSrv) -> bool
    {
        const std::uint32_t arrayTextures =
            std::max<std::uint32_t>(static_cast<std::uint32_t>(textures.size()), 1u);
        const std::uint32_t totalArrayLayers = arrayTextures * layersPerTexture;
        const std::uint32_t bpp              = bytesPerPixel(prototype.pixelFormat);
        if (bpp == 0u)
        {
            return false;
        }

        Diligent::TextureDesc textureDesc{};
        textureDesc.Name      = debugName;
        textureDesc.Type      = type;
        textureDesc.Width     = prototype.width;
        textureDesc.Height    = prototype.height;
        textureDesc.MipLevels = prototype.mipLevelCount;
        textureDesc.ArraySize = totalArrayLayers;
        textureDesc.Format    = resolveTextureFormat(prototype);
        textureDesc.BindFlags = Diligent::BIND_SHADER_RESOURCE;
        textureDesc.Usage     = Diligent::USAGE_IMMUTABLE;

        std::vector<Diligent::TextureSubResData> subresources(
            static_cast<std::size_t>(textureDesc.MipLevels) *
            static_cast<std::size_t>(totalArrayLayers));
        for (std::uint32_t textureIndex = 0u; textureIndex < arrayTextures; ++textureIndex)
        {
            for (std::uint32_t layer = 0u; layer < layersPerTexture; ++layer)
            {
                const std::uint32_t dstLayer = textureIndex * layersPerTexture + layer;
                for (std::uint32_t mipLevel = 0u; mipLevel < prototype.mipLevelCount; ++mipLevel)
                {
                    const std::uint32_t mipWidth  = std::max(prototype.width >> mipLevel, 1u);
                    const std::uint32_t rowStride = mipWidth * bpp;
                    const std::size_t dstIndex =
                        diligentSubresourceIndex(dstLayer, mipLevel, prototype.mipLevelCount);
                    if (textureIndex < textures.size())
                    {
                        const TextureResourceDesc &source = *textures[textureIndex];
                        const auto &sourceSubresource =
                            source
                                .subresources[subresourceIndex(mipLevel, layer, layersPerTexture)];
                        subresources[dstIndex] = Diligent::TextureSubResData{
                            sourceSubresource.pixelData.data(), rowStride};
                    }
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

        outSrv = createTextureSrv(texture, mMaterialSampler);
        return outSrv != nullptr;
    };

    if (!uploadLookupBuffer(lookupEntries))
    {
        return false;
    }

    if (canonicalIrradiance != nullptr && !irradianceTextures.empty())
    {
        if (!buildArrayTexture(irradianceTextures, *canonicalIrradiance,
                               Diligent::RESOURCE_DIM_TEX_CUBE_ARRAY, 6u,
                               "CRESSimNeo.ForwardOpaquePass.IrradianceArray", mIrradianceArraySrv))
        {
            return false;
        }

        if (requiresPrefiltered && canonicalPrefiltered != nullptr && !prefilteredTextures.empty())
        {
            if (!buildArrayTexture(prefilteredTextures, *canonicalPrefiltered,
                                   Diligent::RESOURCE_DIM_TEX_CUBE_ARRAY, 6u,
                                   "CRESSimNeo.ForwardOpaquePass.PrefilteredSpecularArray",
                                   mPrefilteredSpecularArraySrv))
            {
                return false;
            }
            mEnvironmentIblPrefilteredMipCount =
                static_cast<float>(std::max(canonicalPrefiltered->mipLevelCount, 1u));
            if (mBrdfLutSrv == nullptr &&
                !createBrdfLutTexture(renderDevice, mMaterialSampler,
                                      "CRESSimNeo.ForwardOpaquePass.BuiltInBrdfLut", mBrdfLutSrv))
            {
                return false;
            }
        }
        else
        {
            mPrefilteredSpecularArraySrv.Release();
            mBrdfLutSrv.Release();
            mEnvironmentIblPrefilteredMipCount = 0.0f;
        }
    }
    else
    {
        if (!createFallbackCubeArray(renderDevice, mMaterialSampler,
                                     "CRESSimNeo.ForwardOpaquePass.FallbackIrradianceArray",
                                     Diligent::TEX_FORMAT_RGBA8_UNORM, {0u, 0u, 0u, 0u},
                                     mIrradianceArraySrv))
        {
            return false;
        }
        mPrefilteredSpecularArraySrv.Release();
        mBrdfLutSrv.Release();
        mEnvironmentIblPrefilteredMipCount = 0.0f;
    }

    mEnvironmentIblStateHash = stateHash;
    return true;
}

bool ForwardOpaquePass::bindEnvironmentIblResources(
    MaterialProgramRegistry::ProgramResources &program) const
{
    if (mIblQualityTier == IblQualityTier::Off)
    {
        return true;
    }

    if (program.shaderResourceBinding == nullptr || mEnvironmentIblLookupBuffer == nullptr ||
        mIrradianceArraySrv == nullptr)
    {
        return false;
    }

    struct TextureBinding
    {
        const char *name;
        Diligent::ITextureView *view;
    };
    std::vector<TextureBinding> textureBindings;
    textureBindings.push_back({"g_IrradianceMap", mIrradianceArraySrv.RawPtr()});
    if (mIblQualityTier == IblQualityTier::Full)
    {
        textureBindings.push_back(
            {"g_PrefilteredSpecularMap", mPrefilteredSpecularArraySrv.RawPtr()});
        textureBindings.push_back({"g_BrdfLut", mBrdfLutSrv.RawPtr()});
    }

    for (const TextureBinding &binding : textureBindings)
    {
        Diligent::IShaderResourceVariable *variable =
            program.shaderResourceBinding->GetVariableByName(Diligent::SHADER_TYPE_PIXEL,
                                                             binding.name);
        if (variable == nullptr || binding.view == nullptr)
        {
            return false;
        }
        variable->Set(binding.view);
    }

    Diligent::IShaderResourceVariable *lookupVar = program.shaderResourceBinding->GetVariableByName(
        Diligent::SHADER_TYPE_PIXEL, "g_EnvironmentIblLookup");
    if (lookupVar != nullptr)
    {
        Diligent::IBufferView *lookupSrv =
            mEnvironmentIblLookupBuffer->GetDefaultView(Diligent::BUFFER_VIEW_SHADER_RESOURCE);
        if (lookupSrv == nullptr)
        {
            return false;
        }
        lookupVar->Set(lookupSrv);
    }

    return true;
}

bool ForwardOpaquePass::updatePerDrawConstants(Diligent::IDeviceContext *graphicsContext,
                                               const ForwardDrawCommand &drawCommand)
{
    PerObjectConstants objectConstants{};
    objectConstants.instanceIndex     = drawCommand.instanceIndex;
    objectConstants.drawListOffset    = drawCommand.drawListOffset;
    objectConstants.useDrawListBuffer = drawCommand.useDrawListBuffer;

    const MaterialResourceDesc *material =
        mResourceManager.tryGetMaterial(MaterialHandle{drawCommand.materialId});
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

    void *mappedConstants = nullptr;
    graphicsContext->MapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                               mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &objectConstants, sizeof(objectConstants));
    graphicsContext->UnmapBuffer(mPerObjectBuffer, Diligent::MAP_WRITE);

    mappedConstants = nullptr;
    graphicsContext->MapBuffer(mForwardPerMaterialBuffer, Diligent::MAP_WRITE,
                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &materialConstants, sizeof(materialConstants));
    graphicsContext->UnmapBuffer(mForwardPerMaterialBuffer, Diligent::MAP_WRITE);
    return true;
}

void ForwardOpaquePass::bindGeometry(Diligent::IDeviceContext *graphicsContext,
                                     const MeshGpuCache::CachedBuffers &meshBuffers) const
{
    const Diligent::Uint64 vertexOffset = 0;
    Diligent::IBuffer *vertexBuffers[]  = {meshBuffers.vertexBuffer};
    graphicsContext->SetVertexBuffers(0, 1, vertexBuffers, &vertexOffset,
                                      Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION,
                                      Diligent::SET_VERTEX_BUFFERS_FLAG_RESET);
    graphicsContext->SetIndexBuffer(meshBuffers.indexBuffer, 0,
                                    Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
}

bool ForwardOpaquePass::drawIndexed(const gpu::GpuRenderTargetBinding &targetBinding,
                                    const ForwardDrawCommand &drawCommand)
{
    DrawSetup setup{};
    if (!prepareDraw(targetBinding, drawCommand, setup) || !bindShadowMaps(*setup.program))
    {
        return false;
    }
    if (!bindSceneBuffers(*setup.program, drawCommand.programFamily))
    {
        return false;
    }
    if (!bindMaterialTextures(*setup.program, setup.backendContext.renderDevice,
                              setup.backendContext.graphicsContext, drawCommand.materialId))
    {
        return false;
    }
    if (!bindEnvironmentIblResources(*setup.program))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.graphicsContext, drawCommand))
    {
        return false;
    }
    bindGeometry(setup.backendContext.graphicsContext, *setup.meshBuffers);
    setup.backendContext.graphicsContext->SetPipelineState(setup.program->pipelineState);
    setup.backendContext.graphicsContext->CommitShaderResources(
        setup.program->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedAttribs drawAttrs{};
    drawAttrs.IndexType    = Diligent::VT_UINT32;
    drawAttrs.NumIndices   = drawCommand.indexCount;
    drawAttrs.NumInstances = 1u;
    drawAttrs.Flags        = Diligent::DRAW_FLAG_VERIFY_ALL;
    setup.backendContext.graphicsContext->DrawIndexed(drawAttrs);
    return true;
}

bool ForwardOpaquePass::drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                                     const ForwardDrawCommand &drawCommand,
                                     Diligent::IBuffer *indirectArgsBuffer,
                                     Diligent::Uint64 argsOffsetBytes)
{
    DrawSetup setup{};
    if (!prepareDraw(targetBinding, drawCommand, setup) || indirectArgsBuffer == nullptr ||
        !bindShadowMaps(*setup.program))
    {
        return false;
    }
    if (!bindSceneBuffers(*setup.program, drawCommand.programFamily))
    {
        return false;
    }
    if (!bindMaterialTextures(*setup.program, setup.backendContext.renderDevice,
                              setup.backendContext.graphicsContext, drawCommand.materialId))
    {
        return false;
    }
    if (!bindEnvironmentIblResources(*setup.program))
    {
        return false;
    }
    if (!updatePerDrawConstants(setup.backendContext.graphicsContext, drawCommand))
    {
        return false;
    }
    bindGeometry(setup.backendContext.graphicsContext, *setup.meshBuffers);
    setup.backendContext.graphicsContext->SetPipelineState(setup.program->pipelineState);
    setup.backendContext.graphicsContext->CommitShaderResources(
        setup.program->shaderResourceBinding, Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    Diligent::DrawIndexedIndirectAttribs drawAttrs{};
    drawAttrs.IndexType      = Diligent::VT_UINT32;
    drawAttrs.pAttribsBuffer = indirectArgsBuffer;
    drawAttrs.DrawArgsOffset = argsOffsetBytes;
    drawAttrs.Flags          = Diligent::DRAW_FLAG_VERIFY_ALL;
    drawAttrs.AttribsBufferStateTransitionMode =
        Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION;
    setup.backendContext.graphicsContext->DrawIndexedIndirect(drawAttrs);
    return true;
}

std::size_t ForwardOpaquePass::cachedProgramCount() const noexcept
{
    return mProgramRegistry != nullptr ? mProgramRegistry->cachedProgramCount() : 0u;
}

bool ForwardOpaquePass::ensureConstantBuffers(Diligent::IRenderDevice *renderDevice)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    if (mForwardPerFrameBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name           = "CRESSimNeo.ForwardOpaquePass.GriphicsForwardPerFrame";
        constantBufferDesc.Size           = sizeof(ForwardPerFrameConstants);
        constantBufferDesc.Usage          = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
        constantBufferDesc.ImmediateContextMask = mGraphicsContextMask;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mForwardPerFrameBuffer);
        if (mForwardPerFrameBuffer == nullptr)
        {
            return false;
        }
    }

    if (mPerObjectBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name                 = "CRESSimNeo.ForwardOpaquePass.GraphicsPerObject";
        constantBufferDesc.Size                 = sizeof(PerObjectConstants);
        constantBufferDesc.Usage                = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantBufferDesc.ImmediateContextMask = mGraphicsContextMask;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mPerObjectBuffer);
        if (mPerObjectBuffer == nullptr)
        {
            return false;
        }
    }

    if (mForwardPerMaterialBuffer == nullptr)
    {
        Diligent::BufferDesc constantBufferDesc{};
        constantBufferDesc.Name      = "CRESSimNeo.ForwardOpaquePass.GraphicsForwardPerMaterial";
        constantBufferDesc.Size      = sizeof(ForwardPerMaterialConstants);
        constantBufferDesc.Usage     = Diligent::USAGE_DYNAMIC;
        constantBufferDesc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
        constantBufferDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
        constantBufferDesc.ImmediateContextMask = mGraphicsContextMask;
        renderDevice->CreateBuffer(constantBufferDesc, nullptr, &mForwardPerMaterialBuffer);
        if (mForwardPerMaterialBuffer == nullptr)
        {
            return false;
        }
    }

    return true;
}

bool ForwardOpaquePass::bindProgramConstants(MaterialProgramRegistry::ProgramResources &program)
{
    if (program.pipelineState == nullptr || mForwardPerFrameBuffer == nullptr ||
        mPerObjectBuffer == nullptr || mForwardPerMaterialBuffer == nullptr)
    {
        return false;
    }

    auto bindIfPresent = [&](Diligent::SHADER_TYPE shaderType, const char *varName,
                             Diligent::IBuffer *buffer, bool required)
    {
        Diligent::IShaderResourceVariable *variable =
            program.pipelineState->GetStaticVariableByName(shaderType, varName);
        if (variable == nullptr)
        {
            return !required;
        }
        variable->Set(buffer);
        return true;
    };

    if (!bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GriphicsForwardPerFrame",
                       mForwardPerFrameBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GriphicsForwardPerFrame",
                       mForwardPerFrameBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GraphicsPerObject", mPerObjectBuffer, true) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GraphicsPerObject", mPerObjectBuffer, false) ||
        !bindIfPresent(Diligent::SHADER_TYPE_VERTEX, "GraphicsForwardPerMaterial",
                       mForwardPerMaterialBuffer, false) ||
        !bindIfPresent(Diligent::SHADER_TYPE_PIXEL, "GraphicsForwardPerMaterial",
                       mForwardPerMaterialBuffer, true))
    {
        return false;
    }

    return true;
}

bool ForwardOpaquePass::hasAnyShadowMap() const
{
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < mShadowMapCount; ++cascadeIdx)
    {
        Diligent::ITexture *depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(
                mShadowMapTargets[cascadeIdx], depthTexture) &&
            depthTexture != nullptr)
        {
            return true;
        }
    }
    if (mLocalShadowMap2D.id != common::kInvalidResourceId)
    {
        Diligent::ITexture *depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(mLocalShadowMap2D,
                                                                        depthTexture) &&
            depthTexture != nullptr)
        {
            return true;
        }
    }
    if (mPointShadowMap.id != common::kInvalidResourceId)
    {
        Diligent::ITexture *depthTexture = nullptr;
        if (mDevice.renderTargetSystem().tryGetRenderTargetDepthTexture(mPointShadowMap,
                                                                        depthTexture) &&
            depthTexture != nullptr)
        {
            return true;
        }
    }
    return false;
}

} // namespace cressim::neo::graphics::detail
