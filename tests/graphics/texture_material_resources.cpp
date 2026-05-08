#include "common/logger.h"
#include "engine/runtime.h"
#include "graphics/services/texture_gpu_cache.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuGraphicsBackendContext;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::MaterialFeatureFlags;
using cressim::neo::graphics::MaterialRenderMode;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureDimension;
using cressim::neo::graphics::TextureHandle;
using cressim::neo::graphics::TexturePixelFormat;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::graphics::detail::TextureGpuCache;

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

TextureResourceDesc makeTextureDesc(const char *debugName, TextureColorSpace colorSpace,
                                    std::initializer_list<std::uint8_t> pixels)
{
    TextureResourceDesc desc{};
    desc.debugName = debugName;
    desc.width = 1u;
    desc.height = 1u;
    desc.colorSpace = colorSpace;
    desc.pixelData.assign(pixels.begin(), pixels.end());
    return desc;
}

TextureResourceDesc makeHdrBrdfLutDesc()
{
    TextureResourceDesc desc{};
    desc.debugName   = "TextureMaterial.HdrBrdfLut";
    desc.width       = 2u;
    desc.height      = 2u;
    desc.mipLevelCount = 1u;
    desc.dimension   = TextureDimension::Texture2D;
    desc.pixelFormat = TexturePixelFormat::RGBA16F;
    desc.colorSpace  = TextureColorSpace::Linear;
    desc.subresources.resize(1u);

    const std::array<float, 16> texels = {
        0.10f, 0.20f, 0.0f, 1.0f,
        0.20f, 0.30f, 0.0f, 1.0f,
        0.30f, 0.40f, 0.0f, 1.0f,
        0.40f, 0.50f, 0.0f, 1.0f,
    };
    desc.subresources.front().pixelData.resize(texels.size() * sizeof(std::uint16_t));
    auto *encoded = reinterpret_cast<std::uint16_t *>(desc.subresources.front().pixelData.data());
    for (std::size_t i = 0; i < texels.size(); ++i)
    {
        encoded[i] = encodeFloat16(texels[i]);
    }
    return desc;
}

TextureResourceDesc makeHdrPrefilteredCubeDesc()
{
    TextureResourceDesc desc{};
    desc.debugName     = "TextureMaterial.HdrCube";
    desc.width         = 2u;
    desc.height        = 2u;
    desc.mipLevelCount = 2u;
    desc.dimension     = TextureDimension::TextureCube;
    desc.pixelFormat   = TexturePixelFormat::RGBA16F;
    desc.colorSpace    = TextureColorSpace::Linear;
    desc.subresources.resize(12u);

    for (std::uint32_t mip = 0u; mip < desc.mipLevelCount; ++mip)
    {
        const std::uint32_t mipWidth = std::max(desc.width >> mip, 1u);
        const std::uint32_t mipHeight = std::max(desc.height >> mip, 1u);
        const std::size_t texelCount =
            static_cast<std::size_t>(mipWidth) * static_cast<std::size_t>(mipHeight) * 4u;
        for (std::uint32_t face = 0u; face < 6u; ++face)
        {
            auto &subresource = desc.subresources[mip * 6u + face];
            subresource.pixelData.resize(texelCount * sizeof(std::uint16_t));
            auto *encoded = reinterpret_cast<std::uint16_t *>(subresource.pixelData.data());
            for (std::size_t texel = 0u; texel < texelCount; texel += 4u)
            {
                encoded[texel + 0u] = encodeFloat16(0.1f * static_cast<float>(face + 1u));
                encoded[texel + 1u] = encodeFloat16(0.05f * static_cast<float>(mip + 1u));
                encoded[texel + 2u] = encodeFloat16(0.02f);
                encoded[texel + 3u] = encodeFloat16(1.0f);
            }
        }
    }

    return desc;
}

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = cressim::neo::gpu::GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_WARNING(
            "Skipping texture material resource GPU checks because Vulkan runtime initialization failed.\n");
        return 0;
    }

    auto &resources = runtime.getResources();
    const auto srgbTexture = resources.registerTexture(
        makeTextureDesc("TextureMaterial.Srgb", TextureColorSpace::Srgb, {255u, 64u, 32u, 255u}));
    const auto linearTexture = resources.registerTexture(
        makeTextureDesc("TextureMaterial.Linear", TextureColorSpace::Linear, {128u, 255u, 0u, 255u}));
    const auto hdrBrdfLut = resources.registerTexture(makeHdrBrdfLutDesc());
    const auto hdrCube = resources.registerTexture(makeHdrPrefilteredCubeDesc());

    MaterialResourceDesc materialDesc{};
    materialDesc.debugName = "TextureMaterial.Material";
    materialDesc.baseColorTexture = srgbTexture;
    materialDesc.normalTexture = TextureHandle{};
    materialDesc.metallicRoughnessTexture = linearTexture;
    materialDesc.emissiveTexture = srgbTexture;
    materialDesc.aoTexture = linearTexture;
    materialDesc.emissiveFactor = {0.25f, 0.5f, 1.0f};
    const auto material = resources.registerMaterial(materialDesc);

    const auto *storedMaterial = resources.tryGetMaterial(material);
    const auto *storedSrgb = resources.tryGetTexture(srgbTexture);
    const auto *storedLinear = resources.tryGetTexture(linearTexture);
    const auto *storedHdrBrdfLut = resources.tryGetTexture(hdrBrdfLut);
    const auto *storedHdrCube = resources.tryGetTexture(hdrCube);
    if (storedMaterial == nullptr || storedSrgb == nullptr || storedLinear == nullptr ||
        storedHdrBrdfLut == nullptr || storedHdrCube == nullptr)
    {
        CRESSIM_LOG_ERROR("Registered resources could not be retrieved.\n");
        runtime.shutdown();
        return 1;
    }

    if (storedMaterial->baseColorTexture.id != srgbTexture.id ||
        storedMaterial->normalTexture.id != cressim::neo::common::kInvalidResourceId ||
        storedMaterial->metallicRoughnessTexture.id != linearTexture.id ||
        storedMaterial->emissiveTexture.id != srgbTexture.id ||
        storedMaterial->aoTexture.id != linearTexture.id ||
        storedMaterial->renderMode != MaterialRenderMode::Opaque ||
        storedMaterial->renderOrder != 0)
    {
        CRESSIM_LOG_ERROR("Material resource properties were not preserved.\n");
        runtime.shutdown();
        return 1;
    }

    MaterialResourceDesc alphaTestDesc{};
    alphaTestDesc.renderMode = MaterialRenderMode::Cutout;
    if (cressim::neo::graphics::usesTransparentPass(materialDesc) ||
        cressim::neo::graphics::usesTransparentPass(alphaTestDesc) ||
        !cressim::neo::graphics::hasFlag(
            cressim::neo::graphics::effectiveMaterialFeatureFlags(alphaTestDesc),
            MaterialFeatureFlags::AlphaTest))
    {
        CRESSIM_LOG_ERROR("Unexpected opaque or cutout material mode behavior.\n");
        runtime.shutdown();
        return 1;
    }

    MaterialResourceDesc transparentDesc{};
    transparentDesc.renderMode = MaterialRenderMode::Transparent;
    if (!cressim::neo::graphics::usesTransparentPass(transparentDesc) ||
        cressim::neo::graphics::hasFlag(
            cressim::neo::graphics::effectiveMaterialFeatureFlags(transparentDesc),
            MaterialFeatureFlags::AlphaTest))
    {
        CRESSIM_LOG_ERROR("Unexpected transparent material mode behavior.\n");
        runtime.shutdown();
        return 1;
    }

    if (storedSrgb->colorSpace != TextureColorSpace::Srgb ||
        storedLinear->colorSpace != TextureColorSpace::Linear)
    {
        CRESSIM_LOG_ERROR("Texture color space metadata was not preserved.\n");
        runtime.shutdown();
        return 1;
    }
    if (storedHdrBrdfLut->pixelFormat != TexturePixelFormat::RGBA16F ||
        storedHdrBrdfLut->dimension != TextureDimension::Texture2D ||
        storedHdrBrdfLut->subresources.size() != 1u ||
        storedHdrCube->pixelFormat != TexturePixelFormat::RGBA16F ||
        storedHdrCube->dimension != TextureDimension::TextureCube ||
        storedHdrCube->mipLevelCount != 2u || storedHdrCube->subresources.size() != 12u)
    {
        CRESSIM_LOG_ERROR("HDR texture metadata was not preserved.\n");
        runtime.shutdown();
        return 1;
    }

    cressim::neo::gpu::GpuDevice *gpuDevice = runtime.getGpuDevice();
    if (gpuDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("GPU device was not available.\n");
        runtime.shutdown();
        return 1;
    }

    GpuGraphicsBackendContext backendContext{};
    if (!gpuDevice->tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("Graphics backend context was not available.\n");
        runtime.shutdown();
        return 1;
    }

    Diligent::SamplerDesc samplerDesc{};
    samplerDesc.MinFilter = Diligent::FILTER_TYPE_LINEAR;
    samplerDesc.MagFilter = Diligent::FILTER_TYPE_LINEAR;
    samplerDesc.MipFilter = Diligent::FILTER_TYPE_LINEAR;
    Diligent::RefCntAutoPtr<Diligent::ISampler> sampler;
    backendContext.renderDevice->CreateSampler(samplerDesc, &sampler);
    if (sampler == nullptr)
    {
        CRESSIM_LOG_ERROR("Sampler creation failed.\n");
        runtime.shutdown();
        return 1;
    }

    TextureGpuCache textureCache("TextureMaterial.Test");
    TextureGpuCache::CachedTexture *cachedSrgb =
        textureCache.getOrCreate(resources, srgbTexture, backendContext.renderDevice,
                                 backendContext.graphicsContext, sampler.RawPtr());
    TextureGpuCache::CachedTexture *cachedLinear =
        textureCache.getOrCreate(resources, linearTexture, backendContext.renderDevice,
                                 backendContext.graphicsContext, sampler.RawPtr());
    TextureGpuCache::CachedTexture *cachedHdrBrdfLut =
        textureCache.getOrCreate(resources, hdrBrdfLut, backendContext.renderDevice,
                                 backendContext.graphicsContext, sampler.RawPtr());
    TextureGpuCache::CachedTexture *cachedHdrCube =
        textureCache.getOrCreate(resources, hdrCube, backendContext.renderDevice,
                                 backendContext.graphicsContext, sampler.RawPtr());

    if (cachedSrgb == nullptr || cachedLinear == nullptr ||
        cachedHdrBrdfLut == nullptr || cachedHdrCube == nullptr ||
        cachedSrgb->shaderResourceView == nullptr || cachedLinear->shaderResourceView == nullptr ||
        cachedHdrBrdfLut->shaderResourceView == nullptr ||
        cachedHdrCube->shaderResourceView == nullptr)
    {
        CRESSIM_LOG_ERROR("Texture GPU cache failed to create SRVs.\n");
        runtime.shutdown();
        return 1;
    }

    if (cachedSrgb->format != Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB ||
        cachedLinear->format != Diligent::TEX_FORMAT_RGBA8_UNORM ||
        cachedHdrBrdfLut->format != Diligent::TEX_FORMAT_RGBA16_FLOAT ||
        cachedHdrCube->format != Diligent::TEX_FORMAT_RGBA16_FLOAT)
    {
        CRESSIM_LOG_ERROR("Texture GPU cache selected unexpected formats.\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();

    CRESSIM_LOG_INFO("Texture material resource checks passed.\n");
    return 0;
}
