#include "graphics/services/texture_gpu_cache.h"

#include "common/logger.h"

#include <algorithm>
#include <utility>

namespace cressim::neo::graphics::detail
{

namespace
{

std::size_t sourceSubresourceIndex(std::uint32_t mipLevel, std::uint32_t layer,
                                   std::uint32_t layerCount) noexcept
{
    return static_cast<std::size_t>(mipLevel) * static_cast<std::size_t>(layerCount) +
           static_cast<std::size_t>(layer);
}

std::size_t diligentSubresourceIndex(std::uint32_t layer, std::uint32_t mipLevel,
                                     std::uint32_t mipLevelCount) noexcept
{
    return static_cast<std::size_t>(layer) * static_cast<std::size_t>(mipLevelCount) +
           static_cast<std::size_t>(mipLevel);
}

} // namespace

TextureGpuCache::TextureGpuCache(std::string debugPrefix) : mDebugPrefix(std::move(debugPrefix)) {}

TextureGpuCache::CachedTexture *TextureGpuCache::getOrCreate(
    const RenderResourceManager &resources, TextureHandle texture,
    Diligent::IRenderDevice *renderDevice, Diligent::IDeviceContext *graphicsContext,
    Diligent::ISampler *sampler)
{
    if (renderDevice == nullptr || texture.id == common::kInvalidResourceId)
    {
        return nullptr;
    }

    const TextureResourceDesc *textureDesc = resources.tryGetTexture(texture);
    if (textureDesc == nullptr)
    {
        return nullptr;
    }

    auto &cachedTexture = mCachedTextures[texture.id];
    if (cachedTexture.shaderResourceView != nullptr)
    {
        return &cachedTexture;
    }

    if (textureDesc->width == 0u || textureDesc->height == 0u)
    {
        CRESSIM_LOG_ERROR("TextureGpuCache rejected texture with invalid size. id=", texture.id,
                          " width=", textureDesc->width, " height=", textureDesc->height, '\n');
        mCachedTextures.erase(texture.id);
        return nullptr;
    }

    const std::uint32_t bpp        = bytesPerPixel(textureDesc->pixelFormat);
    const std::uint32_t layerCount = arrayLayerCount(textureDesc->dimension);
    const std::size_t expectedSubresourceCount =
        static_cast<std::size_t>(std::max(textureDesc->mipLevelCount, 1u)) *
        static_cast<std::size_t>(layerCount);
    if (textureDesc->subresources.size() != expectedSubresourceCount)
    {
        CRESSIM_LOG_ERROR("TextureGpuCache rejected texture with unexpected subresource count. id=",
                          texture.id, " expected=", expectedSubresourceCount,
                          " actual=", textureDesc->subresources.size(), '\n');
        mCachedTextures.erase(texture.id);
        return nullptr;
    }

    std::vector<Diligent::TextureSubResData> subresources(expectedSubresourceCount);
    const std::uint32_t mipLevelCount = std::max(textureDesc->mipLevelCount, 1u);
    for (std::uint32_t layer = 0u; layer < layerCount; ++layer)
    {
        for (std::uint32_t mipLevel = 0u; mipLevel < mipLevelCount; ++mipLevel)
        {
            const std::uint32_t mipWidth   = std::max(textureDesc->width >> mipLevel, 1u);
            const std::uint32_t mipHeight  = std::max(textureDesc->height >> mipLevel, 1u);
            const std::size_t expectedSize = static_cast<std::size_t>(mipWidth) *
                                             static_cast<std::size_t>(mipHeight) *
                                             static_cast<std::size_t>(bpp);
            const auto &subresourceDesc =
                textureDesc->subresources[sourceSubresourceIndex(mipLevel, layer, layerCount)];
            const std::size_t index = diligentSubresourceIndex(layer, mipLevel, mipLevelCount);
            if (subresourceDesc.pixelData.size() != expectedSize)
            {
                CRESSIM_LOG_ERROR("TextureGpuCache rejected texture with unexpected subresource "
                                  "payload size. id=",
                                  texture.id, " subresource=", index, " expected=", expectedSize,
                                  " actual=", subresourceDesc.pixelData.size(), '\n');
                mCachedTextures.erase(texture.id);
                return nullptr;
            }

            subresources[index] = Diligent::TextureSubResData{
                subresourceDesc.pixelData.data(),
                static_cast<Diligent::Uint64>(mipWidth) * static_cast<Diligent::Uint64>(bpp)};
        }
    }

    Diligent::TextureDesc gpuDesc{};
    const std::string textureName = mDebugPrefix + ".Texture";
    gpuDesc.Name                  = textureName.c_str();
    gpuDesc.Type                  = textureDesc->dimension == TextureDimension::TextureCube
                                        ? Diligent::RESOURCE_DIM_TEX_CUBE
                                        : Diligent::RESOURCE_DIM_TEX_2D;
    gpuDesc.Width                 = textureDesc->width;
    gpuDesc.Height                = textureDesc->height;
    gpuDesc.Format                = resolveTextureFormat(*textureDesc);
    gpuDesc.BindFlags             = Diligent::BIND_SHADER_RESOURCE;
    gpuDesc.Usage                 = Diligent::USAGE_IMMUTABLE;
    gpuDesc.MipLevels             = std::max(textureDesc->mipLevelCount, 1u);
    gpuDesc.ArraySize             = layerCount;

    Diligent::TextureData initialData{subresources.data(),
                                      static_cast<Diligent::Uint32>(subresources.size())};
    renderDevice->CreateTexture(gpuDesc, &initialData, &cachedTexture.texture);
    if (cachedTexture.texture == nullptr)
    {
        CRESSIM_LOG_ERROR("TextureGpuCache failed to create GPU texture. id=", texture.id, '\n');
        mCachedTextures.erase(texture.id);
        return nullptr;
    }

    cachedTexture.format = gpuDesc.Format;
    cachedTexture.shaderResourceView =
        cachedTexture.texture->GetDefaultView(Diligent::TEXTURE_VIEW_SHADER_RESOURCE);
    if (cachedTexture.shaderResourceView == nullptr)
    {
        CRESSIM_LOG_ERROR("TextureGpuCache failed to create SRV. id=", texture.id, '\n');
        mCachedTextures.erase(texture.id);
        return nullptr;
    }
    if (sampler != nullptr)
    {
        cachedTexture.shaderResourceView->SetSampler(sampler);
    }

    (void)graphicsContext;

    return &cachedTexture;
}

Diligent::TEXTURE_FORMAT TextureGpuCache::resolveTextureFormat(
    const TextureResourceDesc &desc) noexcept
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

std::uint32_t TextureGpuCache::bytesPerPixel(TexturePixelFormat format) noexcept
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

std::uint32_t TextureGpuCache::arrayLayerCount(TextureDimension dimension) noexcept
{
    return dimension == TextureDimension::TextureCube ? 6u : 1u;
}

} // namespace cressim::neo::graphics::detail
