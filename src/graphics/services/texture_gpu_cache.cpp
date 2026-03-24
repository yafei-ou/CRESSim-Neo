#include "graphics/services/texture_gpu_cache.h"

#include "common/logger.h"

#include <utility>

namespace cressim::neo::graphics::detail
{

TextureGpuCache::TextureGpuCache(std::string debugPrefix) : mDebugPrefix(std::move(debugPrefix)) {}

TextureGpuCache::CachedTexture *TextureGpuCache::getOrCreate(
    const RenderResourceManager &resources, TextureHandle texture,
    Diligent::IRenderDevice *renderDevice, Diligent::IDeviceContext *immediateContext,
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
    const std::size_t expectedSize = static_cast<std::size_t>(textureDesc->width) *
                                     static_cast<std::size_t>(textureDesc->height) *
                                     static_cast<std::size_t>(bpp);
    if (textureDesc->pixelData.size() != expectedSize)
    {
        CRESSIM_LOG_ERROR(
            "TextureGpuCache rejected texture with unexpected pixel payload size. id=", texture.id,
            " expected=", expectedSize, " actual=", textureDesc->pixelData.size(), '\n');
        mCachedTextures.erase(texture.id);
        return nullptr;
    }

    Diligent::TextureDesc gpuDesc{};
    const std::string textureName = mDebugPrefix + ".Texture";
    gpuDesc.Name                  = textureName.c_str();
    gpuDesc.Type                  = Diligent::RESOURCE_DIM_TEX_2D;
    gpuDesc.Width                 = textureDesc->width;
    gpuDesc.Height                = textureDesc->height;
    gpuDesc.Format                = resolveTextureFormat(*textureDesc);
    gpuDesc.BindFlags             = Diligent::BIND_SHADER_RESOURCE;
    gpuDesc.Usage                 = Diligent::USAGE_IMMUTABLE;
    gpuDesc.MipLevels             = 1u;

    Diligent::TextureSubResData subresource{textureDesc->pixelData.data(),
                                            static_cast<Diligent::Uint64>(textureDesc->width) *
                                                static_cast<Diligent::Uint64>(bpp)};
    Diligent::TextureData initialData{&subresource, 1u};
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

    (void)immediateContext;

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
    default:
        return 0u;
    }
}

} // namespace cressim::neo::graphics::detail
