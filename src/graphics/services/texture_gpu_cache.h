#ifndef CRESSIM_NEO_GRAPHICS_SERVICES_TEXTURE_GPU_CACHE_H
#define CRESSIM_NEO_GRAPHICS_SERVICES_TEXTURE_GPU_CACHE_H

#include "graphics/render_resource_manager.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Texture.h"

#include <string>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class TextureGpuCache
{
public:
    explicit TextureGpuCache(std::string debugPrefix);

    struct CachedTexture
    {
        Diligent::TEXTURE_FORMAT format = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::RefCntAutoPtr<Diligent::ITexture> texture;
        Diligent::RefCntAutoPtr<Diligent::ITextureView> shaderResourceView;
    };

    CachedTexture *getOrCreate(const RenderResourceManager &resources, TextureHandle texture,
                               Diligent::IRenderDevice *renderDevice,
                               Diligent::IDeviceContext *immediateContext,
                               Diligent::ISampler *sampler);

private:
    static Diligent::TEXTURE_FORMAT resolveTextureFormat(const TextureResourceDesc &desc) noexcept;
    static std::uint32_t bytesPerPixel(TexturePixelFormat format) noexcept;
    static std::uint32_t arrayLayerCount(TextureDimension dimension) noexcept;

private:
    std::string mDebugPrefix;
    std::unordered_map<common::ResourceId, CachedTexture> mCachedTextures;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_SERVICES_TEXTURE_GPU_CACHE_H
