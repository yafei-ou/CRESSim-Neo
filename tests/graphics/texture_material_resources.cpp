#include "common/logger.h"
#include "engine/runtime.h"
#include "graphics/services/texture_gpu_cache.h"

namespace
{

using cressim::neo::engine::Runtime;
using cressim::neo::engine::RuntimeConfig;
using cressim::neo::gpu::GpuBackend;
using cressim::neo::gpu::GpuBackendContext;
using cressim::neo::graphics::MaterialResourceDesc;
using cressim::neo::graphics::TextureColorSpace;
using cressim::neo::graphics::TextureHandle;
using cressim::neo::graphics::TextureResourceDesc;
using cressim::neo::graphics::detail::TextureGpuCache;

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

} // namespace

int main()
{
    RuntimeConfig config{};
    config.gpuDeviceDesc.preferredBackend = GpuBackend::Vulkan;

    Runtime runtime;
    if (!runtime.initialize(config))
    {
        CRESSIM_LOG_ERROR("Runtime initialization failed.\n");
        return 1;
    }

    auto &resources = runtime.getResources();
    const auto srgbTexture = resources.registerTexture(
        makeTextureDesc("TextureMaterial.Srgb", TextureColorSpace::Srgb, {255u, 64u, 32u, 255u}));
    const auto linearTexture = resources.registerTexture(
        makeTextureDesc("TextureMaterial.Linear", TextureColorSpace::Linear, {128u, 255u, 0u, 255u}));

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
    if (storedMaterial == nullptr || storedSrgb == nullptr || storedLinear == nullptr)
    {
        CRESSIM_LOG_ERROR("Registered resources could not be retrieved.\n");
        runtime.shutdown();
        return 1;
    }

    if (storedMaterial->baseColorTexture.id != srgbTexture.id ||
        storedMaterial->metallicRoughnessTexture.id != linearTexture.id ||
        storedMaterial->emissiveTexture.id != srgbTexture.id ||
        storedMaterial->aoTexture.id != linearTexture.id)
    {
        CRESSIM_LOG_ERROR("Material texture handles were not preserved.\n");
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

    cressim::neo::gpu::GpuDevice *gpuDevice = runtime.getGpuDevice();
    if (gpuDevice == nullptr)
    {
        CRESSIM_LOG_ERROR("GPU device was not available.\n");
        runtime.shutdown();
        return 1;
    }

    GpuBackendContext backendContext{};
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
                                 backendContext.immediateContext, sampler.RawPtr());
    TextureGpuCache::CachedTexture *cachedLinear =
        textureCache.getOrCreate(resources, linearTexture, backendContext.renderDevice,
                                 backendContext.immediateContext, sampler.RawPtr());

    if (cachedSrgb == nullptr || cachedLinear == nullptr ||
        cachedSrgb->shaderResourceView == nullptr || cachedLinear->shaderResourceView == nullptr)
    {
        CRESSIM_LOG_ERROR("Texture GPU cache failed to create SRVs.\n");
        runtime.shutdown();
        return 1;
    }

    if (cachedSrgb->format != Diligent::TEX_FORMAT_RGBA8_UNORM_SRGB ||
        cachedLinear->format != Diligent::TEX_FORMAT_RGBA8_UNORM)
    {
        CRESSIM_LOG_ERROR("Texture GPU cache selected unexpected formats.\n");
        runtime.shutdown();
        return 1;
    }

    runtime.shutdown();

    CRESSIM_LOG_INFO("Texture material resource checks passed.\n");
    return 0;
}
