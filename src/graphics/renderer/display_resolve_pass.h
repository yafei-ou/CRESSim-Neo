#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_DISPLAY_RESOLVE_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_DISPLAY_RESOLVE_PASS_H

#include "graphics/renderer/passes/render_pass_types.h"

#include "gpu/gpu_device.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <cstdint>
#include <unordered_map>

namespace cressim::neo::graphics::detail
{

class DisplayResolvePass
{
public:
    explicit DisplayResolvePass(gpu::GpuDevice& device);

    bool initialize();
    bool resolve(const common::FrameContext& frameContext, const DisplayResolveRequest& request);

private:
    struct ResolveConstants
    {
        std::uint32_t layer    = 0u;
        std::uint32_t padding0 = 0u;
        std::uint32_t padding1 = 0u;
        std::uint32_t padding2 = 0u;
    };

    struct PipelineKey
    {
        Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
        Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;

        bool operator==(const PipelineKey& rhs) const noexcept
        {
            return colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat;
        }
    };

    struct PipelineKeyHasher
    {
        std::size_t operator()(const PipelineKey& key) const noexcept;
    };

    bool ensureConstants(Diligent::IRenderDevice* renderDevice);
    Diligent::IPipelineState* getOrCreatePipeline(Diligent::IRenderDevice* renderDevice,
                                                  const PipelineKey& key);
    Diligent::RefCntAutoPtr<Diligent::ITextureView> createArraySrv(
        Diligent::ITexture* texture) const;

    gpu::GpuDevice& mDevice;
    bool mInitialized = false;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mSampler;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mPipelines;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_DISPLAY_RESOLVE_PASS_H
