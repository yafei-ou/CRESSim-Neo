#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"
#include "graphics/renderer/services/shader_source_provider.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

#include <cstdint>
#include <unordered_map>

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl;

namespace detail
{

class ShadowPass
{
public:
    explicit ShadowPass(GraphicsDeviceImpl& device);

    bool initialize();
    bool draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand);

private:
    struct CachedMeshGpuData
    {
        std::uint64_t version = 0;
        std::uint32_t indexCount = 0;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> vertexBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> indexBuffer;
    };

    struct DrawConstants
    {
        float modelMatrix[16] = {};
        float lightViewProjectionMatrix[16] = {};
    };

    CachedMeshGpuData* getOrCreateMeshBuffers(
        const ForwardDrawCommand& drawCommand,
        Diligent::IRenderDevice* renderDevice);
    bool createPipeline(Diligent::IRenderDevice* renderDevice);

private:
    GraphicsDeviceImpl& mDevice;
    bool mInitialized = false;
    ShaderSourceProvider mShaderSourceProvider;

    std::unordered_map<common::ResourceId, CachedMeshGpuData> mCachedMeshes;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPipelineState;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mShaderResourceBinding;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mConstantBuffer;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
