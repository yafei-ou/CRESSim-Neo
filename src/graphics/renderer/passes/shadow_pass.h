#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/passes/forward_draw_types.h"
#include "graphics/renderer/services/mesh_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

namespace cressim::neo::graphics
{

namespace detail
{

class ShadowPass
{
public:
    explicit ShadowPass(gpu::GpuDevice& device);

    bool initialize();
    bool draw(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand,
              const Diligent::float4x4& lightViewProjectionMatrix);

private:
    struct PerObjectConstants
    {
        Diligent::float4x4 modelMatrix  = Diligent::float4x4::Identity();
        Diligent::float4x4 normalMatrix = Diligent::float4x4::Identity();
    };

    struct ShadowPerPassConstants
    {
        Diligent::float4x4 lightViewProjectionMatrix = Diligent::float4x4::Identity();
    };

    bool createPipeline(Diligent::IRenderDevice* renderDevice);
    bool ensureConstantBuffers(Diligent::IRenderDevice* renderDevice);

private:
    gpu::GpuDevice& mDevice;
    bool mInitialized = false;
    gpu::ShaderLibrary mShaderLibrary;

    MeshGpuCache mMeshGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPipelineState;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mShaderResourceBinding;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mShadowPerPassBuffer;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
