#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
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
    ShadowPass(gpu::GpuDevice& device, RenderResourceManager& resourceManager);

    bool initialize();
    void setGpuSceneView(const gpu::GpuEntitySceneView& sceneView) noexcept;
    void setVisibleObjectIndexBuffer(Diligent::IBuffer* buffer) noexcept;
    bool drawIndirect(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand,
                      std::uint32_t currentCameraIndex, std::uint32_t cascadeIndex,
                      Diligent::IBuffer* indirectArgsBuffer, Diligent::Uint64 argsOffsetBytes);

private:
    struct DrawSetup
    {
        gpu::GpuBackendContext backendContext{};
        MeshGpuCache::CachedBuffers* meshBuffers = nullptr;
    };

    struct PerObjectConstants
    {
        std::uint32_t instanceIndex     = 0xffffffffu;
        std::uint32_t drawListOffset    = 0u;
        std::uint32_t useDrawListBuffer = 0u;
        std::uint32_t padding0          = 0u;
    };

    struct ShadowPerPassConstants
    {
        std::uint32_t shadowPassParams[4] = {0u, 0u, 0u, 0u};
    };

    bool createPipeline(Diligent::IRenderDevice* renderDevice);
    bool ensureConstantBuffers(Diligent::IRenderDevice* renderDevice);
    bool prepareDraw(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand,
                     DrawSetup& outSetup);
    bool bindSceneBuffers() const;
    bool updatePerDrawConstants(Diligent::IDeviceContext* immediateContext,
                                const ForwardDrawCommand& drawCommand,
                                std::uint32_t currentCameraIndex, std::uint32_t cascadeIndex);
    void bindGeometry(Diligent::IDeviceContext* immediateContext,
                      const MeshGpuCache::CachedBuffers& meshBuffers) const;

private:
    gpu::GpuDevice& mDevice;
    RenderResourceManager& mResourceManager;
    bool mInitialized = false;
    gpu::ShaderLibrary mShaderLibrary;

    MeshGpuCache mMeshGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPipelineState;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mShaderResourceBinding;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mShadowPerPassBuffer;
    gpu::GpuEntitySceneView mSceneView{};
    Diligent::IBuffer* mVisibleObjectIndexBuffer = nullptr;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_SHADOW_PASS_H
