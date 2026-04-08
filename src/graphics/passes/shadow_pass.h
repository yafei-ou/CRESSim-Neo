#ifndef CRESSIM_NEO_GRAPHICS_PASSES_SHADOW_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_SHADOW_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/shader_library.h"
#include "graphics/gpu_scene.h"
#include "graphics/passes/forward_draw_types.h"
#include "graphics/services/mesh_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics::detail
{

namespace physics = cressim::neo::physics;

class ShadowPass
{
public:
    ShadowPass(gpu::GpuDevice &device, RenderResourceManager &resourceManager);

    bool initialize();
    void setGpuSceneView(const GpuEntitySceneView &sceneView) noexcept;
    void setPhysicsSceneView(const physics::PhysicsGpuSceneView *physicsScene) noexcept;
    void setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept;
    void setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept;
    void setLocalShadowViewBuffer(Diligent::IBuffer *buffer) noexcept;
    bool drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                      const ForwardDrawCommand &drawCommand, std::uint32_t shadowMatrixIndex,
                      std::uint32_t shadowPassMode, Diligent::IBuffer *indirectArgsBuffer,
                      Diligent::Uint64 argsOffsetBytes, std::uint32_t drawCount,
                      std::uint32_t drawArgsStride);

private:
    struct DrawSetup
    {
        gpu::GpuGraphicsBackendContext backendContext{};
        MeshGpuCache::CachedBuffers *meshBuffers = nullptr;
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

    bool createPipeline(Diligent::IRenderDevice *renderDevice, MaterialProgramFamily programFamily);
    bool ensureConstantBuffers(Diligent::IRenderDevice *renderDevice);
    bool prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                     const ForwardDrawCommand &drawCommand, DrawSetup &outSetup);
    bool bindSceneBuffers(MaterialProgramFamily programFamily) const;
    bool updatePerDrawConstants(Diligent::IDeviceContext *graphicsContext,
                                const ForwardDrawCommand &drawCommand,
                                std::uint32_t shadowMatrixIndex, std::uint32_t shadowPassMode);
    void bindGeometry(Diligent::IDeviceContext *graphicsContext,
                      const MeshGpuCache::CachedBuffers &meshBuffers) const;

private:
    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    bool mInitialized = false;
    gpu::ShaderLibrary mShaderLibrary;

    MeshGpuCache mMeshGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mPipelineState;
    Diligent::RefCntAutoPtr<Diligent::IPipelineState> mSoftBodyPipelineState;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mShaderResourceBinding;
    Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding> mSoftBodyShaderResourceBinding;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mShadowPerPassBuffer;
    GpuEntitySceneView mSceneView{};
    const physics::PhysicsGpuSceneView *mPhysicsScene = nullptr;
    Diligent::IBuffer *mVisiblePairBuffer             = nullptr;
    Diligent::IBuffer *mBatchCameraBuffer             = nullptr;
    Diligent::IBuffer *mLocalShadowViewBuffer         = nullptr;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_SHADOW_PASS_H
