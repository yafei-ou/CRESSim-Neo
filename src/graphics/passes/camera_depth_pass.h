#ifndef CRESSIM_NEO_GRAPHICS_PASSES_CAMERA_DEPTH_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_CAMERA_DEPTH_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/shader_library.h"
#include "graphics/gpu_scene.h"
#include "graphics/passes/forward_draw_types.h"
#include "graphics/services/mesh_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/PipelineState.h"

#include <unordered_map>

namespace cressim::neo::physics
{
struct PhysicsGpuSceneView;
}

namespace cressim::neo::graphics::detail
{

namespace physics = cressim::neo::physics;

class CameraDepthPass
{
public:
    CameraDepthPass(gpu::GpuDevice &device, RenderResourceManager &resourceManager);

    bool initialize();
    bool beginCamera(std::uint32_t currentCameraIndex);
    void setGpuSceneView(const GpuEntitySceneView &sceneView) noexcept;
    void setPhysicsSceneView(const physics::PhysicsGpuSceneView *physicsScene) noexcept;
    bool drawIndexed(const gpu::GpuRenderTargetBinding &targetBinding,
                     const ForwardDrawCommand &drawCommand);

private:
    struct DrawSetup
    {
        gpu::GpuGraphicsBackendContext backendContext{};
        MeshGpuCache::CachedBuffers *meshBuffers = nullptr;
    };

    struct PerFrameConstants
    {
        std::uint32_t shadowPassParams[4] = {0u, 0u, 0u, 0u};
    };

    struct PerObjectConstants
    {
        std::uint32_t instanceIndex     = 0xffffffffu;
        std::uint32_t drawListOffset    = 0u;
        std::uint32_t useDrawListBuffer = 0u;
        std::uint32_t padding0          = 0u;
    };

    struct PipelineKey
    {
        MaterialProgramFamily programFamily       = MaterialProgramFamily::StandardLit;
        Diligent::TEXTURE_FORMAT depthFormat      = Diligent::TEX_FORMAT_UNKNOWN;

        bool operator==(const PipelineKey &rhs) const noexcept
        {
            return programFamily == rhs.programFamily && depthFormat == rhs.depthFormat;
        }
    };

    struct PipelineKeyHasher
    {
        std::size_t operator()(const PipelineKey &key) const noexcept;
    };

    bool ensureConstantBuffers(Diligent::IRenderDevice *renderDevice);
    Diligent::IPipelineState *getOrCreatePipeline(Diligent::IRenderDevice *renderDevice,
                                                  const PipelineKey &key);
    Diligent::IShaderResourceBinding *getOrCreateShaderBinding(
        Diligent::IPipelineState *pipeline) noexcept;
    bool prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                     const ForwardDrawCommand &drawCommand, DrawSetup &outSetup);
    bool bindSceneBuffers(Diligent::IShaderResourceBinding *shaderBinding,
                          MaterialProgramFamily programFamily) const;
    bool updatePerDrawConstants(Diligent::IDeviceContext *graphicsContext,
                                const ForwardDrawCommand &drawCommand);
    void bindGeometry(Diligent::IDeviceContext *graphicsContext,
                      const MeshGpuCache::CachedBuffers &meshBuffers) const;

    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    bool mInitialized = false;
    gpu::ShaderLibrary mShaderLibrary;
    MeshGpuCache mMeshGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerFrameBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mFallbackVisiblePairsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mFallbackBatchCamerasBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mFallbackLocalShadowViewsBuffer;
    std::unordered_map<PipelineKey, Diligent::RefCntAutoPtr<Diligent::IPipelineState>,
                       PipelineKeyHasher>
        mPipelines;
    std::unordered_map<Diligent::IPipelineState *,
                       Diligent::RefCntAutoPtr<Diligent::IShaderResourceBinding>>
        mShaderBindings;
    GpuEntitySceneView mSceneView{};
    const physics::PhysicsGpuSceneView *mPhysicsScene = nullptr;
    std::uint32_t mCurrentCameraIndex                 = 0u;
};

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_PASSES_CAMERA_DEPTH_PASS_H
