#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "gpu/shader_library.h"
#include "graphics/passes/forward_draw_types.h"
#include "graphics/passes/material_program_registry.h"
#include "graphics/passes/render_pass_types.h"
#include "graphics/services/mesh_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <array>
#include <cstdint>
#include <memory>

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardOpaquePass
{
public:
    ForwardOpaquePass(gpu::GpuDevice &device, RenderResourceManager &resourceManager);

    bool initialize();
    bool beginBatchFrame(std::uint32_t currentCameraIndex);
    void setGpuSceneView(const gpu::GpuEntitySceneView &sceneView) noexcept;
    void setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept;
    void setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept;
    void setShadowMapTargets(
        const std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> &shadowMapTargets,
        std::uint32_t shadowMapCount);
    bool drawIndirect(const gpu::GpuRenderTargetBinding &targetBinding,
                      const ForwardDrawCommand &drawCommand, Diligent::IBuffer *indirectArgsBuffer,
                      Diligent::Uint64 argsOffsetBytes);
    std::size_t cachedProgramCount() const noexcept;

private:
    struct DrawSetup
    {
        gpu::GpuBackendContext backendContext{};
        MeshGpuCache::CachedBuffers *meshBuffers           = nullptr;
        MaterialProgramRegistry::ProgramResources *program = nullptr;
    };

    struct ForwardPerFrameConstants
    {
        Diligent::float4 shadowParams{0.0015f, 0.0f, 0.0f, 0.0f};
        std::uint32_t currentCameraIndex = 0u;
        std::uint32_t padding0           = 0u;
        std::uint32_t padding1           = 0u;
        std::uint32_t padding2           = 0u;
    };

    struct PerObjectConstants
    {
        std::uint32_t instanceIndex     = 0xffffffffu;
        std::uint32_t drawListOffset    = 0u;
        std::uint32_t useDrawListBuffer = 0u;
        std::uint32_t padding0          = 0u;
    };

    struct ForwardPerMaterialConstants
    {
        Diligent::float4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 materialParams{0.0f, 0.5f, 0.5f, 0.0f};
    };

    bool ensureConstantBuffers(Diligent::IRenderDevice *renderDevice);
    bool bindProgramConstants(MaterialProgramRegistry::ProgramResources &program);
    bool hasAnyShadowMap() const;
    bool prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                     const ForwardDrawCommand &drawCommand, DrawSetup &outSetup);
    bool bindShadowMaps(MaterialProgramRegistry::ProgramResources &program);
    bool bindSceneBuffers(MaterialProgramRegistry::ProgramResources &program) const;
    bool updatePerDrawConstants(Diligent::IDeviceContext *immediateContext,
                                const ForwardDrawCommand &drawCommand);
    void bindGeometry(Diligent::IDeviceContext *immediateContext,
                      const MeshGpuCache::CachedBuffers &meshBuffers) const;

private:
    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    bool mInitialized = false;
    gpu::ShaderLibrary mShaderLibrary;
    std::unique_ptr<MaterialProgramRegistry> mProgramRegistry;

    MeshGpuCache mMeshGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerFrameBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerMaterialBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mShadowSampler;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackShadowMapSrv;
    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    std::uint32_t mShadowMapCount = 0;
    gpu::GpuEntitySceneView mSceneView{};
    Diligent::IBuffer *mVisiblePairBuffer = nullptr;
    Diligent::IBuffer *mBatchCameraBuffer = nullptr;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H
