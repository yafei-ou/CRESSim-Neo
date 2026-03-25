#ifndef CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H
#define CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "gpu/shader_library.h"
#include "graphics/passes/forward_draw_types.h"
#include "graphics/passes/material_program_registry.h"
#include "graphics/services/mesh_gpu_cache.h"
#include "graphics/services/texture_gpu_cache.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace cressim::neo::graphics
{

namespace detail
{

class ForwardOpaquePass
{
public:
    ForwardOpaquePass(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                      IblQualityTier iblQualityTier);

    bool initialize();
    bool beginBatchFrame(std::uint32_t currentCameraIndex);
    void setGpuSceneView(const gpu::GpuEntitySceneView &sceneView) noexcept;
    void setEnvironmentIbls(const std::vector<EnvironmentIblDesc> *ibls,
                            std::uint32_t envCount) noexcept;
    void setVisiblePairBuffer(Diligent::IBuffer *buffer) noexcept;
    void setBatchCameraBuffer(Diligent::IBuffer *buffer) noexcept;
    void setShadowMapTargets(
        const std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> &shadowMapTargets,
        std::uint32_t shadowMapCount);
    void setLocalShadowResources(gpu::GpuRenderTargetHandle localShadowMap2D,
                                 gpu::GpuRenderTargetHandle pointShadowMap,
                                 Diligent::IBuffer *localShadowViewBuffer,
                                 Diligent::IBuffer *lightShadowAssignmentBuffer,
                                 std::uint32_t localShadowViewCount) noexcept;
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
        Diligent::float4 iblSpecularParams{0.0f, 0.0f, 0.0f, 0.0f};
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
        Diligent::float4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 materialParams{0.0f, 0.5f, 0.5f, 1.0f};
    };

    struct EnvironmentIblLookupEntry
    {
        std::uint32_t sliceIndex = 0u;
        std::uint32_t enabled    = 0u;
        float intensity          = 1.0f;
        float padding0           = 0.0f;
    };

    bool ensureConstantBuffers(Diligent::IRenderDevice *renderDevice);
    bool ensureEnvironmentIblResources(Diligent::IRenderDevice *renderDevice,
                                       Diligent::IDeviceContext *immediateContext);
    bool bindProgramConstants(MaterialProgramRegistry::ProgramResources &program);
    bool hasAnyShadowMap() const;
    bool prepareDraw(const gpu::GpuRenderTargetBinding &targetBinding,
                     const ForwardDrawCommand &drawCommand, DrawSetup &outSetup);
    bool bindShadowMaps(MaterialProgramRegistry::ProgramResources &program);
    bool bindSceneBuffers(MaterialProgramRegistry::ProgramResources &program) const;
    bool bindEnvironmentIblResources(MaterialProgramRegistry::ProgramResources &program) const;
    bool bindMaterialTextures(MaterialProgramRegistry::ProgramResources &program,
                              Diligent::IRenderDevice *renderDevice,
                              Diligent::IDeviceContext *immediateContext,
                              common::ResourceId materialId);
    bool updatePerDrawConstants(Diligent::IDeviceContext *immediateContext,
                                const ForwardDrawCommand &drawCommand);
    void bindGeometry(Diligent::IDeviceContext *immediateContext,
                      const MeshGpuCache::CachedBuffers &meshBuffers) const;

private:
    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    IblQualityTier mIblQualityTier = IblQualityTier::Off;
    bool mInitialized              = false;
    gpu::ShaderLibrary mShaderLibrary;
    std::unique_ptr<MaterialProgramRegistry> mProgramRegistry;

    MeshGpuCache mMeshGpuCache;
    TextureGpuCache mTextureGpuCache;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerFrameBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerMaterialBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mShadowSampler;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mMaterialSampler;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackShadowMapSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackBaseColorSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackNormalSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackMetallicRoughnessSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackEmissiveSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackAoSrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mIrradianceArraySrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mPrefilteredSpecularArraySrv;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mBrdfLutSrv;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mEnvironmentIblLookupBuffer;
    std::uint32_t mEnvironmentIblLookupCapacity = 0u;
    float mEnvironmentIblPrefilteredMipCount    = 1.0f;
    std::size_t mEnvironmentIblStateHash        = 0u;
    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    std::uint32_t mShadowMapCount = 0;
    gpu::GpuRenderTargetHandle mLocalShadowMap2D{};
    gpu::GpuRenderTargetHandle mPointShadowMap{};
    Diligent::IBuffer *mLocalShadowViewBuffer       = nullptr;
    Diligent::IBuffer *mLightShadowAssignmentBuffer = nullptr;
    std::uint32_t mLocalShadowViewCount             = 0u;
    gpu::GpuEntitySceneView mSceneView{};
    const std::vector<EnvironmentIblDesc> *mEnvironmentIbls = nullptr;
    std::uint32_t mEnvironmentIblEnvCount                   = 0u;
    Diligent::IBuffer *mVisiblePairBuffer                   = nullptr;
    Diligent::IBuffer *mBatchCameraBuffer                   = nullptr;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_PASSES_FORWARD_OPAQUE_PASS_H
