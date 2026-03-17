#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H

#include "gpu/gpu_device.h"
#include "gpu/gpu_scene.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/passes/forward_draw_types.h"
#include "graphics/renderer/passes/material_program_registry.h"
#include "graphics/renderer/passes/render_pass_types.h"
#include "graphics/renderer/services/mesh_gpu_cache.h"

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
    explicit ForwardOpaquePass(gpu::GpuDevice& device);

    bool initialize();
    bool beginCameraFrame(const FrameViewData& frameView);
    void setGpuSceneView(const gpu::GpuEntitySceneView& sceneView) noexcept;
    void setShadowMapTargets(
        const std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount>& shadowMapTargets,
        std::uint32_t shadowMapCount);
    bool draw(gpu::GpuRenderTargetHandle target, const ForwardDrawCommand& drawCommand);
    std::size_t cachedProgramCount() const noexcept;

private:
    struct ForwardPerFrameConstants
    {
        Diligent::float4x4 viewMatrix           = Diligent::float4x4::Identity();
        Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
        std::array<Diligent::float4x4, kShadowCascadeCount> lightViewProjectionMatrices{};
        Diligent::float4 cameraPosition{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 lightDirectionIntensity{0.0f, -1.0f, 0.0f, 1.0f};
        Diligent::float4 lightColor{1.0f, 1.0f, 1.0f, 0.0f};
        Diligent::float4 cascadeSplits{1000.0f, 1000.0f, 1000.0f, 1000.0f};
        Diligent::float4 shadowTexelSizeCascadeCount{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 shadowParams{0.0015f, 0.0f, 0.0f, 0.0f};
    };

    struct PerObjectConstants
    {
        Diligent::float4x4 modelMatrix  = Diligent::float4x4::Identity();
        Diligent::float4x4 normalMatrix = Diligent::float4x4::Identity();
        std::uint32_t instanceIndex = 0xffffffffu;
        std::uint32_t useSceneBuffers = 0u;
        std::uint32_t padding0 = 0u;
        std::uint32_t padding1 = 0u;
    };

    struct ForwardPerMaterialConstants
    {
        Diligent::float4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 materialParams{0.0f, 0.5f, 0.5f, 0.0f};
    };

    bool ensureConstantBuffers(Diligent::IRenderDevice* renderDevice);
    bool bindProgramConstants(MaterialProgramRegistry::ProgramResources& program);
    bool hasAnyShadowMap() const;

private:
    gpu::GpuDevice& mDevice;
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
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H
