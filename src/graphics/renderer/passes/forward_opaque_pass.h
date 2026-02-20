#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H

#include "graphics/graphics_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"
#include "graphics/renderer/passes/material_program_registry.h"
#include "graphics/renderer/services/shader_source_provider.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/RefCntAutoPtr.hpp"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Buffer.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/RenderDevice.h"
#include "DiligentEngine/DiligentCore/Graphics/GraphicsEngine/interface/Sampler.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace cressim::neo::graphics
{

class GraphicsDeviceImpl;

namespace detail
{

class ForwardOpaquePass
{
public:
    explicit ForwardOpaquePass(GraphicsDeviceImpl& device);

    bool initialize();
    void setShadowMapTargets(const std::array<RenderTargetHandle, kShadowCascadeCount>& shadowMapTargets, std::uint32_t shadowMapCount);
    bool draw(RenderTargetHandle target, const ForwardDrawCommand& drawCommand);
    std::size_t cachedProgramCount() const noexcept;

private:
    struct CachedMeshGpuData
    {
        std::uint64_t version = 0;
        std::uint32_t indexCount = 0;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> vertexBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> indexBuffer;
    };

    struct ForwardPerFrameConstants
    {
        Diligent::float4x4 viewMatrix = Diligent::float4x4::Identity();
        Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
        std::array<Diligent::float4x4, kShadowCascadeCount> lightViewProjectionMatrices{};
        Diligent::float4 cameraPositionMetallic{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 lightDirectionIntensity{0.0f, -1.0f, 0.0f, 1.0f};
        Diligent::float4 lightColorRoughness{1.0f, 1.0f, 1.0f, 0.5f};
        Diligent::float4 cascadeSplits{1000.0f, 1000.0f, 1000.0f, 1000.0f};
        Diligent::float4 shadowTexelSizeCascadeCount{0.0f, 0.0f, 0.0f, 0.0f};
        Diligent::float4 shadowParams{0.0015f, 0.0f, 0.0f, 0.0f};
    };

    struct PerObjectConstants
    {
        Diligent::float4x4 modelMatrix = Diligent::float4x4::Identity();
        Diligent::float4x4 normalMatrix = Diligent::float4x4::Identity();
    };

    struct ForwardPerMaterialConstants
    {
        Diligent::float4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
        Diligent::float4 pipelineParams{0.5f, 0.0f, 0.0f, 0.0f};
    };

    CachedMeshGpuData* getOrCreateMeshBuffers(
        const ForwardDrawCommand& drawCommand,
        Diligent::IRenderDevice* renderDevice);
    bool ensureConstantBuffers(Diligent::IRenderDevice* renderDevice);
    bool bindProgramConstants(MaterialProgramRegistry::ProgramResources& program);

private:
    GraphicsDeviceImpl& mDevice;
    bool mInitialized = false;
    ShaderSourceProvider mShaderSourceProvider;
    std::unique_ptr<MaterialProgramRegistry> mProgramRegistry;

    std::unordered_map<common::ResourceId, CachedMeshGpuData> mCachedMeshes;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerFrameBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mPerObjectBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> mForwardPerMaterialBuffer;
    Diligent::RefCntAutoPtr<Diligent::ISampler> mShadowSampler;
    Diligent::RefCntAutoPtr<Diligent::ITextureView> mFallbackShadowMapSrv;
    std::array<RenderTargetHandle, kShadowCascadeCount> mShadowMapTargets{};
    std::uint32_t mShadowMapCount = 0;
};

} // namespace detail
} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_FORWARD_OPAQUE_PASS_H
