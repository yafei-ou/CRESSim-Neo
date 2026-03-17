#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H

#include "common/id.h"
#include "gpu/gpu_device.h"
#include "graphics/renderer/passes/forward_draw_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/AdvancedMath.hpp"
#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <array>
#include <vector>

namespace cressim::neo::graphics
{

enum class MainPassClass
{
    ForwardOpaque,
    ForwardTransparent,
};

struct FrameViewData
{
    gpu::GpuRenderTargetHandle target{};
    gpu::GpuRenderViewport viewport{};
    std::uint32_t envIndex                  = 0u;
    std::uint32_t cameraSlot                = 0u;
    std::uint32_t outputWidth               = 0;
    std::uint32_t outputHeight              = 0;
    Diligent::float4x4 viewMatrix           = Diligent::float4x4::Identity();
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    std::array<Diligent::float4x4, kShadowCascadeCount> lightViewProjectionMatrices{};
    std::array<float, kShadowCascadeCount> cascadeSplits{};
    ForwardDirectionalLightData light{};
    Diligent::float3 cameraWorldPosition = {0.0f, 0.0f, 0.0f};
    Diligent::ViewFrustum viewFrustum{};
    std::array<Diligent::ViewFrustum, kShadowCascadeCount> lightFrustums{};
    std::uint32_t shadowCascadeCount = 0;
    float shadowMapInvSizeX          = 0.0f;
    float shadowMapInvSizeY          = 0.0f;
    bool hasDirectionalLight         = false;
};

struct QueuedDraw
{
    common::EntityId entityId       = common::kInvalidEntityId;
    common::ResourceId meshId       = common::kInvalidResourceId;
    common::ResourceId materialId   = common::kInvalidResourceId;
    float depth                     = 0.0f;
    bool castsShadows               = true;
    bool receivesShadows            = true;
    bool transparent                = false;
    MainPassClass mainPassClass     = MainPassClass::ForwardOpaque;
    std::uint32_t shadowCascadeMask = 0;
    ForwardDrawCommand drawCommand{};
};

struct GpuIndirectCandidate
{
    std::uint32_t objectIndex    = 0xffffffffu;
    std::uint32_t commandIndex   = 0u;
    std::uint32_t visibilityMask = 0u;
    std::uint32_t reserved       = 0u;
};

struct GpuIndirectBucket
{
    ForwardDrawCommand drawCommand{};
    std::uint32_t candidateOffset = 0u;
    std::uint32_t candidateCount  = 0u;
    std::uint32_t drawListOffset  = 0u;
    std::uint32_t commandIndex    = 0u;
};

struct CameraRenderQueues
{
    std::vector<QueuedDraw> opaque;
    std::vector<QueuedDraw> shadowCasters;
    std::vector<QueuedDraw> transparent;
    std::vector<GpuIndirectBucket> gpuOpaqueBuckets;
    std::vector<GpuIndirectCandidate> gpuOpaqueCandidates;
    std::vector<GpuIndirectBucket> gpuShadowBuckets;
    std::vector<GpuIndirectCandidate> gpuShadowCandidates;
};

struct ForwardPassExecutionStats
{
    std::uint32_t opaqueDrawCalls      = 0;
    std::uint32_t shadowDrawCalls      = 0;
    std::uint32_t transparentDrawCalls = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
