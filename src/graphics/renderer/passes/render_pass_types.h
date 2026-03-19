#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H

#include "common/id.h"
#include "gpu/gpu_types.h"
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
};

struct FrameViewData
{
    gpu::GpuRenderTargetHandle target{};
    gpu::GpuRenderViewport viewport{};
    bool clearColor                         = true;
    bool clearDepth                         = true;
    Diligent::float4 clearColorValue        = {0.02f, 0.02f, 0.03f, 1.0f};
    float clearDepthValue                   = 1.0f;
    std::uint32_t envIndex                  = 0u;
    std::uint32_t cameraSlot                = 0u;
    std::uint32_t outputWidth               = 0;
    std::uint32_t outputHeight              = 0;
    Diligent::float4x4 viewMatrix           = Diligent::float4x4::Identity();
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    ForwardDirectionalLightData light{};
    Diligent::float3 cameraWorldPosition = {0.0f, 0.0f, 0.0f};
};

struct QueuedDraw
{
    std::uint32_t objectIndex     = 0xffffffffu;
    common::ResourceId meshId     = common::kInvalidResourceId;
    common::ResourceId materialId = common::kInvalidResourceId;
    bool castsShadows             = true;
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
    std::vector<GpuIndirectBucket> gpuOpaqueBuckets;
    std::vector<GpuIndirectCandidate> gpuOpaqueCandidates;
    std::vector<GpuIndirectBucket> gpuShadowBuckets;
    std::vector<GpuIndirectCandidate> gpuShadowCandidates;
};

struct ForwardPassExecutionStats
{
    std::uint32_t opaqueDrawCalls = 0;
    std::uint32_t shadowDrawCalls = 0;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_PASSES_RENDER_PASS_TYPES_H
