#include "graphics/renderer/passes/forward_pipeline.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/passes/forward_opaque_pass.h"
#include "graphics/renderer/passes/forward_transparent_pass.h"
#include "graphics/renderer/passes/shadow_pass.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace cressim::neo::graphics::detail
{

namespace
{

constexpr std::uint32_t kIndirectThreadGroupSize = 64u;

struct IndirectCommandDesc
{
    std::uint32_t visibleOffset   = 0u;
    std::uint32_t maxVisibleCount = 0u;
    std::uint32_t indexCount      = 0u;
    std::uint32_t reserved        = 0u;
};

struct GraphicsIndirectPassConstants
{
    std::uint32_t count    = 0u;
    std::uint32_t padding0 = 0u;
    std::uint32_t padding1 = 0u;
    std::uint32_t padding2 = 0u;
};

constexpr Diligent::ShaderResourceVariableDesc kIndirectResetVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsIndirectResetConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandDescs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandCountsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_DrawIndexedCommandsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kIndirectResetPassDefinition = {
    "graphics/graphics_indirect_reset.cs.hlsl",
    "CRESSimNeo.Graphics.IndirectReset",
    "CRESSimNeo.Graphics.IndirectReset.PSO",
    kIndirectResetVars,
    std::size(kIndirectResetVars),
};

constexpr Diligent::ShaderResourceVariableDesc kIndirectFilterVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsIndirectFilterConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandDescs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_Candidates",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableVisibilityFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableShadowCascadeMasks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandCountsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_VisibleObjectIndicesRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kIndirectFilterPassDefinition = {
    "graphics/graphics_indirect_filter.cs.hlsl",
    "CRESSimNeo.Graphics.IndirectFilter",
    "CRESSimNeo.Graphics.IndirectFilter.PSO",
    kIndirectFilterVars,
    std::size(kIndirectFilterVars),
};

constexpr Diligent::ShaderResourceVariableDesc kIndirectComposeVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsIndirectComposeConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandCounts",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_DrawIndexedCommandsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kIndirectComposePassDefinition = {
    "graphics/graphics_indirect_compose.cs.hlsl",
    "CRESSimNeo.Graphics.IndirectCompose",
    "CRESSimNeo.Graphics.IndirectCompose.PSO",
    kIndirectComposeVars,
    std::size(kIndirectComposeVars),
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kIndirectThreadGroupSize - 1u) / kIndirectThreadGroupSize;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice* renderDevice, const char* name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer>& outBuffer)
{
    if (renderDevice == nullptr)
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name      = name;
    desc.Size      = static_cast<Diligent::Uint64>(std::max(elementCount, 1u)) * elementStride;
    desc.BindFlags = bindFlags;
    desc.Usage     = Diligent::USAGE_DEFAULT;
    desc.Mode      = Diligent::BUFFER_MODE_STRUCTURED;
    desc.ElementByteStride = elementStride;
    renderDevice->CreateBuffer(desc, nullptr, &outBuffer);
    return outBuffer != nullptr;
}

bool writeBuffer(Diligent::IDeviceContext* context, Diligent::IBuffer* buffer, const void* data,
                 std::size_t sizeBytes)
{
    if (context == nullptr || buffer == nullptr || data == nullptr || sizeBytes == 0u)
    {
        return false;
    }

    context->UpdateBuffer(buffer, 0u, static_cast<Diligent::Uint32>(sizeBytes), data,
                          Diligent::RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    return true;
}

} // namespace

struct ForwardPipeline::GpuIndirectState
{
    struct BufferSet
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> candidateBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandDescBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> visibleObjectIndicesBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> drawIndexedCommandsBuffer;
        std::uint32_t candidateCapacity    = 0u;
        std::uint32_t commandCapacity      = 0u;
        std::uint32_t visibleIndexCapacity = 0u;
    };

    gpu::GpuComputePass resetPass;
    gpu::GpuComputePass filterPass;
    gpu::GpuComputePass composePass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> resetConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> filterConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> composeConstantsBuffer;
    BufferSet opaque;
    BufferSet shadow;
    bool initialized = false;
};

ForwardPipeline::ForwardPipeline(gpu::GpuDevice& device)
    : mDevice(device), mGpuIndirectState(std::make_unique<GpuIndirectState>())
{
}

ForwardPipeline::~ForwardPipeline()
{
    for (gpu::GpuRenderTargetHandle target : mShadowMapTargets)
    {
        if (mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            mDevice.renderTargetSystem().destroyRenderTarget(target);
        }
    }
}

bool ForwardPipeline::initialize()
{
    mForwardOpaquePass = std::make_unique<ForwardOpaquePass>(mDevice);
    if (!mForwardOpaquePass->initialize())
    {
        mForwardOpaquePass.reset();
        return false;
    }

    mForwardTransparentPass = std::make_unique<ForwardTransparentPass>(mDevice);
    if (!mForwardTransparentPass->initialize())
    {
        mForwardTransparentPass.reset();
        mForwardOpaquePass.reset();
        return false;
    }

    mShadowPass = std::make_unique<ShadowPass>(mDevice);
    if (!mShadowPass->initialize())
    {
        mShadowPass.reset();
        mForwardTransparentPass.reset();
        mForwardOpaquePass.reset();
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }
    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory* streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr || mGpuIndirectState == nullptr)
    {
        return false;
    }
    if (!mGpuIndirectState->resetPass.initialize(backendContext.renderDevice, streamFactory, 1ull,
                                                 kIndirectResetPassDefinition) ||
        !mGpuIndirectState->filterPass.initialize(backendContext.renderDevice, streamFactory, 1ull,
                                                  kIndirectFilterPassDefinition) ||
        !mGpuIndirectState->composePass.initialize(backendContext.renderDevice, streamFactory, 1ull,
                                                   kIndirectComposePassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Usage          = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags      = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;

    constantsDesc.Name = "CRESSimNeo.ForwardPipeline.IndirectResetConstants";
    constantsDesc.Size = sizeof(GraphicsIndirectPassConstants);
    backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                              &mGpuIndirectState->resetConstantsBuffer);
    if (mGpuIndirectState->resetConstantsBuffer == nullptr)
    {
        return false;
    }

    constantsDesc.Name = "CRESSimNeo.ForwardPipeline.IndirectFilterConstants";
    backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                              &mGpuIndirectState->filterConstantsBuffer);
    if (mGpuIndirectState->filterConstantsBuffer == nullptr)
    {
        return false;
    }

    constantsDesc.Name = "CRESSimNeo.ForwardPipeline.IndirectComposeConstants";
    backendContext.renderDevice->CreateBuffer(constantsDesc, nullptr,
                                              &mGpuIndirectState->composeConstantsBuffer);
    if (mGpuIndirectState->composeConstantsBuffer == nullptr)
    {
        return false;
    }
    mGpuIndirectState->initialized = true;

    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        gpu::GpuRenderTargetDesc shadowDesc{};
        shadowDesc.width              = kShadowMapResolution;
        shadowDesc.height             = kShadowMapResolution;
        shadowDesc.color              = false;
        shadowDesc.depth              = true;
        shadowDesc.shaderReadable     = true;
        shadowDesc.debugName          = "CRESSimNeo.ShadowMap.Cascade" + std::to_string(cascadeIdx);
        mShadowMapTargets[cascadeIdx] = mDevice.renderTargetSystem().createRenderTarget(shadowDesc);
        if (!mDevice.renderTargetSystem().isValidRenderTarget(mShadowMapTargets[cascadeIdx]))
        {
            mShadowMapTargets[cascadeIdx] = {};
        }
    }

    mInitialized = true;
    return true;
}

bool ForwardPipeline::execute(const common::FrameContext& frameContext,
                              const FrameViewData& frameView,
                              const gpu::GpuEntitySceneView& sceneView,
                              const CameraRenderQueues& queues,
                              ForwardPassExecutionStats& outStats)
{
    if (!mInitialized || mForwardOpaquePass == nullptr)
    {
        return false;
    }

    outStats = {};
    mForwardOpaquePass->setGpuSceneView(sceneView);
    if (mShadowPass != nullptr)
    {
        mShadowPass->setGpuSceneView(sceneView);
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr ||
        mGpuIndirectState == nullptr || !mGpuIndirectState->initialized)
    {
        return false;
    }

    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> activeShadowMaps{};
    std::uint32_t activeShadowMapCount = 0;
    const std::uint32_t currentCameraIndex =
        frameView.envIndex * std::max(sceneView.layout.maxCamerasPerEnv, 1u) + frameView.cameraSlot;

    const auto uploadIndirectSet =
        [&](GpuIndirectState::BufferSet& bufferSet, const std::vector<GpuIndirectBucket>& buckets,
            const std::vector<GpuIndirectCandidate>& candidates, const char* namePrefix) -> bool
    {
        const std::uint32_t commandCount   = static_cast<std::uint32_t>(buckets.size());
        const std::uint32_t candidateCount = static_cast<std::uint32_t>(candidates.size());
        std::uint32_t visibleCapacity      = 0u;
        for (const GpuIndirectBucket& bucket : buckets)
        {
            visibleCapacity =
                std::max(visibleCapacity, bucket.drawListOffset + bucket.candidateCount);
        }

        if (commandCount == 0u || candidateCount == 0u || visibleCapacity == 0u)
        {
            return true;
        }

        if (bufferSet.commandCapacity < commandCount || bufferSet.commandDescBuffer == nullptr ||
            bufferSet.commandCountsBuffer == nullptr ||
            bufferSet.drawIndexedCommandsBuffer == nullptr)
        {
            const std::string descName  = std::string{namePrefix} + ".CommandDescs";
            const std::string countName = std::string{namePrefix} + ".CommandCounts";
            const std::string argsName  = std::string{namePrefix} + ".DrawArgs";
            if (!ensureStructuredBuffer(
                    backendContext.renderDevice, descName.c_str(), sizeof(IndirectCommandDesc),
                    commandCount, Diligent::BIND_SHADER_RESOURCE, bufferSet.commandDescBuffer) ||
                !ensureStructuredBuffer(
                    backendContext.renderDevice, countName.c_str(), sizeof(std::uint32_t),
                    commandCount, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                    bufferSet.commandCountsBuffer) ||
                !ensureStructuredBuffer(backendContext.renderDevice, argsName.c_str(),
                                        sizeof(std::uint32_t) * 5u, commandCount,
                                        Diligent::BIND_UNORDERED_ACCESS |
                                            Diligent::BIND_INDIRECT_DRAW_ARGS,
                                        bufferSet.drawIndexedCommandsBuffer))
            {
                return false;
            }
            bufferSet.commandCapacity = commandCount;
        }

        if (bufferSet.candidateCapacity < candidateCount || bufferSet.candidateBuffer == nullptr)
        {
            const std::string candidateName = std::string{namePrefix} + ".Candidates";
            if (!ensureStructuredBuffer(backendContext.renderDevice, candidateName.c_str(),
                                        sizeof(GpuIndirectCandidate), candidateCount,
                                        Diligent::BIND_SHADER_RESOURCE, bufferSet.candidateBuffer))
            {
                return false;
            }
            bufferSet.candidateCapacity = candidateCount;
        }

        if (bufferSet.visibleIndexCapacity < visibleCapacity ||
            bufferSet.visibleObjectIndicesBuffer == nullptr)
        {
            const std::string visibleName = std::string{namePrefix} + ".VisibleObjectIndices";
            if (!ensureStructuredBuffer(backendContext.renderDevice, visibleName.c_str(),
                                        sizeof(std::uint32_t), visibleCapacity,
                                        Diligent::BIND_SHADER_RESOURCE |
                                            Diligent::BIND_UNORDERED_ACCESS,
                                        bufferSet.visibleObjectIndicesBuffer))
            {
                return false;
            }
            bufferSet.visibleIndexCapacity = visibleCapacity;
        }

        std::vector<IndirectCommandDesc> commandDescs(commandCount);
        for (const GpuIndirectBucket& bucket : buckets)
        {
            commandDescs[bucket.commandIndex] = IndirectCommandDesc{
                bucket.drawListOffset, bucket.candidateCount, bucket.drawCommand.indexCount, 0u};
        }

        if (!writeBuffer(backendContext.immediateContext, bufferSet.commandDescBuffer,
                         commandDescs.data(), commandDescs.size() * sizeof(IndirectCommandDesc)) ||
            !writeBuffer(backendContext.immediateContext, bufferSet.candidateBuffer,
                         candidates.data(), candidates.size() * sizeof(GpuIndirectCandidate)))
        {
            return false;
        }

        const auto updateConstants = [&](Diligent::IBuffer* constantBuffer,
                                         std::uint32_t count) -> bool
        {
            void* mapped = nullptr;
            backendContext.immediateContext->MapBuffer(constantBuffer, Diligent::MAP_WRITE,
                                                       Diligent::MAP_FLAG_DISCARD, mapped);
            if (mapped == nullptr)
            {
                return false;
            }
            const GraphicsIndirectPassConstants constants{count, 0u, 0u, 0u};
            std::memcpy(mapped, &constants, sizeof(constants));
            backendContext.immediateContext->UnmapBuffer(constantBuffer, Diligent::MAP_WRITE);
            return true;
        };

        if (!updateConstants(mGpuIndirectState->resetConstantsBuffer, commandCount))
        {
            return false;
        }
        const std::array resetBindings{
            gpu::GpuBufferBinding{"GraphicsIndirectResetConstants",
                                  mGpuIndirectState->resetConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandDescs", bufferSet.commandDescBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCountsRW", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_DrawIndexedCommandsRW", bufferSet.drawIndexedCommandsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!mGpuIndirectState->resetPass.dispatch(backendContext.immediateContext, 0u,
                                                   resetBindings, dispatchGroupCount(commandCount)))
        {
            return false;
        }

        if (!updateConstants(mGpuIndirectState->filterConstantsBuffer, candidateCount))
        {
            return false;
        }
        const std::array filterBindings{
            gpu::GpuBufferBinding{"GraphicsIndirectFilterConstants",
                                  mGpuIndirectState->filterConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandDescs", bufferSet.commandDescBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_Candidates", bufferSet.candidateBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableVisibilityFlags",
                                  sceneView.renderableVisibilityFlagsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableShadowCascadeMasks",
                                  sceneView.renderableShadowCascadeMasksBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCountsRW", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_VisibleObjectIndicesRW", bufferSet.visibleObjectIndicesBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!mGpuIndirectState->filterPass.dispatch(backendContext.immediateContext, 0u,
                                                    filterBindings,
                                                    dispatchGroupCount(candidateCount)))
        {
            return false;
        }

        if (!updateConstants(mGpuIndirectState->composeConstantsBuffer, commandCount))
        {
            return false;
        }
        const std::array composeBindings{
            gpu::GpuBufferBinding{"GraphicsIndirectComposeConstants",
                                  mGpuIndirectState->composeConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCounts", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_DrawIndexedCommandsRW", bufferSet.drawIndexedCommandsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        return mGpuIndirectState->composePass.dispatch(
            backendContext.immediateContext, 0u, composeBindings, dispatchGroupCount(commandCount));
    };

    if (!uploadIndirectSet(mGpuIndirectState->opaque, queues.gpuOpaqueBuckets,
                           queues.gpuOpaqueCandidates, "CRESSimNeo.ForwardPipeline.Opaque"))
    {
        return false;
    }
    if (!uploadIndirectSet(mGpuIndirectState->shadow, queues.gpuShadowBuckets,
                           queues.gpuShadowCandidates, "CRESSimNeo.ForwardPipeline.Shadow"))
    {
        return false;
    }

    if (frameView.hasDirectionalLight && frameView.shadowCascadeCount > 0 &&
        mShadowPass != nullptr && !queues.gpuShadowBuckets.empty())
    {
        mShadowPass->setVisibleObjectIndexBuffer(
            mGpuIndirectState->shadow.visibleObjectIndicesBuffer);
        for (std::uint32_t cascadeIdx = 0; cascadeIdx < frameView.shadowCascadeCount; ++cascadeIdx)
        {
            const gpu::GpuRenderTargetHandle cascadeShadowMap = mShadowMapTargets[cascadeIdx];
            if (!mDevice.renderTargetSystem().isValidRenderTarget(cascadeShadowMap))
            {
                continue;
            }

            mDevice.renderTargetSystem().setRenderTargetViewport(cascadeShadowMap,
                                                                 gpu::GpuRenderViewport{});

            gpu::GpuRenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor      = false;
            shadowBegin.clearDepth      = true;
            shadowBegin.clearDepthValue = 1.0f;

            mDevice.renderTargetSystem().beginRenderTarget(cascadeShadowMap, frameContext,
                                                           shadowBegin);
            for (const GpuIndirectBucket& bucket : queues.gpuShadowBuckets)
            {
                if ((bucket.commandIndex % kShadowCascadeCount) != cascadeIdx)
                {
                    continue;
                }

                ForwardDrawCommand drawCommand = bucket.drawCommand;
                drawCommand.drawListOffset     = bucket.drawListOffset;
                drawCommand.useDrawListBuffer  = 1u;
                if (mShadowPass->drawIndirect(cascadeShadowMap, drawCommand, currentCameraIndex,
                                              frameView.lightViewProjectionMatrices[cascadeIdx],
                                              cascadeIdx,
                                              mGpuIndirectState->shadow.drawIndexedCommandsBuffer,
                                              static_cast<Diligent::Uint64>(bucket.commandIndex) *
                                                  sizeof(std::uint32_t) * 5u))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(cascadeShadowMap, frameContext);
            activeShadowMaps[cascadeIdx] = cascadeShadowMap;
            activeShadowMapCount         = std::max(activeShadowMapCount, cascadeIdx + 1);
        }
    }

    mForwardOpaquePass->setShadowMapTargets(activeShadowMaps, activeShadowMapCount);
    mForwardOpaquePass->setVisibleObjectIndexBuffer(
        mGpuIndirectState->opaque.visibleObjectIndicesBuffer);
    if (!mForwardOpaquePass->beginCameraFrame(frameView))
    {
        return false;
    }

    mDevice.renderTargetSystem().setRenderTargetViewport(frameView.target, frameView.viewport);
    gpu::GpuRenderPassBeginDesc mainBegin{};
    mainBegin.clearColor = frameView.clearColor;
    mainBegin.clearDepth = frameView.clearDepth;
    mainBegin.clearColorValue[0] = frameView.clearColorValue.x;
    mainBegin.clearColorValue[1] = frameView.clearColorValue.y;
    mainBegin.clearColorValue[2] = frameView.clearColorValue.z;
    mainBegin.clearColorValue[3] = frameView.clearColorValue.w;
    mainBegin.clearDepthValue = frameView.clearDepthValue;
    mDevice.renderTargetSystem().beginRenderTarget(frameView.target, frameContext, mainBegin);

    for (const GpuIndirectBucket& bucket : queues.gpuOpaqueBuckets)
    {
        ForwardDrawCommand drawCommand = bucket.drawCommand;
        drawCommand.drawListOffset     = bucket.drawListOffset;
        drawCommand.useDrawListBuffer  = 1u;
        if (mForwardOpaquePass->drawIndirect(
                frameView.target, drawCommand, mGpuIndirectState->opaque.drawIndexedCommandsBuffer,
                static_cast<Diligent::Uint64>(bucket.commandIndex) * sizeof(std::uint32_t) * 5u))
        {
            ++outStats.opaqueDrawCalls;
        }
    }

    if (mForwardTransparentPass != nullptr)
    {
        for (const QueuedDraw& draw : queues.transparent)
        {
            if (mForwardTransparentPass->draw(frameView.target, draw.drawCommand))
            {
                ++outStats.transparentDrawCalls;
            }
        }
    }

    mDevice.renderTargetSystem().endRenderTarget(frameView.target, frameContext);
    return true;
}

} // namespace cressim::neo::graphics::detail
