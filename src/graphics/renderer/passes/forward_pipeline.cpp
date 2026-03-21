#include "graphics/renderer/passes/forward_pipeline.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/passes/forward_opaque_pass.h"
#include "graphics/renderer/passes/shadow_pass.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

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
    std::uint32_t countOrObjectCount = 0u;
    std::uint32_t cameraCount        = 0u;
    std::uint32_t queueMode          = 0u;
    std::uint32_t padding0           = 0u;
};

constexpr std::uint32_t kQueueModeOpaque = 0u;
constexpr std::uint32_t kQueueModeShadow = 1u;

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
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreparedCameras",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableQueueInfo",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableVisibilityFlags",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableShadowCascadeMasks",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BatchCameraIndices",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_BatchCameraLayers",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandDescs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandCountsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_VisiblePairsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kIndirectFilterPassDefinition = {
    "graphics/graphics_indirect_pair_filter.cs.hlsl",
    "CRESSimNeo.Graphics.IndirectPairFilter",
    "CRESSimNeo.Graphics.IndirectPairFilter.PSO",
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
                            Diligent::RefCntAutoPtr<Diligent::IBuffer>& outBuffer,
                            std::uint32_t& inOutCapacity, std::uint32_t minimumCapacity)
{
    return gpu::detail::ensureStructuredBufferCapacity(
        renderDevice, name, elementStride, elementCount, minimumCapacity, bindFlags,
        Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, 1ull, outBuffer, inOutCapacity);
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

bool updateConstants(Diligent::IDeviceContext* context, Diligent::IBuffer* constantBuffer,
                     const GraphicsIndirectPassConstants& constants)
{
    if (context == nullptr || constantBuffer == nullptr)
    {
        return false;
    }

    void* mapped = nullptr;
    context->MapBuffer(constantBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    context->UnmapBuffer(constantBuffer, Diligent::MAP_WRITE);
    return true;
}

gpu::GpuRenderViewport viewportForBatch(const CameraBatchView& batchView)
{
    if (batchView.cameras.size() == 1u && batchView.cameras.front().useOutputViewport)
    {
        return batchView.cameras.front().viewport;
    }

    return {};
}

} // namespace

struct ForwardPipeline::GpuIndirectState
{
    struct BufferSet
    {
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandDescBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> visiblePairBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> drawIndexedCommandsBuffer;
        std::uint32_t commandCapacity     = 0u;
        std::uint32_t visiblePairCapacity = 0u;
        std::vector<std::uint32_t> drawListOffsets;
    };

    gpu::GpuComputePass resetPass;
    gpu::GpuComputePass filterPass;
    gpu::GpuComputePass composePass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> resetConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> filterConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> composeConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> batchCameraIndicesBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> batchCameraLayersBuffer;
    std::uint32_t batchCameraCapacity = 0u;
    BufferSet opaque;
    BufferSet shadow;
    bool initialized = false;
};

ForwardPipeline::ForwardPipeline(gpu::GpuDevice& device, RenderResourceManager& resourceManager)
    : mDevice(device), mResourceManager(resourceManager),
      mGpuIndirectState(std::make_unique<GpuIndirectState>())
{
}

ForwardPipeline::~ForwardPipeline()
{
    for (const auto& [key, target] : mLayeredTargetCache)
    {
        (void)key;
        if (mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            mDevice.renderTargetSystem().destroyRenderTarget(target);
        }
    }
}

bool ForwardPipeline::initialize()
{
    mForwardOpaquePass = std::make_unique<ForwardOpaquePass>(mDevice, mResourceManager);
    if (!mForwardOpaquePass->initialize())
    {
        mForwardOpaquePass.reset();
        return false;
    }

    mShadowPass = std::make_unique<ShadowPass>(mDevice, mResourceManager);
    if (!mShadowPass->initialize())
    {
        mShadowPass.reset();
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
    constantsDesc.Size           = sizeof(GraphicsIndirectPassConstants);

    constantsDesc.Name = "CRESSimNeo.ForwardPipeline.IndirectResetConstants";
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
    mInitialized                   = true;
    return true;
}

bool ForwardPipeline::executeBatch(const common::FrameContext& frameContext,
                                   const CameraBatchView& batchView, const HostSceneView& sceneView,
                                   ForwardPassExecutionStats& outStats)
{
    outStats = {};
    if (!mInitialized || mForwardOpaquePass == nullptr || batchView.cameras.empty())
    {
        return false;
    }

    const gpu::GpuEntitySceneView emptyGpuScene{};
    const gpu::GpuEntitySceneView& gpuScene =
        sceneView.gpuEntityScene != nullptr ? *sceneView.gpuEntityScene : emptyGpuScene;
    const std::vector<IndirectCommandRegistryEntry> emptyRegistry;
    const std::vector<IndirectCommandRegistryEntry>& opaqueRegistry =
        sceneView.opaqueDrawRegistry != nullptr ? *sceneView.opaqueDrawRegistry : emptyRegistry;
    const std::vector<IndirectCommandRegistryEntry>& shadowRegistry =
        sceneView.shadowDrawRegistry != nullptr ? *sceneView.shadowDrawRegistry : emptyRegistry;

    if (gpuScene.preparedCamerasBuffer == nullptr ||
        gpuScene.renderableQueueInfoBuffer == nullptr ||
        gpuScene.renderableVisibilityFlagsBuffer == nullptr ||
        gpuScene.renderableShadowCascadeMasksBuffer == nullptr)
    {
        return false;
    }

    mForwardOpaquePass->setGpuSceneView(gpuScene);
    if (mShadowPass != nullptr)
    {
        mShadowPass->setGpuSceneView(gpuScene);
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.immediateContext == nullptr ||
        mGpuIndirectState == nullptr || !mGpuIndirectState->initialized)
    {
        return false;
    }

    const std::uint32_t batchCameraCount = static_cast<std::uint32_t>(batchView.cameras.size());
    std::vector<std::uint32_t> batchCameraIndices(batchCameraCount, 0u);
    std::vector<std::uint32_t> batchCameraLayers(batchCameraCount, 0u);
    for (std::uint32_t i = 0u; i < batchCameraCount; ++i)
    {
        batchCameraIndices[i] = batchView.cameras[i].globalCameraIndex;
        batchCameraLayers[i] =
            batchView.cameras[i].outputBinding.firstLayer - batchView.renderBinding.firstLayer;
    }

    if (mGpuIndirectState->batchCameraCapacity < batchCameraCount ||
        mGpuIndirectState->batchCameraIndicesBuffer == nullptr ||
        mGpuIndirectState->batchCameraLayersBuffer == nullptr)
    {
        if (!ensureStructuredBuffer(
                backendContext.renderDevice, "CRESSimNeo.ForwardPipeline.BatchCameraIndices",
                sizeof(std::uint32_t), batchCameraCount, Diligent::BIND_SHADER_RESOURCE,
                mGpuIndirectState->batchCameraIndicesBuffer, mGpuIndirectState->batchCameraCapacity,
                1u) ||
            !ensureStructuredBuffer(
                backendContext.renderDevice, "CRESSimNeo.ForwardPipeline.BatchCameraLayers",
                sizeof(std::uint32_t), batchCameraCount, Diligent::BIND_SHADER_RESOURCE,
                mGpuIndirectState->batchCameraLayersBuffer, mGpuIndirectState->batchCameraCapacity,
                1u))
        {
            return false;
        }
    }
    if (!writeBuffer(backendContext.immediateContext, mGpuIndirectState->batchCameraIndicesBuffer,
                     batchCameraIndices.data(), batchCameraIndices.size() * sizeof(std::uint32_t)))
    {
        return false;
    }
    if (!writeBuffer(backendContext.immediateContext, mGpuIndirectState->batchCameraLayersBuffer,
                     batchCameraLayers.data(), batchCameraLayers.size() * sizeof(std::uint32_t)))
    {
        return false;
    }

    const auto uploadIndirectSet = [&](GpuIndirectState::BufferSet& bufferSet,
                                       const std::vector<IndirectCommandRegistryEntry>& registry,
                                       const char* namePrefix, std::uint32_t queueMode) -> bool
    {
        const std::uint32_t commandCount = static_cast<std::uint32_t>(registry.size());
        std::uint32_t visibleCapacity    = 0u;
        std::vector<IndirectCommandDesc> commandDescs(commandCount);
        bufferSet.drawListOffsets.assign(commandCount, 0u);
        for (std::uint32_t commandIndex = 0u; commandIndex < commandCount; ++commandIndex)
        {
            const std::uint32_t bucketCapacity =
                registry[commandIndex].maxVisibleCount * batchCameraCount;
            bufferSet.drawListOffsets[commandIndex] = visibleCapacity;
            commandDescs[commandIndex]              = IndirectCommandDesc{
                visibleCapacity, bucketCapacity, registry[commandIndex].drawCommand.indexCount, 0u};
            visibleCapacity += bucketCapacity;
        }

        if (commandCount == 0u || visibleCapacity == 0u)
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
            if (!ensureStructuredBuffer(backendContext.renderDevice, descName.c_str(),
                                        sizeof(IndirectCommandDesc), commandCount,
                                        Diligent::BIND_SHADER_RESOURCE, bufferSet.commandDescBuffer,
                                        bufferSet.commandCapacity, 1u) ||
                !ensureStructuredBuffer(
                    backendContext.renderDevice, countName.c_str(), sizeof(std::uint32_t),
                    commandCount, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                    bufferSet.commandCountsBuffer, bufferSet.commandCapacity, 1u) ||
                !ensureStructuredBuffer(
                    backendContext.renderDevice, argsName.c_str(), sizeof(std::uint32_t) * 5u,
                    commandCount,
                    Diligent::BIND_UNORDERED_ACCESS | Diligent::BIND_INDIRECT_DRAW_ARGS,
                    bufferSet.drawIndexedCommandsBuffer, bufferSet.commandCapacity, 1u))
            {
                return false;
            }
        }

        if (bufferSet.visiblePairCapacity < visibleCapacity ||
            bufferSet.visiblePairBuffer == nullptr)
        {
            const std::string visibleName = std::string{namePrefix} + ".VisiblePairs";
            if (!ensureStructuredBuffer(
                    backendContext.renderDevice, visibleName.c_str(),
                    sizeof(gpu::GpuVisiblePairInstance), visibleCapacity,
                    Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                    bufferSet.visiblePairBuffer, bufferSet.visiblePairCapacity, 1u))
            {
                return false;
            }
        }

        if (!writeBuffer(backendContext.immediateContext, bufferSet.commandDescBuffer,
                         commandDescs.data(), commandDescs.size() * sizeof(IndirectCommandDesc)))
        {
            return false;
        }

        if (!updateConstants(backendContext.immediateContext,
                             mGpuIndirectState->resetConstantsBuffer,
                             GraphicsIndirectPassConstants{commandCount, 0u, 0u, 0u}))
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

        if (!updateConstants(backendContext.immediateContext,
                             mGpuIndirectState->filterConstantsBuffer,
                             GraphicsIndirectPassConstants{gpuScene.layout.maxObjectsPerEnv,
                                                           batchCameraCount, queueMode, 0u}))
        {
            return false;
        }
        const std::array filterBindings{
            gpu::GpuBufferBinding{"GraphicsIndirectFilterConstants",
                                  mGpuIndirectState->filterConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_PreparedCameras", gpuScene.preparedCamerasBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableQueueInfo", gpuScene.renderableQueueInfoBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableVisibilityFlags",
                                  gpuScene.renderableVisibilityFlagsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableShadowCascadeMasks",
                                  gpuScene.renderableShadowCascadeMasksBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BatchCameraIndices",
                                  mGpuIndirectState->batchCameraIndicesBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_BatchCameraLayers", mGpuIndirectState->batchCameraLayersBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandDescs", bufferSet.commandDescBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCountsRW", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_VisiblePairsRW", bufferSet.visiblePairBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!mGpuIndirectState->filterPass.dispatch(
                backendContext.immediateContext, 0u, filterBindings,
                dispatchGroupCount(gpuScene.layout.maxObjectsPerEnv * batchCameraCount)))
        {
            return false;
        }

        if (!updateConstants(backendContext.immediateContext,
                             mGpuIndirectState->composeConstantsBuffer,
                             GraphicsIndirectPassConstants{commandCount, 0u, 0u, 0u}))
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

    if (!uploadIndirectSet(mGpuIndirectState->opaque, opaqueRegistry,
                           "CRESSimNeo.ForwardPipeline.Opaque", kQueueModeOpaque))
    {
        return false;
    }
    if (!uploadIndirectSet(mGpuIndirectState->shadow, shadowRegistry,
                           "CRESSimNeo.ForwardPipeline.Shadow", kQueueModeShadow))
    {
        return false;
    }

    auto acquireCachedTarget =
        [&](const gpu::GpuRenderTargetDesc& desc) -> gpu::GpuRenderTargetHandle
    {
        const RenderTargetCacheKey key = makeRenderTargetCacheKey(desc);
        const auto it                  = mLayeredTargetCache.find(key);
        if (it != mLayeredTargetCache.end() &&
            mDevice.renderTargetSystem().isValidRenderTarget(it->second))
        {
            return it->second;
        }

        gpu::GpuRenderTargetHandle handle = mDevice.renderTargetSystem().createRenderTarget(desc);
        if (mDevice.renderTargetSystem().isValidRenderTarget(handle))
        {
            mLayeredTargetCache[key] = handle;
        }
        return handle;
    };

    std::array<gpu::GpuRenderTargetHandle, kShadowCascadeCount> shadowTargets{};
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        gpu::GpuRenderTargetDesc shadowDesc{};
        shadowDesc.width            = kShadowMapResolution;
        shadowDesc.height           = kShadowMapResolution;
        shadowDesc.arraySize        = batchCameraCount;
        shadowDesc.color            = false;
        shadowDesc.depth            = true;
        shadowDesc.shaderReadable   = true;
        shadowDesc.layeredRendering = true;
        shadowDesc.debugName = "CRESSimNeo.ShadowMap.Batch.Cascade" + std::to_string(cascadeIdx);
        shadowTargets[cascadeIdx] = acquireCachedTarget(shadowDesc);
        if (!mDevice.renderTargetSystem().isValidRenderTarget(shadowTargets[cascadeIdx]))
        {
            return false;
        }
    }

    if (batchView.light.intensity > 0.0f && mShadowPass != nullptr && !shadowRegistry.empty())
    {
        mShadowPass->setVisiblePairBuffer(mGpuIndirectState->shadow.visiblePairBuffer);
        for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
        {
            const gpu::GpuRenderTargetBinding shadowBinding{shadowTargets[cascadeIdx], 0u,
                                                            batchCameraCount};
            mDevice.renderTargetSystem().setRenderTargetViewport(shadowBinding,
                                                                 gpu::GpuRenderViewport{});
            gpu::GpuRenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor      = false;
            shadowBegin.clearDepth      = true;
            shadowBegin.clearDepthValue = 1.0f;
            mDevice.renderTargetSystem().beginRenderTarget(shadowBinding, frameContext,
                                                           shadowBegin);
            for (std::uint32_t commandIndex = 0u;
                 commandIndex < static_cast<std::uint32_t>(shadowRegistry.size()); ++commandIndex)
            {
                if ((commandIndex % kShadowCascadeCount) != cascadeIdx)
                {
                    continue;
                }

                ForwardDrawCommand drawCommand = shadowRegistry[commandIndex].drawCommand;
                drawCommand.drawListOffset =
                    mGpuIndirectState->shadow.drawListOffsets[commandIndex];
                drawCommand.useDrawListBuffer = 1u;
                if (mShadowPass->drawIndirect(shadowBinding, drawCommand, 0u, cascadeIdx,
                                              mGpuIndirectState->shadow.drawIndexedCommandsBuffer,
                                              static_cast<Diligent::Uint64>(commandIndex) *
                                                  sizeof(std::uint32_t) * 5u))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(shadowBinding, frameContext);
        }
    }

    mForwardOpaquePass->setShadowMapTargets(shadowTargets, kShadowCascadeCount);
    mForwardOpaquePass->setVisiblePairBuffer(mGpuIndirectState->opaque.visiblePairBuffer);
    if (!mForwardOpaquePass->beginBatchFrame(batchView))
    {
        return false;
    }

    mDevice.renderTargetSystem().setRenderTargetViewport(batchView.renderBinding,
                                                         viewportForBatch(batchView));
    gpu::GpuRenderPassBeginDesc mainBegin{};
    mainBegin.clearColor         = batchView.cameras.front().clearColor;
    mainBegin.clearDepth         = batchView.cameras.front().clearDepth;
    mainBegin.clearColorValue[0] = batchView.cameras.front().clearColorValue.x;
    mainBegin.clearColorValue[1] = batchView.cameras.front().clearColorValue.y;
    mainBegin.clearColorValue[2] = batchView.cameras.front().clearColorValue.z;
    mainBegin.clearColorValue[3] = batchView.cameras.front().clearColorValue.w;
    mainBegin.clearDepthValue    = batchView.cameras.front().clearDepthValue;
    mDevice.renderTargetSystem().beginRenderTarget(batchView.renderBinding, frameContext,
                                                   mainBegin);

    for (std::uint32_t commandIndex = 0u;
         commandIndex < static_cast<std::uint32_t>(opaqueRegistry.size()); ++commandIndex)
    {
        ForwardDrawCommand drawCommand = opaqueRegistry[commandIndex].drawCommand;
        drawCommand.drawListOffset     = mGpuIndirectState->opaque.drawListOffsets[commandIndex];
        drawCommand.useDrawListBuffer  = 1u;
        if (mForwardOpaquePass->drawIndirect(batchView.renderBinding, drawCommand,
                                             mGpuIndirectState->opaque.drawIndexedCommandsBuffer,
                                             static_cast<Diligent::Uint64>(commandIndex) *
                                                 sizeof(std::uint32_t) * 5u))
        {
            ++outStats.opaqueDrawCalls;
        }
    }
    mDevice.renderTargetSystem().endRenderTarget(batchView.renderBinding, frameContext);

    return true;
}

} // namespace cressim::neo::graphics::detail
