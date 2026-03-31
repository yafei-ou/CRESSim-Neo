#include "graphics/passes/forward_pipeline.h"

#include "gpu/gpu_buffer_utils.h"
#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/passes/forward_opaque_pass.h"
#include "graphics/passes/shadow_pass.h"

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

struct LocalShadowPrepareConstants
{
    std::uint32_t envCount               = 0u;
    std::uint32_t maxObjectsPerEnv       = 0u;
    std::uint32_t maxLightsPerEnv        = 0u;
    std::uint32_t localShadowBucketCount = 0u;
};

struct DrawIndexedIndirectArgs
{
    std::uint32_t numIndices          = 0u;
    std::uint32_t numInstances        = 0u;
    std::uint32_t firstIndexLocation  = 0u;
    std::int32_t baseVertex           = 0;
    std::uint32_t firstInstanceOffset = 0u;
};

constexpr std::uint32_t kQueueModeOpaque          = 0u;
constexpr std::uint32_t kQueueModeShadow          = 1u;
constexpr std::uint32_t kShadowPassModeLocal      = 1u;
constexpr std::uint32_t kLocalShadowMapResolution = 1024u;
constexpr std::uint32_t kPointShadowMapResolution = 512u;
constexpr std::uint32_t kLocalShadowViewsPerEnv   = kShadowedLocalLightCap + kShadowedPointLightCap;
constexpr std::uint32_t kLocalShadowEnvBoundsWords = 8u;

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

constexpr gpu::GpuComputePassDefinition kLocalShadowIndirectResetPassDefinition = {
    "graphics/graphics_local_shadow_indirect_reset.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowIndirectReset",
    "CRESSimNeo.Graphics.LocalShadowIndirectReset.PSO",
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
    {Diligent::SHADER_TYPE_COMPUTE, "g_BatchCameras",
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

constexpr Diligent::ShaderResourceVariableDesc kLocalShadowResetVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsLocalShadowPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LightShadowAssignmentsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowViewsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kLocalShadowResetPassDefinition = {
    "graphics/graphics_local_shadow_reset.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowReset",
    "CRESSimNeo.Graphics.LocalShadowReset.PSO",
    kLocalShadowResetVars,
    std::size(kLocalShadowResetVars),
};

constexpr Diligent::ShaderResourceVariableDesc kLocalShadowEnvBoundsResetVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsLocalShadowPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowEnvBoundsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kLocalShadowEnvBoundsResetPassDefinition = {
    "graphics/graphics_local_shadow_env_bounds_reset.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowEnvBoundsReset",
    "CRESSimNeo.Graphics.LocalShadowEnvBoundsReset.PSO",
    kLocalShadowEnvBoundsResetVars,
    std::size(kLocalShadowEnvBoundsResetVars),
};

constexpr Diligent::ShaderResourceVariableDesc kLocalShadowEnvBoundsPrepareVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsLocalShadowPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowEnvBoundsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kLocalShadowEnvBoundsPreparePassDefinition = {
    "graphics/graphics_local_shadow_env_bounds_prepare.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowEnvBoundsPrepare",
    "CRESSimNeo.Graphics.LocalShadowEnvBoundsPrepare.PSO",
    kLocalShadowEnvBoundsPrepareVars,
    std::size(kLocalShadowEnvBoundsPrepareVars),
};

constexpr Diligent::ShaderResourceVariableDesc kLocalShadowViewPrepareVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsLocalShadowPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalLightSelections",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LightInputs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowEnvBounds",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LightShadowAssignmentsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowViewsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kLocalShadowViewPreparePassDefinition = {
    "graphics/graphics_local_shadow_view_prepare.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowViewPrepare",
    "CRESSimNeo.Graphics.LocalShadowViewPrepare.PSO",
    kLocalShadowViewPrepareVars,
    std::size(kLocalShadowViewPrepareVars),
};

constexpr Diligent::ShaderResourceVariableDesc kLocalShadowFilterVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsLocalShadowPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableQueueInfo",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LocalShadowViews",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandDescs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CommandCountsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
    {Diligent::SHADER_TYPE_COMPUTE, "g_VisiblePairsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_DYNAMIC},
};

constexpr gpu::GpuComputePassDefinition kLocalShadowFilter2DPassDefinition = {
    "graphics/graphics_local_shadow_filter_2d.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowFilter2D",
    "CRESSimNeo.Graphics.LocalShadowFilter2D.PSO",
    kLocalShadowFilterVars,
    std::size(kLocalShadowFilterVars),
};

constexpr gpu::GpuComputePassDefinition kLocalShadowFilterPointPassDefinition = {
    "graphics/graphics_local_shadow_filter_point.cs.hlsl",
    "CRESSimNeo.Graphics.LocalShadowFilterPoint",
    "CRESSimNeo.Graphics.LocalShadowFilterPoint.PSO",
    kLocalShadowFilterVars,
    std::size(kLocalShadowFilterVars),
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kIndirectThreadGroupSize - 1u) / kIndirectThreadGroupSize;
}

bool ensureStructuredBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                            std::uint32_t elementStride, std::uint32_t elementCount,
                            Diligent::BIND_FLAGS bindFlags, Diligent::Uint64 contextMask,
                            Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer,
                            std::uint32_t &inOutCapacity, std::uint32_t minimumCapacity)
{
    return gpu::detail::ensureStructuredBufferCapacity(
        renderDevice, name, elementStride, elementCount, minimumCapacity, bindFlags,
        Diligent::USAGE_DEFAULT, Diligent::CPU_ACCESS_NONE, contextMask, outBuffer, inOutCapacity);
}

bool writeBuffer(Diligent::IDeviceContext *context, Diligent::IBuffer *buffer, const void *data,
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

bool updateConstants(Diligent::IDeviceContext *context, Diligent::IBuffer *constantBuffer,
                     const GraphicsIndirectPassConstants &constants)
{
    if (context == nullptr || constantBuffer == nullptr)
    {
        return false;
    }

    void *mapped = nullptr;
    context->MapBuffer(constantBuffer, Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD, mapped);
    if (mapped == nullptr)
    {
        return false;
    }
    std::memcpy(mapped, &constants, sizeof(constants));
    context->UnmapBuffer(constantBuffer, Diligent::MAP_WRITE);
    return true;
}

gpu::GpuRenderViewport viewportForBatch(const CameraBatchView &batchView)
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
        Diligent::RefCntAutoPtr<Diligent::IBuffer> batchCameraBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandDescBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> visiblePairBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> commandCountsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> drawIndexedCommandsBuffer;
        std::uint32_t batchCameraCapacity = 0u;
        std::uint32_t commandCapacity     = 0u;
        std::uint32_t visiblePairCapacity = 0u;
        std::vector<std::uint32_t> drawListOffsets;
    };

    gpu::GpuComputePass resetPass;
    gpu::GpuComputePass localShadowIndirectResetPass;
    gpu::GpuComputePass filterPass;
    gpu::GpuComputePass composePass;
    gpu::GpuComputePass localShadowResetPass;
    gpu::GpuComputePass localShadowEnvBoundsResetPass;
    gpu::GpuComputePass localShadowEnvBoundsPreparePass;
    gpu::GpuComputePass localShadowViewPreparePass;
    gpu::GpuComputePass localShadowFilter2DPass;
    gpu::GpuComputePass localShadowFilterPointPass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> resetConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> filterConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> composeConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> localShadowPrepareConstantsBuffer;
    BufferSet opaque;
    BufferSet shadow;
    BufferSet localShadow2D;
    BufferSet localShadowPoint;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> localShadowEnvBoundsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> localShadowViewBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> localShadowAssignmentBuffer;
    std::uint32_t localShadowEnvBoundsCapacity  = 0u;
    std::uint32_t localShadowViewCapacity       = 0u;
    std::uint32_t localShadowAssignmentCapacity = 0u;
    bool initialized                            = false;
};

ForwardPipeline::ForwardPipeline(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                                 IblQualityTier iblQualityTier)
    : mDevice(device), mResourceManager(resourceManager), mIblQualityTier(iblQualityTier),
      mGpuIndirectState(std::make_unique<GpuIndirectState>())
{
}

ForwardPipeline::~ForwardPipeline()
{
    for (const auto &[key, target] : mLayeredTargetCache)
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
    mForwardOpaquePass =
        std::make_unique<ForwardOpaquePass>(mDevice, mResourceManager, mIblQualityTier);
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

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }
    const Diligent::Uint64 graphicsContextMask = gpu::contextMaskForId(backendContext.contextId);

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr || mGpuIndirectState == nullptr)
    {
        return false;
    }
    if (!mGpuIndirectState->resetPass.initialize(mDevice, streamFactory, graphicsContextMask,
                                                 kIndirectResetPassDefinition) ||
        !mGpuIndirectState->localShadowIndirectResetPass.initialize(
            mDevice, streamFactory, graphicsContextMask, kLocalShadowIndirectResetPassDefinition) ||
        !mGpuIndirectState->filterPass.initialize(mDevice, streamFactory, graphicsContextMask,
                                                  kIndirectFilterPassDefinition) ||
        !mGpuIndirectState->localShadowResetPass.initialize(
            mDevice, streamFactory, graphicsContextMask, kLocalShadowResetPassDefinition) ||
        !mGpuIndirectState->localShadowEnvBoundsResetPass.initialize(
            mDevice, streamFactory, graphicsContextMask,
            kLocalShadowEnvBoundsResetPassDefinition) ||
        !mGpuIndirectState->localShadowEnvBoundsPreparePass.initialize(
            mDevice, streamFactory, graphicsContextMask,
            kLocalShadowEnvBoundsPreparePassDefinition) ||
        !mGpuIndirectState->localShadowViewPreparePass.initialize(
            mDevice, streamFactory, graphicsContextMask, kLocalShadowViewPreparePassDefinition) ||
        !mGpuIndirectState->localShadowFilter2DPass.initialize(
            mDevice, streamFactory, graphicsContextMask, kLocalShadowFilter2DPassDefinition) ||
        !mGpuIndirectState->localShadowFilterPointPass.initialize(
            mDevice, streamFactory, graphicsContextMask, kLocalShadowFilterPointPassDefinition) ||
        !mGpuIndirectState->composePass.initialize(mDevice, streamFactory, graphicsContextMask,
                                                   kIndirectComposePassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc constantsDesc{};
    constantsDesc.Usage                = Diligent::USAGE_DYNAMIC;
    constantsDesc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    constantsDesc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    constantsDesc.Size                 = sizeof(GraphicsIndirectPassConstants);
    constantsDesc.ImmediateContextMask = graphicsContextMask;

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

    constantsDesc.Name = "CRESSimNeo.ForwardPipeline.LocalShadowPrepareConstants";
    constantsDesc.Size = sizeof(LocalShadowPrepareConstants);
    backendContext.renderDevice->CreateBuffer(
        constantsDesc, nullptr, &mGpuIndirectState->localShadowPrepareConstantsBuffer);
    if (mGpuIndirectState->localShadowPrepareConstantsBuffer == nullptr)
    {
        return false;
    }

    mGpuIndirectState->initialized = true;
    mInitialized                   = true;
    return true;
}

bool ForwardPipeline::executeBatch(const common::FrameContext &frameContext,
                                   const CameraBatchView &batchView, const HostSceneView &sceneView,
                                   const std::vector<EnvMainLightState> &envMainLights,
                                   ForwardPassExecutionStats &outStats)
{
    outStats = {};
    if (!mInitialized || mForwardOpaquePass == nullptr || batchView.cameras.empty())
    {
        return false;
    }

    const GpuEntitySceneView emptyGpuScene{};
    const GpuEntitySceneView &gpuScene =
        sceneView.gpuEntityScene != nullptr ? *sceneView.gpuEntityScene : emptyGpuScene;
    const std::vector<IndirectCommandRegistryEntry> emptyRegistry;
    const std::vector<IndirectCommandRegistryEntry> &opaqueRegistry =
        sceneView.opaqueDrawRegistry != nullptr ? *sceneView.opaqueDrawRegistry : emptyRegistry;
    const std::vector<IndirectCommandRegistryEntry> &shadowRegistry =
        sceneView.shadowDrawRegistry != nullptr ? *sceneView.shadowDrawRegistry : emptyRegistry;
    const std::vector<IndirectCommandRegistryEntry> &localShadowRegistry =
        sceneView.localShadowDrawRegistry != nullptr ? *sceneView.localShadowDrawRegistry
                                                     : emptyRegistry;
    if (gpuScene.poses.positionsBuffer == nullptr || gpuScene.poses.orientationsBuffer == nullptr ||
        gpuScene.poses.scalesBuffer == nullptr || gpuScene.renderableMetadataBuffer == nullptr ||
        gpuScene.preparedCamerasBuffer == nullptr ||
        gpuScene.renderableQueueInfoBuffer == nullptr ||
        gpuScene.renderableVisibilityFlagsBuffer == nullptr ||
        gpuScene.renderableShadowCascadeMasksBuffer == nullptr ||
        gpuScene.lightInputsBuffer == nullptr || gpuScene.localLightSelectionBuffer == nullptr)
    {
        return false;
    }

    mForwardOpaquePass->setGpuSceneView(gpuScene);
    mForwardOpaquePass->setEnvironmentIbls(sceneView.environmentIbls, gpuScene.layout.envCount);
    if (mShadowPass != nullptr)
    {
        mShadowPass->setGpuSceneView(gpuScene);
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr || backendContext.graphicsContext == nullptr ||
        mGpuIndirectState == nullptr || !mGpuIndirectState->initialized)
    {
        return false;
    }
    const Diligent::Uint64 graphicsContextMask = gpu::contextMaskForId(backendContext.contextId);

    const std::uint32_t envCount                  = gpuScene.layout.envCount;
    const std::uint32_t totalLightCount           = gpuScene.layout.totalLightCapacity();
    const std::uint32_t totalObjectCount          = gpuScene.layout.totalRenderableObjectCapacity();
    const std::uint32_t totalLocalShadowViewCount = envCount * kLocalShadowViewsPerEnv;
    const std::uint32_t totalLocal2DSubviewCount  = envCount * kShadowedLocalLightCap;
    const std::uint32_t totalPointFaceCount =
        envCount * kShadowedPointLightCap * kLocalShadowMaxFaceCount;

    const std::uint32_t batchCameraCount = static_cast<std::uint32_t>(batchView.cameras.size());
    std::vector<GpuBatchCameraMetadata> opaqueBatchCameras(batchCameraCount);
    std::vector<GpuBatchCameraMetadata> shadowBatchCameras;
    shadowBatchCameras.reserve(batchCameraCount);
    for (std::uint32_t i = 0u; i < batchCameraCount; ++i)
    {
        const ResolvedCameraView &camera = batchView.cameras[i];
        GpuBatchCameraMetadata batchCamera{};
        batchCamera.globalCameraIndex = camera.globalCameraIndex;
        batchCamera.envIndex          = camera.envIndex;
        batchCamera.mainLightIndex    = mainDirectionalLightIndex(gpuScene.layout, camera.envIndex);
        batchCamera.colorLayer =
            camera.outputBinding.firstLayer - batchView.renderBinding.firstLayer;
        batchCamera.shadowLayer = kInvalidBatchCameraLayer;

        if (camera.envIndex < envMainLights.size() && envMainLights[camera.envIndex].castsShadows)
        {
            batchCamera.shadowLayer = static_cast<std::uint32_t>(shadowBatchCameras.size());
            shadowBatchCameras.push_back(batchCamera);
        }

        opaqueBatchCameras[i] = batchCamera;
    }

    const auto uploadBatchCameraBuffer =
        [&](GpuIndirectState::BufferSet &bufferSet, const char *name,
            const std::vector<GpuBatchCameraMetadata> &batchCameras) -> bool
    {
        if (batchCameras.empty())
        {
            return true;
        }

        if (bufferSet.batchCameraCapacity < batchCameras.size() ||
            bufferSet.batchCameraBuffer == nullptr)
        {
            if (!ensureStructuredBuffer(
                    backendContext.renderDevice, name, sizeof(GpuBatchCameraMetadata),
                    static_cast<std::uint32_t>(batchCameras.size()), Diligent::BIND_SHADER_RESOURCE,
                    graphicsContextMask, bufferSet.batchCameraBuffer, bufferSet.batchCameraCapacity,
                    1u))
            {
                return false;
            }
        }

        return writeBuffer(backendContext.graphicsContext, bufferSet.batchCameraBuffer,
                           batchCameras.data(),
                           batchCameras.size() * sizeof(GpuBatchCameraMetadata));
    };

    if (!uploadBatchCameraBuffer(mGpuIndirectState->opaque,
                                 "CRESSimNeo.ForwardPipeline.OpaqueBatchCameras",
                                 opaqueBatchCameras) ||
        !uploadBatchCameraBuffer(mGpuIndirectState->shadow,
                                 "CRESSimNeo.ForwardPipeline.ShadowBatchCameras",
                                 shadowBatchCameras))
    {
        return false;
    }

    const auto uploadIndirectSet = [&](GpuIndirectState::BufferSet &bufferSet,
                                       const std::vector<IndirectCommandRegistryEntry> &registry,
                                       const char *namePrefix, std::uint32_t queueMode,
                                       std::uint32_t cameraCount) -> bool
    {
        const std::uint32_t commandCount = static_cast<std::uint32_t>(registry.size());
        std::uint32_t visibleCapacity    = 0u;
        std::vector<IndirectCommandDesc> commandDescs(commandCount);
        bufferSet.drawListOffsets.assign(commandCount, 0u);
        for (std::uint32_t commandIndex = 0u; commandIndex < commandCount; ++commandIndex)
        {
            const std::uint32_t bucketCapacity =
                registry[commandIndex].maxVisibleCount * cameraCount;
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
            if (!ensureStructuredBuffer(
                    backendContext.renderDevice, descName.c_str(), sizeof(IndirectCommandDesc),
                    commandCount, Diligent::BIND_SHADER_RESOURCE, graphicsContextMask,
                    bufferSet.commandDescBuffer, bufferSet.commandCapacity, 1u) ||
                !ensureStructuredBuffer(
                    backendContext.renderDevice, countName.c_str(), sizeof(std::uint32_t),
                    commandCount, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                    graphicsContextMask, bufferSet.commandCountsBuffer, bufferSet.commandCapacity,
                    1u) ||
                !ensureStructuredBuffer(backendContext.renderDevice, argsName.c_str(),
                                        sizeof(std::uint32_t) * 5u, commandCount,
                                        Diligent::BIND_UNORDERED_ACCESS |
                                            Diligent::BIND_INDIRECT_DRAW_ARGS,
                                        graphicsContextMask, bufferSet.drawIndexedCommandsBuffer,
                                        bufferSet.commandCapacity, 1u))
            {
                return false;
            }
        }

        if (bufferSet.visiblePairCapacity < visibleCapacity ||
            bufferSet.visiblePairBuffer == nullptr)
        {
            const std::string visibleName = std::string{namePrefix} + ".VisiblePairs";
            if (!ensureStructuredBuffer(backendContext.renderDevice, visibleName.c_str(),
                                        sizeof(GpuVisiblePairInstance), visibleCapacity,
                                        Diligent::BIND_SHADER_RESOURCE |
                                            Diligent::BIND_UNORDERED_ACCESS,
                                        graphicsContextMask, bufferSet.visiblePairBuffer,
                                        bufferSet.visiblePairCapacity, 1u))
            {
                return false;
            }
        }

        if (!writeBuffer(backendContext.graphicsContext, bufferSet.commandDescBuffer,
                         commandDescs.data(), commandDescs.size() * sizeof(IndirectCommandDesc)))
        {
            return false;
        }

        if (!updateConstants(backendContext.graphicsContext,
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
        if (!mGpuIndirectState->resetPass.dispatch(backendContext.graphicsContext, 0u,
                                                   resetBindings, dispatchGroupCount(commandCount)))
        {
            return false;
        }

        if (!updateConstants(
                backendContext.graphicsContext, mGpuIndirectState->filterConstantsBuffer,
                GraphicsIndirectPassConstants{gpuScene.layout.maxRenderableObjectsPerEnv,
                                              cameraCount, queueMode, 0u}))
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
            gpu::GpuBufferBinding{"g_BatchCameras", bufferSet.batchCameraBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandDescs", bufferSet.commandDescBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCountsRW", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_VisiblePairsRW", bufferSet.visiblePairBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!mGpuIndirectState->filterPass.dispatch(
                backendContext.graphicsContext, 0u, filterBindings,
                dispatchGroupCount(gpuScene.layout.maxRenderableObjectsPerEnv * cameraCount)))
        {
            return false;
        }

        if (!updateConstants(backendContext.graphicsContext,
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
            backendContext.graphicsContext, 0u, composeBindings, dispatchGroupCount(commandCount));
    };

    const auto uploadLocalShadowIndirectSet =
        [&](GpuIndirectState::BufferSet &bufferSet,
            const std::vector<IndirectCommandRegistryEntry> &registry, const char *namePrefix,
            std::uint32_t subviewCount, gpu::GpuComputePass &filterPass) -> bool
    {
        const std::uint32_t bucketCount  = static_cast<std::uint32_t>(registry.size());
        const std::uint32_t commandCount = bucketCount * subviewCount;
        if (commandCount == 0u || subviewCount == 0u)
        {
            return true;
        }

        std::vector<IndirectCommandDesc> commandDescs(commandCount);
        std::uint32_t visibleCapacity = 0u;
        for (std::uint32_t bucketIndex = 0u; bucketIndex < bucketCount; ++bucketIndex)
        {
            for (std::uint32_t subviewIndex = 0u; subviewIndex < subviewCount; ++subviewIndex)
            {
                const std::uint32_t commandIndex = bucketIndex * subviewCount + subviewIndex;
                commandDescs[commandIndex] =
                    IndirectCommandDesc{visibleCapacity, registry[bucketIndex].maxVisibleCount,
                                        registry[bucketIndex].drawCommand.indexCount, 0u};
                visibleCapacity += registry[bucketIndex].maxVisibleCount;
            }
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
                    commandCount, Diligent::BIND_SHADER_RESOURCE, graphicsContextMask,
                    bufferSet.commandDescBuffer, bufferSet.commandCapacity, 1u) ||
                !ensureStructuredBuffer(
                    backendContext.renderDevice, countName.c_str(), sizeof(std::uint32_t),
                    commandCount, Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                    graphicsContextMask, bufferSet.commandCountsBuffer, bufferSet.commandCapacity,
                    1u) ||
                !ensureStructuredBuffer(backendContext.renderDevice, argsName.c_str(),
                                        sizeof(std::uint32_t) * 5u, commandCount,
                                        Diligent::BIND_UNORDERED_ACCESS |
                                            Diligent::BIND_INDIRECT_DRAW_ARGS,
                                        graphicsContextMask, bufferSet.drawIndexedCommandsBuffer,
                                        bufferSet.commandCapacity, 1u))
            {
                return false;
            }
        }

        if (bufferSet.visiblePairCapacity < visibleCapacity ||
            bufferSet.visiblePairBuffer == nullptr)
        {
            const std::string visibleName = std::string{namePrefix} + ".VisiblePairs";
            if (!ensureStructuredBuffer(backendContext.renderDevice, visibleName.c_str(),
                                        sizeof(GpuVisiblePairInstance), visibleCapacity,
                                        Diligent::BIND_SHADER_RESOURCE |
                                            Diligent::BIND_UNORDERED_ACCESS,
                                        graphicsContextMask, bufferSet.visiblePairBuffer,
                                        bufferSet.visiblePairCapacity, 1u))
            {
                return false;
            }
        }

        if (!writeBuffer(backendContext.graphicsContext, bufferSet.commandDescBuffer,
                         commandDescs.data(), commandDescs.size() * sizeof(IndirectCommandDesc)))
        {
            return false;
        }

        if (!updateConstants(backendContext.graphicsContext,
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
        if (!mGpuIndirectState->localShadowIndirectResetPass.dispatch(
                backendContext.graphicsContext, 0u, resetBindings,
                dispatchGroupCount(commandCount)))
        {
            return false;
        }

        const std::array filterBindings{
            gpu::GpuBufferBinding{"GraphicsLocalShadowPrepareConstants",
                                  mGpuIndirectState->localShadowPrepareConstantsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_EntityPositions", gpuScene.poses.positionsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_EntityOrientations", gpuScene.poses.orientationsBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_EntityScales", gpuScene.poses.scalesBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableMetadata", gpuScene.renderableMetadataBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_RenderableQueueInfo", gpuScene.renderableQueueInfoBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_LocalShadowViews", mGpuIndirectState->localShadowViewBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandDescs", bufferSet.commandDescBuffer,
                                  Diligent::BUFFER_VIEW_SHADER_RESOURCE},
            gpu::GpuBufferBinding{"g_CommandCountsRW", bufferSet.commandCountsBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            gpu::GpuBufferBinding{"g_VisiblePairsRW", bufferSet.visiblePairBuffer,
                                  Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        };
        if (!filterPass.dispatch(backendContext.graphicsContext, 0u, filterBindings,
                                 dispatchGroupCount(totalObjectCount * subviewCount)))
        {
            return false;
        }

        if (!updateConstants(backendContext.graphicsContext,
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
            backendContext.graphicsContext, 0u, composeBindings, dispatchGroupCount(commandCount));
    };

    if (!uploadIndirectSet(mGpuIndirectState->opaque, opaqueRegistry,
                           "CRESSimNeo.ForwardPipeline.Opaque", kQueueModeOpaque, batchCameraCount))
    {
        return false;
    }
    const std::uint32_t shadowCameraCount = static_cast<std::uint32_t>(shadowBatchCameras.size());
    if (!uploadIndirectSet(mGpuIndirectState->shadow, shadowRegistry,
                           "CRESSimNeo.ForwardPipeline.Shadow", kQueueModeShadow,
                           shadowCameraCount))
    {
        return false;
    }

    if (mGpuIndirectState->localShadowAssignmentCapacity < std::max(totalLightCount, 1u) ||
        mGpuIndirectState->localShadowAssignmentBuffer == nullptr)
    {
        if (!ensureStructuredBuffer(
                backendContext.renderDevice, "CRESSimNeo.ForwardPipeline.LocalShadowAssignments",
                sizeof(GpuLightShadowAssignment), std::max(totalLightCount, 1u),
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                graphicsContextMask, mGpuIndirectState->localShadowAssignmentBuffer,
                mGpuIndirectState->localShadowAssignmentCapacity, 1u))
        {
            return false;
        }
    }
    if (mGpuIndirectState->localShadowViewCapacity < std::max(totalLocalShadowViewCount, 1u) ||
        mGpuIndirectState->localShadowViewBuffer == nullptr)
    {
        if (!ensureStructuredBuffer(
                backendContext.renderDevice, "CRESSimNeo.ForwardPipeline.LocalShadowViews",
                sizeof(GpuLocalShadowView), std::max(totalLocalShadowViewCount, 1u),
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                graphicsContextMask, mGpuIndirectState->localShadowViewBuffer,
                mGpuIndirectState->localShadowViewCapacity, 1u))
        {
            return false;
        }
    }
    if (mGpuIndirectState->localShadowEnvBoundsCapacity <
            std::max(envCount * kLocalShadowEnvBoundsWords, 1u) ||
        mGpuIndirectState->localShadowEnvBoundsBuffer == nullptr)
    {
        if (!ensureStructuredBuffer(
                backendContext.renderDevice, "CRESSimNeo.ForwardPipeline.LocalShadowEnvBounds",
                sizeof(std::uint32_t), std::max(envCount * kLocalShadowEnvBoundsWords, 1u),
                Diligent::BIND_SHADER_RESOURCE | Diligent::BIND_UNORDERED_ACCESS,
                graphicsContextMask, mGpuIndirectState->localShadowEnvBoundsBuffer,
                mGpuIndirectState->localShadowEnvBoundsCapacity, 1u))
        {
            return false;
        }
    }

    {
        LocalShadowPrepareConstants localShadowConstants{};
        localShadowConstants.envCount         = envCount;
        localShadowConstants.maxObjectsPerEnv = gpuScene.layout.maxRenderableObjectsPerEnv;
        localShadowConstants.maxLightsPerEnv  = gpuScene.layout.maxLightsPerEnv;
        localShadowConstants.localShadowBucketCount =
            static_cast<std::uint32_t>(localShadowRegistry.size());

        void *mapped = nullptr;
        backendContext.graphicsContext->MapBuffer(
            mGpuIndirectState->localShadowPrepareConstantsBuffer, Diligent::MAP_WRITE,
            Diligent::MAP_FLAG_DISCARD, mapped);
        if (mapped == nullptr)
        {
            return false;
        }
        std::memcpy(mapped, &localShadowConstants, sizeof(localShadowConstants));
        backendContext.graphicsContext->UnmapBuffer(
            mGpuIndirectState->localShadowPrepareConstantsBuffer, Diligent::MAP_WRITE);
    }

    {
        const std::uint32_t resetCount = std::max(totalLightCount, totalLocalShadowViewCount);
        if (resetCount > 0u)
        {
            const std::array localShadowResetBindings{
                gpu::GpuBufferBinding{"GraphicsLocalShadowPrepareConstants",
                                      mGpuIndirectState->localShadowPrepareConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LightShadowAssignmentsRW",
                                      mGpuIndirectState->localShadowAssignmentBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
                gpu::GpuBufferBinding{"g_LocalShadowViewsRW",
                                      mGpuIndirectState->localShadowViewBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };
            if (!mGpuIndirectState->localShadowResetPass.dispatch(backendContext.graphicsContext,
                                                                  0u, localShadowResetBindings,
                                                                  dispatchGroupCount(resetCount)))
            {
                return false;
            }
        }

        if (envCount > 0u)
        {
            const std::array envBoundsResetBindings{
                gpu::GpuBufferBinding{"GraphicsLocalShadowPrepareConstants",
                                      mGpuIndirectState->localShadowPrepareConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LocalShadowEnvBoundsRW",
                                      mGpuIndirectState->localShadowEnvBoundsBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };
            if (!mGpuIndirectState->localShadowEnvBoundsResetPass.dispatch(
                    backendContext.graphicsContext, 0u, envBoundsResetBindings,
                    dispatchGroupCount(envCount)))
            {
                return false;
            }

            const std::array envBoundsPrepareBindings{
                gpu::GpuBufferBinding{"GraphicsLocalShadowPrepareConstants",
                                      mGpuIndirectState->localShadowPrepareConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_RenderableMetadata", gpuScene.renderableMetadataBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_EntityPositions", gpuScene.poses.positionsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LocalShadowEnvBoundsRW",
                                      mGpuIndirectState->localShadowEnvBoundsBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };
            if (!mGpuIndirectState->localShadowEnvBoundsPreparePass.dispatch(
                    backendContext.graphicsContext, 0u, envBoundsPrepareBindings,
                    dispatchGroupCount(totalObjectCount)))
            {
                return false;
            }

            const std::array viewPrepareBindings{
                gpu::GpuBufferBinding{"GraphicsLocalShadowPrepareConstants",
                                      mGpuIndirectState->localShadowPrepareConstantsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LocalLightSelections", gpuScene.localLightSelectionBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LightInputs", gpuScene.lightInputsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LocalShadowEnvBounds",
                                      mGpuIndirectState->localShadowEnvBoundsBuffer,
                                      Diligent::BUFFER_VIEW_SHADER_RESOURCE},
                gpu::GpuBufferBinding{"g_LightShadowAssignmentsRW",
                                      mGpuIndirectState->localShadowAssignmentBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
                gpu::GpuBufferBinding{"g_LocalShadowViewsRW",
                                      mGpuIndirectState->localShadowViewBuffer,
                                      Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
            };
            if (!mGpuIndirectState->localShadowViewPreparePass.dispatch(
                    backendContext.graphicsContext, 0u, viewPrepareBindings,
                    dispatchGroupCount(envCount)))
            {
                return false;
            }
        }
    }

    if (!uploadLocalShadowIndirectSet(mGpuIndirectState->localShadow2D, localShadowRegistry,
                                      "CRESSimNeo.ForwardPipeline.LocalShadow2D",
                                      totalLocal2DSubviewCount,
                                      mGpuIndirectState->localShadowFilter2DPass) ||
        !uploadLocalShadowIndirectSet(mGpuIndirectState->localShadowPoint, localShadowRegistry,
                                      "CRESSimNeo.ForwardPipeline.LocalShadowPoint",
                                      totalPointFaceCount,
                                      mGpuIndirectState->localShadowFilterPointPass))
    {
        return false;
    }

    auto acquireCachedTarget =
        [&](const gpu::GpuRenderTargetDesc &desc) -> gpu::GpuRenderTargetHandle
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
    if (shadowCameraCount > 0u && !shadowRegistry.empty())
    {
        for (std::uint32_t cascadeIdx = 0u; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
        {
            gpu::GpuRenderTargetDesc shadowDesc{};
            shadowDesc.width            = kShadowMapResolution;
            shadowDesc.height           = kShadowMapResolution;
            shadowDesc.arraySize        = shadowCameraCount;
            shadowDesc.color            = false;
            shadowDesc.depth            = true;
            shadowDesc.shaderReadable   = true;
            shadowDesc.layeredRendering = true;
            shadowDesc.debugName =
                "CRESSimNeo.ShadowMap.Batch.Cascade" + std::to_string(cascadeIdx);
            shadowTargets[cascadeIdx] = acquireCachedTarget(shadowDesc);
            if (!mDevice.renderTargetSystem().isValidRenderTarget(shadowTargets[cascadeIdx]))
            {
                return false;
            }
        }
    }

    gpu::GpuRenderTargetHandle localShadowTarget2D{};
    if (totalLocal2DSubviewCount > 0u)
    {
        gpu::GpuRenderTargetDesc shadowDesc{};
        shadowDesc.width            = kLocalShadowMapResolution;
        shadowDesc.height           = kLocalShadowMapResolution;
        shadowDesc.arraySize        = totalLocal2DSubviewCount;
        shadowDesc.color            = false;
        shadowDesc.depth            = true;
        shadowDesc.shaderReadable   = true;
        shadowDesc.layeredRendering = true;
        shadowDesc.debugName        = "CRESSimNeo.ShadowMap.Local2D";
        localShadowTarget2D         = acquireCachedTarget(shadowDesc);
        if (!mDevice.renderTargetSystem().isValidRenderTarget(localShadowTarget2D))
        {
            return false;
        }
    }

    gpu::GpuRenderTargetHandle pointShadowTarget{};
    if (totalPointFaceCount > 0u)
    {
        gpu::GpuRenderTargetDesc shadowDesc{};
        shadowDesc.width            = kPointShadowMapResolution;
        shadowDesc.height           = kPointShadowMapResolution;
        shadowDesc.arraySize        = totalPointFaceCount;
        shadowDesc.color            = false;
        shadowDesc.depth            = true;
        shadowDesc.shaderReadable   = true;
        shadowDesc.layeredRendering = true;
        shadowDesc.debugName        = "CRESSimNeo.ShadowMap.PointFaces";
        pointShadowTarget           = acquireCachedTarget(shadowDesc);
        if (!mDevice.renderTargetSystem().isValidRenderTarget(pointShadowTarget))
        {
            return false;
        }
    }

    if (shadowCameraCount > 0u && mShadowPass != nullptr && !shadowRegistry.empty())
    {
        mShadowPass->setVisiblePairBuffer(mGpuIndirectState->shadow.visiblePairBuffer);
        mShadowPass->setBatchCameraBuffer(mGpuIndirectState->shadow.batchCameraBuffer);
        mShadowPass->setLocalShadowViewBuffer(nullptr);
        for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
        {
            const gpu::GpuRenderTargetBinding shadowBinding{shadowTargets[cascadeIdx], 0u,
                                                            shadowCameraCount};
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
                if (mShadowPass->drawIndirect(shadowBinding, drawCommand, cascadeIdx, 0u,
                                              mGpuIndirectState->shadow.drawIndexedCommandsBuffer,
                                              static_cast<Diligent::Uint64>(commandIndex) *
                                                  sizeof(DrawIndexedIndirectArgs),
                                              1u, sizeof(DrawIndexedIndirectArgs)))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(shadowBinding, frameContext);
        }
    }

    if (mShadowPass != nullptr && !localShadowRegistry.empty())
    {
        mShadowPass->setBatchCameraBuffer(nullptr);
        mShadowPass->setLocalShadowViewBuffer(mGpuIndirectState->localShadowViewBuffer);

        if (mDevice.renderTargetSystem().isValidRenderTarget(localShadowTarget2D) &&
            totalLocal2DSubviewCount > 0u)
        {
            mShadowPass->setVisiblePairBuffer(mGpuIndirectState->localShadow2D.visiblePairBuffer);
            const gpu::GpuRenderTargetBinding shadowBinding{localShadowTarget2D, 0u,
                                                            totalLocal2DSubviewCount};
            mDevice.renderTargetSystem().setRenderTargetViewport(shadowBinding,
                                                                 gpu::GpuRenderViewport{});
            gpu::GpuRenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor      = false;
            shadowBegin.clearDepth      = true;
            shadowBegin.clearDepthValue = 1.0f;
            mDevice.renderTargetSystem().beginRenderTarget(shadowBinding, frameContext,
                                                           shadowBegin);
            for (std::uint32_t bucketIndex = 0u;
                 bucketIndex < static_cast<std::uint32_t>(localShadowRegistry.size());
                 ++bucketIndex)
            {
                ForwardDrawCommand drawCommand = localShadowRegistry[bucketIndex].drawCommand;
                drawCommand.useDrawListBuffer  = 1u;
                if (mShadowPass->drawIndirect(
                        shadowBinding, drawCommand, 0u, kShadowPassModeLocal,
                        mGpuIndirectState->localShadow2D.drawIndexedCommandsBuffer,
                        static_cast<Diligent::Uint64>(bucketIndex * totalLocal2DSubviewCount) *
                            sizeof(DrawIndexedIndirectArgs),
                        totalLocal2DSubviewCount, sizeof(DrawIndexedIndirectArgs)))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(shadowBinding, frameContext);
        }

        if (mDevice.renderTargetSystem().isValidRenderTarget(pointShadowTarget) &&
            totalPointFaceCount > 0u)
        {
            mShadowPass->setVisiblePairBuffer(
                mGpuIndirectState->localShadowPoint.visiblePairBuffer);
            const gpu::GpuRenderTargetBinding shadowBinding{pointShadowTarget, 0u,
                                                            totalPointFaceCount};
            mDevice.renderTargetSystem().setRenderTargetViewport(shadowBinding,
                                                                 gpu::GpuRenderViewport{});
            gpu::GpuRenderPassBeginDesc shadowBegin{};
            shadowBegin.clearColor      = false;
            shadowBegin.clearDepth      = true;
            shadowBegin.clearDepthValue = 1.0f;
            mDevice.renderTargetSystem().beginRenderTarget(shadowBinding, frameContext,
                                                           shadowBegin);
            for (std::uint32_t bucketIndex = 0u;
                 bucketIndex < static_cast<std::uint32_t>(localShadowRegistry.size());
                 ++bucketIndex)
            {
                ForwardDrawCommand drawCommand = localShadowRegistry[bucketIndex].drawCommand;
                drawCommand.useDrawListBuffer  = 1u;
                if (mShadowPass->drawIndirect(
                        shadowBinding, drawCommand, 0u, kShadowPassModeLocal,
                        mGpuIndirectState->localShadowPoint.drawIndexedCommandsBuffer,
                        static_cast<Diligent::Uint64>(bucketIndex * totalPointFaceCount) *
                            sizeof(DrawIndexedIndirectArgs),
                        totalPointFaceCount, sizeof(DrawIndexedIndirectArgs)))
                {
                    ++outStats.shadowDrawCalls;
                }
            }
            mDevice.renderTargetSystem().endRenderTarget(shadowBinding, frameContext);
        }
    }

    mForwardOpaquePass->setShadowMapTargets(shadowTargets,
                                            shadowCameraCount > 0u ? kShadowCascadeCount : 0u);
    mForwardOpaquePass->setLocalShadowResources(
        localShadowTarget2D, pointShadowTarget, mGpuIndirectState->localShadowViewBuffer,
        mGpuIndirectState->localShadowAssignmentBuffer, totalLocalShadowViewCount);
    mForwardOpaquePass->setVisiblePairBuffer(mGpuIndirectState->opaque.visiblePairBuffer);
    mForwardOpaquePass->setBatchCameraBuffer(mGpuIndirectState->opaque.batchCameraBuffer);
    if (!mForwardOpaquePass->beginBatchFrame(batchView.cameras.front().globalCameraIndex))
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
