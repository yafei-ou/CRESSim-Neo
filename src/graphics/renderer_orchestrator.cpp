#include "graphics/renderer.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_source_provider.h"
#include "graphics/output_planner.h"
#include "graphics/passes/display_resolve_pass.h"
#include "graphics/passes/forward_pipeline.h"
#include "graphics/render_plan_builder.h"
#include "graphics/renderer_internal.h"
#include "physics/physics_gpu_scene_view.h"

#include <array>
#include <cstring>

namespace cressim::neo::graphics
{

namespace
{

constexpr std::uint32_t kScenePrepareThreadGroupSize = 64u;

struct SoftBodyWorldAabbFallback
{
    Diligent::float4 minBounds = {0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 maxBounds = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct GraphicsCameraPrepareConstants
{
    std::uint32_t cameraCount         = 0u;
    std::uint32_t maxObjectsPerEnv    = 0u;
    std::uint32_t maxLightsPerEnv     = 0u;
    std::uint32_t shadowMapResolution = kShadowMapResolution;
    std::uint32_t entityPoseCount     = 0u;
    std::uint32_t padding0            = 0u;
    std::uint32_t padding1            = 0u;
    std::uint32_t padding2            = 0u;
};

struct GraphicsScenePrepareConstants
{
    std::uint32_t cameraCount      = 0u;
    std::uint32_t maxObjectsPerEnv = 0u;
    std::uint32_t padding0         = 0u;
    std::uint32_t padding1         = 0u;
};

constexpr Diligent::ShaderResourceVariableDesc kCameraPrepareVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsCameraPrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CameraInputs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_LightInputs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreparedCamerasRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr gpu::GpuComputePassDefinition kCameraPreparePassDefinition = {
    "graphics/graphics_camera_prepare.cs.hlsl",
    "CRESSimNeo.Graphics.CameraPrepare",
    "CRESSimNeo.Graphics.CameraPrepare.PSO",
    kCameraPrepareVars,
    std::size(kCameraPrepareVars),
};

constexpr Diligent::ShaderResourceVariableDesc kScenePrepareVars[] = {
    {Diligent::SHADER_TYPE_COMPUTE, "GraphicsScenePrepareConstants",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityPositions",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityOrientations",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_EntityScales",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableMetadata",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_SoftBodyWorldAabbs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_CurveWorldAabbs",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_PreparedCameras",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableVisibilityFlagsRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
    {Diligent::SHADER_TYPE_COMPUTE, "g_RenderableShadowCascadeMasksRW",
     Diligent::SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE},
};

constexpr gpu::GpuComputePassDefinition kScenePreparePassDefinition = {
    "graphics/graphics_scene_prepare.cs.hlsl",
    "CRESSimNeo.Graphics.ScenePrepare",
    "CRESSimNeo.Graphics.ScenePrepare.PSO",
    kScenePrepareVars,
    std::size(kScenePrepareVars),
};

std::uint32_t dispatchGroupCount(std::uint32_t threadCount)
{
    return (threadCount + kScenePrepareThreadGroupSize - 1u) / kScenePrepareThreadGroupSize;
}

bool createStructuredSrvBuffer(Diligent::IRenderDevice *renderDevice, const char *name,
                               std::uint32_t elementStride, const void *initialData,
                               Diligent::Uint64 immediateContextMask,
                               Diligent::RefCntAutoPtr<Diligent::IBuffer> &outBuffer)
{
    if (renderDevice == nullptr || elementStride == 0u || initialData == nullptr)
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = name;
    desc.Size                 = elementStride;
    desc.BindFlags            = Diligent::BIND_SHADER_RESOURCE;
    desc.Usage                = Diligent::USAGE_DEFAULT;
    desc.CPUAccessFlags       = Diligent::CPU_ACCESS_NONE;
    desc.ImmediateContextMask = immediateContextMask;
    desc.Mode                 = Diligent::BUFFER_MODE_STRUCTURED;
    desc.ElementByteStride    = elementStride;

    Diligent::BufferData bufferData{};
    bufferData.pData    = initialData;
    bufferData.DataSize = elementStride;

    renderDevice->CreateBuffer(desc, &bufferData, &outBuffer);
    return outBuffer != nullptr;
}

std::uint32_t countActiveRenderables(const std::vector<RenderableInstance> &renderables)
{
    std::uint32_t count = 0u;
    for (const RenderableInstance &renderable : renderables)
    {
        if (renderable.entityId != common::kInvalidEntityId &&
            renderable.objectSlot != 0xffffffffu && renderable.visible)
        {
            ++count;
        }
    }
    return count;
}

std::uint32_t countActiveLights(const std::vector<LightData> &lights)
{
    std::uint32_t count = 0u;
    for (const LightData &light : lights)
    {
        if (light.entityId != common::kInvalidEntityId && light.lightSlot != 0xffffffffu)
        {
            ++count;
        }
    }
    return count;
}

bool hasActiveDeformableRenderables(const std::vector<GpuRenderableMetadata> *metadata)
{
    if (metadata == nullptr)
    {
        return false;
    }

    for (const GpuRenderableMetadata &entry : *metadata)
    {
        if ((entry.flags & static_cast<std::uint32_t>(GpuRenderableFlags::Active)) != 0u &&
            entry.deformableType != static_cast<std::uint32_t>(GpuRenderableDeformableType::None))
        {
            return true;
        }
    }
    return false;
}

bool isMainDirectionalLightActive(const LightData &light)
{
    if (light.entityId == common::kInvalidEntityId ||
        light.lightSlot != kMainDirectionalLightSlot || light.type != GpuLightType::Directional)
    {
        return false;
    }

    const float directionLengthSq = light.direction.x * light.direction.x +
                                    light.direction.y * light.direction.y +
                                    light.direction.z * light.direction.z;
    return directionLengthSq > 1.0e-6f && light.intensity > 0.0f;
}

std::vector<EnvMainLightState> buildEnvMainLightStates(const HostSceneView &sceneView,
                                                       const GpuEntitySceneView &gpuScene)
{
    std::vector<EnvMainLightState> states(gpuScene.layout.envCount);
    for (std::uint32_t envIndex = 0u; envIndex < gpuScene.layout.envCount; ++envIndex)
    {
        states[envIndex].mainLightIndex = mainDirectionalLightIndex(gpuScene.layout, envIndex);
    }

    if (sceneView.lights == nullptr)
    {
        return states;
    }

    for (const LightData &light : *sceneView.lights)
    {
        if (light.envIndex >= states.size() || light.lightSlot != kMainDirectionalLightSlot)
        {
            continue;
        }

        const bool active                   = isMainDirectionalLightActive(light);
        states[light.envIndex].active       = active;
        states[light.envIndex].castsShadows = active && light.castsShadows;
    }

    return states;
}

std::optional<DisplayResolveRequest> buildExplicitDisplayResolveRequest(
    const RenderFrameOptions &options)
{
    if (!options.presentationTarget.has_value() || !options.presentedExplicitOutput.has_value() ||
        !options.presentedExplicitOutput->isValid())
    {
        return std::nullopt;
    }

    DisplayResolveRequest resolveRequest{};
    resolveRequest.sourceKind             = options.presentedExplicitOutput->sourceKind;
    resolveRequest.sourceBinding          = options.presentedExplicitOutput->binding;
    resolveRequest.sourceTargetDesc       = options.presentedExplicitOutput->sourceTargetDesc;
    resolveRequest.sourceIsDisplayEncoded = options.presentedExplicitOutput->sourceIsDisplayEncoded;
    resolveRequest.presentationTarget     = *options.presentationTarget;
    resolveRequest.toneMapper             = options.toneMapper;
    resolveRequest.exposure               = options.exposure;
    resolveRequest.nearClip               = options.presentedExplicitOutput->nearClip;
    resolveRequest.farClip                = options.presentedExplicitOutput->farClip;
    resolveRequest.clearColor             = true;
    resolveRequest.clearDepth             = false;
    resolveRequest.preserveAspectRatio    = true;
    return resolveRequest;
}

} // namespace

struct Renderer::Impl
{
    struct GpuScenePrepareState
    {
        gpu::GpuComputePass cameraPreparePass;
        gpu::GpuComputePass scenePreparePass;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> cameraPrepareConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> scenePrepareConstantsBuffer;
        Diligent::RefCntAutoPtr<Diligent::IBuffer> fallbackSoftBodyWorldAabbsBuffer;
        bool initialized = false;
    };

    struct OutputPlanningState
    {
        std::unordered_map<detail::RenderTargetFamilyKey, gpu::GpuRenderTargetHandle,
                           detail::RenderTargetFamilyKeyHasher>
            managedPrimaryTargets;
    };

    Impl(gpu::GpuDevice &deviceIn, RenderResourceManager &resourceManagerIn,
         const RendererDesc &descIn)
        : mDevice(deviceIn), mResourceManager(resourceManagerIn), mDesc(descIn),
          mGpuScenePrepare(std::make_unique<GpuScenePrepareState>()),
          mOutputPlanningState(std::make_unique<OutputPlanningState>())
    {
    }

    bool ensureGpuScenePrepareState();
    bool prepareGpuScene(const HostSceneView &world, const GpuEntitySceneView &sceneView,
                         const physics::PhysicsGpuSceneView *physicsScene);

    gpu::GpuDevice &mDevice;
    RenderResourceManager &mResourceManager;
    RendererDesc mDesc{};
    std::unique_ptr<detail::ForwardPipeline> mForwardPipeline;
    std::unique_ptr<detail::DisplayResolvePass> mDisplayResolvePass;
    std::unique_ptr<GpuScenePrepareState> mGpuScenePrepare;
    std::unique_ptr<OutputPlanningState> mOutputPlanningState;
    bool mInitialized = false;
};

Renderer::Renderer(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                   const RendererDesc &desc)
    : mImpl(std::make_unique<Impl>(device, resourceManager, desc))
{
}

Renderer::~Renderer()
{
    if (mImpl->mOutputPlanningState != nullptr)
    {
        for (const auto &[key, target] : mImpl->mOutputPlanningState->managedPrimaryTargets)
        {
            (void)key;
            if (mImpl->mDevice.renderTargetSystem().isValidRenderTarget(target))
            {
                mImpl->mDevice.renderTargetSystem().destroyRenderTarget(target);
            }
        }
    }
}

bool Renderer::Impl::ensureGpuScenePrepareState()
{
    if (mGpuScenePrepare == nullptr)
    {
        mGpuScenePrepare = std::make_unique<GpuScenePrepareState>();
    }
    if (mGpuScenePrepare == nullptr)
    {
        return false;
    }
    if (mGpuScenePrepare->initialized)
    {
        return true;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }
    const Diligent::Uint64 graphicsContextMask = gpu::contextMaskForId(backendContext.contextId);

    gpu::ShaderSourceProvider shaderSourceProvider(mDevice.shaderSourceConfig());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderSourceProvider.streamFactory();
    if (streamFactory == nullptr)
    {
        return false;
    }

    if (!mGpuScenePrepare->cameraPreparePass.initialize(mDevice, streamFactory, graphicsContextMask,
                                                        kCameraPreparePassDefinition) ||
        !mGpuScenePrepare->scenePreparePass.initialize(mDevice, streamFactory, graphicsContextMask,
                                                       kScenePreparePassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = "CRESSimNeo.Graphics.CameraPrepare.Constants";
    desc.Size                 = sizeof(GraphicsCameraPrepareConstants);
    desc.Usage                = Diligent::USAGE_DYNAMIC;
    desc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    desc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    desc.ImmediateContextMask = graphicsContextMask;
    backendContext.renderDevice->CreateBuffer(desc, nullptr,
                                              &mGpuScenePrepare->cameraPrepareConstantsBuffer);
    if (mGpuScenePrepare->cameraPrepareConstantsBuffer == nullptr)
    {
        return false;
    }

    desc.Name = "CRESSimNeo.Graphics.ScenePrepare.Constants";
    desc.Size = sizeof(GraphicsScenePrepareConstants);
    backendContext.renderDevice->CreateBuffer(desc, nullptr,
                                              &mGpuScenePrepare->scenePrepareConstantsBuffer);
    if (mGpuScenePrepare->scenePrepareConstantsBuffer == nullptr)
    {
        return false;
    }

    const SoftBodyWorldAabbFallback fallbackSoftBodyWorldAabb{};
    if (!createStructuredSrvBuffer(backendContext.renderDevice,
                                   "CRESSimNeo.Graphics.ScenePrepare.FallbackSoftBodyWorldAabbs",
                                   sizeof(SoftBodyWorldAabbFallback), &fallbackSoftBodyWorldAabb,
                                   graphicsContextMask,
                                   mGpuScenePrepare->fallbackSoftBodyWorldAabbsBuffer))
    {
        return false;
    }

    mGpuScenePrepare->initialized = true;
    return true;
}

bool Renderer::Impl::prepareGpuScene(const HostSceneView &world,
                                     const GpuEntitySceneView &sceneView,
                                     const physics::PhysicsGpuSceneView *physicsScene)
{
    if (sceneView.renderableCount == 0u || sceneView.cameraCount == 0u)
    {
        return true;
    }
    if (sceneView.poses.positionsBuffer == nullptr ||
        sceneView.poses.orientationsBuffer == nullptr || sceneView.poses.scalesBuffer == nullptr ||
        sceneView.renderableMetadataBuffer == nullptr || sceneView.cameraInputsBuffer == nullptr ||
        sceneView.preparedCamerasBuffer == nullptr || sceneView.lightInputsBuffer == nullptr ||
        sceneView.localLightSelectionBuffer == nullptr ||
        sceneView.renderableVisibilityFlagsBuffer == nullptr ||
        sceneView.renderableShadowCascadeMasksBuffer == nullptr)
    {
        return false;
    }
    if (!ensureGpuScenePrepareState())
    {
        return false;
    }
    const bool needsSoftBodyWorldAabbs = hasActiveDeformableRenderables(world.renderableMetadata);
    if (needsSoftBodyWorldAabbs &&
        (physicsScene == nullptr || physicsScene->soft.worldAabbsBuffer == nullptr ||
         physicsScene->curve.worldAabbsBuffer == nullptr))
    {
        return false;
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.graphicsContext == nullptr)
    {
        return false;
    }

    GraphicsCameraPrepareConstants cameraPrepareConstants{};
    cameraPrepareConstants.cameraCount         = sceneView.cameraCount;
    cameraPrepareConstants.maxObjectsPerEnv    = sceneView.layout.maxRenderableObjectsPerEnv;
    cameraPrepareConstants.maxLightsPerEnv     = sceneView.layout.maxLightsPerEnv;
    cameraPrepareConstants.shadowMapResolution = kShadowMapResolution;
    cameraPrepareConstants.entityPoseCount     = sceneView.entityCount;

    void *mappedConstants = nullptr;
    backendContext.graphicsContext->MapBuffer(mGpuScenePrepare->cameraPrepareConstantsBuffer,
                                              Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                              mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &cameraPrepareConstants, sizeof(cameraPrepareConstants));
    backendContext.graphicsContext->UnmapBuffer(mGpuScenePrepare->cameraPrepareConstantsBuffer,
                                                Diligent::MAP_WRITE);

    const std::array cameraPrepareBindings{
        gpu::GpuBufferBinding{"GraphicsCameraPrepareConstants",
                              mGpuScenePrepare->cameraPrepareConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CameraInputs", sceneView.cameraInputsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityPositions", sceneView.poses.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityOrientations", sceneView.poses.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_LightInputs", sceneView.lightInputsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreparedCamerasRW", sceneView.preparedCamerasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!mGpuScenePrepare->cameraPreparePass.dispatch(backendContext.graphicsContext, 0u,
                                                      cameraPrepareBindings,
                                                      dispatchGroupCount(sceneView.cameraCount)))
    {
        return false;
    }

    GraphicsScenePrepareConstants constants{};
    constants.cameraCount      = sceneView.cameraCount;
    constants.maxObjectsPerEnv = sceneView.layout.maxRenderableObjectsPerEnv;
    mappedConstants            = nullptr;
    backendContext.graphicsContext->MapBuffer(mGpuScenePrepare->scenePrepareConstantsBuffer,
                                              Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                              mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &constants, sizeof(constants));
    backendContext.graphicsContext->UnmapBuffer(mGpuScenePrepare->scenePrepareConstantsBuffer,
                                                Diligent::MAP_WRITE);
    Diligent::IBuffer *softBodyWorldAabbsBuffer =
        mGpuScenePrepare->fallbackSoftBodyWorldAabbsBuffer;
    Diligent::IBuffer *curveWorldAabbsBuffer = mGpuScenePrepare->fallbackSoftBodyWorldAabbsBuffer;
    if (needsSoftBodyWorldAabbs)
    {
        softBodyWorldAabbsBuffer = physicsScene->soft.worldAabbsBuffer;
        curveWorldAabbsBuffer    = physicsScene->curve.worldAabbsBuffer;
    }

    const std::array bindings{
        gpu::GpuBufferBinding{"GraphicsScenePrepareConstants",
                              mGpuScenePrepare->scenePrepareConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityPositions", sceneView.poses.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityOrientations", sceneView.poses.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityScales", sceneView.poses.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RenderableMetadata", sceneView.renderableMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_SoftBodyWorldAabbs", softBodyWorldAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CurveWorldAabbs", curveWorldAabbsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreparedCameras", sceneView.preparedCamerasBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RenderableVisibilityFlagsRW",
                              sceneView.renderableVisibilityFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RenderableShadowCascadeMasksRW",
                              sceneView.renderableShadowCascadeMasksBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGpuScenePrepare->scenePreparePass.dispatch(
        backendContext.graphicsContext, 0u, bindings,
        dispatchGroupCount(sceneView.layout.maxRenderableObjectsPerEnv * sceneView.cameraCount));
}

bool Renderer::initialize()
{
    gpu::GpuGraphicsBackendContext backendContext{};
    const bool hasGraphicsBackend = mImpl->mDevice.tryGetGraphicsBackendContext(backendContext) &&
                                    backendContext.renderDevice != nullptr &&
                                    backendContext.graphicsContext != nullptr;
    if (!hasGraphicsBackend)
    {
        mImpl->mInitialized = true;
        return true;
    }

    mImpl->mForwardPipeline = std::make_unique<detail::ForwardPipeline>(
        mImpl->mDevice, mImpl->mResourceManager, mImpl->mDesc.iblQualityTier);
    if (!mImpl->mForwardPipeline->initialize())
    {
        mImpl->mForwardPipeline.reset();
        return false;
    }

    mImpl->mDisplayResolvePass = std::make_unique<detail::DisplayResolvePass>(mImpl->mDevice);
    if (!mImpl->mDisplayResolvePass->initialize())
    {
        mImpl->mDisplayResolvePass.reset();
        mImpl->mForwardPipeline.reset();
        return false;
    }

    if (!mImpl->ensureGpuScenePrepareState())
    {
        mImpl->mDisplayResolvePass.reset();
        mImpl->mForwardPipeline.reset();
        return false;
    }

    mImpl->mInitialized = true;
    return mImpl->mInitialized;
}

RenderStats Renderer::render(const common::FrameContext &frameContext, const HostSceneView &world,
                             const cressim::neo::physics::PhysicsGpuSceneView *physicsScene,
                             const RenderFrameOptions &options)
{
    RenderStats stats{};

    if (!mImpl->mInitialized)
    {
        return stats;
    }

    const std::vector<RenderableInstance> &renderables =
        world.renderables != nullptr ? *world.renderables : std::vector<RenderableInstance>{};
    const std::vector<LightData> emptyLights;
    const std::vector<LightData> &directionalLights =
        world.lights != nullptr ? *world.lights : emptyLights;
    const GpuEntitySceneView emptySceneView{};
    const GpuEntitySceneView &gpuScene =
        world.gpuEntityScene != nullptr ? *world.gpuEntityScene : emptySceneView;

    stats.renderableCount = countActiveRenderables(renderables);
    stats.lightCount      = countActiveLights(directionalLights);

    std::vector<CameraData> cameras = detail::sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(detail::defaultCamera());
    }

    gpu::GpuGraphicsBackendContext backendContext{};
    const bool hasGraphicsBackend = mImpl->mDevice.tryGetGraphicsBackendContext(backendContext) &&
                                    backendContext.renderDevice != nullptr &&
                                    backendContext.graphicsContext != nullptr;
    if (!hasGraphicsBackend)
    {
        return stats;
    }

    if (!mImpl->prepareGpuScene(world, gpuScene, physicsScene))
    {
        return stats;
    }

    if (mImpl->mOutputPlanningState == nullptr)
    {
        mImpl->mOutputPlanningState = std::make_unique<Impl::OutputPlanningState>();
    }

    gpu::GpuRenderTargetDesc defaultRenderTargetDesc{};
    if (!mImpl->mDevice.tryGetDefaultRenderTargetDesc(defaultRenderTargetDesc))
    {
        return stats;
    }

    stats.renderedCameraCount = 0u;
    detail::CameraOutputPlanningResult outputPlan =
        detail::planCameraOutputs(cameras, gpuScene, mImpl->mDevice.renderTargetSystem(),
                                  defaultRenderTargetDesc, options.presentationTarget, options,
                                  mImpl->mOutputPlanningState->managedPrimaryTargets, stats);

    if (const std::optional<DisplayResolveRequest> explicitResolve =
            buildExplicitDisplayResolveRequest(options);
        explicitResolve.has_value())
    {
        outputPlan.displayResolve = *explicitResolve;
    }

    for (auto it = mImpl->mOutputPlanningState->managedPrimaryTargets.begin();
         it != mImpl->mOutputPlanningState->managedPrimaryTargets.end();)
    {
        if (outputPlan.usedManagedFamilies.find(it->first) != outputPlan.usedManagedFamilies.end())
        {
            ++it;
            continue;
        }

        if (mImpl->mDevice.renderTargetSystem().isValidRenderTarget(it->second))
        {
            mImpl->mDevice.renderTargetSystem().destroyRenderTarget(it->second);
        }
        it = mImpl->mOutputPlanningState->managedPrimaryTargets.erase(it);
    }

    FrameRenderPlan renderPlan = detail::buildFrameRenderPlan(std::move(outputPlan.resolvedCameras),
                                                              outputPlan.displayResolve);
    renderPlan.envMainLights   = buildEnvMainLightStates(world, gpuScene);

    for (const CameraBatchView &batch : renderPlan.cameraBatches)
    {
        ForwardPassExecutionStats passStats{};
        if (mImpl->mForwardPipeline != nullptr)
        {
            (void)mImpl->mForwardPipeline->executeBatch(frameContext, batch, world, physicsScene,
                                                        options, renderPlan.envMainLights,
                                                        passStats);
        }
        stats.opaqueDrawCalls += passStats.opaqueDrawCalls;
        stats.transparentDrawCalls += passStats.transparentDrawCalls;
        stats.shadowDrawCalls += passStats.shadowDrawCalls;
    }

    if (renderPlan.displayResolve.has_value() && mImpl->mDisplayResolvePass != nullptr)
    {
        (void)mImpl->mDisplayResolvePass->resolve(frameContext, *renderPlan.displayResolve);
    }

    stats.drawCalls = stats.opaqueDrawCalls + stats.transparentDrawCalls + stats.shadowDrawCalls;
    return stats;
}

} // namespace cressim::neo::graphics
