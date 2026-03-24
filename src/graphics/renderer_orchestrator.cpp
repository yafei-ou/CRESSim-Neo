#include "graphics/renderer.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/display_resolve_pass.h"
#include "graphics/output_planner.h"
#include "graphics/passes/forward_pipeline.h"
#include "graphics/render_plan_builder.h"
#include "graphics/renderer_internal.h"

#include <array>
#include <cstring>

namespace cressim::neo::graphics
{

namespace
{

constexpr std::uint32_t kScenePrepareThreadGroupSize = 64u;

struct GraphicsCameraPrepareConstants
{
    std::uint32_t cameraCount         = 0u;
    std::uint32_t maxObjectsPerEnv    = 0u;
    std::uint32_t maxLightsPerEnv     = 0u;
    std::uint32_t shadowMapResolution = kShadowMapResolution;
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

std::uint32_t countActiveLights(const std::vector<DirectionalLightData> &lights)
{
    std::uint32_t count = 0u;
    for (const DirectionalLightData &light : lights)
    {
        if (light.entityId != common::kInvalidEntityId && light.lightSlot != 0xffffffffu)
        {
            ++count;
        }
    }
    return count;
}

bool isMainDirectionalLightActive(const DirectionalLightData &light)
{
    if (light.entityId == common::kInvalidEntityId ||
        light.lightSlot != gpu::kMainDirectionalLightSlot)
    {
        return false;
    }

    const float directionLengthSq = light.direction.x * light.direction.x +
                                    light.direction.y * light.direction.y +
                                    light.direction.z * light.direction.z;
    return directionLengthSq > 1.0e-6f && light.intensity > 0.0f;
}

std::vector<EnvMainLightState> buildEnvMainLightStates(const HostSceneView &sceneView,
                                                       const gpu::GpuEntitySceneView &gpuScene)
{
    std::vector<EnvMainLightState> states(gpuScene.layout.envCount);
    for (std::uint32_t envIndex = 0u; envIndex < gpuScene.layout.envCount; ++envIndex)
    {
        states[envIndex].mainLightIndex = gpu::mainDirectionalLightIndex(gpuScene.layout, envIndex);
    }

    if (sceneView.directionalLights == nullptr)
    {
        return states;
    }

    for (const DirectionalLightData &light : *sceneView.directionalLights)
    {
        if (light.envIndex >= states.size() || light.lightSlot != gpu::kMainDirectionalLightSlot)
        {
            continue;
        }

        const bool active                   = isMainDirectionalLightActive(light);
        states[light.envIndex].active       = active;
        states[light.envIndex].castsShadows = active && light.castsShadows;
    }

    return states;
}

} // namespace

struct Renderer::GpuScenePrepareState
{
    gpu::GpuComputePass cameraPreparePass;
    gpu::GpuComputePass scenePreparePass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> cameraPrepareConstantsBuffer;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> scenePrepareConstantsBuffer;
    bool initialized = false;
};

struct RendererOutputPlanningState
{
    std::unordered_map<detail::RenderTargetFamilyKey, gpu::GpuRenderTargetHandle,
                       detail::RenderTargetFamilyKeyHasher>
        managedPrimaryTargets;
};

Renderer::Renderer(gpu::GpuDevice &device, RenderResourceManager &resourceManager,
                   const RendererDesc &desc)
    : mDevice(device), mResourceManager(resourceManager), mDesc(desc),
      mGpuScenePrepare(std::make_unique<GpuScenePrepareState>()),
      mOutputPlanningState(std::make_unique<RendererOutputPlanningState>())
{
}

Renderer::~Renderer()
{
    if (mOutputPlanningState != nullptr)
    {
        for (const auto &[key, target] : mOutputPlanningState->managedPrimaryTargets)
        {
            (void)key;
            if (mDevice.renderTargetSystem().isValidRenderTarget(target))
            {
                mDevice.renderTargetSystem().destroyRenderTarget(target);
            }
        }
    }
}

bool Renderer::ensureGpuScenePrepareState()
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

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.renderDevice == nullptr)
    {
        return false;
    }

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory *streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return false;
    }

    if (!mGpuScenePrepare->cameraPreparePass.initialize(backendContext.renderDevice, streamFactory,
                                                        1ull, kCameraPreparePassDefinition) ||
        !mGpuScenePrepare->scenePreparePass.initialize(backendContext.renderDevice, streamFactory,
                                                       1ull, kScenePreparePassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name                 = "CRESSimNeo.Graphics.CameraPrepare.Constants";
    desc.Size                 = sizeof(GraphicsCameraPrepareConstants);
    desc.Usage                = Diligent::USAGE_DYNAMIC;
    desc.BindFlags            = Diligent::BIND_UNIFORM_BUFFER;
    desc.CPUAccessFlags       = Diligent::CPU_ACCESS_WRITE;
    desc.ImmediateContextMask = 1ull;
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

    mGpuScenePrepare->initialized = true;
    return true;
}

bool Renderer::prepareGpuScene(const gpu::GpuEntitySceneView &sceneView)
{
    if (sceneView.renderableCount == 0u || sceneView.cameraCount == 0u)
    {
        return true;
    }
    if (sceneView.poses.positionsBuffer == nullptr ||
        sceneView.poses.orientationsBuffer == nullptr || sceneView.poses.scalesBuffer == nullptr ||
        sceneView.renderableMetadataBuffer == nullptr || sceneView.cameraInputsBuffer == nullptr ||
        sceneView.preparedCamerasBuffer == nullptr || sceneView.lightInputsBuffer == nullptr ||
        sceneView.renderableVisibilityFlagsBuffer == nullptr ||
        sceneView.renderableShadowCascadeMasksBuffer == nullptr)
    {
        return false;
    }
    if (!ensureGpuScenePrepareState())
    {
        return false;
    }

    gpu::GpuBackendContext backendContext{};
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) ||
        backendContext.immediateContext == nullptr)
    {
        return false;
    }

    GraphicsCameraPrepareConstants cameraPrepareConstants{};
    cameraPrepareConstants.cameraCount         = sceneView.cameraCount;
    cameraPrepareConstants.maxObjectsPerEnv    = sceneView.layout.maxObjectsPerEnv;
    cameraPrepareConstants.maxLightsPerEnv     = sceneView.layout.maxLightsPerEnv;
    cameraPrepareConstants.shadowMapResolution = kShadowMapResolution;

    void *mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mGpuScenePrepare->cameraPrepareConstantsBuffer,
                                               Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                               mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &cameraPrepareConstants, sizeof(cameraPrepareConstants));
    backendContext.immediateContext->UnmapBuffer(mGpuScenePrepare->cameraPrepareConstantsBuffer,
                                                 Diligent::MAP_WRITE);

    const std::array cameraPrepareBindings{
        gpu::GpuBufferBinding{"GraphicsCameraPrepareConstants",
                              mGpuScenePrepare->cameraPrepareConstantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_CameraInputs", sceneView.cameraInputsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_LightInputs", sceneView.lightInputsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_PreparedCamerasRW", sceneView.preparedCamerasBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };
    if (!mGpuScenePrepare->cameraPreparePass.dispatch(backendContext.immediateContext, 0u,
                                                      cameraPrepareBindings,
                                                      dispatchGroupCount(sceneView.cameraCount)))
    {
        return false;
    }

    GraphicsScenePrepareConstants constants{};
    constants.cameraCount      = sceneView.cameraCount;
    constants.maxObjectsPerEnv = sceneView.layout.maxObjectsPerEnv;
    mappedConstants            = nullptr;
    backendContext.immediateContext->MapBuffer(mGpuScenePrepare->scenePrepareConstantsBuffer,
                                               Diligent::MAP_WRITE, Diligent::MAP_FLAG_DISCARD,
                                               mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &constants, sizeof(constants));
    backendContext.immediateContext->UnmapBuffer(mGpuScenePrepare->scenePrepareConstantsBuffer,
                                                 Diligent::MAP_WRITE);

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
        backendContext.immediateContext, 0u, bindings,
        dispatchGroupCount(sceneView.layout.maxObjectsPerEnv * sceneView.cameraCount));
}

bool Renderer::initialize()
{
    mForwardPipeline = std::make_unique<detail::ForwardPipeline>(mDevice, mResourceManager);
    if (!mForwardPipeline->initialize())
    {
        mForwardPipeline.reset();
        return false;
    }

    mDisplayResolvePass = std::make_unique<detail::DisplayResolvePass>(mDevice);
    if (!mDisplayResolvePass->initialize())
    {
        mDisplayResolvePass.reset();
        mForwardPipeline.reset();
        return false;
    }

    if (!ensureGpuScenePrepareState())
    {
        mDisplayResolvePass.reset();
        mForwardPipeline.reset();
        return false;
    }

    mInitialized = true;
    return mInitialized;
}

RenderStats Renderer::render(const common::FrameContext &frameContext, const HostSceneView &world,
                             const RenderFrameOptions &options)
{
    RenderStats stats{};

    if (!mInitialized)
    {
        return stats;
    }

    mDevice.beginFrame(frameContext);

    const std::vector<RenderableInstance> &renderables =
        world.renderables != nullptr ? *world.renderables : std::vector<RenderableInstance>{};
    const std::vector<DirectionalLightData> emptyLights;
    const std::vector<DirectionalLightData> &directionalLights =
        world.directionalLights != nullptr ? *world.directionalLights : emptyLights;
    const gpu::GpuEntitySceneView emptySceneView{};
    const gpu::GpuEntitySceneView &gpuScene =
        world.gpuEntityScene != nullptr ? *world.gpuEntityScene : emptySceneView;

    stats.renderableCount = countActiveRenderables(renderables);
    stats.lightCount      = countActiveLights(directionalLights);

    std::vector<CameraData> cameras = detail::sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(detail::defaultCamera());
    }

    if (!prepareGpuScene(gpuScene))
    {
        mDevice.endFrame(frameContext);
        return stats;
    }

    if (mOutputPlanningState == nullptr)
    {
        mOutputPlanningState = std::make_unique<RendererOutputPlanningState>();
    }

    gpu::GpuRenderTargetDesc defaultRenderTargetDesc{};
    if (!mDevice.tryGetDefaultRenderTargetDesc(defaultRenderTargetDesc))
    {
        mDevice.endFrame(frameContext);
        return stats;
    }

    detail::CameraOutputPlanningResult outputPlan = detail::planCameraOutputs(
        cameras, gpuScene, mDevice.renderTargetSystem(), defaultRenderTargetDesc,
        options.presentationTarget, options, mOutputPlanningState->managedPrimaryTargets, stats);

    for (auto it = mOutputPlanningState->managedPrimaryTargets.begin();
         it != mOutputPlanningState->managedPrimaryTargets.end();)
    {
        if (outputPlan.usedManagedFamilies.find(it->first) != outputPlan.usedManagedFamilies.end())
        {
            ++it;
            continue;
        }

        if (mDevice.renderTargetSystem().isValidRenderTarget(it->second))
        {
            mDevice.renderTargetSystem().destroyRenderTarget(it->second);
        }
        it = mOutputPlanningState->managedPrimaryTargets.erase(it);
    }

    FrameRenderPlan renderPlan = detail::buildFrameRenderPlan(std::move(outputPlan.resolvedCameras),
                                                              outputPlan.displayResolve);
    renderPlan.envMainLights   = buildEnvMainLightStates(world, gpuScene);

    for (const CameraBatchView &batch : renderPlan.cameraBatches)
    {
        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->executeBatch(frameContext, batch, world,
                                                 renderPlan.envMainLights, passStats);
        }
        stats.opaqueDrawCalls += passStats.opaqueDrawCalls;
        stats.shadowDrawCalls += passStats.shadowDrawCalls;
    }

    if (renderPlan.displayResolve.has_value() && mDisplayResolvePass != nullptr)
    {
        (void)mDisplayResolvePass->resolve(frameContext, *renderPlan.displayResolve);
    }

    stats.drawCalls = stats.opaqueDrawCalls + stats.shadowDrawCalls;

    mDevice.endFrame(frameContext);
    return stats;
}

} // namespace cressim::neo::graphics
