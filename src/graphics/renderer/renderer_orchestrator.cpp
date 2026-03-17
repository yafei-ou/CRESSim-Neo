#include "graphics/renderer.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/renderer_internal.h"

#include <array>
#include <cstring>
#include <unordered_map>

namespace cressim::neo::graphics
{

namespace
{

constexpr std::uint32_t kScenePrepareThreadGroupSize = 64u;

struct GraphicsCameraPrepareConstants
{
    std::uint32_t currentCameraIndex  = 0u;
    std::uint32_t maxLightsPerEnv     = 0u;
    std::uint32_t shadowMapResolution = kShadowMapResolution;
    float frameAspectRatio            = 1.0f;
};

struct GraphicsScenePrepareConstants
{
    std::uint32_t currentCameraIndex = 0u;
    std::uint32_t renderableCount    = 0u;
    std::uint32_t padding0           = 0u;
    std::uint32_t padding1           = 0u;
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

std::uint32_t countActiveRenderables(const std::vector<RenderableInstance>& renderables)
{
    std::uint32_t count = 0u;
    for (const RenderableInstance& renderable : renderables)
    {
        if (renderable.entityId != common::kInvalidEntityId && renderable.objectSlot != 0xffffffffu &&
            renderable.visible)
        {
            ++count;
        }
    }
    return count;
}

std::uint32_t countActiveLights(const std::vector<DirectionalLightData>& lights)
{
    std::uint32_t count = 0u;
    for (const DirectionalLightData& light : lights)
    {
        if (light.entityId != common::kInvalidEntityId && light.lightSlot != 0xffffffffu)
        {
            ++count;
        }
    }
    return count;
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

Renderer::Renderer(gpu::GpuDevice& device, RenderResourceManager& resourceManager,
                   const RendererDesc& desc)
    : mDevice(device), mResourceManager(resourceManager), mDesc(desc),
      mGpuScenePrepare(std::make_unique<GpuScenePrepareState>())
{
}

Renderer::~Renderer() = default;

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
    Diligent::IShaderSourceInputStreamFactory* streamFactory = shaderLibrary.streamFactory();
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

bool Renderer::prepareGpuScene(const FrameViewData& frameView,
                               const gpu::GpuEntitySceneView& sceneView)
{
    if (sceneView.renderableCount == 0u)
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

    const std::uint32_t currentCameraIndex =
        frameView.envIndex * sceneView.layout.maxCamerasPerEnv + frameView.cameraSlot;
    if (currentCameraIndex >= sceneView.cameraCount)
    {
        return true;
    }

    GraphicsCameraPrepareConstants cameraPrepareConstants{};
    cameraPrepareConstants.currentCameraIndex  = currentCameraIndex;
    cameraPrepareConstants.maxLightsPerEnv     = sceneView.layout.maxLightsPerEnv;
    cameraPrepareConstants.shadowMapResolution = kShadowMapResolution;
    cameraPrepareConstants.frameAspectRatio =
        frameView.outputHeight > 0u ? static_cast<float>(frameView.outputWidth) /
                                          static_cast<float>(std::max(frameView.outputHeight, 1u))
                                    : 1.0f;

    void* mappedConstants = nullptr;
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
                                                      cameraPrepareBindings, 1u))
    {
        return false;
    }

    GraphicsScenePrepareConstants constants{};
    constants.currentCameraIndex = currentCameraIndex;
    constants.renderableCount    = sceneView.renderableCount;
    mappedConstants              = nullptr;
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
        dispatchGroupCount(sceneView.renderableCount));
}

bool Renderer::initialize()
{
    mForwardPipeline = std::make_unique<detail::ForwardPipeline>(mDevice);
    if (!mForwardPipeline->initialize())
    {
        mForwardPipeline.reset();
        return false;
    }

    if (!ensureGpuScenePrepareState())
    {
        mForwardPipeline.reset();
        return false;
    }

    mInitialized = true;
    return mInitialized;
}

RenderStats Renderer::render(const common::FrameContext& frameContext, const HostSceneView& world)
{
    RenderStats stats{};

    if (!mInitialized)
    {
        return stats;
    }

    mDevice.beginFrame(frameContext);

    const std::vector<RenderableInstance>& renderables =
        world.renderables != nullptr ? *world.renderables : std::vector<RenderableInstance>{};
    const std::unordered_map<common::EntityId, std::uint32_t> emptyPoseIndices;
    const std::unordered_map<common::EntityId, std::uint32_t>& poseIndices =
        world.gpuEntityPoseIndices != nullptr ? *world.gpuEntityPoseIndices : emptyPoseIndices;
    const std::vector<DirectionalLightData> emptyLights;
    const std::vector<DirectionalLightData>& directionalLights =
        world.directionalLights != nullptr ? *world.directionalLights : emptyLights;
    const auto preparedRenderables =
        detail::buildPreparedRenderables(renderables, mResourceManager, poseIndices);
    const ForwardDirectionalLightData lightData = detail::buildMainLight(directionalLights);

    stats.renderableCount      = countActiveRenderables(renderables);
    stats.validRenderableCount = static_cast<std::uint32_t>(preparedRenderables.size());
    stats.lightCount           = countActiveLights(directionalLights);

    std::vector<CameraData> cameras = detail::sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(detail::defaultCamera());
    }

    struct RequestedExtent
    {
        std::uint32_t width  = 0;
        std::uint32_t height = 0;
    };
    std::unordered_map<common::ResourceId, RequestedExtent> requestedExtents;

    const auto renderCamera = [&](const CameraData& camera)
    {
        gpu::GpuRenderTargetHandle target = camera.outputTarget;
        if (!mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            target = mDevice.renderTargetSystem().defaultRenderTarget();
        }
        if (!mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            return;
        }

        gpu::GpuRenderTargetDesc targetDesc{};
        if (!mDevice.renderTargetSystem().tryGetRenderTargetDesc(target, targetDesc))
        {
            return;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            ++stats.renderTargetResizeRequests;

            RequestedExtent desired{};
            desired.width  = (camera.outputWidth == 0 ? targetDesc.width : camera.outputWidth);
            desired.height = (camera.outputHeight == 0 ? targetDesc.height : camera.outputHeight);

            const auto requestedIt = requestedExtents.find(target.id);
            if (requestedIt == requestedExtents.end())
            {
                requestedExtents.emplace(target.id, desired);
            }
            else
            {
                const bool conflict = requestedIt->second.width != desired.width ||
                                      requestedIt->second.height != desired.height;
                if (conflict)
                {
                    ++stats.renderTargetResizeConflicts;
                }
                desired = requestedIt->second;
            }

            if (targetDesc.width != desired.width || targetDesc.height != desired.height)
            {
                const gpu::GpuRenderTargetUpdateResult updateResult =
                    mDevice.renderTargetSystem().resizeRenderTarget(target, desired.width,
                                                                    desired.height);
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Unchanged)
                {
                    ++stats.renderTargetResizeNoOps;
                }
                else if (updateResult == gpu::GpuRenderTargetUpdateResult::Recreated)
                {
                    ++stats.renderTargetRecreateCount;
                }
                if (updateResult != gpu::GpuRenderTargetUpdateResult::Failed)
                {
                    if (!mDevice.renderTargetSystem().tryGetRenderTargetDesc(target, targetDesc))
                    {
                        return;
                    }
                }
            }
            else
            {
                ++stats.renderTargetResizeNoOps;
            }
        }

        const gpu::GpuRenderViewport viewport = detail::normalizeViewport(camera.viewport);
        const FrameViewData frameView =
            detail::buildFrameViewData(camera, targetDesc, target, viewport, lightData);
        const CameraRenderQueues queues = detail::buildCameraRenderQueues(
            preparedRenderables, frameView, mResourceManager, stats);

        const gpu::GpuEntitySceneView emptySceneView{};
        const gpu::GpuEntitySceneView& gpuScene =
            world.gpuEntityScene != nullptr ? *world.gpuEntityScene : emptySceneView;

        if (!prepareGpuScene(frameView, gpuScene))
        {
            return;
        }

        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->execute(frameContext, frameView, gpuScene, queues, passStats);
        }

        stats.opaqueDrawCalls += passStats.opaqueDrawCalls;
        stats.shadowDrawCalls += passStats.shadowDrawCalls;
        stats.transparentDrawCalls += passStats.transparentDrawCalls;
        ++stats.cameraCount;
    };

    for (const CameraData& camera : cameras)
    {
        renderCamera(camera);
    }

    stats.drawCalls = stats.opaqueDrawCalls + stats.shadowDrawCalls + stats.transparentDrawCalls;

    mDevice.endFrame(frameContext);
    return stats;
}

} // namespace cressim::neo::graphics
