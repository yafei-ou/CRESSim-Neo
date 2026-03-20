#include "graphics/renderer.h"

#include "gpu/gpu_compute_pass.h"
#include "gpu/shader_library.h"
#include "graphics/renderer/display_resolve_pass.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/render_plan_builder.h"
#include "graphics/renderer/renderer_internal.h"

#include <array>
#include <cstring>
#include <string>
#include <unordered_map>

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

std::uint32_t countActiveRenderables(const std::vector<RenderableInstance>& renderables)
{
    std::uint32_t count = 0u;
    for (const RenderableInstance& renderable : renderables)
    {
        if (renderable.entityId != common::kInvalidEntityId &&
            renderable.objectSlot != 0xffffffffu && renderable.visible)
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

struct ManagedTargetKey
{
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t arraySize = 1u;
    bool color = true;
    bool depth = true;
    bool shaderReadable = true;
    bool layeredRendering = true;
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_UNKNOWN;
    Diligent::TEXTURE_FORMAT depthFormat = Diligent::TEX_FORMAT_UNKNOWN;
    std::string debugName{};

    bool operator==(const ManagedTargetKey& rhs) const noexcept
    {
        return width == rhs.width && height == rhs.height && arraySize == rhs.arraySize &&
               color == rhs.color && depth == rhs.depth &&
               shaderReadable == rhs.shaderReadable &&
               layeredRendering == rhs.layeredRendering &&
               colorFormat == rhs.colorFormat && depthFormat == rhs.depthFormat &&
               debugName == rhs.debugName;
    }
};

struct ManagedTargetKeyHasher
{
    std::size_t operator()(const ManagedTargetKey& key) const noexcept
    {
        std::size_t seed = 0u;
        auto hashCombine = [&](std::size_t value)
        { seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u); };
        hashCombine(std::hash<std::uint32_t>{}(key.width));
        hashCombine(std::hash<std::uint32_t>{}(key.height));
        hashCombine(std::hash<std::uint32_t>{}(key.arraySize));
        hashCombine(std::hash<bool>{}(key.color));
        hashCombine(std::hash<bool>{}(key.depth));
        hashCombine(std::hash<bool>{}(key.shaderReadable));
        hashCombine(std::hash<bool>{}(key.layeredRendering));
        hashCombine(std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.colorFormat)));
        hashCombine(std::hash<std::uint32_t>{}(static_cast<std::uint32_t>(key.depthFormat)));
        hashCombine(std::hash<std::string>{}(key.debugName));
        return seed;
    }
};

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
    std::unordered_map<ManagedTargetKey, gpu::GpuRenderTargetHandle, ManagedTargetKeyHasher>
        managedPrimaryTargets;
};

Renderer::Renderer(gpu::GpuDevice& device, RenderResourceManager& resourceManager,
                   const RendererDesc& desc)
    : mDevice(device), mResourceManager(resourceManager), mDesc(desc),
      mGpuScenePrepare(std::make_unique<GpuScenePrepareState>()),
      mOutputPlanningState(std::make_unique<RendererOutputPlanningState>())
{
}

Renderer::~Renderer()
{
    if (mOutputPlanningState != nullptr)
    {
        for (const auto& [key, target] : mOutputPlanningState->managedPrimaryTargets)
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

bool Renderer::prepareGpuScene(const gpu::GpuEntitySceneView& sceneView)
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
                                                      cameraPrepareBindings, sceneView.cameraCount))
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
    const std::vector<DirectionalLightData> emptyLights;
    const std::vector<DirectionalLightData>& directionalLights =
        world.directionalLights != nullptr ? *world.directionalLights : emptyLights;

    // TODO: we are only using one global light
    const ForwardDirectionalLightData lightData = detail::buildMainLight(directionalLights);
    const gpu::GpuEntitySceneView emptySceneView{};
    const gpu::GpuEntitySceneView& gpuScene =
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

    struct RequestedExtent
    {
        std::uint32_t width  = 0;
        std::uint32_t height = 0;
    };
    std::unordered_map<common::ResourceId, RequestedExtent> requestedExtents;
    gpu::GpuRenderTargetDesc defaultTargetDesc{};
    const gpu::GpuRenderTargetHandle defaultTarget = mDevice.renderTargetSystem().defaultRenderTarget();
    const bool hasDefaultTarget =
        mDevice.renderTargetSystem().isValidRenderTarget(defaultTarget) &&
        mDevice.renderTargetSystem().tryGetRenderTargetDesc(defaultTarget, defaultTargetDesc);

    if (mOutputPlanningState == nullptr)
    {
        mOutputPlanningState = std::make_unique<RendererOutputPlanningState>();
    }

    const auto acquireManagedPrimaryTarget =
        [&](const gpu::GpuRenderTargetDesc& desc) -> gpu::GpuRenderTargetHandle
    {
        if (mOutputPlanningState == nullptr)
        {
            return {};
        }

        const ManagedTargetKey key{desc.width,          desc.height,         desc.arraySize,
                                   desc.color,          desc.depth,          desc.shaderReadable,
                                   desc.layeredRendering, desc.colorFormat, desc.depthFormat,
                                   desc.debugName};
        const auto it = mOutputPlanningState->managedPrimaryTargets.find(key);
        if (it != mOutputPlanningState->managedPrimaryTargets.end() &&
            mDevice.renderTargetSystem().isValidRenderTarget(it->second))
        {
            return it->second;
        }

        gpu::GpuRenderTargetHandle handle = mDevice.renderTargetSystem().createRenderTarget(desc);
        if (mDevice.renderTargetSystem().isValidRenderTarget(handle))
        {
            mOutputPlanningState->managedPrimaryTargets[key] = handle;
        }
        return handle;
    };

    std::vector<ResolvedCameraView> resolvedCameras;
    resolvedCameras.reserve(cameras.size());
    std::optional<DisplayResolveRequest> displayResolve;

    const auto buildManagedDesc = [&](const CameraData& camera)
    {
        gpu::GpuRenderTargetDesc desc = defaultTargetDesc;
        desc.width = camera.outputWidth == 0 ? defaultTargetDesc.width : camera.outputWidth;
        desc.height = camera.outputHeight == 0 ? defaultTargetDesc.height : camera.outputHeight;
        desc.arraySize = 1u;
        desc.layeredRendering = true;
        desc.shaderReadable = true;
        desc.debugName = "CRESSimNeo.ManagedPrimary";
        return desc;
    };

    const auto sameManagedCompatibility = [&](const CameraData& lhs, const CameraData& rhs)
    {
        const gpu::GpuRenderTargetDesc lhsDesc = buildManagedDesc(lhs);
        const gpu::GpuRenderTargetDesc rhsDesc = buildManagedDesc(rhs);
        return lhsDesc.width == rhsDesc.width && lhsDesc.height == rhsDesc.height &&
               lhsDesc.color == rhsDesc.color && lhsDesc.depth == rhsDesc.depth &&
               lhsDesc.shaderReadable == rhsDesc.shaderReadable &&
               lhsDesc.colorFormat == rhsDesc.colorFormat &&
               lhsDesc.depthFormat == rhsDesc.depthFormat &&
               lhs.clearColor == rhs.clearColor && lhs.clearDepth == rhs.clearDepth &&
               lhs.clearColorValue.x == rhs.clearColorValue.x &&
               lhs.clearColorValue.y == rhs.clearColorValue.y &&
               lhs.clearColorValue.z == rhs.clearColorValue.z &&
               lhs.clearColorValue.w == rhs.clearColorValue.w &&
               lhs.clearDepthValue == rhs.clearDepthValue;
    };

    const auto resolveExplicitTarget = [&](const CameraData& camera,
                                           ResolvedCameraView& outView) -> bool
    {
        gpu::GpuRenderTargetHandle target = camera.output.binding.target;
        if (!mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            target = defaultTarget;
        }
        if (!mDevice.renderTargetSystem().isValidRenderTarget(target))
        {
            return false;
        }

        gpu::GpuRenderTargetDesc targetDesc{};
        if (!mDevice.renderTargetSystem().tryGetRenderTargetDesc(target, targetDesc))
        {
            return false;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            ++stats.renderTargetResizeRequests;

            RequestedExtent desired{};
            desired.width  = camera.outputWidth == 0 ? targetDesc.width : camera.outputWidth;
            desired.height = camera.outputHeight == 0 ? targetDesc.height : camera.outputHeight;

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
                if (updateResult == gpu::GpuRenderTargetUpdateResult::Failed ||
                    !mDevice.renderTargetSystem().tryGetRenderTargetDesc(target, targetDesc))
                {
                    return false;
                }
            }
            else
            {
                ++stats.renderTargetResizeNoOps;
            }
        }

        outView.entityId          = camera.entityId;
        outView.outputBinding     = camera.output.binding;
        outView.outputBinding.target = target;
        outView.outputBinding.layerCount = 1u;
        outView.outputBinding.firstLayer =
            std::min(outView.outputBinding.firstLayer, targetDesc.arraySize - 1u);
        outView.outputTargetDesc  = targetDesc;
        outView.viewport          = detail::normalizeViewport(camera.viewport);
        outView.clearColor        = camera.clearColor;
        outView.clearDepth        = camera.clearDepth;
        outView.clearColorValue   = camera.clearColorValue;
        outView.clearDepthValue   = camera.clearDepthValue;
        outView.envIndex          = camera.envIndex;
        outView.cameraSlot        = camera.cameraSlot;
        outView.globalCameraIndex =
            camera.envIndex * std::max(gpuScene.layout.maxCamerasPerEnv, 1u) + camera.cameraSlot;
        return true;
    };

    for (std::size_t cameraIndex = 0; cameraIndex < cameras.size();)
    {
        const CameraData& camera = cameras[cameraIndex];
        if (camera.output.mode == gpu::CameraOutputMode::ManagedPrimary && hasDefaultTarget)
        {
            std::size_t runEnd = cameraIndex + 1u;
            while (runEnd < cameras.size() &&
                   cameras[runEnd].output.mode == gpu::CameraOutputMode::ManagedPrimary &&
                   sameManagedCompatibility(cameras[cameraIndex], cameras[runEnd]))
            {
                ++runEnd;
            }

            gpu::GpuRenderTargetDesc managedDesc = buildManagedDesc(cameras[cameraIndex]);
            managedDesc.arraySize = static_cast<std::uint32_t>(runEnd - cameraIndex);
            managedDesc.layeredRendering = true;
            const gpu::GpuRenderTargetHandle managedTarget = acquireManagedPrimaryTarget(managedDesc);
            if (mDevice.renderTargetSystem().isValidRenderTarget(managedTarget))
            {
                for (std::size_t runIndex = cameraIndex; runIndex < runEnd; ++runIndex)
                {
                    const CameraData& managedCamera = cameras[runIndex];
                    ResolvedCameraView resolved{};
                    resolved.entityId = managedCamera.entityId;
                    resolved.outputBinding = gpu::GpuRenderTargetBinding{
                        managedTarget, static_cast<std::uint32_t>(runIndex - cameraIndex), 1u};
                    resolved.outputTargetDesc  = managedDesc;
                    resolved.viewport          = detail::normalizeViewport(managedCamera.viewport);
                    resolved.clearColor        = managedCamera.clearColor;
                    resolved.clearDepth        = managedCamera.clearDepth;
                    resolved.clearColorValue   = managedCamera.clearColorValue;
                    resolved.clearDepthValue   = managedCamera.clearDepthValue;
                    resolved.envIndex          = managedCamera.envIndex;
                    resolved.cameraSlot        = managedCamera.cameraSlot;
                    resolved.globalCameraIndex = managedCamera.envIndex *
                                                     std::max(gpuScene.layout.maxCamerasPerEnv, 1u) +
                                                 managedCamera.cameraSlot;
                    resolvedCameras.push_back(resolved);
                    ++stats.cameraCount;

                    if (!displayResolve.has_value())
                    {
                        displayResolve = DisplayResolveRequest{
                            resolved.outputBinding,
                            managedDesc,
                            mDevice.renderTargetSystem().defaultRenderTargetBinding(),
                            defaultTargetDesc,
                            false,
                            false,
                            resolved.clearColorValue,
                            resolved.clearDepthValue};
                    }
                }
            }

            cameraIndex = runEnd;
            continue;
        }

        ResolvedCameraView resolved{};
        if (resolveExplicitTarget(camera, resolved))
        {
            resolvedCameras.push_back(resolved);
            ++stats.cameraCount;
        }
        ++cameraIndex;
    }

    const FrameRenderPlan renderPlan =
        detail::buildFrameRenderPlan(std::move(resolvedCameras), lightData, displayResolve);

    for (const CameraBatchView& batch : renderPlan.cameraBatches)
    {
        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->executeBatch(frameContext, batch, world, passStats);
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
