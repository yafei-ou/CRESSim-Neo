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

struct GraphicsScenePrepareConstants
{
    Diligent::float4x4 viewProjectionMatrix = Diligent::float4x4::Identity();
    std::array<Diligent::float4x4, kShadowCascadeCount> lightViewProjectionMatrices{};
    std::uint32_t renderableCount = 0u;
    std::uint32_t shadowCascadeCount = 0u;
    std::uint32_t cameraEnvIndex = 0u;
    std::uint32_t padding = 0u;
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

} // namespace

struct Renderer::GpuScenePrepareState
{
    gpu::GpuComputePass pass;
    Diligent::RefCntAutoPtr<Diligent::IBuffer> constantsBuffer;
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
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) || backendContext.renderDevice == nullptr)
    {
        return false;
    }

    gpu::ShaderLibrary shaderLibrary(mDevice.shaderSourceDirectory());
    Diligent::IShaderSourceInputStreamFactory* streamFactory = shaderLibrary.streamFactory();
    if (streamFactory == nullptr)
    {
        return false;
    }

    if (!mGpuScenePrepare->pass.initialize(backendContext.renderDevice, streamFactory, 1ull,
                                           kScenePreparePassDefinition))
    {
        return false;
    }

    Diligent::BufferDesc desc{};
    desc.Name = "CRESSimNeo.Graphics.ScenePrepare.Constants";
    desc.Size = sizeof(GraphicsScenePrepareConstants);
    desc.Usage = Diligent::USAGE_DYNAMIC;
    desc.BindFlags = Diligent::BIND_UNIFORM_BUFFER;
    desc.CPUAccessFlags = Diligent::CPU_ACCESS_WRITE;
    desc.ImmediateContextMask = 1ull;
    backendContext.renderDevice->CreateBuffer(desc, nullptr, &mGpuScenePrepare->constantsBuffer);
    if (mGpuScenePrepare->constantsBuffer == nullptr)
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
    if (sceneView.poses.positionsBuffer == nullptr || sceneView.poses.orientationsBuffer == nullptr ||
        sceneView.poses.scalesBuffer == nullptr || sceneView.renderableMetadataBuffer == nullptr ||
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
    if (!mDevice.tryGetGraphicsBackendContext(backendContext) || backendContext.immediateContext == nullptr)
    {
        return false;
    }

    GraphicsScenePrepareConstants constants{};
    constants.viewProjectionMatrix = frameView.viewProjectionMatrix.Transpose();
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        constants.lightViewProjectionMatrices[cascadeIdx] =
            frameView.lightViewProjectionMatrices[cascadeIdx].Transpose();
    }
    constants.renderableCount = sceneView.renderableCount;
    constants.shadowCascadeCount = frameView.shadowCascadeCount;
    constants.cameraEnvIndex = 0u;

    void* mappedConstants = nullptr;
    backendContext.immediateContext->MapBuffer(mGpuScenePrepare->constantsBuffer, Diligent::MAP_WRITE,
                                               Diligent::MAP_FLAG_DISCARD, mappedConstants);
    if (mappedConstants == nullptr)
    {
        return false;
    }
    std::memcpy(mappedConstants, &constants, sizeof(constants));
    backendContext.immediateContext->UnmapBuffer(mGpuScenePrepare->constantsBuffer,
                                                 Diligent::MAP_WRITE);

    const std::array bindings{
        gpu::GpuBufferBinding{"GraphicsScenePrepareConstants", mGpuScenePrepare->constantsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityPositions", sceneView.poses.positionsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityOrientations", sceneView.poses.orientationsBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_EntityScales", sceneView.poses.scalesBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RenderableMetadata", sceneView.renderableMetadataBuffer,
                              Diligent::BUFFER_VIEW_SHADER_RESOURCE},
        gpu::GpuBufferBinding{"g_RenderableVisibilityFlagsRW",
                              sceneView.renderableVisibilityFlagsBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
        gpu::GpuBufferBinding{"g_RenderableShadowCascadeMasksRW",
                              sceneView.renderableShadowCascadeMasksBuffer,
                              Diligent::BUFFER_VIEW_UNORDERED_ACCESS},
    };

    return mGpuScenePrepare->pass.dispatch(backendContext.immediateContext, 0u, bindings,
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

RenderStats Renderer::render(const common::FrameContext& frameContext, const RenderWorld& world)
{
    RenderStats stats{};

    if (!mInitialized)
    {
        return stats;
    }

    mDevice.beginFrame(frameContext);

    const auto& renderables = world.renderables();
    const auto preparedRenderables = detail::buildPreparedRenderables(
        renderables, mResourceManager, world.gpuEntityPoseIndices());
    const ForwardDirectionalLightData lightData = detail::buildMainLight(world.directionalLights());

    stats.renderableCount = static_cast<std::uint32_t>(renderables.size());
    stats.validRenderableCount = static_cast<std::uint32_t>(preparedRenderables.size());
    stats.lightCount = static_cast<std::uint32_t>(world.directionalLights().size());

    std::vector<CameraData> cameras = detail::sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(detail::defaultCamera());
    }

    struct RequestedExtent
    {
        std::uint32_t width = 0;
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
            desired.width = (camera.outputWidth == 0 ? targetDesc.width : camera.outputWidth);
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

        if (!prepareGpuScene(frameView, world.gpuEntityScene()))
        {
            return;
        }

        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->execute(frameContext, frameView, world.gpuEntityScene(), queues,
                                            passStats);
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
