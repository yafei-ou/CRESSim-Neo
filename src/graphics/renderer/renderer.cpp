#include "graphics/renderer.h"

#include "graphics/device/graphics_device_impl.h"
#include "graphics/math/diligent_math_utils.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/passes/render_pass_types.h"
#include "graphics/renderer/services/debug_view_presenter.h"

#include "DiligentEngine/DiligentCore/Common/interface/AdvancedMath.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{

float clamp01(float value)
{
    return std::max(0.0f, std::min(value, 1.0f));
}

float clampPositive(float value, float fallback)
{
    return value > 0.0f ? value : fallback;
}

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    RenderViewport normalized{};
    normalized.x = clamp01(viewport.x);
    normalized.y = clamp01(viewport.y);
    normalized.width = clamp01(viewport.width);
    normalized.height = clamp01(viewport.height);

    const float maxWidth = std::max(0.0f, 1.0f - normalized.x);
    const float maxHeight = std::max(0.0f, 1.0f - normalized.y);
    normalized.width = std::min(normalized.width, maxWidth);
    normalized.height = std::min(normalized.height, maxHeight);

    if (normalized.width == 0.0f)
    {
        normalized.width = 1.0f;
        normalized.x = 0.0f;
    }
    if (normalized.height == 0.0f)
    {
        normalized.height = 1.0f;
        normalized.y = 0.0f;
    }

    return normalized;
}

struct ValidRenderable
{
    const RenderableInstance* instance = nullptr;
    const MeshResourceDesc* mesh = nullptr;
    const MaterialResourceDesc* material = nullptr;
    bool hasLocalBounds = false;
    common::Vec3f localBoundsMin{};
    common::Vec3f localBoundsMax{};
};

std::vector<ValidRenderable> gatherValidRenderables(const std::vector<RenderableInstance>& renderables, const RenderResourceManager& resources)
{
    std::vector<ValidRenderable> valid;
    valid.reserve(renderables.size());

    for (const RenderableInstance& renderable : renderables)
    {
        const MeshResourceDesc* mesh = resources.tryGetMesh(renderable.mesh);
        const MaterialResourceDesc* material = resources.tryGetMaterial(renderable.material);
        if (mesh == nullptr || material == nullptr)
        {
            continue;
        }
        if (mesh->vertices.empty() || mesh->indices.size() < 3)
        {
            continue;
        }

        ValidRenderable validRenderable{};
        validRenderable.instance = &renderable;
        validRenderable.mesh = mesh;
        validRenderable.material = material;
        validRenderable.hasLocalBounds = resources.tryGetMeshLocalBounds(renderable.mesh, validRenderable.localBoundsMin, validRenderable.localBoundsMax);
        valid.push_back(validRenderable);
    }

    return valid;
}

ForwardDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights)
{
    ForwardDirectionalLightData out{};
    if (lights.empty())
    {
        return out;
    }

    const DirectionalLightData& light = lights.front();
    out.direction[0] = light.direction.x;
    out.direction[1] = light.direction.y;
    out.direction[2] = light.direction.z;
    out.color[0] = light.color.x;
    out.color[1] = light.color.y;
    out.color[2] = light.color.z;
    out.intensity = light.intensity;
    return out;
}

bool buildDrawCommand(
    const ValidRenderable& renderable,
    const FrameViewData& frameView,
    const ForwardDirectionalLightData& light,
    const RenderResourceManager& resources,
    ForwardDrawCommand& outCommand)
{
    if (renderable.instance == nullptr || renderable.mesh == nullptr || renderable.material == nullptr)
    {
        return false;
    }

    const auto& mesh = *renderable.mesh;
    const auto& material = *renderable.material;
    if (mesh.vertices.empty() || mesh.indices.size() < 3)
    {
        return false;
    }

    const Diligent::float4x4 modelMatrix = math::transformMatrix(renderable.instance->worldTransform);

    outCommand = {};
    outCommand.shadingModel = material.shadingModel;
    outCommand.meshId = renderable.instance->mesh.id;
    outCommand.materialId = renderable.instance->material.id;
    outCommand.meshVersion = resources.meshVersion(renderable.instance->mesh);
    outCommand.vertexData = mesh.vertices.data();
    outCommand.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData = mesh.indices.data();
    outCommand.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
    math::copyMatrixRowMajor(outCommand.modelMatrix, modelMatrix);
    math::copyMatrixRowMajor(outCommand.viewProjectionMatrix, frameView.viewProjectionMatrix);
    outCommand.cameraPosition[0] = frameView.cameraWorldPosition.x;
    outCommand.cameraPosition[1] = frameView.cameraWorldPosition.y;
    outCommand.cameraPosition[2] = frameView.cameraWorldPosition.z;
    outCommand.material.baseColor[0] = material.baseColor.x;
    outCommand.material.baseColor[1] = material.baseColor.y;
    outCommand.material.baseColor[2] = material.baseColor.z;
    outCommand.material.metallic = material.metallic;
    outCommand.material.roughness = material.roughness;
    outCommand.material.opacity = clamp01(material.opacity);
    outCommand.material.receivesShadows = material.receivesShadows ? 1.0f : 0.0f;
    outCommand.light = light;
    return true;
}

Diligent::float4x4 buildViewProjection(const CameraData& camera, float outputWidth, float outputHeight)
{
    const float aspect = clampPositive(outputWidth, 1.0f) / clampPositive(outputHeight, 1.0f);

    const Diligent::float4x4 view = math::viewMatrixFromTransform(camera.worldTransform);
    const Diligent::float4x4 projection = math::perspectiveMatrix(camera.verticalFovDegrees, aspect, camera.nearClip, camera.farClip);
    return view * projection;
}

Diligent::float3 cameraPosition(const CameraData& camera)
{
    return math::toFloat3(camera.worldTransform.position);
}

CameraData defaultCamera()
{
    CameraData camera{};
    camera.verticalFovDegrees = 60.0f;
    camera.nearClip = 0.01f;
    camera.farClip = 1000.0f;
    camera.viewport = {};
    return camera;
}

std::vector<CameraData> sortedCameras(const RenderWorld& world)
{
    std::vector<CameraData> cameras = world.cameras();
    std::sort(cameras.begin(), cameras.end(), [](const CameraData& lhs, const CameraData& rhs) {
        if (lhs.renderOrder != rhs.renderOrder)
        {
            return lhs.renderOrder < rhs.renderOrder;
        }
        return lhs.entityId < rhs.entityId;
    });
    return cameras;
}

float squaredDistanceToCamera(const common::Transform& transform, const Diligent::float3& cameraWorldPosition)
{
    const float dx = transform.position.x - cameraWorldPosition.x;
    const float dy = transform.position.y - cameraWorldPosition.y;
    const float dz = transform.position.z - cameraWorldPosition.z;
    return dx * dx + dy * dy + dz * dz;
}

bool isVisibleByFrustum(const ValidRenderable& renderable, const Diligent::ViewFrustum& frustum)
{
    if (renderable.instance == nullptr)
    {
        return false;
    }

    if (!renderable.hasLocalBounds)
    {
        return true;
    }

    const Diligent::BoundBox localBounds{
        Diligent::float3{renderable.localBoundsMin.x, renderable.localBoundsMin.y, renderable.localBoundsMin.z},
        Diligent::float3{renderable.localBoundsMax.x, renderable.localBoundsMax.y, renderable.localBoundsMax.z}};
    const Diligent::float4x4 modelMatrix = math::transformMatrix(renderable.instance->worldTransform);
    const Diligent::BoundBox worldBounds = localBounds.Transform(modelMatrix);

    const Diligent::BoxVisibility visibility = Diligent::GetBoxVisibility(frustum, worldBounds);
    return visibility != Diligent::BoxVisibility::Invisible;
}

CameraRenderQueues buildCameraRenderQueues(
    const std::vector<ValidRenderable>& validRenderables,
    const FrameViewData& frameView,
    const ForwardDirectionalLightData& lightData,
    const RenderResourceManager& resources,
    RenderStats& stats)
{
    CameraRenderQueues queues{};
    queues.opaque.reserve(validRenderables.size());
    queues.transparent.reserve(validRenderables.size());
    queues.shadowCasters.reserve(validRenderables.size());

    for (const ValidRenderable& renderable : validRenderables)
    {
        if (renderable.instance == nullptr)
        {
            continue;
        }

        if (!isVisibleByFrustum(renderable, frameView.viewFrustum))
        {
            ++stats.culledRenderableCount;
            continue;
        }

        ForwardDrawCommand drawCommand{};
        if (!buildDrawCommand(renderable, frameView, lightData, resources, drawCommand))
        {
            continue;
        }

        QueuedDraw queuedDraw{};
        queuedDraw.entityId = renderable.instance->entityId;
        queuedDraw.meshId = renderable.instance->mesh.id;
        queuedDraw.materialId = renderable.instance->material.id;
        queuedDraw.depth = squaredDistanceToCamera(renderable.instance->worldTransform, frameView.cameraWorldPosition);
        queuedDraw.castsShadows = renderable.material->castsShadows;
        queuedDraw.receivesShadows = renderable.material->receivesShadows;
        queuedDraw.transparent = (renderable.material->blendMode == BlendMode::Transparent);
        queuedDraw.drawCommand = drawCommand;

        if (queuedDraw.transparent)
        {
            queues.transparent.push_back(queuedDraw);
        }
        else
        {
            queues.opaque.push_back(queuedDraw);
        }

        if (queuedDraw.castsShadows && !queuedDraw.transparent)
        {
            queues.shadowCasters.push_back(queuedDraw);
        }
    }

    std::sort(queues.opaque.begin(), queues.opaque.end(), [](const QueuedDraw& lhs, const QueuedDraw& rhs) {
        if (lhs.drawCommand.shadingModel != rhs.drawCommand.shadingModel)
        {
            return static_cast<int>(lhs.drawCommand.shadingModel) < static_cast<int>(rhs.drawCommand.shadingModel);
        }
        if (lhs.materialId != rhs.materialId)
        {
            return lhs.materialId < rhs.materialId;
        }
        if (lhs.meshId != rhs.meshId)
        {
            return lhs.meshId < rhs.meshId;
        }
        if (lhs.depth != rhs.depth)
        {
            return lhs.depth < rhs.depth;
        }
        return lhs.entityId < rhs.entityId;
    });

    std::sort(queues.transparent.begin(), queues.transparent.end(), [](const QueuedDraw& lhs, const QueuedDraw& rhs) {
        if (lhs.depth != rhs.depth)
        {
            return lhs.depth > rhs.depth;
        }
        if (lhs.materialId != rhs.materialId)
        {
            return lhs.materialId < rhs.materialId;
        }
        if (lhs.meshId != rhs.meshId)
        {
            return lhs.meshId < rhs.meshId;
        }
        return lhs.entityId < rhs.entityId;
    });

    std::sort(queues.shadowCasters.begin(), queues.shadowCasters.end(), [](const QueuedDraw& lhs, const QueuedDraw& rhs) {
        if (lhs.materialId != rhs.materialId)
        {
            return lhs.materialId < rhs.materialId;
        }
        if (lhs.meshId != rhs.meshId)
        {
            return lhs.meshId < rhs.meshId;
        }
        return lhs.entityId < rhs.entityId;
    });

    stats.opaqueQueueCount += static_cast<std::uint32_t>(queues.opaque.size());
    stats.transparentQueueCount += static_cast<std::uint32_t>(queues.transparent.size());
    stats.shadowCasterQueueCount += static_cast<std::uint32_t>(queues.shadowCasters.size());
    return queues;
}

FrameViewData buildFrameViewData(
    const CameraData& camera,
    const RenderTargetDesc& targetDesc,
    RenderTargetHandle target,
    const RenderViewport& viewport)
{
    FrameViewData frameView{};
    frameView.target = target;
    frameView.viewport = viewport;
    frameView.outputWidth = targetDesc.width;
    frameView.outputHeight = targetDesc.height;
    frameView.viewProjectionMatrix = buildViewProjection(
        camera,
        static_cast<float>(targetDesc.width),
        static_cast<float>(targetDesc.height));
    frameView.cameraWorldPosition = cameraPosition(camera);
    Diligent::ExtractViewFrustumPlanesFromMatrix(frameView.viewProjectionMatrix, frameView.viewFrustum, false);
    return frameView;
}

} // namespace

Renderer::Renderer(GraphicsDevice& device, RenderResourceManager& resourceManager, const RendererDesc& desc) :
    mDevice(device),
    mResourceManager(resourceManager),
    mDesc(desc)
{
}

Renderer::~Renderer() = default;

bool Renderer::initialize()
{
    GraphicsDeviceImpl& deviceImpl = static_cast<GraphicsDeviceImpl&>(mDevice);
    mForwardPipeline = std::make_unique<detail::ForwardPipeline>(deviceImpl);
    if (!mForwardPipeline->initialize())
    {
        mForwardPipeline.reset();
        return false;
    }

    if (mDesc.debugViewer.enabled)
    {
        mDebugViewPresenter = std::make_unique<detail::DebugViewPresenter>(deviceImpl, mDesc.debugViewer);
        if (!mDebugViewPresenter->initialize())
        {
            mDebugViewPresenter.reset();
            mForwardPipeline.reset();
            return false;
        }
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
    const auto validRenderables = gatherValidRenderables(renderables, mResourceManager);
    const ForwardDirectionalLightData lightData = buildMainLight(world.directionalLights());

    stats.renderableCount = static_cast<std::uint32_t>(renderables.size());
    stats.validRenderableCount = static_cast<std::uint32_t>(validRenderables.size());
    stats.lightCount = static_cast<std::uint32_t>(world.directionalLights().size());

    std::vector<CameraData> cameras = sortedCameras(world);
    if (cameras.empty())
    {
        cameras.push_back(defaultCamera());
    }

    const auto renderCamera = [&](const CameraData& camera) {
        RenderTargetHandle target = camera.outputTarget;
        if (!mDevice.isValidRenderTarget(target))
        {
            target = mDevice.defaultRenderTarget();
        }
        if (!mDevice.isValidRenderTarget(target))
        {
            return;
        }

        if (camera.outputWidth > 0 || camera.outputHeight > 0)
        {
            (void)mDevice.resizeRenderTarget(target, camera.outputWidth, camera.outputHeight);
        }

        const RenderViewport viewport = normalizeViewport(camera.viewport);
        mDevice.setRenderTargetViewport(target, viewport);

        RenderTargetDesc targetDesc{};
        if (!mDevice.tryGetRenderTargetDesc(target, targetDesc))
        {
            return;
        }

        const FrameViewData frameView = buildFrameViewData(camera, targetDesc, target, viewport);
        const CameraRenderQueues queues = buildCameraRenderQueues(validRenderables, frameView, lightData, mResourceManager, stats);

        const RenderPassBeginDesc beginDesc{};
        mDevice.beginRenderTarget(target, frameContext, beginDesc);

        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->execute(target, queues, passStats);
        }

        mDevice.endRenderTarget(target, frameContext);

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
    if (mDebugViewPresenter != nullptr)
    {
        (void)mDebugViewPresenter->present(mDevice.defaultRenderTarget());
    }

    return stats;
}

} // namespace cressim::neo::graphics
