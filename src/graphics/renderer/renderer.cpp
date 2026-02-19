#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "graphics/device/graphics_device_impl.h"
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

constexpr float kEpsilon = 1.0e-12f;

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
    Diligent::float3 localBoundsMin{};
    Diligent::float3 localBoundsMax{};
};

Diligent::float3 safeNormalize(const Diligent::float3& value, const Diligent::float3& fallback)
{
    const float lengthSq = Diligent::dot(value, value);
    if (lengthSq <= kEpsilon)
    {
        return fallback;
    }
    return value * (1.0f / std::sqrt(lengthSq));
}

Diligent::QuaternionF normalizeQuaternion(const Diligent::QuaternionF& value)
{
    const float lengthSq = Diligent::dot(value.q, value.q);
    if (lengthSq <= kEpsilon)
    {
        return Diligent::QuaternionF{0.0f, 0.0f, 0.0f, 1.0f};
    }
    return Diligent::normalize(value);
}

Diligent::float4x4 worldMatrixFromTransform(const common::Transform& transform)
{
    const Diligent::QuaternionF rotation = normalizeQuaternion(transform.rotation);
    return Diligent::float4x4::Scale(transform.scale) * rotation.ToMatrix() * Diligent::float4x4::Translation(transform.position);
}

Diligent::float4x4 normalMatrixFromModelMatrix(const Diligent::float4x4& model)
{
    // Normal matrix = inverse-transpose of the model's linear (upper-left 3x3) part.
    // Return as 4x4 with last row/col = identity-ish so we can use it easily in shaders.

    const auto linear3x3 = Diligent::float3x3{
        Diligent::float3{model._11, model._12, model._13},
        Diligent::float3{model._21, model._22, model._23},
        Diligent::float3{model._31, model._32, model._33},
    };

    const Diligent::float3x3 normal3x3 = linear3x3.Inverse().Transpose();

    return Diligent::float4x4{
        normal3x3._11, normal3x3._12, normal3x3._13, 0.0f,
        normal3x3._21, normal3x3._22, normal3x3._23, 0.0f,
        normal3x3._31, normal3x3._32, normal3x3._33, 0.0f,
        0.0f,          0.0f,          0.0f,          1.0f
    };
}

Diligent::float4x4 viewMatrixFromCameraTransform(const common::Transform& cameraTransform)
{
    const Diligent::QuaternionF rotation = normalizeQuaternion(cameraTransform.rotation);
    const Diligent::float3 eye = cameraTransform.position;
    const Diligent::float3 zAxis = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f}), Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 up = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f}), Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float3 xAxis = safeNormalize(Diligent::cross(up, zAxis), Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis = safeNormalize(Diligent::cross(zAxis, xAxis), Diligent::float3{0.0f, 1.0f, 0.0f});

    const Diligent::float4x4 viewRotation = Diligent::float4x4::ViewFromBasis(xAxis, yAxis, zAxis);
    const Diligent::float4x4 viewTranslation = Diligent::float4x4::Translation(-eye);
    return viewTranslation * viewRotation;
}

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
        out.direction = Diligent::float3{0.0f, 0.0f, 0.0f};
        out.intensity = 0.0f;
        return out;
    }

    const DirectionalLightData& light = lights.front();
    out.direction = light.direction;
    out.color = light.color;
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

    const Diligent::float4x4 modelMatrix = worldMatrixFromTransform(renderable.instance->worldTransform);

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
    outCommand.modelMatrix = modelMatrix;
    outCommand.viewProjectionMatrix = frameView.viewProjectionMatrix;
    outCommand.lightViewProjectionMatrix = frameView.lightViewProjectionMatrix;
    outCommand.normalMatrix = normalMatrixFromModelMatrix(modelMatrix);
    outCommand.cameraPosition = frameView.cameraWorldPosition;
    outCommand.material.baseColor = material.baseColor;
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
    const float fovRadians = camera.verticalFovDegrees * 0.017453292519943295769f;
    const float nearPlane = std::max(camera.nearClip, 0.001f);
    const float farPlane = std::max(camera.farClip, nearPlane + 0.001f);

    const Diligent::float4x4 view = viewMatrixFromCameraTransform(camera.worldTransform);
    const Diligent::float4x4 projection = Diligent::float4x4::Projection(fovRadians, aspect, nearPlane, farPlane, false);
    return view * projection;
}

Diligent::float4x4 lookAtMatrix(const Diligent::float3& eye, const Diligent::float3& at, const Diligent::float3& up)
{
    const Diligent::float3 zAxis = safeNormalize(at - eye, Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 xAxis = safeNormalize(Diligent::cross(up, zAxis), Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis = safeNormalize(Diligent::cross(zAxis, xAxis), Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float4x4 viewRotation = Diligent::float4x4::ViewFromBasis(xAxis, yAxis, zAxis);
    const Diligent::float4x4 viewTranslation = Diligent::float4x4::Translation(-eye);
    return viewTranslation * viewRotation;
}

Diligent::float4x4 buildDirectionalLightViewProjection(
    const ForwardDirectionalLightData& light,
    const Diligent::float3& shadowFocusWorldPosition,
    bool& outHasDirectionalLight)
{
    const Diligent::float3 lightDirection = light.direction;
    outHasDirectionalLight = light.intensity > 0.0f && Diligent::dot(lightDirection, lightDirection) > 1.0e-6f;
    if (!outHasDirectionalLight)
    {
        return Diligent::float4x4::Identity();
    }

    const Diligent::float3 dir = Diligent::normalize(lightDirection);
    const Diligent::float3 sceneCenter = shadowFocusWorldPosition;
    const Diligent::float3 lightPosition = sceneCenter - dir * 18.0f;
    const Diligent::float3 upCandidate =
        std::abs(Diligent::dot(dir, Diligent::float3{0.0f, 1.0f, 0.0f})) > 0.98f ?
            Diligent::float3{0.0f, 0.0f, 1.0f} :
            Diligent::float3{0.0f, 1.0f, 0.0f};

    const Diligent::float4x4 lightView = lookAtMatrix(lightPosition, sceneCenter, upCandidate);
    const Diligent::float4x4 lightProjection = Diligent::float4x4::Ortho(24.0f, 24.0f, 0.1f, 60.0f, false);
    return lightView * lightProjection;
}

bool buildWorldBoundsCenter(const std::vector<ValidRenderable>& validRenderables, Diligent::float3& outCenter)
{
    bool hasBounds = false;
    Diligent::float3 minBounds{0.0f, 0.0f, 0.0f};
    Diligent::float3 maxBounds{0.0f, 0.0f, 0.0f};

    for (const ValidRenderable& renderable : validRenderables)
    {
        if (renderable.instance == nullptr)
        {
            continue;
        }

        Diligent::float3 worldMin{};
        Diligent::float3 worldMax{};
        if (renderable.hasLocalBounds)
        {
            const Diligent::BoundBox localBounds{
                renderable.localBoundsMin,
                renderable.localBoundsMax};
            const Diligent::float4x4 modelMatrix = worldMatrixFromTransform(renderable.instance->worldTransform);
            const Diligent::BoundBox worldBounds = localBounds.Transform(modelMatrix);
            worldMin = worldBounds.Min;
            worldMax = worldBounds.Max;
        }
        else
        {
            const Diligent::float3 position = renderable.instance->worldTransform.position;
            worldMin = position;
            worldMax = position;
        }

        if (!hasBounds)
        {
            minBounds = worldMin;
            maxBounds = worldMax;
            hasBounds = true;
            continue;
        }

        minBounds.x = std::min(minBounds.x, worldMin.x);
        minBounds.y = std::min(minBounds.y, worldMin.y);
        minBounds.z = std::min(minBounds.z, worldMin.z);
        maxBounds.x = std::max(maxBounds.x, worldMax.x);
        maxBounds.y = std::max(maxBounds.y, worldMax.y);
        maxBounds.z = std::max(maxBounds.z, worldMax.z);
    }

    if (!hasBounds)
    {
        outCenter = Diligent::float3{0.0f, 0.0f, 0.0f};
        return false;
    }

    outCenter = (minBounds + maxBounds) * 0.5f;
    return true;
}

Diligent::float3 cameraPosition(const CameraData& camera)
{
    return camera.worldTransform.position;
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
        renderable.localBoundsMin,
        renderable.localBoundsMax};
    const Diligent::float4x4 modelMatrix = worldMatrixFromTransform(renderable.instance->worldTransform);
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

        const bool cameraVisible = isVisibleByFrustum(renderable, frameView.viewFrustum);
        const bool transparent = (renderable.material->blendMode == BlendMode::Transparent);
        const bool canCastShadows = renderable.material->castsShadows && !transparent;
        const bool lightVisible = frameView.hasDirectionalLight && isVisibleByFrustum(renderable, frameView.lightFrustum);

        if (!cameraVisible)
        {
            ++stats.culledRenderableCount;
        }

        const bool needsMainPass = cameraVisible;
        const bool needsShadowPass = canCastShadows && lightVisible;
        if (!needsMainPass && !needsShadowPass)
        {
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
        queuedDraw.transparent = transparent;
        queuedDraw.drawCommand = drawCommand;

        if (needsMainPass && queuedDraw.transparent)
        {
            queues.transparent.push_back(queuedDraw);
        }
        else if (needsMainPass)
        {
            queues.opaque.push_back(queuedDraw);
        }

        if (needsShadowPass)
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
    const RenderViewport& viewport,
    const ForwardDirectionalLightData& lightData,
    const Diligent::float3& shadowFocusWorldPosition)
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
    frameView.lightViewProjectionMatrix =
        buildDirectionalLightViewProjection(lightData, shadowFocusWorldPosition, frameView.hasDirectionalLight);
    Diligent::ExtractViewFrustumPlanesFromMatrix(frameView.viewProjectionMatrix, frameView.viewFrustum, false);
    if (frameView.hasDirectionalLight)
    {
        Diligent::ExtractViewFrustumPlanesFromMatrix(frameView.lightViewProjectionMatrix, frameView.lightFrustum, false);
    }
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
    Diligent::float3 worldCenter = Diligent::float3{0.0f, 0.0f, 0.0f};
    const bool hasWorldCenter = buildWorldBoundsCenter(validRenderables, worldCenter);

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

        RenderTargetDesc targetDesc{};
        if (!mDevice.tryGetRenderTargetDesc(target, targetDesc))
        {
            return;
        }

        const Diligent::float3 shadowFocus = hasWorldCenter ? worldCenter : cameraPosition(camera);
        const FrameViewData frameView = buildFrameViewData(camera, targetDesc, target, viewport, lightData, shadowFocus);
        const CameraRenderQueues queues = buildCameraRenderQueues(validRenderables, frameView, lightData, mResourceManager, stats);

        ForwardPassExecutionStats passStats{};
        if (mForwardPipeline != nullptr)
        {
            (void)mForwardPipeline->execute(frameContext, frameView, queues, passStats);
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
    if (mDebugViewPresenter != nullptr)
    {
        (void)mDebugViewPresenter->present(mDevice.defaultRenderTarget());
    }

    return stats;
}

} // namespace cressim::neo::graphics
