#include "graphics/renderer.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"
#include "graphics/device/graphics_device_impl.h"
#include "graphics/renderer/passes/forward_pipeline.h"
#include "graphics/renderer/passes/render_pass_types.h"
#include "graphics/renderer/services/debug_view_presenter.h"

#include "DiligentEngine/DiligentCore/Common/interface/AdvancedMath.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace cressim::neo::graphics
{

namespace
{

constexpr float kEpsilon = 1.0e-12f;
constexpr float kDegreesToRadians = 0.017453292519943295769f;
constexpr float kCascadeSplitLambda = 0.85f;
constexpr float kCascadeStabilization = 16.0f;
constexpr float kCascadeDepthPadding = 16.0f;

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
    out.shadowDistance = light.shadowDistance;
    out.shadowFadeDistance = light.shadowFadeDistance;
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
    outCommand.programFamily = material.pipeline.programFamily;
    outCommand.materialFeatureFlags = material.pipeline.featureFlags;
    outCommand.meshId = renderable.instance->mesh.id;
    outCommand.materialId = renderable.instance->material.id;
    outCommand.meshVersion = resources.meshVersion(renderable.instance->mesh);
    outCommand.vertexData = mesh.vertices.data();
    outCommand.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    outCommand.vertexStrideBytes = static_cast<std::uint32_t>(sizeof(MeshResourceDesc::Vertex));
    outCommand.indexData = mesh.indices.data();
    outCommand.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
    outCommand.modelMatrix = modelMatrix;
    outCommand.viewMatrix = frameView.viewMatrix;
    outCommand.viewProjectionMatrix = frameView.viewProjectionMatrix;
    outCommand.lightViewProjectionMatrix = Diligent::float4x4::Identity();
    for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        outCommand.lightViewProjectionMatrices[i] = frameView.lightViewProjectionMatrices[i];
        outCommand.cascadeSplits[i] = frameView.cascadeSplits[i];
    }
    if (frameView.shadowCascadeCount > 0)
    {
        outCommand.lightViewProjectionMatrix = frameView.lightViewProjectionMatrices[0];
    }
    outCommand.normalMatrix = normalMatrixFromModelMatrix(modelMatrix);
    outCommand.cameraPosition = frameView.cameraWorldPosition;
    outCommand.shadowCascadeCount = static_cast<float>(frameView.shadowCascadeCount);
    outCommand.shadowMapInvSizeX = frameView.shadowMapInvSizeX;
    outCommand.shadowMapInvSizeY = frameView.shadowMapInvSizeY;
    outCommand.material.baseColor = material.baseColor;
    outCommand.material.metallic = material.metallic;
    outCommand.material.roughness = material.roughness;
    outCommand.material.opacity = clamp01(material.opacity);
    outCommand.material.alphaCutoff = clamp01(material.pipeline.alphaCutoff);
    outCommand.material.receivesShadows = material.receivesShadows ? 1.0f : 0.0f;
    outCommand.light = light;
    return true;
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

struct CameraFrustumInfo
{
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::float3 right{1.0f, 0.0f, 0.0f};
    Diligent::float3 up{0.0f, 1.0f, 0.0f};
    Diligent::float3 forward{0.0f, 0.0f, 1.0f};
    float aspect = 1.0f;
    float fovRadians = 60.0f * kDegreesToRadians;
    float nearPlane = 0.01f;
    float farPlane = 1000.0f;
    Diligent::float4x4 viewMatrix = Diligent::float4x4::Identity();
    Diligent::float4x4 projectionMatrix = Diligent::float4x4::Identity();
};

CameraFrustumInfo buildCameraFrustumInfo(const CameraData& camera, float outputWidth, float outputHeight)
{
    CameraFrustumInfo info{};
    info.position = camera.worldTransform.position;
    info.aspect = clampPositive(outputWidth, 1.0f) / clampPositive(outputHeight, 1.0f);
    info.fovRadians = std::max(camera.verticalFovDegrees, 1.0f) * kDegreesToRadians;
    info.nearPlane = std::max(camera.nearClip, 0.001f);
    info.farPlane = std::max(camera.farClip, info.nearPlane + 0.001f);

    const Diligent::QuaternionF rotation = normalizeQuaternion(camera.worldTransform.rotation);
    info.forward = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f}), Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 worldUp = safeNormalize(rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f}), Diligent::float3{0.0f, 1.0f, 0.0f});
    info.right = safeNormalize(Diligent::cross(worldUp, info.forward), Diligent::float3{1.0f, 0.0f, 0.0f});
    info.up = safeNormalize(Diligent::cross(info.forward, info.right), Diligent::float3{0.0f, 1.0f, 0.0f});
    info.viewMatrix = viewMatrixFromCameraTransform(camera.worldTransform);
    info.projectionMatrix = Diligent::float4x4::Projection(info.fovRadians, info.aspect, info.nearPlane, info.farPlane, false);
    return info;
}

void computeCascadeSplits(float nearPlane, float farPlane, std::array<float, kShadowCascadeCount>& outSplits)
{
    for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascadeCount);
        const float logarithmic = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniform = nearPlane + (farPlane - nearPlane) * p;
        outSplits[i] = kCascadeSplitLambda * logarithmic + (1.0f - kCascadeSplitLambda) * uniform;
    }
}

std::array<Diligent::float3, 8> buildFrustumCornersForRange(
    const CameraFrustumInfo& cameraInfo,
    float splitNear,
    float splitFar)
{
    const float tanHalfFov = std::tan(cameraInfo.fovRadians * 0.5f);
    const float nearHalfHeight = tanHalfFov * splitNear;
    const float nearHalfWidth = nearHalfHeight * cameraInfo.aspect;
    const float farHalfHeight = tanHalfFov * splitFar;
    const float farHalfWidth = farHalfHeight * cameraInfo.aspect;

    const Diligent::float3 nearCenter = cameraInfo.position + cameraInfo.forward * splitNear;
    const Diligent::float3 farCenter = cameraInfo.position + cameraInfo.forward * splitFar;

    return {
        nearCenter - cameraInfo.right * nearHalfWidth - cameraInfo.up * nearHalfHeight,
        nearCenter + cameraInfo.right * nearHalfWidth - cameraInfo.up * nearHalfHeight,
        nearCenter + cameraInfo.right * nearHalfWidth + cameraInfo.up * nearHalfHeight,
        nearCenter - cameraInfo.right * nearHalfWidth + cameraInfo.up * nearHalfHeight,
        farCenter - cameraInfo.right * farHalfWidth - cameraInfo.up * farHalfHeight,
        farCenter + cameraInfo.right * farHalfWidth - cameraInfo.up * farHalfHeight,
        farCenter + cameraInfo.right * farHalfWidth + cameraInfo.up * farHalfHeight,
        farCenter - cameraInfo.right * farHalfWidth + cameraInfo.up * farHalfHeight};
}

Diligent::float4x4 buildDirectionalLightCascadeViewProjection(
    const Diligent::float3& lightDirection,
    const std::array<Diligent::float3, 8>& cascadeCorners)
{
    Diligent::float3 cascadeCenter{0.0f, 0.0f, 0.0f};
    for (const Diligent::float3& corner : cascadeCorners)
    {
        cascadeCenter += corner;
    }
    cascadeCenter /= static_cast<float>(cascadeCorners.size());

    float radius = 1.0f;
    for (const Diligent::float3& corner : cascadeCorners)
    {
        radius = std::max(radius, Diligent::length(corner - cascadeCenter));
    }
    radius = std::ceil(radius * kCascadeStabilization) / kCascadeStabilization;

    const Diligent::float3 upCandidate =
        std::abs(Diligent::dot(lightDirection, Diligent::float3{0.0f, 1.0f, 0.0f})) > 0.98f ?
            Diligent::float3{0.0f, 0.0f, 1.0f} :
            Diligent::float3{0.0f, 1.0f, 0.0f};

    const Diligent::float3 lightPosition = cascadeCenter - lightDirection * (radius * 2.0f + kCascadeDepthPadding);
    const Diligent::float4x4 lightView = lookAtMatrix(lightPosition, cascadeCenter, upCandidate);

    const Diligent::float4 lightSpaceCenter4 = Diligent::float4{cascadeCenter, 1.0f} * lightView;
    const float texelWorldSize = (radius * 2.0f) / static_cast<float>(kShadowMapResolution);
    float snappedCenterX = lightSpaceCenter4.x;
    float snappedCenterY = lightSpaceCenter4.y;
    if (texelWorldSize > 0.0f)
    {
        snappedCenterX = std::floor(snappedCenterX / texelWorldSize + 0.5f) * texelWorldSize;
        snappedCenterY = std::floor(snappedCenterY / texelWorldSize + 0.5f) * texelWorldSize;
    }

    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const Diligent::float3& corner : cascadeCorners)
    {
        const Diligent::float4 lightSpaceCorner = Diligent::float4{corner, 1.0f} * lightView;
        minZ = std::min(minZ, lightSpaceCorner.z);
        maxZ = std::max(maxZ, lightSpaceCorner.z);
    }

    // Use a fixed XY extent from the frustum slice's bounding sphere for stable cascades.
    const float left = snappedCenterX - radius;
    const float right = snappedCenterX + radius;
    const float bottom = snappedCenterY - radius;
    const float top = snappedCenterY + radius;
    const float nearPlane = std::max(0.1f, minZ - kCascadeDepthPadding);
    const float farPlane = std::max(nearPlane + 1.0f, maxZ + kCascadeDepthPadding);

    const Diligent::float4x4 lightProjection =
        Diligent::float4x4::OrthoOffCenter(left, right, bottom, top, nearPlane, farPlane, false);
    return lightView * lightProjection;
}

void populateDirectionalLightCascades(
    const CameraFrustumInfo& cameraInfo,
    const ForwardDirectionalLightData& lightData,
    FrameViewData& outFrameView)
{
    outFrameView.hasDirectionalLight = lightData.intensity > 0.0f && Diligent::dot(lightData.direction, lightData.direction) > 1.0e-6f;
    if (!outFrameView.hasDirectionalLight)
    {
        outFrameView.shadowCascadeCount = 0;
        outFrameView.shadowMapInvSizeX = 0.0f;
        outFrameView.shadowMapInvSizeY = 0.0f;
        for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i)
        {
            outFrameView.lightViewProjectionMatrices[i] = Diligent::float4x4::Identity();
            outFrameView.cascadeSplits[i] = cameraInfo.farPlane;
            Diligent::ExtractViewFrustumPlanesFromMatrix(
                outFrameView.lightViewProjectionMatrices[i],
                outFrameView.lightFrustums[i],
                false);
        }
        return;
    }

    const Diligent::float3 lightDirection = Diligent::normalize(lightData.direction);
    const float shadowFarDistance = std::min(
        cameraInfo.farPlane,
        std::max(lightData.shadowDistance, cameraInfo.nearPlane + 0.001f));
    computeCascadeSplits(cameraInfo.nearPlane, shadowFarDistance, outFrameView.cascadeSplits);
    outFrameView.shadowCascadeCount = kShadowCascadeCount;
    outFrameView.shadowMapInvSizeX = 1.0f / static_cast<float>(kShadowMapResolution);
    outFrameView.shadowMapInvSizeY = 1.0f / static_cast<float>(kShadowMapResolution);

    float splitNear = cameraInfo.nearPlane;
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        const float splitFar = outFrameView.cascadeSplits[cascadeIdx];
        const auto cascadeCorners = buildFrustumCornersForRange(cameraInfo, splitNear, splitFar);
        outFrameView.lightViewProjectionMatrices[cascadeIdx] =
            buildDirectionalLightCascadeViewProjection(lightDirection, cascadeCorners);
        Diligent::ExtractViewFrustumPlanesFromMatrix(
            outFrameView.lightViewProjectionMatrices[cascadeIdx],
            outFrameView.lightFrustums[cascadeIdx],
            false);
        splitNear = splitFar;
    }
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
        std::uint32_t shadowCascadeMask = 0;
        if (frameView.hasDirectionalLight)
        {
            for (std::uint32_t cascadeIdx = 0; cascadeIdx < frameView.shadowCascadeCount; ++cascadeIdx)
            {
                if (isVisibleByFrustum(renderable, frameView.lightFrustums[cascadeIdx]))
                {
                    shadowCascadeMask |= (1u << cascadeIdx);
                }
            }
        }
        const bool lightVisible = shadowCascadeMask != 0;

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
        queuedDraw.mainPassClass = transparent ? MainPassClass::ForwardTransparent : MainPassClass::ForwardOpaque;
        queuedDraw.shadowCascadeMask = shadowCascadeMask;
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
        if (lhs.drawCommand.programFamily != rhs.drawCommand.programFamily)
        {
            return static_cast<std::uint32_t>(lhs.drawCommand.programFamily) < static_cast<std::uint32_t>(rhs.drawCommand.programFamily);
        }
        if (lhs.drawCommand.materialFeatureFlags != rhs.drawCommand.materialFeatureFlags)
        {
            return lhs.drawCommand.materialFeatureFlags < rhs.drawCommand.materialFeatureFlags;
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
    const ForwardDirectionalLightData& lightData)
{
    FrameViewData frameView{};
    frameView.target = target;
    frameView.viewport = viewport;
    frameView.outputWidth = targetDesc.width;
    frameView.outputHeight = targetDesc.height;

    const CameraFrustumInfo cameraInfo = buildCameraFrustumInfo(
        camera,
        static_cast<float>(targetDesc.width),
        static_cast<float>(targetDesc.height));
    frameView.viewMatrix = cameraInfo.viewMatrix;
    frameView.viewProjectionMatrix = cameraInfo.viewMatrix * cameraInfo.projectionMatrix;
    frameView.cameraWorldPosition = cameraPosition(camera);
    Diligent::ExtractViewFrustumPlanesFromMatrix(frameView.viewProjectionMatrix, frameView.viewFrustum, false);
    populateDirectionalLightCascades(cameraInfo, lightData, frameView);
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

        RenderTargetDesc targetDesc{};
        if (!mDevice.tryGetRenderTargetDesc(target, targetDesc))
        {
            return;
        }

        const FrameViewData frameView = buildFrameViewData(camera, targetDesc, target, viewport, lightData);
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
