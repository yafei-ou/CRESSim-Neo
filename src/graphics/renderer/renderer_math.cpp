#include "graphics/renderer/renderer_internal.h"

#include "common/math_utils_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace cressim::neo::graphics::detail
{

namespace
{

constexpr float kCascadeSplitLambda   = 0.85f;
constexpr float kCascadeStabilization = 16.0f;
constexpr float kCascadeDepthPadding  = 16.0f;

Diligent::float4x4 viewMatrixFromCameraTransform(const common::Transform& cameraTransform)
{
    const Diligent::QuaternionF rotation =
        common::runtime_math::normalizeQuaternion(cameraTransform.rotation);
    const Diligent::float3 eye   = cameraTransform.position;
    const Diligent::float3 zAxis = common::runtime_math::safeNormalize(
        rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f}),
        Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 up = common::runtime_math::safeNormalize(
        rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f}),
        Diligent::float3{0.0f, 1.0f, 0.0f});
    const Diligent::float3 xAxis = common::runtime_math::safeNormalize(
        Diligent::cross(up, zAxis), Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis = common::runtime_math::safeNormalize(
        Diligent::cross(zAxis, xAxis), Diligent::float3{0.0f, 1.0f, 0.0f});

    const Diligent::float4x4 viewRotation = Diligent::float4x4::ViewFromBasis(xAxis, yAxis, zAxis);
    const Diligent::float4x4 viewTranslation = Diligent::float4x4::Translation(-eye);
    return viewTranslation * viewRotation;
}

Diligent::float4x4 lookAtMatrix(const Diligent::float3& eye, const Diligent::float3& at,
                                const Diligent::float3& up)
{
    const Diligent::float3 zAxis =
        common::runtime_math::safeNormalize(at - eye, Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 xAxis = common::runtime_math::safeNormalize(
        Diligent::cross(up, zAxis), Diligent::float3{1.0f, 0.0f, 0.0f});
    const Diligent::float3 yAxis = common::runtime_math::safeNormalize(
        Diligent::cross(zAxis, xAxis), Diligent::float3{0.0f, 1.0f, 0.0f});
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
    float aspect                        = 1.0f;
    float fovRadians                    = common::runtime_math::degreesToRadians(60.0f);
    float nearPlane                     = 0.01f;
    float farPlane                      = 1000.0f;
    Diligent::float4x4 viewMatrix       = Diligent::float4x4::Identity();
    Diligent::float4x4 projectionMatrix = Diligent::float4x4::Identity();
};

CameraFrustumInfo buildCameraFrustumInfo(const CameraData& camera, float outputWidth,
                                         float outputHeight)
{
    CameraFrustumInfo info{};
    info.position = camera.worldTransform.position;
    info.aspect   = common::runtime_math::clampPositive(outputWidth, 1.0f) /
                  common::runtime_math::clampPositive(outputHeight, 1.0f);
    info.fovRadians =
        std::max(camera.verticalFovDegrees, 1.0f) * common::runtime_math::degreesToRadians(1.0f);
    info.nearPlane = std::max(camera.nearClip, 0.001f);
    info.farPlane  = std::max(camera.farClip, info.nearPlane + 0.001f);

    const Diligent::QuaternionF rotation =
        common::runtime_math::normalizeQuaternion(camera.worldTransform.rotation);
    info.forward = common::runtime_math::safeNormalize(
        rotation.RotateVector(Diligent::float3{0.0f, 0.0f, 1.0f}),
        Diligent::float3{0.0f, 0.0f, 1.0f});
    const Diligent::float3 worldUp = common::runtime_math::safeNormalize(
        rotation.RotateVector(Diligent::float3{0.0f, 1.0f, 0.0f}),
        Diligent::float3{0.0f, 1.0f, 0.0f});
    info.right      = common::runtime_math::safeNormalize(Diligent::cross(worldUp, info.forward),
                                                          Diligent::float3{1.0f, 0.0f, 0.0f});
    info.up         = common::runtime_math::safeNormalize(Diligent::cross(info.forward, info.right),
                                                          Diligent::float3{0.0f, 1.0f, 0.0f});
    info.viewMatrix = viewMatrixFromCameraTransform(camera.worldTransform);
    info.projectionMatrix = Diligent::float4x4::Projection(info.fovRadians, info.aspect,
                                                           info.nearPlane, info.farPlane, false);
    return info;
}

void computeCascadeSplits(float nearPlane, float farPlane,
                          std::array<float, kShadowCascadeCount>& outSplits)
{
    for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i)
    {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascadeCount);
        const float logarithmic = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniform     = nearPlane + (farPlane - nearPlane) * p;
        outSplits[i] = kCascadeSplitLambda * logarithmic + (1.0f - kCascadeSplitLambda) * uniform;
    }
}

std::array<Diligent::float3, 8> buildFrustumCornersForRange(const CameraFrustumInfo& cameraInfo,
                                                            float splitNear, float splitFar)
{
    const float tanHalfFov     = std::tan(cameraInfo.fovRadians * 0.5f);
    const float nearHalfHeight = tanHalfFov * splitNear;
    const float nearHalfWidth  = nearHalfHeight * cameraInfo.aspect;
    const float farHalfHeight  = tanHalfFov * splitFar;
    const float farHalfWidth   = farHalfHeight * cameraInfo.aspect;

    const Diligent::float3 nearCenter = cameraInfo.position + cameraInfo.forward * splitNear;
    const Diligent::float3 farCenter  = cameraInfo.position + cameraInfo.forward * splitFar;

    return {nearCenter - cameraInfo.right * nearHalfWidth - cameraInfo.up * nearHalfHeight,
            nearCenter + cameraInfo.right * nearHalfWidth - cameraInfo.up * nearHalfHeight,
            nearCenter + cameraInfo.right * nearHalfWidth + cameraInfo.up * nearHalfHeight,
            nearCenter - cameraInfo.right * nearHalfWidth + cameraInfo.up * nearHalfHeight,
            farCenter - cameraInfo.right * farHalfWidth - cameraInfo.up * farHalfHeight,
            farCenter + cameraInfo.right * farHalfWidth - cameraInfo.up * farHalfHeight,
            farCenter + cameraInfo.right * farHalfWidth + cameraInfo.up * farHalfHeight,
            farCenter - cameraInfo.right * farHalfWidth + cameraInfo.up * farHalfHeight};
}

Diligent::float4x4 buildDirectionalLightCascadeViewProjection(
    const Diligent::float3& lightDirection, const std::array<Diligent::float3, 8>& cascadeCorners)
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
        std::abs(Diligent::dot(lightDirection, Diligent::float3{0.0f, 1.0f, 0.0f})) > 0.98f
            ? Diligent::float3{0.0f, 0.0f, 1.0f}
            : Diligent::float3{0.0f, 1.0f, 0.0f};

    const Diligent::float3 lightPosition =
        cascadeCenter - lightDirection * (radius * 2.0f + kCascadeDepthPadding);
    const Diligent::float4x4 lightView = lookAtMatrix(lightPosition, cascadeCenter, upCandidate);

    const Diligent::float4 lightSpaceCenter4 = Diligent::float4{cascadeCenter, 1.0f} * lightView;
    const float texelWorldSize = (radius * 2.0f) / static_cast<float>(kShadowMapResolution);
    float snappedCenterX       = lightSpaceCenter4.x;
    float snappedCenterY       = lightSpaceCenter4.y;
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
        minZ                                    = std::min(minZ, lightSpaceCorner.z);
        maxZ                                    = std::max(maxZ, lightSpaceCorner.z);
    }

    const float left      = snappedCenterX - radius;
    const float right     = snappedCenterX + radius;
    const float bottom    = snappedCenterY - radius;
    const float top       = snappedCenterY + radius;
    const float nearPlane = std::max(0.1f, minZ - kCascadeDepthPadding);
    const float farPlane  = std::max(nearPlane + 1.0f, maxZ + kCascadeDepthPadding);

    const Diligent::float4x4 lightProjection =
        Diligent::float4x4::OrthoOffCenter(left, right, bottom, top, nearPlane, farPlane, false);
    return lightView * lightProjection;
}

void populateDirectionalLightCascades(const CameraFrustumInfo& cameraInfo,
                                      const ForwardDirectionalLightData& lightData,
                                      FrameViewData& outFrameView)
{
    outFrameView.hasDirectionalLight =
        lightData.intensity > 0.0f &&
        Diligent::dot(lightData.direction, lightData.direction) > 1.0e-6f;
    if (!outFrameView.hasDirectionalLight)
    {
        outFrameView.shadowCascadeCount = 0;
        outFrameView.shadowMapInvSizeX  = 0.0f;
        outFrameView.shadowMapInvSizeY  = 0.0f;
        for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i)
        {
            outFrameView.lightViewProjectionMatrices[i] = Diligent::float4x4::Identity();
            outFrameView.cascadeSplits[i]               = cameraInfo.farPlane;
            Diligent::ExtractViewFrustumPlanesFromMatrix(
                outFrameView.lightViewProjectionMatrices[i], outFrameView.lightFrustums[i], false);
        }
        return;
    }

    const Diligent::float3 lightDirection = Diligent::normalize(lightData.direction);
    const float shadowFarDistance         = std::min(
        cameraInfo.farPlane, std::max(lightData.shadowDistance, cameraInfo.nearPlane + 0.001f));
    computeCascadeSplits(cameraInfo.nearPlane, shadowFarDistance, outFrameView.cascadeSplits);
    outFrameView.shadowCascadeCount = kShadowCascadeCount;
    outFrameView.shadowMapInvSizeX  = 1.0f / static_cast<float>(kShadowMapResolution);
    outFrameView.shadowMapInvSizeY  = 1.0f / static_cast<float>(kShadowMapResolution);

    float splitNear = cameraInfo.nearPlane;
    for (std::uint32_t cascadeIdx = 0; cascadeIdx < kShadowCascadeCount; ++cascadeIdx)
    {
        const float splitFar      = outFrameView.cascadeSplits[cascadeIdx];
        const auto cascadeCorners = buildFrustumCornersForRange(cameraInfo, splitNear, splitFar);
        outFrameView.lightViewProjectionMatrices[cascadeIdx] =
            buildDirectionalLightCascadeViewProjection(lightDirection, cascadeCorners);
        Diligent::ExtractViewFrustumPlanesFromMatrix(
            outFrameView.lightViewProjectionMatrices[cascadeIdx],
            outFrameView.lightFrustums[cascadeIdx], false);
        splitNear = splitFar;
    }
}

Diligent::float3 cameraPosition(const CameraData& camera)
{
    return camera.worldTransform.position;
}

} // namespace

RenderViewport normalizeViewport(const RenderViewport& viewport)
{
    return common::runtime_math::normalizeViewport(viewport);
}

Diligent::float4x4 worldMatrixFromTransform(const common::Transform& transform)
{
    const Diligent::QuaternionF rotation =
        common::runtime_math::normalizeQuaternion(transform.rotation);
    return Diligent::float4x4::Scale(transform.scale) * rotation.ToMatrix() *
           Diligent::float4x4::Translation(transform.position);
}

Diligent::float4x4 normalMatrixFromModelMatrix(const Diligent::float4x4& modelMatrix)
{
    const auto linear3x3 = Diligent::float3x3{
        Diligent::float3{modelMatrix._11, modelMatrix._12, modelMatrix._13},
        Diligent::float3{modelMatrix._21, modelMatrix._22, modelMatrix._23},
        Diligent::float3{modelMatrix._31, modelMatrix._32, modelMatrix._33},
    };

    const Diligent::float3x3 normal3x3 = linear3x3.Inverse().Transpose();

    return Diligent::float4x4{normal3x3._11, normal3x3._12, normal3x3._13, 0.0f,
                              normal3x3._21, normal3x3._22, normal3x3._23, 0.0f,
                              normal3x3._31, normal3x3._32, normal3x3._33, 0.0f,
                              0.0f,          0.0f,          0.0f,          1.0f};
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
    out.direction                     = light.direction;
    out.color                         = light.color;
    out.intensity                     = light.intensity;
    out.shadowDistance                = light.shadowDistance;
    out.shadowFadeDistance            = light.shadowFadeDistance;
    return out;
}

FrameViewData buildFrameViewData(const CameraData& camera, const RenderTargetDesc& targetDesc,
                                 RenderTargetHandle target, const RenderViewport& viewport,
                                 const ForwardDirectionalLightData& lightData)
{
    FrameViewData frameView{};
    frameView.target       = target;
    frameView.viewport     = viewport;
    frameView.outputWidth  = targetDesc.width;
    frameView.outputHeight = targetDesc.height;
    frameView.light        = lightData;

    const CameraFrustumInfo cameraInfo = buildCameraFrustumInfo(
        camera, static_cast<float>(targetDesc.width), static_cast<float>(targetDesc.height));
    frameView.viewMatrix           = cameraInfo.viewMatrix;
    frameView.viewProjectionMatrix = cameraInfo.viewMatrix * cameraInfo.projectionMatrix;
    frameView.cameraWorldPosition  = cameraPosition(camera);
    Diligent::ExtractViewFrustumPlanesFromMatrix(frameView.viewProjectionMatrix,
                                                 frameView.viewFrustum, false);
    populateDirectionalLightCascades(cameraInfo, lightData, frameView);
    return frameView;
}

CameraData defaultCamera()
{
    CameraData camera{};
    camera.verticalFovDegrees = 60.0f;
    camera.nearClip           = 0.01f;
    camera.farClip            = 1000.0f;
    camera.viewport           = {};
    return camera;
}

std::vector<CameraData> sortedCameras(const RenderWorld& world)
{
    std::vector<CameraData> cameras = world.cameras();
    std::sort(cameras.begin(), cameras.end(),
              [](const CameraData& lhs, const CameraData& rhs)
              {
                  if (lhs.renderOrder != rhs.renderOrder)
                  {
                      return lhs.renderOrder < rhs.renderOrder;
                  }
                  return lhs.entityId < rhs.entityId;
              });
    return cameras;
}

} // namespace cressim::neo::graphics::detail
