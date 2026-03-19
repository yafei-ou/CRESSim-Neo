#include "graphics/renderer/renderer_internal.h"

#include "common/math_utils_runtime.h"

#include <algorithm>

namespace cressim::neo::graphics::detail
{

namespace
{

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

Diligent::float3 cameraPosition(const CameraData& camera)
{
    return camera.worldTransform.position;
}

} // namespace

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport& viewport)
{
    return common::runtime_math::normalizeViewport(viewport);
}

ForwardDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights)
{
    /**
     * This gets the first directional light.
     */

    // TODO: use the brightest one as the main light
    // TODO: allow multiple lights
    // TODO: add other types of lights (spot, point)

    ForwardDirectionalLightData out{};
    for (const DirectionalLightData& light : lights)
    {
        if (light.entityId == common::kInvalidEntityId || light.lightSlot == 0xffffffffu)
        {
            continue;
        }
        out.direction          = light.direction;
        out.color              = light.color;
        out.intensity          = light.intensity;
        out.shadowDistance     = light.shadowDistance;
        out.shadowFadeDistance = light.shadowFadeDistance;
        return out;
    }

    out.direction = Diligent::float3{0.0f, 0.0f, 0.0f};
    out.intensity = 0.0f;
    return out;
}

FrameViewData buildFrameViewData(const CameraData& camera,
                                 const gpu::GpuRenderTargetDesc& targetDesc,
                                 gpu::GpuRenderTargetHandle target,
                                 const gpu::GpuRenderViewport& viewport,
                                 const ForwardDirectionalLightData& lightData)
{
    FrameViewData frameView{};
    frameView.target          = target;
    frameView.viewport        = viewport;
    frameView.clearColor      = camera.clearColor;
    frameView.clearDepth      = camera.clearDepth;
    frameView.clearColorValue = camera.clearColorValue;
    frameView.clearDepthValue = camera.clearDepthValue;
    frameView.envIndex        = camera.envIndex;
    frameView.cameraSlot      = camera.cameraSlot;
    frameView.outputWidth     = targetDesc.width;
    frameView.outputHeight    = targetDesc.height;
    frameView.light           = lightData;

    const CameraFrustumInfo cameraInfo = buildCameraFrustumInfo(
        camera, static_cast<float>(targetDesc.width), static_cast<float>(targetDesc.height));
    frameView.viewMatrix           = cameraInfo.viewMatrix;
    frameView.viewProjectionMatrix = cameraInfo.viewMatrix * cameraInfo.projectionMatrix;
    frameView.cameraWorldPosition  = cameraPosition(camera);
    return frameView;
}

CameraData defaultCamera()
{
    CameraData camera{};
    camera.envIndex           = 0u;
    camera.cameraSlot         = 0u;
    camera.verticalFovDegrees = 60.0f;
    camera.nearClip           = 0.01f;
    camera.farClip            = 1000.0f;
    camera.viewport           = {};
    camera.clearColor         = true;
    camera.clearDepth         = true;
    camera.clearColorValue    = Diligent::float4{0.02f, 0.02f, 0.03f, 1.0f};
    camera.clearDepthValue    = 1.0f;
    return camera;
}

std::vector<CameraData> sortedCameras(const HostSceneView& sceneView)
{
    std::vector<CameraData> cameras;
    if (sceneView.cameras != nullptr)
    {
        cameras.reserve(sceneView.cameras->size());
        for (const CameraData& camera : *sceneView.cameras)
        {
            if (camera.entityId == common::kInvalidEntityId || camera.cameraSlot == 0xffffffffu)
            {
                continue;
            }
            cameras.push_back(camera);
        }
    }
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
