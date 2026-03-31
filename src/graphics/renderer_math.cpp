#include "graphics/renderer_internal.h"

#include <algorithm>
#include <cmath>

namespace cressim::neo::graphics::detail
{

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
    camera.clearColorValue    = Diligent::float4{0.0f, 0.0f, 0.0f, 1.0f};
    camera.clearDepthValue    = 1.0f;
    return camera;
}

Diligent::float3 normalizeOrFallback(const Diligent::float3 &value,
                                     const Diligent::float3 &fallback)
{
    const float lengthSq = dot3(value, value);
    if (lengthSq <= 1.0e-8f)
    {
        return fallback;
    }

    const float invLength = 1.0f / std::sqrt(lengthSq);
    return Diligent::float3{value.x * invLength, value.y * invLength, value.z * invLength};
}

float dot3(const Diligent::float3 &lhs, const Diligent::float3 &rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Diligent::float4x4 buildLookAtMatrix(const Diligent::float3 &eye, const Diligent::float3 &target,
                                     const Diligent::float3 &up)
{
    const Diligent::float3 zaxis = normalizeOrFallback(target - eye, Diligent::float3{0, 0, 1});
    const Diligent::float3 xaxis =
        normalizeOrFallback(Diligent::cross(up, zaxis), Diligent::float3{1, 0, 0});
    const Diligent::float3 yaxis = Diligent::cross(zaxis, xaxis);

    Diligent::float4x4 result = Diligent::float4x4::Identity();
    result._11                = xaxis.x;
    result._21                = xaxis.y;
    result._31                = xaxis.z;
    result._12                = yaxis.x;
    result._22                = yaxis.y;
    result._32                = yaxis.z;
    result._13                = zaxis.x;
    result._23                = zaxis.y;
    result._33                = zaxis.z;
    result._41                = -dot3(xaxis, eye);
    result._42                = -dot3(yaxis, eye);
    result._43                = -dot3(zaxis, eye);
    return result;
}

std::vector<CameraData> sortedCameras(const HostSceneView &sceneView)
{
    std::vector<CameraData> cameras;
    if (sceneView.cameras != nullptr)
    {
        cameras.reserve(sceneView.cameras->size());
        for (const CameraData &camera : *sceneView.cameras)
        {
            if (camera.entityId == common::kInvalidEntityId || camera.cameraSlot == 0xffffffffu)
            {
                continue;
            }
            cameras.push_back(camera);
        }
    }
    std::sort(cameras.begin(), cameras.end(),
              [](const CameraData &lhs, const CameraData &rhs)
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
