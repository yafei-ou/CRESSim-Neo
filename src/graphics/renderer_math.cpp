#include "graphics/renderer_internal.h"

#include "common/math_utils_runtime.h"

#include <algorithm>

namespace cressim::neo::graphics::detail
{

namespace
{

} // namespace

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport &viewport)
{
    return common::runtime_math::normalizeViewport(viewport);
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
