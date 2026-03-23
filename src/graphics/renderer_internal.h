#ifndef CRESSIM_NEO_GRAPHICS_INTERNAL_H
#define CRESSIM_NEO_GRAPHICS_INTERNAL_H

#include "graphics/host_scene.h"

#include <vector>

namespace cressim::neo::graphics::detail
{

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport &viewport);
CameraData defaultCamera();
std::vector<CameraData> sortedCameras(const HostSceneView &sceneView);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_INTERNAL_H
