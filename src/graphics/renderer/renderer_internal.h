#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H

#include "graphics/host_scene.h"
#include "graphics/renderer.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include <vector>

namespace cressim::neo::graphics::detail
{

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport& viewport);
ForwardDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights);
CameraData defaultCamera();
std::vector<CameraData> sortedCameras(const HostSceneView& sceneView);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H
