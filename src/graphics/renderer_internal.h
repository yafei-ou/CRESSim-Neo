#ifndef CRESSIM_NEO_GRAPHICS_INTERNAL_H
#define CRESSIM_NEO_GRAPHICS_INTERNAL_H

#include "graphics/host_scene.h"

#include <vector>

namespace cressim::neo::graphics::detail
{

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport &viewport);
CameraData defaultCamera();
std::vector<CameraData> sortedCameras(const HostSceneView &sceneView);
Diligent::float3 normalizeOrFallback(const Diligent::float3 &value,
                                     const Diligent::float3 &fallback);
float dot3(const Diligent::float3 &lhs, const Diligent::float3 &rhs);
Diligent::float4x4 buildLookAtMatrix(const Diligent::float3 &eye, const Diligent::float3 &target,
                                     const Diligent::float3 &up);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_INTERNAL_H
