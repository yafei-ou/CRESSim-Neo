#ifndef CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H
#define CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H

#include "graphics/host_scene.h"
#include "graphics/render_resource_manager.h"
#include "graphics/renderer.h"
#include "graphics/renderer/passes/render_pass_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/AdvancedMath.hpp"

#include <unordered_map>
#include <vector>

namespace cressim::neo::graphics::detail
{

struct PreparedRenderable
{
    const RenderableInstance* instance   = nullptr;
    const MeshResourceDesc* mesh         = nullptr;
    const MaterialResourceDesc* material = nullptr;
    std::uint32_t instanceIndex          = 0xffffffffu;
    common::Transform worldTransform{};
    Diligent::float4x4 modelMatrix  = Diligent::float4x4::Identity();
    Diligent::float4x4 normalMatrix = Diligent::float4x4::Identity();
    bool hasWorldBounds             = false;
    Diligent::BoundBox worldBounds{};
};

gpu::GpuRenderViewport normalizeViewport(const gpu::GpuRenderViewport& viewport);
Diligent::float4x4 worldMatrixFromTransform(const common::Transform& transform);
Diligent::float4x4 normalMatrixFromModelMatrix(const Diligent::float4x4& modelMatrix);
ForwardDirectionalLightData buildMainLight(const std::vector<DirectionalLightData>& lights);
std::vector<PreparedRenderable> buildPreparedRenderables(
    const std::vector<RenderableInstance>& renderables, const RenderResourceManager& resources,
    const std::unordered_map<common::EntityId, std::uint32_t>& gpuPoseIndices);
bool isVisibleByFrustum(const PreparedRenderable& renderable, const Diligent::ViewFrustum& frustum);
FrameViewData buildFrameViewData(const CameraData& camera,
                                 const gpu::GpuRenderTargetDesc& targetDesc,
                                 gpu::GpuRenderTargetHandle target,
                                 const gpu::GpuRenderViewport& viewport,
                                 const ForwardDirectionalLightData& lightData);
CameraRenderQueues buildCameraRenderQueues(
    const std::vector<PreparedRenderable>& preparedRenderables, const FrameViewData& frameView,
    const RenderResourceManager& resources, RenderStats& stats);
CameraData defaultCamera();
std::vector<CameraData> sortedCameras(const HostSceneView& sceneView);

} // namespace cressim::neo::graphics::detail

#endif // CRESSIM_NEO_GRAPHICS_RENDERER_INTERNAL_H
