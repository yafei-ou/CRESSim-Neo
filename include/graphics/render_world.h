#ifndef CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H
#define CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H

#include "common/id.h"
#include "common/math_types.h"
#include "gpu/gpu_device.h"
#include "graphics/export.h"
#include "graphics/render_resource_manager.h"

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace cressim::neo::graphics
{

struct RenderableInstance
{
    common::EntityId entityId = common::kInvalidEntityId;
    common::Transform worldTransform{};
    MeshHandle mesh{};
    MaterialHandle material{};
};

struct CameraData
{
    common::EntityId entityId = common::kInvalidEntityId;
    common::Transform worldTransform{};
    float verticalFovDegrees = 60.0f;
    float nearClip           = 0.01f;
    float farClip            = 1000.0f;

    // Render output and scheduling controls copied from engine::CameraComponent.
    gpu::GpuRenderTargetHandle outputTarget{};
    std::uint32_t outputWidth  = 0;
    std::uint32_t outputHeight = 0;
    gpu::GpuRenderViewport viewport{};

    std::uint32_t renderOrder = 0;
};

struct DirectionalLightData
{
    common::EntityId entityId = common::kInvalidEntityId;
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity          = 1.0f;
    float shadowDistance     = 120.0f;
    float shadowFadeDistance = 20.0f;
};

class CRESSIM_NEO_GRAPHICS_API RenderWorld
{
public:
    void clear();

    void upsertRenderable(const RenderableInstance& instance);
    void upsertCamera(const CameraData& camera);
    void upsertDirectionalLight(const DirectionalLightData& light);
    bool removeRenderable(common::EntityId entityId);
    bool removeCamera(common::EntityId entityId);
    bool removeDirectionalLight(common::EntityId entityId);

    const std::vector<RenderableInstance>& renderables() const noexcept;
    const std::vector<CameraData>& cameras() const noexcept;
    const std::vector<DirectionalLightData>& directionalLights() const noexcept;

private:
    std::vector<RenderableInstance> mRenderables;
    std::vector<CameraData> mCameras;
    std::vector<DirectionalLightData> mDirectionalLights;

    std::unordered_map<common::EntityId, std::size_t> mRenderableIndices;
    std::unordered_map<common::EntityId, std::size_t> mCameraIndices;
    std::unordered_map<common::EntityId, std::size_t> mDirectionalLightIndices;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H
