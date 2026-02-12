#ifndef CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H
#define CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H

#include "common/id.h"
#include "common/math_types.h"
#include "graphics/export.h"
#include "graphics/render_resource_manager.h"

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
    float nearClip = 0.01f;
    float farClip = 1000.0f;
};

struct DirectionalLightData
{
    common::EntityId entityId = common::kInvalidEntityId;
    common::Vec3f direction{0.0f, -1.0f, 0.0f};
    common::Vec3f color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

class CRESSIM_NEO_GRAPHICS_API RenderWorld
{
public:
    void clear();

    void upsertRenderable(const RenderableInstance& instance);
    void upsertCamera(const CameraData& camera);
    void upsertDirectionalLight(const DirectionalLightData& light);

    const std::vector<RenderableInstance>& renderables() const noexcept;
    const std::vector<CameraData>& cameras() const noexcept;
    const std::vector<DirectionalLightData>& directionalLights() const noexcept;

private:
    std::vector<RenderableInstance> mRenderables;
    std::vector<CameraData> mCameras;
    std::vector<DirectionalLightData> mDirectionalLights;
};

} // namespace cressim::neo::graphics

#endif // CRESSIM_NEO_GRAPHICS_RENDER_WORLD_H
