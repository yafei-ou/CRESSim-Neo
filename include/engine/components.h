#ifndef CRESSIM_NEO_ENGINE_COMPONENTS_H
#define CRESSIM_NEO_ENGINE_COMPONENTS_H

#include "common/math_types.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"

#include <cstdint>

namespace cressim::neo::engine
{

struct ColliderHandle
{
    physics::ColliderId id = physics::kInvalidColliderId;

    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

struct TransformComponent
{
    common::Transform worldTransform{};
};

struct MeshRendererComponent
{
    graphics::MeshHandle mesh{};
    graphics::MaterialHandle material{};
    bool visible = true;
};

struct CameraComponent
{
    float verticalFovDegrees = 60.0f;
    float nearClip           = 0.01f;
    float farClip            = 1000.0f;

    // ManagedPrimary renders into the renderer-managed primary layered surface for presentation.
    // ExplicitSurface renders directly into the bound target layer. Binding a non-array target
    // still means every camera targeting it shares layer 0.
    gpu::CameraOutputBinding output{};
    // Optional per-camera output resize request (0 keeps current target size).
    std::uint32_t outputWidth  = 0;
    std::uint32_t outputHeight = 0;
    // Normalized viewport on the chosen output target.
    // Renderer support is intentionally narrow: sub-rect viewport rendering is only honored for
    // ExplicitSurface cameras targeting non-layered render targets. ManagedPrimary and layered
    // targets still render to the full target.
    // Clears are still whole-target clears, not viewport-scoped clears, so viewport users should
    // disable clearColor / clearDepth when preserving surrounding pixels matters.
    gpu::GpuRenderViewport viewport{};
    bool clearColor = true;
    bool clearDepth = true;
    // This value should follow the color-space semantics of the target path in use.
    // For example, HDR/scene-linear targets expect linear values.
    Diligent::float4 clearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    float clearDepthValue = 1.0f;

    // Cameras are rendered in ascending order.
    std::uint32_t renderOrder = 0;
};

struct DirectionalLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity          = 1.0f;
    float range              = 0.0f;
    float shadowDistance     = 120.0f;
    float shadowFadeDistance = 20.0f;
    float shadowBias         = 0.0015f;
    bool castsShadows        = true;
};

struct PointLightComponent
{
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity   = 1.0f;
    float range       = 10.0f;
    float shadowBias  = 0.0015f;
    bool castsShadows = false;
};

struct SpotLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity      = 1.0f;
    float range          = 10.0f;
    float innerConeAngle = 25.0f;
    float outerConeAngle = 35.0f;
    float shadowBias     = 0.0015f;
    bool castsShadows    = false;
};

struct RigidBodyComponent
{
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};
    physics::RigidBodyType bodyType = physics::RigidBodyType::Dynamic;
    float inverseMass               = 1.0f;
    Diligent::float3 kinematicTargetPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF kinematicTargetRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool kinematicTargetEnabled = false;
    bool simulated              = true;
};

struct ColliderComponent
{
    physics::ColliderShapeType shapeType = physics::ColliderShapeType::Sphere;
    Diligent::float4 shapeParams{0.5f, 0.0f, 0.0f, 0.0f};
    Diligent::float3 localPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool enabled = true;

    float friction               = 0.0f;
    float restitution            = 0.0f;
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_COMPONENTS_H
