#ifndef CRESSIM_NEO_ENGINE_COMPONENTS_H
#define CRESSIM_NEO_ENGINE_COMPONENTS_H

#include "common/math_types.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"

#include <cstdint>

namespace cressim::neo::engine
{

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

    // If invalid, renderer falls back to device.renderTargets().defaultRenderTarget().
    gpu::GpuRenderTargetHandle outputTarget{};
    // Optional per-camera output resize request (0 keeps current target size).
    std::uint32_t outputWidth  = 0;
    std::uint32_t outputHeight = 0;
    // Normalized viewport on the chosen output target.
    gpu::GpuRenderViewport viewport{};

    // Cameras are rendered in ascending order.
    std::uint32_t renderOrder = 0;
};

struct DirectionalLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity          = 1.0f;
    float shadowDistance     = 120.0f;
    float shadowFadeDistance = 20.0f;
};

struct RigidBodyComponent
{
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};
    float inverseMass                        = 1.0f;
    physics::ColliderShapeType colliderShape = physics::ColliderShapeType::Sphere;
    Diligent::float4 colliderParams{0.5f, 0.0f, 0.0f, 0.0f};
    bool simulated = true;
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_COMPONENTS_H
