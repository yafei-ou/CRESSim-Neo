#ifndef CRESSIM_NEO_ENGINE_COMPONENTS_H
#define CRESSIM_NEO_ENGINE_COMPONENTS_H

#include "common/math_types.h"
#include "gpu/gpu_types.h"
#include "graphics/render_resource_manager.h"
#include "physics/physics_types.h"

#include <cstdint>
#include <vector>

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
    enum class BackgroundMode : std::uint32_t
    {
        ClearColor         = 0u,
        EnvironmentCubemap = 1u,
    };

    float verticalFovDegrees = 60.0f;
    float nearClip           = 0.01f;
    float farClip            = 1000.0f;

    // ManagedPrimary renders into the renderer-managed primary layered surface for presentation.
    // ExplicitSurface renders directly into the bound target layer. Binding a non-array target
    // still means every camera targeting it shares layer 0.
    gpu::RenderOutputBinding output{};
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
    float clearDepthValue         = 1.0f;
    BackgroundMode backgroundMode = BackgroundMode::ClearColor;

    // Cameras are rendered in ascending order.
    std::uint32_t renderOrder = 0;
};

struct DirectionalLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f};
    Diligent::float3 color{1.0f, 1.0f, 1.0f};
    float intensity          = 1.0f;
    float range              = 0.0f;
    float shadowDistance     = 50.0f;
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
    float staticFriction         = -1.0f;
    float restitution            = 0.0f;
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
};

struct SoftBodyComponent
{
    physics::SoftBodySourceDesc source{};
    physics::SoftBodyMaterialDesc material{};
    float particleMass           = 1.0f;
    float particleRadius         = 0.125f;
    float edgeCompliance         = 0.0f;
    float volumeCompliance       = 0.001f;
    bool simulated               = true;
    bool selfCollisionEnabled    = false;
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
};

struct FluidComponent
{
    physics::FluidSourceDesc source{};
    physics::FluidMaterialDesc material{};
    Diligent::float4 visualColor{0.32f, 0.62f, 0.95f, 0.72f};
    float particleMass           = 1.0f;
    float particleRadius         = 0.125f;
    bool simulated               = true;
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
};

struct UltrasoundProbeComponent
{
    bool enabled                       = true;
    std::uint32_t numScanlines         = 16u;
    float lineLength                   = 12.0f;
    float scanlineSpacing              = 0.5f;
    float soundSpeed                   = 1540.0f;
    float worldUnitsPerMeter           = 10.0f;
    float noiseAmplitude               = 0.0f;
    float samplingFrequency            = 100e6f;
    float demodulationFrequency        = 2.5e6f;
    float centerFrequency              = 2.5e6f;
    float fractionalBandwidth          = 0.2f;
    float beamSigmaLateral             = 1.0f;
    float beamSigmaElevational         = 1.0f;
    std::uint32_t radialDecimation     = 4u;
    std::uint32_t threadsPerBlock      = 128u;
    std::uint32_t cudaNumStreams       = 1u;
    std::uint32_t numTimeSamples       = 0u;
    bool useArcProjection              = false;
    bool enablePhaseDelay              = true;
    bool imageEnabled                  = true;
    std::uint32_t imageBaseHeight      = 512u;
    bool imageUseFixedMaxNormalization = false;
    float imageFixedMaxSignal          = 1.0f;
    float imageDynamicRangeDb          = 60.0f;
};

struct UltrasoundScattererSourceComponent
{
    bool enabled                = true;
    float density               = 0.0f;
    float amplitudeMin          = 0.0f;
    float amplitudeMax          = 1.0f;
    float pointDistanceOverride = 0.0f;
};

struct UltrasoundProbeResult
{
    bool valid                        = false;
    bool imageValid                   = false;
    std::uint64_t frameIndex          = 0u;
    std::uint32_t numScanlines        = 0u;
    std::uint32_t samplesPerScanline  = 0u;
    std::uint64_t totalScattererCount = 0u;
    std::uint32_t imageWidth          = 0u;
    std::uint32_t imageHeight         = 0u;
    gpu::GpuRenderTargetHandle imageTarget{};
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_COMPONENTS_H
