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

/// @brief Lightweight handle wrapper for a physics collider instance.
struct ColliderHandle
{
    physics::ColliderId id = physics::kInvalidColliderId; ///< Unique collider ID.

    /// @brief Checks if the collider handle is valid.
    /// @return True if valid, false otherwise.
    bool isValid() const noexcept
    {
        return id != 0u;
    }
};

/// @brief World transform component for spatial positioning, orientation, and scaling.
struct TransformComponent
{
    common::Transform worldTransform{}; ///< 3D world transform matrix/pose.
};

/// @brief Mesh renderer component binding a 3D mesh and material for visual rendering.
struct MeshRendererComponent
{
    graphics::MeshHandle mesh{};        ///< Handle to the geometry mesh resource.
    graphics::MaterialHandle material{};///< Handle to the visual material resource.
    std::uint32_t segmentationId = 0u;  ///< ID for semantic image segmentation passes.
    bool visible                 = true;///< Visibility flag for camera rendering.
};

/// @brief Camera component defining projection, view targets, and rendering modes.
struct CameraComponent
{
    /// @brief Output product modes rendered by the camera.
    enum class Product : std::uint32_t
    {
        ColorDepth        = 0u, ///< Standard RGBA color and depth output.
        Depth             = 1u, ///< Single-channel depth map output.
        SegmentationDepth = 2u, ///< Semantic segmentation mask and depth output.
    };

    /// @brief Background clear modes for camera rendering.
    enum class BackgroundMode : std::uint32_t
    {
        ClearColor         = 0u, ///< Solid clear color background.
        EnvironmentCubemap = 1u, ///< Skybox / Image-Based Lighting cubemap background.
    };

    float verticalFovDegrees = 60.0f;           ///< Vertical Field-of-View in degrees.
    float nearClip           = 0.01f;           ///< Near clipping plane distance.
    float farClip            = 1000.0f;         ///< Far clipping plane distance.
    Product product          = Product::ColorDepth; ///< Rendered camera output product type.

    gpu::RenderOutputBinding output{};          ///< Target render output binding descriptor.
    std::uint32_t outputWidth  = 0;             ///< Optional explicit target width (0 for default).
    std::uint32_t outputHeight = 0;             ///< Optional explicit target height (0 for default).
    gpu::GpuRenderViewport viewport{};          ///< Viewport rectangle on the output target.
    bool clearColor = true;                     ///< Whether to clear target color buffer before rendering.
    bool clearDepth = true;                     ///< Whether to clear target depth buffer before rendering.
    Diligent::float4 clearColorValue{0.0f, 0.0f, 0.0f, 1.0f}; ///< Clear color RGBA values.
    float clearDepthValue         = 1.0f;       ///< Clear depth value.
    BackgroundMode backgroundMode = BackgroundMode::ClearColor; ///< Camera background rendering mode.

    std::uint32_t renderOrder = 0;              ///< Rendering priority order (ascending).
};

/// @brief Directional light source for global scene illumination and shadow mapping.
struct DirectionalLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f}; ///< Light direction vector.
    Diligent::float3 color{1.0f, 1.0f, 1.0f};      ///< Light color RGB values.
    float intensity          = 1.0f;             ///< Illumination intensity multiplier.
    float range              = 0.0f;             ///< Maximum light range (0 for infinite).
    float shadowDistance     = 50.0f;            ///< Maximum shadow rendering distance.
    float shadowFadeDistance = 20.0f;            ///< Distance over which shadows fade out.
    float shadowBias         = 0.0015f;          ///< Shadow depth comparison bias.
    bool castsShadows        = true;             ///< Enable shadow map generation.
};

/// @brief Point light source emitting light uniformly in all directions.
struct PointLightComponent
{
    Diligent::float3 color{1.0f, 1.0f, 1.0f}; ///< Light color RGB values.
    float intensity   = 1.0f;               ///< Illumination intensity multiplier.
    float range       = 10.0f;              ///< Attenuation distance range.
    float shadowBias  = 0.0015f;            ///< Shadow depth bias.
    bool castsShadows = false;             ///< Enable shadow map generation.
};

/// @brief Spot light source emitting a cone of light in a specified direction.
struct SpotLightComponent
{
    Diligent::float3 direction{0.0f, -1.0f, 0.0f}; ///< Spot light emission direction.
    Diligent::float3 color{1.0f, 1.0f, 1.0f};      ///< Light color RGB values.
    float intensity      = 1.0f;                 ///< Illumination intensity multiplier.
    float range          = 10.0f;                ///< Attenuation distance range.
    float innerConeAngle = 25.0f;                ///< Inner full-intensity cone angle (degrees).
    float outerConeAngle = 35.0f;                ///< Outer zero-intensity cone angle (degrees).
    float shadowBias     = 0.0015f;              ///< Shadow depth bias.
    bool castsShadows    = false;               ///< Enable shadow map generation.
};

/// @brief Rigid body component defining linear/angular velocity, mass properties, and kinematic targets.
struct RigidBodyComponent
{
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};   ///< Linear velocity vector.
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};  ///< Angular velocity vector.
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f}; ///< Local inverse inertia tensor diagonal.
    std::vector<Diligent::float3> proxyParticleLocalPositions{}; ///< Local positions of proxy particles.
    physics::ParticleContactMaterialDesc proxyParticleMaterial{}; ///< Proxy particle contact material properties.
    physics::RigidBodyType bodyType   = physics::RigidBodyType::Dynamic; ///< Rigid body type (Dynamic, Static, Kinematic).
    float inverseMass                 = 1.0f;             ///< Inverse mass (1/kg).
    float proxyParticleRadius         = 0.0f;             ///< Radius of proxy collision particles.
    std::uint32_t proxyCollisionLayer = 1u;               ///< Bitmask collision layer for proxy particles.
    std::uint32_t proxyCollisionMask  = 0xffffffffu;        ///< Bitmask collision mask for proxy particles.
    bool suturingEnabled              = false;            ///< Enable surgical suturing needle proxy interactions.
    std::uint32_t needleTipProxyIndex = 0u;               ///< Proxy particle index representing needle tip.
    Diligent::float3 kinematicTargetPosition{0.0f, 0.0f, 0.0f}; ///< Kinematic target position for interpolation.
    Diligent::QuaternionF kinematicTargetRotation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Kinematic target orientation quaternion.
    bool kinematicTargetEnabled = false;                  ///< Enable kinematic target positioning.
};

/// @brief Collider component attached to entities for physical collision queries and dynamics.
struct ColliderComponent
{
    physics::ColliderShapeType shapeType = physics::ColliderShapeType::Sphere; ///< Collision primitive geometry shape type.
    Diligent::float4 shapeParams{0.5f, 0.0f, 0.0f, 0.0f};                     ///< Shape dimensions (e.g. radius, box extents).
    Diligent::float3 localPosition{0.0f, 0.0f, 0.0f};                         ///< Local offset position relative to transform.
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f};              ///< Local offset rotation relative to transform.
    bool enabled = true;                                                       ///< Active collision state.

    float friction               = 0.0f;        ///< Dynamic friction coefficient.
    float staticFriction         = -1.0f;       ///< Static friction coefficient (-1 to reuse dynamic friction).
    float restitution            = 0.0f;        ///< Coefficient of restitution (bounciness).
    std::uint32_t collisionLayer = 1u;          ///< Bitmask collision layer.
    std::uint32_t collisionMask  = 0xffffffffu; ///< Bitmask collision filtering mask.
};

/// @brief Soft body component supporting tetrahedral and meshfree particle sources.
struct SoftBodyComponent
{
    physics::SoftBodySourceDesc source{};             ///< Source mesh file or tetrahedral asset descriptor.
    physics::SoftBodyMaterialDesc material{};         ///< Particle contact material parameters.
    float particleMass           = 1.0f;              ///< Mass per node particle.
    float particleRadius         = 0.125f;            ///< Collision radius per particle.
    float edgeCompliance         = 0.0f;              ///< Extended Position Based Dynamics (XPBD) edge constraint compliance.
    float volumeCompliance       = 0.001f;            ///< XPBD volume conservation constraint compliance.
    bool selfCollisionEnabled    = false;             ///< Enable internal self-collision handling.
    bool supportsSuturing        = false;             ///< Enable surgical thread suturing insertion.
    std::uint32_t collisionLayer = 1u;                 ///< Collision bitmask layer.
    std::uint32_t collisionMask  = 0xffffffffu;        ///< Collision bitmask filter.
};

/// @brief Meshfree / particle-based soft body component for point cloud elastic simulation.
struct MeshfreeSoftBodyComponent
{
    std::vector<Diligent::float3> particles;            ///< Particle rest position list.
    std::vector<Diligent::float3> surfaceRestPositions; ///< Surface mesh rest position coordinates.
    std::vector<Diligent::float3> surfaceNormals;       ///< Surface mesh normal vectors.
    std::vector<Diligent::uint3> surfaceTriangles;      ///< Surface mesh triangle indices.
    std::vector<std::uint32_t> staticParticleIndices;   ///< Particle indices fixed in space.
    physics::SoftBodyMaterialDesc material{};          ///< Material property descriptor.
    float particleRadius         = 0.001f;             ///< Particle radius.
    float particleMass           = 0.001f;             ///< Mass per particle.
    std::uint32_t neighbourCount = 12u;                ///< Particle neighbor interaction count.
    float compliance             = 1.0e-6f;            ///< Constraint compliance parameter.
    bool selfCollisionEnabled    = false;             ///< Enable self-collision.
    std::uint32_t collisionLayer = 1u;                 ///< Collision layer.
    std::uint32_t collisionMask  = 0xffffffffu;        ///< Collision mask.
};

/// @brief 1D elastic strand component for surgical threads and sutures.
struct StrandComponent
{
    physics::StrandMaterialDesc material{};            ///< Material parameters for strand bending/stretching.
    std::vector<Diligent::float3> restPositions{};      ///< Rest positions of strand particles.
    std::vector<std::uint32_t> staticParticleIndices;  ///< Fixed node particle indices.
    float particleMass           = 1.0f;              ///< Mass per strand node particle.
    float particleRadius         = 0.125f;            ///< Collision radius per strand node.
    float stretchShearCompliance = 0.0f;              ///< Stretching and shearing constraint compliance.
    float bendCompliance         = 0.0f;              ///< Bending constraint compliance.
    float twistCompliance        = 0.0f;              ///< Torsional twisting compliance.
    float distanceCompliance     = 0.0f;              ///< Distance constraint compliance.
    Diligent::float3 rootMaterialNormal{0.0f, 1.0f, 0.0f}; ///< Normal vector at strand root constraint.
    bool selfCollisionEnabled    = false;             ///< Enable strand self-collision.
    bool suturingEnabled         = false;             ///< Enable suturing path tracking.
    float pathNodeSpacing        = 0.2f;              ///< Node spacing along suturing path.
    std::uint32_t collisionLayer = 1u;                 ///< Collision layer.
    std::uint32_t collisionMask  = 0xffffffffu;        ///< Collision mask.
};

/// @brief Procedural render component for generating tube meshes along deformable curves (strands/sutures).
struct ProceduralDeformableCurveRenderComponent
{
    physics::ParticleSequenceId sequenceId = physics::kInvalidParticleSequenceId; ///< Target particle sequence ID.
    float radius                           = 0.05f;                             ///< Tube mesh cross-section radius.
    std::uint32_t radialResolution         = 8u;                                ///< Number of radial tube segments.
    bool enabled                           = true;                              ///< Enable rendering.
};

/// @brief Particle-based fluid simulation component (SPH / Position-Based Fluids).
struct FluidComponent
{
    physics::FluidSourceDesc source{};                 ///< Fluid particle emitter/initializer source descriptor.
    physics::FluidMaterialDesc material{};             ///< Fluid material properties (viscosity, surface tension).
    Diligent::float4 visualColor{0.32f, 0.62f, 0.95f, 0.72f}; ///< Visual RGBA color for fluid rendering.
    float particleMass           = 1.0f;              ///< Fluid particle mass.
    float particleRadius         = 0.125f;            ///< Particle interaction radius.
    std::uint32_t collisionLayer = 1u;                 ///< Collision layer.
    std::uint32_t collisionMask  = 0xffffffffu;        ///< Collision mask.
};

/// @brief Ultrasound transducer probe component for simulated B-mode ultrasound imaging.
struct UltrasoundProbeComponent
{
    /// @brief Ultrasound transducer probe array geometry type.
    enum class Geometry : std::uint32_t
    {
        Linear      = 0u, ///< Linear array transducer.
        Curvilinear = 1u, ///< Convex / curvilinear array transducer.
    };

    bool enabled                   = true;              ///< Enable ultrasound beam simulation.
    Geometry geometry              = Geometry::Linear;  ///< Transducer array geometry.
    std::uint32_t numScanlines     = 50u;               ///< Number of acoustic scanlines.
    float lineLength               = 1.2f;              ///< Scanline penetration depth.
    float scanlineSpacing          = 0.01f;             ///< Spacing between adjacent scanlines.
    float sectorAngleDegrees       = 60.0f;             ///< Sector sweep angle for curvilinear arrays.
    float probeRadius              = 0.35f;             ///< Physical transducer head curvature radius.
    float soundSpeed               = 1540.0f;           ///< Acoustic speed of sound in medium (m/s).
    float worldUnitsPerMeter       = 10.0f;             ///< World unit scaling factor per meter.
    float noiseAmplitude           = 0.0f;              ///< Thermal and speckle noise amplitude.
    float samplingFrequency        = 100e6f;            ///< Acoustic RF signal sampling frequency (Hz).
    float demodulationFrequency    = 2.5e6f;            ///< RF demodulation carrier frequency (Hz).
    float centerFrequency          = 2.5e6f;            ///< Transducer center frequency (Hz).
    float fractionalBandwidth      = 0.2f;              ///< Transducer fractional bandwidth.
    float beamSigmaLateral         = 0.001f;            ///< Lateral acoustic beam profile Gaussian sigma.
    float beamSigmaElevational     = 0.001f;            ///< Elevational acoustic beam profile Gaussian sigma.
    std::uint32_t radialDecimation = 4u;                ///< Radial decimation factor for image generation.
    std::uint32_t threadsPerBlock  = 128u;              ///< CUDA compute block thread size.
    std::uint32_t cudaNumStreams   = 1u;                ///< Number of concurrent CUDA streams.
    std::uint32_t numTimeSamples   = 0u;                ///< RF time-domain samples per scanline.
    bool useArcProjection          = false;             ///< Enable arc projection geometry.
    bool enablePhaseDelay          = true;              ///< Enable phase delay beamforming.
};

/// @brief Renderer binding component for ultrasound B-mode image output generation.
struct UltrasoundRendererComponent
{
    bool enabled = true;                                ///< Active rendering state.
    gpu::RenderOutputBinding output{};                  ///< Output target binding descriptor.
    std::uint32_t outputWidth     = 0u;                 ///< Output width in pixels.
    std::uint32_t outputHeight    = 0u;                 ///< Output height in pixels.
    bool useFixedMaxNormalization = false;                ///< Enable fixed maximum signal intensity normalization.
    float fixedMaxSignal          = 1.0f;               ///< Fixed maximum signal normalization value.
};

/// @brief Output layout descriptor for ultrasound probe imaging targets.
struct UltrasoundProbeLayout
{
    std::uint32_t numScanlines           = 0u;          ///< Total scanlines in probe image.
    std::uint32_t samplesPerScanline     = 0u;          ///< Acoustic samples per scanline.
    std::uint32_t imageWidth             = 0u;          ///< Output B-mode image width (pixels).
    std::uint32_t imageHeight            = 0u;          ///< Output B-mode image height (pixels).
    Diligent::TEXTURE_FORMAT colorFormat = Diligent::TEX_FORMAT_RGBA8_UNORM; ///< Target texture format.
    bool layeredOutputSupported          = true;        ///< Support for array/layered render targets.
};

/// @brief Signal amplitude range for ultrasound acoustic scatterer reflections.
struct UltrasoundAmplitudeRange
{
    float minimum = 0.0f; ///< Minimum signal amplitude.
    float maximum = 0.0f; ///< Maximum signal amplitude.
};

/// @brief Scatterer point cloud source for generating tissue ultrasound acoustic backscatter.
struct UltrasoundScattererSourceComponent
{
    bool enabled                = true;        ///< Enable acoustic backscattering.
    float density               = 1000000.0f;  ///< Acoustic scatterer point density per cubic meter.
    float pointDistanceOverride = 0.0f;        ///< Override spacing between scatterer points.
};

/// @brief Authoring particle position container for soft body asset creation.
struct SoftBodyAuthoringParticles
{
    std::uint32_t particleCount = 0u;           ///< Total particle count.
    std::vector<Diligent::float3> restPositions{};///< Rest position coordinate array.
};

/// @brief Execution result containing output handles and metadata for ultrasound probe simulation.
struct UltrasoundProbeResult
{
    bool prepared                     = false; ///< True if metadata/handles are published for current frame.
    bool completed                    = false; ///< True if simulation completed for completedFrameIndex.
    std::uint64_t completedFrameIndex = 0u;    ///< Frame index corresponding to completed ultrasound output.
    std::uint32_t numScanlines        = 0u;    ///< Scanline count in output.
    std::uint32_t samplesPerScanline  = 0u;    ///< Samples per scanline.
    std::uint64_t totalScattererCount = 0u;    ///< Evaluated scatterer count.
    std::uint32_t imageWidth          = 0u;    ///< Generated B-mode image width.
    std::uint32_t imageHeight         = 0u;    ///< Generated B-mode image height.
    gpu::GpuRenderTargetBinding imageBinding{};///< Render target binding for generated B-mode texture.
};

} // namespace cressim::neo::engine

#endif // CRESSIM_NEO_ENGINE_COMPONENTS_H
