#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H

#include "common/id.h"
#include "common/math_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// @file physics_types.h
/// @brief Core physics data structures, collision primitives, constraint descriptors, joints, and SoA storage containers.

namespace cressim::neo::physics
{

/// @brief Geometric primitive shapes supported for rigid-body collision detection.
enum class ColliderShapeType : std::uint32_t
{
    Sphere  = 0u, ///< Spherical collider defined by radius (`shapeParams.x`).
    Box     = 1u, ///< Box collider defined by half-extents (`shapeParams.xyz`).
    Capsule = 2u, ///< Capsule collider defined by radius (`shapeParams.x`) and half-height (`shapeParams.y`).
};

/// @brief Simulation behavior types for rigid bodies.
enum class RigidBodyType : std::uint32_t
{
    Static    = 0u, ///< Immobile obstacle with infinite mass and zero velocity.
    Kinematic = 1u, ///< User-driven or keyframed body unaffected by external forces.
    Dynamic   = 2u, ///< Fully simulated body responding to gravity, forces, and collisions.
};

/// @brief Unique numeric identifier for a rigid body.
using RigidBodyId                         = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated RigidBodyId.
constexpr RigidBodyId kInvalidRigidBodyId = 0u;

/// @brief Unique numeric identifier for a physical collider.
using ColliderId                        = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated ColliderId.
constexpr ColliderId kInvalidColliderId = 0u;

/// @brief Unique numeric identifier for an articulated ball joint.
using BallJointId                         = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated BallJointId.
constexpr BallJointId kInvalidBallJointId = 0u;

/// @brief Unique numeric identifier for an articulated spherical joint.
using SphericalJointId                              = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated SphericalJointId.
constexpr SphericalJointId kInvalidSphericalJointId = 0u;

/// @brief Unique numeric identifier for an articulated hinge joint.
using HingeJointId                          = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated HingeJointId.
constexpr HingeJointId kInvalidHingeJointId = 0u;

/// @brief Unique numeric identifier for an articulated slider (prismatic) joint.
using SliderJointId                           = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated SliderJointId.
constexpr SliderJointId kInvalidSliderJointId = 0u;

/// @brief Unique numeric identifier for a rigid-to-rigid distance constraint.
using RigidDistanceConstraintId                                       = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated RigidDistanceConstraintId.
constexpr RigidDistanceConstraintId kInvalidRigidDistanceConstraintId = 0u;

/// @brief Unique numeric identifier for an inter-particle distance constraint.
using ParticleConstraintId                                  = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated ParticleConstraintId.
constexpr ParticleConstraintId kInvalidParticleConstraintId = 0u;

/// @brief Unique numeric identifier for a rigid-body to particle attachment constraint.
using RigidParticleAttachmentConstraintId = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated RigidParticleAttachmentConstraintId.
constexpr RigidParticleAttachmentConstraintId kInvalidRigidParticleAttachmentConstraintId = 0u;

/// @brief Unique numeric identifier for a strand-to-rigid attachment constraint.
using StrandRigidAttachmentConstraintId = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated StrandRigidAttachmentConstraintId.
constexpr StrandRigidAttachmentConstraintId kInvalidStrandRigidAttachmentConstraintId = 0u;

/// @brief Unique numeric identifier for a routed cable constraint.
using RoutedCableConstraintId                                     = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated RoutedCableConstraintId.
constexpr RoutedCableConstraintId kInvalidRoutedCableConstraintId = 0u;

/// @brief Unique numeric identifier for an authored particle collision filter.
using ParticleCollisionFilterId                                       = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated ParticleCollisionFilterId.
constexpr ParticleCollisionFilterId kInvalidParticleCollisionFilterId = 0u;

/// @brief Unique numeric identifier for an authored particle sequence.
using ParticleSequenceId                                = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated ParticleSequenceId.
constexpr ParticleSequenceId kInvalidParticleSequenceId = 0u;

/// @brief Unique numeric identifier for an authored suturing sequence.
using SuturingSequenceId                                = std::uint32_t;
/// @brief Sentinel constant representing an invalid or unallocated SuturingSequenceId.
constexpr SuturingSequenceId kInvalidSuturingSequenceId = 0u;

/// @brief Actuator drive control modes for articulated rigid joints.
enum class RigidJointDriveMode : std::uint32_t
{
    None              = 0u, ///< Joint drive disabled (passive joint).
    TargetPosition    = 1u, ///< Drive targets a desired joint position or angle.
    TargetVelocity    = 2u, ///< Drive targets a desired angular or linear velocity.
    TargetOrientation = 3u, ///< Drive targets a desired 3D orientation quaternion (spherical joints).
};

/// @brief Generation methods and mesh representations for deformable soft bodies.
enum class SoftBodySourceKind : std::uint32_t
{
    RegularGrid       = 0u, ///< Procedural voxel regular grid.
    TetMesh           = 1u, ///< Explicit tetrahedral mesh vertex array and index buffer.
    TetGenFiles       = 2u, ///< TetGen `.node` and `.ele` file paths.
    MeshfreeParticles = 3u, ///< Meshfree particle point cloud with k-nearest neighbors.
};

/// @brief Initialization source types for fluid simulations.
enum class FluidSourceKind : std::uint32_t
{
    RegularGrid = 0u, ///< Grid-aligned particle emitter block.
};

/// @brief Fundamental particle physics classification.
enum class ParticleKind : std::uint32_t
{
    SoftSolid = 0u, ///< Deformable soft solid, strand, or rigid proxy particle.
    Fluid     = 1u, ///< Incompressible SPH / Position-Based Fluid particle.
};

/// @brief Owner classification for unified simulation particles.
enum class ParticleOwnerType : std::uint32_t
{
    None      = 0u, ///< Unallocated particle slot.
    SoftBody  = 1u, ///< Particle owned by a 3D deformable SoftBody.
    FluidBody = 2u, ///< Particle owned by a FluidBody.
    Strand    = 3u, ///< Particle owned by a 1D elastic Strand.
    RigidBody = 4u, ///< Proxy particle attached to a RigidBody.
};

/// @brief Target owner classification when referencing particles in authored constraints.
enum class AuthoredParticleReferenceType : std::uint32_t
{
    SoftBodyParticle   = 0u, ///< Reference indexes into a soft body's particle array.
    StrandParticle     = 1u, ///< Reference indexes into a strand's particle array.
    RigidProxyParticle = 2u, ///< Reference indexes into a rigid body's proxy particle array.
};

/// @brief Role of a particle in surgical needle and suturing thread sequences.
enum class ParticleStrandRole : std::uint32_t
{
    None       = 0u, ///< Normal strand particle.
    NeedleTip  = 1u, ///< Surgical needle piercing tip node.
    NeedleBody = 2u, ///< Rigid curved surgical needle body node.
    Thread     = 3u, ///< Flexible suture thread trailing node.
};

/// @brief State tracking whether a needle tip or particle is inside a soft-tissue volume.
enum class SuturingInsertionState : std::uint32_t
{
    Outside = 0u, ///< Particle is in free space outside the soft tissue.
    Inside  = 1u, ///< Particle has penetrated into the soft tissue volume.
};

/// @brief Sentinel index indicating an uninitialized or invalid suturing index.
constexpr std::uint32_t kInvalidSuturingIndex = 0xffffffffu;

/// @brief GPU tracking structure representing needle penetration state and barycentric coordinate mapping.
struct GpuSuturingInsertionState
{
    std::uint32_t state               = static_cast<std::uint32_t>(SuturingInsertionState::Outside); ///< Current SuturingInsertionState.
    std::uint32_t softBodyIndex       = kInvalidSuturingIndex;                                       ///< Index of penetrated soft body.
    std::uint32_t tetIndex            = kInvalidSuturingIndex;                                       ///< Index of containing tetrahedron.
    std::uint32_t pathIndex           = kInvalidSuturingIndex;                                       ///< Index of active suturing path.
    std::uint32_t nearestNodeIndex    = kInvalidSuturingIndex;                                       ///< Nearest generated path node index.
    std::uint32_t closestSegmentTBits = 0u;                                                          ///< Encoded parametric t along segment.
    std::uint32_t reserved1           = 0u;                                                          ///< Reserved padding.
    std::uint32_t reserved2           = 0u;                                                          ///< Reserved padding.
    Diligent::float4 barycentrics{0.0f, 0.0f, 0.0f, 0.0f};                                           ///< Barycentric coordinates inside tetrahedron.
};

/// @brief Header metadata for an authored or generated suturing path through soft tissue.
struct SuturingPathHeader
{
    std::uint32_t suturingGroupId = kInvalidSuturingIndex; ///< Identifier for the suturing interaction group.
    std::uint32_t softBodyIndex   = kInvalidSuturingIndex; ///< Index of the pierced soft body.
    std::uint32_t nodeStart       = 0u;                    ///< Start offset into the path node buffer.
    std::uint32_t nodeCount       = 0u;                    ///< Total nodes in this path.
    std::uint32_t flags           = 0u;                    ///< Path status flags.
    std::uint32_t reserved0       = 0u;                    ///< Reserved padding.
    std::uint32_t reserved1       = 0u;                    ///< Reserved padding.
    std::uint32_t reserved2       = 0u;                    ///< Reserved padding.
};

/// @brief Single interpolation node along a punctured suturing path.
struct SuturingPathNode
{
    std::uint32_t softBodyIndex = 0xffffffffu;             ///< Soft body index.
    std::uint32_t tetIndex      = 0xffffffffu;             ///< Containing tetrahedron index.
    Diligent::float4 barycentrics{0.0f, 0.0f, 0.0f, 0.0f}; ///< Interpolated barycentric weights.
    Diligent::float3 tangent{0.0f, 0.0f, 0.0f};            ///< Tangent direction of path at this node.
    float arcLength = 0.0f;                                ///< Cumulative arc-length distance along path.
};

/// @brief Pairing descriptor binding a surgical strand to a deformable soft body for needle puncture tracking.
struct StrandSoftSuturingPair
{
    std::uint32_t suturingGroupId     = kInvalidSuturingIndex; ///< Suturing interaction group ID.
    std::uint32_t softBodyIndex       = kInvalidSuturingIndex; ///< Target soft-body index.
    std::uint32_t strandParticleStart = 0u;                    ///< Start offset of strand particles.
    std::uint32_t strandParticleCount = 0u;                    ///< Number of strand particles.
    std::uint32_t tipParticleIndex    = kInvalidSuturingIndex; ///< Particle index designated as needle tip.
    std::uint32_t softTetStart        = 0u;                    ///< Start offset of soft-body tetrahedra.
    std::uint32_t softTetCount        = 0u;                    ///< Number of soft-body tetrahedra.
    std::uint32_t pathStart           = 0u;                    ///< Offset into path header buffer.
    std::uint32_t pathCount           = 0u;                    ///< Number of paths allocated.
    std::uint32_t nodeStart           = 0u;                    ///< Offset into path node buffer.
    std::uint32_t nodeCount           = 0u;                    ///< Maximum path node capacity.
    std::uint32_t activePathIndex     = kInvalidSuturingIndex; ///< Currently active path index.
    std::uint32_t environmentIndex    = 0u;                    ///< Environment index.
    float pathNodeSpacing             = 0.0f;                  ///< Node sampling interval along path.
    std::uint32_t reserved0           = 0u;                    ///< Reserved padding.
    std::uint32_t reserved1           = 0u;                    ///< Reserved padding.
};

/// @brief Source parameters for generating a regular voxel grid soft body.
struct SoftBodyRegularGridSource
{
    Diligent::float3 size{1.0f, 1.0f, 1.0f};         ///< 3D box extents of the grid volume.
    float targetParticleSpacing = 0.25f;             ///< Desired inter-particle spacing distance.
    std::vector<std::uint32_t> staticParticleIndices;///< Local particle indices fixed in space (infinite mass).
};

/// @brief Source parameters for generating a soft body from an explicit tetrahedral mesh.
struct SoftBodyTetMeshSource
{
    std::vector<Diligent::float3> objectSpaceRestPositions; ///< Rest coordinates of mesh vertices.
    std::vector<std::uint32_t> tetVertexIndices;            ///< 4-tuples of vertex indices defining tetrahedra.
    std::vector<std::uint32_t> staticParticleIndices;       ///< Local particle indices fixed in space.
};

/// @brief Source parameters for loading a tetrahedral soft body from TetGen format files.
struct SoftBodyTetGenSource
{
    std::string nodeFile;                            ///< Path to `.node` vertex coordinate file.
    std::string eleFile;                             ///< Path to `.ele` tetrahedral element file.
    std::vector<std::uint32_t> staticParticleIndices;///< Local particle indices fixed in space.
};

/// @brief Source parameters for meshfree particle-based soft bodies.
struct SoftBodyMeshfreeParticleSource
{
    std::vector<Diligent::float3> particleRestPositions; ///< Interior simulation particle positions.
    std::vector<Diligent::float3> surfaceRestPositions;  ///< High-resolution surface mesh vertex coordinates.
    std::vector<Diligent::float3> surfaceNormals;        ///< Surface normal vectors.
    std::vector<Diligent::uint3> surfaceTriangles;       ///< Surface triangle indices.
    std::vector<std::uint32_t> staticParticleIndices;    ///< Local particle indices fixed in space.
    std::uint32_t neighbourCount = 12u;                  ///< Number of nearest neighbors per particle for shape matching.
};

/// @brief Discriminated union descriptor selecting the geometry source for a soft body.
struct SoftBodySourceDesc
{
    SoftBodySourceKind kind = SoftBodySourceKind::RegularGrid; ///< Selected source kind.
    SoftBodyRegularGridSource regularGrid;                    ///< Grid configuration (used when kind == RegularGrid).
    SoftBodyTetMeshSource tetMesh;                            ///< Explicit mesh configuration (used when kind == TetMesh).
    SoftBodyTetGenSource tetGen;                              ///< TetGen file paths (used when kind == TetGenFiles).
    SoftBodyMeshfreeParticleSource meshfreeParticles;         ///< Meshfree configuration (used when kind == MeshfreeParticles).
};

/// @brief Source parameters for generating a regular grid fluid volume.
struct FluidRegularGridSource
{
    Diligent::float3 size{1.0f, 1.0f, 1.0f}; ///< 3D bounding box extents for fluid particle placement.
    float targetParticleSpacing = 0.25f;     ///< Desired inter-particle spacing distance.
};

/// @brief Discriminated union descriptor selecting the source configuration for fluid particles.
struct FluidSourceDesc
{
    FluidSourceKind kind = FluidSourceKind::RegularGrid; ///< Selected fluid source kind.
    FluidRegularGridSource regularGrid;                  ///< Regular grid parameters.
};

/// @brief Contact and frictional material properties for particle interactions.
struct ParticleContactMaterialDesc
{
    float friction       = 0.0f;  ///< Dynamic kinetic friction coefficient.
    float restitution    = 0.0f;  ///< Coefficient of restitution (bounciness).
    float damping        = 0.0f;  ///< Contact velocity damping factor.
    float staticFriction = -1.0f; ///< Static friction coefficient (<0 reuses dynamic friction).
};

/// @brief Material parameters for deformable soft bodies.
struct SoftBodyMaterialDesc
{
    ParticleContactMaterialDesc contact{}; ///< Contact interaction properties.
};

/// @brief Material parameters for elastic strands.
struct StrandMaterialDesc
{
    ParticleContactMaterialDesc contact{}; ///< Contact interaction properties.
};

/// @brief Physical and fluid-dynamic material parameters for Position-Based Fluids.
struct FluidMaterialDesc
{
    ParticleContactMaterialDesc contact{}; ///< Contact interaction properties against solids.
    float viscosity            = 0.01f;    ///< Dynamic kinematic viscosity coefficient.
    float cohesion             = 0.0f;     ///< Inter-particle cohesion attraction force coefficient.
    float surfaceTension       = 0.0f;     ///< Surface tension energy coefficient.
    float vorticityConfinement = 0.0f;     ///< Vorticity confinement force multiplier to preserve turbulent eddies.
    float gravityScale         = 1.0f;     ///< Multiplier for gravitational acceleration on fluid.
    float cflCoefficient       = 1.0f;     ///< CFL stability condition time-step factor.
};

/// @brief GPU representation of fluid material properties and derived numerical constants.
///
/// Occupies 48 bytes matching its GPU uniform layout.
struct FluidMaterialGpu
{
    float restDensity                   = 1000.0f;         ///< Fluid rest density (kg/m^3).
    float invRestDensity                = 1.0f / 1000.0f;  ///< Reciprocal rest density.
    float smoothingRadius               = 0.4f;            ///< SPH kernel smoothing radius (h).
    float densityConstraintScaleDerived = 1.0f;            ///< Precomputed density constraint normalization scale.
    float viscosityDerived              = 0.01f / 1000.0f; ///< Derived kinematic viscosity.
    float cohesionDerived               = 0.0f;            ///< Derived cohesion factor.
    float cohesion1                     = 0.0f;            ///< Derived cohesion term 1.
    float cohesion2                     = 0.0f;            ///< Derived cohesion term 2.
    float surfaceTensionDerived         = 0.0f;            ///< Derived surface tension factor.
    float vorticityConfinementDerived   = 0.0f;            ///< Derived vorticity confinement factor.
    float gravityScale                  = 1.0f;            ///< Gravity scaling factor.
    float cflRadius                     = 0.25f;           ///< Radius for CFL velocity limit.

    constexpr bool operator==(const FluidMaterialGpu &rhs) const noexcept
    {
        return restDensity == rhs.restDensity && invRestDensity == rhs.invRestDensity &&
               smoothingRadius == rhs.smoothingRadius &&
               densityConstraintScaleDerived == rhs.densityConstraintScaleDerived &&
               viscosityDerived == rhs.viscosityDerived && cohesionDerived == rhs.cohesionDerived &&
               cohesion1 == rhs.cohesion1 && cohesion2 == rhs.cohesion2 &&
               surfaceTensionDerived == rhs.surfaceTensionDerived &&
               vorticityConfinementDerived == rhs.vorticityConfinementDerived &&
               gravityScale == rhs.gravityScale && cflRadius == rhs.cflRadius;
    }
};

static_assert(sizeof(FluidMaterialGpu) == 48u);

/// @brief Complete dynamic and kinematic state descriptor for an authored rigid body.
struct RigidBodyState
{
    RigidBodyId rigidBodyId        = kInvalidRigidBodyId;   ///< Assigned rigid body identifier.
    common::EntityId entityId      = common::kInvalidEntityId; ///< Associated ECS scene entity ID.
    std::uint32_t environmentIndex = 0u;                    ///< Multi-environment partition index.
    Diligent::float3 position{0.0f, 0.0f, 0.0f};           ///< Center-of-mass world position (x, y, z).
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};///< World orientation quaternion.
    Diligent::float3 scale{1.0f, 1.0f, 1.0f};              ///< 3D scale factors.
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};     ///< Linear velocity vector (m/s).
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};    ///< Angular velocity vector (rad/s).
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};///< Diagonal elements of inverted body-local inertia tensor.
    RigidBodyType bodyType = RigidBodyType::Dynamic;        ///< Rigid body simulation type (Static, Kinematic, Dynamic).
    float inverseMass      = 1.0f;                          ///< Reciprocal mass (1/m), 0 for static bodies.
    Diligent::float3 kinematicTargetPosition{0.0f, 0.0f, 0.0f}; ///< Kinematic target position when driven.
    Diligent::QuaternionF kinematicTargetRotation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Kinematic target rotation when driven.
    bool kinematicTargetEnabled = false;                    ///< Whether kinematic target driving is active.
    float proxyParticleRadius   = 0.0f;                     ///< Radius of proxy collision particles.
    ParticleContactMaterialDesc proxyParticleMaterial{};    ///< Contact material for proxy collision particles.
    std::uint32_t proxyCollisionLayer        = 1u;          ///< Collision layer bit for proxy particles.
    std::uint32_t proxyCollisionMask         = 0xffffffffu; ///< Collision mask bitfield for proxy particles.
    std::uint32_t proxyParticleOffset        = 0u;          ///< Offset into global particle buffer.
    std::uint32_t proxyParticleCount         = 0u;          ///< Number of proxy particles attached to this rigid body.
    std::uint32_t proxyParticleMaterialIndex = 0u;          ///< Material table index for proxy contact.
    bool suturingEnabled                     = false;       ///< Whether proxy particles participate in suturing needle puncture tracking.
    std::uint32_t needleTipProxyIndex        = 0u;          ///< Proxy particle index acting as the needle tip.
    Diligent::float4 proxyParticleContactMaterial{0.0f, 0.0f, 0.0f, 0.0f}; ///< Packed proxy contact material.
    std::vector<Diligent::float3> proxyParticleLocalPositions; ///< Local-space positions of rigid proxy particles.
};

/// @brief Physical collision shape attachment for rigid bodies.
struct ColliderState
{
    ColliderId colliderId        = kInvalidColliderId;     ///< Unique collider identifier.
    common::EntityId entityId    = common::kInvalidEntityId; ///< Associated ECS entity ID.
    RigidBodyId ownerRigidBodyId = kInvalidRigidBodyId;     ///< Owning rigid body identifier.
    ColliderShapeType shapeType  = ColliderShapeType::Sphere;///< Geometric shape type (Sphere, Box, Capsule).
    Diligent::float4 shapeParams{0.5f, 0.0f, 0.0f, 0.0f};  ///< Shape geometric dimensions (radius, half-extents).
    Diligent::float3 localPosition{0.0f, 0.0f, 0.0f};      ///< Local translation offset from rigid body center.
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Local rotation offset.
    bool enabled                 = true;                   ///< Active collision state.
    float friction               = 0.5f;                   ///< Dynamic kinetic friction coefficient.
    float staticFriction         = -1.0f;                  ///< Static friction coefficient (-1 reuses friction).
    float restitution            = 0.0f;                   ///< Restitution coefficient (bounciness).
    std::uint32_t collisionLayer = 1u;                     ///< Bitmask collision layer.
    std::uint32_t collisionMask  = 0xffffffffu;            ///< Bitmask collision filter mask.
};

/// @brief State descriptor for an authored 3D deformable tetrahedral or meshfree soft body.
struct SoftBodyState
{
    common::EntityId entityId      = common::kInvalidEntityId; ///< Associated ECS entity ID.
    std::uint32_t environmentIndex = 0u;                    ///< Multi-environment index.
    std::uint32_t collisionLayer   = 1u;                    ///< Collision layer bitmask.
    std::uint32_t collisionMask    = 0xffffffffu;           ///< Collision filtering bitmask.
    bool supportsSuturing          = false;                 ///< Whether this soft body can be pierced by surgical sutures.
    SoftBodySourceDesc source{};                            ///< Geometry source definition.
    SoftBodyMaterialDesc material{};                        ///< Soft-body contact material properties.
    common::Transform restTransform{};                      ///< Initial rest transform pose in world space.
    float particleMass                 = 1.0f;              ///< Mass per soft-body particle.
    float particleRadius               = 0.125f;            ///< Collision radius per particle.
    float edgeCompliance               = 0.0f;              ///< Distance/elastic edge constraint compliance.
    float volumeCompliance             = 0.0f;              ///< Tetrahedral hydrostatic volume constraint compliance.
    bool selfCollisionEnabled          = false;             ///< Whether particles of this soft body collide with one another.
    std::uint32_t contactMaterialIndex = 0u;                ///< Index into contact material table.
    std::uint32_t particleOffset       = 0u;                ///< Start offset in global particle buffer.
    std::uint32_t particleCount        = 0u;                ///< Number of particles in this soft body.
    std::uint32_t edgeOffset           = 0u;                ///< Start offset in global edge constraint buffer.
    std::uint32_t edgeCount            = 0u;                ///< Number of distance edge constraints.
    std::uint32_t tetOffset            = 0u;                ///< Start offset in tetrahedral constraint buffer.
    std::uint32_t tetCount             = 0u;                ///< Number of tetrahedral volume constraints.
    std::vector<Diligent::float3> restPositions;            ///< Object-space rest positions of particles.
    std::vector<Diligent::uint3> boundaryFaces;             ///< Triangle indices for surface boundary rendering.
};

/// @brief State descriptor for an authored 1D elastic Cosserat-like strand or surgical suture thread.
struct StrandState
{
    common::EntityId entityId      = common::kInvalidEntityId; ///< Associated ECS entity ID.
    std::uint32_t environmentIndex = 0u;                    ///< Multi-environment index.
    std::uint32_t collisionLayer   = 1u;                    ///< Collision layer bitmask.
    std::uint32_t collisionMask    = 0xffffffffu;           ///< Collision filter bitmask.
    bool suturingEnabled           = false;                 ///< Whether suturing needle tracking is active for this strand.
    float pathNodeSpacing          = 0.2f;                  ///< Node sampling interval when generating puncture paths.
    StrandMaterialDesc material{};                          ///< Strand contact material parameters.
    float particleMass           = 1.0f;                    ///< Mass per node particle.
    float particleRadius         = 0.125f;                  ///< Collision radius per node particle.
    float stretchShearCompliance = 0.0f;                    ///< Stretch and shear compliance coefficient.
    float bendCompliance         = 0.0f;                    ///< Bending compliance coefficient.
    float twistCompliance        = 0.0f;                    ///< Torsional twist compliance coefficient.
    float distanceCompliance     = 0.0f;                    ///< Segment distance compliance coefficient.
    Diligent::float3 rootMaterialNormal{0.0f, 1.0f, 0.0f};  ///< Normal vector at the root node for orientation framing.
    bool selfCollisionEnabled          = false;             ///< Whether strand nodes collide with one another.
    std::uint32_t contactMaterialIndex = 0u;                ///< Index in contact material table.
    std::uint32_t particleOffset       = 0u;                ///< Start offset in global particle buffer.
    std::uint32_t particleCount        = 0u;                ///< Number of particle nodes in this strand.
    std::uint32_t segmentOffset        = 0u;                ///< Start offset in strand segment buffer.
    std::uint32_t segmentCount         = 0u;                ///< Number of segments in this strand.
    std::uint32_t jointOffset          = 0u;                ///< Start offset in strand joint buffer.
    std::uint32_t jointCount           = 0u;                ///< Number of inter-segment joints in this strand.
    std::vector<Diligent::float3> restPositions;            ///< Node rest positions in object space.
    std::vector<std::uint32_t> staticParticleIndices;       ///< Node indices fixed in space.
};

/// @brief State descriptor for an authored particle-based fluid body.
struct FluidState
{
    common::EntityId entityId      = common::kInvalidEntityId; ///< Associated ECS entity ID.
    std::uint32_t environmentIndex = 0u;                    ///< Multi-environment index.
    std::uint32_t collisionLayer   = 1u;                    ///< Collision layer bitmask.
    std::uint32_t collisionMask    = 0xffffffffu;           ///< Collision filter bitmask.
    FluidSourceDesc source{};                               ///< Source emitter/generator configuration.
    FluidMaterialDesc material{};                           ///< Fluid material parameters (viscosity, cohesion).
    Diligent::float4 visualColor{0.32f, 0.62f, 0.95f, 0.72f};///< RGBA color for rendering fluid particles.
    common::Transform restTransform{};                      ///< World transform placement of fluid block.
    float particleMass                 = 1.0f;              ///< Mass per fluid particle.
    float particleRadius               = 0.125f;            ///< SPH interaction radius.
    std::uint32_t contactMaterialIndex = 0u;                ///< Contact material table index.
    std::uint32_t fluidMaterialIndex   = 0u;                ///< Fluid material table index.
    std::uint32_t particleOffset       = 0u;                ///< Offset in global particle buffer.
    std::uint32_t particleCount        = 0u;                ///< Number of fluid particles.
    std::vector<Diligent::float3> restPositions;            ///< Rest positions of fluid particles.
};

/// @brief Reference locating a specific simulation particle within an authored entity.
struct AuthoredParticleReference
{
    common::EntityId entityId          = common::kInvalidEntityId;            ///< Target entity ID.
    AuthoredParticleReferenceType type = AuthoredParticleReferenceType::StrandParticle; ///< Entity particle owner type.
    std::uint32_t localParticleIndex   = 0u;                                  ///< Zero-based local index of the particle.
};

/// @brief Authored distance constraint between two referenced particles.
struct AuthoredParticleDistanceConstraintState
{
    ParticleConstraintId constraintId = kInvalidParticleConstraintId; ///< Unique constraint ID.
    AuthoredParticleReference particleA{};                            ///< First referenced particle.
    AuthoredParticleReference particleB{};                            ///< Second referenced particle.
    float restLength = 0.0f;                                          ///< Target rest distance separation.
    float compliance = 0.0f;                                          ///< XPBD constraint compliance.
    bool enabled     = true;                                          ///< Whether constraint is active.
};

/// @brief Authored distance constraint between anchor points on two rigid bodies.
struct AuthoredRigidDistanceConstraintState
{
    RigidDistanceConstraintId constraintId = kInvalidRigidDistanceConstraintId; ///< Unique constraint ID.
    common::EntityId entityA               = common::kInvalidEntityId;          ///< First rigid-body entity ID.
    common::EntityId entityB               = common::kInvalidEntityId;          ///< Second rigid-body entity ID.
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};                           ///< Anchor position in body A local space.
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};                           ///< Anchor position in body B local space.
    float restDistance = 0.0f;                                                 ///< Target rest distance between anchors.
    float compliance   = 0.0f;                                                 ///< XPBD constraint compliance.
    bool enabled       = true;                                                 ///< Whether constraint is active.
};

/// @brief Authored attachment constraint between a referenced particle and a rigid body.
struct AuthoredRigidParticleAttachmentConstraintState
{
    RigidParticleAttachmentConstraintId constraintId = kInvalidRigidParticleAttachmentConstraintId; ///< Unique constraint ID.
    AuthoredParticleReference particle{};                                                           ///< Target particle reference.
    common::EntityId rigidBodyEntityId = common::kInvalidEntityId;                                  ///< Target rigid-body entity ID.
    Diligent::float3 localAnchor{0.0f, 0.0f, 0.0f};                                                ///< Attachment anchor in rigid-body local space.
    float compliance = 0.0f;                                                                        ///< XPBD constraint compliance.
    bool enabled     = true;                                                                        ///< Whether attachment is active.
};

/// @brief One-way attachment constraint where a rigid body kinematic/dynamic pose drives a strand segment.
struct AuthoredStrandRigidAttachmentConstraintState
{
    // This constraint is intentionally one-way: the rigid body drives the strand
    // station pose, but the strand does not push force or torque back into the rigid body.
    StrandRigidAttachmentConstraintId constraintId = kInvalidStrandRigidAttachmentConstraintId; ///< Unique constraint ID.
    common::EntityId strandEntityId                = common::kInvalidEntityId;                 ///< Strand entity ID.
    std::uint32_t localSegmentIndex                = 0u;                                       ///< Attached segment index along the strand.
    float segmentT                                 = 0.0f;                                     ///< Parametric position [0, 1] along segment.
    common::EntityId rigidBodyEntityId             = common::kInvalidEntityId;                 ///< Driving rigid-body entity ID.
    Diligent::float3 localAnchor{0.0f, 0.0f, 0.0f};                                            ///< Local anchor offset in rigid body.
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f};                               ///< Relative orientation in rigid body.
    float translationCompliance = 0.0f;                                                        ///< Positional compliance.
    float rotationCompliance    = 0.0f;                                                        ///< Angular rotational compliance.
    bool enabled                = true;                                                        ///< Whether attachment is active.
};

/// @brief Guide point on a rigid body through which a routed cable passes.
struct AuthoredRoutedCableRoutePoint
{
    common::EntityId entityId = common::kInvalidEntityId; ///< Rigid-body entity ID hosting the guide.
    Diligent::float3 localGuideOffset{0.0f, 0.0f, 0.0f}; ///< Local coordinate offset of guide on the body.

    constexpr bool operator==(const AuthoredRoutedCableRoutePoint &rhs) const noexcept
    {
        return entityId == rhs.entityId && localGuideOffset == rhs.localGuideOffset;
    }
};

/// @brief Cable length constraint routed through an ordered sequence of rigid-body guide points.
struct AuthoredRoutedCableConstraintState
{
    RoutedCableConstraintId constraintId = kInvalidRoutedCableConstraintId; ///< Unique constraint ID.
    std::vector<AuthoredRoutedCableRoutePoint> routePoints{};               ///< Ordered guide points along the cable path.
    float targetLength = 0.0f;                                              ///< Target total cable length.
    float compliance   = 0.0f;                                              ///< XPBD constraint compliance.
    bool tensionOnly   = true;                                              ///< True if constraint resists stretching only (slack allowed).
    bool enabled       = true;                                              ///< Whether constraint is active.
};

/// @brief Collision filtering override for an individual particle.
struct AuthoredParticleCollisionFilterState
{
    ParticleCollisionFilterId filterId = kInvalidParticleCollisionFilterId; ///< Unique filter ID.
    AuthoredParticleReference particle{};                                   ///< Target particle reference.
    std::uint32_t collisionLayer = 1u;                                      ///< Collision layer bitmask.
    std::uint32_t collisionMask  = 0xffffffffu;                             ///< Collision filter bitmask.
    bool enabled                 = true;                                    ///< Whether collision filtering is active.
};

/// @brief Ordered sequence of particle references.
struct AuthoredParticleSequenceState
{
    ParticleSequenceId sequenceId = kInvalidParticleSequenceId; ///< Unique sequence ID.
    std::vector<AuthoredParticleReference> entries{};           ///< Ordered list of particle references.
    bool enabled = true;                                        ///< Whether sequence is active.
};

/// @brief Authored surgical needle and suture thread sequence for soft-tissue puncture tracking.
struct AuthoredSuturingSequenceState
{
    SuturingSequenceId sequenceId = kInvalidSuturingSequenceId; ///< Unique suturing sequence ID.
    std::vector<AuthoredParticleReference> entries{};           ///< Ordered needle and thread particle references.
    // The selected tip entry authors the suturing path. In the current prototype,
    // the sequence tip and tail also suppress same-soft-body exterior contact.
    std::uint32_t tipEntryIndex = 0u;                           ///< Index into entries designating the needle tip.
    float pathNodeSpacing       = 0.0f;                         ///< Desired spacing between generated path interpolation nodes.
    bool enabled                = true;                         ///< Whether suturing tracking is active.
};

/// @brief XPBD distance constraint between two deformable particles.
struct DeformableDistanceConstraint
{
    std::uint32_t particleA = 0u;   ///< Index of first particle.
    std::uint32_t particleB = 0u;   ///< Index of second particle.
    float restLength        = 0.0f; ///< Target rest distance.
    float compliance        = 0.0f; ///< XPBD constraint compliance.
    // Keep authored constraints resident in the GPU edge buffer so enable/disable can be
    // evaluated by the solver instead of changing the constraint's GPU identity.
    std::uint32_t enabled   = 1u;   ///< Constraint enabled flag (0 or 1).
    std::uint32_t reserved0 = 0u;   ///< Reserved padding.
    std::uint32_t reserved1 = 0u;   ///< Reserved padding.
    std::uint32_t reserved2 = 0u;   ///< Reserved padding.
};

/// @brief Alias for deformable distance edge constraint.
using SoftEdge = DeformableDistanceConstraint;

/// @brief Hydrostatic volume conservation constraint for tetrahedral elements.
struct DeformableVolumeConstraint
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u}; ///< 4 particle indices forming the tetrahedron.
    float restVolume        = 0.0f;                  ///< Target rest volume.
    float compliance        = 0.0f;                  ///< Hydrostatic volume compliance.
    std::uint32_t reserved0 = 0u;                    ///< Reserved padding.
    std::uint32_t reserved1 = 0u;                    ///< Reserved padding.
};

/// @brief Alias for tetrahedral volume constraint.
using SoftTet = DeformableVolumeConstraint;

/// @brief Bending angle constraint among a triplet of deformable particles.
struct DeformableBendConstraint
{
    std::uint32_t particle0 = 0u;   ///< Index of first particle.
    std::uint32_t particle1 = 0u;   ///< Index of vertex hinge particle.
    std::uint32_t particle2 = 0u;   ///< Index of third particle.
    float restAngle         = 0.0f; ///< Target rest bending angle in radians.
    float compliance        = 0.0f; ///< XPBD bending compliance.
    std::uint32_t reserved0 = 0u;   ///< Reserved padding.
    std::uint32_t reserved1 = 0u;   ///< Reserved padding.
    std::uint32_t reserved2 = 0u;   ///< Reserved padding.
};

/// @brief Alias for deformable bending constraint.
using SoftBend = DeformableBendConstraint;

/// @brief Segment constraint connecting adjacent particles along an elastic strand.
struct StrandSegmentConstraint
{
    std::uint32_t particleA      = 0u;                      ///< First particle node.
    std::uint32_t particleB      = 0u;                      ///< Second particle node.
    float restLength             = 0.0f;                    ///< Segment rest length.
    float stretchShearCompliance = 0.0f;                    ///< Stretching and shearing compliance.
    Diligent::float4 restOrientation{0.0f, 0.0f, 0.0f, 1.0f};///< Rest segment orientation quaternion.
};

/// @brief Angular joint constraint connecting two adjacent strand segments.
struct StrandJointConstraint
{
    std::uint32_t segmentA = 0u;                              ///< First strand segment index.
    std::uint32_t segmentB = 0u;                              ///< Second strand segment index.
    float bendCompliance   = 0.0f;                            ///< Bending compliance.
    float twistCompliance  = 0.0f;                            ///< Torsional twisting compliance.
    Diligent::float4 restRelativeOrientation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Rest relative rotation between segments.
};

/// @brief Distance constraint between strand particle nodes.
struct StrandDistanceConstraint
{
    std::uint32_t particleA  = 0u;   ///< First particle index.
    std::uint32_t particleB  = 0u;   ///< Second particle index.
    float restLength         = 0.0f; ///< Target rest separation distance.
    float distanceCompliance = 0.0f; ///< Distance compliance.
};

/// @brief Dynamic runtime orientation state of a strand segment.
struct StrandSegmentState
{
    Diligent::float4 orientation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Current segment orientation quaternion.
};

/// @brief GPU runtime descriptor for a routed cable constraint.
struct RoutedCableConstraint
{
    std::uint32_t routePointStart = 0u;   ///< Start offset in route points buffer.
    std::uint32_t routePointCount = 0u;   ///< Total guide points.
    float targetLength            = 0.0f; ///< Target cable length.
    float compliance              = 0.0f; ///< Compliance coefficient.
    std::uint32_t tensionOnly     = 1u;   ///< True (1) if cable resists stretching only.
    std::uint32_t enabled         = 1u;   ///< Active state flag.
    std::uint32_t reserved1       = 0u;   ///< Reserved padding.
    std::uint32_t reserved2       = 0u;   ///< Reserved padding.
};

/// @brief GPU runtime descriptor for a routed cable guide point.
struct RoutedCableRoutePoint
{
    std::uint32_t rigidBodyIndex = 0u;                    ///< Hosting rigid body index.
    std::uint32_t reserved0      = 0u;                    ///< Reserved padding.
    std::uint32_t reserved1      = 0u;                    ///< Reserved padding.
    std::uint32_t reserved2      = 0u;                    ///< Reserved padding.
    Diligent::float4 localGuideOffset{0.0f, 0.0f, 0.0f, 0.0f}; ///< Local guide coordinate offset on rigid body.
};

/// @brief GPU runtime descriptor for a rigid-to-rigid distance constraint.
struct RigidDistanceConstraint
{
    std::uint32_t rigidBodyIndexA = 0u;                   ///< First rigid body index.
    std::uint32_t rigidBodyIndexB = 0u;                   ///< Second rigid body index.
    float restDistance            = 0.0f;                 ///< Rest separation distance.
    float compliance              = 0.0f;                 ///< Distance compliance.
    std::uint32_t enabled         = 1u;                   ///< Active state flag.
    std::uint32_t reserved0       = 0u;                   ///< Reserved padding.
    Diligent::float4 localAnchorA{0.0f, 0.0f, 0.0f, 0.0f};///< Local anchor offset in body A.
    Diligent::float4 localAnchorB{0.0f, 0.0f, 0.0f, 0.0f};///< Local anchor offset in body B.
};

/// @brief GPU runtime descriptor for an attachment constraint between a particle and rigid body.
struct RigidParticleAttachmentConstraint
{
    std::uint32_t particleIndex  = 0u;                    ///< Target particle index.
    std::uint32_t rigidBodyIndex = 0u;                    ///< Target rigid body index.
    float compliance             = 0.0f;                  ///< Attachment compliance.
    std::uint32_t enabled        = 1u;                    ///< Active state flag.
    Diligent::float4 localAnchor{0.0f, 0.0f, 0.0f, 0.0f}; ///< Local anchor offset on rigid body.
};

/// @brief GPU runtime descriptor for a one-way rigid-driven strand follow attachment.
struct StrandRigidAttachmentConstraint
{
    std::uint32_t segmentIndex   = 0u;                      ///< Attached strand segment index.
    std::uint32_t rigidBodyIndex = 0u;                      ///< Driving rigid body index.
    float segmentT               = 0.0f;                    ///< Parametric position [0, 1] on segment.
    float translationCompliance  = 0.0f;                    ///< Translation compliance.
    float rotationCompliance     = 0.0f;                    ///< Orientation compliance.
    std::uint32_t enabled        = 1u;                      ///< Active state flag.
    std::uint32_t reserved1      = 0u;                      ///< Reserved padding.
    std::uint32_t reserved2      = 0u;                      ///< Reserved padding.
    Diligent::float4 localAnchor{0.0f, 0.0f, 0.0f, 0.0f};   ///< Local anchor on rigid body.
    Diligent::float4 localRotation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Local relative orientation on rigid body.
};

/// @brief Structure-of-Arrays (SoA) host memory container holding unified particle attributes.
struct ParticleSoAHost
{
    std::vector<Diligent::float4> positionsInvMass;
    std::vector<Diligent::float4> previousPositions;
    std::vector<Diligent::float4> velocities;
    std::vector<float> radii;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> particleKinds;
    std::vector<std::uint32_t> ownerTypes;
    std::vector<std::uint32_t> ownerIndices;
    std::vector<std::uint32_t> strandIds;
    std::vector<std::uint32_t> strandOrders;
    std::vector<std::uint32_t> strandRoles;
    std::vector<Diligent::uint4> suturingNeighborLinks;
    std::vector<std::uint32_t> suturingParticleIndices;
    std::vector<std::uint32_t> owningSoftBodyIndices;
    std::vector<std::uint32_t> particleMaterialIndices;
    std::vector<std::uint32_t> fluidMaterialIndices;
    std::vector<std::uint32_t> phases;
    std::vector<std::uint32_t> collisionLayers;
    std::vector<std::uint32_t> collisionMasks;
    std::vector<std::uint32_t> adjacencyOffsets;
    std::vector<std::uint32_t> adjacencyCounts;
    std::vector<std::uint32_t> adjacencyIndices;
    std::vector<Diligent::float4> rigidProxyLocalPositions;

    std::size_t size() const noexcept
    {
        return positionsInvMass.size();
    }

    bool empty() const noexcept
    {
        return positionsInvMass.empty();
    }

    void clear()
    {
        positionsInvMass.clear();
        previousPositions.clear();
        velocities.clear();
        radii.clear();
        environmentIndices.clear();
        particleKinds.clear();
        ownerTypes.clear();
        ownerIndices.clear();
        strandIds.clear();
        strandOrders.clear();
        strandRoles.clear();
        suturingNeighborLinks.clear();
        suturingParticleIndices.clear();
        owningSoftBodyIndices.clear();
        particleMaterialIndices.clear();
        fluidMaterialIndices.clear();
        phases.clear();
        collisionLayers.clear();
        collisionMasks.clear();
        adjacencyOffsets.clear();
        adjacencyCounts.clear();
        adjacencyIndices.clear();
        rigidProxyLocalPositions.clear();
    }
};

/// @brief Structure-of-Arrays (SoA) host memory container holding rigid-body attributes.
struct RigidBodySoAHost
{
    std::vector<RigidBodyId> rigidBodyIds;
    std::vector<common::EntityId> entityIds;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<Diligent::float4> positionsInvMass;
    std::vector<Diligent::float4> orientations;
    std::vector<Diligent::float4> scales;
    std::vector<Diligent::float4> linearVelocities;
    std::vector<Diligent::float4> angularVelocities;
    std::vector<Diligent::float4> inverseInertiaLocal;
    std::vector<std::uint32_t> bodyTypes;
    std::vector<Diligent::float4> kinematicTargetPositions;
    std::vector<Diligent::float4> kinematicTargetOrientations;
    std::vector<std::uint32_t> kinematicTargetFlags;
    std::vector<Diligent::float4> proxyParticleContactMaterials;

    std::size_t size() const noexcept
    {
        return entityIds.size();
    }

    bool empty() const noexcept
    {
        return entityIds.empty();
    }

    void clear()
    {
        rigidBodyIds.clear();
        entityIds.clear();
        environmentIndices.clear();
        positionsInvMass.clear();
        orientations.clear();
        scales.clear();
        linearVelocities.clear();
        angularVelocities.clear();
        inverseInertiaLocal.clear();
        bodyTypes.clear();
        kinematicTargetPositions.clear();
        kinematicTargetOrientations.clear();
        kinematicTargetFlags.clear();
        proxyParticleContactMaterials.clear();
    }
};

/// @brief Structure-of-Arrays (SoA) host memory container holding collider descriptors.
struct ColliderSoAHost
{
    std::vector<ColliderId> colliderIds;
    std::vector<common::EntityId> entityIds;
    std::vector<RigidBodyId> ownerRigidBodyIds;
    std::vector<std::uint32_t> ownerRigidBodyIndices;
    std::vector<std::uint32_t> shapeTypes;
    std::vector<Diligent::float4> shapeParams;
    std::vector<Diligent::float4> localPositions;
    std::vector<Diligent::float4> localOrientations;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<Diligent::float4> frictionRestitution;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> collisionLayers;
    std::vector<std::uint32_t> collisionMasks;

    std::size_t size() const noexcept
    {
        return colliderIds.size();
    }

    bool empty() const noexcept
    {
        return colliderIds.empty();
    }

    void clear()
    {
        colliderIds.clear();
        entityIds.clear();
        ownerRigidBodyIds.clear();
        ownerRigidBodyIndices.clear();
        shapeTypes.clear();
        shapeParams.clear();
        localPositions.clear();
        localOrientations.clear();
        enabledFlags.clear();
        frictionRestitution.clear();
        environmentIndices.clear();
        collisionLayers.clear();
        collisionMasks.clear();
    }
};

/// @brief Host mapping table relating rigid bodies to their attached collider indices.
struct BodyColliderMappingHost
{
    std::vector<std::uint32_t> colliderOffsets;
    std::vector<std::uint32_t> colliderCounts;
    std::vector<std::uint32_t> colliderIndices;

    void clear()
    {
        colliderOffsets.clear();
        colliderCounts.clear();
        colliderIndices.clear();
    }
};

/// @brief State descriptor for an articulated ball (spherical point-to-point) joint.
struct BallJointState
{
    BallJointId jointId                  = kInvalidBallJointId; ///< Unique joint identifier.
    bool enabled                         = true;                ///< Active state flag.
    bool suppressConnectedBodyCollisions = false;               ///< Whether collision between connected bodies is suppressed.
    RigidBodyId bodyA                    = kInvalidRigidBodyId; ///< First connected rigid body ID.
    RigidBodyId bodyB                    = kInvalidRigidBodyId; ///< Second connected rigid body ID.
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};           ///< Anchor position in body A local space.
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};           ///< Anchor position in body B local space.
};

/// @brief State descriptor for an articulated 1-DOF revolute hinge joint with limits and actuator drive.
struct HingeJointState
{
    HingeJointId jointId                 = kInvalidHingeJointId;   ///< Unique joint identifier.
    bool enabled                         = true;                  ///< Active state flag.
    bool suppressConnectedBodyCollisions = false;                 ///< Suppress collision between connected bodies.
    RigidJointDriveMode driveMode        = RigidJointDriveMode::None; ///< Actuator drive control mode.
    bool limitEnabled                    = false;                 ///< Whether angular limits are enabled.
    RigidBodyId bodyA                    = kInvalidRigidBodyId;   ///< First connected body ID.
    RigidBodyId bodyB                    = kInvalidRigidBodyId;   ///< Second connected body ID.
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};             ///< Anchor in body A local space.
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};             ///< Anchor in body B local space.
    Diligent::QuaternionF localRotationA{0.0f, 0.0f, 0.0f, 1.0f};///< Joint orientation in body A.
    Diligent::QuaternionF localRotationB{0.0f, 0.0f, 0.0f, 1.0f};///< Joint orientation in body B.
    float limitMin                   = 0.0f;                      ///< Minimum hinge angle limit (radians).
    float limitMax                   = 0.0f;                      ///< Maximum hinge angle limit (radians).
    float constraintCompliance       = 0.0f;                      ///< Hinge constraint compliance.
    float driveCompliance            = 0.0f;                      ///< Actuator drive compliance.
    float driveTargetAngle           = 0.0f;                      ///< Target angle for position drive (radians).
    float driveDamping               = 0.0f;                      ///< Damping coefficient for hinge drive.
    float driveMaxAngularVelocity    = 0.0f;                      ///< Maximum velocity clamp for drive.
    float driveTargetAngularVelocity = 0.0f;                      ///< Target angular velocity for velocity drive.
};

/// @brief State descriptor for a 3-DOF spherical joint with swing limits, twist limits, and orientation drive.
struct SphericalJointState
{
    SphericalJointId jointId             = kInvalidSphericalJointId; ///< Unique joint identifier.
    bool enabled                         = true;                    ///< Active state flag.
    bool suppressConnectedBodyCollisions = false;                   ///< Suppress collision between connected bodies.
    RigidJointDriveMode driveMode        = RigidJointDriveMode::None;///< Drive control mode.
    bool limitEnabled                    = false;                   ///< Whether swing/twist limits are enabled.
    RigidBodyId bodyA                    = kInvalidRigidBodyId;     ///< First connected body ID.
    RigidBodyId bodyB                    = kInvalidRigidBodyId;     ///< Second connected body ID.
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};               ///< Anchor in body A local space.
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};               ///< Anchor in body B local space.
    Diligent::QuaternionF localRotationA{0.0f, 0.0f, 0.0f, 1.0f};  ///< Joint frame in body A.
    Diligent::QuaternionF localRotationB{0.0f, 0.0f, 0.0f, 1.0f};  ///< Joint frame in body B.
    float swingLimitY          = 0.0f;                              ///< Maximum swing angle about joint Y axis (radians).
    float swingLimitZ          = 0.0f;                              ///< Maximum swing angle about joint Z axis (radians).
    float twistLimitMin        = 0.0f;                              ///< Minimum twist angle limit (radians).
    float twistLimitMax        = 0.0f;                              ///< Maximum twist angle limit (radians).
    float constraintCompliance = 0.0f;                              ///< Anchor position compliance.
    float swingCompliance      = 0.0f;                              ///< Swing-limit compliance.
    float twistCompliance      = 0.0f;                              ///< Twist-limit compliance.
    float driveCompliance      = 0.0f;                              ///< Orientation drive compliance.
    Diligent::QuaternionF driveTargetOrientation{0.0f, 0.0f, 0.0f, 1.0f}; ///< Target orientation for drive.
};

/// @brief State descriptor for a 1-DOF prismatic slider joint with linear travel limits and linear actuator drive.
struct SliderJointState
{
    SliderJointId jointId                = kInvalidSliderJointId;   ///< Unique joint identifier.
    bool enabled                         = true;                    ///< Active state flag.
    bool suppressConnectedBodyCollisions = false;                   ///< Suppress collision between connected bodies.
    RigidJointDriveMode driveMode        = RigidJointDriveMode::None;///< Drive control mode.
    bool limitEnabled                    = false;                   ///< Whether linear travel limits are enabled.
    RigidBodyId bodyA                    = kInvalidRigidBodyId;     ///< First connected body ID.
    RigidBodyId bodyB                    = kInvalidRigidBodyId;     ///< Second connected body ID.
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};               ///< Anchor in body A local space.
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};               ///< Anchor in body B local space.
    Diligent::QuaternionF localRotationA{0.0f, 0.0f, 0.0f, 1.0f};  ///< Slider axis frame in body A.
    Diligent::QuaternionF localRotationB{0.0f, 0.0f, 0.0f, 1.0f};  ///< Slider axis frame in body B.
    float limitMin             = 0.0f;                              ///< Minimum linear translation limit (m).
    float limitMax             = 0.0f;                              ///< Maximum linear translation limit (m).
    float constraintCompliance = 0.0f;                              ///< Slider constraint compliance.
    float driveCompliance      = 0.0f;                              ///< Linear drive compliance.
    float driveDamping         = 0.0f;                              ///< Damping coefficient for slider drive.
    float driveMaxVelocity     = 0.0f;                              ///< Maximum linear velocity clamp for drive.
    float driveTargetPosition  = 0.0f;                              ///< Target position for position drive (m).
    float driveTargetVelocity  = 0.0f;                              ///< Target velocity for velocity drive (m/s).
};

/// @brief Structure-of-Arrays (SoA) host memory container holding ball joint descriptors.
struct BallJointSoAHost
{
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;

    std::size_t size() const noexcept
    {
        return bodyIndicesA.size();
    }
    bool empty() const noexcept
    {
        return bodyIndicesA.empty();
    }
    void clear()
    {
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
    }
};

/// @brief Structure-of-Arrays (SoA) host memory container holding hinge joint descriptors.
struct HingeJointSoAHost
{
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<std::uint32_t> driveModes;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;
    std::vector<Diligent::float4> localAxesA0;
    std::vector<Diligent::float4> localAxesA1;
    std::vector<Diligent::float4> localAxesB1;
    std::vector<std::uint32_t> limitEnabledFlags;
    std::vector<float> limitMins;
    std::vector<float> limitMaxs;
    std::vector<float> constraintCompliances;
    std::vector<float> driveCompliances;
    std::vector<float> driveTargetAngles;
    std::vector<float> driveDampings;
    std::vector<float> driveMaxAngularVelocities;
    std::vector<float> driveTargetAngularVelocities;
    std::vector<Diligent::float4> projectionRow0;
    std::vector<Diligent::float4> projectionRow1;
    std::vector<Diligent::float4> projectionRow2;

    std::size_t size() const noexcept
    {
        return bodyIndicesA.size();
    }
    bool empty() const noexcept
    {
        return bodyIndicesA.empty();
    }
    void clear()
    {
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        driveModes.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
        localAxesA0.clear();
        localAxesA1.clear();
        localAxesB1.clear();
        limitEnabledFlags.clear();
        limitMins.clear();
        limitMaxs.clear();
        constraintCompliances.clear();
        driveCompliances.clear();
        driveTargetAngles.clear();
        driveDampings.clear();
        driveMaxAngularVelocities.clear();
        driveTargetAngularVelocities.clear();
        projectionRow0.clear();
        projectionRow1.clear();
        projectionRow2.clear();
    }
};

/// @brief Structure-of-Arrays (SoA) host memory container holding spherical joint descriptors.
struct SphericalJointSoAHost
{
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<std::uint32_t> driveModes;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;
    std::vector<Diligent::float4> localRotationsA;
    std::vector<Diligent::float4> localRotationsB;
    std::vector<std::uint32_t> limitEnabledFlags;
    std::vector<float> swingLimitYs;
    std::vector<float> swingLimitZs;
    std::vector<float> twistLimitMins;
    std::vector<float> twistLimitMaxs;
    std::vector<float> constraintCompliances;
    std::vector<float> swingCompliances;
    std::vector<float> twistCompliances;
    std::vector<float> driveCompliances;
    std::vector<Diligent::float4> driveTargetOrientations;

    std::size_t size() const noexcept
    {
        return bodyIndicesA.size();
    }
    bool empty() const noexcept
    {
        return bodyIndicesA.empty();
    }
    void clear()
    {
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        driveModes.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
        localRotationsA.clear();
        localRotationsB.clear();
        limitEnabledFlags.clear();
        swingLimitYs.clear();
        swingLimitZs.clear();
        twistLimitMins.clear();
        twistLimitMaxs.clear();
        constraintCompliances.clear();
        swingCompliances.clear();
        twistCompliances.clear();
        driveCompliances.clear();
        driveTargetOrientations.clear();
    }
};

/// @brief Structure-of-Arrays (SoA) host memory container holding slider joint descriptors.
struct SliderJointSoAHost
{
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<std::uint32_t> driveModes;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;
    std::vector<std::uint32_t> limitEnabledFlags;
    std::vector<float> limitMins;
    std::vector<float> limitMaxs;
    std::vector<float> constraintCompliances;
    std::vector<float> driveCompliances;
    std::vector<float> driveDampings;
    std::vector<float> driveMaxVelocities;
    std::vector<float> driveTargetPositions;
    std::vector<float> driveTargetVelocities;
    std::vector<float> driveRestOffsets;
    std::vector<Diligent::float4> localAxesA0;
    std::vector<Diligent::float4> localAxesA1;
    std::vector<Diligent::float4> localAxesA2;
    std::vector<Diligent::float4> projectionRow0;
    std::vector<Diligent::float4> projectionRow1;
    std::vector<Diligent::float4> projectionRow2;

    std::size_t size() const noexcept
    {
        return bodyIndicesA.size();
    }
    bool empty() const noexcept
    {
        return bodyIndicesA.empty();
    }
    void clear()
    {
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        driveModes.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
        limitEnabledFlags.clear();
        limitMins.clear();
        limitMaxs.clear();
        constraintCompliances.clear();
        driveCompliances.clear();
        driveDampings.clear();
        driveMaxVelocities.clear();
        driveTargetPositions.clear();
        driveTargetVelocities.clear();
        driveRestOffsets.clear();
        localAxesA0.clear();
        localAxesA1.clear();
        localAxesA2.clear();
        projectionRow0.clear();
        projectionRow1.clear();
        projectionRow2.clear();
    }
};

/// @brief Combined host SoA container for all joint categories (ball, spherical, hinge, slider).
struct RigidJointSceneHost
{
    BallJointSoAHost ball;
    SphericalJointSoAHost spherical;
    HingeJointSoAHost hinge;
    SliderJointSoAHost slider;

    void clear()
    {
        ball.clear();
        spherical.clear();
        hinge.clear();
        slider.clear();
    }
};

/// @brief Host container for collision suppression pairs between connected joint bodies.
struct JointCollisionSuppressionHost
{
    std::vector<std::uint32_t> neighborOffsets;
    std::vector<std::uint32_t> neighbors;

    void clear()
    {
        neighborOffsets.clear();
        neighbors.clear();
    }
};

/// @brief Vertex-to-triangle mapping range for surface normal computation.
struct SoftRenderVertexTriangleRange
{
    std::uint32_t start     = 0u;
    std::uint32_t count     = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

/// @brief Skinning binding connecting a render surface vertex to four simulation particles.
struct SoftRenderVertexBinding
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u}; ///< Four simulation particle indices.
    Diligent::float4 weights{1.0f, 0.0f, 0.0f, 0.0f}; ///< Interpolation weights summing to 1.0.
};

/// @brief Host container for soft body surface rendering data and barycentric vertex bindings.
struct SoftRenderDataHost
{
    std::vector<SoftRenderVertexBinding> vertexBindings;
    std::vector<Diligent::float4> fallbackNormals;
    std::vector<SoftRenderVertexTriangleRange> vertexTriangleRanges;
    std::vector<std::uint32_t> vertexTriangleIndices;
    std::vector<Diligent::uint4> triangleParticleIndices;
    std::vector<Diligent::uint2> softBodyParticleRanges;

    void clear()
    {
        vertexBindings.clear();
        fallbackNormals.clear();
        vertexTriangleRanges.clear();
        vertexTriangleIndices.clear();
        triangleParticleIndices.clear();
        softBodyParticleRanges.clear();
    }
};

/// @brief Curve rendering descriptor for strand tubular geometry generation.
struct CurveRenderDescriptorHost
{
    std::uint32_t particleIndexStart = 0u;
    std::uint32_t particleCount      = 0u;
    std::uint32_t vertexBase         = 0u;
    std::uint32_t vertexCount        = 0u;
    std::uint32_t radialResolution   = 0u;
    std::uint32_t environmentIndex   = 0u;
    float radius                     = 0.0f;
};

/// @brief Host container for curve rendering descriptors.
struct CurveRenderDataHost
{
    std::vector<CurveRenderDescriptorHost> descriptors;
    std::vector<std::uint32_t> particleIndices;

    void clear()
    {
        descriptors.clear();
        particleIndices.clear();
    }
};

/// @brief GPU metadata descriptor for rigid body SoA buffers.
struct RigidBodySoAGpu
{
    std::uint32_t bodyCount = 0;
    std::uint32_t capacity  = 0;
};

/// @brief GPU metadata descriptor for collider SoA buffers.
struct ColliderSoAGpu
{
    std::uint32_t colliderCount = 0;
    std::uint32_t capacity      = 0;
};

/// @brief Aggregate GPU and host buffer container for physics scene state.
struct PhysicsGpuBuffers
{
    RigidBodySoAHost rigidBodies{};
    ColliderSoAHost colliders{};
    RigidBodySoAGpu rigidBodyGpu{};
    ColliderSoAGpu colliderGpu{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
