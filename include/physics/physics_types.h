#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H

#include "common/id.h"
#include "common/math_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cressim::neo::physics
{

enum class ColliderShapeType : std::uint32_t
{
    Sphere  = 0u,
    Box     = 1u,
    Capsule = 2u,
};

enum class RigidBodyType : std::uint32_t
{
    Static    = 0u,
    Kinematic = 1u,
    Dynamic   = 2u,
};

using RigidBodyId                         = std::uint32_t;
constexpr RigidBodyId kInvalidRigidBodyId = 0u;

using ColliderId                        = std::uint32_t;
constexpr ColliderId kInvalidColliderId = 0u;

using BallJointId                         = std::uint32_t;
constexpr BallJointId kInvalidBallJointId = 0u;

using HingeJointId                          = std::uint32_t;
constexpr HingeJointId kInvalidHingeJointId = 0u;

using SliderJointId                           = std::uint32_t;
constexpr SliderJointId kInvalidSliderJointId = 0u;

using RigidDistanceConstraintId = std::uint32_t;
constexpr RigidDistanceConstraintId kInvalidRigidDistanceConstraintId = 0u;

using ParticleConstraintId                                  = std::uint32_t;
constexpr ParticleConstraintId kInvalidParticleConstraintId = 0u;

using RigidParticleAttachmentConstraintId = std::uint32_t;
constexpr RigidParticleAttachmentConstraintId kInvalidRigidParticleAttachmentConstraintId = 0u;

using RoutedCableConstraintId                                  = std::uint32_t;
constexpr RoutedCableConstraintId kInvalidRoutedCableConstraintId = 0u;

using ParticleCollisionFilterId                                       = std::uint32_t;
constexpr ParticleCollisionFilterId kInvalidParticleCollisionFilterId = 0u;

using ParticleSequenceId                                = std::uint32_t;
constexpr ParticleSequenceId kInvalidParticleSequenceId = 0u;

using SuturingSequenceId                                = std::uint32_t;
constexpr SuturingSequenceId kInvalidSuturingSequenceId = 0u;

enum class RigidJointDriveMode : std::uint32_t
{
    None           = 0u,
    TargetPosition = 1u,
    TargetVelocity = 2u,
};

enum class SoftBodySourceKind : std::uint32_t
{
    RegularGrid       = 0u,
    TetMesh           = 1u,
    TetGenFiles       = 2u,
    MeshfreeParticles = 3u,
};

enum class FluidSourceKind : std::uint32_t
{
    RegularGrid = 0u,
};

enum class ParticleKind : std::uint32_t
{
    SoftSolid = 0u,
    Fluid     = 1u,
};

enum class ParticleOwnerType : std::uint32_t
{
    None      = 0u,
    SoftBody  = 1u,
    FluidBody = 2u,
    Strand    = 3u,
    RigidBody = 4u,
};

enum class AuthoredParticleReferenceType : std::uint32_t
{
    SoftBodyParticle   = 0u,
    StrandParticle     = 1u,
    RigidProxyParticle = 2u,
};

enum class ParticleStrandRole : std::uint32_t
{
    None       = 0u,
    NeedleTip  = 1u,
    NeedleBody = 2u,
    Thread     = 3u,
};

enum class SuturingInsertionState : std::uint32_t
{
    Outside = 0u,
    Inside  = 1u,
};

constexpr std::uint32_t kInvalidSuturingIndex = 0xffffffffu;

struct GpuSuturingInsertionState
{
    std::uint32_t state            = static_cast<std::uint32_t>(SuturingInsertionState::Outside);
    std::uint32_t softBodyIndex    = kInvalidSuturingIndex;
    std::uint32_t tetIndex         = kInvalidSuturingIndex;
    std::uint32_t pathIndex        = kInvalidSuturingIndex;
    std::uint32_t nearestNodeIndex = kInvalidSuturingIndex;
    std::uint32_t reserved0        = 0u;
    std::uint32_t reserved1        = 0u;
    std::uint32_t reserved2        = 0u;
    Diligent::float4 barycentrics{0.0f, 0.0f, 0.0f, 0.0f};
};

struct SuturingPathHeader
{
    std::uint32_t suturingGroupId          = kInvalidSuturingIndex;
    std::uint32_t softBodyIndex            = kInvalidSuturingIndex;
    std::uint32_t nodeStart                = 0u;
    std::uint32_t nodeCount                = 0u;
    std::uint32_t flags                    = 0u;
    std::uint32_t needleTangentialDragBits = 0u;
    std::uint32_t threadTangentialDragBits = 0u;
    std::uint32_t reserved2                = 0u;
};

struct SuturingPathNode
{
    std::uint32_t softBodyIndex = 0xffffffffu;
    std::uint32_t tetIndex      = 0xffffffffu;
    Diligent::float4 barycentrics{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float3 tangent{0.0f, 0.0f, 0.0f};
    float arcLength = 0.0f;
};

struct StrandSoftSuturingPair
{
    std::uint32_t suturingGroupId          = kInvalidSuturingIndex;
    std::uint32_t softBodyIndex            = kInvalidSuturingIndex;
    std::uint32_t strandParticleStart      = 0u;
    std::uint32_t strandParticleCount      = 0u;
    std::uint32_t tipParticleIndex         = kInvalidSuturingIndex;
    std::uint32_t softTetStart             = 0u;
    std::uint32_t softTetCount             = 0u;
    std::uint32_t pathStart                = 0u;
    std::uint32_t pathCount                = 0u;
    std::uint32_t nodeStart                = 0u;
    std::uint32_t nodeCount                = 0u;
    std::uint32_t activePathIndex          = kInvalidSuturingIndex;
    std::uint32_t environmentIndex         = 0u;
    float pathNodeSpacing                  = 0.0f;
    std::uint32_t needleTangentialDragBits = 0u;
    std::uint32_t threadTangentialDragBits = 0u;
};

struct SoftBodyRegularGridSource
{
    Diligent::float3 size{1.0f, 1.0f, 1.0f};
    float targetParticleSpacing = 0.25f;
    std::vector<std::uint32_t> staticParticleIndices;
};

struct SoftBodyTetMeshSource
{
    std::vector<Diligent::float3> objectSpaceRestPositions;
    std::vector<std::uint32_t> tetVertexIndices;
    std::vector<std::uint32_t> staticParticleIndices;
};

struct SoftBodyTetGenSource
{
    std::string nodeFile;
    std::string eleFile;
    std::vector<std::uint32_t> staticParticleIndices;
};

struct SoftBodyMeshfreeParticleSource
{
    std::vector<Diligent::float3> particleRestPositions;
    std::vector<Diligent::float3> surfaceRestPositions;
    std::vector<Diligent::float3> surfaceNormals;
    std::vector<Diligent::uint3> surfaceTriangles;
    std::vector<std::uint32_t> staticParticleIndices;
    std::uint32_t neighbourCount = 12u;
};

struct SoftBodySourceDesc
{
    SoftBodySourceKind kind = SoftBodySourceKind::RegularGrid;
    SoftBodyRegularGridSource regularGrid;
    SoftBodyTetMeshSource tetMesh;
    SoftBodyTetGenSource tetGen;
    SoftBodyMeshfreeParticleSource meshfreeParticles;
};

struct FluidRegularGridSource
{
    Diligent::float3 size{1.0f, 1.0f, 1.0f};
    float targetParticleSpacing = 0.25f;
};

struct FluidSourceDesc
{
    FluidSourceKind kind = FluidSourceKind::RegularGrid;
    FluidRegularGridSource regularGrid;
};

struct ParticleContactMaterialDesc
{
    float friction       = 0.0f;
    float restitution    = 0.0f;
    float damping        = 0.0f;
    float staticFriction = -1.0f;
};

struct SoftBodyMaterialDesc
{
    ParticleContactMaterialDesc contact{};
};

struct StrandMaterialDesc
{
    ParticleContactMaterialDesc contact{};
};

struct FluidMaterialDesc
{
    ParticleContactMaterialDesc contact{};
    float viscosity            = 0.01f;
    float cohesion             = 0.0f;
    float surfaceTension       = 0.0f;
    float vorticityConfinement = 0.0f;
    float gravityScale         = 1.0f;
    float cflCoefficient       = 1.0f;
};

struct FluidMaterialGpu
{
    float restDensity                   = 1000.0f;
    float invRestDensity                = 1.0f / 1000.0f;
    float smoothingRadius               = 0.4f;
    float densityConstraintScaleDerived = 1.0f;
    float viscosityDerived              = 0.01f / 1000.0f;
    float cohesionDerived               = 0.0f;
    float cohesion1                     = 0.0f;
    float cohesion2                     = 0.0f;
    float surfaceTensionDerived         = 0.0f;
    float vorticityConfinementDerived   = 0.0f;
    float gravityScale                  = 1.0f;
    float cflRadius                     = 0.25f;

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

struct RigidBodyState
{
    RigidBodyId rigidBodyId        = kInvalidRigidBodyId;
    common::EntityId entityId      = common::kInvalidEntityId;
    std::uint32_t environmentIndex = 0u;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float3 scale{1.0f, 1.0f, 1.0f};
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};
    RigidBodyType bodyType = RigidBodyType::Dynamic;
    float inverseMass      = 1.0f;
    Diligent::float3 kinematicTargetPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF kinematicTargetRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool kinematicTargetEnabled = false;
    float proxyParticleRadius   = 0.0f;
    ParticleContactMaterialDesc proxyParticleMaterial{};
    std::uint32_t proxyCollisionLayer        = 1u;
    std::uint32_t proxyCollisionMask         = 0xffffffffu;
    std::uint32_t proxyParticleOffset        = 0u;
    std::uint32_t proxyParticleCount         = 0u;
    std::uint32_t proxyParticleMaterialIndex = 0u;
    bool suturingEnabled                     = false;
    std::uint32_t needleTipProxyIndex        = 0u;
    Diligent::float4 proxyParticleContactMaterial{0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<Diligent::float3> proxyParticleLocalPositions;
};

struct ColliderState
{
    ColliderId colliderId        = kInvalidColliderId;
    common::EntityId entityId    = common::kInvalidEntityId;
    RigidBodyId ownerRigidBodyId = kInvalidRigidBodyId;
    ColliderShapeType shapeType  = ColliderShapeType::Sphere;
    Diligent::float4 shapeParams{0.5f, 0.0f, 0.0f, 0.0f};
    Diligent::float3 localPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool enabled                 = true;
    float friction               = 0.5f;
    float staticFriction         = -1.0f;
    float restitution            = 0.0f;
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
};

struct SoftBodyState
{
    common::EntityId entityId      = common::kInvalidEntityId;
    std::uint32_t environmentIndex = 0u;
    std::uint32_t collisionLayer   = 1u;
    std::uint32_t collisionMask    = 0xffffffffu;
    bool supportsSuturing          = false;
    SoftBodySourceDesc source{};
    SoftBodyMaterialDesc material{};
    common::Transform restTransform{};
    float particleMass                 = 1.0f;
    float particleRadius               = 0.125f;
    float edgeCompliance               = 0.0f;
    float volumeCompliance             = 0.0f;
    bool simulated                     = true;
    bool selfCollisionEnabled          = false;
    std::uint32_t contactMaterialIndex = 0u;
    std::uint32_t particleOffset       = 0u;
    std::uint32_t particleCount        = 0u;
    std::uint32_t edgeOffset           = 0u;
    std::uint32_t edgeCount            = 0u;
    std::uint32_t tetOffset            = 0u;
    std::uint32_t tetCount             = 0u;
    std::vector<Diligent::float3> restPositions;
    std::vector<Diligent::uint3> boundaryFaces;
};

struct StrandState
{
    common::EntityId entityId      = common::kInvalidEntityId;
    std::uint32_t environmentIndex = 0u;
    std::uint32_t collisionLayer   = 1u;
    std::uint32_t collisionMask    = 0xffffffffu;
    bool suturingEnabled           = false;
    float pathNodeSpacing          = 0.2f;
    StrandMaterialDesc material{};
    float particleMass                 = 1.0f;
    float particleRadius               = 0.125f;
    float stretchShearCompliance       = 0.0f;
    float bendCompliance               = 0.0f;
    float twistCompliance              = 0.0f;
    float distanceCompliance           = 0.0f;
    enum class DistanceSolverMode : std::uint32_t
    {
        None   = 0u,
        Jacobi = 1u,
        Direct = 2u,
    };
    DistanceSolverMode distanceSolverMode = DistanceSolverMode::Jacobi;
    Diligent::float3 rootMaterialNormal{0.0f, 1.0f, 0.0f};
    bool simulated                     = true;
    bool selfCollisionEnabled          = false;
    std::uint32_t contactMaterialIndex = 0u;
    std::uint32_t particleOffset       = 0u;
    std::uint32_t particleCount        = 0u;
    std::uint32_t segmentOffset        = 0u;
    std::uint32_t segmentCount         = 0u;
    std::uint32_t jointOffset          = 0u;
    std::uint32_t jointCount           = 0u;
    std::vector<Diligent::float3> restPositions;
    std::vector<std::uint32_t> staticParticleIndices;
};

struct FluidState
{
    common::EntityId entityId      = common::kInvalidEntityId;
    std::uint32_t environmentIndex = 0u;
    std::uint32_t collisionLayer   = 1u;
    std::uint32_t collisionMask    = 0xffffffffu;
    FluidSourceDesc source{};
    FluidMaterialDesc material{};
    Diligent::float4 visualColor{0.32f, 0.62f, 0.95f, 0.72f};
    common::Transform restTransform{};
    float particleMass                 = 1.0f;
    float particleRadius               = 0.125f;
    bool simulated                     = true;
    std::uint32_t contactMaterialIndex = 0u;
    std::uint32_t fluidMaterialIndex   = 0u;
    std::uint32_t particleOffset       = 0u;
    std::uint32_t particleCount        = 0u;
    std::vector<Diligent::float3> restPositions;
};

struct AuthoredParticleReference
{
    common::EntityId entityId          = common::kInvalidEntityId;
    AuthoredParticleReferenceType type = AuthoredParticleReferenceType::StrandParticle;
    std::uint32_t localParticleIndex   = 0u;
};

struct AuthoredParticleDistanceConstraintState
{
    ParticleConstraintId constraintId = kInvalidParticleConstraintId;
    AuthoredParticleReference particleA{};
    AuthoredParticleReference particleB{};
    float restLength = 0.0f;
    float compliance = 0.0f;
    bool enabled     = true;
};

struct AuthoredRigidDistanceConstraintState
{
    RigidDistanceConstraintId constraintId = kInvalidRigidDistanceConstraintId;
    common::EntityId entityA               = common::kInvalidEntityId;
    common::EntityId entityB               = common::kInvalidEntityId;
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};
    float restDistance = 0.0f;
    float compliance   = 0.0f;
    bool enabled       = true;
};

struct AuthoredRigidParticleAttachmentConstraintState
{
    RigidParticleAttachmentConstraintId constraintId =
        kInvalidRigidParticleAttachmentConstraintId;
    AuthoredParticleReference particle{};
    common::EntityId rigidBodyEntityId = common::kInvalidEntityId;
    Diligent::float3 localAnchor{0.0f, 0.0f, 0.0f};
    float compliance = 0.0f;
    bool enabled     = true;
};

struct AuthoredRoutedCableRoutePoint
{
    common::EntityId entityId = common::kInvalidEntityId;
    Diligent::float3 localGuideOffset{0.0f, 0.0f, 0.0f};

    constexpr bool operator==(const AuthoredRoutedCableRoutePoint &rhs) const noexcept
    {
        return entityId == rhs.entityId && localGuideOffset == rhs.localGuideOffset;
    }
};

struct AuthoredRoutedCableConstraintState
{
    RoutedCableConstraintId constraintId = kInvalidRoutedCableConstraintId;
    std::vector<AuthoredRoutedCableRoutePoint> routePoints{};
    float targetLength = 0.0f;
    float compliance   = 0.0f;
    bool tensionOnly   = true;
    bool enabled       = true;
};

struct AuthoredParticleCollisionFilterState
{
    ParticleCollisionFilterId filterId = kInvalidParticleCollisionFilterId;
    AuthoredParticleReference particle{};
    std::uint32_t collisionLayer = 1u;
    std::uint32_t collisionMask  = 0xffffffffu;
    bool enabled                 = true;
};

struct AuthoredParticleSequenceState
{
    ParticleSequenceId sequenceId = kInvalidParticleSequenceId;
    std::vector<AuthoredParticleReference> entries{};
    bool enabled = true;
};

struct AuthoredSuturingSequenceState
{
    SuturingSequenceId sequenceId = kInvalidSuturingSequenceId;
    std::vector<AuthoredParticleReference> entries{};
    // The selected tip entry authors the suturing path. In the current prototype,
    // the sequence tip and tail also suppress same-soft-body exterior contact.
    std::uint32_t tipEntryIndex = 0u;
    float pathNodeSpacing       = 0.0f;
    float needleTangentialDrag  = 0.0f;
    float threadTangentialDrag  = 0.0f;
    bool enabled                = true;
};

struct DeformableDistanceConstraint
{
    std::uint32_t particleA = 0u;
    std::uint32_t particleB = 0u;
    float restLength        = 0.0f;
    float compliance        = 0.0f;
};

using SoftEdge = DeformableDistanceConstraint;

struct DeformableVolumeConstraint
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u};
    float restVolume        = 0.0f;
    float compliance        = 0.0f;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

using SoftTet = DeformableVolumeConstraint;

struct DeformableBendConstraint
{
    std::uint32_t particle0 = 0u;
    std::uint32_t particle1 = 0u;
    std::uint32_t particle2 = 0u;
    float restAngle         = 0.0f;
    float compliance        = 0.0f;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint32_t reserved2 = 0u;
};

using SoftBend = DeformableBendConstraint;

struct StrandSegmentConstraint
{
    std::uint32_t particleA = 0u;
    std::uint32_t particleB = 0u;
    float restLength        = 0.0f;
    float stretchShearCompliance = 0.0f;
    Diligent::float4 restOrientation{0.0f, 0.0f, 0.0f, 1.0f};
};

struct StrandJointConstraint
{
    std::uint32_t segmentA  = 0u;
    std::uint32_t segmentB  = 0u;
    float bendCompliance    = 0.0f;
    float twistCompliance   = 0.0f;
    Diligent::float4 restRelativeOrientation{0.0f, 0.0f, 0.0f, 1.0f};
};

struct StrandDistanceConstraint
{
    std::uint32_t particleA          = 0u;
    std::uint32_t particleB          = 0u;
    float restLength                 = 0.0f;
    float distanceCompliance = 0.0f;
    std::uint32_t solverMode         = 0u;
    std::uint32_t reserved0          = 0u;
    std::uint32_t reserved1          = 0u;
    std::uint32_t reserved2          = 0u;
};

struct StrandSegmentState
{
    Diligent::float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
};

struct RoutedCableConstraint
{
    std::uint32_t routePointStart = 0u;
    std::uint32_t routePointCount = 0u;
    float targetLength            = 0.0f;
    float compliance              = 0.0f;
    std::uint32_t tensionOnly     = 1u;
    std::uint32_t reserved0       = 0u;
    std::uint32_t reserved1       = 0u;
    std::uint32_t reserved2       = 0u;
};

struct RoutedCableRoutePoint
{
    std::uint32_t rigidBodyIndex = 0u;
    std::uint32_t reserved0      = 0u;
    std::uint32_t reserved1      = 0u;
    std::uint32_t reserved2      = 0u;
    Diligent::float4 localGuideOffset{0.0f, 0.0f, 0.0f, 0.0f};
};

struct RigidDistanceConstraint
{
    std::uint32_t rigidBodyIndexA = 0u;
    std::uint32_t rigidBodyIndexB = 0u;
    float restDistance            = 0.0f;
    float compliance              = 0.0f;
    Diligent::float4 localAnchorA{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 localAnchorB{0.0f, 0.0f, 0.0f, 0.0f};
};

struct RigidParticleAttachmentConstraint
{
    std::uint32_t particleIndex  = 0u;
    std::uint32_t rigidBodyIndex = 0u;
    float compliance             = 0.0f;
    std::uint32_t reserved0      = 0u;
    Diligent::float4 localAnchor{0.0f, 0.0f, 0.0f, 0.0f};
};

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

struct RigidBodySoAHost
{
    std::vector<RigidBodyId> rigidBodyIds;
    std::vector<common::EntityId> entityIds;
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

struct BallJointState
{
    BallJointId jointId                  = kInvalidBallJointId;
    bool enabled                         = true;
    bool suppressConnectedBodyCollisions = false;
    RigidBodyId bodyA                    = kInvalidRigidBodyId;
    RigidBodyId bodyB                    = kInvalidRigidBodyId;
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};
};

struct HingeJointState
{
    HingeJointId jointId                 = kInvalidHingeJointId;
    bool enabled                         = true;
    bool suppressConnectedBodyCollisions = false;
    RigidJointDriveMode driveMode        = RigidJointDriveMode::None;
    bool limitEnabled                    = false;
    RigidBodyId bodyA                    = kInvalidRigidBodyId;
    RigidBodyId bodyB                    = kInvalidRigidBodyId;
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF localRotationA{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::QuaternionF localRotationB{0.0f, 0.0f, 0.0f, 1.0f};
    float limitMin                   = 0.0f;
    float limitMax                   = 0.0f;
    float constraintCompliance       = 0.0f;
    float driveCompliance            = 0.0f;
    float driveTargetAngle           = 0.0f;
    float driveTargetAngularVelocity = 0.0f;
};

struct SliderJointState
{
    SliderJointId jointId                = kInvalidSliderJointId;
    bool enabled                         = true;
    bool suppressConnectedBodyCollisions = false;
    RigidJointDriveMode driveMode        = RigidJointDriveMode::None;
    bool limitEnabled                    = false;
    RigidBodyId bodyA                    = kInvalidRigidBodyId;
    RigidBodyId bodyB                    = kInvalidRigidBodyId;
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF localRotationA{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::QuaternionF localRotationB{0.0f, 0.0f, 0.0f, 1.0f};
    float limitMin             = 0.0f;
    float limitMax             = 0.0f;
    float constraintCompliance = 0.0f;
    float driveCompliance      = 0.0f;
    float driveTargetPosition  = 0.0f;
    float driveTargetVelocity  = 0.0f;
};

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

struct HingeJointSoAHost
{
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<std::uint32_t> driveModes;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;
    std::vector<Diligent::float4> localAxesA0;
    std::vector<std::uint32_t> limitEnabledFlags;
    std::vector<float> limitMins;
    std::vector<float> limitMaxs;
    std::vector<float> constraintCompliances;
    std::vector<float> driveCompliances;
    std::vector<float> driveTargetAngles;
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
        limitEnabledFlags.clear();
        limitMins.clear();
        limitMaxs.clear();
        constraintCompliances.clear();
        driveCompliances.clear();
        driveTargetAngles.clear();
        driveTargetAngularVelocities.clear();
        projectionRow0.clear();
        projectionRow1.clear();
        projectionRow2.clear();
    }
};

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

struct RigidJointSceneHost
{
    BallJointSoAHost ball;
    HingeJointSoAHost hinge;
    SliderJointSoAHost slider;

    void clear()
    {
        ball.clear();
        hinge.clear();
        slider.clear();
    }
};

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

struct SoftRenderVertexTriangleRange
{
    std::uint32_t start     = 0u;
    std::uint32_t count     = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct SoftRenderVertexBinding
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u};
    Diligent::float4 weights{1.0f, 0.0f, 0.0f, 0.0f};
};

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

struct RigidBodySoAGpu
{
    std::uint32_t bodyCount = 0;
    std::uint32_t capacity  = 0;
};

struct ColliderSoAGpu
{
    std::uint32_t colliderCount = 0;
    std::uint32_t capacity      = 0;
};

struct PhysicsGpuBuffers
{
    RigidBodySoAHost rigidBodies{};
    ColliderSoAHost colliders{};
    RigidBodySoAGpu rigidBodyGpu{};
    ColliderSoAGpu colliderGpu{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
