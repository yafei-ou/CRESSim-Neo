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

using RigidJointId                          = std::uint32_t;
constexpr RigidJointId kInvalidRigidJointId = 0u;

enum class RigidJointDriveMode : std::uint32_t
{
    None           = 0u,
    TargetPosition = 1u,
    TargetVelocity = 2u,
};

enum class SoftBodySourceKind : std::uint32_t
{
    RegularGrid = 0u,
    TetMesh     = 1u,
    TetGenFiles = 2u,
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

struct SoftBodySourceDesc
{
    SoftBodySourceKind kind = SoftBodySourceKind::RegularGrid;
    SoftBodyRegularGridSource regularGrid;
    SoftBodyTetMeshSource tetMesh;
    SoftBodyTetGenSource tetGen;
};

struct SoftBodyMaterialDesc
{
    float friction       = 0.0f;
    float restitution    = 0.0f;
    float damping        = 0.0f;
    float staticFriction = -1.0f;
};

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
    SoftBodySourceDesc source{};
    SoftBodyMaterialDesc material{};
    common::Transform restTransform{};
    float particleMass           = 1.0f;
    float particleRadius         = 0.125f;
    float edgeCompliance         = 0.0f;
    float volumeCompliance       = 0.0f;
    bool simulated               = true;
    bool selfCollisionEnabled    = false;
    std::uint32_t particleOffset = 0u;
    std::uint32_t particleCount  = 0u;
    std::uint32_t edgeOffset     = 0u;
    std::uint32_t edgeCount      = 0u;
    std::uint32_t tetOffset      = 0u;
    std::uint32_t tetCount       = 0u;
    std::vector<Diligent::float3> restPositions;
    std::vector<Diligent::uint3> boundaryFaces;
};

struct SoftEdge
{
    std::uint32_t particleA = 0u;
    std::uint32_t particleB = 0u;
    float restLength        = 0.0f;
    float compliance        = 0.0f;
};

struct SoftTet
{
    Diligent::uint4 particleIndices{0u, 0u, 0u, 0u};
    float restVolume        = 0.0f;
    float compliance        = 0.0f;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct SoftParticleSoAHost
{
    std::vector<Diligent::float4> positionsInvMass;
    std::vector<Diligent::float4> previousPositions;
    std::vector<Diligent::float4> velocities;
    std::vector<Diligent::float4> materials;
    std::vector<float> radii;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> owningSoftBodyIndices;
    std::vector<std::uint32_t> phases;
    std::vector<std::uint32_t> collisionLayers;
    std::vector<std::uint32_t> collisionMasks;
    std::vector<std::uint32_t> adjacencyOffsets;
    std::vector<std::uint32_t> adjacencyCounts;
    std::vector<std::uint32_t> adjacencyIndices;

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
        materials.clear();
        radii.clear();
        environmentIndices.clear();
        owningSoftBodyIndices.clear();
        phases.clear();
        collisionLayers.clear();
        collisionMasks.clear();
        adjacencyOffsets.clear();
        adjacencyCounts.clear();
        adjacencyIndices.clear();
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
    RigidJointId jointId                 = kInvalidRigidJointId;
    bool enabled                         = true;
    bool suppressConnectedBodyCollisions = false;
    RigidBodyId bodyA                    = kInvalidRigidBodyId;
    RigidBodyId bodyB                    = kInvalidRigidBodyId;
    Diligent::float3 localAnchorA{0.0f, 0.0f, 0.0f};
    Diligent::float3 localAnchorB{0.0f, 0.0f, 0.0f};
};

struct HingeJointState
{
    RigidJointId jointId                 = kInvalidRigidJointId;
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
    float driveTargetAngle           = 0.0f;
    float driveTargetAngularVelocity = 0.0f;
};

struct SliderJointState
{
    RigidJointId jointId                 = kInvalidRigidJointId;
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
    float limitMin            = 0.0f;
    float limitMax            = 0.0f;
    float driveTargetPosition = 0.0f;
    float driveTargetVelocity = 0.0f;
};

struct BallJointSoAHost
{
    std::vector<RigidJointId> jointIds;
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;

    std::size_t size() const noexcept
    {
        return jointIds.size();
    }
    bool empty() const noexcept
    {
        return jointIds.empty();
    }
    void clear()
    {
        jointIds.clear();
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
    }
};

struct HingeJointSoAHost
{
    std::vector<RigidJointId> jointIds;
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
    std::vector<float> driveTargetAngles;
    std::vector<float> driveTargetAngularVelocities;
    std::vector<Diligent::float4> projectionRow0;
    std::vector<Diligent::float4> projectionRow1;
    std::vector<Diligent::float4> projectionRow2;

    std::size_t size() const noexcept
    {
        return jointIds.size();
    }
    bool empty() const noexcept
    {
        return jointIds.empty();
    }
    void clear()
    {
        jointIds.clear();
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
        driveTargetAngles.clear();
        driveTargetAngularVelocities.clear();
        projectionRow0.clear();
        projectionRow1.clear();
        projectionRow2.clear();
    }
};

struct SliderJointSoAHost
{
    std::vector<RigidJointId> jointIds;
    std::vector<std::uint32_t> bodyIndicesA;
    std::vector<std::uint32_t> bodyIndicesB;
    std::vector<std::uint32_t> enabledFlags;
    std::vector<std::uint32_t> driveModes;
    std::vector<Diligent::float4> localAnchorsA;
    std::vector<Diligent::float4> localAnchorsB;
    std::vector<std::uint32_t> limitEnabledFlags;
    std::vector<float> limitMins;
    std::vector<float> limitMaxs;
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
        return jointIds.size();
    }
    bool empty() const noexcept
    {
        return jointIds.empty();
    }
    void clear()
    {
        jointIds.clear();
        bodyIndicesA.clear();
        bodyIndicesB.clear();
        enabledFlags.clear();
        driveModes.clear();
        localAnchorsA.clear();
        localAnchorsB.clear();
        limitEnabledFlags.clear();
        limitMins.clear();
        limitMaxs.clear();
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

struct SoftRenderDataHost
{
    std::vector<Diligent::float4> fallbackNormals;
    std::vector<SoftRenderVertexTriangleRange> vertexTriangleRanges;
    std::vector<std::uint32_t> vertexTriangleIndices;
    std::vector<Diligent::uint4> triangleParticleIndices;
    std::vector<Diligent::uint2> softBodyParticleRanges;

    void clear()
    {
        fallbackNormals.clear();
        vertexTriangleRanges.clear();
        vertexTriangleIndices.clear();
        triangleParticleIndices.clear();
        softBodyParticleRanges.clear();
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
