#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H

#include "common/id.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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

struct UInt3
{
    std::uint32_t x = 1u;
    std::uint32_t y = 1u;
    std::uint32_t z = 1u;
};

struct RigidBodyState
{
    RigidBodyId rigidBodyId   = kInvalidRigidBodyId;
    common::EntityId entityId = common::kInvalidEntityId;
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
    ColliderId colliderId          = kInvalidColliderId;
    common::EntityId entityId      = common::kInvalidEntityId;
    RigidBodyId ownerRigidBodyId   = kInvalidRigidBodyId;
    std::uint32_t environmentIndex = 0u;
    ColliderShapeType shapeType    = ColliderShapeType::Sphere;
    Diligent::float4 shapeParams{0.5f, 0.0f, 0.0f, 0.0f};
    Diligent::float3 localPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF localRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool enabled                 = true;
    float friction               = 0.5f;
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
    UInt3 gridResolution{};
    Diligent::float3 origin{0.0f, 0.0f, 0.0f};
    Diligent::float3 size{1.0f, 1.0f, 1.0f};
    float particleSpacing        = 0.25f;
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
    std::array<std::uint32_t, 4> particleIndices{0u, 0u, 0u, 0u};
    float restVolume = 0.0f;
    float compliance = 0.0f;
};

struct SoftParticleSoAHost
{
    std::vector<Diligent::float4> positionsInvMass;
    std::vector<Diligent::float4> previousPositions;
    std::vector<Diligent::float4> velocities;
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

struct RigidSurfaceParticleSoAHost
{
    std::vector<Diligent::float4> localPositions;
    std::vector<std::uint32_t> owningRigidBodyIndices;
    std::vector<RigidBodyId> owningRigidBodyIds;
    std::vector<std::uint32_t> owningColliderIndices;
    std::vector<ColliderId> owningColliderIds;
    std::vector<float> sampleRadii;
    std::vector<std::uint32_t> environmentIndices;
    std::vector<std::uint32_t> collisionLayers;
    std::vector<std::uint32_t> collisionMasks;

    std::size_t size() const noexcept
    {
        return localPositions.size();
    }

    bool empty() const noexcept
    {
        return localPositions.empty();
    }

    void clear()
    {
        localPositions.clear();
        owningRigidBodyIndices.clear();
        owningRigidBodyIds.clear();
        owningColliderIndices.clear();
        owningColliderIds.clear();
        sampleRadii.clear();
        environmentIndices.clear();
        collisionLayers.clear();
        collisionMasks.clear();
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
