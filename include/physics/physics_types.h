#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H

#include "common/id.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

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

struct RigidBodyState
{
    common::EntityId entityId = common::kInvalidEntityId;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float3 scale{1.0f, 1.0f, 1.0f};
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 inverseInertiaLocal{1.0f, 1.0f, 1.0f};
    RigidBodyType bodyType          = RigidBodyType::Dynamic;
    float inverseMass               = 1.0f;
    ColliderShapeType colliderShape = ColliderShapeType::Sphere;
    Diligent::float4 colliderParams{0.5f, 0.0f, 0.0f, 0.0f};
    Diligent::float3 kinematicTargetPosition{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF kinematicTargetRotation{0.0f, 0.0f, 0.0f, 1.0f};
    bool kinematicTargetEnabled = false;
};

struct PhysicsSoADirtyRange
{
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    bool valid          = false;

    void clear() noexcept
    {
        begin = 0;
        end   = 0;
        valid = false;
    }

    void include(std::uint32_t index) noexcept
    {
        if (!valid)
        {
            begin = index;
            end   = index + 1u;
            valid = true;
            return;
        }

        if (index < begin)
        {
            begin = index;
        }
        if (index + 1u > end)
        {
            end = index + 1u;
        }
    }
};

struct RigidBodySoAHost
{
    std::vector<common::EntityId> entityIds;
    std::vector<Diligent::float4> positionsInvMass;
    std::vector<Diligent::float4> orientations;
    std::vector<Diligent::float4> scales;
    std::vector<Diligent::float4> linearVelocities;
    std::vector<Diligent::float4> angularVelocities;
    std::vector<Diligent::float4> inverseInertiaLocal;
    std::vector<std::uint32_t> bodyTypes;
    std::vector<std::uint32_t> colliderShapeTypes;
    std::vector<Diligent::float4> colliderParams;
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
        entityIds.clear();
        positionsInvMass.clear();
        orientations.clear();
        scales.clear();
        linearVelocities.clear();
        angularVelocities.clear();
        inverseInertiaLocal.clear();
        bodyTypes.clear();
        colliderShapeTypes.clear();
        colliderParams.clear();
        kinematicTargetPositions.clear();
        kinematicTargetOrientations.clear();
        kinematicTargetFlags.clear();
    }
};

struct RigidBodySoAGpu
{
    std::uint32_t bodyCount = 0;
    std::uint32_t capacity  = 0;
    PhysicsSoADirtyRange dirtyRange{};
};

struct PhysicsGpuBuffers
{
    RigidBodySoAHost rigidBodies{};
    RigidBodySoAGpu rigidBodyGpu{};
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
