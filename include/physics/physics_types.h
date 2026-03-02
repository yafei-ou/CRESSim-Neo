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

struct RigidBodyState
{
    common::EntityId entityId = common::kInvalidEntityId;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    Diligent::float3 angularVelocity{0.0f, 0.0f, 0.0f};
    float inverseMass               = 1.0f;
    ColliderShapeType colliderShape = ColliderShapeType::Sphere;
    Diligent::float4 colliderParams{0.5f, 0.0f, 0.0f, 0.0f};
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
    std::vector<Diligent::float4> linearVelocities;
    std::vector<Diligent::float4> angularVelocities;
    std::vector<std::uint32_t> colliderShapeTypes;
    std::vector<Diligent::float4> colliderParams;

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
        linearVelocities.clear();
        angularVelocities.clear();
        colliderShapeTypes.clear();
        colliderParams.clear();
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
