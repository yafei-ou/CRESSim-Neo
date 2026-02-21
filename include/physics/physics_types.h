#ifndef CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
#define CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H

#include "common/id.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <vector>

namespace cressim::neo::physics
{

enum class ColliderShapeType
{
    Sphere,
    Box,
    Capsule,
};

struct RigidBodyState
{
    common::EntityId entityId = common::kInvalidEntityId;
    Diligent::float3 position{0.0f, 0.0f, 0.0f};
    Diligent::QuaternionF rotation{0.0f, 0.0f, 0.0f, 1.0f};
    Diligent::float3 linearVelocity{0.0f, 0.0f, 0.0f};
    float inverseMass = 1.0f;
};

struct PhysicsGpuBuffers
{
    // TODO: promote to full PBD SoA buffers (constraints, contacts, lambdas, scratch).
    std::vector<RigidBodyState> rigidBodies;
};

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_PHYSICS_TYPES_H
