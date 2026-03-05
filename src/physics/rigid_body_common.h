#ifndef CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H
#define CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H

#include "physics/physics_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>

namespace cressim::neo::physics
{

constexpr std::uint32_t kRigidContactsPerPair = 4u;

struct GpuRigidDispatchConstants
{
    float dt                     = 0.0f;
    std::uint32_t rigidBodyCount = 0;
    std::uint32_t pairCount      = 0;
    std::uint32_t substepIndex   = 0;
    std::uint32_t iterationIndex = 0;
    std::uint32_t solverIterations = 0;
    std::uint32_t reserved0        = 0;
    std::uint32_t reserved1        = 0;
};

struct GpuRigidContact
{
    std::uint32_t bodyA = 0;
    std::uint32_t bodyB = 0;
    std::uint32_t active = 0;
    std::uint32_t reserved = 0;
    Diligent::float4 normalPenetration{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 localPointA{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 localPointB{0.0f, 0.0f, 0.0f, 0.0f};
};

static_assert(sizeof(GpuRigidDispatchConstants) == 32u);
static_assert(sizeof(GpuRigidContact) == 64u);

struct EffectiveColliderDimensions
{
    float sphereRadius = 0.0f;
    Diligent::float3 boxHalfExtents{0.0f, 0.0f, 0.0f};
    float capsuleRadius     = 0.0f;
    float capsuleHalfHeight = 0.0f;
};

std::uint32_t computeRigidPairCount(std::uint32_t bodyCount) noexcept;

EffectiveColliderDimensions computeEffectiveColliderDimensions(
    ColliderShapeType shape, const Diligent::float4& colliderParams,
    const Diligent::float3& scale) noexcept;

Diligent::QuaternionF rotationVectorToQuaternion(const Diligent::float3& rotationVector) noexcept;

Diligent::QuaternionF integrateOrientation(const Diligent::QuaternionF& orientation,
                                           const Diligent::float3& angularVelocity,
                                           float dt) noexcept;

Diligent::float3 angularVelocityFromOrientationDelta(
    const Diligent::QuaternionF& previous, const Diligent::QuaternionF& current,
    float dt) noexcept;

Diligent::float3 multiplyWorldInverseInertia(const Diligent::float3& inverseInertiaLocal,
                                             const Diligent::QuaternionF& orientation,
                                             const Diligent::float3& vector) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H
