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
    float dt                            = 0.0f;
    std::uint32_t rigidBodyCount        = 0;
    std::uint32_t activeDynamicCount    = 0;
    std::uint32_t candidatePairCount    = 0;
    std::uint32_t candidatePairCapacity = 0;
    std::uint32_t substepIndex          = 0;
    std::uint32_t iterationIndex        = 0;
    std::uint32_t solverIterations      = 0;
};

struct GpuBodyAabb
{
    Diligent::float4 minBounds{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 maxBounds{0.0f, 0.0f, 0.0f, 0.0f};
};

struct GpuBodyMeta
{
    std::uint32_t bodyId      = 0;
    std::uint32_t flags       = 0;
    std::uint32_t activeIndex = 0xffffffffu;
    std::uint32_t reserved    = 0;
};

struct GpuBroadPhaseElement
{
    std::uint32_t primitiveIdx = 0;
    float aabbMinX             = 0.0f;
    float aabbMinY             = 0.0f;
    float aabbMinZ             = 0.0f;
    float aabbMaxX             = 0.0f;
    float aabbMaxY             = 0.0f;
    float aabbMaxZ             = 0.0f;
    float reserved             = 0.0f;
};

struct GpuMortonCodeElement
{
    std::uint32_t mortonCode = 0;
    std::uint32_t elementIdx = 0;
};

struct GpuBvhNode
{
    std::int32_t left          = -1;
    std::int32_t right         = -1;
    std::uint32_t primitiveIdx = 0;
    float aabbMinX             = 0.0f;
    float aabbMinY             = 0.0f;
    float aabbMinZ             = 0.0f;
    float aabbMaxX             = 0.0f;
    float aabbMaxY             = 0.0f;
    float aabbMaxZ             = 0.0f;
    float reserved             = 0.0f;
};

struct GpuBvhConstructionInfo
{
    std::uint32_t parent         = 0;
    std::int32_t visitationCount = 0;
};

struct GpuCandidatePair
{
    std::uint32_t bodyA     = 0;
    std::uint32_t bodyB     = 0;
    std::uint32_t reserved0 = 0;
    std::uint32_t reserved1 = 0;
};

struct GpuBroadPhaseMeta
{
    std::uint32_t activeDynamicCount = 0;
    std::uint32_t candidatePairCount = 0;
    std::uint32_t requiredPairCount  = 0;
    std::uint32_t overflow           = 0;
};

struct GpuRigidContact
{
    std::uint32_t bodyA    = 0;
    std::uint32_t bodyB    = 0;
    std::uint32_t active   = 0;
    std::uint32_t reserved = 0;
    Diligent::float4 normalPenetration{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 localPointA{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 localPointB{0.0f, 0.0f, 0.0f, 0.0f};
};

static_assert(sizeof(GpuRigidDispatchConstants) == 32u);
static_assert(sizeof(GpuBodyAabb) == 32u);
static_assert(sizeof(GpuBodyMeta) == 16u);
static_assert(sizeof(GpuBroadPhaseElement) == 32u);
static_assert(sizeof(GpuMortonCodeElement) == 8u);
static_assert(sizeof(GpuBvhNode) == 40u);
static_assert(sizeof(GpuBvhConstructionInfo) == 8u);
static_assert(sizeof(GpuCandidatePair) == 16u);
static_assert(sizeof(GpuBroadPhaseMeta) == 16u);
static_assert(sizeof(GpuRigidContact) == 64u);

struct EffectiveColliderDimensions
{
    float sphereRadius = 0.0f;
    Diligent::float3 boxHalfExtents{0.0f, 0.0f, 0.0f};
    float capsuleRadius     = 0.0f;
    float capsuleHalfHeight = 0.0f;
};

std::uint32_t estimateRigidCandidatePairCapacity(std::uint32_t bodyCount) noexcept;

EffectiveColliderDimensions computeEffectiveColliderDimensions(
    ColliderShapeType shape, const Diligent::float4& colliderParams,
    const Diligent::float3& scale) noexcept;

Diligent::QuaternionF rotationVectorToQuaternion(const Diligent::float3& rotationVector) noexcept;

Diligent::QuaternionF integrateOrientation(const Diligent::QuaternionF& orientation,
                                           const Diligent::float3& angularVelocity,
                                           float dt) noexcept;

Diligent::float3 angularVelocityFromOrientationDelta(const Diligent::QuaternionF& previous,
                                                     const Diligent::QuaternionF& current,
                                                     float dt) noexcept;

Diligent::float3 multiplyWorldInverseInertia(const Diligent::float3& inverseInertiaLocal,
                                             const Diligent::QuaternionF& orientation,
                                             const Diligent::float3& vector) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H
