#ifndef CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H
#define CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H

#include "physics/physics_types.h"

#include "DiligentEngine/DiligentCore/Common/interface/BasicMath.hpp"

#include <cstdint>

namespace cressim::neo::physics
{

constexpr std::uint32_t kRigidContactsPerPair   = 4u;
constexpr std::uint32_t kRigidPairTypeCount     = 6u;
constexpr std::uint32_t kBodyFlagStatic         = 1u << 0u;
constexpr std::uint32_t kBodyFlagKinematic      = 1u << 1u;
constexpr std::uint32_t kBodyFlagDynamic        = 1u << 2u;
constexpr std::uint32_t kBodyFlagMoving         = kBodyFlagKinematic | kBodyFlagDynamic;
constexpr std::uint32_t kKinematicTargetEnabled = 1u << 0u;

enum class GpuRigidPairType : std::uint32_t
{
    SphereSphere = 0u,
    SphereBox,
    SphereCapsule,
    BoxBox,
    BoxCapsule,
    CapsuleCapsule,
};

struct GpuRigidDispatchConstants
{
    float dt                            = 0.0f;
    std::uint32_t rigidBodyCount        = 0;
    std::uint32_t colliderCount         = 0;
    std::uint32_t activeMovingCount     = 0;
    std::uint32_t staticBodyCount       = 0;
    std::uint32_t candidatePairCount    = 0;
    std::uint32_t candidatePairCapacity = 0;
    std::uint32_t substepIndex          = 0;
    std::uint32_t iterationIndex        = 0;
    std::uint32_t solverIterations      = 0;
    std::uint32_t reserved0             = 0;
    std::uint32_t reserved1             = 0;
};

struct GpuPhysicsScanConstants
{
    std::uint32_t elementCount = 0;
    std::uint32_t reserved0    = 0;
    std::uint32_t reserved1    = 0;
    std::uint32_t reserved2    = 0;
};

struct GpuPhysicsRadixConstants
{
    std::uint32_t elementCount = 0;
    std::uint32_t bitIndex     = 0;
    std::uint32_t reserved0    = 0;
    std::uint32_t reserved1    = 0;
};

struct GpuSoftDispatchConstants
{
    float dt                                = 0.0f;
    std::uint32_t softParticleCount         = 0;
    std::uint32_t rigidSurfaceParticleCount = 0;
    float particleGridCellSize              = 0.0f;
    std::uint32_t softRigidCandidatePairCapacity = 0;
    std::uint32_t softEdgeCount             = 0;
    std::uint32_t softTetCount              = 0;
    std::uint32_t reserved2                 = 0;
};

constexpr std::uint32_t kSoftRigidBroadPhaseParticleTypeSoft         = 0u;
constexpr std::uint32_t kSoftRigidBroadPhaseParticleTypeRigidSurface = 1u;
constexpr std::uint32_t kSoftRigidCandidatePairTypeSoftSoft          = 0u;
constexpr std::uint32_t kSoftRigidCandidatePairTypeSoftRigid         = 1u;

struct GpuSoftRigidBroadPhaseParticle
{
    std::uint32_t cellKey       = 0;
    std::int32_t cellX          = 0;
    std::int32_t cellY          = 0;
    std::int32_t cellZ          = 0;
    std::uint32_t particleIndex = 0;
    std::uint32_t particleType  = 0;
    std::uint32_t ownerIndex    = 0;
    std::uint32_t reserved0     = 0;
};

struct GpuSoftRigidCandidatePair
{
    std::uint32_t pairType = 0;
    std::uint32_t indexA   = 0;
    std::uint32_t indexB   = 0;
    std::uint32_t auxIndex = 0;
};

struct GpuSoftRigidContact
{
    std::uint32_t softParticleIndex = 0;
    std::uint32_t rigidBodyIndex    = 0;
    std::uint32_t colliderIndex     = 0;
    std::uint32_t active            = 0;
    Diligent::float4 normalPenetration{0.0f, 0.0f, 0.0f, 0.0f};
};

struct GpuBroadPhaseBuildConstants
{
    std::uint32_t elementCount = 0;
    std::uint32_t reserved0    = 0;
    std::uint32_t reserved1    = 0;
    std::uint32_t reserved2    = 0;
};

struct GpuBroadPhaseReductionConstants
{
    std::uint32_t elementCount = 0;
    std::uint32_t reserved0    = 0;
    std::uint32_t reserved1    = 0;
    std::uint32_t reserved2    = 0;
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

struct GpuBroadPhaseExtent
{
    Diligent::float4 minBounds{0.0f, 0.0f, 0.0f, 0.0f};
    Diligent::float4 maxBounds{0.0f, 0.0f, 0.0f, 0.0f};
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
    std::uint32_t colliderA = 0;
    std::uint32_t colliderB = 0;
    std::uint32_t reserved0 = 0;
    std::uint32_t reserved1 = 0;
};

struct GpuRigidPairRange
{
    std::uint32_t type     = 0;
    std::uint32_t start    = 0;
    std::uint32_t count    = 0;
    std::uint32_t reserved = 0;
};

struct GpuNarrowPhaseChunk
{
    std::uint32_t pairType  = 0;
    std::uint32_t pairStart = 0;
    std::uint32_t pairCount = 0;
    std::uint32_t reserved  = 0;
};

struct GpuNarrowPhaseMeta
{
    std::uint32_t chunkCount = 0;
    std::uint32_t reserved0  = 0;
    std::uint32_t reserved1  = 0;
    std::uint32_t reserved2  = 0;
};

struct GpuBroadPhaseMeta
{
    std::uint32_t activeMovingCount  = 0;
    std::uint32_t staticBodyCount    = 0;
    std::uint32_t candidatePairCount = 0;
    std::uint32_t requiredPairCount  = 0;
    std::uint32_t overflow           = 0;
    std::uint32_t staticBvhReady     = 0;
    std::uint32_t reserved0          = 0;
    std::uint32_t reserved1          = 0;
};

struct GpuColliderBroadPhaseData
{
    std::uint32_t ownerBody        = 0;
    std::uint32_t shapeType        = 0;
    std::uint32_t environmentIndex = 0;
    std::uint32_t collisionLayer   = 0;
    std::uint32_t collisionMask    = 0;
    std::uint32_t enabledFlag      = 0;
    std::uint32_t reserved1        = 0;
    std::uint32_t reserved2        = 0;
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
    Diligent::float4 material{0.0f, 0.0f, 0.0f, 0.0f};
};

static_assert(sizeof(GpuRigidDispatchConstants) == 48u);
static_assert(sizeof(GpuPhysicsScanConstants) == 16u);
static_assert(sizeof(GpuPhysicsRadixConstants) == 16u);
static_assert(sizeof(GpuSoftDispatchConstants) == 32u);
static_assert(sizeof(GpuSoftRigidBroadPhaseParticle) == 32u);
static_assert(sizeof(GpuSoftRigidCandidatePair) == 16u);
static_assert(sizeof(GpuSoftRigidContact) == 32u);
static_assert(sizeof(GpuBroadPhaseBuildConstants) == 16u);
static_assert(sizeof(GpuBroadPhaseReductionConstants) == 16u);
static_assert(sizeof(GpuBodyAabb) == 32u);
static_assert(sizeof(GpuBodyMeta) == 16u);
static_assert(sizeof(GpuBroadPhaseElement) == 32u);
static_assert(sizeof(GpuMortonCodeElement) == 8u);
static_assert(sizeof(GpuBroadPhaseExtent) == 32u);
static_assert(sizeof(GpuBvhNode) == 40u);
static_assert(sizeof(GpuBvhConstructionInfo) == 8u);
static_assert(sizeof(GpuCandidatePair) == 16u);
static_assert(sizeof(GpuRigidPairRange) == 16u);
static_assert(sizeof(GpuNarrowPhaseChunk) == 16u);
static_assert(sizeof(GpuNarrowPhaseMeta) == 16u);
static_assert(sizeof(GpuBroadPhaseMeta) == 32u);
static_assert(sizeof(GpuColliderBroadPhaseData) == 32u);
static_assert(sizeof(GpuRigidContact) == 80u);

struct EffectiveColliderDimensions
{
    float sphereRadius = 0.0f;
    Diligent::float3 boxHalfExtents{0.0f, 0.0f, 0.0f};
    float capsuleRadius     = 0.0f;
    float capsuleHalfHeight = 0.0f;
};

std::uint32_t estimateRigidCandidatePairCapacity(std::uint32_t bodyCount) noexcept;

EffectiveColliderDimensions computeEffectiveColliderDimensions(
    ColliderShapeType shape, const Diligent::float4 &colliderParams,
    const Diligent::float3 &scale) noexcept;

Diligent::QuaternionF rotationVectorToQuaternion(const Diligent::float3 &rotationVector) noexcept;

Diligent::QuaternionF integrateOrientation(const Diligent::QuaternionF &orientation,
                                           const Diligent::float3 &angularVelocity,
                                           float dt) noexcept;

Diligent::float3 angularVelocityFromOrientationDelta(const Diligent::QuaternionF &previous,
                                                     const Diligent::QuaternionF &current,
                                                     float dt) noexcept;

Diligent::float3 multiplyWorldInverseInertia(const Diligent::float3 &inverseInertiaLocal,
                                             const Diligent::QuaternionF &orientation,
                                             const Diligent::float3 &vector) noexcept;

} // namespace cressim::neo::physics

#endif // CRESSIM_NEO_PHYSICS_RIGID_BODY_COMMON_H
