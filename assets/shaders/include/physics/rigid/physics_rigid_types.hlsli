#ifndef CRESSIM_NEO_PHYSICS_RIGID_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_TYPES_HLSLI

#include "../collision/physics_shape_common.hlsli"
#include "../shared/physics_indirect_dispatch.hlsli"

static const uint kRigidBodyTypeStatic = 0u;
static const uint kRigidBodyTypeKinematic = 1u;
static const uint kRigidBodyTypeDynamic = 2u;
static const uint kKinematicTargetEnabled = 1u << 0u;

struct GpuRigidContact
{
    uint bodyA;
    uint bodyB;
    uint active;
    uint reserved;
    float4 normalPenetration;
    float4 localPointA;
    float4 localPointB;
    float4 material;
};

struct GpuBodyAabb
{
    float4 minBounds;
    float4 maxBounds;
};

struct GpuBodyMeta
{
    uint bodyId;
    uint bodyType;
    uint activeIndex;
    uint reserved0;
};

struct GpuCandidatePair
{
    uint colliderA;
    uint colliderB;
    uint reserved0;
    uint reserved1;
};

struct GpuRigidPairRange
{
    uint type;
    uint start;
    uint count;
    uint reserved;
};

struct GpuNarrowPhaseChunk
{
    uint pairType;
    uint pairStart;
    uint pairCount;
    uint reserved;
};

struct GpuNarrowPhaseMeta
{
    uint chunkCount;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct GpuColliderContactData
{
    uint ownerBody;
    uint shapeType;
    uint reserved0;
    uint reserved1;
    float4 shapeParams;
    float4 localPosition;
    float4 localOrientation;
    float4 material;
};

struct GpuBallJoint
{
    uint bodyA;
    uint bodyB;
    uint enabled;
    uint reserved0;
    float4 localAnchorA;
    float4 localAnchorB;
};

struct GpuHingeJoint
{
    uint bodyA;
    uint bodyB;
    uint enabled;
    uint reserved0;
    float4 localAnchorA;
    float4 localAnchorB;
    float4 localAxisA;
    float4 localAxisB;
    float4 projectionRow0;
    float4 projectionRow1;
};

struct GpuSliderJoint
{
    uint bodyA;
    uint bodyB;
    uint enabled;
    uint reserved0;
    float4 localAxisA0;
    float4 localAxisA1;
    float4 localAxisA2;
    float4 localAxisB0;
    float4 localAxisB1;
    float4 localAxisB2;
    float4 referenceOffset;
};

uint ComputeRigidPairType(uint shapeTypeA, uint shapeTypeB)
{
    const uint lo = min(shapeTypeA, shapeTypeB);
    const uint hi = max(shapeTypeA, shapeTypeB);

    if (lo == kColliderSphere && hi == kColliderSphere)
        return 0u;
    if (lo == kColliderSphere && hi == kColliderBox)
        return 1u;
    if (lo == kColliderSphere && hi == kColliderCapsule)
        return 2u;
    if (lo == kColliderBox && hi == kColliderBox)
        return 3u;
    if (lo == kColliderBox && hi == kColliderCapsule)
        return 4u;
    return 5u;
}

void CanonicalizeRigidPair(uint bodyA, uint bodyB, uint shapeTypeA, uint shapeTypeB,
                           out uint outBodyA, out uint outBodyB, out uint pairType)
{
    pairType = ComputeRigidPairType(shapeTypeA, shapeTypeB);

    if (shapeTypeA < shapeTypeB)
    {
        outBodyA = bodyA;
        outBodyB = bodyB;
        return;
    }

    if (shapeTypeA > shapeTypeB)
    {
        outBodyA = bodyB;
        outBodyB = bodyA;
        return;
    }

    outBodyA = bodyA;
    outBodyB = bodyB;
}

void PairTypeToShapeTypes(uint pairType, out uint shapeTypeA, out uint shapeTypeB)
{
    if (pairType == 0u)
    {
        shapeTypeA = kColliderSphere;
        shapeTypeB = kColliderSphere;
        return;
    }

    if (pairType == 1u)
    {
        shapeTypeA = kColliderSphere;
        shapeTypeB = kColliderBox;
        return;
    }

    if (pairType == 2u)
    {
        shapeTypeA = kColliderSphere;
        shapeTypeB = kColliderCapsule;
        return;
    }

    if (pairType == 3u)
    {
        shapeTypeA = kColliderBox;
        shapeTypeB = kColliderBox;
        return;
    }

    if (pairType == 4u)
    {
        shapeTypeA = kColliderBox;
        shapeTypeB = kColliderCapsule;
        return;
    }

    shapeTypeA = kColliderCapsule;
    shapeTypeB = kColliderCapsule;
}

void PairIndexToBodies(uint pairIndex, uint bodyCount, out uint bodyA, out uint bodyB)
{
    uint remaining = pairIndex;
    bodyA = 0u;
    bodyB = 0u;

    [loop] for (uint a = 0u; a + 1u < bodyCount; ++a)
    {
        const uint pairsForA = bodyCount - a - 1u;
        if (remaining < pairsForA)
        {
            bodyA = a;
            bodyB = a + 1u + remaining;
            return;
        }
        remaining -= pairsForA;
    }
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_TYPES_HLSLI
