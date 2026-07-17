#ifndef CRESSIM_NEO_PHYSICS_RIGID_BROAD_PHASE_TYPES_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_BROAD_PHASE_TYPES_HLSLI

#include "physics/collision/physics_shape_common.hlsli"

struct GpuBroadPhaseElement
{
    uint primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
    float reserved;
};

struct GpuMortonCodeElement
{
    uint mortonCode;
    uint elementIdx;
};

struct GpuBroadPhaseExtent
{
    float4 minBounds;
    float4 maxBounds;
};

struct GpuBvhNode
{
    int left;
    int right;
    uint primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
    float reserved;
};

struct GpuBvhConstructionInfo
{
    uint parent;
    int visitationCount;
};

struct GpuBroadPhaseMeta
{
    uint activeMovingCount;
    uint staticBodyCount;
    uint candidatePairCount;
    uint requiredPairCount;
    uint overflow;
    uint staticBvhReady;
    uint reserved0;
    uint reserved1;
};

struct GpuProxyRigidContactMeta
{
    uint activeContactCount;
    uint requiredContactCount;
    uint overflow;
    uint reserved0;
};

struct GpuColliderBroadPhaseData
{
    uint ownerBody;
    uint shapeType;
    uint environmentIndex;
    uint collisionLayer;
    uint collisionMask;
    uint enabledFlag;
    uint reserved1;
    uint reserved2;
};

bool NodeAabbOverlapsQuery(GpuBvhNode node, float3 queryMin, float3 queryMax)
{
    return AabbOverlaps(queryMin, queryMax, float3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                        float3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_BROAD_PHASE_TYPES_HLSLI
