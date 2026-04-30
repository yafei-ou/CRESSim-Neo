#ifndef CRESSIM_NEO_PHYSICS_COLLISION_SHAPE_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_COLLISION_SHAPE_COMMON_HLSLI

#include "../core/physics_math.hlsli"

static const uint kColliderSphere = 0u;
static const uint kColliderBox = 1u;
static const uint kColliderCapsule = 2u;
static const uint kRigidPairTypeCount = 6u;
static const float kBroadPhaseMargin = 0.05f;

bool ShouldBroadPhaseCollide(uint environmentA, uint environmentB, uint layerA, uint maskA,
                             uint layerB, uint maskB)
{
    if (environmentA != environmentB)
    {
        return false;
    }

    return (maskA & layerB) != 0u && (maskB & layerA) != 0u;
}

float SphereRadius(float4 colliderParams, float4 scale)
{
    const float3 absScale = Abs3(scale.xyz);
    return colliderParams.x * max(absScale.x, max(absScale.y, absScale.z));
}

float3 BoxHalfExtents(float4 colliderParams, float4 scale)
{
    return colliderParams.xyz * Abs3(scale.xyz);
}

void CapsuleSegment(float3 position, float4 orientation, float4 colliderParams, float4 scale,
                    out float3 segmentA, out float3 segmentB, out float radius)
{
    const float3 absScale = Abs3(scale.xyz);
    radius = colliderParams.x * max(absScale.x, absScale.z);
    const float halfHeight = colliderParams.y * absScale.y;
    const float3 axis = QuaternionRotate(orientation, float3(0.0, 1.0, 0.0));
    segmentA = position - axis * halfHeight;
    segmentB = position + axis * halfHeight;
}

float3 BoxAxisX(float4 q) { return QuaternionRotate(q, float3(1.0, 0.0, 0.0)); }
float3 BoxAxisY(float4 q) { return QuaternionRotate(q, float3(0.0, 1.0, 0.0)); }
float3 BoxAxisZ(float4 q) { return QuaternionRotate(q, float3(0.0, 0.0, 1.0)); }

void ComputeBodyAabb(uint shapeType, float3 position, float4 orientation, float4 colliderParams,
                     float4 scale, out float3 aabbMin, out float3 aabbMax)
{
    float3 extents = 0.0;
    if (shapeType == kColliderSphere)
    {
        extents = SphereRadius(colliderParams, scale).xxx;
    }
    else if (shapeType == kColliderBox)
    {
        const float3 halfExtents = BoxHalfExtents(colliderParams, scale);
        extents = abs(BoxAxisX(orientation)) * halfExtents.x +
                  abs(BoxAxisY(orientation)) * halfExtents.y +
                  abs(BoxAxisZ(orientation)) * halfExtents.z;
    }
    else
    {
        float3 capsuleA, capsuleB;
        float capsuleRadius = 0.0;
        CapsuleSegment(position, orientation, colliderParams, scale, capsuleA, capsuleB, capsuleRadius);
        const float3 capsuleAxis = abs(QuaternionRotate(orientation, float3(0.0, 1.0, 0.0)));
        extents = capsuleAxis * (0.5 * distance(capsuleA, capsuleB)) + capsuleRadius.xxx;
    }

    aabbMin = position - extents;
    aabbMax = position + extents;
}

bool AabbOverlaps(float3 minA, float3 maxA, float3 minB, float3 maxB)
{
    return minA.x <= maxB.x && maxA.x >= minB.x &&
           minA.y <= maxB.y && maxA.y >= minB.y &&
           minA.z <= maxB.z && maxA.z >= minB.z;
}

float3 ClosestPointOnBox(float3 queryPoint, float3 boxCenter, float4 boxOrientation, float3 halfExtents)
{
    const float3 localPoint = QuaternionInverseRotate(boxOrientation, queryPoint - boxCenter);
    const float3 clamped = clamp(localPoint, -halfExtents, halfExtents);
    return boxCenter + QuaternionRotate(boxOrientation, clamped);
}

float3 BoxSupportPoint(float3 boxCenter, float4 boxOrientation, float3 halfExtents, float3 direction)
{
    const float3 localDir = QuaternionInverseRotate(boxOrientation, direction);
    const float3 localSupport = float3(localDir.x >= 0.0 ? halfExtents.x : -halfExtents.x,
                                       localDir.y >= 0.0 ? halfExtents.y : -halfExtents.y,
                                       localDir.z >= 0.0 ? halfExtents.z : -halfExtents.z);
    return boxCenter + QuaternionRotate(boxOrientation, localSupport);
}

float3 ClosestFaceNormalLocal(float3 localPoint, float3 halfExtents)
{
    const float3 distances = halfExtents - abs(localPoint);
    if (distances.x <= distances.y && distances.x <= distances.z)
        return float3(localPoint.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    if (distances.y <= distances.z)
        return float3(0.0, localPoint.y >= 0.0 ? 1.0 : -1.0, 0.0);
    return float3(0.0, 0.0, localPoint.z >= 0.0 ? 1.0 : -1.0);
}

float3 SupportPointForShape(uint shapeType, float3 position, float4 orientation,
                            float4 colliderParams, float4 scale, float3 direction)
{
    const float3 dir = SafeNormalize(direction, float3(0.0, 1.0, 0.0));

    if (shapeType == kColliderSphere)
        return position + dir * SphereRadius(colliderParams, scale);

    if (shapeType == kColliderBox)
        return BoxSupportPoint(position, orientation, BoxHalfExtents(colliderParams, scale), dir);

    float3 a, b;
    float r = 0.0;
    CapsuleSegment(position, orientation, colliderParams, scale, a, b, r);
    const float3 seg = b - a;
    const float3 segPoint = dot(dir, seg) >= 0.0 ? b : a;
    return segPoint + dir * r;
}

#endif // CRESSIM_NEO_PHYSICS_COLLISION_SHAPE_COMMON_HLSLI
