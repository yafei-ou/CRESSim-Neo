#ifndef CRESSIM_NEO_PHYSICS_RIGID_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_COMMON_HLSLI

#include "include/structured_buffer_compat.hlsli"

static const uint kColliderSphere = 0u;
static const uint kColliderBox = 1u;
static const uint kColliderCapsule = 2u;
static const uint kRigidBodyTypeStatic = 0u;
static const uint kRigidBodyTypeKinematic = 1u;
static const uint kRigidBodyTypeDynamic = 2u;
static const uint kKinematicTargetEnabled = 1u << 0u;
static const uint kInvalidIndex = 0xffffffffu;
static const uint kRigidPairTypeCount = 6u;

static const uint kRigidContactsPerPair = 4u;
static const uint kParticleBroadPhaseEntryTypeSoft = 0u;
static const uint kParticleBroadPhaseEntryTypeRigidSurface = 1u;
static const uint kSoftCandidatePairTypeSoftSoft = 0u;
static const uint kSoftCandidatePairTypeSoftRigid = 1u;
static const uint kSoftPhaseGroupMask = 0x7fffffffu;
static const uint kSoftPhaseSelfCollideFlag = 0x80000000u;
static const float kBroadPhaseMargin = 0.05f;

static const float3 kGravity = float3(0.0, -9.81, 0.0);

// Numerical eps for normalization / divide checks
static const float kEpsilon = 1.0e-6;

// Contact tolerances
static const float kContactSlop = 1.0e-3; // meters-ish (or your unit)

// Scale-aware manifold merging (world units). Tune to your world scale.
// Using slop as a baseline is typical.
static const float kManifoldMergeDistance = 4.0 * kContactSlop;

struct GpuRigidContact
{
    uint bodyA;
    uint bodyB;
    uint active;
    uint reserved;

    // xyz = normal (A -> B), w = penetration (>= 0 when overlapping)
    float4 normalPenetration;

    // xyz = point on surface of A in A-local space
    float4 localPointA;

    // xyz = point on surface of B in B-local space
    float4 localPointB;

    // x = friction, y = restitution
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

struct GpuParticleBroadPhaseEntry
{
    uint cellKey;
    int cellX;
    int cellY;
    int cellZ;
    uint particleIndex;
    uint particleType;
    uint ownerIndex;
    uint reserved0;
};

struct GpuSoftCandidatePair
{
    uint pairType;
    uint indexA;
    uint indexB;
    uint auxIndex;
};

struct GpuParticleCellRange
{
    uint cellKey;
    uint startIndex;
    uint endIndex;
    uint reserved0;
};

struct GpuSoftRigidContact
{
    uint softParticleIndex;
    uint rigidBodyIndex;
    uint colliderIndex;
    uint active;
    float4 normalPenetration;
    float4 rigidLocalPoint;
};

struct GpuSoftContact
{
    uint particleA;
    uint particleB;
    uint active;
    uint reserved0;
    float4 normalPenetration;
};

uint SoftParticlePhaseGroup(uint phase)
{
    return phase & kSoftPhaseGroupMask;
}

bool SoftParticlePhaseSelfCollideEnabled(uint phase)
{
    return (phase & kSoftPhaseSelfCollideFlag) != 0u;
}

struct GpuSoftEdge
{
    uint particleA;
    uint particleB;
    float restLength;
    float compliance;
};

struct GpuSoftTet
{
    uint4 particleIndices;
    float restVolume;
    float compliance;
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

bool ShouldBroadPhaseCollide(uint environmentA, uint environmentB, uint layerA, uint maskA,
                             uint layerB, uint maskB)
{
    if (environmentA != environmentB)
    {
        return false;
    }

    return (maskA & layerB) != 0u && (maskB & layerA) != 0u;
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

float3 SafeNormalize(float3 value, float3 fallback)
{
    const float lengthSq = dot(value, value);
    if (lengthSq <= kEpsilon)
        return fallback;
    return value * rsqrt(lengthSq);
}

float4 QuaternionNormalize(float4 q)
{
    const float lengthSq = dot(q, q);
    if (lengthSq <= kEpsilon)
        return float4(0.0, 0.0, 0.0, 1.0);
    return q * rsqrt(lengthSq);
}

float4 QuaternionConjugate(float4 q) { return float4(-q.xyz, q.w); }

float4 QuaternionMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 QuaternionRotate(float4 q, float3 v)
{
    const float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 ComposeColliderWorldPosition(float3 bodyPosition, float4 bodyOrientation,
                                    float3 colliderLocalPosition)
{
    return bodyPosition + QuaternionRotate(bodyOrientation, colliderLocalPosition);
}

float4 ComposeColliderWorldOrientation(float4 bodyOrientation, float4 colliderLocalOrientation)
{
    return QuaternionNormalize(QuaternionMul(bodyOrientation, colliderLocalOrientation));
}

float3 QuaternionInverseRotate(float4 q, float3 v)
{
    return QuaternionRotate(QuaternionConjugate(q), v);
}

float4 QuaternionFromRotationVector(float3 rotationVector)
{
    const float angleSq = dot(rotationVector, rotationVector);
    if (angleSq <= kEpsilon)
    {
        return QuaternionNormalize(float4(rotationVector * 0.5, 1.0));
    }

    const float angle = sqrt(angleSq);
    const float halfAngle = 0.5 * angle;
    const float scale = sin(halfAngle) / angle;
    return QuaternionNormalize(float4(rotationVector * scale, cos(halfAngle)));
}

float4 IntegrateOrientation(float4 orientation, float3 angularVelocity, float dt)
{
    return QuaternionNormalize(QuaternionMul(QuaternionFromRotationVector(angularVelocity * dt),
                                             orientation));
}

float3 AngularVelocityFromQuaternionDelta(float4 previous, float4 current, float dt)
{
    if (dt <= kEpsilon)
        return float3(0.0, 0.0, 0.0);

    float4 delta =
        QuaternionMul(QuaternionNormalize(current),
                      QuaternionConjugate(QuaternionNormalize(previous)));
    if (delta.w < 0.0)
        delta = -delta;

    const float imagLenSq = dot(delta.xyz, delta.xyz);
    if (imagLenSq <= kEpsilon)
        return delta.xyz * (2.0 / dt);

    const float imagLen = sqrt(imagLenSq);
    const float angle = 2.0 * atan2(imagLen, delta.w);
    return delta.xyz * (angle / (imagLen * dt));
}

float3 Abs3(float3 v) { return abs(v); }

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

// --- Canonical penetration convention ---
// normal is A->B. penetration = positive overlap along normal.
float ComputePenetrationFromPoints(float3 pointA, float3 pointB, float3 normalAtoB)
{
    const float sep = dot(pointB - pointA, normalAtoB); // positive when separated along normal
    return max(0.0, -sep);
}

// ------------------------- Contacts with points on each shape -------------------------

bool GenerateSphereSphereContactPoints(float3 centerA, float radiusA,
                                       float3 centerB, float radiusB,
                                       out float3 normalAtoB,
                                       out float3 pointA,
                                       out float3 pointB,
                                       out float penetration)
{
    const float3 delta = centerB - centerA;
    const float distSq = dot(delta, delta);
    const float r = radiusA + radiusB;

    if (distSq > r * r)
    {
        normalAtoB = 0.0;
        pointA = 0.0;
        pointB = 0.0;
        penetration = 0.0;
        return false;
    }

    const float dist = sqrt(max(distSq, 0.0));
    normalAtoB = (dist > kEpsilon) ? (delta / dist) : float3(0.0, 1.0, 0.0);

    // Points on surfaces
    pointA = centerA + normalAtoB * radiusA;
    pointB = centerB - normalAtoB * radiusB;

    penetration = ComputePenetrationFromPoints(pointA, pointB, normalAtoB);
    return (penetration > 0.0);
}

bool GenerateSphereBoxContactPoints(float3 sphereCenter, float sphereRadius,
                                    float3 boxCenter, float4 boxOrientation, float3 halfExtents,
                                    out float3 normalAtoB, // sphere -> box
                                    out float3 pointA,
                                    out float3 pointB,
                                    out float penetration)
{
    const float3 localSphere = QuaternionInverseRotate(boxOrientation, sphereCenter - boxCenter);
    const float3 clamped = clamp(localSphere, -halfExtents, halfExtents);
    const bool inside = all(localSphere >= -halfExtents) && all(localSphere <= halfExtents);

    if (!inside)
    {
        // Closest point on box to sphere center
        const float3 boxPoint = boxCenter + QuaternionRotate(boxOrientation, clamped);

        const float3 delta = boxPoint - sphereCenter; // sphere -> box
        const float distSq = dot(delta, delta);
        if (distSq > sphereRadius * sphereRadius)
        {
            normalAtoB = 0.0;
            pointA = 0.0;
            pointB = 0.0;
            penetration = 0.0;
            return false;
        }

        const float dist = sqrt(max(distSq, 0.0));
        normalAtoB = (dist > kEpsilon) ? (delta / dist) : float3(0.0, 1.0, 0.0);

        pointA = sphereCenter + normalAtoB * sphereRadius;
        pointB = boxPoint;

        penetration = ComputePenetrationFromPoints(pointA, pointB, normalAtoB);
        return (penetration > 0.0);
    }

    // Sphere center is inside box: use nearest face depth and choose normal so
    // the solver pushes the sphere toward that face (out of the box).
    const float3 localFaceNormal = ClosestFaceNormalLocal(localSphere, halfExtents);
    const float3 faceNormalWorld = QuaternionRotate(boxOrientation, localFaceNormal);

    const float faceDepth =
        min(halfExtents.x - abs(localSphere.x),
            min(halfExtents.y - abs(localSphere.y), halfExtents.z - abs(localSphere.z)));

    // Keep solver convention: body A moves by -normalAtoB. For containment we
    // need A to move toward the nearest face, so flip this branch's normal.
    normalAtoB = -faceNormalWorld;

    // Box point on nearest face.
    const float3 localBoxSurface = localSphere + localFaceNormal * faceDepth;
    pointB = boxCenter + QuaternionRotate(boxOrientation, localBoxSurface);

    // Sphere point consistent with normal orientation.
    pointA = sphereCenter + normalAtoB * sphereRadius;

    penetration = faceDepth + sphereRadius;
    return (penetration > 0.0);
}

bool GenerateSphereCapsuleContactPoints(float3 sphereCenter, float sphereRadius,
                                        float3 capsuleA, float3 capsuleB, float capsuleRadius,
                                        out float3 normalAtoB, // sphere -> capsule
                                        out float3 pointA,
                                        out float3 pointB,
                                        out float penetration)
{
    const float3 seg = capsuleB - capsuleA;
    const float segLenSq = dot(seg, seg);

    float t = 0.0;
    if (segLenSq > kEpsilon)
        t = saturate(dot(sphereCenter - capsuleA, seg) / segLenSq);

    const float3 capsuleCenter = capsuleA + seg * t;
    return GenerateSphereSphereContactPoints(sphereCenter, sphereRadius,
                                             capsuleCenter, capsuleRadius,
                                             normalAtoB, pointA, pointB, penetration);
}

float BoxProjectionRadius(float3 axis, float3 axes[3], float3 halfExtents)
{
    return halfExtents.x * abs(dot(axis, axes[0])) +
           halfExtents.y * abs(dot(axis, axes[1])) +
           halfExtents.z * abs(dot(axis, axes[2]));
}

static const uint kObbAxisFaceA = 0u;
static const uint kObbAxisFaceB = 1u;
static const uint kObbAxisEdge = 2u;

void ClosestPointsSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1, out float3 outA, out float3 outB);

bool TestObbAxis(float3 axis, float3 centerDelta,
                 float3 axesA[3], float3 extentsA,
                 float3 axesB[3], float3 extentsB,
                 uint candidateAxisType, uint candidateAxisA, uint candidateAxisB,
                 inout float minPenetration, inout float3 bestAxis,
                 inout uint bestAxisType, inout uint bestAxisA, inout uint bestAxisB)
{
    const float axisLenSq = dot(axis, axis);
    if (axisLenSq <= kEpsilon)
        return true;

    const float3 n = axis * rsqrt(axisLenSq);
    const float distance = abs(dot(centerDelta, n));

    const float radiusA = BoxProjectionRadius(n, axesA, extentsA);
    const float radiusB = BoxProjectionRadius(n, axesB, extentsB);

    const float overlap = radiusA + radiusB - distance;
    if (overlap < 0.0)
        return false;

    if (overlap < minPenetration)
    {
        minPenetration = overlap;
        // Ensure axis points A->B
        bestAxis = (dot(centerDelta, n) >= 0.0) ? n : -n;
        bestAxisType = candidateAxisType;
        bestAxisA = candidateAxisA;
        bestAxisB = candidateAxisB;
    }
    return true;
}

void BuildBoxSupportEdge(uint axisIndex, float3 center, float3 axes[3], float3 halfExtents,
                         float3 supportDirection, out float3 edgeStart, out float3 edgeEnd)
{
    const uint orthoA = (axisIndex + 1u) % 3u;
    const uint orthoB = (axisIndex + 2u) % 3u;

    const float signA = dot(axes[orthoA], supportDirection) >= 0.0 ? 1.0 : -1.0;
    const float signB = dot(axes[orthoB], supportDirection) >= 0.0 ? 1.0 : -1.0;

    const float3 edgeCenter =
        center + axes[orthoA] * (halfExtents[orthoA] * signA) +
        axes[orthoB] * (halfExtents[orthoB] * signB);
    const float3 edgeHalfVector = axes[axisIndex] * halfExtents[axisIndex];

    edgeStart = edgeCenter - edgeHalfVector;
    edgeEnd = edgeCenter + edgeHalfVector;
}

bool GenerateBoxBoxContactPoints(float3 centerA, float4 orientationA, float3 halfExtentsA,
                                 float3 centerB, float4 orientationB, float3 halfExtentsB,
                                 out float3 normalAtoB,
                                 out float3 pointA,
                                 out float3 pointB,
                                 out float penetration)
{
    float3 axesA[3] = {BoxAxisX(orientationA), BoxAxisY(orientationA), BoxAxisZ(orientationA)};
    float3 axesB[3] = {BoxAxisX(orientationB), BoxAxisY(orientationB), BoxAxisZ(orientationB)};
    const float3 centerDelta = centerB - centerA;

    float minPen = 3.402823466e+38;
    float3 bestAxis = float3(0.0, 1.0, 0.0);
    uint bestAxisType = kObbAxisFaceA;
    uint bestAxisA = 0u;
    uint bestAxisB = 0u;

    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        if (!TestObbAxis(axesA[i], centerDelta, axesA, halfExtentsA, axesB, halfExtentsB,
                         kObbAxisFaceA, i, 0u,
                         minPen, bestAxis, bestAxisType, bestAxisA, bestAxisB))
        {
            normalAtoB = 0.0;
            pointA = 0.0;
            pointB = 0.0;
            penetration = 0.0;
            return false;
        }
    }

    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        if (!TestObbAxis(axesB[i], centerDelta, axesA, halfExtentsA, axesB, halfExtentsB,
                         kObbAxisFaceB, i, 0u,
                         minPen, bestAxis, bestAxisType, bestAxisA, bestAxisB))
        {
            normalAtoB = 0.0;
            pointA = 0.0;
            pointB = 0.0;
            penetration = 0.0;
            return false;
        }
    }

    [unroll] for (uint a = 0u; a < 3u; ++a)
    {
        [unroll] for (uint b = 0u; b < 3u; ++b)
        {
            if (!TestObbAxis(cross(axesA[a], axesB[b]), centerDelta, axesA, halfExtentsA, axesB, halfExtentsB,
                             kObbAxisEdge, a, b,
                             minPen, bestAxis, bestAxisType, bestAxisA, bestAxisB))
            {
                normalAtoB = 0.0;
                pointA = 0.0;
                pointB = 0.0;
                penetration = 0.0;
                return false;
            }
        }
    }

    normalAtoB = bestAxis;

    if (bestAxisType == kObbAxisEdge)
    {
        float3 edgeA0, edgeA1, edgeB0, edgeB1;
        BuildBoxSupportEdge(bestAxisA, centerA, axesA, halfExtentsA, normalAtoB, edgeA0, edgeA1);
        BuildBoxSupportEdge(bestAxisB, centerB, axesB, halfExtentsB, -normalAtoB, edgeB0, edgeB1);
        ClosestPointsSegmentSegment(edgeA0, edgeA1, edgeB0, edgeB1, pointA, pointB);
    }
    else if (bestAxisType == kObbAxisFaceB)
    {
        pointB = BoxSupportPoint(centerB, orientationB, halfExtentsB, -normalAtoB);
        pointA = ClosestPointOnBox(pointB, centerA, orientationA, halfExtentsA);
    }
    else
    {
        pointA = BoxSupportPoint(centerA, orientationA, halfExtentsA, normalAtoB);
        pointB = ClosestPointOnBox(pointA, centerB, orientationB, halfExtentsB);
    }

    penetration = minPen; // max(minPen, ComputePenetrationFromPoints(pointA, pointB, normalAtoB));
    return (penetration > 0.0);
}

void ClosestPointsSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1, out float3 outA, out float3 outB)
{
    const float3 d1 = a1 - a0;
    const float3 d2 = b1 - b0;
    const float3 r = a0 - b0;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);

    float s = 0.0;
    float t = 0.0;

    if (a <= kEpsilon && e <= kEpsilon)
    {
        outA = a0;
        outB = b0;
        return;
    }
    if (a <= kEpsilon)
    {
        t = saturate(f / e);
    }
    else
    {
        const float c = dot(d1, r);
        if (e <= kEpsilon)
        {
            s = saturate(-c / a);
        }
        else
        {
            const float b = dot(d1, d2);
            const float denom = a * e - b * b;
            if (denom != 0.0)
                s = saturate((b * f - c * e) / denom);

            const float tNom = b * s + f;
            if (tNom < 0.0)
            {
                t = 0.0;
                s = saturate(-c / a);
            }
            else if (tNom > e)
            {
                t = 1.0;
                s = saturate((b - c) / a);
            }
            else
            {
                t = tNom / e;
            }
        }
    }

    outA = a0 + d1 * s;
    outB = b0 + d2 * t;
}

// Segment-box helpers unchanged (your existing code) ...
// (Keep your SegmentIntersectsAabbLocal / ClosestPointsSegmentBox helpers as-is.)

bool SegmentIntersectsAabbLocal(float3 a, float3 b, float3 halfExtents, out float hitT, out float3 hitPoint)
{
    float tMin = 0.0, tMax = 1.0;
    const float3 d = b - a;

    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float origin = axis == 0u ? a.x : (axis == 1u ? a.y : a.z);
        const float delta = axis == 0u ? d.x : (axis == 1u ? d.y : d.z);
        const float minV = axis == 0u ? -halfExtents.x : (axis == 1u ? -halfExtents.y : -halfExtents.z);
        const float maxV = axis == 0u ? halfExtents.x : (axis == 1u ? halfExtents.y : halfExtents.z);

        if (abs(delta) <= kEpsilon)
        {
            if (origin < minV || origin > maxV)
            {
                hitT = 0.0;
                hitPoint = a;
                return false;
            }
            continue;
        }

        const float invD = 1.0 / delta;
        float t0 = (minV - origin) * invD;
        float t1 = (maxV - origin) * invD;
        if (t0 > t1)
        {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        tMin = max(tMin, t0);
        tMax = min(tMax, t1);
        if (tMin > tMax)
        {
            hitT = 0.0;
            hitPoint = a;
            return false;
        }
    }

    hitT = tMin;
    hitPoint = a + d * tMin;
    return true;
}

void ConsiderSegmentBoxCandidate(float t, float3 localA, float3 localB, float3 halfExtents,
                                 inout float bestDistanceSq, inout float3 bestSegPoint, inout float3 bestBoxPoint)
{
    if (t < 0.0 || t > 1.0)
        return;

    const float3 segP = lerp(localA, localB, t);
    const float3 boxP = clamp(segP, -halfExtents, halfExtents);
    const float d2 = dot(segP - boxP, segP - boxP);
    if (d2 < bestDistanceSq)
    {
        bestDistanceSq = d2;
        bestSegPoint = segP;
        bestBoxPoint = boxP;
    }
}

void ClosestPointsSegmentBox(float3 segmentA, float3 segmentB, float3 boxCenter,
                             float4 boxOrientation, float3 halfExtents,
                             out float3 outSegPoint, out float3 outBoxPoint, out bool intersects)
{
    const float3 localA = QuaternionInverseRotate(boxOrientation, segmentA - boxCenter);
    const float3 localB = QuaternionInverseRotate(boxOrientation, segmentB - boxCenter);

    float hitT = 0.0;
    float3 hitPoint = 0.0;
    intersects = SegmentIntersectsAabbLocal(localA, localB, halfExtents, hitT, hitPoint);
    if (intersects)
    {
        outSegPoint = boxCenter + QuaternionRotate(boxOrientation, hitPoint);
        outBoxPoint = outSegPoint;
        return;
    }

    float bestDistanceSq = 3.402823466e+38;
    float3 bestSegPoint = localA;
    float3 bestBoxPoint = clamp(localA, -halfExtents, halfExtents);
    const float3 d = localB - localA;

    ConsiderSegmentBoxCandidate(0.0, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    ConsiderSegmentBoxCandidate(1.0, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);

    if (abs(d.x) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.x - localA.x) / d.x, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.x - localA.x) / d.x, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }
    if (abs(d.y) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.y - localA.y) / d.y, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.y - localA.y) / d.y, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }
    if (abs(d.z) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.z - localA.z) / d.z, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.z - localA.z) / d.z, localA, localB, halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }

    outSegPoint = boxCenter + QuaternionRotate(boxOrientation, bestSegPoint);
    outBoxPoint = boxCenter + QuaternionRotate(boxOrientation, bestBoxPoint);
}

bool GenerateCapsuleCapsuleContactPoints(float3 capsuleA0, float3 capsuleA1, float radiusA,
                                         float3 capsuleB0, float3 capsuleB1, float radiusB,
                                         out float3 normalAtoB,
                                         out float3 pointA,
                                         out float3 pointB,
                                         out float penetration)
{
    float3 cA, cB;
    ClosestPointsSegmentSegment(capsuleA0, capsuleA1, capsuleB0, capsuleB1, cA, cB);
    return GenerateSphereSphereContactPoints(cA, radiusA, cB, radiusB, normalAtoB, pointA, pointB, penetration);
}

bool GenerateBoxCapsuleContactPoints(float3 boxCenter, float4 boxOrientation, float3 halfExtents,
                                     float3 capsuleA, float3 capsuleB, float capsuleRadius,
                                     out float3 normalAtoB, // box -> capsule
                                     out float3 pointA,
                                     out float3 pointB,
                                     out float penetration)
{
    float3 segPoint, boxPoint;
    bool intersects = false;
    ClosestPointsSegmentBox(capsuleA, capsuleB, boxCenter, boxOrientation, halfExtents,
                            segPoint, boxPoint, intersects);

    if (!intersects)
    {
        // normal A->B = box -> capsule
        const float3 delta = segPoint - boxPoint;
        const float distSq = dot(delta, delta);
        if (distSq > capsuleRadius * capsuleRadius)
        {
            normalAtoB = 0.0;
            pointA = 0.0;
            pointB = 0.0;
            penetration = 0.0;
            return false;
        }

        const float dist = sqrt(max(distSq, 0.0));
        normalAtoB = (dist > kEpsilon) ? (delta / dist) : float3(0.0, 1.0, 0.0);

        pointA = boxPoint;
        pointB = segPoint - normalAtoB * capsuleRadius;

        penetration = ComputePenetrationFromPoints(pointA, pointB, normalAtoB);
        return (penetration > 0.0);
    }

    // Segment intersects box: treat like capsule center is "inside" the box, choose closest face.
    const float3 localSeg = QuaternionInverseRotate(boxOrientation, segPoint - boxCenter);
    const float3 localFaceNormal = ClosestFaceNormalLocal(localSeg, halfExtents);
    normalAtoB = QuaternionRotate(boxOrientation, localFaceNormal); // box -> capsule (outward)

    const float faceDepth =
        min(halfExtents.x - abs(localSeg.x),
            min(halfExtents.y - abs(localSeg.y), halfExtents.z - abs(localSeg.z)));

    const float3 localBoxSurface = localSeg + localFaceNormal * faceDepth;
    pointA = boxCenter + QuaternionRotate(boxOrientation, localBoxSurface);

    // Capsule surface toward that normal (capsule centerline point is segPoint)
    pointB = segPoint - normalAtoB * capsuleRadius;

    penetration = ComputePenetrationFromPoints(pointA, pointB, normalAtoB);
    return (penetration > 0.0);
}

bool GenerateRigidContact(uint shapeTypeA, float3 positionA, float4 orientationA,
                          float4 colliderParamsA, float4 scaleA,
                          uint shapeTypeB, float3 positionB, float4 orientationB,
                          float4 colliderParamsB, float4 scaleB,
                          out float3 normalAtoB,
                          out float3 pointAWorld,
                          out float3 pointBWorld,
                          out float penetration)
{
    // Sphere-Sphere
    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderSphere)
    {
        return GenerateSphereSphereContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                                 positionB, SphereRadius(colliderParamsB, scaleB),
                                                 normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Sphere-Box (A sphere -> B box)
    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderBox)
    {
        return GenerateSphereBoxContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                              positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                              normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Box-Sphere (A box -> B sphere): call sphere-box and flip
    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderSphere)
    {
        float3 nS2B, pS, pB;
        const bool hit = GenerateSphereBoxContactPoints(positionB, SphereRadius(colliderParamsB, scaleB),
                                                        positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                                        nS2B, pS, pB, penetration);
        // We asked (sphere=B)->(box=A). Convert to (box=A)->(sphere=B)
        normalAtoB = -nS2B;
        pointAWorld = pB; // point on box A
        pointBWorld = pS; // point on sphere B
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    // Sphere-Capsule (A sphere -> B capsule)
    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderCapsule)
    {
        float3 b0, b1;
        float rB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, b0, b1, rB);
        return GenerateSphereCapsuleContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                                  b0, b1, rB,
                                                  normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Capsule-Sphere (A capsule -> B sphere): call sphere-capsule and flip
    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderSphere)
    {
        float3 a0, a1;
        float rA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, a0, a1, rA);

        float3 nS2C, pS, pC;
        const bool hit = GenerateSphereCapsuleContactPoints(positionB, SphereRadius(colliderParamsB, scaleB),
                                                            a0, a1, rA,
                                                            nS2C, pS, pC, penetration);
        normalAtoB = -nS2C; // capsule -> sphere
        pointAWorld = pC;   // point on capsule
        pointBWorld = pS;   // point on sphere
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    // Box-Box
    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderBox)
    {
        return GenerateBoxBoxContactPoints(positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                           positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                           normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Capsule-Capsule
    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderCapsule)
    {
        float3 a0, a1;
        float rA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, a0, a1, rA);

        float3 b0, b1;
        float rB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, b0, b1, rB);

        return GenerateCapsuleCapsuleContactPoints(a0, a1, rA, b0, b1, rB,
                                                   normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Box-Capsule (A box -> B capsule)
    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderCapsule)
    {
        float3 b0, b1;
        float rB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, b0, b1, rB);
        return GenerateBoxCapsuleContactPoints(positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                               b0, b1, rB, normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    // Capsule-Box (A capsule -> B box): call box-capsule and flip
    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderBox)
    {
        float3 a0, a1;
        float rA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, a0, a1, rA);

        float3 nB2C, pBox, pCap;
        const bool hit = GenerateBoxCapsuleContactPoints(positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                                         a0, a1, rA, nB2C, pBox, pCap, penetration);
        normalAtoB = -nB2C; // capsule -> box
        pointAWorld = pCap; // point on capsule
        pointBWorld = pBox; // point on box
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    normalAtoB = 0.0;
    pointAWorld = 0.0;
    pointBWorld = 0.0;
    penetration = 0.0;
    return false;
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

float3 MultiplyWorldInverseInertia(float3 inverseInertiaLocal, float4 orientation, float3 value)
{
    const float3 ax = BoxAxisX(orientation);
    const float3 ay = BoxAxisY(orientation);
    const float3 az = BoxAxisZ(orientation);
    return ax * (inverseInertiaLocal.x * dot(ax, value)) +
           ay * (inverseInertiaLocal.y * dot(ay, value)) +
           az * (inverseInertiaLocal.z * dot(az, value));
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

#endif
