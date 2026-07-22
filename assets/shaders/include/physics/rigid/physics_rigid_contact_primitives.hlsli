#ifndef CRESSIM_NEO_PHYSICS_RIGID_CONTACT_PRIMITIVES_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_CONTACT_PRIMITIVES_HLSLI

#include "physics_shape_common.hlsli"
#include "physics_solver_config.hlsli"

static const uint kObbAxisFaceA = 0u;
static const uint kObbAxisFaceB = 1u;
static const uint kObbAxisEdge = 2u;

float ComputePenetrationFromPoints(float3 pointA, float3 pointB, float3 normalAtoB)
{
    const float sep = dot(pointB - pointA, normalAtoB);
    return max(0.0, -sep);
}

bool GenerateSphereSphereContactPoints(float3 centerA, float radiusA,
                                       float3 centerB, float radiusB,
                                       out float3 normalAtoB,
                                       out float3 pointA,
                                       out float3 pointB,
                                       out float penetration)
{
    normalAtoB = 0.0;
    pointA = 0.0;
    pointB = 0.0;
    penetration = 0.0;

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
    pointA = centerA + normalAtoB * radiusA;
    pointB = centerB - normalAtoB * radiusB;
    penetration = ComputePenetrationFromPoints(pointA, pointB, normalAtoB);
    return (penetration > 0.0);
}

bool GenerateSphereBoxContactPoints(float3 sphereCenter, float sphereRadius,
                                    float3 boxCenter, float4 boxOrientation, float3 halfExtents,
                                    out float3 normalAtoB,
                                    out float3 pointA,
                                    out float3 pointB,
                                    out float penetration)
{
    normalAtoB = 0.0;
    pointA = 0.0;
    pointB = 0.0;
    penetration = 0.0;

    const float3 localSphere = QuaternionInverseRotate(boxOrientation, sphereCenter - boxCenter);
    const float3 clamped = clamp(localSphere, -halfExtents, halfExtents);
    const bool inside = all(localSphere >= -halfExtents) && all(localSphere <= halfExtents);

    if (!inside)
    {
        const float3 boxPoint = boxCenter + QuaternionRotate(boxOrientation, clamped);
        const float3 delta = boxPoint - sphereCenter;
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

    const float3 localFaceNormal = ClosestFaceNormalLocal(localSphere, halfExtents);
    const float3 faceNormalWorld = QuaternionRotate(boxOrientation, localFaceNormal);
    const float faceDepth =
        min(halfExtents.x - abs(localSphere.x),
            min(halfExtents.y - abs(localSphere.y), halfExtents.z - abs(localSphere.z)));
    normalAtoB = -faceNormalWorld;
    const float3 localBoxSurface = localSphere + localFaceNormal * faceDepth;
    pointB = boxCenter + QuaternionRotate(boxOrientation, localBoxSurface);
    pointA = sphereCenter + normalAtoB * sphereRadius;
    penetration = faceDepth + sphereRadius;
    return (penetration > 0.0);
}

bool GenerateSphereCapsuleContactPoints(float3 sphereCenter, float sphereRadius,
                                        float3 capsuleA, float3 capsuleB, float capsuleRadius,
                                        out float3 normalAtoB,
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

void ClosestPointsSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1,
                                 out float3 outA, out float3 outB);

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
    normalAtoB = 0.0;
    pointA = 0.0;
    pointB = 0.0;
    penetration = 0.0;

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

    penetration = minPen;
    return (penetration > 0.0);
}

void ClosestPointsSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1, out float3 outA, out float3 outB)
{
    outA = a0;
    outB = b0;

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

bool SegmentIntersectsAabbLocal(float3 a, float3 b, float3 halfExtents, out float hitT, out float3 hitPoint)
{
    hitT = 0.0;
    hitPoint = a;

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
    outSegPoint = segmentA;
    outBoxPoint = ClosestPointOnBox(segmentA, boxCenter, boxOrientation, halfExtents);
    intersects = false;

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
    normalAtoB = 0.0;
    pointA = 0.0;
    pointB = 0.0;
    penetration = 0.0;

    float3 cA, cB;
    ClosestPointsSegmentSegment(capsuleA0, capsuleA1, capsuleB0, capsuleB1, cA, cB);
    return GenerateSphereSphereContactPoints(cA, radiusA, cB, radiusB, normalAtoB, pointA, pointB, penetration);
}

bool GenerateBoxCapsuleContactPoints(float3 boxCenter, float4 boxOrientation, float3 halfExtents,
                                     float3 capsuleA, float3 capsuleB, float capsuleRadius,
                                     out float3 normalAtoB,
                                     out float3 pointA,
                                     out float3 pointB,
                                     out float penetration)
{
    normalAtoB = 0.0;
    pointA = 0.0;
    pointB = 0.0;
    penetration = 0.0;

    float3 segPoint, boxPoint;
    bool intersects = false;
    ClosestPointsSegmentBox(capsuleA, capsuleB, boxCenter, boxOrientation, halfExtents,
                            segPoint, boxPoint, intersects);

    if (!intersects)
    {
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

    const float3 localSeg = QuaternionInverseRotate(boxOrientation, segPoint - boxCenter);
    const float3 localFaceNormal = ClosestFaceNormalLocal(localSeg, halfExtents);
    normalAtoB = QuaternionRotate(boxOrientation, localFaceNormal);

    const float faceDepth =
        min(halfExtents.x - abs(localSeg.x),
            min(halfExtents.y - abs(localSeg.y), halfExtents.z - abs(localSeg.z)));

    const float3 localBoxSurface = localSeg + localFaceNormal * faceDepth;
    pointA = boxCenter + QuaternionRotate(boxOrientation, localBoxSurface);
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
    normalAtoB = 0.0;
    pointAWorld = 0.0;
    pointBWorld = 0.0;
    penetration = 0.0;

    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderSphere)
    {
        return GenerateSphereSphereContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                                 positionB, SphereRadius(colliderParamsB, scaleB),
                                                 normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderBox)
    {
        return GenerateSphereBoxContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                              positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                              normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderSphere)
    {
        float3 nS2B, pS, pB;
        const bool hit = GenerateSphereBoxContactPoints(positionB, SphereRadius(colliderParamsB, scaleB),
                                                        positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                                        nS2B, pS, pB, penetration);
        normalAtoB = -nS2B;
        pointAWorld = pB;
        pointBWorld = pS;
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderCapsule)
    {
        float3 b0, b1;
        float rB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, b0, b1, rB);
        return GenerateSphereCapsuleContactPoints(positionA, SphereRadius(colliderParamsA, scaleA),
                                                  b0, b1, rB,
                                                  normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderSphere)
    {
        float3 a0, a1;
        float rA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, a0, a1, rA);

        float3 nS2C, pS, pC;
        const bool hit = GenerateSphereCapsuleContactPoints(positionB, SphereRadius(colliderParamsB, scaleB),
                                                            a0, a1, rA,
                                                            nS2C, pS, pC, penetration);
        normalAtoB = -nS2C;
        pointAWorld = pC;
        pointBWorld = pS;
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderBox)
    {
        return GenerateBoxBoxContactPoints(positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                           positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                           normalAtoB, pointAWorld, pointBWorld, penetration);
    }

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

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderCapsule)
    {
        float3 b0, b1;
        float rB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, b0, b1, rB);
        return GenerateBoxCapsuleContactPoints(positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                               b0, b1, rB, normalAtoB, pointAWorld, pointBWorld, penetration);
    }

    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderBox)
    {
        float3 a0, a1;
        float rA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, a0, a1, rA);

        float3 nB2C, pBox, pCap;
        const bool hit = GenerateBoxCapsuleContactPoints(positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                                         a0, a1, rA, nB2C, pBox, pCap, penetration);
        normalAtoB = -nB2C;
        pointAWorld = pCap;
        pointBWorld = pBox;
        penetration = ComputePenetrationFromPoints(pointAWorld, pointBWorld, normalAtoB);
        return hit && (penetration > 0.0);
    }

    normalAtoB = 0.0;
    pointAWorld = 0.0;
    pointBWorld = 0.0;
    penetration = 0.0;
    return false;
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_CONTACT_PRIMITIVES_HLSLI
