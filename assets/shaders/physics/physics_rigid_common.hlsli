#ifndef CRESSIM_NEO_PHYSICS_RIGID_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_COMMON_HLSLI

static const uint kColliderSphere = 0u;
static const uint kColliderBox = 1u;
static const uint kColliderCapsule = 2u;
static const uint kRigidContactsPerPair = 4u;
static const float3 kGravity = float3(0.0, -9.81, 0.0);
static const float kEpsilon = 1.0e-6;
static const float kContactSlop = 1.0e-3;
static const float kContactBias = 0.8;

struct GpuRigidContact
{
    uint bodyA;
    uint bodyB;
    uint active;
    uint reserved;
    float4 normalPenetration;
    float4 worldPoint;
};

float3 SafeNormalize(float3 value, float3 fallback)
{
    const float lengthSq = dot(value, value);
    if (lengthSq <= kEpsilon)
    {
        return fallback;
    }
    return value * rsqrt(lengthSq);
}

float4 QuaternionNormalize(float4 q)
{
    const float lengthSq = dot(q, q);
    if (lengthSq <= kEpsilon)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return q * rsqrt(lengthSq);
}

float4 QuaternionConjugate(float4 q)
{
    return float4(-q.xyz, q.w);
}

float4 QuaternionMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 QuaternionRotate(float4 q, float3 value)
{
    const float3 t = 2.0 * cross(q.xyz, value);
    return value + q.w * t + cross(q.xyz, t);
}

float3 QuaternionInverseRotate(float4 q, float3 value)
{
    return QuaternionRotate(QuaternionConjugate(q), value);
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
    {
        return float3(0.0, 0.0, 0.0);
    }

    float4 delta =
        QuaternionMul(QuaternionNormalize(current), QuaternionConjugate(QuaternionNormalize(previous)));
    if (delta.w < 0.0)
    {
        delta = -delta;
    }

    const float imagLengthSq = dot(delta.xyz, delta.xyz);
    if (imagLengthSq <= kEpsilon)
    {
        return delta.xyz * (2.0 / dt);
    }

    const float imagLength = sqrt(imagLengthSq);
    const float angle = 2.0 * atan2(imagLength, delta.w);
    return delta.xyz * (angle / (imagLength * dt));
}

float3 Abs3(float3 value)
{
    return abs(value);
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

float3 BoxAxisX(float4 orientation)
{
    return QuaternionRotate(orientation, float3(1.0, 0.0, 0.0));
}

float3 BoxAxisY(float4 orientation)
{
    return QuaternionRotate(orientation, float3(0.0, 1.0, 0.0));
}

float3 BoxAxisZ(float4 orientation)
{
    return QuaternionRotate(orientation, float3(0.0, 0.0, 1.0));
}

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
        float3 capsuleA;
        float3 capsuleB;
        float capsuleRadius = 0.0;
        CapsuleSegment(position, orientation, colliderParams, scale, capsuleA, capsuleB,
                       capsuleRadius);
        const float3 capsuleExtents =
            abs(QuaternionRotate(orientation, float3(0.0, 1.0, 0.0))) *
                (0.5 * distance(capsuleA, capsuleB)) +
            capsuleRadius.xxx;
        extents = capsuleExtents;
    }

    aabbMin = position - extents;
    aabbMax = position + extents;
}

bool AabbOverlaps(float3 minA, float3 maxA, float3 minB, float3 maxB)
{
    return minA.x <= maxB.x && maxA.x >= minB.x && minA.y <= maxB.y && maxA.y >= minB.y &&
           minA.z <= maxB.z && maxA.z >= minB.z;
}

float3 ClosestPointOnBox(float3 queryPoint, float3 boxCenter, float4 boxOrientation,
                         float3 halfExtents)
{
    const float3 localPoint = QuaternionInverseRotate(boxOrientation, queryPoint - boxCenter);
    const float3 clamped = clamp(localPoint, -halfExtents, halfExtents);
    return boxCenter + QuaternionRotate(boxOrientation, clamped);
}

float3 BoxSupportPoint(float3 boxCenter, float4 boxOrientation, float3 halfExtents, float3 direction)
{
    const float3 localDirection = QuaternionInverseRotate(boxOrientation, direction);
    const float3 localSupport = float3(localDirection.x >= 0.0 ? halfExtents.x : -halfExtents.x,
                                       localDirection.y >= 0.0 ? halfExtents.y : -halfExtents.y,
                                       localDirection.z >= 0.0 ? halfExtents.z : -halfExtents.z);
    return boxCenter + QuaternionRotate(boxOrientation, localSupport);
}

float3 ClosestFaceNormalLocal(float3 localPoint, float3 halfExtents)
{
    const float3 distances = halfExtents - abs(localPoint);
    if (distances.x <= distances.y && distances.x <= distances.z)
    {
        return float3(localPoint.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    }
    if (distances.y <= distances.z)
    {
        return float3(0.0, localPoint.y >= 0.0 ? 1.0 : -1.0, 0.0);
    }
    return float3(0.0, 0.0, localPoint.z >= 0.0 ? 1.0 : -1.0);
}

void ClosestPointsSegmentSegment(float3 a0, float3 a1, float3 b0, float3 b1, out float3 outA,
                                 out float3 outB)
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
            {
                s = saturate((b * f - c * e) / denom);
            }
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

bool SegmentIntersectsAabbLocal(float3 a, float3 b, float3 halfExtents, out float hitT,
                                out float3 hitPoint)
{
    float tMin = 0.0;
    float tMax = 1.0;
    const float3 d = b - a;

    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float origin = axis == 0u ? a.x : (axis == 1u ? a.y : a.z);
        const float delta = axis == 0u ? d.x : (axis == 1u ? d.y : d.z);
        const float minValue = axis == 0u ? -halfExtents.x : (axis == 1u ? -halfExtents.y : -halfExtents.z);
        const float maxValue = axis == 0u ? halfExtents.x : (axis == 1u ? halfExtents.y : halfExtents.z);

        if (abs(delta) <= kEpsilon)
        {
            if (origin < minValue || origin > maxValue)
            {
                hitT = 0.0;
                hitPoint = a;
                return false;
            }
            continue;
        }

        const float invDelta = 1.0 / delta;
        float t0 = (minValue - origin) * invDelta;
        float t1 = (maxValue - origin) * invDelta;
        if (t0 > t1)
        {
            const float temp = t0;
            t0 = t1;
            t1 = temp;
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
                                 inout float bestDistanceSq, inout float3 bestSegPoint,
                                 inout float3 bestBoxPoint)
{
    if (t < 0.0 || t > 1.0)
    {
        return;
    }

    const float3 segmentPoint = lerp(localA, localB, t);
    const float3 boxPoint = clamp(segmentPoint, -halfExtents, halfExtents);
    const float distanceSq = dot(segmentPoint - boxPoint, segmentPoint - boxPoint);
    if (distanceSq < bestDistanceSq)
    {
        bestDistanceSq = distanceSq;
        bestSegPoint = segmentPoint;
        bestBoxPoint = boxPoint;
    }
}

void ClosestPointsSegmentBox(float3 segmentA, float3 segmentB, float3 boxCenter,
                             float4 boxOrientation, float3 halfExtents, out float3 outSegPoint,
                             out float3 outBoxPoint, out bool intersects)
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

    ConsiderSegmentBoxCandidate(0.0, localA, localB, halfExtents, bestDistanceSq, bestSegPoint,
                                bestBoxPoint);
    ConsiderSegmentBoxCandidate(1.0, localA, localB, halfExtents, bestDistanceSq, bestSegPoint,
                                bestBoxPoint);

    if (abs(d.x) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.x - localA.x) / d.x, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.x - localA.x) / d.x, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }
    if (abs(d.y) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.y - localA.y) / d.y, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.y - localA.y) / d.y, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }
    if (abs(d.z) > kEpsilon)
    {
        ConsiderSegmentBoxCandidate((-halfExtents.z - localA.z) / d.z, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
        ConsiderSegmentBoxCandidate((halfExtents.z - localA.z) / d.z, localA, localB,
                                    halfExtents, bestDistanceSq, bestSegPoint, bestBoxPoint);
    }

    outSegPoint = boxCenter + QuaternionRotate(boxOrientation, bestSegPoint);
    outBoxPoint = boxCenter + QuaternionRotate(boxOrientation, bestBoxPoint);
}

bool GenerateSphereSphereContact(float3 centerA, float radiusA, float3 centerB, float radiusB,
                                 out float3 normal, out float3 contactPoint,
                                 out float penetration)
{
    const float3 delta = centerB - centerA;
    const float distanceSq = dot(delta, delta);
    const float combinedRadius = radiusA + radiusB;
    if (distanceSq > combinedRadius * combinedRadius)
    {
        normal = 0.0;
        contactPoint = 0.0;
        penetration = 0.0;
        return false;
    }

    const float distance = sqrt(max(distanceSq, 0.0));
    normal = distance > kEpsilon ? delta / distance : float3(0.0, 1.0, 0.0);
    penetration = combinedRadius - distance;
    const float3 pointA = centerA + normal * radiusA;
    const float3 pointB = centerB - normal * radiusB;
    contactPoint = 0.5 * (pointA + pointB);
    return true;
}

bool GenerateSphereBoxContact(float3 sphereCenter, float sphereRadius, float3 boxCenter,
                              float4 boxOrientation, float3 halfExtents, out float3 normal,
                              out float3 contactPoint, out float penetration)
{
    const float3 localSphere = QuaternionInverseRotate(boxOrientation, sphereCenter - boxCenter);
    const float3 clamped = clamp(localSphere, -halfExtents, halfExtents);
    const bool inside = all(localSphere >= -halfExtents) && all(localSphere <= halfExtents);

    if (!inside)
    {
        const float3 boxPoint = boxCenter + QuaternionRotate(boxOrientation, clamped);
        const float3 delta = sphereCenter - boxPoint;
        const float distanceSq = dot(delta, delta);
        if (distanceSq > sphereRadius * sphereRadius)
        {
            normal = 0.0;
            contactPoint = 0.0;
            penetration = 0.0;
            return false;
        }

        const float distance = sqrt(max(distanceSq, 0.0));
        normal = distance > kEpsilon ? delta / distance : float3(0.0, 1.0, 0.0);
        penetration = sphereRadius - distance;
        contactPoint = 0.5 * (boxPoint + (sphereCenter - normal * sphereRadius));
        return true;
    }

    const float3 localNormal = ClosestFaceNormalLocal(localSphere, halfExtents);
    normal = QuaternionRotate(boxOrientation, localNormal);
    const float faceDepth =
        min(halfExtents.x - abs(localSphere.x),
            min(halfExtents.y - abs(localSphere.y), halfExtents.z - abs(localSphere.z)));
    const float3 boxSurface =
        boxCenter + QuaternionRotate(boxOrientation, localSphere - localNormal * faceDepth);
    penetration = sphereRadius + faceDepth;
    contactPoint = 0.5 * (boxSurface + (sphereCenter - normal * sphereRadius));
    return true;
}

bool GenerateSphereCapsuleContact(float3 sphereCenter, float sphereRadius, float3 capsuleA,
                                  float3 capsuleB, float capsuleRadius, out float3 normal,
                                  out float3 contactPoint, out float penetration)
{
    const float3 segment = capsuleB - capsuleA;
    const float segmentLengthSq = dot(segment, segment);
    float t = 0.0;
    if (segmentLengthSq > kEpsilon)
    {
        t = saturate(dot(sphereCenter - capsuleA, segment) / segmentLengthSq);
    }
    const float3 capsulePoint = capsuleA + segment * t;
    return GenerateSphereSphereContact(sphereCenter, sphereRadius, capsulePoint, capsuleRadius,
                                       normal, contactPoint, penetration);
}

float BoxProjectionRadius(float3 axis, float3 axes[3], float3 halfExtents)
{
    return halfExtents.x * abs(dot(axis, axes[0])) + halfExtents.y * abs(dot(axis, axes[1])) +
           halfExtents.z * abs(dot(axis, axes[2]));
}

bool TestObbAxis(float3 axis, float3 centerDelta, float3 axesA[3], float3 extentsA,
                 float3 axesB[3], float3 extentsB, inout float minPenetration,
                 inout float3 bestAxis)
{
    const float axisLengthSq = dot(axis, axis);
    if (axisLengthSq <= kEpsilon)
    {
        return true;
    }

    const float3 n = axis * rsqrt(axisLengthSq);
    const float distance = abs(dot(centerDelta, n));
    const float radiusA = BoxProjectionRadius(n, axesA, extentsA);
    const float radiusB = BoxProjectionRadius(n, axesB, extentsB);
    const float overlap = radiusA + radiusB - distance;
    if (overlap < 0.0)
    {
        return false;
    }

    if (overlap < minPenetration)
    {
        minPenetration = overlap;
        bestAxis = dot(centerDelta, n) >= 0.0 ? n : -n;
    }
    return true;
}

bool GenerateBoxBoxContact(float3 centerA, float4 orientationA, float3 halfExtentsA,
                           float3 centerB, float4 orientationB, float3 halfExtentsB,
                           out float3 normal, out float3 contactPoint,
                           out float penetration)
{
    float3 axesA[3] = {BoxAxisX(orientationA), BoxAxisY(orientationA), BoxAxisZ(orientationA)};
    float3 axesB[3] = {BoxAxisX(orientationB), BoxAxisY(orientationB), BoxAxisZ(orientationB)};
    const float3 centerDelta = centerB - centerA;

    float minPenetration = 3.402823466e+38;
    float3 bestAxis = float3(0.0, 1.0, 0.0);

    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (!TestObbAxis(axesA[axis], centerDelta, axesA, halfExtentsA, axesB, halfExtentsB,
                         minPenetration, bestAxis))
        {
            normal = 0.0;
            contactPoint = 0.0;
            penetration = 0.0;
            return false;
        }
    }

    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (!TestObbAxis(axesB[axis], centerDelta, axesA, halfExtentsA, axesB, halfExtentsB,
                         minPenetration, bestAxis))
        {
            normal = 0.0;
            contactPoint = 0.0;
            penetration = 0.0;
            return false;
        }
    }

    [unroll]
    for (uint axisA = 0u; axisA < 3u; ++axisA)
    {
        [unroll]
        for (uint axisB = 0u; axisB < 3u; ++axisB)
        {
            if (!TestObbAxis(cross(axesA[axisA], axesB[axisB]), centerDelta, axesA,
                             halfExtentsA, axesB, halfExtentsB, minPenetration, bestAxis))
            {
                normal = 0.0;
                contactPoint = 0.0;
                penetration = 0.0;
                return false;
            }
        }
    }

    normal = bestAxis;
    penetration = minPenetration;
    contactPoint = 0.5 * (BoxSupportPoint(centerA, orientationA, halfExtentsA, normal) +
                          BoxSupportPoint(centerB, orientationB, halfExtentsB, -normal));
    return true;
}

bool GenerateCapsuleCapsuleContact(float3 capsuleA0, float3 capsuleA1, float radiusA,
                                   float3 capsuleB0, float3 capsuleB1, float radiusB,
                                   out float3 normal, out float3 contactPoint,
                                   out float penetration)
{
    float3 pointA;
    float3 pointB;
    ClosestPointsSegmentSegment(capsuleA0, capsuleA1, capsuleB0, capsuleB1, pointA, pointB);
    return GenerateSphereSphereContact(pointA, radiusA, pointB, radiusB, normal, contactPoint,
                                       penetration);
}

bool GenerateBoxCapsuleContact(float3 boxCenter, float4 boxOrientation, float3 halfExtents,
                               float3 capsuleA, float3 capsuleB, float capsuleRadius,
                               out float3 normal, out float3 contactPoint,
                               out float penetration)
{
    float3 segmentPoint;
    float3 boxPoint;
    bool intersects = false;
    ClosestPointsSegmentBox(capsuleA, capsuleB, boxCenter, boxOrientation, halfExtents,
                            segmentPoint, boxPoint, intersects);

    if (!intersects)
    {
        const float3 delta = segmentPoint - boxPoint;
        const float distanceSq = dot(delta, delta);
        if (distanceSq > capsuleRadius * capsuleRadius)
        {
            normal = 0.0;
            contactPoint = 0.0;
            penetration = 0.0;
            return false;
        }

        const float distance = sqrt(max(distanceSq, 0.0));
        normal = distance > kEpsilon ? delta / distance : float3(0.0, 1.0, 0.0);
        penetration = capsuleRadius - distance;
        contactPoint = 0.5 * (boxPoint + (segmentPoint - normal * capsuleRadius));
        return true;
    }

    const float3 localSegment = QuaternionInverseRotate(boxOrientation, segmentPoint - boxCenter);
    const float3 localNormal = ClosestFaceNormalLocal(localSegment, halfExtents);
    normal = QuaternionRotate(boxOrientation, localNormal);
    const float faceDepth =
        min(halfExtents.x - abs(localSegment.x),
            min(halfExtents.y - abs(localSegment.y), halfExtents.z - abs(localSegment.z)));
    const float3 boxSurface =
        boxCenter + QuaternionRotate(boxOrientation, localSegment - localNormal * faceDepth);
    penetration = capsuleRadius + faceDepth;
    contactPoint = 0.5 * (boxSurface + (segmentPoint - normal * capsuleRadius));
    return true;
}

bool GenerateRigidContact(uint shapeTypeA, float3 positionA, float4 orientationA,
                          float4 colliderParamsA, float4 scaleA, uint shapeTypeB,
                          float3 positionB, float4 orientationB, float4 colliderParamsB,
                          float4 scaleB, out float3 normal, out float3 contactPoint,
                          out float penetration)
{
    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderSphere)
    {
        return GenerateSphereSphereContact(positionA, SphereRadius(colliderParamsA, scaleA),
                                           positionB, SphereRadius(colliderParamsB, scaleB),
                                           normal, contactPoint, penetration);
    }

    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderBox)
    {
        return GenerateSphereBoxContact(positionA, SphereRadius(colliderParamsA, scaleA), positionB,
                                        orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                        normal, contactPoint, penetration);
    }

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderSphere)
    {
        const bool result = GenerateSphereBoxContact(positionB, SphereRadius(colliderParamsB, scaleB),
                                                     positionA, orientationA,
                                                     BoxHalfExtents(colliderParamsA, scaleA),
                                                     normal, contactPoint, penetration);
        normal = -normal;
        return result;
    }

    if (shapeTypeA == kColliderSphere && shapeTypeB == kColliderCapsule)
    {
        float3 capsuleB0;
        float3 capsuleB1;
        float capsuleRadiusB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, capsuleB0, capsuleB1,
                       capsuleRadiusB);
        return GenerateSphereCapsuleContact(positionA, SphereRadius(colliderParamsA, scaleA),
                                            capsuleB0, capsuleB1, capsuleRadiusB, normal,
                                            contactPoint,
                                            penetration);
    }

    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderSphere)
    {
        float3 capsuleA0;
        float3 capsuleA1;
        float capsuleRadiusA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, capsuleA0, capsuleA1,
                       capsuleRadiusA);
        const bool result = GenerateSphereCapsuleContact(positionB,
                                                         SphereRadius(colliderParamsB, scaleB),
                                                         capsuleA0, capsuleA1, capsuleRadiusA,
                                                         normal, contactPoint, penetration);
        normal = -normal;
        return result;
    }

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderBox)
    {
        return GenerateBoxBoxContact(positionA, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                                     positionB, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                                     normal, contactPoint, penetration);
    }

    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderCapsule)
    {
        float3 capsuleA0;
        float3 capsuleA1;
        float capsuleRadiusA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, capsuleA0, capsuleA1,
                       capsuleRadiusA);

        float3 capsuleB0;
        float3 capsuleB1;
        float capsuleRadiusB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, capsuleB0, capsuleB1,
                       capsuleRadiusB);

        return GenerateCapsuleCapsuleContact(capsuleA0, capsuleA1, capsuleRadiusA, capsuleB0,
                                             capsuleB1, capsuleRadiusB, normal, contactPoint,
                                             penetration);
    }

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderCapsule)
    {
        float3 capsuleB0;
        float3 capsuleB1;
        float capsuleRadiusB = 0.0;
        CapsuleSegment(positionB, orientationB, colliderParamsB, scaleB, capsuleB0, capsuleB1,
                       capsuleRadiusB);
        return GenerateBoxCapsuleContact(positionA, orientationA,
                                         BoxHalfExtents(colliderParamsA, scaleA), capsuleB0,
                                         capsuleB1, capsuleRadiusB, normal, contactPoint,
                                         penetration);
    }

    if (shapeTypeA == kColliderCapsule && shapeTypeB == kColliderBox)
    {
        float3 capsuleA0;
        float3 capsuleA1;
        float capsuleRadiusA = 0.0;
        CapsuleSegment(positionA, orientationA, colliderParamsA, scaleA, capsuleA0, capsuleA1,
                       capsuleRadiusA);
        const bool result = GenerateBoxCapsuleContact(positionB, orientationB,
                                                      BoxHalfExtents(colliderParamsB, scaleB),
                                                      capsuleA0, capsuleA1, capsuleRadiusA,
                                                      normal, contactPoint, penetration);
        normal = -normal;
        return result;
    }

    normal = 0.0;
    contactPoint = 0.0;
    penetration = 0.0;
    return false;
}

void PairIndexToBodies(uint pairIndex, uint bodyCount, out uint bodyA, out uint bodyB)
{
    uint remaining = pairIndex;
    bodyA = 0u;
    bodyB = 0u;

    [loop]
    for (uint a = 0u; a + 1u < bodyCount; ++a)
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
    const float3 axisX = BoxAxisX(orientation);
    const float3 axisY = BoxAxisY(orientation);
    const float3 axisZ = BoxAxisZ(orientation);
    return axisX * (inverseInertiaLocal.x * dot(axisX, value)) +
           axisY * (inverseInertiaLocal.y * dot(axisY, value)) +
           axisZ * (inverseInertiaLocal.z * dot(axisZ, value));
}

#endif
