#ifndef CRESSIM_NEO_PHYSICS_RIGID_BOX_BOX_MANIFOLD_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_BOX_BOX_MANIFOLD_HLSLI

#include "physics_rigid_contact_primitives.hlsli"
#include "physics_rigid_types.hlsli"

static const float kManifoldPointMergeDistanceSq = kManifoldMergeDistance * kManifoldMergeDistance;
static const uint kBoxBoxCandidateCapacity = 16u;
static const float kEdgeAxisRelativeTolerance = 0.95;
static const float kEdgeAxisAbsoluteTolerance = 1.0e-3;
static const float kFaceAxisAbsoluteTolerance = 5.0e-4;
static const bool kDebugBoxBoxContactSource = false;

static const uint kBoxBoxContactSourceFaceClip = 1u;
static const uint kBoxBoxContactSourceEdge = 2u;
static const uint kBoxBoxContactSourceCornerRecovery = 3u;
static const uint kBoxBoxContactSourceGenericFallback = 4u;

uint EncodeBoxBoxContactReserved(uint source)
{
    return kDebugBoxBoxContactSource ? source : 0u;
}

float3 BoxCornerFromIndex(float3 center, float4 orientation, float3 halfExtents, uint cornerIndex)
{
    const float3 localCorner = float3((cornerIndex & 1u) != 0u ? halfExtents.x : -halfExtents.x,
                                      (cornerIndex & 2u) != 0u ? halfExtents.y : -halfExtents.y,
                                      (cornerIndex & 4u) != 0u ? halfExtents.z : -halfExtents.z);
    return center + QuaternionRotate(orientation, localCorner);
}

bool IsPointInsideBox(float3 queryPoint, float3 boxCenter, float4 boxOrientation, float3 halfExtents)
{
    const float3 localPoint = QuaternionInverseRotate(boxOrientation, queryPoint - boxCenter);
    return all(localPoint >= -halfExtents) && all(localPoint <= halfExtents);
}

float3 ProjectInsidePointToBoxSurfaceAlongDirection(float3 queryPoint, float3 boxCenter,
                                                    float4 boxOrientation, float3 halfExtents,
                                                    float3 direction)
{
    const float3 localPoint = QuaternionInverseRotate(boxOrientation, queryPoint - boxCenter);
    const float3 localDir =
        QuaternionInverseRotate(boxOrientation, SafeNormalize(direction, float3(0.0, 1.0, 0.0)));

    if (!(all(localPoint >= -halfExtents) && all(localPoint <= halfExtents)))
    {
        return ClosestPointOnBox(queryPoint, boxCenter, boxOrientation, halfExtents);
    }

    float bestT = 3.402823466e+38;

    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float d = axis == 0u ? localDir.x : (axis == 1u ? localDir.y : localDir.z);
        if (abs(d) <= kEpsilon)
        {
            continue;
        }

        const float p = axis == 0u ? localPoint.x : (axis == 1u ? localPoint.y : localPoint.z);
        const float face = d > 0.0
                               ? (axis == 0u ? halfExtents.x
                                             : (axis == 1u ? halfExtents.y : halfExtents.z))
                               : (axis == 0u ? -halfExtents.x
                                             : (axis == 1u ? -halfExtents.y : -halfExtents.z));

        const float t = (face - p) / d;
        if (t >= 0.0 && t < bestT)
        {
            bestT = t;
        }
    }

    float3 localSurface = localPoint;
    if (bestT < 3.402823466e+37)
    {
        localSurface = localPoint + localDir * bestT;
    }
    else
    {
        const float3 localFaceNormal = ClosestFaceNormalLocal(localPoint, halfExtents);
        const float faceDepth =
            min(halfExtents.x - abs(localPoint.x),
                min(halfExtents.y - abs(localPoint.y), halfExtents.z - abs(localPoint.z)));
        localSurface = localPoint + localFaceNormal * faceDepth;
    }

    return boxCenter + QuaternionRotate(boxOrientation, localSurface);
}

struct BoxFaceFrame
{
    float3 center;
    float3 normal;
    float3 tangentU;
    float3 tangentV;
    float halfU;
    float halfV;
};

void BuildObbAxes(float4 orientation, out float3 axes[3])
{
    axes[0] = BoxAxisX(orientation);
    axes[1] = BoxAxisY(orientation);
    axes[2] = BoxAxisZ(orientation);
}

void BuildBoxFaceFrame(float3 boxCenter, float3 boxAxes[3], float3 halfExtents,
                       uint axisIndex, float faceSign, out BoxFaceFrame frame)
{
    const uint tangentUIndex = (axisIndex + 1u) % 3u;
    const uint tangentVIndex = (axisIndex + 2u) % 3u;

    frame.normal = boxAxes[axisIndex] * faceSign;
    frame.center = boxCenter + frame.normal * halfExtents[axisIndex];
    frame.tangentU = boxAxes[tangentUIndex];
    frame.tangentV = boxAxes[tangentVIndex];
    frame.halfU = halfExtents[tangentUIndex];
    frame.halfV = halfExtents[tangentVIndex];
}

void BuildIncidentFaceQuad(float3 boxCenter, float3 boxAxes[3], float3 halfExtents,
                           float3 referenceNormal, out float3 quad[4])
{
    uint axisIndex = 0u;
    float bestAlignment = abs(dot(boxAxes[0], referenceNormal));

    [unroll] for (uint i = 1u; i < 3u; ++i)
    {
        const float alignment = abs(dot(boxAxes[i], referenceNormal));
        if (alignment > bestAlignment)
        {
            bestAlignment = alignment;
            axisIndex = i;
        }
    }

    const float faceSign = dot(boxAxes[axisIndex], referenceNormal) >= 0.0 ? -1.0 : 1.0;
    BoxFaceFrame incidentFace;
    BuildBoxFaceFrame(boxCenter, boxAxes, halfExtents, axisIndex, faceSign, incidentFace);

    const float3 du = incidentFace.tangentU * incidentFace.halfU;
    const float3 dv = incidentFace.tangentV * incidentFace.halfV;

    quad[0] = incidentFace.center + du + dv;
    quad[1] = incidentFace.center - du + dv;
    quad[2] = incidentFace.center - du - dv;
    quad[3] = incidentFace.center + du - dv;
}

uint ClipPolygonAgainstPlane(float3 inputVerts[8], uint inputCount,
                             float3 planeOrigin, float3 planeNormal, float planeOffset,
                             bool keepLessEqual, out float3 outputVerts[8])
{
    if (inputCount == 0u)
        return 0u;

    uint outputCount = 0u;
    float3 previous = inputVerts[inputCount - 1u];
    float previousCoord = dot(previous - planeOrigin, planeNormal);
    bool previousInside = keepLessEqual ? (previousCoord <= planeOffset)
                                        : (previousCoord >= planeOffset);

    [loop] for (uint i = 0u; i < 8u; ++i)
    {
        if (i >= inputCount)
            break;

        const float3 current = inputVerts[i];
        const float currentCoord = dot(current - planeOrigin, planeNormal);
        const bool currentInside = keepLessEqual ? (currentCoord <= planeOffset)
                                                 : (currentCoord >= planeOffset);

        if (currentInside != previousInside)
        {
            const float denom = currentCoord - previousCoord;
            const float t = (abs(denom) > kEpsilon)
                                ? ((planeOffset - previousCoord) / denom)
                                : 0.0;
            if (outputCount < 8u)
            {
                outputVerts[outputCount] = lerp(previous, current, saturate(t));
                ++outputCount;
            }
        }

        if (currentInside && outputCount < 8u)
        {
            outputVerts[outputCount] = current;
            ++outputCount;
        }

        previous = current;
        previousCoord = currentCoord;
        previousInside = currentInside;
    }

    return outputCount;
}

bool ComputeBoxBoxSat(float3 centerA, float4 orientationA, float3 halfExtentsA,
                      float3 centerB, float4 orientationB, float3 halfExtentsB,
                      out float3 normalAtoB, out float penetration,
                      out uint axisType, out uint axisA, out uint axisB)
{
    normalAtoB = 0.0;
    penetration = 0.0;
    axisType = kObbAxisFaceA;
    axisA = 0u;
    axisB = 0u;

    float3 axesA[3];
    float3 axesB[3];
    BuildObbAxes(orientationA, axesA);
    BuildObbAxes(orientationB, axesB);

    const float3 centerDelta = centerB - centerA;

    float bestFacePen = 3.402823466e+38;
    float3 bestFaceAxis = float3(0.0, 1.0, 0.0);
    uint bestFaceType = kObbAxisFaceA;
    uint bestFaceA = 0u;
    uint bestFaceB = 0u;

    float bestEdgePen = 3.402823466e+38;
    float3 bestEdgeAxis = float3(0.0, 1.0, 0.0);
    uint bestEdgeType = kObbAxisEdge;
    uint bestEdgeA = 0u;
    uint bestEdgeB = 0u;
    bool hasEdgeAxis = false;

    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        float3 n = axesA[i];
        const float distance = abs(dot(centerDelta, n));
        const float radiusA = BoxProjectionRadius(n, axesA, halfExtentsA);
        const float radiusB = BoxProjectionRadius(n, axesB, halfExtentsB);
        const float overlap = radiusA + radiusB - distance;
        if (overlap < 0.0)
        {
            normalAtoB = 0.0;
            penetration = 0.0;
            axisType = kObbAxisFaceA;
            axisA = 0u;
            axisB = 0u;
            return false;
        }

        if (dot(centerDelta, n) < 0.0)
            n = -n;

        if (overlap < bestFacePen - kFaceAxisAbsoluteTolerance)
        {
            bestFacePen = overlap;
            bestFaceAxis = n;
            bestFaceType = kObbAxisFaceA;
            bestFaceA = i;
            bestFaceB = 0u;
        }
    }

    [unroll] for (uint i = 0u; i < 3u; ++i)
    {
        float3 n = axesB[i];
        const float distance = abs(dot(centerDelta, n));
        const float radiusA = BoxProjectionRadius(n, axesA, halfExtentsA);
        const float radiusB = BoxProjectionRadius(n, axesB, halfExtentsB);
        const float overlap = radiusA + radiusB - distance;
        if (overlap < 0.0)
        {
            normalAtoB = 0.0;
            penetration = 0.0;
            axisType = kObbAxisFaceA;
            axisA = 0u;
            axisB = 0u;
            return false;
        }

        if (dot(centerDelta, n) < 0.0)
            n = -n;

        if (overlap < bestFacePen - kFaceAxisAbsoluteTolerance)
        {
            bestFacePen = overlap;
            bestFaceAxis = n;
            bestFaceType = kObbAxisFaceB;
            bestFaceA = i;
            bestFaceB = 0u;
        }
    }

    [unroll] for (uint a = 0u; a < 3u; ++a)
    {
        [unroll] for (uint b = 0u; b < 3u; ++b)
        {
            float3 axis = cross(axesA[a], axesB[b]);
            const float axisLenSq = dot(axis, axis);
            if (axisLenSq <= kEpsilon)
                continue;
            hasEdgeAxis = true;

            float3 n = axis * rsqrt(axisLenSq);
            const float distance = abs(dot(centerDelta, n));
            const float radiusA = BoxProjectionRadius(n, axesA, halfExtentsA);
            const float radiusB = BoxProjectionRadius(n, axesB, halfExtentsB);
            const float overlap = radiusA + radiusB - distance;
            if (overlap < 0.0)
            {
                normalAtoB = 0.0;
                penetration = 0.0;
                axisType = kObbAxisFaceA;
                axisA = 0u;
                axisB = 0u;
                return false;
            }

            if (dot(centerDelta, n) < 0.0)
                n = -n;

            if (overlap < bestEdgePen)
            {
                bestEdgePen = overlap;
                bestEdgeAxis = n;
                bestEdgeType = kObbAxisEdge;
                bestEdgeA = a;
                bestEdgeB = b;
            }
        }
    }

    const bool edgeClearlyBetter =
        hasEdgeAxis &&
        (bestEdgePen < bestFacePen * kEdgeAxisRelativeTolerance - kEdgeAxisAbsoluteTolerance);

    if (edgeClearlyBetter)
    {
        normalAtoB = bestEdgeAxis;
        penetration = bestEdgePen;
        axisType = bestEdgeType;
        axisA = bestEdgeA;
        axisB = bestEdgeB;
    }
    else
    {
        normalAtoB = bestFaceAxis;
        penetration = bestFacePen;
        axisType = bestFaceType;
        axisA = bestFaceA;
        axisB = bestFaceB;
    }
    return (penetration > 0.0);
}

struct BoxBoxManifoldCandidate
{
    float3 pointAWorld;
    float3 pointBWorld;
    float penetration;
};

void TryInsertManifoldCandidate(float3 pointAWorld, float3 pointBWorld, float penetration,
                                inout uint candidateCount,
                                inout BoxBoxManifoldCandidate candidates[kBoxBoxCandidateCapacity])
{
    if (penetration <= kEpsilon)
        return;

    const float3 candidateMid = 0.5 * (pointAWorld + pointBWorld);

    [unroll] for (uint i = 0u; i < kBoxBoxCandidateCapacity; ++i)
    {
        if (i >= candidateCount)
            break;

        const float3 existingMid = 0.5 * (candidates[i].pointAWorld + candidates[i].pointBWorld);
        const float3 d = existingMid - candidateMid;
        if (dot(d, d) <= kManifoldPointMergeDistanceSq)
        {
            if (penetration > candidates[i].penetration)
            {
                candidates[i].pointAWorld = pointAWorld;
                candidates[i].pointBWorld = pointBWorld;
                candidates[i].penetration = penetration;
            }
            return;
        }
    }

    if (candidateCount < kBoxBoxCandidateCapacity)
    {
        candidates[candidateCount].pointAWorld = pointAWorld;
        candidates[candidateCount].pointBWorld = pointBWorld;
        candidates[candidateCount].penetration = penetration;
        ++candidateCount;
        return;
    }

    uint replaceIndex = 0u;
    float minPen = candidates[0].penetration;

    [unroll] for (uint i = 1u; i < kBoxBoxCandidateCapacity; ++i)
    {
        if (candidates[i].penetration < minPen)
        {
            minPen = candidates[i].penetration;
            replaceIndex = i;
        }
    }

    if (penetration > minPen)
    {
        candidates[replaceIndex].pointAWorld = pointAWorld;
        candidates[replaceIndex].pointBWorld = pointBWorld;
        candidates[replaceIndex].penetration = penetration;
    }
}

uint GenerateBoxBoxManifoldContacts(uint bodyA, uint bodyB,
                                    float3 bodyPositionA, float4 bodyOrientationA,
                                    float3 centerA, float4 orientationA, float4 colliderParamsA,
                                    float4 scaleA, float3 bodyPositionB, float4 bodyOrientationB,
                                    float3 centerB, float4 orientationB, float4 colliderParamsB,
                                    float4 scaleB,
                                    float3 normalAtoB, float3 contactMaterial,
                                    out uint contactSource,
                                    inout GpuRigidContact contacts[kRigidContactsPerPair])
{
    contactSource = 0u;
    BoxBoxManifoldCandidate candidates[kBoxBoxCandidateCapacity];
    [unroll] for (uint i = 0u; i < kBoxBoxCandidateCapacity; ++i)
    {
        candidates[i].pointAWorld = 0.0;
        candidates[i].pointBWorld = 0.0;
        candidates[i].penetration = 0.0;
    }
    uint candidateCount = 0u;

    const float3 halfA = BoxHalfExtents(colliderParamsA, scaleA);
    const float3 halfB = BoxHalfExtents(colliderParamsB, scaleB);
    float3 manifoldNormalAtoB = SafeNormalize(normalAtoB, float3(0.0, 1.0, 0.0));

    float3 satNormal = 0.0;
    float satPenetration = 0.0;
    uint satAxisType = kObbAxisFaceA;
    uint satAxisA = 0u;
    uint satAxisB = 0u;
    const bool satHit = ComputeBoxBoxSat(centerA, orientationA, halfA, centerB, orientationB, halfB,
                                         satNormal, satPenetration, satAxisType, satAxisA, satAxisB);
    if (satHit)
    {
        manifoldNormalAtoB = SafeNormalize(satNormal, manifoldNormalAtoB);
    }

    if (satHit && satAxisType == kObbAxisEdge)
    {
        float3 axesA[3];
        float3 axesB[3];
        BuildObbAxes(orientationA, axesA);
        BuildObbAxes(orientationB, axesB);

        float3 edgeA0, edgeA1, edgeB0, edgeB1;
        BuildBoxSupportEdge(satAxisA, centerA, axesA, halfA, manifoldNormalAtoB, edgeA0, edgeA1);
        BuildBoxSupportEdge(satAxisB, centerB, axesB, halfB, -manifoldNormalAtoB, edgeB0, edgeB1);

        float3 pointAEdge = 0.0;
        float3 pointBEdge = 0.0;
        ClosestPointsSegmentSegment(edgeA0, edgeA1, edgeB0, edgeB1, pointAEdge, pointBEdge);
        const float penetrationEdge =
            max(satPenetration, ComputePenetrationFromPoints(pointAEdge, pointBEdge, manifoldNormalAtoB));
        TryInsertManifoldCandidate(pointAEdge, pointBEdge, penetrationEdge, candidateCount, candidates);
        if (candidateCount > 0u)
        {
            contactSource = kBoxBoxContactSourceEdge;
        }
    }

    if (satHit && (satAxisType == kObbAxisFaceA || satAxisType == kObbAxisFaceB))
    {
        float3 axesA[3];
        float3 axesB[3];
        BuildObbAxes(orientationA, axesA);
        BuildObbAxes(orientationB, axesB);

        const bool referenceIsA = (satAxisType == kObbAxisFaceA);
        BoxFaceFrame referenceFace;

        if (referenceIsA)
        {
            const float refSign = dot(axesA[satAxisA], manifoldNormalAtoB) >= 0.0 ? 1.0 : -1.0;
            BuildBoxFaceFrame(centerA, axesA, halfA, satAxisA, refSign, referenceFace);
        }
        else
        {
            const float refSign = dot(axesB[satAxisA], -manifoldNormalAtoB) >= 0.0 ? 1.0 : -1.0;
            BuildBoxFaceFrame(centerB, axesB, halfB, satAxisA, refSign, referenceFace);
        }

        float3 incidentQuad[4] = {(float3)0, (float3)0, (float3)0, (float3)0};
        if (referenceIsA)
        {
            BuildIncidentFaceQuad(centerB, axesB, halfB, referenceFace.normal, incidentQuad);
        }
        else
        {
            BuildIncidentFaceQuad(centerA, axesA, halfA, referenceFace.normal, incidentQuad);
        }

        float3 polyA[8];
        float3 polyB[8];
        [unroll] for (uint i = 0u; i < 8u; ++i)
        {
            polyA[i] = 0.0;
            polyB[i] = 0.0;
            if (i < 4u)
            {
                polyA[i] = incidentQuad[i];
            }
        }

        uint polyCount = 4u;
        polyCount = ClipPolygonAgainstPlane(polyA, polyCount, referenceFace.center,
                                            referenceFace.tangentU, referenceFace.halfU, true,
                                            polyB);
        polyCount = ClipPolygonAgainstPlane(polyB, polyCount, referenceFace.center,
                                            referenceFace.tangentU, -referenceFace.halfU, false,
                                            polyA);
        polyCount = ClipPolygonAgainstPlane(polyA, polyCount, referenceFace.center,
                                            referenceFace.tangentV, referenceFace.halfV, true,
                                            polyB);
        polyCount = ClipPolygonAgainstPlane(polyB, polyCount, referenceFace.center,
                                            referenceFace.tangentV, -referenceFace.halfV, false,
                                            polyA);

        [loop] for (uint i = 0u; i < 8u; ++i)
        {
            if (i >= polyCount)
                break;

            const float3 incidentPoint = polyA[i];
            const float separation = dot(incidentPoint - referenceFace.center, referenceFace.normal);
            if (separation > kContactSlop)
                continue;

            const float3 referencePoint = incidentPoint - referenceFace.normal * separation;
            const float penetration = max(0.0, -separation);

            if (referenceIsA)
            {
                TryInsertManifoldCandidate(referencePoint, incidentPoint, penetration,
                                           candidateCount, candidates);
            }
            else
            {
                TryInsertManifoldCandidate(incidentPoint, referencePoint, penetration,
                                           candidateCount, candidates);
            }
        }

        if (candidateCount > 0u)
        {
            contactSource = kBoxBoxContactSourceFaceClip;
        }
    }

    if (candidateCount == 0u)
    {
        [unroll] for (uint corner = 0u; corner < 8u; ++corner)
        {
            const float3 cornerA = BoxCornerFromIndex(centerA, orientationA, halfA, corner);
            float3 pointBFromA = ClosestPointOnBox(cornerA, centerB, orientationB, halfB);
            if (IsPointInsideBox(cornerA, centerB, orientationB, halfB))
            {
                pointBFromA = ProjectInsidePointToBoxSurfaceAlongDirection(
                    cornerA, centerB, orientationB, halfB, -manifoldNormalAtoB);
            }

            const float penetrationA = ComputePenetrationFromPoints(cornerA, pointBFromA,
                                                                    manifoldNormalAtoB);
            TryInsertManifoldCandidate(cornerA, pointBFromA, penetrationA, candidateCount,
                                       candidates);

            const float3 cornerB = BoxCornerFromIndex(centerB, orientationB, halfB, corner);
            float3 pointAFromB = ClosestPointOnBox(cornerB, centerA, orientationA, halfA);
            if (IsPointInsideBox(cornerB, centerA, orientationA, halfA))
            {
                pointAFromB = ProjectInsidePointToBoxSurfaceAlongDirection(
                    cornerB, centerA, orientationA, halfA, manifoldNormalAtoB);
            }

            const float penetrationB = ComputePenetrationFromPoints(pointAFromB, cornerB,
                                                                    manifoldNormalAtoB);
            TryInsertManifoldCandidate(pointAFromB, cornerB, penetrationB, candidateCount,
                                       candidates);
        }

        if (candidateCount > 0u)
        {
            contactSource = kBoxBoxContactSourceCornerRecovery;
        }
    }

    if (candidateCount == 0u)
    {
        return 0u;
    }

    const float3 n = manifoldNormalAtoB;
    const float3 tangentX =
        SafeNormalize(cross(abs(n.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), n),
                      float3(1.0, 0.0, 0.0));
    const float3 tangentY = cross(n, tangentX);

    float2 projectedMids[kBoxBoxCandidateCapacity];
    [unroll] for (uint i = 0u; i < kBoxBoxCandidateCapacity; ++i)
    {
        if (i >= candidateCount)
            break;
        const float3 mid = 0.5 * (candidates[i].pointAWorld + candidates[i].pointBWorld);
        projectedMids[i] = float2(dot(mid, tangentX), dot(mid, tangentY));
    }

    bool selected[kBoxBoxCandidateCapacity];
    [unroll] for (uint i = 0u; i < kBoxBoxCandidateCapacity; ++i)
    {
        selected[i] = false;
    }

    const uint targetCount = min(candidateCount, kRigidContactsPerPair);
    uint selectedCount = 0u;

    uint deepestIndex = 0u;
    float deepestPen = candidates[0].penetration;
    [unroll] for (uint i = 1u; i < kBoxBoxCandidateCapacity; ++i)
    {
        if (i >= candidateCount)
            break;
        if (candidates[i].penetration > deepestPen)
        {
            deepestPen = candidates[i].penetration;
            deepestIndex = i;
        }
    }
    selected[deepestIndex] = true;
    contacts[selectedCount].bodyA = bodyA;
    contacts[selectedCount].bodyB = bodyB;
    contacts[selectedCount].active = 1u;
    contacts[selectedCount].reserved = EncodeBoxBoxContactReserved(contactSource);
    contacts[selectedCount].normalPenetration = float4(n, candidates[deepestIndex].penetration);
    contacts[selectedCount].material = float4(contactMaterial, 0.0);
    contacts[selectedCount].localPointA =
        float4(QuaternionInverseRotate(bodyOrientationA,
                                       candidates[deepestIndex].pointAWorld - bodyPositionA),
               1.0);
    contacts[selectedCount].localPointB =
        float4(QuaternionInverseRotate(bodyOrientationB,
                                       candidates[deepestIndex].pointBWorld - bodyPositionB),
               1.0);
    ++selectedCount;

    [loop] while (selectedCount < targetCount)
    {
        uint bestIndex = 0u;
        float bestScore = -1.0;
        bool found = false;

        [unroll] for (uint i = 0u; i < kBoxBoxCandidateCapacity; ++i)
        {
            if (i >= candidateCount)
                break;
            if (selected[i])
                continue;

            float minDistSq = 3.402823466e+38;
            [unroll] for (uint j = 0u; j < kBoxBoxCandidateCapacity; ++j)
            {
                if (j >= candidateCount)
                    break;
                if (!selected[j])
                    continue;
                const float2 d = projectedMids[i] - projectedMids[j];
                minDistSq = min(minDistSq, dot(d, d));
            }

            const float score = minDistSq + candidates[i].penetration * 1.0e-4;
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
                found = true;
            }
        }

        if (!found)
            break;

        selected[bestIndex] = true;
        contacts[selectedCount].bodyA = bodyA;
        contacts[selectedCount].bodyB = bodyB;
        contacts[selectedCount].active = 1u;
        contacts[selectedCount].reserved = EncodeBoxBoxContactReserved(contactSource);
        contacts[selectedCount].normalPenetration = float4(n, candidates[bestIndex].penetration);
        contacts[selectedCount].material = float4(contactMaterial, 0.0);
        contacts[selectedCount].localPointA =
            float4(QuaternionInverseRotate(bodyOrientationA,
                                           candidates[bestIndex].pointAWorld - bodyPositionA),
                   1.0);
        contacts[selectedCount].localPointB =
            float4(QuaternionInverseRotate(bodyOrientationB,
                                           candidates[bestIndex].pointBWorld - bodyPositionB),
                   1.0);
        ++selectedCount;
    }

    return selectedCount;
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_BOX_BOX_MANIFOLD_HLSLI
