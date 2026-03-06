cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint activeDynamicCount;
    uint candidatePairCount;
    uint candidatePairCapacity;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyScales;
StructuredBuffer<uint> g_RigidBodyColliderShapeTypes;
StructuredBuffer<float4> g_RigidBodyColliderParams;
StructuredBuffer<GpuCandidatePair> g_CandidatePairs;

RWStructuredBuffer<GpuRigidContact> g_RigidContacts;

// Scale-aware manifold merge radius
static const float kManifoldPointMergeDistanceSq = kManifoldMergeDistance * kManifoldMergeDistance;
static const uint kBoxBoxCandidateCapacity = 16u;
static const float kEdgeAxisRelativeTolerance = 0.95;
static const float kEdgeAxisAbsoluteTolerance = 1.0e-3;

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
        // Degenerate direction: fallback to nearest face projection.
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

        if (overlap < bestFacePen)
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

        if (overlap < bestFacePen)
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

    // Replace the shallowest penetration
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
                                    float3 centerA, float4 orientationA, float4 colliderParamsA, float4 scaleA,
                                    float3 centerB, float4 orientationB, float4 colliderParamsB, float4 scaleB,
                                    float3 normalAtoB,
                                    inout GpuRigidContact contacts[kRigidContactsPerPair])
{
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
    }

    // Industrial-style face manifold generation: incident face clipped to reference face.
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

        float3 incidentQuad[4];
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
        [unroll] for (uint i = 0u; i < 4u; ++i)
        {
            polyA[i] = incidentQuad[i];
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
    }

    // Corner projection fallback/supplement for face clipping degeneracies.
    [unroll] for (uint corner = 0u; corner < 8u; ++corner)
    {
        // A corner against B
        const float3 cornerA = BoxCornerFromIndex(centerA, orientationA, halfA, corner);
        float3 pointBFromA = ClosestPointOnBox(cornerA, centerB, orientationB, halfB);
        if (IsPointInsideBox(cornerA, centerB, orientationB, halfB))
        {
            pointBFromA = ProjectInsidePointToBoxSurfaceAlongDirection(
                cornerA, centerB, orientationB, halfB, -manifoldNormalAtoB);
        }

        // Store separate points: point on A is the corner, point on B is closest point
        const float penetrationA = ComputePenetrationFromPoints(cornerA, pointBFromA,
                                                                manifoldNormalAtoB);
        TryInsertManifoldCandidate(cornerA, pointBFromA, penetrationA, candidateCount, candidates);

        // B corner against A
        const float3 cornerB = BoxCornerFromIndex(centerB, orientationB, halfB, corner);
        float3 pointAFromB = ClosestPointOnBox(cornerB, centerA, orientationA, halfA);
        if (IsPointInsideBox(cornerB, centerA, orientationA, halfA))
        {
            pointAFromB = ProjectInsidePointToBoxSurfaceAlongDirection(
                cornerB, centerA, orientationA, halfA, manifoldNormalAtoB);
        }

        // point on A is closestA, point on B is cornerB
        const float penetrationB = ComputePenetrationFromPoints(pointAFromB, cornerB,
                                                                manifoldNormalAtoB);
        TryInsertManifoldCandidate(pointAFromB, cornerB, penetrationB, candidateCount, candidates);
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

    // Start with deepest contact.
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
    contacts[selectedCount].reserved = 0u;
    contacts[selectedCount].normalPenetration = float4(n, candidates[deepestIndex].penetration);
    contacts[selectedCount].localPointA =
        float4(QuaternionInverseRotate(orientationA, candidates[deepestIndex].pointAWorld - centerA), 1.0);
    contacts[selectedCount].localPointB =
        float4(QuaternionInverseRotate(orientationB, candidates[deepestIndex].pointBWorld - centerB), 1.0);
    ++selectedCount;

    // Fill remaining contacts by maximizing planar spread.
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
        contacts[selectedCount].reserved = 0u;
        contacts[selectedCount].normalPenetration = float4(n, candidates[bestIndex].penetration);
        contacts[selectedCount].localPointA =
            float4(QuaternionInverseRotate(orientationA, candidates[bestIndex].pointAWorld - centerA), 1.0);
        contacts[selectedCount].localPointB =
            float4(QuaternionInverseRotate(orientationB, candidates[bestIndex].pointBWorld - centerB), 1.0);
        ++selectedCount;
    }

    return selectedCount;
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint pairIndex = dispatchThreadID.x;
    if (pairIndex >= candidatePairCount)
    {
        return;
    }

    const uint contactBaseIndex = pairIndex * kRigidContactsPerPair;
    [unroll] for (uint contactOffset = 0u; contactOffset < kRigidContactsPerPair; ++contactOffset)
    {
        GpuRigidContact cleared;
        cleared.bodyA = 0u;
        cleared.bodyB = 0u;
        cleared.active = 0u;
        cleared.reserved = 0u;
        cleared.normalPenetration = 0.0;
        cleared.localPointA = 0.0;
        cleared.localPointB = 0.0;
        g_RigidContacts[contactBaseIndex + contactOffset] = cleared;
    }

    const GpuCandidatePair pair = g_CandidatePairs[pairIndex];
    const uint bodyA = pair.bodyA;
    const uint bodyB = pair.bodyB;

    const float4 positionInvMassA = g_PredictedRigidBodyPositionsInvMass[bodyA];
    const float4 positionInvMassB = g_PredictedRigidBodyPositionsInvMass[bodyB];
    if (positionInvMassA.w == 0.0 && positionInvMassB.w == 0.0)
    {
        return;
    }

    const float4 orientationA = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyA]);
    const float4 orientationB = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyB]);
    const uint shapeTypeA = g_RigidBodyColliderShapeTypes[bodyA];
    const uint shapeTypeB = g_RigidBodyColliderShapeTypes[bodyB];
    const float4 colliderParamsA = g_RigidBodyColliderParams[bodyA];
    const float4 colliderParamsB = g_RigidBodyColliderParams[bodyB];
    const float4 scaleA = g_RigidBodyScales[bodyA];
    const float4 scaleB = g_RigidBodyScales[bodyB];

    float3 aabbMinA;
    float3 aabbMaxA;
    float3 aabbMinB;
    float3 aabbMaxB;
    ComputeBodyAabb(shapeTypeA, positionInvMassA.xyz, orientationA, colliderParamsA, scaleA,
                    aabbMinA, aabbMaxA);
    ComputeBodyAabb(shapeTypeB, positionInvMassB.xyz, orientationB, colliderParamsB, scaleB,
                    aabbMinB, aabbMaxB);
    if (!AabbOverlaps(aabbMinA, aabbMaxA, aabbMinB, aabbMaxB))
    {
        return;
    }

    float3 normalAtoB = 0.0;
    float3 pointAWorld = 0.0;
    float3 pointBWorld = 0.0;
    float penetration = 0.0;

    if (shapeTypeA == kColliderBox && shapeTypeB == kColliderBox)
    {
        float3 satNormal = 0.0;
        float satPenetration = 0.0;
        uint satAxisType = kObbAxisFaceA;
        uint satAxisA = 0u;
        uint satAxisB = 0u;
        if (!ComputeBoxBoxSat(positionInvMassA.xyz, orientationA, BoxHalfExtents(colliderParamsA, scaleA),
                              positionInvMassB.xyz, orientationB, BoxHalfExtents(colliderParamsB, scaleB),
                              satNormal, satPenetration, satAxisType, satAxisA, satAxisB))
        {
            return;
        }

        const float3 contactNormal = SafeNormalize(satNormal, float3(0.0, 1.0, 0.0));

        GpuRigidContact manifoldContacts[kRigidContactsPerPair];
        [unroll] for (uint i = 0u; i < kRigidContactsPerPair; ++i)
        {
            GpuRigidContact cleared;
            cleared.bodyA = bodyA;
            cleared.bodyB = bodyB;
            cleared.active = 0u;
            cleared.reserved = 0u;
            cleared.normalPenetration = 0.0;
            cleared.localPointA = 0.0;
            cleared.localPointB = 0.0;
            manifoldContacts[i] = cleared;
        }

        const uint manifoldCount =
            GenerateBoxBoxManifoldContacts(bodyA, bodyB, positionInvMassA.xyz, orientationA,
                                           colliderParamsA, scaleA, positionInvMassB.xyz,
                                           orientationB, colliderParamsB, scaleB, contactNormal,
                                           manifoldContacts);
        if (manifoldCount > 0u)
        {
            [unroll] for (uint i = 0u; i < kRigidContactsPerPair; ++i)
            {
                if (i >= manifoldCount)
                {
                    break;
                }
                g_RigidContacts[contactBaseIndex + i] = manifoldContacts[i];
            }
            return;
        }
    }

    if (!GenerateRigidContact(shapeTypeA, positionInvMassA.xyz, orientationA, colliderParamsA, scaleA,
                              shapeTypeB, positionInvMassB.xyz, orientationB, colliderParamsB, scaleB,
                              normalAtoB, pointAWorld, pointBWorld, penetration) ||
        penetration <= 0.0)
    {
        return;
    }

    const float3 contactNormal = SafeNormalize(normalAtoB, float3(0.0, 1.0, 0.0));

    GpuRigidContact contact;
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.active = 1u;
    contact.reserved = 0u;
    contact.normalPenetration = float4(contactNormal, penetration);
    contact.localPointA = float4(QuaternionInverseRotate(orientationA, pointAWorld - positionInvMassA.xyz), 1.0);
    contact.localPointB = float4(QuaternionInverseRotate(orientationB, pointBWorld - positionInvMassB.xyz), 1.0);
    g_RigidContacts[contactBaseIndex] = contact;
}
