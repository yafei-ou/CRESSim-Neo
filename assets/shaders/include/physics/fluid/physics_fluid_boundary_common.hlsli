#ifndef CRESSIM_NEO_PHYSICS_FLUID_BOUNDARY_COMMON_HLSLI
#define CRESSIM_NEO_PHYSICS_FLUID_BOUNDARY_COMMON_HLSLI

#include "../core/physics_math.hlsli"
#include "../particle/physics_particle_types.hlsli"
#include "../collision/physics_shape_common.hlsli"
#include "../rigid/physics_rigid_types.hlsli"
#include "../rigid/physics_rigid_broad_phase_types.hlsli"

bool ComputeFluidBoundaryGhostDelta(uint rigidBodyIndex, uint colliderIndex, float3 selfPosition,
                                    float smoothingRadius, out float3 ghostDelta)
{
    ghostDelta = float3(0.0, 0.0, 0.0);

    const GpuColliderBroadPhaseData broadPhase =
        CRESSIM_SB_LOAD(g_ColliderBroadPhaseData, colliderIndex);
    if (broadPhase.enabledFlag == 0u)
    {
        return false;
    }

    const float3 bodyPosition =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex).xyz;
    const float4 bodyOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
    const float4 bodyScale = CRESSIM_SB_LOAD(g_RigidBodyScales, rigidBodyIndex);
    const GpuColliderGeometryData colliderGeometry =
        CRESSIM_SB_LOAD(g_ColliderGeometryData, colliderIndex);
    const float3 colliderWorldPosition = ComposeColliderWorldPosition(
        bodyPosition, bodyOrientation, colliderGeometry.localPosition.xyz);
    const float4 colliderWorldOrientation = ComposeColliderWorldOrientation(
        bodyOrientation, QuaternionNormalize(colliderGeometry.localOrientation));

    float signedDistance = 0.0;
    float3 normalWorld = float3(0.0, 1.0, 0.0);
    const float4 shapeParams = colliderGeometry.shapeParams;

    if (broadPhase.shapeType == kColliderSphere)
    {
        const float3 delta = selfPosition - colliderWorldPosition;
        const float distance = length(delta);
        const float radius = SphereRadius(shapeParams, bodyScale);
        normalWorld = distance > kEpsilon ? (delta / distance) : float3(0.0, 1.0, 0.0);
        signedDistance = distance - radius;
    }
    else if (broadPhase.shapeType == kColliderBox)
    {
        const float3 halfExtents = BoxHalfExtents(shapeParams, bodyScale);
        const float3 selfLocal =
            QuaternionInverseRotate(colliderWorldOrientation, selfPosition - colliderWorldPosition);
        const float3 closest = clamp(selfLocal, -halfExtents, halfExtents);
        const float3 delta = selfLocal - closest;
        const float deltaLength = length(delta);
        if (deltaLength > kEpsilon)
        {
            const float3 normalLocal = delta / deltaLength;
            normalWorld = QuaternionRotate(colliderWorldOrientation, normalLocal);
            signedDistance = deltaLength;
        }
        else
        {
            const float3 distances = halfExtents - abs(selfLocal);
            const float3 normalLocal = ClosestFaceNormalLocal(selfLocal, halfExtents);
            normalWorld = QuaternionRotate(colliderWorldOrientation, normalLocal);
            signedDistance = -min(distances.x, min(distances.y, distances.z));
        }
    }
    else
    {
        float3 segmentA, segmentB;
        float capsuleRadius = 0.0;
        CapsuleSegment(colliderWorldPosition, colliderWorldOrientation, shapeParams, bodyScale,
                       segmentA, segmentB, capsuleRadius);
        const float3 segment = segmentB - segmentA;
        const float segmentLengthSq = dot(segment, segment);
        float t = 0.0;
        if (segmentLengthSq > kEpsilon)
        {
            t = saturate(dot(selfPosition - segmentA, segment) / segmentLengthSq);
        }
        const float3 closest = lerp(segmentA, segmentB, t);
        const float3 delta = selfPosition - closest;
        const float distance = length(delta);
        normalWorld = distance > kEpsilon ? (delta / distance) : float3(0.0, 1.0, 0.0);
        signedDistance = distance - capsuleRadius;
    }

    if (signedDistance < 0.0 || signedDistance >= smoothingRadius)
    {
        return false;
    }

    ghostDelta = normalWorld * (2.0 * signedDistance);
    return true;
}

void AccumulateFluidBoundaryDensityFromCandidates(uint particleIndex, float3 selfPosition,
                                                  float smoothingRadius, float surfaceTension,
                                                  inout float density,
                                                  inout float3 surfaceNormal)
{
    const uint2 candidateRange = CRESSIM_SB_LOAD(g_FluidBoundaryCandidateRanges, particleIndex);
    const uint candidateOffset = candidateRange.x;
    const uint candidateCount = candidateRange.y;

    [loop]
    for (uint i = 0u; i < candidateCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidBoundaryCandidatePairs, candidateOffset + i);
        const uint colliderIndex = pair.indexB;
        const uint rigidBodyIndex = pair.auxIndex;
        float3 ghostDelta = float3(0.0, 0.0, 0.0);
        if (!ComputeFluidBoundaryGhostDelta(rigidBodyIndex, colliderIndex, selfPosition,
                                            smoothingRadius, ghostDelta))
        {
            continue;
        }

        AccumulateBoundaryDensityContribution(ghostDelta, smoothingRadius, surfaceTension,
                                              density, surfaceNormal);
    }
}

float3 ComputeFluidBoundaryDeltaFromCandidates(uint particleIndex, float3 selfPosition,
                                               float smoothingRadius, float selfConstraint)
{
    float3 boundaryDelta = float3(0.0, 0.0, 0.0);
    const uint2 candidateRange = CRESSIM_SB_LOAD(g_FluidBoundaryCandidateRanges, particleIndex);
    const uint candidateOffset = candidateRange.x;
    const uint candidateCount = candidateRange.y;

    [loop]
    for (uint i = 0u; i < candidateCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidBoundaryCandidatePairs, candidateOffset + i);
        const uint colliderIndex = pair.indexB;
        const uint rigidBodyIndex = pair.auxIndex;
        float3 ghostDelta = float3(0.0, 0.0, 0.0);
        if (!ComputeFluidBoundaryGhostDelta(rigidBodyIndex, colliderIndex, selfPosition,
                                            smoothingRadius, ghostDelta))
        {
            continue;
        }

        boundaryDelta += ComputeBoundaryDeltaContribution(ghostDelta, smoothingRadius,
                                                          max(selfConstraint, 0.0));
    }

    return boundaryDelta * kFluidBoundaryDeltaScale;
}

#endif // CRESSIM_NEO_PHYSICS_FLUID_BOUNDARY_COMMON_HLSLI
