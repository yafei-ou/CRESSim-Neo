#include "physics_particle_dispatch_constants.hlsli"
#include "physics_math.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuParticleShapeMembershipRange, g_ParticleShapeMembershipRanges);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleShapeMembershipIndices);
CRESSIM_STRUCTURED_BUFFER(uint, g_MembershipShapeClusterIndices);
CRESSIM_STRUCTURED_BUFFER(GpuShapeCluster, g_ShapeClusters);
CRESSIM_STRUCTURED_BUFFER(GpuShapeClusterMember, g_ShapeClusterMembers);
CRESSIM_STRUCTURED_BUFFER(GpuShapeClusterPose, g_ShapeClusterPoses);

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount)
    {
        return;
    }

    const float4 positionInvMass = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    if (positionInvMass.w <= kEpsilon)
    {
        return;
    }

    const GpuParticleShapeMembershipRange range =
        CRESSIM_SB_LOAD(g_ParticleShapeMembershipRanges, particleIndex);
    float3 weightedCorrection = float3(0.0, 0.0, 0.0);
    float accumulatedWeight = 0.0;
    float stiffnessSum = 0.0;

    for (uint i = 0u; i < range.count; ++i)
    {
        const uint membershipIndexOffset = range.offset + i;
        if (membershipIndexOffset >= particleShapeMembershipIndexCount)
        {
            break;
        }

        const uint memberIndex =
            CRESSIM_SB_LOAD(g_ParticleShapeMembershipIndices, membershipIndexOffset);
        const GpuShapeClusterMember member = CRESSIM_SB_LOAD(g_ShapeClusterMembers, memberIndex);
        if (member.particleIndex != particleIndex)
        {
            continue;
        }

        const uint clusterIndex = CRESSIM_SB_LOAD(g_MembershipShapeClusterIndices, memberIndex);
        if (clusterIndex >= shapeClusterCount)
        {
            continue;
        }

        const GpuShapeCluster cluster = CRESSIM_SB_LOAD(g_ShapeClusters, clusterIndex);
        if ((cluster.flags & kShapeClusterActive) == 0u)
        {
            continue;
        }

        const GpuShapeClusterPose pose = CRESSIM_SB_LOAD(g_ShapeClusterPoses, clusterIndex);
        if (pose.currentCenterAndStatus.w <= 0.0)
        {
            continue;
        }

        const float blendWeight = max(member.blendWeight, 0.0);
        const float3 goal = pose.currentCenterAndStatus.xyz +
                            QuaternionRotate(pose.rotationQuaternion, member.restOffset.xyz);
        weightedCorrection += blendWeight * (goal - positionInvMass.xyz);
        accumulatedWeight += blendWeight;
        stiffnessSum += blendWeight * saturate(cluster.stiffness);
    }

    if (accumulatedWeight <= kEpsilon)
    {
        return;
    }

    float3 correction = (stiffnessSum / accumulatedWeight) *
                        (weightedCorrection / accumulatedWeight);
    if (maximumShapeCorrection > 0.0)
    {
        const float magnitude = length(correction);
        if (magnitude > maximumShapeCorrection)
        {
            correction *= maximumShapeCorrection / magnitude;
        }
    }

    CRESSIM_SB_STORE(g_ParticlePositionsInvMass, particleIndex,
                     float4(positionInvMass.xyz + correction, positionInvMass.w));
}
