#include "../../../include/physics/physics_particle_dispatch_constants.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float, g_ParticleRadii);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleKinds);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleContactMaterials);
CRESSIM_RW_STRUCTURED_BUFFER(float4, g_ParticleVelocities);
CRESSIM_STRUCTURED_BUFFER(uint2, g_FluidBoundaryCandidateRanges);
CRESSIM_STRUCTURED_BUFFER(GpuParticleCandidatePair, g_FluidBoundaryCandidatePairs);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyScales);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuColliderGeometryData, g_ColliderGeometryData);
CRESSIM_STRUCTURED_BUFFER(GpuColliderBroadPhaseData, g_ColliderBroadPhaseData);

#include "../../../include/physics/fluid/physics_fluid_common.hlsli"
#include "../../../include/physics/fluid/physics_fluid_boundary_common.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint particleIndex = dispatchThreadID.x;
    if (particleIndex >= particleCount ||
        CRESSIM_SB_LOAD(g_ParticleKinds, particleIndex) != kParticleKindFluid)
    {
        return;
    }

    const float3 position = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex).xyz;
    const float particleRadius = max(CRESSIM_SB_LOAD(g_ParticleRadii, particleIndex), 0.0);
    float3 velocity = CRESSIM_SB_LOAD(g_ParticleVelocities, particleIndex).xyz;

    const uint materialIndex = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, particleIndex);
    const float4 particleMaterial = CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndex);
    const float particleDamping = max(particleMaterial.z, 0.0);

    const uint2 candidateRange = CRESSIM_SB_LOAD(g_FluidBoundaryCandidateRanges, particleIndex);
    const uint candidateOffset = candidateRange.x;
    const uint candidateCount = candidateRange.y;

    float bestDistance = 3.402823466e+38;
    float3 bestNormal = float3(0.0, 1.0, 0.0);
    float3 bestCombinedMaterial = float3(0.0, 0.0, 0.0);
    bool foundBoundary = false;

    [loop]
    for (uint i = 0u; i < candidateCount; ++i)
    {
        const GpuParticleCandidatePair pair =
            CRESSIM_SB_LOAD(g_FluidBoundaryCandidatePairs, candidateOffset + i);
        const uint colliderIndex = pair.indexB;
        const uint rigidBodyIndex = pair.auxIndex;

        float signedDistance = 0.0;
        float3 normalWorld = float3(0.0, 1.0, 0.0);
        if (!ComputeFluidBoundaryContactInfo(rigidBodyIndex, colliderIndex, position, signedDistance,
                                             normalWorld))
        {
            continue;
        }

        const float contactDistance = particleRadius + 2.0 * kContactSlop;
        if (signedDistance > contactDistance)
        {
            continue;
        }

        if (signedDistance < bestDistance)
        {
            bestDistance = signedDistance;
            bestNormal = normalWorld;
            bestCombinedMaterial = CombineContactMaterial(
                particleMaterial, CRESSIM_SB_LOAD(g_ColliderMaterials, colliderIndex));
            foundBoundary = true;
        }
    }

    if (!foundBoundary)
    {
        return;
    }

    const float penetration = max(particleRadius - bestDistance, 0.0);
    const float contactDistance = max(particleRadius + 2.0 * kContactSlop, 1.0e-5);
    const float contactWeight = saturate((contactDistance - bestDistance) / contactDistance);

    float3 relativeVelocity = velocity;
    const float normalVelocity = dot(relativeVelocity, bestNormal);
    if (normalVelocity < 0.0)
    {
        relativeVelocity -= bestNormal * normalVelocity;
    }

    float3 tangentialVelocity = ProjectOntoContactTangent(relativeVelocity, bestNormal);
    if (dt > kEpsilon)
    {
        const float3 frictionDisplacement = ComputePositionFrictionDelta(
            tangentialVelocity * dt, max(penetration, kContactSlop),
            bestCombinedMaterial.x, bestCombinedMaterial.z);
        tangentialVelocity -= frictionDisplacement / dt;
    }

    const float tangentialDampingScale = max(1.0 - particleDamping * contactWeight * dt, 0.0);
    tangentialVelocity *= tangentialDampingScale;

    relativeVelocity =
        bestNormal * max(dot(relativeVelocity, bestNormal), 0.0) + tangentialVelocity;
    velocity = relativeVelocity;

    CRESSIM_SB_STORE(g_ParticleVelocities, particleIndex, float4(velocity, 0.0));
}
