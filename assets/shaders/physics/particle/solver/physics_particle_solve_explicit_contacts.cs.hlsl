#include "physics/physics_atomic_float.hlsli"
#include "physics/particle/physics_particle_types.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"
#include "physics/core/physics_math.hlsli"

static const float kSoftContactRelaxation = 0.95;
static const float kSoftMaxCorrectionPerIter = 0.05;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerTypes);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleOwnerIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidProxyLocalPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuParticleContact, g_ParticleContacts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (contactIndex >= meta.activeParticleContactCount)
    {
        return;
    }

    const GpuParticleContact contact = CRESSIM_SB_LOAD(g_ParticleContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint particleA = contact.particleA;
    const uint particleB = contact.particleB;
    const float4 particleAState = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleA);
    const float4 particleBState = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleB);
    const float3 particlePreviousPositionA =
        CRESSIM_SB_LOAD(g_ParticlePreviousPositions, particleA).xyz;
    const float3 particlePreviousPositionB =
        CRESSIM_SB_LOAD(g_ParticlePreviousPositions, particleB).xyz;
    const uint ownerTypeA = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleA);
    const uint ownerTypeB = CRESSIM_SB_LOAD(g_ParticleOwnerTypes, particleB);
    const bool proxyA = ownerTypeA == kParticleOwnerTypeRigidBody;
    const bool proxyB = ownerTypeB == kParticleOwnerTypeRigidBody;

    float invMassA = particleAState.w;
    float invMassB = particleBState.w;
    float3 currentPointA = particleAState.xyz;
    float3 currentPointB = particleBState.xyz;
    float3 previousPointA = particlePreviousPositionA;
    float3 previousPointB = particlePreviousPositionB;
    float3 rA = 0.0;
    float3 rB = 0.0;
    float3 invInertiaA = 0.0;
    float3 invInertiaB = 0.0;
    float4 orientationA = float4(0.0, 0.0, 0.0, 1.0);
    float4 orientationB = float4(0.0, 0.0, 0.0, 1.0);
    uint rigidBodyIndexA = 0u;
    uint rigidBodyIndexB = 0u;

    if (proxyA)
    {
        rigidBodyIndexA = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleA);
        const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndexA);
        const float4 rigidPositionInvMassA =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndexA);
        const float4 previousRigidPositionInvMassA =
            CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, rigidBodyIndexA);
        orientationA =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndexA));
        const float4 previousOrientationA =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, rigidBodyIndexA));
        invMassA = bodyTypeA == kRigidBodyTypeDynamic ? rigidPositionInvMassA.w : 0.0;
        invInertiaA = invMassA > kEpsilon
                          ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndexA).xyz
                          : 0.0;
        const float3 localProxyA = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleA).xyz;
        currentPointA = rigidPositionInvMassA.xyz + QuaternionRotate(orientationA, localProxyA);
        previousPointA =
            previousRigidPositionInvMassA.xyz + QuaternionRotate(previousOrientationA, localProxyA);
        rA = currentPointA - rigidPositionInvMassA.xyz;
    }

    if (proxyB)
    {
        rigidBodyIndexB = CRESSIM_SB_LOAD(g_ParticleOwnerIndices, particleB);
        const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndexB);
        const float4 rigidPositionInvMassB =
            CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndexB);
        const float4 previousRigidPositionInvMassB =
            CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, rigidBodyIndexB);
        orientationB =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndexB));
        const float4 previousOrientationB =
            QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, rigidBodyIndexB));
        invMassB = bodyTypeB == kRigidBodyTypeDynamic ? rigidPositionInvMassB.w : 0.0;
        invInertiaB = invMassB > kEpsilon
                          ? CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndexB).xyz
                          : 0.0;
        const float3 localProxyB = CRESSIM_SB_LOAD(g_RigidProxyLocalPositions, particleB).xyz;
        currentPointB = rigidPositionInvMassB.xyz + QuaternionRotate(orientationB, localProxyB);
        previousPointB =
            previousRigidPositionInvMassB.xyz + QuaternionRotate(previousOrientationB, localProxyB);
        rB = currentPointB - rigidPositionInvMassB.xyz;
    }

    if (invMassA <= kEpsilon && invMassB <= kEpsilon)
    {
        return;
    }

    const float penetration =
        min(max(contact.normalPenetration.w - kContactSlop, 0.0), kSoftMaxCorrectionPerIter);
    if (penetration <= 0.0)
    {
        return;
    }

    const float3 combinedMaterial = contact.material.xyz;
    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 relativeDisplacement =
        (currentPointB - previousPointB) - (currentPointA - previousPointA);
    const float normalMassA = proxyA
                                  ? ComputeContactEffectiveMass(invMassA, invInertiaA, orientationA,
                                                                rA, normal)
                                  : invMassA;
    const float normalMassB = proxyB
                                  ? ComputeContactEffectiveMass(invMassB, invInertiaB, orientationB,
                                                                rB, normal)
                                  : invMassB;
    const float denom = normalMassA + normalMassB;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = (penetration / denom) * kSoftContactRelaxation;
    float3 correctionA = 0.0;
    float3 correctionB = 0.0;
    float3 rigidTranslationA = 0.0;
    float3 rigidTranslationB = 0.0;
    float3 rigidRotationA = 0.0;
    float3 rigidRotationB = 0.0;

    if (proxyA)
    {
        rigidTranslationA = -normal * (invMassA * lambda);
        rigidRotationA =
            -MultiplyWorldInverseInertia(invInertiaA, orientationA, cross(rA, normal)) * lambda;
    }
    else
    {
        correctionA = -normal * (invMassA * lambda);
    }

    if (proxyB)
    {
        rigidTranslationB = normal * (invMassB * lambda);
        rigidRotationB =
            MultiplyWorldInverseInertia(invInertiaB, orientationB, cross(rB, normal)) * lambda;
    }
    else
    {
        correctionB = normal * (invMassB * lambda);
    }

    const float3 tangentialDisplacement = ProjectOntoContactTangent(relativeDisplacement, normal);
    const float3 frictionDelta = ComputePositionFrictionDelta(
        tangentialDisplacement, penetration, combinedMaterial.x, combinedMaterial.z);
    const float frictionDistance = length(frictionDelta);
    if (frictionDistance > 0.0)
    {
        const float3 tangent = frictionDelta / frictionDistance;
        const float tangentMassA = proxyA
                                       ? ComputeContactEffectiveMass(invMassA, invInertiaA,
                                                                     orientationA, rA, tangent)
                                       : invMassA;
        const float tangentMassB = proxyB
                                       ? ComputeContactEffectiveMass(invMassB, invInertiaB,
                                                                     orientationB, rB, tangent)
                                       : invMassB;
        const float tangentDenom = tangentMassA + tangentMassB;
        if (tangentDenom > kEpsilon)
        {
            const float tangentLambda = (frictionDistance / tangentDenom) * kSoftContactRelaxation;
            if (proxyA)
            {
                rigidTranslationA += tangent * (invMassA * tangentLambda);
                rigidRotationA += MultiplyWorldInverseInertia(invInertiaA, orientationA,
                                                              cross(rA, tangent)) *
                                  tangentLambda;
            }
            else
            {
                correctionA += tangent * (invMassA * tangentLambda);
            }

            if (proxyB)
            {
                rigidTranslationB -= tangent * (invMassB * tangentLambda);
                rigidRotationB -= MultiplyWorldInverseInertia(invInertiaB, orientationB,
                                                              cross(rB, tangent)) *
                                  tangentLambda;
            }
            else
            {
                correctionB -= tangent * (invMassB * tangentLambda);
            }
        }
    }

    if (!proxyA && invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleA, correctionA);
    }

    if (!proxyB && invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleB, correctionB);
    }

    if (proxyA && invMassA > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, rigidBodyIndexA,
                                      rigidTranslationA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, rigidBodyIndexA,
                                      rigidRotationA);
    }

    if (proxyB && invMassB > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, rigidBodyIndexB,
                                      rigidTranslationB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, rigidBodyIndexB,
                                      rigidRotationB);
    }
}
