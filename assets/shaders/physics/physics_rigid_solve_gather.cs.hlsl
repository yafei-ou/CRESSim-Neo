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

static const float kMaxCorrectionPerIter = 0.02; // world units, tune (e.g. 2 cm)
static const float kRelaxation = 0.90;           // try 0.8 if jittery

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyInverseInertiaLocal;
StructuredBuffer<GpuRigidContact> g_RigidContacts;

RWStructuredBuffer<float4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<float4> g_RigidBodyRotationCorrections;

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    float3 translationCorrection = 0.0;
    float3 rotationCorrection = 0.0;

    for (uint pairIdx = 0u; pairIdx < candidatePairCount; ++pairIdx)
    {
        const uint contactBaseIndex = pairIdx * kRigidContactsPerPair;
        [unroll] for (uint contactOffset = 0u; contactOffset < kRigidContactsPerPair; ++contactOffset)
        {
            const GpuRigidContact contact = g_RigidContacts[contactBaseIndex + contactOffset];
            if (contact.active == 0u)
                continue;
            if (contact.bodyA != bodyIndex && contact.bodyB != bodyIndex)
                continue;

            const uint bodyA = contact.bodyA;
            const uint bodyB = contact.bodyB;

            const float4 posInvMassA = g_PredictedRigidBodyPositionsInvMass[bodyA];
            const float4 posInvMassB = g_PredictedRigidBodyPositionsInvMass[bodyB];

            const float invMassA = posInvMassA.w;
            const float invMassB = posInvMassB.w;
            if (invMassA == 0.0 && invMassB == 0.0)
                continue;

            const float4 qA = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyA]);
            const float4 qB = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyB]);

            float3 invInertiaA = g_RigidBodyInverseInertiaLocal[bodyA].xyz;
            float3 invInertiaB = g_RigidBodyInverseInertiaLocal[bodyB].xyz;
            // If static, inertia must be zero
            if (invMassA == 0.0)
                invInertiaA = 0.0;
            if (invMassB == 0.0)
                invInertiaB = 0.0;

            const float3 pA = posInvMassA.xyz + QuaternionRotate(qA, contact.localPointA.xyz);
            const float3 pB = posInvMassB.xyz + QuaternionRotate(qB, contact.localPointB.xyz);

            float3 n = SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));

            const float measuredPenetration = -dot(pB - pA, n);
            float penetration = min(measuredPenetration - kContactSlop, kMaxCorrectionPerIter);

            if (penetration <= 0.0)
                continue;

            const float3 rA = pA - posInvMassA.xyz;
            const float3 rB = pB - posInvMassB.xyz;

            const float3 angJA = cross(rA, n);
            const float3 angJB = cross(rB, n);

            const float3 angMassA = MultiplyWorldInverseInertia(invInertiaA, qA, angJA);
            const float3 angMassB = MultiplyWorldInverseInertia(invInertiaB, qB, angJB);

            const float angA = dot(cross(angMassA, rA), n);
            const float angB = dot(cross(angMassB, rB), n);

            const float denom = invMassA + invMassB + angA + angB;
            if (denom <= kEpsilon)
                continue;

            const float lambda = (penetration / denom) * kRelaxation;

            if (bodyIndex == bodyA)
            {
                translationCorrection += -n * (invMassA * lambda);
                rotationCorrection += -angMassA * lambda;
            }
            else
            {
                translationCorrection += n * (invMassB * lambda);
                rotationCorrection += angMassB * lambda;
            }
        }
    }

    // // If you want damping, scale by something stable like 1/solverIterations.
    // const float stableScale = 1.0; // 1.0 / max((float)solverIterations, 1.0); // or 1.0
    // translationCorrection *= stableScale;
    // rotationCorrection *= stableScale;

    g_RigidBodyTranslationCorrections[bodyIndex] = float4(translationCorrection, 0.0);
    g_RigidBodyRotationCorrections[bodyIndex] = float4(rotationCorrection, 0.0);
}
