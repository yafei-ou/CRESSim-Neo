cbuffer PhysicsDispatchConstantsBuffer
{
    float dt;
    uint rigidBodyCount;
    uint pairCount;
    uint substepIndex;
    uint iterationIndex;
    uint solverIterations;
    uint reserved0;
    uint reserved1;
};

#include "physics/physics_rigid_common.hlsli"

StructuredBuffer<float4> g_PredictedRigidBodyPositionsInvMass;
StructuredBuffer<float4> g_PredictedRigidBodyOrientations;
StructuredBuffer<float4> g_RigidBodyInverseInertiaLocal;
StructuredBuffer<GpuRigidContact> g_RigidContacts;

RWStructuredBuffer<float4> g_RigidBodyTranslationCorrections;
RWStructuredBuffer<float4> g_RigidBodyRotationCorrections;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const float4 bodyPositionInvMass = g_PredictedRigidBodyPositionsInvMass[bodyIndex];
    const float4 bodyOrientation = QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyIndex]);
    const float3 bodyInverseInertiaLocal = g_RigidBodyInverseInertiaLocal[bodyIndex].xyz;

    float3 translationCorrection = 0.0;
    float3 rotationCorrection = 0.0;
    float contactCount = 0.0;

    [loop]
    for (uint pairIdx = 0u; pairIdx < pairCount; ++pairIdx)
    {
        const uint contactBaseIndex = pairIdx * kRigidContactsPerPair;
        [unroll]
        for (uint contactOffset = 0u; contactOffset < kRigidContactsPerPair; ++contactOffset)
        {
            const GpuRigidContact contact = g_RigidContacts[contactBaseIndex + contactOffset];
            if (contact.active == 0u)
            {
                continue;
            }
            if (contact.bodyA != bodyIndex && contact.bodyB != bodyIndex)
            {
                continue;
            }

            const uint bodyA = contact.bodyA;
            const uint bodyB = contact.bodyB;
            const float4 positionInvMassA = g_PredictedRigidBodyPositionsInvMass[bodyA];
            const float4 positionInvMassB = g_PredictedRigidBodyPositionsInvMass[bodyB];
            const float4 orientationA =
                QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyA]);
            const float4 orientationB =
                QuaternionNormalize(g_PredictedRigidBodyOrientations[bodyB]);
            const float3 inverseInertiaLocalA = g_RigidBodyInverseInertiaLocal[bodyA].xyz;
            const float3 inverseInertiaLocalB = g_RigidBodyInverseInertiaLocal[bodyB].xyz;
            const float3 normal = contact.normalPenetration.xyz;
            const float3 contactPoint = contact.worldPoint.xyz;

            const float penetration =
                max(contact.normalPenetration.w - kContactSlop, 0.0) * kContactBias;
            if (penetration <= 0.0)
            {
                continue;
            }

            const float3 rA = contactPoint - positionInvMassA.xyz;
            const float3 rB = contactPoint - positionInvMassB.xyz;
            const float3 angularJacobianA = cross(rA, normal);
            const float3 angularJacobianB = cross(rB, normal);
            const float3 angularMassA =
                MultiplyWorldInverseInertia(inverseInertiaLocalA, orientationA, angularJacobianA);
            const float3 angularMassB =
                MultiplyWorldInverseInertia(inverseInertiaLocalB, orientationB, angularJacobianB);
            const float angularA = dot(cross(angularMassA, rA), normal);
            const float angularB = dot(cross(angularMassB, rB), normal);
            const float denominator =
                positionInvMassA.w + positionInvMassB.w + angularA + angularB;
            if (denominator <= kEpsilon)
            {
                continue;
            }

            const float lambda = penetration / denominator;
            if (bodyIndex == bodyA)
            {
                translationCorrection += -normal * (positionInvMassA.w * lambda);
                rotationCorrection += -angularMassA * lambda;
            }
            else
            {
                translationCorrection += normal * (positionInvMassB.w * lambda);
                rotationCorrection += angularMassB * lambda;
            }
            contactCount += 1.0;
        }
    }

    if (contactCount > 0.0)
    {
        translationCorrection /= contactCount;
        rotationCorrection /= contactCount;
    }

    g_RigidBodyTranslationCorrections[bodyIndex] = float4(translationCorrection, 0.0);
    g_RigidBodyRotationCorrections[bodyIndex] = float4(rotationCorrection, 0.0);
}
