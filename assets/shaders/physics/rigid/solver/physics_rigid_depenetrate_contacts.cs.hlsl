#include "physics/physics_rigid_dispatch_constants.hlsli"
#include "physics/physics_atomic_float.hlsli"
#include "physics/rigid/physics_rigid_types.hlsli"
#include "physics/rigid/physics_rigid_broad_phase_types.hlsli"
#include "physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "physics/rigid/physics_rigid_solver_shared.hlsli"

static const float kBaumgarte = 0.25;
// Final rigid-rigid cleanup is a single normal-only depenetration pass.
static const float kMaxCorrectionPerIter = 0.1;
static const float kRelaxation = 0.90;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuBroadPhaseMeta, g_BroadPhaseMeta);
CRESSIM_STRUCTURED_BUFFER(GpuProxyRigidContactMeta, g_ProxyRigidContactMeta);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuBroadPhaseMeta broadPhaseMeta = CRESSIM_SB_LOAD(g_BroadPhaseMeta, 0);
    const GpuProxyRigidContactMeta proxyMeta = CRESSIM_SB_LOAD(g_ProxyRigidContactMeta, 0u);
    const uint totalContactSlots =
        broadPhaseMeta.candidatePairCount * kRigidContactsPerPair + proxyMeta.activeContactCount;
    if (contactIndex >= totalContactSlots)
    {
        return;
    }

    const GpuRigidContact contact = CRESSIM_SB_LOAD(g_RigidContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint bodyA = contact.bodyA;
    const uint bodyB = contact.bodyB;

    const float4 posInvMassA = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyA);
    const float4 posInvMassB = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyB);
    const float4 previousPosInvMassA = CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, bodyA);
    const float4 previousPosInvMassB = CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, bodyB);
    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyA);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyB);

    const float invMassA = bodyTypeA == kRigidBodyTypeDynamic ? posInvMassA.w : 0.0;
    const float invMassB = bodyTypeB == kRigidBodyTypeDynamic ? posInvMassB.w : 0.0;
    if (invMassA == 0.0 && invMassB == 0.0)
        return;

    const float4 qA = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 qB = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));
    const float4 previousQA =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, bodyA));
    const float4 previousQB =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, bodyB));

    float3 invInertiaA = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyA).xyz;
    float3 invInertiaB = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyB).xyz;
    if (invMassA == 0.0)
        invInertiaA = 0.0;
    if (invMassB == 0.0)
        invInertiaB = 0.0;

    const float3 pA = posInvMassA.xyz + QuaternionRotate(qA, contact.localPointA.xyz);
    const float3 pB = posInvMassB.xyz + QuaternionRotate(qB, contact.localPointB.xyz);
    const float3 previousPA =
        previousPosInvMassA.xyz + QuaternionRotate(previousQA, contact.localPointA.xyz);
    const float3 previousPB =
        previousPosInvMassB.xyz + QuaternionRotate(previousQB, contact.localPointB.xyz);

    const float3 n = SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float measuredPenetration = -dot(pB - pA, n);
    const float rawPenetration = max(measuredPenetration - kContactSlop, 0.0);
    const float penetration = min(rawPenetration * kBaumgarte, kMaxCorrectionPerIter);
    if (penetration <= 0.0)
        return;

    const float3 rA = pA - posInvMassA.xyz;
    const float3 rB = pB - posInvMassB.xyz;
    const float normalMassA = ComputeContactEffectiveMass(invMassA, invInertiaA, qA, rA, n);
    const float normalMassB = ComputeContactEffectiveMass(invMassB, invInertiaB, qB, rB, n);
    const float denom = normalMassA + normalMassB;
    if (denom <= kEpsilon)
        return;

    const float lambda = (penetration / denom) * kRelaxation;
    float3 translationA = -n * (invMassA * lambda);
    float3 rotationA =
        -MultiplyWorldInverseInertia(invInertiaA, qA, cross(rA, n)) * lambda;
    float3 translationB = n * (invMassB * lambda);
    float3 rotationB =
        MultiplyWorldInverseInertia(invInertiaB, qB, cross(rB, n)) * lambda;

    const float3 relativeDisplacement = (pB - previousPB) - (pA - previousPA);
    const float3 tangentialDisplacement = ProjectOntoContactTangent(relativeDisplacement, n);
    const float3 frictionDelta = ComputePositionFrictionDelta(
        tangentialDisplacement, penetration, saturate(contact.material.x), saturate(contact.material.z));
    const float frictionDistance = length(frictionDelta);
    if (frictionDistance > 0.0)
    {
        const float3 tangent = frictionDelta / frictionDistance;
        const float tangentMassA =
            ComputeContactEffectiveMass(invMassA, invInertiaA, qA, rA, tangent);
        const float tangentMassB =
            ComputeContactEffectiveMass(invMassB, invInertiaB, qB, rB, tangent);
        const float tangentDenom = tangentMassA + tangentMassB;
        if (tangentDenom > kEpsilon)
        {
            const float tangentLambda = (frictionDistance / tangentDenom) * kRelaxation;
            translationA += tangent * (invMassA * tangentLambda);
            rotationA +=
                MultiplyWorldInverseInertia(invInertiaA, qA, cross(rA, tangent)) * tangentLambda;
            translationB -= tangent * (invMassB * tangentLambda);
            rotationB -=
                MultiplyWorldInverseInertia(invInertiaB, qB, cross(rB, tangent)) * tangentLambda;
        }
    }

    if (bodyTypeA == kRigidBodyTypeDynamic && invMassA != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyA, translationA);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyA, rotationA);
    }

    if (bodyTypeB == kRigidBodyTypeDynamic && invMassB != 0.0)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, bodyB, translationB);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, bodyB, rotationB);
    }
}
