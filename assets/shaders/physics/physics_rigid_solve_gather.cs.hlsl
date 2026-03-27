#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

// TODO: this is okay for now but with bad accuracy and shrinks
// the allowed world range a lot
// We could split this into several stages

static const float kMaxCorrectionPerIter = 0.02; // world units, tune (e.g. 2 cm)
static const float kRelaxation = 0.90;           // try 0.8 if jittery
static const float kCorrectionAtomicScale = 100000.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);

CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyTranslationCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyRotationCorrections);

int3 QuantizeCorrection(float3 value)
{
    return int3(round(value * kCorrectionAtomicScale));
}

[numthreads(64, 1, 1)] void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const uint totalContactSlots = candidatePairCount * kRigidContactsPerPair;
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
    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyA);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyB);

    const float invMassA = bodyTypeA == 2u ? posInvMassA.w : 0.0;
    const float invMassB = bodyTypeB == 2u ? posInvMassB.w : 0.0;
    if (invMassA == 0.0 && invMassB == 0.0)
        return;

    const float4 qA = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyA));
    const float4 qB = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyB));

    float3 invInertiaA = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyA).xyz;
    float3 invInertiaB = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyB).xyz;
    if (invMassA == 0.0)
        invInertiaA = 0.0;
    if (invMassB == 0.0)
        invInertiaB = 0.0;

    const float3 pA = posInvMassA.xyz + QuaternionRotate(qA, contact.localPointA.xyz);
    const float3 pB = posInvMassB.xyz + QuaternionRotate(qB, contact.localPointB.xyz);

    const float3 n = SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float measuredPenetration = -dot(pB - pA, n);
    const float penetration = min(measuredPenetration - kContactSlop, kMaxCorrectionPerIter);
    if (penetration <= 0.0)
        return;

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
        return;

    const float lambda = (penetration / denom) * kRelaxation;
    const int3 translationA = QuantizeCorrection(-n * (invMassA * lambda));
    const int3 rotationA = QuantizeCorrection(-angMassA * lambda);
    const int3 translationB = QuantizeCorrection(n * (invMassB * lambda));
    const int3 rotationB = QuantizeCorrection(angMassB * lambda);

    if (bodyTypeA == 2u && invMassA != 0.0)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyA).x, translationA.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyA).y, translationA.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyA).z, translationA.z);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyA).x, rotationA.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyA).y, rotationA.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyA).z, rotationA.z);
    }

    if (bodyTypeB == 2u && invMassB != 0.0)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyB).x, translationB.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyB).y, translationB.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, bodyB).z, translationB.z);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyB).x, rotationB.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyB).y, rotationB.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, bodyB).z, rotationB.z);
    }
}
