#include "physics/include/physics_rigid_common.hlsli"
#include "physics/include/physics_soft_dispatch_constants.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;
static const float kSoftContactRelaxation = 0.90;
static const float kSoftMaxCorrectionPerIter = 0.02;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuSoftRigidContact, g_SoftRigidContacts);

CRESSIM_RW_STRUCTURED_BUFFER(int4, g_SoftPositionCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyTranslationCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyRotationCorrections);

int3 QuantizeSoftCorrection(float3 value)
{
    return int3(round(value * kSoftCorrectionAtomicScale));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    if (contactIndex >= softRigidCandidatePairCapacity)
    {
        return;
    }

    const GpuSoftRigidContact contact = CRESSIM_SB_LOAD(g_SoftRigidContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint softParticleIndex = contact.softParticleIndex;
    const float4 softPositionInvMass =
        CRESSIM_SB_LOAD(g_SoftParticlePositionsInvMass, softParticleIndex);
    const float invMassSoft = softPositionInvMass.w;
    const uint rigidBodyIndex = contact.rigidBodyIndex;
    const float4 rigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
    const float4 rigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, rigidBodyIndex));
    const uint rigidBodyType = CRESSIM_SB_LOAD(g_RigidBodyTypes, rigidBodyIndex);
    const float invMassRigid = rigidBodyType == kRigidBodyTypeDynamic ? rigidPositionInvMass.w : 0.0;
    if (invMassSoft <= kEpsilon && invMassRigid <= kEpsilon)
    {
        return;
    }

    const float penetration =
        min(max(contact.normalPenetration.w - kContactSlop, 0.0), kSoftMaxCorrectionPerIter);
    if (penetration <= 0.0)
    {
        return;
    }

    const float3 normal =
        SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    float3 invInertiaRigid = CRESSIM_SB_LOAD(g_RigidBodyInverseInertiaLocal, rigidBodyIndex).xyz;
    if (invMassRigid <= kEpsilon)
    {
        invInertiaRigid = 0.0;
    }

    const float3 rigidContactPoint =
        rigidPositionInvMass.xyz + QuaternionRotate(rigidOrientation, contact.rigidLocalPoint.xyz);
    const float3 rRigid = rigidContactPoint - rigidPositionInvMass.xyz;
    const float3 angJ = cross(rRigid, normal);
    const float3 angMass = MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation, angJ);
    const float angTerm = dot(cross(angMass, rRigid), normal);
    const float denom = invMassSoft + invMassRigid + angTerm;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = (penetration / denom) * kSoftContactRelaxation;
    const float3 softCorrection = normal * (invMassSoft * lambda);
    const float3 rigidTranslationCorrection = -normal * (invMassRigid * lambda);
    const float3 rigidRotationCorrection = -angMass * lambda;
    const int3 softQuantized = QuantizeSoftCorrection(softCorrection);
    const int3 rigidTranslationQuantized = QuantizeSoftCorrection(rigidTranslationCorrection);
    const int3 rigidRotationQuantized = QuantizeSoftCorrection(rigidRotationCorrection);

    if (invMassSoft > kEpsilon)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).x, softQuantized.x);
        InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).y, softQuantized.y);
        InterlockedAdd(CRESSIM_SB_REF(g_SoftPositionCorrections, softParticleIndex).z, softQuantized.z);
    }

    if (invMassRigid > kEpsilon)
    {
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, rigidBodyIndex).x,
                       rigidTranslationQuantized.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, rigidBodyIndex).y,
                       rigidTranslationQuantized.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyTranslationCorrections, rigidBodyIndex).z,
                       rigidTranslationQuantized.z);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, rigidBodyIndex).x,
                       rigidRotationQuantized.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, rigidBodyIndex).y,
                       rigidRotationQuantized.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyRotationCorrections, rigidBodyIndex).z,
                       rigidRotationQuantized.z);
    }
}
