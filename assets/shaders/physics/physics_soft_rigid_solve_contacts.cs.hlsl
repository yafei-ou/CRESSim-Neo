#include "physics/include/physics_rigid_common.hlsli"

static const float kSoftCorrectionAtomicScale = 100000.0;
static const float kSoftContactRelaxation = 0.95;
static const float kSoftMaxCorrectionPerIter = 0.05;

CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(float4, g_SoftParticleMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuSoftRigidContact, g_SoftRigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuSoftNeighborMeta, g_SoftNeighborMeta);

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
    const GpuSoftNeighborMeta meta = CRESSIM_SB_LOAD(g_SoftNeighborMeta, 0u);
    if (contactIndex >= meta.activeSoftRigidContactCount)
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
    const float3 previousSoftPosition =
        CRESSIM_SB_LOAD(g_SoftParticlePreviousPositions, softParticleIndex).xyz;
    const float invMassSoft = softPositionInvMass.w;
    const uint rigidBodyIndex = contact.rigidBodyIndex;
    const float4 previousRigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PreviousRigidBodyPositionsInvMass, rigidBodyIndex);
    const float4 rigidPositionInvMass =
        CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, rigidBodyIndex);
    const float4 previousRigidOrientation =
        QuaternionNormalize(CRESSIM_SB_LOAD(g_PreviousRigidBodyOrientations, rigidBodyIndex));
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
    const float2 combinedMaterial = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_SoftParticleMaterials, softParticleIndex),
        CRESSIM_SB_LOAD(g_ColliderMaterials, contact.colliderIndex));
    if (invMassRigid <= kEpsilon)
    {
        invInertiaRigid = 0.0;
    }

    const float3 previousRigidContactPoint =
        previousRigidPositionInvMass.xyz +
        QuaternionRotate(previousRigidOrientation, contact.rigidLocalPoint.xyz);
    const float3 rigidContactPoint =
        rigidPositionInvMass.xyz + QuaternionRotate(rigidOrientation, contact.rigidLocalPoint.xyz);
    const float3 rRigid = rigidContactPoint - rigidPositionInvMass.xyz;
    const float normalMassRigid =
        ComputeContactEffectiveMass(invMassRigid, invInertiaRigid, rigidOrientation, rRigid, normal);
    const float denom = invMassSoft + normalMassRigid;
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = (penetration / denom) * kSoftContactRelaxation;
    float3 softCorrection = normal * (invMassSoft * lambda);
    float3 rigidTranslationCorrection = -normal * (invMassRigid * lambda);
    float3 rigidRotationCorrection =
        -MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation, cross(rRigid, normal)) * lambda;

    const float3 relativeDisplacement =
        (rigidContactPoint - previousRigidContactPoint) - (softPositionInvMass.xyz - previousSoftPosition);
    const float3 tangentialDisplacement = ProjectOntoContactTangent(relativeDisplacement, normal);
    const float tangentialDistance = length(tangentialDisplacement);
    if (tangentialDistance > 1.0e-5 && penetration > kEpsilon)
    {
        const float3 tangent = tangentialDisplacement / tangentialDistance;
        const float tangentMassRigid = ComputeContactEffectiveMass(invMassRigid, invInertiaRigid,
                                                                  rigidOrientation, rRigid, tangent);
        const float tangentDenom = invMassSoft + tangentMassRigid;
        if (tangentDenom > kEpsilon)
        {
            const float frictionScale =
                min(saturate(combinedMaterial.x) * penetration / tangentialDistance, 1.0);
            const float3 frictionDelta = tangentialDisplacement * frictionScale;
            const float tangentLambda = length(frictionDelta) / tangentDenom;
            softCorrection += tangent * (tangentLambda * invMassSoft * kSoftContactRelaxation);
            rigidTranslationCorrection -=
                tangent * (tangentLambda * invMassRigid * kSoftContactRelaxation);
            rigidRotationCorrection -=
                MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation, cross(rRigid, tangent)) *
                (tangentLambda * kSoftContactRelaxation);
        }
    }

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
