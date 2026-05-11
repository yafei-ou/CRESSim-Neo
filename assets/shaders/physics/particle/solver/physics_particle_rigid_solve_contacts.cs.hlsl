#include "../../../include/physics/physics_atomic_float.hlsli"
#include "../../../include/physics/particle/physics_particle_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_types.hlsli"
#include "../../../include/physics/rigid/physics_rigid_contact_primitives.hlsli"
#include "../../../include/physics/rigid/physics_rigid_solver_shared.hlsli"
#include "../../../include/physics/core/physics_math.hlsli"

static const float kSoftContactRelaxation = 0.95;
static const float kSoftMaxCorrectionPerIter = 0.05;

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePreviousPositions);
CRESSIM_STRUCTURED_BUFFER(uint, g_ParticleMaterialIndices);
CRESSIM_STRUCTURED_BUFFER(float4, g_ParticleContactMaterials);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PreviousRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(float4, g_ColliderMaterials);
CRESSIM_STRUCTURED_BUFFER(GpuParticleRigidContact, g_ParticleRigidContacts);
CRESSIM_STRUCTURED_BUFFER(GpuParticleNeighborMeta, g_ParticleNeighborMeta);

CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_ParticlePositionCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyTranslationCorrections);
CRESSIM_RW_ATOMIC_FLOAT_BUFFER(g_RigidBodyRotationCorrections);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint contactIndex = dispatchThreadID.x;
    const GpuParticleNeighborMeta meta = CRESSIM_SB_LOAD(g_ParticleNeighborMeta, 0u);
    if (contactIndex >= meta.activeParticleRigidContactCount)
    {
        return;
    }

    const GpuParticleRigidContact contact = CRESSIM_SB_LOAD(g_ParticleRigidContacts, contactIndex);
    if (contact.active == 0u)
    {
        return;
    }

    const uint particleIndex = contact.particleIndex;
    const float4 softPositionInvMass =
        CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, particleIndex);
    const float3 previousSoftPosition =
        CRESSIM_SB_LOAD(g_ParticlePreviousPositions, particleIndex).xyz;
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
    const uint materialIndex = CRESSIM_SB_LOAD(g_ParticleMaterialIndices, particleIndex);
    const float3 combinedMaterial = CombineContactMaterial(
        CRESSIM_SB_LOAD(g_ParticleContactMaterials, materialIndex),
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
    const float3 frictionDelta = ComputePositionFrictionDelta(
        tangentialDisplacement, penetration, combinedMaterial.x, combinedMaterial.z);
    const float frictionDistance = length(frictionDelta);
    if (frictionDistance > 0.0)
    {
        const float3 tangent = frictionDelta / frictionDistance;
        const float tangentMassRigid = ComputeContactEffectiveMass(invMassRigid, invInertiaRigid,
                                                                  rigidOrientation, rRigid, tangent);
        const float tangentDenom = invMassSoft + tangentMassRigid;
        if (tangentDenom > kEpsilon)
        {
            const float tangentLambda = frictionDistance / tangentDenom;
            softCorrection += tangent * (tangentLambda * invMassSoft * kSoftContactRelaxation);
            rigidTranslationCorrection -=
                tangent * (tangentLambda * invMassRigid * kSoftContactRelaxation);
            rigidRotationCorrection -=
                MultiplyWorldInverseInertia(invInertiaRigid, rigidOrientation, cross(rRigid, tangent)) *
                (tangentLambda * kSoftContactRelaxation);
        }
    }

    if (invMassSoft > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_ParticlePositionCorrections, particleIndex, softCorrection);
    }

    if (invMassRigid > kEpsilon)
    {
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyTranslationCorrections, rigidBodyIndex,
                                      rigidTranslationCorrection);
        CRESSIM_ATOMIC_ADD_FLOAT3_CAS(g_RigidBodyRotationCorrections, rigidBodyIndex,
                                      rigidRotationCorrection);
    }
}
