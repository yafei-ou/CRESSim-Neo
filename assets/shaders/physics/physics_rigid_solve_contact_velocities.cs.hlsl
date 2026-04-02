#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

static const float kRestitutionVelocityThreshold = 0.5;
static const float kRestitutionPenetrationThreshold = 2.0 * kContactSlop;
static const float kVelocityCorrectionAtomicScale = 100000.0;

CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyPositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyOrientations);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyLinearVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_PredictedRigidBodyAngularVelocities);
CRESSIM_STRUCTURED_BUFFER(float4, g_RigidBodyInverseInertiaLocal);
CRESSIM_STRUCTURED_BUFFER(uint, g_RigidBodyTypes);
CRESSIM_STRUCTURED_BUFFER(GpuRigidContact, g_RigidContacts);

CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyLinearVelocityCorrections);
CRESSIM_RW_STRUCTURED_BUFFER(int4, g_RigidBodyAngularVelocityCorrections);

int3 QuantizeVelocityCorrection(float3 value)
{
    return int3(round(value * kVelocityCorrectionAtomicScale));
}

float ComputeImpulseDenominator(float invMass, float3 invInertiaLocal, float4 orientation,
                                float3 r, float3 direction)
{
    if (invMass == 0.0)
    {
        return 0.0;
    }

    const float3 angJ = cross(r, direction);
    const float3 angMass = MultiplyWorldInverseInertia(invInertiaLocal, orientation, angJ);
    return invMass + dot(cross(angMass, r), direction);
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

    const uint bodyAIndex = contact.bodyA;
    const uint bodyBIndex = contact.bodyB;
    const uint bodyTypeA = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyAIndex);
    const uint bodyTypeB = CRESSIM_SB_LOAD(g_RigidBodyTypes, bodyBIndex);
    const float4 posInvMassA = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyAIndex);
    const float4 posInvMassB = CRESSIM_SB_LOAD(g_PredictedRigidBodyPositionsInvMass, bodyBIndex);
    const float invMassA = bodyTypeA == kRigidBodyTypeDynamic ? posInvMassA.w : 0.0;
    const float invMassB = bodyTypeB == kRigidBodyTypeDynamic ? posInvMassB.w : 0.0;
    if (invMassA == 0.0 && invMassB == 0.0)
    {
        return;
    }

    const float4 orientationA = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyAIndex));
    const float4 orientationB = QuaternionNormalize(CRESSIM_SB_LOAD(g_PredictedRigidBodyOrientations, bodyBIndex));
    const float3 linearVelocityA = CRESSIM_SB_REF(g_PredictedRigidBodyLinearVelocities, bodyAIndex).xyz;
    const float3 linearVelocityB = CRESSIM_SB_REF(g_PredictedRigidBodyLinearVelocities, bodyBIndex).xyz;
    const float3 angularVelocityA = CRESSIM_SB_REF(g_PredictedRigidBodyAngularVelocities, bodyAIndex).xyz;
    const float3 angularVelocityB = CRESSIM_SB_REF(g_PredictedRigidBodyAngularVelocities, bodyBIndex).xyz;
    const float3 invInertiaLocalA = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyAIndex).xyz;
    const float3 invInertiaLocalB = CRESSIM_SB_REF(g_RigidBodyInverseInertiaLocal, bodyBIndex).xyz;

    const float3 normal = SafeNormalize(contact.normalPenetration.xyz, float3(0.0, 1.0, 0.0));
    const float3 worldPointA =
        posInvMassA.xyz + QuaternionRotate(orientationA, contact.localPointA.xyz);
    const float3 worldPointB =
        posInvMassB.xyz + QuaternionRotate(orientationB, contact.localPointB.xyz);
    const float3 rA = worldPointA - posInvMassA.xyz;
    const float3 rB = worldPointB - posInvMassB.xyz;

    const float3 velocityAtA = linearVelocityA + cross(angularVelocityA, rA);
    const float3 velocityAtB = linearVelocityB + cross(angularVelocityB, rB);
    const float3 relativeVelocity = velocityAtB - velocityAtA;
    const float normalVelocity = dot(relativeVelocity, normal);
    if (normalVelocity >= 0.0)
    {
        return;
    }

    const float normalDenomA =
        ComputeImpulseDenominator(invMassA, invInertiaLocalA, orientationA, rA, normal);
    const float normalDenomB =
        ComputeImpulseDenominator(invMassB, invInertiaLocalB, orientationB, rB, normal);
    const float normalDenom = normalDenomA + normalDenomB;
    if (normalDenom <= kEpsilon)
    {
        return;
    }

    const bool enableRestitution =
        (-normalVelocity > kRestitutionVelocityThreshold) &&
        (contact.normalPenetration.w <= kRestitutionPenetrationThreshold);
    const float restitution = enableRestitution ? contact.material.y : 0.0;
    const float desiredNormalVelocity =
        enableRestitution ? (-restitution * normalVelocity) : 0.0;
    const float normalImpulseScalar =
        max(0.0, (desiredNormalVelocity - normalVelocity) / normalDenom);
    const float3 normalImpulse = normal * normalImpulseScalar;

    float3 totalImpulse = normalImpulse;

    const float3 tangentialVelocity = relativeVelocity - normal * normalVelocity;
    const float tangentialSpeed = length(tangentialVelocity);
    if (tangentialSpeed > 1.0e-4 && normalImpulseScalar > 0.0)
    {
        const float3 tangent = tangentialVelocity / tangentialSpeed;
        const float tangentDenomA =
            ComputeImpulseDenominator(invMassA, invInertiaLocalA, orientationA, rA, tangent);
        const float tangentDenomB =
            ComputeImpulseDenominator(invMassB, invInertiaLocalB, orientationB, rB, tangent);
        const float tangentDenom = tangentDenomA + tangentDenomB;
        if (tangentDenom > kEpsilon)
        {
            const float frictionLimit = contact.material.x * normalImpulseScalar;
            const float tangentImpulseScalar =
                clamp(-tangentialSpeed / tangentDenom, -frictionLimit, frictionLimit);
            totalImpulse += tangent * tangentImpulseScalar;
        }
    }

    if (invMassA != 0.0)
    {
        const float3 deltaLinearVelocityA = -totalImpulse * invMassA;
        const float3 deltaAngularVelocityA =
            MultiplyWorldInverseInertia(invInertiaLocalA, orientationA, cross(rA, -totalImpulse));
        const int3 linearA = QuantizeVelocityCorrection(deltaLinearVelocityA);
        const int3 angularA = QuantizeVelocityCorrection(deltaAngularVelocityA);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyAIndex).x, linearA.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyAIndex).y, linearA.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyAIndex).z, linearA.z);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyAIndex).x, angularA.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyAIndex).y, angularA.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyAIndex).z, angularA.z);
    }

    if (invMassB != 0.0)
    {
        const float3 deltaLinearVelocityB = totalImpulse * invMassB;
        const float3 deltaAngularVelocityB =
            MultiplyWorldInverseInertia(invInertiaLocalB, orientationB, cross(rB, totalImpulse));
        const int3 linearB = QuantizeVelocityCorrection(deltaLinearVelocityB);
        const int3 angularB = QuantizeVelocityCorrection(deltaAngularVelocityB);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyBIndex).x, linearB.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyBIndex).y, linearB.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyLinearVelocityCorrections, bodyBIndex).z, linearB.z);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyBIndex).x, angularB.x);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyBIndex).y, angularB.y);
        InterlockedAdd(CRESSIM_SB_REF(g_RigidBodyAngularVelocityCorrections, bodyBIndex).z, angularB.z);
    }
}
