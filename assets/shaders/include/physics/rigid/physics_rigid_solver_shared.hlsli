#ifndef CRESSIM_NEO_PHYSICS_RIGID_SOLVER_SHARED_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_SOLVER_SHARED_HLSLI

#include "physics_shape_common.hlsli"

static const float kRigidRestitutionThreshold = 0.0;
// These affects the needed number of velocity iterations; larger values lead to faster velocity
// solver but can be unstable with more contacts; smaller values improves stability but requires
// more velocity iterations
static const float kMaxTotalLinearVelocityCorrectionPerIter = 0.8;
static const float kMaxTotalAngularVelocityCorrectionPerIter = 0.5;

float3 MultiplyWorldInverseInertia(float3 inverseInertiaLocal, float4 orientation, float3 value)
{
    const float3 ax = BoxAxisX(orientation);
    const float3 ay = BoxAxisY(orientation);
    const float3 az = BoxAxisZ(orientation);
    return ax * (inverseInertiaLocal.x * dot(ax, value)) +
           ay * (inverseInertiaLocal.y * dot(ay, value)) +
           az * (inverseInertiaLocal.z * dot(az, value));
}

float3 ComputeContactPointVelocity(float3 linearVelocity, float3 angularVelocity, float3 leverArm)
{
    return linearVelocity + cross(angularVelocity, leverArm);
}

float ComputeContactEffectiveMass(float invMass, float3 inverseInertiaLocal, float4 orientation,
                                  float3 r, float3 direction)
{
    if (invMass <= kEpsilon)
    {
        return 0.0;
    }

    const float3 angularJacobian = cross(r, direction);
    const float3 angularMass =
        MultiplyWorldInverseInertia(inverseInertiaLocal, orientation, angularJacobian);
    return invMass + dot(cross(angularMass, r), direction);
}

float3 CombineContactMaterial(float4 materialA, float4 materialB)
{
    const float friction = sqrt(max(0.0, materialA.x) * max(0.0, materialB.x));
    const float restitution = max(max(0.0, materialA.y), max(0.0, materialB.y));
    const float staticFriction = sqrt(max(0.0, materialA.w) * max(0.0, materialB.w));
    return float3(friction, restitution, staticFriction);
}

float3 ProjectOntoContactTangent(float3 value, float3 normal)
{
    return value - normal * dot(value, normal);
}

float3 ComputePositionFrictionDelta(float3 tangentialDisplacement, float penetration,
                                    float kineticFriction, float staticFriction)
{
    const float tangentialDistance = length(tangentialDisplacement);
    if (tangentialDistance <= 1.0e-5 || penetration <= kEpsilon)
    {
        return 0.0;
    }

    const float staticLimit = saturate(staticFriction) * penetration;
    if (tangentialDistance <= staticLimit)
    {
        return tangentialDisplacement;
    }

    const float kineticScale =
        min(saturate(kineticFriction) * penetration / tangentialDistance, 1.0);
    return tangentialDisplacement * kineticScale;
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_SOLVER_SHARED_HLSLI
