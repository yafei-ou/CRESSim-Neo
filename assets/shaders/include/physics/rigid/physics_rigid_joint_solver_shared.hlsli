#ifndef CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI

#include "physics_rigid_solver_shared.hlsli"

float3 ChoosePerpendicular(float3 axis)
{
    const float3 reference = abs(axis.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    return normalize(cross(reference, axis));
}

void BuildConstraintBasis(float3 axis, out float3 basis0, out float3 basis1, out float3 basis2)
{
    basis0 = SafeNormalize(axis, float3(1.0, 0.0, 0.0));
    basis1 = ChoosePerpendicular(basis0);
    basis2 = normalize(cross(basis0, basis1));
}

void ApplyLinearConstraintRow(float3 direction, float error, float invMassA, float invMassB,
                              float3 invInertiaA, float4 qA, float3 rA, float3 invInertiaB,
                              float4 qB, float3 rB, out float3 translationA,
                              out float3 rotationA, out float3 translationB,
                              out float3 rotationB)
{
    translationA = 0.0;
    rotationA = 0.0;
    translationB = 0.0;
    rotationB = 0.0;

    const float denom = ComputeContactEffectiveMass(invMassA, invInertiaA, qA, rA, direction) +
                        ComputeContactEffectiveMass(invMassB, invInertiaB, qB, rB, direction);
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = error / denom;
    translationA = direction * (invMassA * lambda);
    rotationA = MultiplyWorldInverseInertia(invInertiaA, qA, cross(rA, direction)) * lambda;
    translationB = -direction * (invMassB * lambda);
    rotationB = -MultiplyWorldInverseInertia(invInertiaB, qB, cross(rB, direction)) * lambda;
}

void ApplyAngularConstraintRow(float3 axis, float error, float3 invInertiaA, float4 qA,
                               float3 invInertiaB, float4 qB, out float3 rotationA,
                               out float3 rotationB)
{
    rotationA = 0.0;
    rotationB = 0.0;

    const float denom = ComputeAngularEffectiveMass(invInertiaA, qA, axis) +
                        ComputeAngularEffectiveMass(invInertiaB, qB, axis);
    if (denom <= kEpsilon)
    {
        return;
    }

    const float lambda = error / denom;
    rotationA = MultiplyWorldInverseInertia(invInertiaA, qA, axis) * lambda;
    rotationB = -MultiplyWorldInverseInertia(invInertiaB, qB, axis) * lambda;
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
