#ifndef CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI

#include "physics/rigid/physics_rigid_solver_shared.hlsli"

float3 ChoosePerpendicular(float3 axis)
{
    const float3 reference = abs(axis.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    return normalize(cross(reference, axis));
}

float3 ClampErrorVector(float3 error, float maxMagnitude)
{
    const float lengthSq = dot(error, error);
    const float maxSq = maxMagnitude * maxMagnitude;
    if (lengthSq <= maxSq)
    {
        return error;
    }

    return error * (maxMagnitude * rsqrt(lengthSq));
}

float ClampErrorScalar(float error, float maxMagnitude)
{
    return clamp(error, -maxMagnitude, maxMagnitude);
}

float ComputeCorrectionLimitScale(float3 translationA, float3 rotationA,
                                  float3 translationB, float3 rotationB,
                                  float maxTranslationMagnitude, float maxRotationMagnitude)
{
    float scale = 1.0;

    if (maxTranslationMagnitude > 0.0)
    {
        const float translationALengthSq = dot(translationA, translationA);
        if (translationALengthSq > maxTranslationMagnitude * maxTranslationMagnitude)
        {
            scale = min(scale, maxTranslationMagnitude * rsqrt(translationALengthSq));
        }

        const float translationBLengthSq = dot(translationB, translationB);
        if (translationBLengthSq > maxTranslationMagnitude * maxTranslationMagnitude)
        {
            scale = min(scale, maxTranslationMagnitude * rsqrt(translationBLengthSq));
        }
    }

    if (maxRotationMagnitude > 0.0)
    {
        const float rotationALengthSq = dot(rotationA, rotationA);
        if (rotationALengthSq > maxRotationMagnitude * maxRotationMagnitude)
        {
            scale = min(scale, maxRotationMagnitude * rsqrt(rotationALengthSq));
        }

        const float rotationBLengthSq = dot(rotationB, rotationB);
        if (rotationBLengthSq > maxRotationMagnitude * maxRotationMagnitude)
        {
            scale = min(scale, maxRotationMagnitude * rsqrt(rotationBLengthSq));
        }
    }

    return scale;
}

float4 QuaternionToWXYZ(float4 q)
{
    return float4(q.w, q.x, q.y, q.z);
}

float3 GRow0(float4 q) { return float3(-0.5 * q.x, -0.5 * q.y, -0.5 * q.z); }
float3 GRow1(float4 q) { return float3( 0.5 * q.w,  0.5 * q.z, -0.5 * q.y); }
float3 GRow2(float4 q) { return float3(-0.5 * q.z,  0.5 * q.w,  0.5 * q.x); }
float3 GRow3(float4 q) { return float3( 0.5 * q.y, -0.5 * q.x,  0.5 * q.w); }

float4 QColumn0(float4 q) { return float4( q.w,  q.x,  q.y,  q.z); }
float4 QColumn1(float4 q) { return float4(-q.x,  q.w,  q.z, -q.y); }
float4 QColumn2(float4 q) { return float4(-q.y, -q.z,  q.w,  q.x); }
float4 QColumn3(float4 q) { return float4(-q.z,  q.y, -q.x,  q.w); }

float3 ComputeProjectionMatrixRow(float4 qA, float4 qB, float4 qColumn)
{
    const float3 g0 = GRow0(qB);
    const float3 g1 = GRow1(qB);
    const float3 g2 = GRow2(qB);
    const float3 g3 = GRow3(qB);
    return qColumn.x * g0 + qColumn.y * g1 + qColumn.z * g2 + qColumn.w * g3;
}

float3 ComputeProjectionJacobianRow(float4 projectionRow, float4 qA, float4 qB)
{
    const float3 m0 = ComputeProjectionMatrixRow(qA, qB, QColumn0(qA));
    const float3 m1 = ComputeProjectionMatrixRow(qA, qB, QColumn1(qA));
    const float3 m2 = ComputeProjectionMatrixRow(qA, qB, QColumn2(qA));
    const float3 m3 = ComputeProjectionMatrixRow(qA, qB, QColumn3(qA));
    return -(projectionRow.x * m0 + projectionRow.y * m1 +
             projectionRow.z * m2 + projectionRow.w * m3);
}

float ComputeProjectionConstraintValue(float4 projectionRow, float4 qA, float4 qB)
{
    const float4 relative = QuaternionMul(QuaternionConjugate(qA), qB);
    // We use the math from InteractiveComputerGraphics/PositionBasedDynamics
    // Their joint projection math is using w, x, y, z quaternion
    return dot(projectionRow, QuaternionToWXYZ(relative));
}

float ComputeHingeAngle(float4 projectionRow, float4 qA, float4 qB)
{
    return 2.0 * asin(clamp(ComputeProjectionConstraintValue(projectionRow, qA, qB), -1.0, 1.0));
}

float WrapAngleDelta(float delta)
{
    static const float kPi = 3.14159265358979323846;
    const float twoPi = 2.0 * kPi;
    delta = fmod(delta + kPi, twoPi);
    if (delta < 0.0)
    {
        delta += twoPi;
    }
    return delta - kPi;
}

float ComputeHingeWrappedAngle(float4 qA, float4 qB, float3 localAxisA0, float3 localAxisA1,
                               float3 localAxisB1)
{
    const float3 hingeAxis =
        SafeNormalize(QuaternionRotate(qA, localAxisA0), float3(1.0, 0.0, 0.0));
    const float3 referenceAWorld =
        SafeNormalize(QuaternionRotate(qA, localAxisA1), float3(0.0, 1.0, 0.0));
    const float3 referenceBWorld =
        SafeNormalize(QuaternionRotate(qB, localAxisB1), float3(0.0, 1.0, 0.0));
    const float3 projectedA = SafeNormalize(
        referenceAWorld - hingeAxis * dot(referenceAWorld, hingeAxis), ChoosePerpendicular(hingeAxis));
    float3 fallbackB = cross(hingeAxis, projectedA);
    fallbackB = SafeNormalize(fallbackB, ChoosePerpendicular(hingeAxis));
    const float3 projectedB =
        SafeNormalize(referenceBWorld - hingeAxis * dot(referenceBWorld, hingeAxis), fallbackB);
    const float sine = dot(hingeAxis, cross(projectedA, projectedB));
    const float cosine = dot(projectedA, projectedB);
    return atan2(sine, cosine);
}

float ComputeHingeUnwrappedAngle(float4 qA, float4 qB, GpuHingeJoint joint,
                                 GpuHingeJointRuntimeState runtimeState, out float wrappedAngle)
{
    wrappedAngle = ComputeHingeWrappedAngle(qA, qB, joint.localAxisA0.xyz, joint.localAxisA1.xyz,
                                            joint.localAxisB1.xyz);
    if (runtimeState.angleState.w < 0.5)
    {
        return wrappedAngle;
    }
    return runtimeState.angleState.y + WrapAngleDelta(wrappedAngle - runtimeState.angleState.x);
}

bool ComputeLimitTarget(float value, float2 limitRange, out float targetValue)
{
    if (value < limitRange.x)
    {
        targetValue = limitRange.x;
        return true;
    }
    if (value > limitRange.y)
    {
        targetValue = limitRange.y;
        return true;
    }

    targetValue = value;
    return false;
}

float ScaleVelocityMotorTargetNearLimits(float targetVelocity, float value, float2 limitRange,
                                         float approachDistance)
{
    if (approachDistance <= 0.0 || targetVelocity == 0.0)
    {
        return targetVelocity;
    }

    if (targetVelocity < 0.0)
    {
        const float distanceToLimit = value - limitRange.x;
        return targetVelocity * saturate(distanceToLimit / approachDistance);
    }

    const float distanceToLimit = limitRange.y - value;
    return targetVelocity * saturate(distanceToLimit / approachDistance);
}

float ComputeConstraintMatrixElement(float invMassA, float3 invInertiaA, float4 qA,
                                     float invMassB, float3 invInertiaB, float4 qB,
                                     float3 linearA0, float3 angularA0,
                                     float3 linearB0, float3 angularB0,
                                     float3 linearA1, float3 angularA1,
                                     float3 linearB1, float3 angularB1)
{
    float result = invMassA * dot(linearA0, linearA1) + invMassB * dot(linearB0, linearB1);
    result += dot(angularA0, MultiplyWorldInverseInertia(invInertiaA, qA, angularA1));
    result += dot(angularB0, MultiplyWorldInverseInertia(invInertiaB, qB, angularB1));
    return result;
}

bool SolveLinearSystem3x3(float a00, float a01, float a02,
                          float a10, float a11, float a12,
                          float a20, float a21, float a22,
                          float b0, float b1, float b2, out float3 x)
{
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a02 * a21 - a01 * a22;
    const float c02 = a01 * a12 - a02 * a11;
    const float det = a00 * c00 + a10 * c01 + a20 * c02;
    if (abs(det) <= kEpsilon)
    {
        x = 0.0;
        return false;
    }

    const float invDet = 1.0 / det;
    const float3 row0 = float3(c00, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20) * invDet;
    const float3 row1 = float3(c01, a00 * a22 - a02 * a20, a01 * a20 - a00 * a21) * invDet;
    const float3 row2 = float3(c02, a02 * a10 - a00 * a12, a00 * a11 - a01 * a10) * invDet;
    x = float3(dot(row0, float3(b0, b1, b2)),
               dot(row1, float3(b0, b1, b2)),
               dot(row2, float3(b0, b1, b2)));
    return true;
}

bool SolveLinearSystem5x5(float a[5][5], float b[5], out float x[5])
{
    float augmented[5][6];
    [unroll] for (uint row = 0u; row < 5u; ++row)
    {
        [unroll] for (uint col = 0u; col < 5u; ++col)
        {
            augmented[row][col] = a[row][col];
        }
        augmented[row][5u] = b[row];
    }

    [unroll] for (uint pivot = 0u; pivot < 5u; ++pivot)
    {
        uint pivotRow = pivot;
        float pivotMagnitude = abs(augmented[pivot][pivot]);
        [unroll] for (uint row = pivot + 1u; row < 5u; ++row)
        {
            const float candidate = abs(augmented[row][pivot]);
            if (candidate > pivotMagnitude)
            {
                pivotMagnitude = candidate;
                pivotRow = row;
            }
        }

        if (pivotMagnitude <= kEpsilon)
        {
            [unroll] for (uint i = 0u; i < 5u; ++i)
            {
                x[i] = 0.0;
            }
            return false;
        }

        if (pivotRow != pivot)
        {
            [unroll] for (uint col = pivot; col < 6u; ++col)
            {
                const float temp = augmented[pivot][col];
                augmented[pivot][col] = augmented[pivotRow][col];
                augmented[pivotRow][col] = temp;
            }
        }

        const float invPivot = 1.0 / augmented[pivot][pivot];
        [unroll] for (uint col = pivot; col < 6u; ++col)
        {
            augmented[pivot][col] *= invPivot;
        }

        [unroll] for (uint row = 0u; row < 5u; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const float factor = augmented[row][pivot];
            [unroll] for (uint col = pivot; col < 6u; ++col)
            {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }

    [unroll] for (uint row = 0u; row < 5u; ++row)
    {
        x[row] = augmented[row][5u];
    }
    return true;
}

bool SolveLinearSystem6x6(float a[6][6], float b[6], out float x[6])
{
    float augmented[6][7];
    [unroll] for (uint row = 0u; row < 6u; ++row)
    {
        [unroll] for (uint col = 0u; col < 6u; ++col)
        {
            augmented[row][col] = a[row][col];
        }
        augmented[row][6u] = b[row];
    }

    [unroll] for (uint pivot = 0u; pivot < 6u; ++pivot)
    {
        uint pivotRow = pivot;
        float pivotMagnitude = abs(augmented[pivot][pivot]);
        [unroll] for (uint row = pivot + 1u; row < 6u; ++row)
        {
            const float candidate = abs(augmented[row][pivot]);
            if (candidate > pivotMagnitude)
            {
                pivotMagnitude = candidate;
                pivotRow = row;
            }
        }

        if (pivotMagnitude <= kEpsilon)
        {
            [unroll] for (uint i = 0u; i < 6u; ++i)
            {
                x[i] = 0.0;
            }
            return false;
        }

        if (pivotRow != pivot)
        {
            [unroll] for (uint col = pivot; col < 7u; ++col)
            {
                const float temp = augmented[pivot][col];
                augmented[pivot][col] = augmented[pivotRow][col];
                augmented[pivotRow][col] = temp;
            }
        }

        const float invPivot = 1.0 / augmented[pivot][pivot];
        [unroll] for (uint col = pivot; col < 7u; ++col)
        {
            augmented[pivot][col] *= invPivot;
        }

        [unroll] for (uint row = 0u; row < 6u; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const float factor = augmented[row][pivot];
            [unroll] for (uint col = pivot; col < 7u; ++col)
            {
                augmented[row][col] -= factor * augmented[pivot][col];
            }
        }
    }

    [unroll] for (uint row = 0u; row < 6u; ++row)
    {
        x[row] = augmented[row][6u];
    }
    return true;
}

#endif // CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
