#ifndef CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
#define CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI

#include "physics_rigid_solver_shared.hlsli"

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

#endif // CRESSIM_NEO_PHYSICS_RIGID_JOINT_SOLVER_SHARED_HLSLI
