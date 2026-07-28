#include "physics_particle_dispatch_constants.hlsli"
#include "physics_math.hlsli"
#include "physics_particle_types.hlsli"

CRESSIM_STRUCTURED_BUFFER(float4, g_ParticlePositionsInvMass);
CRESSIM_STRUCTURED_BUFFER(GpuShapeCluster, g_ShapeClusters);
CRESSIM_STRUCTURED_BUFFER(GpuShapeClusterMember, g_ShapeClusterMembers);
CRESSIM_RW_STRUCTURED_BUFFER(GpuShapeClusterPose, g_ShapeClusterPoses);

float3x3 OuterProduct(float3 a, float3 b)
{
    return float3x3(a.x * b.x, a.x * b.y, a.x * b.z,
                    a.y * b.x, a.y * b.y, a.y * b.z,
                    a.z * b.x, a.z * b.y, a.z * b.z);
}

float3 MatrixColumn(float3x3 m, uint column)
{
    return float3(m[0][column], m[1][column], m[2][column]);
}

float3x3 QuaternionToMatrix(float4 q)
{
    q = QuaternionNormalize(q);
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    return float3x3(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
                    2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
                    2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));
}

float4 QuaternionFromAxisAngle(float3 axis, float angle)
{
    const float halfAngle = 0.5 * angle;
    return QuaternionNormalize(float4(axis * sin(halfAngle), cos(halfAngle)));
}

float4 ExtractClosestRotation(float3x3 Apq, float4 initialRotation, uint iterationCount)
{
    float4 rotation = QuaternionNormalize(initialRotation);
    const uint clampedIterations = max(iterationCount, 1u);
    for (uint iteration = 0u; iteration < clampedIterations; ++iteration)
    {
        const float3x3 R = QuaternionToMatrix(rotation);

        const float3 r0 = MatrixColumn(R, 0u);
        const float3 r1 = MatrixColumn(R, 1u);
        const float3 r2 = MatrixColumn(R, 2u);

        const float3 a0 = MatrixColumn(Apq, 0u);
        const float3 a1 = MatrixColumn(Apq, 1u);
        const float3 a2 = MatrixColumn(Apq, 2u);

        const float denominator =
            abs(dot(r0, a0) + dot(r1, a1) + dot(r2, a2)) + 1.0e-9;
        const float3 omega =
            (cross(r0, a0) + cross(r1, a1) + cross(r2, a2)) / denominator;
        const float angle = length(omega);
        if (angle < 1.0e-7)
        {
            break;
        }

        const float4 deltaRotation = QuaternionFromAxisAngle(omega / angle, angle);
        rotation = QuaternionNormalize(QuaternionMul(deltaRotation, rotation));
    }
    return rotation;
}

[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint clusterIndex = dispatchThreadID.x;
    if (clusterIndex >= shapeClusterCount)
    {
        return;
    }

    const GpuShapeCluster cluster = CRESSIM_SB_LOAD(g_ShapeClusters, clusterIndex);
    if ((cluster.flags & kShapeClusterActive) == 0u)
    {
        return;
    }

    float totalMass = 0.0;
    float3 weightedPositionSum = float3(0.0, 0.0, 0.0);
    float3x3 Apq = (float3x3)0.0;

    for (uint localIndex = 0u; localIndex < cluster.memberCount; ++localIndex)
    {
        const GpuShapeClusterMember member =
            CRESSIM_SB_LOAD(g_ShapeClusterMembers, cluster.memberOffset + localIndex);
        if (member.particleIndex >= particleCount)
        {
            continue;
        }

        const float3 x = CRESSIM_SB_LOAD(g_ParticlePositionsInvMass, member.particleIndex).xyz;
        const float3 q = member.restOffset.xyz;
        const float weight = max(member.fittingWeight, 0.0);

        totalMass += weight;
        weightedPositionSum += weight * x;
        Apq += weight * OuterProduct(x, q);
    }

    if (totalMass <= kEpsilon)
    {
        GpuShapeClusterPose inactivePose = CRESSIM_SB_LOAD(g_ShapeClusterPoses, clusterIndex);
        inactivePose.currentCenterAndStatus.w = 0.0;
        CRESSIM_SB_STORE(g_ShapeClusterPoses, clusterIndex, inactivePose);
        return;
    }

    const float3 currentCenter = weightedPositionSum / totalMass;
    GpuShapeClusterPose pose = CRESSIM_SB_LOAD(g_ShapeClusterPoses, clusterIndex);
    pose.rotationQuaternion =
        ExtractClosestRotation(Apq, pose.rotationQuaternion, shapeRotationIterations);
    pose.currentCenterAndStatus = float4(currentCenter, 1.0);
    CRESSIM_SB_STORE(g_ShapeClusterPoses, clusterIndex, pose);
}
