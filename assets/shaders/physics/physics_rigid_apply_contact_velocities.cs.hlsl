#include "physics/include/physics_rigid_dispatch_constants.hlsli"
#include "physics/include/physics_rigid_common.hlsli"

StructuredBuffer<uint> g_RigidBodyTypes;

RWStructuredBuffer<float4> g_PredictedRigidBodyLinearVelocities;
RWStructuredBuffer<float4> g_PredictedRigidBodyAngularVelocities;
RWStructuredBuffer<int4> g_RigidBodyLinearVelocityCorrections;
RWStructuredBuffer<int4> g_RigidBodyAngularVelocityCorrections;

static const float kVelocityCorrectionAtomicScale = 100000.0;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bodyIndex = dispatchThreadID.x;
    if (bodyIndex >= rigidBodyCount)
    {
        return;
    }

    const uint bodyType = g_RigidBodyTypes[bodyIndex];
    float4 linearVelocity = g_PredictedRigidBodyLinearVelocities[bodyIndex];
    float4 angularVelocity = g_PredictedRigidBodyAngularVelocities[bodyIndex];
    const float3 linearCorrection =
        float3(g_RigidBodyLinearVelocityCorrections[bodyIndex].xyz) / kVelocityCorrectionAtomicScale;
    const float3 angularCorrection =
        float3(g_RigidBodyAngularVelocityCorrections[bodyIndex].xyz) / kVelocityCorrectionAtomicScale;

    if (bodyType == 2u)
    {
        linearVelocity.xyz += linearCorrection;
        angularVelocity.xyz += angularCorrection;
    }

    g_PredictedRigidBodyLinearVelocities[bodyIndex] = linearVelocity;
    g_PredictedRigidBodyAngularVelocities[bodyIndex] = angularVelocity;
    g_RigidBodyLinearVelocityCorrections[bodyIndex] = int4(0, 0, 0, 0);
    g_RigidBodyAngularVelocityCorrections[bodyIndex] = int4(0, 0, 0, 0);
}
